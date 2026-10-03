//! Bounded sampled-pointer tracking. Each pointer belongs to one shard, so
//! cross-thread frees have the same ownership rules as same-thread frees.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Mutex, OnceLock,
    },
};

use super::{session_is_current, splitmix64, StackKey, SAMPLING_CONFIG_GENERATION};

const SHARD_COUNT: usize = 64;
static ENABLED: AtomicBool = AtomicBool::new(false);
static DROPPED: AtomicU64 = AtomicU64::new(0);
static METADATA_PAYLOAD_BYTES: AtomicUsize = AtomicUsize::new(0);
static SHARDS: OnceLock<[LiveShard; SHARD_COUNT]> = OnceLock::new();

// Keep frequently written occupancy counters on separate cache lines.
#[repr(align(64))]
struct LiveShard {
    entries: Mutex<LiveMap>,
    count: AtomicUsize,
    membership: AtomicU64,
}

struct LiveMap {
    samples: HashMap<usize, LiveAllocationSample>,
    capacity: usize,
    // Detached reallocations retain a slot until success or failure is known.
    reservations: usize,
    // Exact bit reference counts let the atomic filter reject most untracked
    // frees without a mutex lookup, while never dropping a tracked removal.
    membership_counts: [usize; 64],
}

#[derive(Debug, Copy, Clone)]
pub(super) struct LiveAllocationSample {
    pub(super) stack: StackKey,
    // Preserve fractional weights until per-stack aggregation. Rounding each
    // pointer's object count would systematically bias similarly sized objects.
    pub(super) weighted_objects: f64,
    pub(super) weighted_bytes: f64,
}

pub(super) fn enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

pub(super) fn prepare(enabled: bool, capacity: usize) {
    clear();
    DROPPED.store(0, Ordering::Relaxed);
    if !enabled {
        return;
    }
    let shards = SHARDS.get_or_init(|| {
        std::array::from_fn(|_| LiveShard {
            entries: Mutex::new(LiveMap {
                samples: HashMap::new(),
                capacity: 0,
                reservations: 0,
                membership_counts: [0; 64],
            }),
            count: AtomicUsize::new(0),
            membership: AtomicU64::new(0),
        })
    });
    let mut payload_bytes = 0_usize;
    for (index, shard) in shards.iter().enumerate() {
        let mut entries = shard.entries.lock().unwrap_or_else(|err| err.into_inner());
        let shard_capacity = capacity / SHARD_COUNT + usize::from(index < capacity % SHARD_COUNT);
        entries.samples = HashMap::with_capacity(shard_capacity);
        payload_bytes = payload_bytes.saturating_add(
            entries
                .samples
                .capacity()
                .saturating_mul(std::mem::size_of::<(usize, LiveAllocationSample)>()),
        );
        entries.capacity = shard_capacity;
        entries.reservations = 0;
        entries.membership_counts.fill(0);
        shard.count.store(0, Ordering::Release);
        shard.membership.store(0, Ordering::Release);
    }
    METADATA_PAYLOAD_BYTES.store(payload_bytes, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Release);
}

pub(super) fn clear() {
    ENABLED.store(false, Ordering::Release);
    if let Some(shards) = SHARDS.get() {
        for shard in shards {
            let mut entries = shard.entries.lock().unwrap_or_else(|err| err.into_inner());
            entries.samples = HashMap::new();
            entries.capacity = 0;
            entries.reservations = 0;
            entries.membership_counts.fill(0);
            shard.count.store(0, Ordering::Release);
            shard.membership.store(0, Ordering::Release);
        }
    }
    METADATA_PAYLOAD_BYTES.store(0, Ordering::Relaxed);
}

fn shard_and_bit(pointer: usize) -> (usize, usize) {
    let hash = splitmix64(pointer as u64);
    (hash as usize % SHARD_COUNT, (hash >> 6) as usize % 64)
}

impl LiveMap {
    fn add_membership(&mut self, shard: &LiveShard, bit: usize) {
        self.membership_counts[bit] += 1;
        shard.membership.fetch_or(1_u64 << bit, Ordering::Release);
        shard.count.fetch_add(1, Ordering::Relaxed);
    }

    fn remove_membership(&mut self, shard: &LiveShard, bit: usize) {
        self.membership_counts[bit] -= 1;
        if self.membership_counts[bit] == 0 {
            shard
                .membership
                .fetch_and(!(1_u64 << bit), Ordering::Release);
        }
        shard.count.fetch_sub(1, Ordering::Relaxed);
    }
}

pub(super) fn record(pointer: usize, stack: StackKey, size: u64, interval: u64, generation: u64) {
    let Some(shards) = SHARDS.get() else {
        return;
    };
    let (index, bit) = shard_and_bit(pointer);
    let shard = &shards[index];
    let Ok(mut entries) = shard.entries.try_lock() else {
        DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    };
    if !enabled() || !session_is_current(generation) {
        return;
    }
    if entries.samples.len() + entries.reservations >= entries.capacity {
        DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let (weighted_objects, weighted_bytes) = weights(size, interval);
    let previous = entries.samples.insert(
        pointer,
        LiveAllocationSample {
            stack,
            weighted_objects,
            weighted_bytes,
        },
    );
    if previous.is_none() {
        entries.add_membership(shard, bit);
    }
}

// One pointer can cross many Poisson intervals, but contributes one live hit.
// Use its inclusion probability, rather than interval crossings, to avoid
// overweighting large objects and rounding small-object bytes to zero.
fn weights(size: u64, interval: u64) -> (f64, f64) {
    let probability = -(-(size as f64) / interval.max(1) as f64).exp_m1();
    let objects = (1.0 / probability).max(1.0);
    let bytes = (size as f64 / probability).max(size as f64);
    (objects, bytes)
}

pub(super) fn remove(pointer: usize) {
    let Some(shards) = SHARDS.get() else {
        return;
    };
    let (index, bit) = shard_and_bit(pointer);
    let shard = &shards[index];
    if shard.membership.load(Ordering::Acquire) & (1_u64 << bit) == 0 {
        return;
    }
    // Dropping a removal on contention would fabricate a live allocation.
    // This short critical section never allocates or resolves symbols.
    let mut entries = shard.entries.lock().unwrap_or_else(|err| err.into_inner());
    if entries.samples.remove(&pointer).is_some() {
        entries.remove_membership(shard, bit);
    }
}

pub(super) struct PendingReallocation {
    pointer: usize,
    sample: LiveAllocationSample,
    generation: u64,
}

pub(super) fn detach(pointer: usize) -> Option<PendingReallocation> {
    let shards = SHARDS.get()?;
    let (index, bit) = shard_and_bit(pointer);
    let shard = &shards[index];
    if shard.membership.load(Ordering::Acquire) & (1_u64 << bit) == 0 {
        return None;
    }
    let mut entries = shard.entries.lock().unwrap_or_else(|err| err.into_inner());
    let sample = entries.samples.remove(&pointer)?;
    entries.reservations += 1;
    entries.remove_membership(shard, bit);
    Some(PendingReallocation {
        pointer,
        sample,
        generation: SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire),
    })
}

impl PendingReallocation {
    pub(super) fn finish(self, failed: bool) {
        let Some(shards) = SHARDS.get() else {
            return;
        };
        let (index, bit) = shard_and_bit(self.pointer);
        let shard = &shards[index];
        let mut entries = shard.entries.lock().unwrap_or_else(|err| err.into_inner());
        if !enabled() || !session_is_current(self.generation) {
            return;
        }
        entries.reservations -= 1;
        if failed {
            // The allocator kept the original pointer alive. Its reserved slot
            // guarantees rollback even if other threads filled this shard.
            entries.samples.insert(self.pointer, self.sample);
            entries.add_membership(shard, bit);
        }
    }
}

pub(super) fn snapshot() -> Vec<LiveAllocationSample> {
    if !enabled() {
        return Vec::new();
    }
    let Some(shards) = SHARDS.get() else {
        return Vec::new();
    };
    let mut samples = Vec::new();
    for shard in shards {
        // Allocate the output storage before taking a shard lock. Samples are
        // copied under the lock, with aggregation/symbolization done afterward.
        let capacity = shard
            .entries
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .capacity;
        samples.reserve(capacity);
        let entries = shard.entries.lock().unwrap_or_else(|err| err.into_inner());
        samples.extend(entries.samples.values().copied());
    }
    samples
}

pub(super) fn sample_count() -> usize {
    SHARDS
        .get()
        .map(|shards| {
            shards
                .iter()
                .map(|shard| shard.count.load(Ordering::Relaxed))
                .sum()
        })
        .unwrap_or(0)
}

pub(super) fn dropped_sample_count() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

pub(super) fn metadata_payload_bytes() -> usize {
    METADATA_PAYLOAD_BYTES.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inclusion_weights_cover_small_and_large_allocations() {
        let (objects, bytes) = weights(1, 1024);
        assert!((objects - 1024.5).abs() < 0.001);
        assert_eq!(objects, bytes);
        assert_eq!(weights(1024 * 1024, 1024), (1.0, (1024 * 1024) as f64));
        let (objects, bytes) = weights(512, 1024);
        assert!((objects - 2.5414940825).abs() < 0.000001);
        assert!((bytes - objects * 512.0).abs() < 0.000001);
    }
}
