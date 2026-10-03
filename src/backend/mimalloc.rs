//! Allocation profiling backend for applications that install
//! [`SamplingMiMalloc`] as their global allocator.
//!
//! The allocator hot path records sampled allocation events into fixed TLS
//! rings with best-effort, non-blocking handoff to sharded global buffers.
//! `report()` is the non-hot path: it may block briefly to drain registered TLS
//! rings, aggregate samples, resolve symbols, and encode memory pprof data.
//! Optional live heap tracking maintains sampled pointers in bounded shards.

mod live;

use std::{
    alloc::{GlobalAlloc, Layout},
    cell::Cell,
    collections::{HashMap, VecDeque},
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, LazyLock, Mutex,
    },
    time::Instant,
};

use rustfs_mimalloc as mimalloc;

use crate::{
    backend::{Backend, BackendImpl, BackendUninitialized, ReportBatch, ReportData, ThreadTag},
    encode::memory_pprof::{self, AllocationSample},
    error::{PyroscopeError, Result},
};

const LOG_TAG: &str = "Pyroscope::Mimalloc";
const DEFAULT_SAMPLE_INTERVAL_BYTES: u64 = 1024 * 1024;
const DEFAULT_MAX_DEPTH: usize = 64;
const DEFAULT_RING_CAPACITY: usize = 512;
const DEFAULT_REPORT_DRAIN_LIMIT: usize = 1_000_000;
const DEFAULT_MAX_LIVE_SAMPLES: usize = 16_384;
const MAX_CAPTURE_DEPTH: usize = 64;
const PROFILER_FRAME_ALLOWANCE: usize = 16;
const TLS_SAMPLE_RING_CAPACITY: usize = 64;
const RECORDED_SAMPLE_SHARD_COUNT: usize = 8;
const SYNTHETIC_FRAME: &str = "[mimalloc] unresolved allocation stack";
const RNG_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;
const RNG_INITIAL_STATE: u64 = 0xa076_1d64_78bd_642f;
// Keep worst-case work bounded inside the allocator hook. Very large
// allocations keep the first intervals stochastic, then fall back to a
// deterministic approximation instead of looping once per sampled interval.
const MAX_POISSON_INTERVALS_PER_ALLOCATION: u64 = 64;

static RECORDER_ACTIVE: AtomicBool = AtomicBool::new(false);
static BACKEND_CLAIMED: AtomicBool = AtomicBool::new(false);
static CAPTURE_DEPTH: AtomicUsize = AtomicUsize::new(MAX_CAPTURE_DEPTH);
static ALLOCATOR_SEEN: AtomicBool = AtomicBool::new(false);
static SAMPLE_INTERVAL_BYTES: AtomicU64 = AtomicU64::new(DEFAULT_SAMPLE_INTERVAL_BYTES);
static SAMPLING_CONFIG_GENERATION: AtomicU64 = AtomicU64::new(0);
static SAMPLING_RNG_SEED: AtomicU64 = AtomicU64::new(RNG_INITIAL_STATE);
static FLUSH_REQUEST_GENERATION: AtomicU64 = AtomicU64::new(0);
static MAX_RECORDED_SAMPLES: AtomicUsize = AtomicUsize::new(DEFAULT_RING_CAPACITY);
static GLOBAL_BUFFERED_SAMPLE_COUNT: AtomicUsize = AtomicUsize::new(0);
static NEXT_RECORDED_SAMPLE_SHARD: AtomicUsize = AtomicUsize::new(0);
static NEXT_DRAINED_SAMPLE_SHARD: AtomicUsize = AtomicUsize::new(0);
static NEXT_REPORT_SOURCE: AtomicUsize = AtomicUsize::new(0);
static NEXT_TLS_DRAIN_BUFFER: AtomicUsize = AtomicUsize::new(0);
static RECORDED_SAMPLE_COUNT: AtomicU64 = AtomicU64::new(0);
static FLUSH_COUNT: AtomicU64 = AtomicU64::new(0);
static FLUSHED_SAMPLE_COUNT: AtomicU64 = AtomicU64::new(0);
static DROPPED_SAMPLES: AtomicU64 = AtomicU64::new(0);
static LAST_PPROF_ENCODE_ELAPSED_MICROS: AtomicU64 = AtomicU64::new(0);

static RECORDED_SAMPLE_SHARDS: LazyLock<Vec<Mutex<VecDeque<RecordedAllocationSample>>>> =
    LazyLock::new(|| {
        (0..RECORDED_SAMPLE_SHARD_COUNT)
            .map(|_| Mutex::new(VecDeque::new()))
            .collect()
    });
static TLS_SAMPLE_BUFFER_REGISTRY: LazyLock<Mutex<TlsSampleBufferRegistry>> =
    LazyLock::new(|| Mutex::new(TlsSampleBufferRegistry::new()));

#[derive(Debug, Copy, Clone)]
struct SamplerState {
    in_profiler: bool,
    profiler_suppressed: bool,
    has_buffer: bool,
    remaining_bytes: u64,
    remaining_config_generation: u64,
    rng_state: u64,
    flush_generation: u64,
}

impl SamplerState {
    const fn new() -> Self {
        Self {
            in_profiler: false,
            profiler_suppressed: false,
            has_buffer: false,
            remaining_bytes: DEFAULT_SAMPLE_INTERVAL_BYTES,
            remaining_config_generation: 0,
            rng_state: 0,
            flush_generation: 0,
        }
    }
}

thread_local! {
    // Keep all allocation-hook state in one TLS cell so the hot path pays one
    // state lookup instead of one lookup per guard, sampler, RNG, and flush flag.
    static SAMPLER_STATE: Cell<SamplerState> = const { Cell::new(SamplerState::new()) };
    static TLS_SAMPLE_BUFFER: RegisteredTlsSampleBuffer = RegisteredTlsSampleBuffer::new();
}

#[derive(Debug, Copy, Clone)]
struct StackKey {
    frames: [usize; MAX_CAPTURE_DEPTH],
    depth: usize,
}

impl PartialEq for StackKey {
    fn eq(&self, other: &Self) -> bool {
        self.frames[..self.depth] == other.frames[..other.depth]
    }
}

impl Eq for StackKey {}

impl Hash for StackKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Unused slots do not identify a stack. Hash only captured addresses,
        // particularly for shallow stacks during report aggregation.
        self.frames[..self.depth].hash(state);
    }
}

impl StackKey {
    fn capture(max_depth: usize) -> Self {
        let mut key = Self {
            frames: [0; MAX_CAPTURE_DEPTH],
            depth: 0,
        };
        let max_depth = max_depth.min(MAX_CAPTURE_DEPTH);

        backtrace::trace(|frame| {
            if key.depth >= max_depth {
                return false;
            }
            key.frames[key.depth] = frame.ip() as usize;
            key.depth += 1;
            true
        });

        key
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.frames[..self.depth].iter().copied()
    }
}

#[derive(Debug, Copy, Clone)]
struct RecordedAllocationSample {
    stack: StackKey,
    weighted_objects: u64,
    weighted_bytes: u64,
}

#[derive(Debug)]
struct TlsSampleBuffer {
    samples: [Option<RecordedAllocationSample>; TLS_SAMPLE_RING_CAPACITY],
    len: usize,
    generation: u64,
}

impl TlsSampleBuffer {
    fn new() -> Self {
        Self {
            samples: [None; TLS_SAMPLE_RING_CAPACITY],
            len: 0,
            generation: SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire),
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn is_full(&self) -> bool {
        self.len == TLS_SAMPLE_RING_CAPACITY
    }

    fn push(&mut self, sample: RecordedAllocationSample) -> bool {
        if self.is_full() {
            return false;
        }

        self.samples[self.len] = Some(sample);
        self.len += 1;
        true
    }

    fn clear(&mut self) {
        for sample in &mut self.samples[..self.len] {
            *sample = None;
        }
        self.len = 0;
    }

    fn drain_into(
        &mut self,
        out: &mut impl Extend<RecordedAllocationSample>,
        limit: usize,
    ) -> usize {
        let drain_len = self.len.min(limit);
        for index in 0..drain_len {
            if let Some(sample) = self.samples[index].take() {
                out.extend(std::iter::once(sample));
            }
        }

        let remaining = self.len - drain_len;
        for index in 0..remaining {
            self.samples[index] = self.samples[drain_len + index].take();
        }
        for index in remaining..self.len {
            self.samples[index] = None;
        }
        self.len = remaining;

        drain_len
    }
}

#[derive(Debug)]
struct RegisteredTlsSampleBuffer {
    id: Cell<Option<usize>>,
    buffer: Arc<Mutex<TlsSampleBuffer>>,
}

impl RegisteredTlsSampleBuffer {
    fn new() -> Self {
        let buffer = Arc::new(Mutex::new(TlsSampleBuffer::new()));
        let id = register_tls_sample_buffer(buffer.clone());
        Self {
            id: Cell::new(id),
            buffer,
        }
    }

    fn ensure_registered(&self) {
        if self.id.get().is_none() {
            self.id.set(register_tls_sample_buffer(self.buffer.clone()));
        }
    }

    fn try_lock(&self) -> Option<std::sync::MutexGuard<'_, TlsSampleBuffer>> {
        let _ = SAMPLER_STATE.try_with(|sampler| {
            let mut state = sampler.get();
            state.has_buffer = true;
            sampler.set(state);
        });
        self.ensure_registered();
        self.buffer.try_lock().ok()
    }
}

impl Drop for RegisteredTlsSampleBuffer {
    fn drop(&mut self) {
        with_profiler_suppressed(|| {
            if RECORDER_ACTIVE.load(Ordering::Acquire) {
                if let Ok(mut buffer) = self.buffer.lock() {
                    if !flush_tls_samples_for_report(&mut buffer) {
                        drop_tls_samples(&mut buffer);
                    }
                }
            }
            // Deregister only after the final handoff attempt so a concurrent
            // report can still see this thread's ring until thread teardown.
            if let Some(id) = self.id.get() {
                deregister_tls_sample_buffer(id);
            }
        });
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct AggregatedAllocationSample {
    alloc_objects: u64,
    alloc_space: u64,
    inuse_objects: f64,
    inuse_space: f64,
}

/// Configuration for the mimalloc allocation memory profiling backend.
///
/// The backend records sampled allocation call stacks and reports memory pprof
/// data with `alloc_objects/count` and `alloc_space/bytes` sample types. Enabling
/// `live_heap_tracking` adds `inuse_objects/count` and `inuse_space/bytes`, with
/// `inuse_space` selected by default, matching jemalloc's live heap view.
/// Samples whose frames
/// cannot be resolved are grouped under a synthetic fallback frame.
///
/// # Examples
///
/// ```rust
/// use pyroscope::backend::mimalloc::MimallocConfig;
///
/// let config = MimallocConfig {
///     sample_interval_bytes: 512 * 1024,
///     max_depth: 48,
///     ..MimallocConfig::default()
/// };
///
/// assert_eq!(config.sample_interval_bytes, 512 * 1024);
/// assert_eq!(config.max_depth, 48);
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MimallocConfig {
    /// Average number of allocated bytes between sampled allocation events.
    ///
    /// The sampler uses a byte-based Poisson process, so this value is the mean
    /// interval rather than a fixed every-N-bytes trigger. Lower values increase
    /// profile detail and hot-path overhead.
    pub sample_interval_bytes: u64,
    /// Maximum number of stack frames captured for each sampled allocation.
    pub max_depth: usize,
    /// Maximum number of samples retained in the global recorder between reports.
    ///
    /// If the recorder is full or contended, new samples are dropped rather than
    /// blocking the allocator hot path.
    pub ring_capacity: usize,
    /// Maximum number of allocation samples drained by one `report()` call.
    ///
    /// A bounded drain keeps large bursts from making a single report interval do
    /// unbounded aggregation and pprof encoding work.
    pub report_drain_limit: usize,
    /// Track sampled live pointers and emit a live heap snapshot on every report.
    ///
    /// Disabled by default. This adds a sharded lookup on deallocation and
    /// successful reallocation. Removals may briefly wait for a shard lock so
    /// cross-thread frees never leave stale live entries. Only Rust allocations
    /// made through `SamplingMiMalloc` after initialization are tracked.
    /// Same-address realloc preserves the original allocation stack and object
    /// weight while updating size. Moving realloc samples the replacement
    /// independently, without depending on the allocation-event byte remainder.
    pub live_heap_tracking: bool,
    /// Maximum number of sampled live pointers retained across all shards.
    ///
    /// Capacity is partitioned across shards and preallocated at initialization.
    /// A full or contended shard drops new live samples, counted separately in
    /// `MimallocStats::dropped_live_samples`. Independent of `ring_capacity` and
    /// `report_drain_limit`; live snapshots always include every retained entry.
    pub max_live_samples: usize,
}

/// Runtime counters for the mimalloc memory profiling backend.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MimallocStats {
    /// Number of samples accepted into the recorder since backend initialization.
    pub recorded_samples: u64,
    /// Number of successful TLS-to-global or TLS-to-report flushes since initialization.
    pub flushes: u64,
    /// Number of samples moved from TLS rings into global buffers or reports.
    pub flushed_samples: u64,
    /// Number of sample records dropped because the recorder was full or locked.
    pub dropped_samples: u64,
    /// Number of samples currently buffered for the next report, if the buffer lock is available.
    pub buffered_samples: Option<usize>,
    /// Duration of the most recent pprof encoding step in microseconds.
    pub last_pprof_encode_elapsed_micros: u64,
    /// Number of sampled live pointers currently retained (zero when disabled).
    pub live_samples: usize,
    /// Live samples omitted because their metadata shard was full or contended.
    pub dropped_live_samples: u64,
    /// Preallocated live-table entry storage in bytes, excluding hash control
    /// bytes, shard headers, and allocator bookkeeping. Zero when disabled.
    pub live_metadata_payload_bytes: usize,
}

/// Return current mimalloc backend recorder counters.
///
/// This is mainly intended for tests, diagnostics, and benchmark reports.
pub fn mimalloc_stats() -> MimallocStats {
    let tls_buffered_samples = registered_tls_buffered_samples();

    MimallocStats {
        recorded_samples: RECORDED_SAMPLE_COUNT.load(Ordering::Relaxed),
        flushes: FLUSH_COUNT.load(Ordering::Relaxed),
        flushed_samples: FLUSHED_SAMPLE_COUNT.load(Ordering::Relaxed),
        dropped_samples: DROPPED_SAMPLES.load(Ordering::Relaxed),
        buffered_samples: tls_buffered_samples.map(|tls| {
            GLOBAL_BUFFERED_SAMPLE_COUNT
                .load(Ordering::Relaxed)
                .saturating_add(tls)
        }),
        last_pprof_encode_elapsed_micros: LAST_PPROF_ENCODE_ELAPSED_MICROS.load(Ordering::Relaxed),
        live_samples: live::sample_count(),
        dropped_live_samples: live::dropped_sample_count(),
        live_metadata_payload_bytes: live::metadata_payload_bytes(),
    }
}

impl Default for MimallocConfig {
    fn default() -> Self {
        Self {
            sample_interval_bytes: DEFAULT_SAMPLE_INTERVAL_BYTES,
            max_depth: DEFAULT_MAX_DEPTH,
            ring_capacity: DEFAULT_RING_CAPACITY,
            report_drain_limit: DEFAULT_REPORT_DRAIN_LIMIT,
            live_heap_tracking: false,
            max_live_samples: DEFAULT_MAX_LIVE_SAMPLES,
        }
    }
}

impl MimallocConfig {
    fn validate(&self) -> Result<()> {
        if self.sample_interval_bytes == 0 {
            return Err(PyroscopeError::new(
                "mimalloc: sample_interval_bytes must be greater than zero",
            ));
        }
        if self.max_depth == 0 || self.max_depth > MAX_CAPTURE_DEPTH {
            return Err(PyroscopeError::new(
                "mimalloc: max_depth must be between 1 and 64",
            ));
        }
        if self.ring_capacity == 0 {
            return Err(PyroscopeError::new(
                "mimalloc: ring_capacity must be greater than zero",
            ));
        }
        if self.report_drain_limit == 0 {
            return Err(PyroscopeError::new(
                "mimalloc: report_drain_limit must be greater than zero",
            ));
        }
        if self.live_heap_tracking && self.max_live_samples == 0 {
            return Err(PyroscopeError::new("mimalloc: max_live_samples must be greater than zero when live heap tracking is enabled"));
        }
        Ok(())
    }
}

/// A mimalloc global allocator wrapper that records allocation samples.
///
/// Use this type as the application's global allocator when enabling
/// `backend-mimalloc`:
///
/// ```rust
/// use pyroscope::backend::mimalloc::SamplingMiMalloc;
///
/// #[global_allocator]
/// static ALLOC: SamplingMiMalloc = SamplingMiMalloc::new();
/// ```
///
/// The backend cannot record allocation call stacks when an application uses
/// `rustfs_mimalloc::MiMalloc` directly.
pub struct SamplingMiMalloc {
    inner: mimalloc::MiMalloc,
}

impl SamplingMiMalloc {
    /// Create a `SamplingMiMalloc` allocator.
    ///
    /// The allocator is always safe to install, but it only records samples while
    /// a `backend-mimalloc` backend is initialized.
    pub const fn new() -> Self {
        Self {
            inner: mimalloc::MiMalloc,
        }
    }
}

impl Default for SamplingMiMalloc {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for SamplingMiMalloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        mark_allocator_seen();
        // SAFETY: `SamplingMiMalloc` preserves the caller's `GlobalAlloc`
        // contract and forwards the exact layout to the wrapped mimalloc
        // allocator.
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            record_allocation(ptr, layout.size() as u64);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        mark_allocator_seen();
        // SAFETY: The caller provided a valid `GlobalAlloc` layout; this wrapper
        // only forwards it to mimalloc and records after successful allocation.
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_allocation(ptr, layout.size() as u64);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Remove before freeing: another thread may immediately reuse this
        // address once mimalloc receives it.
        if live::enabled() {
            remove_live_allocation(ptr as usize);
        }
        // SAFETY: Deallocation is forwarded unchanged; callers must pass a
        // pointer and layout that satisfy the `GlobalAlloc::dealloc` contract.
        unsafe { self.inner.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        mark_allocator_seen();
        // Keep the metadata slot reserved before mimalloc can recycle the old
        // address. Lifecycle maintenance must also run while sampling is suppressed.
        let pending = if live::enabled() {
            with_profiler_suppressed(|| live::begin_reallocation(ptr as usize))
        } else {
            None
        };
        // SAFETY: Reallocation is forwarded unchanged to mimalloc with the
        // caller-provided pointer, old layout, and requested new size.
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        let previous = pending.and_then(|pending| {
            with_profiler_suppressed(|| pending.finish(new_ptr as usize, new_size as u64))
        });
        if !new_ptr.is_null() {
            if new_ptr != ptr && live::enabled() {
                record_live_reallocation(new_ptr as usize, new_size as u64, previous);
            }
            let recorded_size = realloc_recorded_size(ptr, new_ptr, layout.size(), new_size);
            // Live lifecycle updates and replacement sampling are independent
            // of the event sampler, which charges only newly allocated bytes.
            record_allocation(std::ptr::null_mut(), recorded_size as u64);
        }
        new_ptr
    }
}

/// Create a mimalloc allocation memory profiling backend.
///
/// The returned backend should be passed to `PyroscopeAgentBuilder::new`, and
/// the process must install [`SamplingMiMalloc`] as its global allocator.
///
/// # Examples
///
/// ```no_run
/// use pyroscope::backend::mimalloc::{
///     mimalloc_backend, MimallocConfig, SamplingMiMalloc,
/// };
/// use pyroscope::pyroscope::PyroscopeAgentBuilder;
///
/// #[global_allocator]
/// static ALLOC: SamplingMiMalloc = SamplingMiMalloc::new();
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let agent = PyroscopeAgentBuilder::new(
///     "http://localhost:4040",
///     "example.mimalloc",
///     100,
///     "pyroscope-rs",
///     env!("CARGO_PKG_VERSION"),
///     mimalloc_backend(MimallocConfig::default()),
/// )
/// .build()?;
/// # let _ = agent;
/// # Ok(())
/// # }
/// ```
pub fn mimalloc_backend(config: MimallocConfig) -> BackendImpl<BackendUninitialized> {
    BackendImpl::new(Box::new(Mimalloc::new(config)))
}

#[derive(Debug)]
struct Mimalloc {
    config: MimallocConfig,
    last_report: Option<Instant>,
    initialized: bool,
}

impl Mimalloc {
    fn new(config: MimallocConfig) -> Self {
        Self {
            config,
            last_report: None,
            initialized: false,
        }
    }

    fn stop(&mut self) {
        if !self.initialized {
            return;
        }
        RECORDER_ACTIVE.store(false, Ordering::Release);
        with_profiler_suppressed(|| {
            clear_registered_tls_samples();
            live::clear();
        });
        self.initialized = false;
        BACKEND_CLAIMED.store(false, Ordering::Release);
    }
}

impl Drop for Mimalloc {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Backend for Mimalloc {
    fn initialize(&mut self) -> Result<()> {
        self.config.validate()?;
        if !ALLOCATOR_SEEN.load(Ordering::Relaxed) {
            return Err(PyroscopeError::new("mimalloc: SamplingMiMalloc has not observed allocations; install it as #[global_allocator] before initializing the backend"));
        }
        BACKEND_CLAIMED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| PyroscopeError::new("mimalloc: another backend is already initialized"))?;
        self.initialized = true;
        with_profiler_suppressed(|| {
            SAMPLE_INTERVAL_BYTES.store(self.config.sample_interval_bytes, Ordering::Relaxed);
            SAMPLING_CONFIG_GENERATION.fetch_add(1, Ordering::AcqRel);
            CAPTURE_DEPTH.store(
                self.config
                    .max_depth
                    .saturating_add(PROFILER_FRAME_ALLOWANCE)
                    .min(MAX_CAPTURE_DEPTH),
                Ordering::Relaxed,
            );
            NEXT_RECORDED_SAMPLE_SHARD.store(0, Ordering::Relaxed);
            NEXT_DRAINED_SAMPLE_SHARD.store(0, Ordering::Relaxed);
            NEXT_REPORT_SOURCE.store(0, Ordering::Relaxed);
            NEXT_TLS_DRAIN_BUFFER.store(0, Ordering::Relaxed);
            MAX_RECORDED_SAMPLES.store(self.config.ring_capacity, Ordering::Relaxed);
            RECORDED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
            FLUSH_COUNT.store(0, Ordering::Relaxed);
            FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
            DROPPED_SAMPLES.store(0, Ordering::Relaxed);
            LAST_PPROF_ENCODE_ELAPSED_MICROS.store(0, Ordering::Relaxed);
            prepare_sample_buffer(self.config.ring_capacity);
            // A backend can be stopped and started again while worker threads keep
            // their TLS rings alive. Clear registered rings at the session boundary
            // so old samples cannot be flushed into the next profile interval.
            clear_registered_tls_samples();
            reset_current_thread_sample_buffer();
            warm_backtrace();
            live::prepare(self.config.live_heap_tracking, self.config.max_live_samples);
            RECORDER_ACTIVE.store(true, Ordering::Release);
            self.last_report = Some(Instant::now());
        });

        log::info!(target: LOG_TAG, "Mimalloc profiling backend initialized");
        Ok(())
    }

    fn shutdown(mut self: Box<Self>) -> Result<()> {
        self.stop();
        log::trace!(target: LOG_TAG, "Shutting down mimalloc backend");
        Ok(())
    }

    fn report(&mut self) -> Result<ReportBatch> {
        // Reporting allocates for registry snapshots, aggregation, symbol
        // resolution, and pprof encoding. Suppressing this thread avoids
        // profiler self-sampling without disabling worker-thread sampling.
        with_profiler_suppressed(|| {
            let now = Instant::now();
            let duration_nanos = self
                .last_report
                .replace(now)
                .map(|last_report| duration_to_i64_nanos(now.duration_since(last_report)))
                .unwrap_or_default();

            request_tls_sample_flush();
            let recorded = drain_samples_for_report(self.config.report_drain_limit);
            let recorded_count = recorded.len();
            let dropped_count = DROPPED_SAMPLES.load(Ordering::Relaxed);
            if dropped_count > 0 {
                log::debug!(
                    target: LOG_TAG,
                    "Mimalloc report drained {recorded_count} samples; {dropped_count} samples have been dropped since initialization"
                );
            }

            let samples = build_memory_samples(recorded, live::snapshot(), self.config.max_depth);

            let encode_start = Instant::now();
            let pprof_data = memory_pprof::encode_memory_profile(
                &samples,
                self.config.sample_interval_bytes,
                duration_nanos,
                self.config.live_heap_tracking,
            );
            LAST_PPROF_ENCODE_ELAPSED_MICROS.store(
                duration_to_u64_micros(encode_start.elapsed()),
                Ordering::Relaxed,
            );

            Ok(ReportBatch {
                profile_type: "memory".into(),
                data: ReportData::RawPprof(pprof_data),
            })
        })
    }

    fn add_tag(&self, _tag: ThreadTag) -> Result<()> {
        Ok(())
    }

    fn remove_tag(&self, _tag: ThreadTag) -> Result<()> {
        Ok(())
    }
}

fn mark_allocator_seen() {
    if !ALLOCATOR_SEEN.load(Ordering::Relaxed) {
        ALLOCATOR_SEEN.store(true, Ordering::Relaxed);
    }
}

// Keep TLS suppression and table lookups out of the default allocator's
// deallocation body, including their stack/register requirements.
#[inline(never)]
fn remove_live_allocation(pointer: usize) {
    with_profiler_suppressed(|| live::remove(pointer));
}

fn with_profiler_suppressed<R>(f: impl FnOnce() -> R) -> R {
    struct SuppressionGuard<'a> {
        sampler: &'a Cell<SamplerState>,
        previous: bool,
    }

    impl Drop for SuppressionGuard<'_> {
        fn drop(&mut self) {
            let mut state = self.sampler.get();
            state.profiler_suppressed = self.previous;
            self.sampler.set(state);
        }
    }

    let mut f = Some(f);
    match SAMPLER_STATE.try_with(|sampler| {
        let mut state = sampler.get();
        let previous = state.profiler_suppressed;
        state.profiler_suppressed = true;
        sampler.set(state);

        let _guard = SuppressionGuard { sampler, previous };
        f.take().expect("suppression closure was already taken")()
    }) {
        Ok(result) => result,
        Err(_) => f
            .take()
            .expect("suppression fallback closure was already taken")(),
    }
}

fn realloc_recorded_size(
    old_ptr: *mut u8,
    new_ptr: *mut u8,
    old_size: usize,
    new_size: usize,
) -> usize {
    if old_ptr == new_ptr {
        new_size.saturating_sub(old_size)
    } else {
        new_size
    }
}

fn with_recording_guard<R>(f: impl FnOnce(&Cell<SamplerState>) -> R) -> Option<R> {
    struct ProfilerReentryGuard<'a> {
        sampler: &'a Cell<SamplerState>,
    }

    impl Drop for ProfilerReentryGuard<'_> {
        fn drop(&mut self) {
            let mut state = self.sampler.get();
            state.in_profiler = false;
            self.sampler.set(state);
        }
    }

    SAMPLER_STATE
        .try_with(|sampler| {
            let mut state = sampler.get();
            if state.profiler_suppressed || state.in_profiler {
                return None;
            }

            state.in_profiler = true;
            sampler.set(state);
            let _guard = ProfilerReentryGuard { sampler };
            Some(f(sampler))
        })
        .ok()
        .flatten()
}

fn record_allocation(ptr: *mut u8, size: u64) {
    if size == 0 || !RECORDER_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);

    with_recording_guard(|sampler| {
        let mut state = sampler.get();

        let interval = SAMPLE_INTERVAL_BYTES.load(Ordering::Relaxed).max(1);
        let mut current = state.remaining_bytes;
        if state.remaining_config_generation != generation || current == 0 {
            if state.has_buffer {
                clear_current_thread_samples();
            }
            state.rng_state = next_thread_rng_seed();
            current = next_poisson_interval(interval, &mut state.rng_state);
            state.remaining_config_generation = generation;
        }
        flush_requested_tls_samples_with_state(&mut state);

        if size < current {
            state.remaining_bytes = current - size;
            sampler.set(state);
        } else {
            let weight = calculate_sample_weight(size, current, interval, &mut state.rng_state);
            state.remaining_bytes = weight.next_remaining;
            sampler.set(state);
            record_sample(ptr, size, weight, generation);
        }
    });
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct SampleWeight {
    weighted_objects: u64,
    weighted_bytes: u64,
    next_remaining: u64,
}

fn calculate_sample_weight(
    size: u64,
    current: u64,
    sample_interval: u64,
    rng_state: &mut u64,
) -> SampleWeight {
    let sample_interval = sample_interval.max(1);
    let mut remaining_bytes = size.saturating_sub(current.max(1));
    let mut crossed_intervals = 1_u64;
    let mut next_remaining = next_poisson_interval(sample_interval, rng_state);

    while remaining_bytes >= next_remaining
        && crossed_intervals < MAX_POISSON_INTERVALS_PER_ALLOCATION
    {
        remaining_bytes -= next_remaining;
        crossed_intervals = crossed_intervals.saturating_add(1);
        next_remaining = next_poisson_interval(sample_interval, rng_state);
    }

    if remaining_bytes >= next_remaining {
        remaining_bytes -= next_remaining;
        crossed_intervals = crossed_intervals.saturating_add(1);
        // Past the bounded stochastic prefix, approximate the rest of an
        // unusually large allocation with fixed-size intervals. This preserves
        // proportional weighting while keeping hook latency predictable.
        let deterministic_intervals = remaining_bytes / sample_interval;
        crossed_intervals = crossed_intervals.saturating_add(deterministic_intervals);
        let bytes_into_next_interval = remaining_bytes % sample_interval;
        next_remaining = if bytes_into_next_interval == 0 {
            sample_interval
        } else {
            sample_interval - bytes_into_next_interval
        };
    } else {
        next_remaining -= remaining_bytes;
    }

    let weighted_bytes = crossed_intervals.saturating_mul(sample_interval);
    let weighted_objects = weighted_bytes.checked_div(size).unwrap_or_default().max(1);

    SampleWeight {
        weighted_objects,
        weighted_bytes,
        next_remaining,
    }
}

fn next_poisson_interval(sample_interval: u64, rng_state: &mut u64) -> u64 {
    let sample_interval = sample_interval.max(1);
    let random = next_random_u64(rng_state);
    let mantissa = (random >> 11).max(1);
    let uniform = mantissa as f64 * (1.0 / ((1_u64 << 53) as f64));
    let interval = -(uniform.ln()) * sample_interval as f64;

    if interval.is_finite() && interval < u64::MAX as f64 {
        (interval.ceil() as u64).max(1)
    } else {
        u64::MAX
    }
}

fn next_random_u64(rng_state: &mut u64) -> u64 {
    if *rng_state == 0 {
        *rng_state = next_thread_rng_seed();
    }
    *rng_state = (*rng_state).wrapping_add(RNG_INCREMENT);
    splitmix64(*rng_state)
}

fn next_thread_rng_seed() -> u64 {
    let seed = SAMPLING_RNG_SEED.fetch_add(RNG_INCREMENT, Ordering::Relaxed);
    splitmix64(seed.wrapping_add(RNG_INCREMENT))
}

fn splitmix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
fn calculate_deterministic_sample_weight(size: u64, current: u64, interval: u64) -> SampleWeight {
    let interval = interval.max(1);
    let current = current.clamp(1, interval);
    let bytes_after_first_sample = size.saturating_sub(current);
    let crossed_intervals = bytes_after_first_sample
        .checked_div(interval)
        .unwrap_or_default()
        .saturating_add(1);
    let weighted_bytes = crossed_intervals.saturating_mul(interval);
    let weighted_objects = weighted_bytes.checked_div(size).unwrap_or_default().max(1);
    let bytes_into_next_interval = bytes_after_first_sample % interval;
    let next_remaining = if bytes_into_next_interval == 0 {
        interval
    } else {
        interval - bytes_into_next_interval
    };

    SampleWeight {
        weighted_objects,
        weighted_bytes,
        next_remaining,
    }
}

fn session_is_current(generation: u64) -> bool {
    RECORDER_ACTIVE.load(Ordering::Acquire)
        && SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire) == generation
}

fn record_live_reallocation(
    pointer: usize,
    size: u64,
    previous: Option<(live::LiveAllocationSample, u64)>,
) {
    if !RECORDER_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
    let sampled = with_recording_guard(|sampler| {
        let mut state = sampler.get();
        if state.rng_state == 0 {
            state.rng_state = next_thread_rng_seed();
        }
        let interval = SAMPLE_INTERVAL_BYTES.load(Ordering::Relaxed).max(1);
        let hit = next_poisson_interval(interval, &mut state.rng_state) <= size;
        sampler.set(state);
        if hit {
            let stack = StackKey::capture(CAPTURE_DEPTH.load(Ordering::Relaxed));
            live::record(pointer, stack, size, interval, generation);
        }
    });
    if sampled.is_none() {
        // Suppression prevents capturing new profiler allocations, but never
        // abandons the lifecycle of a pointer that was already tracked.
        if let Some((sample, generation)) = previous {
            with_profiler_suppressed(|| live::transfer_sample(pointer, sample, generation));
        }
    }
}

fn record_sample(ptr: *mut u8, size: u64, weight: SampleWeight, generation: u64) {
    let tracking_live = live::enabled() && !ptr.is_null();
    let mut stack = None;
    let _ = TLS_SAMPLE_BUFFER.try_with(|buffer| {
        let Some(mut buffer) = buffer.try_lock() else {
            DROPPED_SAMPLES.fetch_add(1, Ordering::Relaxed);
            return;
        };
        // Check while holding the ring lock so shutdown/restart cannot admit
        // a stack captured for the previous profiling session.
        if !session_is_current(generation) {
            return;
        }
        if buffer.generation != generation {
            buffer.clear();
            buffer.generation = generation;
        }

        if buffer.is_full() {
            flush_tls_samples(&mut buffer);
        }
        if buffer.is_full() && !tracking_live {
            DROPPED_SAMPLES.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let captured = StackKey::capture(CAPTURE_DEPTH.load(Ordering::Relaxed));
        stack = Some(captured);
        let sample = RecordedAllocationSample {
            stack: captured,
            weighted_objects: weight.weighted_objects,
            weighted_bytes: weight.weighted_bytes,
        };

        if buffer.push(sample) {
            RECORDED_SAMPLE_COUNT.fetch_add(1, Ordering::Relaxed);
        } else {
            DROPPED_SAMPLES.fetch_add(1, Ordering::Relaxed);
        }
    });
    // The allocation event queue and live table have independent budgets.
    // Dropping an interval event must not suppress an otherwise valid live hit.
    if tracking_live && session_is_current(generation) {
        let stack =
            stack.unwrap_or_else(|| StackKey::capture(CAPTURE_DEPTH.load(Ordering::Relaxed)));
        live::record(
            ptr as usize,
            stack,
            size,
            SAMPLE_INTERVAL_BYTES.load(Ordering::Relaxed),
            generation,
        );
    }
}

fn flush_current_thread_samples() -> bool {
    TLS_SAMPLE_BUFFER
        .try_with(|buffer| {
            let Some(mut buffer) = buffer.try_lock() else {
                return false;
            };
            flush_tls_samples(&mut buffer)
        })
        .unwrap_or(false)
}

fn request_tls_sample_flush() {
    FLUSH_REQUEST_GENERATION.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
fn flush_requested_tls_samples() {
    SAMPLER_STATE.with(|sampler| {
        let mut state = sampler.get();
        flush_requested_tls_samples_with_state(&mut state);
        sampler.set(state);
    });
}

fn flush_requested_tls_samples_with_state(state: &mut SamplerState) {
    let requested_generation = FLUSH_REQUEST_GENERATION.load(Ordering::Relaxed);
    if state.flush_generation == requested_generation {
        return;
    }
    // Threads without a sampling hit need no ring allocation or registry entry.
    if !state.has_buffer {
        state.flush_generation = requested_generation;
        return;
    }

    if flush_current_thread_samples() {
        state.flush_generation = requested_generation;
    }
}

fn reset_current_thread_sample_buffer() {
    let generation = FLUSH_REQUEST_GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
    SAMPLER_STATE.with(|sampler| {
        let mut state = sampler.get();
        state.flush_generation = generation;
        sampler.set(state);
    });
    clear_current_thread_samples();
}

fn clear_current_thread_samples() {
    let _ = TLS_SAMPLE_BUFFER.try_with(|buffer| {
        if let Some(mut buffer) = buffer.try_lock() {
            buffer.clear();
            buffer.generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
        }
    });
}

#[derive(Debug)]
struct TlsSampleBufferRegistry {
    buffers: Vec<Option<Arc<Mutex<TlsSampleBuffer>>>>,
    free_ids: Vec<usize>,
}

impl TlsSampleBufferRegistry {
    const fn new() -> Self {
        Self {
            buffers: Vec::new(),
            free_ids: Vec::new(),
        }
    }

    fn register(&mut self, buffer: Arc<Mutex<TlsSampleBuffer>>) -> usize {
        if let Some(id) = self.free_ids.pop() {
            self.buffers[id] = Some(buffer);
            id
        } else {
            let id = self.buffers.len();
            self.buffers.push(Some(buffer));
            id
        }
    }

    fn deregister(&mut self, id: usize) {
        if id < self.buffers.len() && self.buffers[id].take().is_some() {
            self.free_ids.push(id);
        }
    }

    fn buffers(&self) -> Vec<Arc<Mutex<TlsSampleBuffer>>> {
        self.buffers.iter().filter_map(Clone::clone).collect()
    }
}

fn register_tls_sample_buffer(buffer: Arc<Mutex<TlsSampleBuffer>>) -> Option<usize> {
    // TLS registration can happen from the allocator hook on first use. Do not
    // wait for the global registry lock there; unregistered threads still keep
    // local TLS buffering and can flush on ring pressure or thread exit.
    let Ok(mut registry) = TLS_SAMPLE_BUFFER_REGISTRY.try_lock() else {
        return None;
    };

    Some(registry.register(buffer))
}

fn deregister_tls_sample_buffer(id: usize) {
    if let Ok(mut registry) = TLS_SAMPLE_BUFFER_REGISTRY.lock() {
        registry.deregister(id);
    }
}

fn registered_tls_sample_buffers() -> Vec<Arc<Mutex<TlsSampleBuffer>>> {
    let Ok(registry) = TLS_SAMPLE_BUFFER_REGISTRY.lock() else {
        DROPPED_SAMPLES.fetch_add(1, Ordering::Relaxed);
        return Vec::new();
    };

    registry.buffers()
}

fn registered_tls_buffered_samples() -> Option<usize> {
    with_profiler_suppressed(|| {
        let buffers = registered_tls_sample_buffers();
        let mut buffered_samples = 0_usize;
        for buffer in buffers {
            let Ok(buffer) = buffer.try_lock() else {
                return None;
            };
            buffered_samples = buffered_samples.saturating_add(buffer.len());
        }
        Some(buffered_samples)
    })
}

fn count_recorded_samples() -> usize {
    RECORDED_SAMPLE_SHARDS
        .iter()
        .filter_map(|shard| match shard.lock() {
            Ok(samples) => Some(samples.len()),
            Err(_) => {
                DROPPED_SAMPLES.fetch_add(1, Ordering::Relaxed);
                None
            }
        })
        .sum()
}

#[cfg(test)]
fn flush_registered_tls_samples() {
    // `report()` is outside the allocation hot path, so it can block briefly to
    // make the profile interval deterministic instead of best-effort.
    for buffer in registered_tls_sample_buffers() {
        if let Ok(mut buffer) = buffer.lock() {
            flush_tls_samples_for_report(&mut buffer);
        }
    }
}

fn drain_registered_tls_samples_for_report(
    out: &mut Vec<RecordedAllocationSample>,
    limit: usize,
) -> usize {
    let mut drained = 0;
    let buffers = registered_tls_sample_buffers();
    if buffers.is_empty() || limit == 0 {
        return 0;
    }
    let start = NEXT_TLS_DRAIN_BUFFER.fetch_add(1, Ordering::Relaxed) % buffers.len();
    for offset in 0..buffers.len() {
        if drained == limit {
            break;
        }
        let buffer = &buffers[(start + offset) % buffers.len()];
        if let Ok(mut buffer) = buffer.lock() {
            let moved = buffer.drain_into(out, limit - drained);
            if moved > 0 {
                FLUSH_COUNT.fetch_add(1, Ordering::Relaxed);
                FLUSHED_SAMPLE_COUNT.fetch_add(moved as u64, Ordering::Relaxed);
                drained += moved;
            }
        }
    }
    drained
}

fn clear_registered_tls_samples() {
    // Used only at backend lifecycle boundaries; clear rather than flush so
    // stale samples from a previous session cannot leak into a new report.
    for buffer in registered_tls_sample_buffers() {
        if let Ok(mut buffer) = buffer.lock() {
            buffer.clear();
            buffer.generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
        }
    }
}

fn drop_tls_samples(buffer: &mut TlsSampleBuffer) {
    let dropped = buffer.len();
    if dropped > 0 {
        DROPPED_SAMPLES.fetch_add(dropped as u64, Ordering::Relaxed);
        buffer.clear();
    }
}

fn flush_tls_samples(buffer: &mut TlsSampleBuffer) -> bool {
    flush_tls_samples_to_global(buffer, GlobalSampleShardLock::Try)
}

fn flush_tls_samples_for_report(buffer: &mut TlsSampleBuffer) -> bool {
    flush_tls_samples_to_global(buffer, GlobalSampleShardLock::Blocking)
}

#[derive(Debug, Copy, Clone)]
enum GlobalSampleShardLock {
    Try,
    Blocking,
}

fn flush_tls_samples_to_global(
    buffer: &mut TlsSampleBuffer,
    lock_mode: GlobalSampleShardLock,
) -> bool {
    if buffer.is_empty() {
        return true;
    }

    let shard_index =
        NEXT_RECORDED_SAMPLE_SHARD.fetch_add(1, Ordering::Relaxed) % RECORDED_SAMPLE_SHARD_COUNT;
    let mut samples = match lock_mode {
        GlobalSampleShardLock::Try => {
            let Ok(samples) = RECORDED_SAMPLE_SHARDS[shard_index].try_lock() else {
                return false;
            };
            samples
        }
        GlobalSampleShardLock::Blocking => {
            let Ok(samples) = RECORDED_SAMPLE_SHARDS[shard_index].lock() else {
                return false;
            };
            samples
        }
    };

    // Generation validation and reservations belong inside the shard lock.
    // Initialization clears each shard under the same lock before resetting
    // the global count, so an old flush cannot reserve slots in a new session.
    if buffer.generation != SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire) {
        buffer.clear();
        return true;
    }
    let reserved_slots = reserve_global_sample_slots(buffer.len());
    if reserved_slots == 0 {
        return false;
    }

    // Keep handoffs inside the storage allocated during initialization, even
    // when contention makes occupancy uneven across shards.
    let writable_slots = reserved_slots.min(samples.capacity().saturating_sub(samples.len()));
    let flushed = buffer.drain_into(&mut *samples, writable_slots);
    if flushed < reserved_slots {
        release_global_sample_slots(reserved_slots - flushed);
    }
    if flushed > 0 {
        FLUSH_COUNT.fetch_add(1, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.fetch_add(flushed as u64, Ordering::Relaxed);
    }
    if !buffer.is_empty() {
        return false;
    }

    flushed > 0
}

fn reserve_global_sample_slots(wanted: usize) -> usize {
    let max_samples = MAX_RECORDED_SAMPLES.load(Ordering::Relaxed);
    let mut current = GLOBAL_BUFFERED_SAMPLE_COUNT.load(Ordering::Relaxed);

    loop {
        let available = max_samples.saturating_sub(current);
        if available == 0 {
            return 0;
        }

        let reserved = wanted.min(available);
        match GLOBAL_BUFFERED_SAMPLE_COUNT.compare_exchange_weak(
            current,
            current + reserved,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return reserved,
            Err(observed) => current = observed,
        }
    }
}

fn release_global_sample_slots(slots: usize) {
    if slots == 0 {
        return;
    }

    let mut current = GLOBAL_BUFFERED_SAMPLE_COUNT.load(Ordering::Relaxed);
    loop {
        let release = slots.min(current);
        if release == 0 {
            return;
        }

        match GLOBAL_BUFFERED_SAMPLE_COUNT.compare_exchange_weak(
            current,
            current - release,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

fn prepare_sample_buffer(capacity: usize) {
    for shard in RECORDED_SAMPLE_SHARDS.iter() {
        let Ok(mut samples) = shard.lock() else {
            DROPPED_SAMPLES.fetch_add(1, Ordering::Relaxed);
            continue;
        };
        samples.clear();
        let current_capacity = samples.capacity();
        let shard_capacity = recorded_sample_shard_capacity(capacity);
        if current_capacity < shard_capacity {
            samples.reserve(shard_capacity);
        }
    }
    GLOBAL_BUFFERED_SAMPLE_COUNT.store(count_recorded_samples(), Ordering::Relaxed);
}

fn drain_recorded_samples(limit: usize) -> Vec<RecordedAllocationSample> {
    let mut drained = Vec::new();
    let mut remaining = limit;
    if remaining == 0 {
        return drained;
    }

    let start_shard =
        NEXT_DRAINED_SAMPLE_SHARD.fetch_add(1, Ordering::Relaxed) % RECORDED_SAMPLE_SHARD_COUNT;
    for offset in 0..RECORDED_SAMPLE_SHARD_COUNT {
        if remaining == 0 {
            break;
        }

        let shard_index = (start_shard + offset) % RECORDED_SAMPLE_SHARD_COUNT;
        let shard = &RECORDED_SAMPLE_SHARDS[shard_index];
        let Ok(mut samples) = shard.lock() else {
            DROPPED_SAMPLES.fetch_add(1, Ordering::Relaxed);
            continue;
        };
        let drain_len = samples.len().min(remaining);
        drained.extend(samples.drain(..drain_len));
        remaining -= drain_len;
    }

    if !drained.is_empty() {
        release_global_sample_slots(drained.len());
    }

    drained
}

fn drain_samples_for_report(limit: usize) -> Vec<RecordedAllocationSample> {
    // Give both sources progress under a small drain limit. For a limit of one,
    // alternate the preferred source instead of starving the global backlog.
    let global_budget = if limit == 1 {
        NEXT_REPORT_SOURCE.fetch_add(1, Ordering::Relaxed) % 2
    } else {
        limit / 2
    };
    let mut recorded = drain_recorded_samples(global_budget);
    let remaining = limit - recorded.len();
    drain_registered_tls_samples_for_report(&mut recorded, remaining);
    recorded.extend(drain_recorded_samples(limit - recorded.len()));
    recorded
}

fn recorded_sample_shard_capacity(total_capacity: usize) -> usize {
    total_capacity.saturating_add(RECORDED_SAMPLE_SHARD_COUNT - 1) / RECORDED_SAMPLE_SHARD_COUNT
}

#[cfg(test)]
fn build_allocation_samples(
    recorded: Vec<RecordedAllocationSample>,
    max_depth: usize,
) -> Vec<AllocationSample> {
    build_memory_samples(recorded, Vec::new(), max_depth)
}

fn build_memory_samples(
    recorded: Vec<RecordedAllocationSample>,
    live_samples: Vec<live::LiveAllocationSample>,
    max_depth: usize,
) -> Vec<AllocationSample> {
    let mut aggregated: HashMap<StackKey, AggregatedAllocationSample> = HashMap::new();
    for sample in recorded {
        let entry = aggregated.entry(sample.stack).or_default();
        entry.alloc_objects = entry.alloc_objects.saturating_add(sample.weighted_objects);
        entry.alloc_space = entry.alloc_space.saturating_add(sample.weighted_bytes);
    }
    for sample in live_samples {
        let entry = aggregated.entry(sample.stack).or_default();
        entry.inuse_objects += sample.weighted_objects;
        entry.inuse_space += sample.weighted_bytes;
    }

    // Stacks commonly share most instruction pointers. Resolve each address
    // once per report, without retaining stale symbols across dynamic unloads.
    let mut frame_cache: HashMap<usize, Vec<String>> = HashMap::new();

    aggregated
        .into_iter()
        .map(|(stack, sample)| {
            let frames = resolve_stack_with(&stack, max_depth, |ip| {
                frame_cache
                    .entry(ip)
                    .or_insert_with(|| resolve_frame_names(ip))
                    .clone()
            });
            AllocationSample {
                frames,
                alloc_objects: i64::try_from(sample.alloc_objects).unwrap_or(i64::MAX),
                alloc_space: i64::try_from(sample.alloc_space).unwrap_or(i64::MAX),
                // Float-to-integer casts saturate at i64::MAX for very large
                // weighted totals. Round once, after summing the entire stack.
                inuse_objects: sample.inuse_objects.round() as i64,
                inuse_space: sample.inuse_space.round() as i64,
            }
        })
        .collect()
}

fn resolve_stack_with(
    stack: &StackKey,
    max_depth: usize,
    resolve: impl FnMut(usize) -> Vec<String>,
) -> Vec<String> {
    let frames: Vec<String> = stack
        .iter()
        .flat_map(resolve)
        .filter(|name| !is_mimalloc_profiler_frame(name))
        .take(max_depth)
        .collect();

    if frames.is_empty() {
        vec![SYNTHETIC_FRAME.to_string()]
    } else {
        frames
    }
}

fn resolve_frame_names(ip: usize) -> Vec<String> {
    let mut resolved = Vec::new();
    backtrace::resolve(ip as *mut std::ffi::c_void, |symbol| {
        if let Some(name) = symbol.name() {
            resolved.push(name.to_string());
        }
    });
    if resolved.is_empty() {
        resolved.push(format!("0x{ip:x}"));
    }
    resolved
}

fn is_mimalloc_profiler_frame(name: &str) -> bool {
    name.contains("pyroscope::backend::mimalloc")
        || name.contains("pyroscope::encode::memory_pprof")
        || name.contains("backtrace::")
}

fn warm_backtrace() {
    let mut frames = 0;
    backtrace::trace(|_frame| {
        frames += 1;
        frames < 2
    });
}

fn duration_to_i64_nanos(duration: std::time::Duration) -> i64 {
    i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX)
}

fn duration_to_u64_micros(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as TestMutex;

    static TEST_LOCK: TestMutex<()> = TestMutex::new(());

    struct RecorderActiveGuard;

    impl Drop for RecorderActiveGuard {
        fn drop(&mut self) {
            RECORDER_ACTIVE.store(false, Ordering::Release);
            live::prepare(false, 0);
        }
    }

    fn test_sample(stack: StackKey) -> RecordedAllocationSample {
        RecordedAllocationSample {
            stack,
            weighted_objects: 1,
            weighted_bytes: 1024,
        }
    }

    fn test_live_sample(stack: StackKey) -> live::LiveAllocationSample {
        live::LiveAllocationSample {
            stack,
            weighted_objects: 1.0,
            weighted_bytes: 1024.0,
        }
    }

    fn clear_test_buffers() {
        for shard in RECORDED_SAMPLE_SHARDS.iter() {
            shard.lock().expect("lock samples").clear();
        }
        GLOBAL_BUFFERED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        // Production initialize preallocates every shard before recording.
        // Tests must honor that contract instead of relying on Vec growth.
        prepare_sample_buffer(DEFAULT_RING_CAPACITY);
        NEXT_RECORDED_SAMPLE_SHARD.store(0, Ordering::Relaxed);
        NEXT_DRAINED_SAMPLE_SHARD.store(0, Ordering::Relaxed);
        NEXT_REPORT_SOURCE.store(0, Ordering::Relaxed);
        NEXT_TLS_DRAIN_BUFFER.store(0, Ordering::Relaxed);
        SAMPLER_STATE.with(|sampler| sampler.set(SamplerState::new()));
        for buffer in registered_tls_sample_buffers() {
            let mut buffer = buffer.lock().expect("lock tls samples");
            buffer.clear();
            buffer.generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
        }
    }

    fn push_global_test_samples(samples: impl IntoIterator<Item = RecordedAllocationSample>) {
        push_global_test_samples_to_shard(0, samples);
    }

    fn push_global_test_samples_to_shard(
        shard_index: usize,
        samples: impl IntoIterator<Item = RecordedAllocationSample>,
    ) {
        let mut shard = RECORDED_SAMPLE_SHARDS[shard_index]
            .lock()
            .expect("lock samples");
        let initial_len = shard.len();
        shard.extend(samples);
        GLOBAL_BUFFERED_SAMPLE_COUNT.fetch_add(shard.len() - initial_len, Ordering::Relaxed);
    }

    #[test]
    fn mimalloc_config_default_is_valid() {
        assert!(MimallocConfig::default().validate().is_ok());
    }

    #[test]
    fn mimalloc_config_rejects_zero_sample_interval() {
        let config = MimallocConfig {
            sample_interval_bytes: 0,
            ..MimallocConfig::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn mimalloc_config_checks_capture_and_live_metadata_bounds() {
        assert!(MimallocConfig {
            max_depth: 65,
            ..MimallocConfig::default()
        }
        .validate()
        .is_err());
        assert!(MimallocConfig {
            live_heap_tracking: true,
            max_live_samples: 0,
            ..MimallocConfig::default()
        }
        .validate()
        .is_err());
        assert!(MimallocConfig {
            max_live_samples: 0,
            ..MimallocConfig::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn unsampled_thread_does_not_allocate_or_register_a_tls_ring() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let previous_interval = SAMPLE_INTERVAL_BYTES.swap(u64::MAX, Ordering::Relaxed);
        SAMPLING_CONFIG_GENERATION.fetch_add(1, Ordering::AcqRel);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active = RecorderActiveGuard;
        let before = registered_tls_sample_buffers().len();
        std::thread::spawn(|| {
            record_allocation(std::ptr::null_mut(), 1);
            SAMPLER_STATE.with(|sampler| assert!(!sampler.get().has_buffer));
        })
        .join()
        .expect("join unsampled thread");
        assert_eq!(registered_tls_sample_buffers().len(), before);
        SAMPLE_INTERVAL_BYTES.store(previous_interval, Ordering::Relaxed);
    }

    #[test]
    fn backend_rejects_concurrent_owners_and_drop_releases_recorder() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let previous_seen = ALLOCATOR_SEEN.swap(true, Ordering::Relaxed);
        let mut first = Mimalloc::new(MimallocConfig::default());
        first.initialize().expect("initialize first backend");
        let mut second = Mimalloc::new(MimallocConfig::default());
        assert!(second.initialize().is_err());
        drop(second);
        assert!(RECORDER_ACTIVE.load(Ordering::Acquire));
        drop(first);
        assert!(!RECORDER_ACTIVE.load(Ordering::Acquire));
        let mut third = Mimalloc::new(MimallocConfig::default());
        third.initialize().expect("initialize after owner drop");
        drop(third);
        ALLOCATOR_SEEN.store(previous_seen, Ordering::Relaxed);
        clear_test_buffers();
    }

    #[test]
    fn live_snapshot_survives_reports_and_cross_thread_free() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        live::prepare(true, 4096);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active = RecorderActiveGuard;
        let generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        live::record(100, stack, 1024 * 1024, 1024, generation);
        assert_eq!(live::snapshot().len(), 1);
        assert_eq!(live::snapshot().len(), 1);
        std::thread::spawn(|| live::remove(100))
            .join()
            .expect("cross-thread removal");
        assert!(live::snapshot().is_empty());
        assert_eq!(live::sample_count(), 0);
    }

    #[test]
    fn live_membership_filter_keeps_colliding_pointers_until_both_are_freed() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        live::prepare(true, 4096);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active = RecorderActiveGuard;
        let generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        let pointer = 100;
        let mut collisions = (101..)
            .filter(|other| splitmix64(*other as u64) % 4096 == splitmix64(pointer as u64) % 4096);
        let second = collisions
            .next()
            .expect("second pointer with same shard and membership bit");
        let untracked = collisions.next().expect("untracked colliding pointer");
        live::record(pointer, stack, 1024, 1, generation);
        live::record(second, stack, 2048, 1, generation);
        live::remove(untracked);
        assert_eq!(live::sample_count(), 2);
        live::remove(pointer);
        assert_eq!(live::sample_count(), 1);
        live::remove(second);
        assert_eq!(live::sample_count(), 0);
    }

    #[test]
    fn failed_reallocation_preserves_metadata_under_capacity_pressure() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        live::prepare(true, 64);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active = RecorderActiveGuard;
        let generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        let pointer = 100;
        let collision = (101..)
            .find(|other| splitmix64(*other as u64) % 64 == splitmix64(pointer as u64) % 64)
            .expect("colliding pointer");
        live::record(pointer, stack, 1024, 1, generation);
        let pending = live::begin_reallocation(pointer).expect("reserve sampled pointer");
        live::record(collision, stack, 2048, 1, generation);
        assert_eq!(live::dropped_sample_count(), 1);
        pending.finish(0, 0);
        assert_eq!(live::sample_count(), 1);
        assert_eq!(live::snapshot()[0].weighted_bytes, 1024.0);
        let (sample, generation) = live::begin_reallocation(pointer)
            .expect("reserve restored pointer")
            .finish(collision, 2048)
            .expect("moved sample");
        live::transfer_sample(collision, sample, generation);
        assert_eq!(live::snapshot()[0].weighted_bytes, 2048.0);
    }

    #[test]
    fn old_live_samples_and_reallocation_tokens_cannot_enter_restarted_session() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        live::prepare(true, 4096);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active = RecorderActiveGuard;
        let old_generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        live::record(100, stack, 1024, 1, old_generation);
        let pending = live::begin_reallocation(100).expect("reserve old-session pointer");
        RECORDER_ACTIVE.store(false, Ordering::Release);
        live::clear();
        let generation = SAMPLING_CONFIG_GENERATION.fetch_add(1, Ordering::AcqRel) + 1;
        live::prepare(true, 4096);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        pending.finish(0, 0);
        live::record(100, stack, 1024, 1, old_generation);
        assert_eq!(live::sample_count(), 0);
        live::record(100, stack, 2048, 1, generation);
        assert_eq!(live::sample_count(), 1);
    }

    #[test]
    fn report_with_one_sample_budget_alternates_tls_and_global_sources() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        clear_test_buffers();
        let global_stack = StackKey {
            frames: [1; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        let tls_stack = StackKey {
            frames: [2; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        push_global_test_samples([test_sample(global_stack)]);
        TLS_SAMPLE_BUFFER.with(|buffer| {
            let mut buffer = buffer.try_lock().expect("lock TLS");
            buffer.push(test_sample(tls_stack));
            buffer.push(test_sample(tls_stack));
        });
        assert_eq!(drain_samples_for_report(1)[0].stack, tls_stack);
        assert_eq!(drain_samples_for_report(1)[0].stack, global_stack);
        clear_test_buffers();
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
    }

    #[test]
    fn memory_aggregation_combines_interval_and_live_values_by_stack() {
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        let samples = build_memory_samples(
            vec![test_sample(stack)],
            vec![test_live_sample(stack), test_live_sample(stack)],
            16,
        );
        assert_eq!(samples.len(), 1);
        assert_eq!(
            (samples[0].alloc_objects, samples[0].alloc_space),
            (1, 1024)
        );
        assert_eq!(
            (samples[0].inuse_objects, samples[0].inuse_space),
            (2, 2048)
        );
    }

    #[test]
    fn live_object_weights_are_rounded_after_stack_aggregation() {
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        let sample = live::LiveAllocationSample {
            stack,
            weighted_objects: 1.582,
            weighted_bytes: 1620.5,
        };
        let samples = build_memory_samples(Vec::new(), vec![sample, sample], 8);
        assert_eq!(samples[0].inuse_objects, 3);
        assert_eq!(samples[0].inuse_space, 3241);
    }

    #[test]
    fn mimalloc_stats_reports_global_recorder_counters() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        clear_test_buffers();
        RECORDED_SAMPLE_COUNT.store(7, Ordering::Relaxed);
        DROPPED_SAMPLES.store(3, Ordering::Relaxed);
        LAST_PPROF_ENCODE_ELAPSED_MICROS.store(11, Ordering::Relaxed);

        let stats = mimalloc_stats();

        assert_eq!(
            stats,
            MimallocStats {
                recorded_samples: 7,
                flushes: 0,
                flushed_samples: 0,
                dropped_samples: 3,
                buffered_samples: Some(0),
                last_pprof_encode_elapsed_micros: 11,
                live_samples: 0,
                dropped_live_samples: 0,
                live_metadata_payload_bytes: 0,
            }
        );

        RECORDED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        DROPPED_SAMPLES.store(0, Ordering::Relaxed);
        LAST_PPROF_ENCODE_ELAPSED_MICROS.store(0, Ordering::Relaxed);
        clear_test_buffers();
    }

    #[test]
    fn stack_identity_ignores_unused_capture_slots() {
        let mut first = StackKey {
            frames: [0; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        first.frames[0] = 42;
        let second = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        assert_eq!(first, second);
        let samples =
            build_memory_samples(vec![test_sample(first), test_sample(second)], Vec::new(), 8);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].alloc_objects, 2);
    }

    #[test]
    fn previous_session_ring_cannot_flush_into_new_global_shards() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        clear_test_buffers();
        MAX_RECORDED_SAMPLES.store(10, Ordering::Relaxed);
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        let mut buffer = TlsSampleBuffer::new();
        buffer.push(test_sample(stack));
        SAMPLING_CONFIG_GENERATION.fetch_add(1, Ordering::AcqRel);
        assert!(flush_tls_samples_for_report(&mut buffer));
        assert!(buffer.is_empty());
        assert_eq!(count_recorded_samples(), 0);
        assert_eq!(GLOBAL_BUFFERED_SAMPLE_COUNT.load(Ordering::Relaxed), 0);
        MAX_RECORDED_SAMPLES.store(DEFAULT_RING_CAPACITY, Ordering::Relaxed);
        clear_test_buffers();
    }

    #[test]
    fn abandoned_reallocation_token_restores_its_live_slot() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        live::prepare(true, 64);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active = RecorderActiveGuard;
        let generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        live::record(100, stack, 1024, 1, generation);
        drop(live::begin_reallocation(100).expect("reserve live slot"));
        assert_eq!(live::sample_count(), 1);
        live::remove(100);
        assert_eq!(live::sample_count(), 0);
    }

    #[test]
    fn suppressed_and_reentrant_frees_still_remove_live_metadata() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        live::prepare(true, 4096);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active = RecorderActiveGuard;
        let allocator = SamplingMiMalloc::new();
        let layout = Layout::from_size_align(1024, 8).expect("layout");
        let generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        for reentrant in [false, true] {
            // SAFETY: Allocate valid storage directly through the wrapped
            // allocator so the test can install an exact live sample.
            let pointer = unsafe { allocator.inner.alloc(layout) };
            assert!(!pointer.is_null());
            live::record(pointer as usize, stack, 1024, 1, generation);
            assert_eq!(live::sample_count(), 1);
            SAMPLER_STATE.with(|sampler| {
                let mut state = sampler.get();
                state.in_profiler = reentrant;
                sampler.set(state);
            });
            // SAFETY: The pointer is live and layout matches its allocation.
            with_profiler_suppressed(|| unsafe { allocator.dealloc(pointer, layout) });
            SAMPLER_STATE.with(|sampler| {
                let mut state = sampler.get();
                state.in_profiler = false;
                sampler.set(state);
            });
            assert_eq!(live::sample_count(), 0);
        }
    }

    #[test]
    fn tracked_reallocation_below_byte_remainder_keeps_live_sample_without_allocation_event() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        clear_test_buffers();
        live::prepare(true, 4096);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active = RecorderActiveGuard;
        let allocator = SamplingMiMalloc::new();
        let layout = Layout::from_size_align(1024, 8).expect("layout");
        for suppressed in [false, true] {
            let new_size = if suppressed { 1536 } else { 768 };
            // SAFETY: Allocate valid storage directly through mimalloc.
            let pointer = unsafe { allocator.inner.alloc(layout) };
            assert!(!pointer.is_null());
            let generation = SAMPLING_CONFIG_GENERATION.load(Ordering::Acquire);
            let stack = StackKey {
                frames: [42; MAX_CAPTURE_DEPTH],
                depth: 1,
            };
            live::record(pointer as usize, stack, 1024, 1, generation);
            SAMPLER_STATE.with(|sampler| {
                let mut state = sampler.get();
                state.remaining_bytes = 1024 * 1024;
                state.remaining_config_generation = generation;
                state.flush_generation = FLUSH_REQUEST_GENERATION.load(Ordering::Relaxed);
                sampler.set(state);
            });
            RECORDED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
            // SAFETY: The pointer is live with the supplied layout.
            let resize = || unsafe { allocator.realloc(pointer, layout, new_size) };
            let resized = if suppressed {
                with_profiler_suppressed(resize)
            } else {
                resize()
            };
            assert!(!resized.is_null());
            if !suppressed {
                assert_eq!(
                    resized, pointer,
                    "shrink within the same mimalloc size class"
                );
            }
            assert_eq!(live::sample_count(), 1);
            assert_eq!(live::snapshot()[0].weighted_bytes, new_size as f64);
            assert_eq!(live::snapshot()[0].stack, stack);
            assert_eq!(RECORDED_SAMPLE_COUNT.load(Ordering::Relaxed), 0);
            // SAFETY: Successful realloc transfers ownership to resized.
            unsafe { allocator.dealloc(resized, Layout::from_size_align(new_size, 8).unwrap()) };
        }
        clear_test_buffers();
    }

    #[test]
    fn moved_reallocation_samples_new_physical_allocations_independently() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        clear_test_buffers();
        let previous_interval = SAMPLE_INTERVAL_BYTES.swap(4096, Ordering::Relaxed);
        live::prepare(true, 4096);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active = RecorderActiveGuard;
        let allocator = SamplingMiMalloc::new();
        let layout = Layout::from_size_align(128, 8).unwrap();
        // SAFETY: A valid layout is supplied to the wrapped allocator.
        let pointer = unsafe { allocator.inner.alloc(layout) };
        assert!(!pointer.is_null());
        assert_eq!(live::sample_count(), 0);
        // SAFETY: pointer is live and layout matches its original allocation.
        let replacement = unsafe { allocator.realloc(pointer, layout, 4 * 1024 * 1024) };
        assert!(!replacement.is_null());
        assert_ne!(replacement, pointer);
        assert_eq!(live::sample_count(), 1);
        assert_eq!(live::snapshot()[0].weighted_bytes, (4 * 1024 * 1024) as f64);
        // SAFETY: Successful realloc supplies the replacement's ownership/layout.
        unsafe {
            allocator.dealloc(
                replacement,
                Layout::from_size_align(4 * 1024 * 1024, 8).unwrap(),
            )
        };
        SAMPLE_INTERVAL_BYTES.store(previous_interval, Ordering::Relaxed);
        clear_test_buffers();
    }

    #[test]
    fn unregistered_thread_exit_discards_its_previous_session_ring() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        clear_test_buffers();
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active = RecorderActiveGuard;
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let registered = RegisteredTlsSampleBuffer::new();
            if let Some(id) = registered.id.take() {
                deregister_tls_sample_buffer(id);
            }
            let stack = StackKey {
                frames: [42; MAX_CAPTURE_DEPTH],
                depth: 1,
            };
            registered
                .buffer
                .lock()
                .expect("lock detached ring")
                .push(test_sample(stack));
            ready_tx.send(()).expect("send old ring ready");
            release_rx.recv().expect("wait for new session");
        });
        ready_rx.recv().expect("wait for old ring");
        SAMPLING_CONFIG_GENERATION.fetch_add(1, Ordering::AcqRel);
        release_tx.send(()).expect("exit worker in new session");
        worker.join().expect("join old-session worker");
        assert_eq!(count_recorded_samples(), 0);
        clear_test_buffers();
    }

    #[test]
    fn mimalloc_stats_includes_current_thread_tls_samples() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        push_global_test_samples([test_sample(stack)]);
        TLS_SAMPLE_BUFFER.with(|buffer| {
            let mut buffer = buffer.try_lock().expect("lock current thread buffer");
            assert!(buffer.push(test_sample(stack)));
            assert!(buffer.push(test_sample(stack)));
        });

        assert_eq!(mimalloc_stats().buffered_samples, Some(3));

        clear_test_buffers();
    }

    #[test]
    fn profiler_suppression_skips_internal_allocations_without_drop_count() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        clear_test_buffers();
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active_guard = RecorderActiveGuard;
        DROPPED_SAMPLES.store(0, Ordering::Relaxed);
        RECORDED_SAMPLE_COUNT.store(0, Ordering::Relaxed);

        with_profiler_suppressed(|| record_allocation(std::ptr::null_mut(), 1));

        assert_eq!(RECORDED_SAMPLE_COUNT.load(Ordering::Relaxed), 0);
        assert_eq!(DROPPED_SAMPLES.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn reentrant_allocation_is_ignored_without_drop_count() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        clear_test_buffers();
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active_guard = RecorderActiveGuard;
        DROPPED_SAMPLES.store(0, Ordering::Relaxed);
        RECORDED_SAMPLE_COUNT.store(0, Ordering::Relaxed);

        SAMPLER_STATE.with(|sampler| {
            let mut state = sampler.get();
            let previous = state.in_profiler;
            state.in_profiler = true;
            sampler.set(state);
            record_allocation(std::ptr::null_mut(), 1);
            let mut state = sampler.get();
            state.in_profiler = previous;
            sampler.set(state);
        });

        assert_eq!(RECORDED_SAMPLE_COUNT.load(Ordering::Relaxed), 0);
        assert_eq!(DROPPED_SAMPLES.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn realloc_recorded_size_tracks_newly_allocated_bytes() {
        let mut old_byte = 0_u8;
        let mut new_byte = 0_u8;
        let old_ptr = &mut old_byte as *mut u8;
        let new_ptr = &mut new_byte as *mut u8;

        assert_eq!(realloc_recorded_size(old_ptr, old_ptr, 1024, 1536), 512);
        assert_eq!(realloc_recorded_size(old_ptr, old_ptr, 1536, 1024), 0);
        assert_eq!(realloc_recorded_size(old_ptr, new_ptr, 1024, 1536), 1536);
        assert_eq!(realloc_recorded_size(old_ptr, new_ptr, 1536, 1024), 1024);
    }

    #[test]
    fn drain_recorded_samples_respects_limit_and_keeps_remaining_samples() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        push_global_test_samples([test_sample(stack), test_sample(stack), test_sample(stack)]);

        let drained = drain_recorded_samples(2);

        assert_eq!(drained.len(), 2);
        assert_eq!(mimalloc_stats().buffered_samples, Some(1));

        clear_test_buffers();
    }

    #[test]
    fn drain_recorded_samples_rotates_starting_shard() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let low_stack = StackKey {
            frames: [1; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        let high_stack = StackKey {
            frames: [2; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        push_global_test_samples_to_shard(0, [test_sample(low_stack)]);
        push_global_test_samples_to_shard(1, [test_sample(high_stack)]);
        NEXT_DRAINED_SAMPLE_SHARD.store(1, Ordering::Relaxed);

        let first = drain_recorded_samples(1);
        let second = drain_recorded_samples(1);

        assert_eq!(first.len(), 1);
        assert_eq!(first[0].stack, high_stack);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].stack, low_stack);
        assert_eq!(GLOBAL_BUFFERED_SAMPLE_COUNT.load(Ordering::Relaxed), 0);

        clear_test_buffers();
    }

    #[test]
    fn release_global_sample_slots_saturates_at_zero() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        GLOBAL_BUFFERED_SAMPLE_COUNT.store(0, Ordering::Relaxed);

        release_global_sample_slots(3);
        assert_eq!(GLOBAL_BUFFERED_SAMPLE_COUNT.load(Ordering::Relaxed), 0);

        GLOBAL_BUFFERED_SAMPLE_COUNT.store(2, Ordering::Relaxed);
        release_global_sample_slots(5);
        assert_eq!(GLOBAL_BUFFERED_SAMPLE_COUNT.load(Ordering::Relaxed), 0);

        GLOBAL_BUFFERED_SAMPLE_COUNT.store(5, Ordering::Relaxed);
        release_global_sample_slots(3);
        assert_eq!(GLOBAL_BUFFERED_SAMPLE_COUNT.load(Ordering::Relaxed), 2);

        GLOBAL_BUFFERED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
    }

    #[test]
    fn prepare_sample_buffer_recounts_global_sample_slots_after_clear() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        push_global_test_samples([test_sample(stack), test_sample(stack)]);
        GLOBAL_BUFFERED_SAMPLE_COUNT.store(usize::MAX, Ordering::Relaxed);

        prepare_sample_buffer(10);

        assert_eq!(count_recorded_samples(), 0);
        assert_eq!(GLOBAL_BUFFERED_SAMPLE_COUNT.load(Ordering::Relaxed), 0);

        clear_test_buffers();
    }

    #[test]
    fn tls_sample_buffer_drain_into_respects_limit() {
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        let mut buffer = TlsSampleBuffer::new();
        assert!(buffer.push(test_sample(stack)));
        assert!(buffer.push(test_sample(stack)));
        assert!(buffer.push(test_sample(stack)));
        let mut out = Vec::new();

        let drained = buffer.drain_into(&mut out, 2);

        assert_eq!(drained, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn tls_sample_buffer_registry_reuses_deregistered_slots() {
        let _guard = TEST_LOCK.lock().expect("lock test");

        let first_id = register_tls_sample_buffer(Arc::new(Mutex::new(TlsSampleBuffer::new())))
            .expect("register first buffer");
        deregister_tls_sample_buffer(first_id);

        let second_id = register_tls_sample_buffer(Arc::new(Mutex::new(TlsSampleBuffer::new())))
            .expect("register second buffer");

        assert_eq!(second_id, first_id);
        deregister_tls_sample_buffer(second_id);
    }

    #[test]
    fn tls_sample_buffer_retries_registration_after_initial_contention() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let registry_guard = TLS_SAMPLE_BUFFER_REGISTRY.lock().expect("lock registry");

        let buffer = RegisteredTlsSampleBuffer::new();
        assert_eq!(buffer.id.get(), None);
        drop(buffer.try_lock().expect("lock local buffer"));
        assert_eq!(buffer.id.get(), None);

        drop(registry_guard);
        drop(buffer.try_lock().expect("lock local buffer after retry"));

        let id = buffer.id.get().expect("registration retried");
        deregister_tls_sample_buffer(id);
        buffer.id.set(None);
    }

    #[test]
    fn flush_tls_samples_moves_buffer_to_global_samples() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        MAX_RECORDED_SAMPLES.store(10, Ordering::Relaxed);
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        let mut buffer = TlsSampleBuffer::new();
        assert!(buffer.push(test_sample(stack)));
        assert!(buffer.push(test_sample(stack)));

        assert!(flush_tls_samples(&mut buffer));

        assert_eq!(buffer.len(), 0);
        assert_eq!(mimalloc_stats().flushes, 1);
        assert_eq!(mimalloc_stats().flushed_samples, 2);
        assert_eq!(mimalloc_stats().buffered_samples, Some(2));
        clear_test_buffers();
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        MAX_RECORDED_SAMPLES.store(DEFAULT_RING_CAPACITY, Ordering::Relaxed);
    }

    #[test]
    fn flush_tls_samples_keeps_tls_samples_when_global_capacity_is_full() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        DROPPED_SAMPLES.store(0, Ordering::Relaxed);
        MAX_RECORDED_SAMPLES.store(0, Ordering::Relaxed);
        let mut buffer = TlsSampleBuffer::new();
        assert!(buffer.push(test_sample(stack)));
        assert!(buffer.push(test_sample(stack)));

        assert!(!flush_tls_samples(&mut buffer));

        assert_eq!(buffer.len(), 2);
        assert_eq!(DROPPED_SAMPLES.load(Ordering::Relaxed), 0);
        MAX_RECORDED_SAMPLES.store(DEFAULT_RING_CAPACITY, Ordering::Relaxed);
        clear_test_buffers();
    }

    #[test]
    fn flush_tls_samples_keeps_unflushed_tls_samples_after_partial_flush() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        DROPPED_SAMPLES.store(0, Ordering::Relaxed);
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        MAX_RECORDED_SAMPLES.store(1, Ordering::Relaxed);
        let mut buffer = TlsSampleBuffer::new();
        assert!(buffer.push(test_sample(stack)));
        assert!(buffer.push(test_sample(stack)));

        assert!(!flush_tls_samples(&mut buffer));

        assert_eq!(buffer.len(), 1);
        assert_eq!(count_recorded_samples(), 1);
        assert_eq!(GLOBAL_BUFFERED_SAMPLE_COUNT.load(Ordering::Relaxed), 1);
        assert_eq!(DROPPED_SAMPLES.load(Ordering::Relaxed), 0);
        MAX_RECORDED_SAMPLES.store(DEFAULT_RING_CAPACITY, Ordering::Relaxed);
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        clear_test_buffers();
    }

    #[test]
    fn flush_requested_tls_samples_flushes_current_thread_buffer_once() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        MAX_RECORDED_SAMPLES.store(10, Ordering::Relaxed);
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        TLS_SAMPLE_BUFFER.with(|buffer| {
            let mut buffer = buffer.try_lock().expect("lock current thread buffer");
            assert!(buffer.push(test_sample(stack)));
            assert!(buffer.push(test_sample(stack)));
        });

        request_tls_sample_flush();
        flush_requested_tls_samples();

        assert_eq!(mimalloc_stats().buffered_samples, Some(2));
        assert_eq!(mimalloc_stats().flushes, 1);
        assert_eq!(mimalloc_stats().flushed_samples, 2);

        flush_requested_tls_samples();
        assert_eq!(mimalloc_stats().buffered_samples, Some(2));
        assert_eq!(mimalloc_stats().flushes, 1);
        assert_eq!(mimalloc_stats().flushed_samples, 2);

        clear_test_buffers();
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        MAX_RECORDED_SAMPLES.store(DEFAULT_RING_CAPACITY, Ordering::Relaxed);
    }

    #[test]
    fn flush_requested_tls_samples_retries_after_lock_failure() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        MAX_RECORDED_SAMPLES.store(10, Ordering::Relaxed);
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);

        TLS_SAMPLE_BUFFER.with(|buffer| {
            let mut locked_buffer = buffer.try_lock().expect("lock current thread buffer");
            assert!(locked_buffer.push(test_sample(stack)));

            request_tls_sample_flush();
            flush_requested_tls_samples();
            assert_eq!(mimalloc_stats().flushes, 0);
            assert_eq!(locked_buffer.len(), 1);
        });

        flush_requested_tls_samples();

        assert_eq!(mimalloc_stats().flushes, 1);
        assert_eq!(mimalloc_stats().flushed_samples, 1);

        clear_test_buffers();
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        MAX_RECORDED_SAMPLES.store(DEFAULT_RING_CAPACITY, Ordering::Relaxed);
    }

    #[test]
    fn flush_registered_tls_samples_drains_live_worker_thread_buffer() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        MAX_RECORDED_SAMPLES.store(10, Ordering::Relaxed);
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        let worker = std::thread::spawn(move || {
            TLS_SAMPLE_BUFFER.with(|buffer| {
                let mut buffer = buffer.try_lock().expect("lock worker buffer");
                assert!(buffer.push(test_sample(stack)));
                assert!(buffer.push(test_sample(stack)));
            });
            ready_tx.send(()).expect("send ready");
            release_rx.recv().expect("wait for release");
        });

        ready_rx.recv().expect("wait for worker buffer");
        assert_eq!(registered_tls_buffered_samples(), Some(2));

        flush_registered_tls_samples();

        let stats = mimalloc_stats();
        assert_eq!(stats.flushes, 1);
        assert_eq!(stats.flushed_samples, 2);
        assert_eq!(stats.buffered_samples, Some(2));

        release_tx.send(()).expect("release worker");
        worker.join().expect("join worker");
        clear_test_buffers();
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        MAX_RECORDED_SAMPLES.store(DEFAULT_RING_CAPACITY, Ordering::Relaxed);
    }

    #[test]
    fn report_tls_drain_bypasses_full_global_buffer() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let global_stack = StackKey {
            frames: [1; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        let tls_stack = StackKey {
            frames: [2; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        MAX_RECORDED_SAMPLES.store(1, Ordering::Relaxed);
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        push_global_test_samples([test_sample(global_stack)]);

        let tls_buffer = Arc::new(Mutex::new(TlsSampleBuffer::new()));
        {
            let mut buffer = tls_buffer.lock().expect("lock tls buffer");
            assert!(buffer.push(test_sample(tls_stack)));
            assert!(buffer.push(test_sample(tls_stack)));
        }
        let id = register_tls_sample_buffer(tls_buffer.clone()).expect("register tls buffer");
        let mut recorded = Vec::new();

        let drained = drain_registered_tls_samples_for_report(&mut recorded, 10);

        assert_eq!(drained, 2);
        assert_eq!(recorded.len(), 2);
        assert!(recorded.iter().all(|sample| sample.stack == tls_stack));
        assert_eq!(tls_buffer.lock().expect("lock tls buffer").len(), 0);
        assert_eq!(count_recorded_samples(), 1);
        assert_eq!(GLOBAL_BUFFERED_SAMPLE_COUNT.load(Ordering::Relaxed), 1);
        assert_eq!(mimalloc_stats().flushes, 1);
        assert_eq!(mimalloc_stats().flushed_samples, 2);

        deregister_tls_sample_buffer(id);
        clear_test_buffers();
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        MAX_RECORDED_SAMPLES.store(DEFAULT_RING_CAPACITY, Ordering::Relaxed);
    }

    #[test]
    fn tls_sample_buffer_flushes_on_thread_exit_when_recorder_is_active() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        MAX_RECORDED_SAMPLES.store(10, Ordering::Relaxed);
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        DROPPED_SAMPLES.store(0, Ordering::Relaxed);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active_guard = RecorderActiveGuard;

        std::thread::spawn(move || {
            TLS_SAMPLE_BUFFER.with(|buffer| {
                let mut buffer = buffer.try_lock().expect("lock worker buffer");
                assert!(buffer.push(test_sample(stack)));
                assert!(buffer.push(test_sample(stack)));
            });
        })
        .join()
        .expect("join allocation thread");

        let stats = mimalloc_stats();
        assert!(matches!(stats.buffered_samples, Some(samples) if samples >= 2));
        assert!(stats.flushes >= 1);
        assert!(stats.flushed_samples >= 2);
        assert_eq!(stats.dropped_samples, 0);
        assert_eq!(registered_tls_buffered_samples(), Some(0));

        clear_test_buffers();
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        DROPPED_SAMPLES.store(0, Ordering::Relaxed);
        MAX_RECORDED_SAMPLES.store(DEFAULT_RING_CAPACITY, Ordering::Relaxed);
    }

    #[test]
    fn tls_sample_buffer_counts_drop_when_thread_exit_flush_has_no_global_capacity() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        MAX_RECORDED_SAMPLES.store(0, Ordering::Relaxed);
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        DROPPED_SAMPLES.store(0, Ordering::Relaxed);
        RECORDER_ACTIVE.store(true, Ordering::Release);
        let _active_guard = RecorderActiveGuard;

        std::thread::spawn(move || {
            TLS_SAMPLE_BUFFER.with(|buffer| {
                let mut buffer = buffer.try_lock().expect("lock worker buffer");
                assert!(buffer.push(test_sample(stack)));
                assert!(buffer.push(test_sample(stack)));
            });
        })
        .join()
        .expect("join allocation thread");

        let stats = mimalloc_stats();
        assert_eq!(stats.dropped_samples, 2);
        assert_eq!(stats.flushes, 0);
        assert_eq!(stats.flushed_samples, 0);
        assert_eq!(stats.buffered_samples, Some(0));
        assert_eq!(registered_tls_buffered_samples(), Some(0));

        clear_test_buffers();
        FLUSH_COUNT.store(0, Ordering::Relaxed);
        FLUSHED_SAMPLE_COUNT.store(0, Ordering::Relaxed);
        DROPPED_SAMPLES.store(0, Ordering::Relaxed);
        MAX_RECORDED_SAMPLES.store(DEFAULT_RING_CAPACITY, Ordering::Relaxed);
    }

    #[test]
    fn clear_registered_tls_samples_drains_all_live_thread_buffers() {
        let _guard = TEST_LOCK.lock().expect("lock test");
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        clear_test_buffers();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        let worker = std::thread::spawn(move || {
            TLS_SAMPLE_BUFFER.with(|buffer| {
                let mut buffer = buffer.try_lock().expect("lock worker buffer");
                assert!(buffer.push(test_sample(stack)));
                assert!(buffer.push(test_sample(stack)));
            });
            ready_tx.send(()).expect("send ready");
            release_rx.recv().expect("wait for release");
        });

        ready_rx.recv().expect("wait for worker buffer");
        assert_eq!(registered_tls_buffered_samples(), Some(2));

        clear_registered_tls_samples();

        assert_eq!(registered_tls_buffered_samples(), Some(0));
        release_tx.send(()).expect("release worker");
        worker.join().expect("join worker");
        clear_test_buffers();
    }

    #[test]
    fn build_allocation_samples_groups_matching_stacks() {
        let stack = StackKey {
            frames: [42; MAX_CAPTURE_DEPTH],
            depth: 1,
        };
        let samples = build_allocation_samples(
            vec![
                RecordedAllocationSample {
                    stack,
                    weighted_objects: 8,
                    weighted_bytes: 1024,
                },
                RecordedAllocationSample {
                    stack,
                    weighted_objects: 4,
                    weighted_bytes: 1024,
                },
            ],
            DEFAULT_MAX_DEPTH,
        );

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].alloc_objects, 12);
        assert_eq!(samples[0].alloc_space, 2048);
    }

    #[test]
    fn resolve_stack_applies_depth_after_profiler_frame_filtering() {
        let mut frames = [0; MAX_CAPTURE_DEPTH];
        frames[..4].copy_from_slice(&[1, 2, 3, 4]);
        let stack = StackKey { frames, depth: 4 };

        let resolved = resolve_stack_with(&stack, 1, |ip| match ip {
            1 => vec!["pyroscope::backend::mimalloc::record_sample".to_string()],
            2 => vec!["backtrace::trace_unsynchronized".to_string()],
            3 => vec!["example::allocate".to_string()],
            4 => vec!["example::caller".to_string()],
            _ => Vec::new(),
        });

        assert_eq!(resolved, vec!["example::allocate"]);
    }

    #[test]
    fn resolve_stack_expands_inline_symbols_before_filtering() {
        let mut frames = [0; MAX_CAPTURE_DEPTH];
        frames[0] = 1;
        let stack = StackKey { frames, depth: 1 };

        let resolved = resolve_stack_with(&stack, 2, |ip| match ip {
            1 => vec![
                "pyroscope::backend::mimalloc::record_sample".to_string(),
                "example::inline_allocate".to_string(),
                "example::caller".to_string(),
            ],
            _ => Vec::new(),
        });

        assert_eq!(
            resolved,
            vec!["example::inline_allocate", "example::caller"]
        );
    }

    #[test]
    fn calculate_sample_weight_uses_interval_for_small_allocation() {
        let mut rng_state = 1;
        let weight = calculate_sample_weight(128, 128, 1024, &mut rng_state);

        assert_eq!(weight.weighted_objects, 8);
        assert_eq!(weight.weighted_bytes, 1024);
        assert!(weight.next_remaining > 0);
    }

    #[test]
    fn calculate_sample_weight_carries_large_allocation_overshoot_with_poisson_intervals() {
        let mut rng_state = 1;
        let weight = calculate_sample_weight(2500, 1000, 1000, &mut rng_state);

        assert!(weight.weighted_objects >= 1);
        assert!(weight.weighted_bytes >= 1000);
        assert!(weight.next_remaining > 0);
    }

    #[test]
    fn calculate_sample_weight_bounds_large_allocation_interval_work() {
        let mut rng_state = 1;
        let size = (MAX_POISSON_INTERVALS_PER_ALLOCATION + 1024) * 1024;
        let weight = calculate_sample_weight(size, 1, 1024, &mut rng_state);

        assert!(weight.weighted_objects >= 1);
        assert!(weight.weighted_bytes >= MAX_POISSON_INTERVALS_PER_ALLOCATION * 1024);
        assert!(weight.next_remaining > 0);
    }

    #[test]
    fn deterministic_sample_weight_documents_previous_interval_semantics() {
        let weight = calculate_deterministic_sample_weight(2500, 1000, 1000);

        assert_eq!(weight.weighted_objects, 1);
        assert_eq!(weight.weighted_bytes, 2000);
        assert_eq!(weight.next_remaining, 500);
    }

    #[test]
    fn next_poisson_interval_uses_thread_rng_state() {
        let mut rng_state = 1;
        let first = next_poisson_interval(1024, &mut rng_state);
        let second = next_poisson_interval(1024, &mut rng_state);

        assert!(first > 0);
        assert!(second > 0);
        assert_ne!(first, second);
    }
}
