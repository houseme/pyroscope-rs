//! SamplingMiMalloc allocation workload for local overhead comparisons.
//!
//! ```sh
//! cargo run --release --example mimalloc_overhead --features backend-mimalloc
//! MIMALLOC_BENCH_MODE=active cargo run --release --example mimalloc_overhead --features backend-mimalloc
//! ```

#[path = "mimalloc_benchmark/support.rs"]
mod support;

use pyroscope::backend::{
    mimalloc::{mimalloc_backend, mimalloc_stats, MimallocConfig, SamplingMiMalloc},
    ReportData,
};
use support::{print_workload, run_workload, WorkloadConfig};

#[global_allocator]
static ALLOC: SamplingMiMalloc = SamplingMiMalloc::new();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = WorkloadConfig::from_env();
    let mode = std::env::var("MIMALLOC_BENCH_MODE").unwrap_or_else(|_| "inactive".to_string());

    if mode == "active" || mode == "live" {
        run_active(config, mode == "live")?;
    } else {
        let result = run_workload(config);
        print_workload("sampling_mimalloc_inactive", config, result);
    }

    Ok(())
}

fn run_active(config: WorkloadConfig, live_heap: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = mimalloc_backend(MimallocConfig {
        sample_interval_bytes: read_env_u64("MIMALLOC_BENCH_SAMPLE_INTERVAL", 1024 * 1024),
        ring_capacity: read_env_usize("MIMALLOC_BENCH_RING_CAPACITY", 512),
        report_drain_limit: read_env_usize("MIMALLOC_BENCH_REPORT_DRAIN_LIMIT", 1_000_000),
        live_heap_tracking: live_heap,
        max_live_samples: read_env_usize("MIMALLOC_BENCH_MAX_LIVE_SAMPLES", 16_384),
        ..MimallocConfig::default()
    })
    .initialize()?;

    // Retain a working set through report generation while the timed workload
    // exercises allocations and frees. Its setup is outside throughput timing.
    let retained: Vec<Vec<u8>> = if live_heap {
        (0..read_env_usize("MIMALLOC_BENCH_LIVE_RETAINED_COUNT", 256))
            .map(|_| vec![0; read_env_usize("MIMALLOC_BENCH_LIVE_RETAINED_SIZE", 64 * 1024)])
            .collect()
    } else {
        Vec::new()
    };

    let result = run_workload(config);
    let report_start = std::time::Instant::now();
    let report = backend.report()?;
    let report_elapsed = report_start.elapsed();
    let encoded_pprof_bytes = match &report.data {
        ReportData::RawPprof(bytes) => bytes.len(),
        ReportData::Reports(_) => 0,
    };
    let stats = mimalloc_stats();

    std::hint::black_box(&retained);
    print_workload(
        if live_heap {
            "sampling_mimalloc_live"
        } else {
            "sampling_mimalloc_active"
        },
        config,
        result,
    );
    println!(
        "sample_interval_bytes={}",
        read_env_u64("MIMALLOC_BENCH_SAMPLE_INTERVAL", 1024 * 1024)
    );
    println!("recorded_samples={}", stats.recorded_samples);
    println!("flushes={}", stats.flushes);
    println!("flushed_samples={}", stats.flushed_samples);
    println!("dropped_samples={}", stats.dropped_samples);
    println!("live_samples={}", stats.live_samples);
    println!("dropped_live_samples={}", stats.dropped_live_samples);
    println!(
        "live_metadata_payload_bytes={}",
        stats.live_metadata_payload_bytes
    );
    println!(
        "buffered_samples={}",
        stats
            .buffered_samples
            .map(|samples| samples.to_string())
            .unwrap_or_else(|| "locked".to_string())
    );
    println!("report_elapsed_ms={}", report_elapsed.as_millis());
    println!("encoded_pprof_bytes={encoded_pprof_bytes}");
    println!(
        "pprof_encode_elapsed_us={}",
        stats.last_pprof_encode_elapsed_micros
    );

    backend.shutdown()?;
    if let (Some(path), ReportData::RawPprof(bytes)) =
        (std::env::var_os("MIMALLOC_BENCH_PPROF_PATH"), report.data)
    {
        use std::io::Write;
        // Export after shutdown and outside both throughput and report timing.
        let file = std::fs::File::create(path)?;
        let mut gzip = libflate::gzip::Encoder::new(file)?;
        gzip.write_all(&bytes)?;
        gzip.finish().into_result()?;
    }
    Ok(())
}

fn read_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn read_env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
