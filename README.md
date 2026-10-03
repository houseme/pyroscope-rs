## Pyroscope Profiler

**Pyroscope Profiler for Rust. Profile your Rust applications.**

[![license](https://img.shields.io/badge/license-Apache2.0-blue.svg)](LICENSE) 
[![Crate](https://img.shields.io/crates/v/pyroscope.svg)](https://crates.io/crates/pyroscope)

### Mimalloc Memory Profiling

Enable the optional `backend-mimalloc` feature and install
`SamplingMiMalloc` as the process global allocator:

```toml
[dependencies]
# Before the next crates.io release, use the branch or a local path that
# contains `backend-mimalloc`.
pyroscope = { git = "https://github.com/grafana/pyroscope-rs", features = ["backend-mimalloc"] }
```

```rust
use pyroscope::backend::mimalloc::{
    mimalloc_backend, MimallocConfig, SamplingMiMalloc,
};
use pyroscope::pyroscope::PyroscopeAgentBuilder;

#[global_allocator]
static ALLOC: SamplingMiMalloc = SamplingMiMalloc::new();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = PyroscopeAgentBuilder::new(
        "http://localhost:4040",
        "my-rust-service",
        100,
        "pyroscope-rs",
        env!("CARGO_PKG_VERSION"),
        mimalloc_backend(MimallocConfig::default()),
    )
    .build()?;

    let agent_running = agent.start()?;
    // Run application workload.
    let agent_ready = agent_running.stop()?;
    agent_ready.shutdown();
    Ok(())
}
```

The default mimalloc backend records allocation events (`alloc_objects` and
`alloc_space`) through the normal Pyroscope upload path. Enable live heap
profiling to also report `inuse_objects` and `inuse_space`, with `inuse_space`
selected by default, matching the jemalloc backend's live memory view:

```rust
let backend = mimalloc_backend(MimallocConfig {
    live_heap_tracking: true,
    max_live_samples: 16_384,
    ..MimallocConfig::default()
});
```

Live tracking covers Rust allocations made through `SamplingMiMalloc` after
initialization, including cross-thread frees and reallocations. It reports
sampling-weighted requested sizes, not allocator usable sizes or allocations
made directly by native libraries. Sampling starts on `alloc` / `alloc_zeroed`.
Same-address reallocations preserve the original stack and sampling weight
while updating size. Moving reallocations independently sample the replacement
allocation, including replacements of previously untracked objects. Enabling live tracking
does not change the interval `alloc_*` accounting for reallocations.

Metadata is bounded, preallocated, and partitioned across shards. Full or
contended shards can omit new live samples, exposed by
`mimalloc_stats().dropped_live_samples`. Freeing a tracked pointer reliably
removes it, which may briefly wait for its shard lock. Reports copy shards one
at a time, so concurrent heap snapshots are approximate rather than atomic.

The feature uses `rustfs-mimalloc` and requires Rust 1.96 or newer. Applications
must install `SamplingMiMalloc`; initialization fails when it has not observed
allocations. Using `rustfs_mimalloc::MiMalloc` directly does not capture call
stacks. Only one mimalloc backend may be active per process. Unresolved frames
fall back to instruction addresses or a synthetic frame. Thread tags are
no-ops, as in the jemalloc backend; agent-wide tags remain supported.

Memory pprof locations retain instruction addresses and inline call chains.
Function filenames and source lines are included when debug information is
available. To retain source information in optimized builds:

```toml
[profile.release]
debug = 1
strip = "none"
```

This backend does not yet export executable mappings or build IDs for external
symbolization. The upload path is covered by loopback HTTP integration tests
for memory profiles, gzip payloads, authentication, tenant headers, and labels;
these tests do not replace validation against a deployed Pyroscope server.

Useful local checks:

```bash
cargo run --example mimalloc --features backend-mimalloc
cargo run --release --example mimalloc_overhead --features backend-mimalloc
MIMALLOC_BENCH_MODE=live cargo run --release --example mimalloc_overhead --features backend-mimalloc
make mimalloc/bench/report
cargo test --locked --test mimalloc_backend --features backend-mimalloc -- --ignored
```

`make mimalloc/bench/report` writes a Markdown report and raw key-value
outputs under `target/mimalloc-benchmark/`. The GitHub Actions
`mimalloc benchmark report` job uploads the same directory as the
`mimalloc-benchmark-report` artifact, including throughput, overhead,
recorder counters, report latency, encoded pprof size, pprof encode time, and
sampled allocation latency percentiles. Live heap scenarios at 1 MiB, 512 KiB,
and 4 KiB sampling intervals also retain a working set through report creation.
Their raw outputs include live sample counts, dropped live samples, and
preallocated metadata payload bytes (excluding hash control bytes, shard
headers, and allocator bookkeeping). Benchmark warnings are diagnostic;
performance claims require repeat runs on an otherwise idle machine.

Latency sampling selects one pseudorandom allocation per window, covering the
workload's size distribution instead of repeatedly timing the smallest size.
Raw outputs include the sampling policy, sample count, and sampled size range.
New history rows use `history/mimalloc-benchmark-history-v2.csv`; existing
history files are retained. Earlier fixed-cadence latency percentiles should
not be compared directly with the new stratified measurements.

### Major Contributors

We'd like to give a big thank you to the following contributors who have made significant contributions to this project:

* [Abid Omar](https://github.com/omarabid)
* [Anatoly Korniltsev](https://github.com/korniltsev)
* [Bernhard Schuster](https://github.com/drahnr)


### License

Pyroscope is distributed under the Apache License (Version 2.0).

See [LICENSE](LICENSE) for details.
