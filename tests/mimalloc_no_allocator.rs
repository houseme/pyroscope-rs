#![cfg(feature = "backend-mimalloc")]

use pyroscope::backend::mimalloc::{mimalloc_backend, MimallocConfig};

// This binary intentionally uses the default allocator. Keep it separate from
// mimalloc_backend.rs so the global-allocator installation check is real.
#[test]
fn initialization_rejects_a_missing_sampling_allocator() {
    let result = mimalloc_backend(MimallocConfig::default()).initialize();
    let Err(error) = result else {
        panic!("missing SamplingMiMalloc must fail initialization");
    };
    assert!(error.to_string().contains("SamplingMiMalloc"));
}
