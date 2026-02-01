# 🔮 Future Work: rust-ta

This document outlines the roadmap for performance optimizations and architectural improvements identified during the initial audit. These tasks are intended for contributors looking to push `rust-ta` to the limit of high-frequency trading (HFT) performance.

## 1. 🚀 SIMD Acceleration (`ta-simd`)

The `ta-simd` crate is currently a skeleton. The goal is to provide vectorized implementations for "hot path" indicators using the [`wide`](https://crates.io/crates/wide) crate for portable SIMD support.

### Objectives
*   **Vectorize Moving Averages**: Implement SIMD versions of SMA, EMA, and WMA.
*   **Vectorize Element-wise Operations**: Accelerate operations like `High - Low` or `Close - PrevClose` used in TR, RSI, and DM calculations.
*   **Feature Gating**: Ensure SIMD support is behind a `simd` feature flag to maintain `no_std` compatibility where `wide` might not be supported (though `wide` is generally quite portable).

### Implementation Strategy
Use `wide::f64x4` (or `f32x4` depending on `TaFloat`) to process 4 candles per CPU cycle.

#### Pattern: Rolling Window Reduction
For a rolling window sum (SMA), the naive loop allows O(1) updates, but batch calculation (`calculate()`) can be vectorized.

```rust
// Conceptual implementation for ta-simd
use wide::f64x4;

pub fn rolling_sum_simd(data: &[f64], window: usize) -> Vec<f64> {
    let mut results = Vec::with_capacity(data.len());
    // ... specialized logic to load 4 windows in parallel or 
    // process 4 independent rolling sums if batching multiple series ...
    
    // NOTE: True rolling window SIMD is tricky for single-series dependency.
    // However, element-wise ops (RSI gains/losses) are trivially parallelizable.
    results
}
```

**Priorities:**
1.  **RSI/StochRSI**: The `Gain/Loss` calculation over the whole array is element-wise and perfect for SIMD.
2.  **Bollinger Bands**: Standard deviation calculation involves sum-of-squares which vectorizes well.

## 2. ⚡ Zero-Allocation Reset Pattern

### The Issue
Currently, many indicators implement `reset()` by re-initializing their buffers. This causes heap deallocation and re-allocation, which is expensive in HFT loops that frequently reset state (e.g., when switching symbols in a worker).

**Current (inefficient):**
```rust
// ta-indicators/src/trend/sma.rs
fn reset(&mut self) {
    // DROPS old Vec, ALLOCATES new Vec
    self.buffer = RingBuffer::new(self.config.window); 
    self.sum = T::ZERO;
    self.count = 0;
}
```

### The Fix
Modify `reset()` to reuse the existing capacity of `RingBuffer` and other internal vectors.

**Proposed (optimized):**
```rust
// ta-indicators/src/trend/sma.rs
fn reset(&mut self) {
    // Reuses existing capacity, sets len=0, head=0, sum=0
    self.buffer.clear(); 
    self.sum = T::ZERO;
    self.count = 0;
}
```

### Action Items
*   **Audit all `reset()` methods**: Check `ta-indicators` for usage of `new()` inside reset.
*   **Verify `RingBuffer::clear`**: Ensure `RingBuffer::clear()` in `ta-core` essentially does `self.len = 0; self.head = 0;` without touching the `Vec`'s capacity. (Confirmed: It currently sets `head=0, len=0, sum=0`, which is correct).
*   **Other Buffers**: Check indicators like `Macd` that own sub-indicators (`fast_ema`, `slow_ema`). Ensure their `reset()` cascades the call to `child.reset()` instead of re-creating them.

## 3. 🧠 Memory Layout Optimizations

### Struct-of-Arrays (SoA) for `OhlcvSeries`
Currently, `OhlcvSeries` is a struct containing 5 separate `Series` (Vecs). This is "Struct of Arrays" (SoA), which is actually **good** for SIMD and cache locality when accessing a single field (e.g., just `Close` for SMA).

**Potential Improvement**:
*   Ensure that `DataFrame` construction from `OhlcvSeries` is zero-copy or minimal-copy.
*   Investigate `arrow-rs` or `polars` integration if interoperability with the wider data engineering ecosystem is required in the future.

## 4. 🧪 Fuzz Testing

While `proptest` covers property-based testing, true fuzzing (using `cargo-fuzz`) could discover edge cases in:
*   Numerical stability (NaN propagation, Infinity handling).
*   Buffer boundary conditions (window sizes vs data lengths).
*   Serialization/Deserialization of corrupt state data.

## 5. 🐍 Python Performance (PyO3)

*   **Numpy Views**: Ensure data passed from Python is accessed as a view where possible, avoiding copy into Rust `Vec` unless necessary for alignment.
*   **Parallel Execution**: Use `par_iter` (Rayon) for batch calculations on large DataFrames if the GIL is released.
