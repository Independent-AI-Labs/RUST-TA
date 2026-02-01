# rust-ta 🦀📈

**High-performance, production-grade technical analysis library for Rust.**

Designed for high-frequency trading (HFT), backtesting, and live data pipelines. Built with correctness, speed, and determinism as first-class citizens.

## 🚀 Key Features

*   **Zero-Copy Streaming**: All indicators implement `StreamingIndicator` for O(1) incremental updates.
*   **Deterministic**: Uses `IndexMap` based DataFrames to ensure reproducible iteration order and state hashes.
*   **Production Ready**: Full `serde` support for state checkpointing and restoration.
*   **no_std Compatible**: Core crates support embedded/WASM environments (via `libm`).
*   **SIMD Optimized**: Critical paths accelerated using `wide` for vectorization (feature-gated).
*   **Python Bindings**: Drop-in replacement for `python-ta` with Rust speed (via `pyo3`).
*   **Golden Tests**: Validated against `python-ta` reference implementations using real market data.

## Performance Benchmarks

Benchmarked on 100,000 candles (5 iterations)

| Indicator | python-ta | rust-ta | Speedup |
|-----------|-----------|---------|---------|
| SMA(14) | 737.0μs | 99.9μs | **7x** |
| SMA(50) | 626.6μs | 74.5μs | **8x** |
| SMA(200) | 632.3μs | 73.4μs | **9x** |
| EMA(14) | 498.8μs | 128.7μs | **4x** |
| EMA(50) | 494.4μs | 129.7μs | **4x** |
| WMA(14) | 2.61s | 228.7μs | **11432x** |
| MACD | 1.94ms | 1.16ms | **2x** |
| ADX(14) | 313.84ms | 1.50ms | **209x** |
| Aroon(25) | 140.66ms | 3.94ms | **36x** |
| RSI(14) | 2.90ms | 859.0μs | **3x** |
| RSI(7) | 2.59ms | 849.7μs | **3x** |
| Stochastic | 3.35ms | 1.09ms | **3x** |
| Williams %R | 3.32ms | 725.4μs | **5x** |
| ROC(12) | 342.7μs | 80.7μs | **4x** |
| Bollinger | 1.97ms | 1.69ms | **1x** |
| ATR(14) | 167.63ms | 540.8μs | **310x** |
| Keltner | 2.66ms | 1.39ms | **2x** |
| Donchian | 2.91ms | 1.62ms | **2x** |
| OBV | 702.3μs | 415.8μs | **2x** |
| MFI(14) | 442.05ms | 1.10ms | **402x** |
| CMF(20) | 1.73ms | 206.4μs | **8x** |
| **Total** | **3.71s** | **17.90ms** | **207x** |

## 📦 Architecture

The workspace is modular to minimize compile times and binary size:

| Crate | Description |
|-------|-------------|
| `ta-core` | Foundational types (`Series`, `DataFrame`, `RingBuffer`) and traits (`Indicator`, `Transform`). |
| `ta-indicators` | The meat. RSI, MACD, Bollinger Bands, ATR, etc. |
| `ta-transforms` | Scikit-learn style data pipelines (`LogReturn`, `StandardScaler`, `RobustScaler`). |
| `ta-simd` | Low-level vectorized math primitives. |
| `ta-python` | Python FFI bindings. |

## 🛠️ Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ta-core = "0.1.0"
ta-indicators = "0.1.0"
# Optional: for data pipelines
# ta-transforms = "0.1.0"
```

## ⚡ Quick Start

### 1. Streaming (Live Trading)

Ideal for processing websocket feeds. Constant memory usage.

```rust
use ta_core::prelude::*;
use ta_indicators::prelude::*;

fn main() -> Result<()> {
    // Configure RSI with 14-period window
    let config = RsiConfig::new(14);
    let mut rsi = Rsi::<f64>::new(config);

    // Simulate incoming price data
    let prices = vec![100.0, 101.0, 102.5, 99.0, 98.5];

    for price in prices {
        // Create a bar (simplified for example)
        let bar = Bar::new(price, price, price, price, 1000.0);
        
        // Update state and get current value
        if let Some(value) = rsi.update(&bar)? {
            println!("Current RSI: {:.2}", value);
        }
    }
    
    // Checkpoint state
    let state = rsi.get_state();
    // save_to_db(&state)?;
    
    Ok(())
}
```

### 2. Batch Processing (Backtesting)

Process entire history at once using vectorized operations where possible.

```rust
use ta_core::prelude::*;
use ta_indicators::prelude::*;

fn main() -> Result<()> {
    // Load your data into an OHLCV series
    let mut ohlcv = OhlcvSeries::new();
    // ... populate ohlcv ...

    // Calculate indicator over the whole series
    let macd = Macd::new(MacdConfig::default());
    let output = macd.calculate(&ohlcv)?;

    println!("Last MACD: {:?}", output.macd.last());
    println!("Last Signal: {:?}", output.signal.last());
    println!("Last Hist: {:?}", output.histogram.last());

    Ok(())
}
```

### 3. Data Pipelines (ML Prep)

Pre-process data for machine learning models.

```rust
use ta_transforms::prelude::*;

fn main() -> Result<()> {
    // Create a pipeline: Log Returns -> Robust Scaling
    let mut pipeline = TransformPipeline::new()
        .add(LogReturnTransform::new(LogReturnConfig::default()))
        .add(RobustNormalizationTransform::new(RobustNormalizationConfig::default()));

    // Fit on training data and transform
    let transformed_df = pipeline.fit_transform(&raw_df)?;
    
    Ok(())
}
```

## 🐍 Python Usage

Install the bindings (requires Rust toolchain):

```bash
pip install .
```

```python
import rust_ta
import pandas as pd

df = pd.read_csv("data.csv")

# Use Rust implementation of RSI
rsi = rust_ta.rsi(df["close"].values, window=14)
```

## ✅ Correctness

We don't just guess. This library includes a comprehensive test suite that compares outputs against `python-ta` using:
1.  **Synthetic Data**: Random walks, trends, and edge cases (NaNs, zero volume).
2.  **Real Market Data**: Fetched from TimescaleDB (crypto/forex) via `scripts/fetch_ohlcv.py`.
3.  **Property Testing**: `proptest` ensures invariants hold (e.g., `Bollinger Lower <= Middle <= Upper`).

Run the golden tests:
```bash
cargo test --package ta-indicators --test golden_tests
```

## 📜 License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See [LICENSE-APACHE](LICENSE-APACHE) for details.
