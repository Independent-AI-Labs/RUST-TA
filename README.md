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
