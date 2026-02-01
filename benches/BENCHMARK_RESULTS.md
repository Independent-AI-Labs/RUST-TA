## Performance Benchmarks

Benchmarked on 100,000 candles (5 iterations)

| Indicator | python-ta | rust-ta | Speedup | Description |
|-----------|-----------|---------|---------|-------------|
| SMA(14) | 737.0μs | 99.9μs | **7x** | Simple moving average |
| SMA(50) | 626.6μs | 74.5μs | **8x** | Simple moving average |
| SMA(200) | 632.3μs | 73.4μs | **9x** | Simple moving average |
| EMA(14) | 498.8μs | 128.7μs | **4x** | Exponential moving average |
| EMA(50) | 494.4μs | 129.7μs | **4x** | Exponential moving average |
| WMA(14) | 2.61s | 228.7μs | **11432x** | Weighted moving average |
| MACD | 1.94ms | 1.16ms | **2x** | Moving average convergence/divergence |
| ADX(14) | 313.84ms | 1.50ms | **209x** | Average directional index (trend strength) |
| Aroon(25) | 140.66ms | 3.94ms | **36x** | Trend direction and strength |
| RSI(14) | 2.90ms | 859.0μs | **3x** | Relative strength index (momentum) |
| RSI(7) | 2.59ms | 849.7μs | **3x** | Relative strength index (momentum) |
| Stochastic | 3.35ms | 1.09ms | **3x** | Stochastic oscillator (%K, %D) |
| Williams %R | 3.32ms | 725.4μs | **5x** | Overbought/oversold oscillator |
| ROC(12) | 342.7μs | 80.7μs | **4x** | Rate of change (price momentum) |
| Bollinger | 1.97ms | 1.69ms | **1x** | Volatility bands around SMA |
| ATR(14) | 167.63ms | 540.8μs | **310x** | Average true range (volatility) |
| Keltner | 2.66ms | 1.39ms | **2x** | Volatility channel around EMA |
| Donchian | 2.91ms | 1.62ms | **2x** | High/low breakout channel |
| OBV | 702.3μs | 415.8μs | **2x** | On-balance volume (accumulation) |
| MFI(14) | 442.05ms | 1.10ms | **402x** | Volume-weighted RSI |
| CMF(20) | 1.73ms | 206.4μs | **8x** | Chaikin money flow |
