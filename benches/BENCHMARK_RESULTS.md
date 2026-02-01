## Performance Benchmarks

Benchmarked on 100,000 candles (25 iterations)

| Indicator | python-ta | rust-ta | Speedup | Description |
|-----------|-----------|---------|---------|-------------|
| SMA(14) | 647.5μs | 82.9μs | **8x** | Simple moving average |
| SMA(50) | 611.0μs | 80.6μs | **8x** | Simple moving average |
| SMA(200) | 595.3μs | 77.8μs | **8x** | Simple moving average |
| EMA(14) | 536.5μs | 137.8μs | **4x** | Exponential moving average |
| EMA(50) | 525.5μs | 138.1μs | **4x** | Exponential moving average |
| WMA(14) | 2.81s | 244.3μs | **11508x** | Weighted moving average |
| MACD | 1.73ms | 1.32ms | **1x** | Moving average convergence/divergence |
| ADX(14) | 341.09ms | 1.62ms | **211x** | Average directional index (trend strength) |
| Aroon(25) | 147.45ms | 4.22ms | **35x** | Trend direction and strength |
| RSI(14) | 2.52ms | 908.6μs | **3x** | Relative strength index (momentum) |
| RSI(7) | 2.55ms | 913.4μs | **3x** | Relative strength index (momentum) |
| Stochastic | 3.28ms | 1.14ms | **3x** | Stochastic oscillator (%K, %D) |
| Williams %R | 3.28ms | 766.0μs | **4x** | Overbought/oversold oscillator |
| ROC(12) | 313.0μs | 78.7μs | **4x** | Rate of change (price momentum) |
| Bollinger | 2.25ms | 1.83ms | **1x** | Volatility bands around SMA |
| ATR(14) | 183.09ms | 540.0μs | **339x** | Average true range (volatility) |
| Keltner | 2.78ms | 1.48ms | **2x** | Volatility channel around EMA |
| Donchian | 3.18ms | 1.73ms | **2x** | High/low breakout channel |
| OBV | 726.0μs | 417.4μs | **2x** | On-balance volume (accumulation) |
| MFI(14) | 479.04ms | 1.21ms | **394x** | Volume-weighted RSI |
| CMF(20) | 1.70ms | 214.2μs | **8x** | Chaikin money flow |
