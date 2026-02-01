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
