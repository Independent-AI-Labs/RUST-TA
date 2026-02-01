//! Property-based tests for ta-indicators.
//!
//! These tests verify invariants that must hold for all inputs.

use proptest::prelude::*;
use ta_core::ohlcv::{Bar, OhlcvSeries};
use ta_core::traits::{Indicator, StreamingIndicator};

use ta_indicators::prelude::*;

// ============================================================================
// Proptest Strategies
// ============================================================================

/// Generate a valid close price (positive, finite).
fn valid_price() -> impl Strategy<Value = f64> {
    (0.01f64..10000.0).prop_filter("must be finite", |x| x.is_finite())
}

/// Generate a valid volume (non-negative, finite).
fn valid_volume() -> impl Strategy<Value = f64> {
    (0.0f64..1_000_000.0).prop_filter("must be finite", |x| x.is_finite())
}

/// Generate a valid bar with plausible OHLCV values.
fn valid_bar() -> impl Strategy<Value = Bar<f64>> {
    (valid_price(), valid_volume()).prop_flat_map(|(close, volume)| {
        // Generate O, H, L relative to close
        let range = close * 0.1; // 10% range
        let low = close - range / 2.0;
        let high = close + range / 2.0;
        let open = close; // Simplify for tests

        Just(Bar::new(open, high, low, close, volume))
    })
}

/// Generate a vector of valid bars.
fn valid_ohlcv_series(min_len: usize, max_len: usize) -> impl Strategy<Value = OhlcvSeries<f64>> {
    prop::collection::vec(valid_bar(), min_len..=max_len).prop_map(|bars| {
        let mut series = OhlcvSeries::new();
        for bar in bars {
            series.push(bar);
        }
        series
    })
}

/// Generate a constant price series.
fn constant_price_series(len: usize) -> impl Strategy<Value = OhlcvSeries<f64>> {
    (valid_price(), valid_volume()).prop_map(move |(price, volume)| {
        let mut series = OhlcvSeries::new();
        for _ in 0..len {
            series.push(Bar::new(price, price, price, price, volume));
        }
        series
    })
}

/// Generate a monotonically increasing price series.
fn increasing_price_series(len: usize) -> impl Strategy<Value = OhlcvSeries<f64>> {
    (valid_price(), valid_volume()).prop_map(move |(start, volume)| {
        let mut series = OhlcvSeries::new();
        for i in 0..len {
            let price = start + (i as f64) * 1.0;
            series.push(Bar::new(price, price, price, price, volume));
        }
        series
    })
}

/// Generate a monotonically decreasing price series.
fn decreasing_price_series(len: usize) -> impl Strategy<Value = OhlcvSeries<f64>> {
    (valid_price(), valid_volume())
        .prop_filter("start must be high enough", |(start, _)| *start > 100.0)
        .prop_map(move |(start, volume)| {
            let mut series = OhlcvSeries::new();
            for i in 0..len {
                let price = start - (i as f64) * 1.0;
                series.push(Bar::new(price, price, price, price, volume));
            }
            series
        })
}

// ============================================================================
// SMA Property Tests
// ============================================================================

proptest! {
    /// SMA of constant series should equal the constant.
    #[test]
    fn sma_constant_equals_input(
        ohlcv in constant_price_series(20),
        window in 2usize..=10,
    ) {
        let config = SmaConfig::new(window);
        let sma = Sma::<f64>::new(config);
        let result = sma.calculate(&ohlcv).unwrap();

        let expected_price = ohlcv.iter().next().unwrap().close;
        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                let diff = (val - expected_price).abs();
                prop_assert!(diff < 1e-10, "SMA of constant should equal constant at index {}: {} vs {}", i, val, expected_price);
            }
        }
    }

    /// SMA should always produce finite or NaN values.
    #[test]
    fn sma_finite_or_nan(
        ohlcv in valid_ohlcv_series(5, 50),
        window in 2usize..=10,
    ) {
        let config = SmaConfig::new(window);
        let sma = Sma::<f64>::new(config);
        let result = sma.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            prop_assert!(val.is_nan() || val.is_finite(), "SMA should be finite or NaN at index {}: {}", i, val);
        }
    }

    /// SMA streaming should equal batch.
    #[test]
    fn sma_streaming_equals_batch(
        ohlcv in valid_ohlcv_series(10, 30),
        window in 2usize..=5,
    ) {
        let config = SmaConfig::new(window);

        // Batch calculation
        let batch_sma = Sma::<f64>::new(config.clone());
        let batch_result = batch_sma.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_sma = Sma::<f64>::new(config);
        for (i, bar) in ohlcv.iter().enumerate() {
            let streaming_result = streaming_sma.update(&bar).unwrap();

            if let Some(val) = streaming_result {
                if !batch_result[i].is_nan() {
                    let diff = (val - batch_result[i]).abs();
                    prop_assert!(diff < 1e-10, "SMA streaming[{}] should equal batch: {} vs {}", i, val, batch_result[i]);
                }
            }
        }
    }
}

// ============================================================================
// EMA Property Tests
// ============================================================================

proptest! {
    /// EMA of constant series should converge to the constant.
    #[test]
    fn ema_constant_converges(
        ohlcv in constant_price_series(30),
        window in 2usize..=10,
    ) {
        let config = EmaConfig::new(window);
        let ema = Ema::<f64>::new(config);
        let result = ema.calculate(&ohlcv).unwrap();

        let expected_price = ohlcv.iter().next().unwrap().close;
        // After enough periods, EMA should converge to the constant
        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() && i >= window * 2 {
                let diff = (val - expected_price).abs();
                prop_assert!(diff < 1e-6, "EMA of constant should converge at index {}: {} vs {}", i, val, expected_price);
            }
        }
    }

    /// EMA streaming should equal batch.
    #[test]
    fn ema_streaming_equals_batch(
        ohlcv in valid_ohlcv_series(10, 30),
        window in 2usize..=5,
    ) {
        let config = EmaConfig::new(window);

        // Batch calculation
        let batch_ema = Ema::<f64>::new(config.clone());
        let batch_result = batch_ema.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_ema = Ema::<f64>::new(config);
        for (i, bar) in ohlcv.iter().enumerate() {
            let streaming_result = streaming_ema.update(&bar).unwrap();

            if let Some(val) = streaming_result {
                if !batch_result[i].is_nan() {
                    let diff = (val - batch_result[i]).abs();
                    prop_assert!(diff < 1e-10, "EMA streaming[{}] should equal batch: {} vs {}", i, val, batch_result[i]);
                }
            }
        }
    }
}

// ============================================================================
// RSI Property Tests
// ============================================================================

proptest! {
    /// RSI should always be in [0, 100] range.
    #[test]
    fn rsi_bounded(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=14,
    ) {
        let config = RsiConfig::new(window);
        let rsi = Rsi::<f64>::new(config);
        let result = rsi.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                prop_assert!(val >= 0.0, "RSI should be >= 0 at index {}: {}", i, val);
                prop_assert!(val <= 100.0, "RSI should be <= 100 at index {}: {}", i, val);
            }
        }
    }

    /// RSI of all gains should be 100.
    #[test]
    fn rsi_all_gains_is_100(
        ohlcv in increasing_price_series(30),
        window in 5usize..=10,
    ) {
        let config = RsiConfig::new(window);
        let rsi = Rsi::<f64>::new(config);
        let result = rsi.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                let diff = (val - 100.0).abs();
                prop_assert!(diff < 1e-6, "RSI of all gains should be 100 at index {}: {}", i, val);
            }
        }
    }

    /// RSI of all losses should be 0.
    #[test]
    fn rsi_all_losses_is_0(
        ohlcv in decreasing_price_series(30),
        window in 5usize..=10,
    ) {
        let config = RsiConfig::new(window);
        let rsi = Rsi::<f64>::new(config);
        let result = rsi.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                let diff = val.abs();
                prop_assert!(diff < 1e-6, "RSI of all losses should be 0 at index {}: {}", i, val);
            }
        }
    }

    /// RSI of constant should be 50 (neutral).
    #[test]
    fn rsi_constant_is_50(
        ohlcv in constant_price_series(30),
        window in 5usize..=10,
    ) {
        let config = RsiConfig::new(window);
        let rsi = Rsi::<f64>::new(config);
        let result = rsi.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                let diff = (val - 50.0).abs();
                prop_assert!(diff < 1e-6, "RSI of constant should be 50 at index {}: {}", i, val);
            }
        }
    }

    /// RSI streaming should equal batch.
    #[test]
    fn rsi_streaming_equals_batch(
        ohlcv in valid_ohlcv_series(20, 40),
        window in 5usize..=10,
    ) {
        let config = RsiConfig::new(window);

        // Batch calculation
        let batch_rsi = Rsi::<f64>::new(config.clone());
        let batch_result = batch_rsi.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_rsi = Rsi::<f64>::new(config);
        for (i, bar) in ohlcv.iter().enumerate() {
            let streaming_result = streaming_rsi.update(&bar).unwrap();

            if let Some(val) = streaming_result {
                if !batch_result[i].is_nan() {
                    let rel_diff = ((val - batch_result[i]) / batch_result[i]).abs();
                    prop_assert!(rel_diff < 1e-8, "RSI streaming[{}] should equal batch: {} vs {}", i, val, batch_result[i]);
                }
            }
        }
    }
}

// ============================================================================
// Bollinger Bands Property Tests
// ============================================================================

proptest! {
    /// Bollinger Bands: lower <= middle <= upper.
    #[test]
    fn bollinger_ordered(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=20,
    ) {
        let config = BollingerConfig::new(window, 2.0);
        let bb = BollingerBands::<f64>::new(config);
        let result = bb.calculate(&ohlcv).unwrap();

        for i in 0..result.middle.len() {
            if !result.middle[i].is_nan() {
                prop_assert!(result.lower[i] <= result.middle[i] + 1e-10,
                    "lower ({}) should be <= middle ({}) at index {}",
                    result.lower[i], result.middle[i], i);
                prop_assert!(result.middle[i] <= result.upper[i] + 1e-10,
                    "middle ({}) should be <= upper ({}) at index {}",
                    result.middle[i], result.upper[i], i);
            }
        }
    }

    /// Bollinger Bands of constant should have zero width.
    #[test]
    fn bollinger_constant_zero_width(
        ohlcv in constant_price_series(30),
        window in 5usize..=15,
    ) {
        let config = BollingerConfig::new(window, 2.0);
        let bb = BollingerBands::<f64>::new(config);
        let result = bb.calculate(&ohlcv).unwrap();

        for (i, &width) in result.width.iter().enumerate() {
            if !width.is_nan() {
                // Allow small floating-point precision error in standard deviation calculation
                prop_assert!(width.abs() < 1e-6,
                    "Bollinger width of constant should be near zero at index {}: {}", i, width);
            }
        }
    }

    /// Bollinger %B should be 0.5 when close equals middle.
    #[test]
    fn bollinger_pct_b_constant_is_half(
        ohlcv in constant_price_series(30),
        window in 5usize..=15,
    ) {
        let config = BollingerConfig::new(window, 2.0);
        let bb = BollingerBands::<f64>::new(config);
        let result = bb.calculate(&ohlcv).unwrap();

        for (i, &pct_b) in result.pct_b.iter().enumerate() {
            if !pct_b.is_nan() {
                let diff = (pct_b - 0.5).abs();
                prop_assert!(diff < 1e-6,
                    "Bollinger %B of constant should be 0.5 at index {}: {}", i, pct_b);
            }
        }
    }
}

// ============================================================================
// ATR Property Tests
// ============================================================================

proptest! {
    /// ATR should always be non-negative.
    #[test]
    fn atr_non_negative(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=14,
    ) {
        let config = AtrConfig::new(window);
        let atr = Atr::<f64>::new(config);
        let result = atr.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                prop_assert!(val >= 0.0, "ATR should be >= 0 at index {}: {}", i, val);
            }
        }
    }

    /// ATR of constant should be zero.
    #[test]
    fn atr_constant_is_zero(
        ohlcv in constant_price_series(30),
        window in 5usize..=14,
    ) {
        let config = AtrConfig::new(window);
        let atr = Atr::<f64>::new(config);
        let result = atr.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                prop_assert!(val.abs() < 1e-10, "ATR of constant should be zero at index {}: {}", i, val);
            }
        }
    }

    /// ATR streaming should equal batch.
    #[test]
    fn atr_streaming_equals_batch(
        ohlcv in valid_ohlcv_series(20, 40),
        window in 5usize..=10,
    ) {
        let config = AtrConfig::new(window);

        // Batch calculation
        let batch_atr = Atr::<f64>::new(config.clone());
        let batch_result = batch_atr.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_atr = Atr::<f64>::new(config);
        for (i, bar) in ohlcv.iter().enumerate() {
            let streaming_result = streaming_atr.update(&bar).unwrap();

            if let Some(val) = streaming_result {
                if !batch_result[i].is_nan() {
                    let diff = (val - batch_result[i]).abs();
                    prop_assert!(diff < 1e-10, "ATR streaming[{}] should equal batch: {} vs {}", i, val, batch_result[i]);
                }
            }
        }
    }
}

// ============================================================================
// Stochastic Property Tests
// ============================================================================

proptest! {
    /// Stochastic %K should be in [0, 100] range.
    #[test]
    fn stochastic_k_bounded(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=14,
    ) {
        let config = StochasticConfig::new(window, 3, 3);
        let stoch = Stochastic::<f64>::new(config);
        let result = stoch.calculate(&ohlcv).unwrap();

        for (i, &k) in result.k.iter().enumerate() {
            if !k.is_nan() {
                prop_assert!(k >= 0.0 - 1e-10, "Stochastic %K should be >= 0 at index {}: {}", i, k);
                prop_assert!(k <= 100.0 + 1e-10, "Stochastic %K should be <= 100 at index {}: {}", i, k);
            }
        }
    }

    /// Stochastic %D should be in [0, 100] range.
    #[test]
    fn stochastic_d_bounded(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=14,
    ) {
        let config = StochasticConfig::new(window, 3, 3);
        let stoch = Stochastic::<f64>::new(config);
        let result = stoch.calculate(&ohlcv).unwrap();

        for (i, &d) in result.d.iter().enumerate() {
            if !d.is_nan() {
                prop_assert!(d >= 0.0 - 1e-10, "Stochastic %D should be >= 0 at index {}: {}", i, d);
                prop_assert!(d <= 100.0 + 1e-10, "Stochastic %D should be <= 100 at index {}: {}", i, d);
            }
        }
    }
}

// ============================================================================
// Williams %R Property Tests
// ============================================================================

proptest! {
    /// Williams %R should be in [-100, 0] range.
    #[test]
    fn williams_r_bounded(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=14,
    ) {
        let config = WilliamsRConfig::new(window);
        let wr = WilliamsR::<f64>::new(config);
        let result = wr.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                prop_assert!(val >= -100.0 - 1e-10, "Williams %R should be >= -100 at index {}: {}", i, val);
                prop_assert!(val <= 0.0 + 1e-10, "Williams %R should be <= 0 at index {}: {}", i, val);
            }
        }
    }
}

// ============================================================================
// ADX Property Tests
// ============================================================================

proptest! {
    /// +DI and -DI should be in [0, 100] range.
    #[test]
    fn adx_di_bounded(
        ohlcv in valid_ohlcv_series(30, 50),
        window in 5usize..=14,
    ) {
        let config = AdxConfig::new(window);
        let adx = Adx::<f64>::new(config);
        let result = adx.calculate(&ohlcv).unwrap();

        for i in 0..result.plus_di.len() {
            if !result.plus_di[i].is_nan() {
                prop_assert!(result.plus_di[i] >= 0.0 - 1e-6, "+DI should be >= 0 at index {}: {}", i, result.plus_di[i]);
                prop_assert!(result.plus_di[i] <= 100.0 + 1e-6, "+DI should be <= 100 at index {}: {}", i, result.plus_di[i]);
            }
            if !result.minus_di[i].is_nan() {
                prop_assert!(result.minus_di[i] >= 0.0 - 1e-6, "-DI should be >= 0 at index {}: {}", i, result.minus_di[i]);
                prop_assert!(result.minus_di[i] <= 100.0 + 1e-6, "-DI should be <= 100 at index {}: {}", i, result.minus_di[i]);
            }
        }
    }

    /// ADX should be in [0, 100] range.
    #[test]
    fn adx_bounded(
        ohlcv in valid_ohlcv_series(40, 60),
        window in 5usize..=14,
    ) {
        let config = AdxConfig::new(window);
        let adx = Adx::<f64>::new(config);
        let result = adx.calculate(&ohlcv).unwrap();

        for (i, &val) in result.adx.iter().enumerate() {
            if !val.is_nan() {
                prop_assert!(val >= 0.0 - 1e-6, "ADX should be >= 0 at index {}: {}", i, val);
                prop_assert!(val <= 100.0 + 1e-6, "ADX should be <= 100 at index {}: {}", i, val);
            }
        }
    }
}

// ============================================================================
// Aroon Property Tests
// ============================================================================

proptest! {
    /// Aroon Up and Down should be in [0, 100] range.
    #[test]
    fn aroon_bounded(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=25,
    ) {
        let config = AroonConfig::new(window);
        let aroon = Aroon::<f64>::new(config);
        let result = aroon.calculate(&ohlcv).unwrap();

        for i in 0..result.aroon_up.len() {
            if !result.aroon_up[i].is_nan() {
                prop_assert!(result.aroon_up[i] >= 0.0 - 1e-10, "Aroon Up should be >= 0 at index {}: {}", i, result.aroon_up[i]);
                prop_assert!(result.aroon_up[i] <= 100.0 + 1e-10, "Aroon Up should be <= 100 at index {}: {}", i, result.aroon_up[i]);
            }
            if !result.aroon_down[i].is_nan() {
                prop_assert!(result.aroon_down[i] >= 0.0 - 1e-10, "Aroon Down should be >= 0 at index {}: {}", i, result.aroon_down[i]);
                prop_assert!(result.aroon_down[i] <= 100.0 + 1e-10, "Aroon Down should be <= 100 at index {}: {}", i, result.aroon_down[i]);
            }
        }
    }
}

// ============================================================================
// MFI Property Tests
// ============================================================================

proptest! {
    /// MFI should be in [0, 100] range.
    #[test]
    fn mfi_bounded(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=14,
    ) {
        let config = MfiConfig::new(window);
        let mfi = Mfi::<f64>::new(config);
        let result = mfi.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                prop_assert!(val >= 0.0 - 1e-6, "MFI should be >= 0 at index {}: {}", i, val);
                prop_assert!(val <= 100.0 + 1e-6, "MFI should be <= 100 at index {}: {}", i, val);
            }
        }
    }
}

// ============================================================================
// CMF Property Tests
// ============================================================================

proptest! {
    /// CMF should be in [-1, 1] range.
    #[test]
    fn cmf_bounded(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=20,
    ) {
        let config = CmfConfig::new(window);
        let cmf = Cmf::<f64>::new(config);
        let result = cmf.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                prop_assert!(val >= -1.0 - 1e-6, "CMF should be >= -1 at index {}: {}", i, val);
                prop_assert!(val <= 1.0 + 1e-6, "CMF should be <= 1 at index {}: {}", i, val);
            }
        }
    }
}

// ============================================================================
// ROC Property Tests
// ============================================================================

proptest! {
    /// ROC of constant should be zero.
    #[test]
    fn roc_constant_is_zero(
        ohlcv in constant_price_series(30),
        window in 1usize..=10,
    ) {
        let config = RocConfig::new(window);
        let roc = Roc::<f64>::new(config);
        let result = roc.calculate(&ohlcv).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            if !val.is_nan() {
                prop_assert!(val.abs() < 1e-10, "ROC of constant should be zero at index {}: {}", i, val);
            }
        }
    }
}

// ============================================================================
// Keltner Channel Property Tests
// ============================================================================

proptest! {
    /// Keltner Channel: lower <= middle <= upper.
    #[test]
    fn keltner_ordered(
        ohlcv in valid_ohlcv_series(30, 50),
        window in 5usize..=20,
    ) {
        let config = KeltnerConfig::new(window, window, 2.0);
        let kc = KeltnerChannel::<f64>::new(config);
        let result = kc.calculate(&ohlcv).unwrap();

        for i in 0..result.middle.len() {
            if !result.middle[i].is_nan() {
                prop_assert!(result.lower[i] <= result.middle[i] + 1e-10,
                    "lower ({}) should be <= middle ({}) at index {}",
                    result.lower[i], result.middle[i], i);
                prop_assert!(result.middle[i] <= result.upper[i] + 1e-10,
                    "middle ({}) should be <= upper ({}) at index {}",
                    result.middle[i], result.upper[i], i);
            }
        }
    }
}

// ============================================================================
// Donchian Channel Property Tests
// ============================================================================

proptest! {
    /// Donchian Channel: lower <= middle <= upper.
    #[test]
    fn donchian_ordered(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=20,
    ) {
        let config = DonchianConfig::new(window);
        let dc = DonchianChannel::<f64>::new(config);
        let result = dc.calculate(&ohlcv).unwrap();

        for i in 0..result.middle.len() {
            if !result.middle[i].is_nan() {
                prop_assert!(result.lower[i] <= result.middle[i] + 1e-10,
                    "lower ({}) should be <= middle ({}) at index {}",
                    result.lower[i], result.middle[i], i);
                prop_assert!(result.middle[i] <= result.upper[i] + 1e-10,
                    "middle ({}) should be <= upper ({}) at index {}",
                    result.middle[i], result.upper[i], i);
            }
        }
    }

    /// Donchian middle should equal (upper + lower) / 2.
    #[test]
    fn donchian_middle_is_average(
        ohlcv in valid_ohlcv_series(20, 50),
        window in 5usize..=20,
    ) {
        let config = DonchianConfig::new(window);
        let dc = DonchianChannel::<f64>::new(config);
        let result = dc.calculate(&ohlcv).unwrap();

        for i in 0..result.middle.len() {
            if !result.middle[i].is_nan() {
                let expected = (result.upper[i] + result.lower[i]) / 2.0;
                let diff = (result.middle[i] - expected).abs();
                prop_assert!(diff < 1e-10,
                    "middle ({}) should be average of upper ({}) and lower ({}) at index {}",
                    result.middle[i], result.upper[i], result.lower[i], i);
            }
        }
    }
}
