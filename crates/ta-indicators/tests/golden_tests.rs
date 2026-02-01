//! Golden data tests for ta-indicators.
//!
//! These tests compare indicator outputs against pre-computed values from python-ta.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::Deserialize;
use ta_core::ohlcv::{Bar, OhlcvSeries};
use ta_core::traits::{Indicator, StreamingIndicator};

use ta_indicators::prelude::*;

// ============================================================================
// Test Utilities
// ============================================================================

/// Golden data entry with input OHLCV and expected outputs.
#[derive(Debug, Deserialize)]
struct GoldenData {
    /// Input OHLCV data.
    ohlcv: OhlcvData,
    /// Expected indicator outputs.
    expected: HashMap<String, Vec<Option<f64>>>,
}

#[derive(Debug, Deserialize)]
struct OhlcvData {
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
}

impl GoldenData {
    /// Load golden data from a JSON file.
    fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data: serde_json::Value = serde_json::from_reader(reader)?;

        let ohlcv = data
            .get("ohlcv")
            .ok_or("Missing 'ohlcv' field")?;

        let ohlcv_data = OhlcvData {
            open: parse_array(ohlcv.get("open"))?,
            high: parse_array(ohlcv.get("high"))?,
            low: parse_array(ohlcv.get("low"))?,
            close: parse_array(ohlcv.get("close"))?,
            volume: parse_array(ohlcv.get("volume"))?,
        };

        let mut expected = HashMap::new();
        if let Some(exp) = data.get("expected").and_then(|e| e.as_object()) {
            for (key, value) in exp {
                expected.insert(key.clone(), parse_option_array(Some(value))?);
            }
        }

        Ok(GoldenData {
            ohlcv: ohlcv_data,
            expected,
        })
    }

    /// Convert to OhlcvSeries for use with indicators.
    fn to_ohlcv_series(&self) -> OhlcvSeries<f64> {
        let mut series = OhlcvSeries::new();
        for i in 0..self.ohlcv.close.len() {
            series.push(Bar::new(
                self.ohlcv.open[i],
                self.ohlcv.high[i],
                self.ohlcv.low[i],
                self.ohlcv.close[i],
                self.ohlcv.volume[i],
            ));
        }
        series
    }

    /// Get expected values for an indicator.
    fn get_expected(&self, name: &str) -> Option<&Vec<Option<f64>>> {
        self.expected.get(name)
    }
}

fn parse_array(value: Option<&serde_json::Value>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    match value {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| format!("Invalid number: {:?}", v).into())
            })
            .collect(),
        _ => Ok(Vec::new()),
    }
}

fn parse_option_array(
    value: Option<&serde_json::Value>,
) -> Result<Vec<Option<f64>>, Box<dyn std::error::Error>> {
    match value {
        Some(serde_json::Value::Array(arr)) => Ok(arr
            .iter()
            .map(|v| {
                if v.is_null() {
                    None
                } else {
                    v.as_f64()
                }
            })
            .collect()),
        _ => Ok(Vec::new()),
    }
}

/// Assert two floats are approximately equal with tiered tolerance.
fn assert_float_eq(actual: f64, expected: f64, epsilon: f64, context: &str) {
    if expected.is_nan() {
        assert!(
            actual.is_nan(),
            "{}: Expected NaN but got {}",
            context,
            actual
        );
        return;
    }

    if actual.is_nan() {
        panic!("{}: Got NaN but expected {}", context, expected);
    }

    if expected.is_infinite() {
        assert!(
            actual.is_infinite() && actual.signum() == expected.signum(),
            "{}: Expected {} but got {}",
            context,
            expected,
            actual
        );
        return;
    }

    let abs_expected = expected.abs();

    // For values very close to zero, use absolute comparison
    if abs_expected < 1e-10 {
        let diff = (actual - expected).abs();
        assert!(
            diff < epsilon,
            "{}: Expected {} but got {} (diff: {})",
            context,
            expected,
            actual,
            diff
        );
        return;
    }

    // For other values, use relative comparison
    let rel_diff = ((actual - expected) / expected).abs();
    assert!(
        rel_diff < epsilon,
        "{}: Expected {} but got {} (rel diff: {:.2e})",
        context,
        expected,
        actual,
        rel_diff
    );
}

/// Assert a series matches expected optional values.
fn assert_series_eq_optional(
    actual: &[f64],
    expected: &[Option<f64>],
    epsilon: f64,
    name: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: Length mismatch: {} vs {}",
        name,
        actual.len(),
        expected.len()
    );

    for (i, (&a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let context = format!("{}[{}]", name, i);
        match e {
            Some(exp) => assert_float_eq(a, *exp, epsilon, &context),
            None => assert!(
                a.is_nan(),
                "{}: Expected NaN but got {}",
                context,
                a
            ),
        }
    }
}

// ============================================================================
// Golden Data Path Helper
// ============================================================================

fn golden_path(filename: &str) -> String {
    // Look for golden data relative to workspace root
    let workspace_root = std::env::var("CARGO_MANIFEST_DIR")
        .map(|dir| {
            std::path::Path::new(&dir)
                .parent()
                .and_then(|p| p.parent())
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf()
        })
        .unwrap_or_else(|_| std::path::PathBuf::from("."));

    workspace_root
        .join("golden")
        .join("python_outputs")
        .join(filename)
        .to_string_lossy()
        .to_string()
}

// ============================================================================
// SMA Tests
// ============================================================================

#[test]
fn test_sma_golden() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();

    if let Some(expected) = golden.get_expected("sma_5") {
        let config = SmaConfig::new(5);
        let sma = Sma::<f64>::new(config);
        let result = sma.calculate(&ohlcv).unwrap();

        assert_series_eq_optional(result.as_slice(), expected, 1e-6, "SMA(5)");
    }
}

#[test]
fn test_sma_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = SmaConfig::new(5);

    // Batch calculation
    let batch_sma = Sma::<f64>::new(config.clone());
    let batch_result = batch_sma.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_sma = Sma::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_sma.update(&bar).unwrap();

        if let Some(val) = streaming_result {
            if !batch_result[i].is_nan() {
                assert_float_eq(
                    val,
                    batch_result[i],
                    1e-10,
                    &format!("SMA streaming[{}]", i),
                );
            }
        }
    }
}

// ============================================================================
// EMA Tests
// ============================================================================

#[test]
fn test_ema_golden() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();

    if let Some(expected) = golden.get_expected("ema_5") {
        let config = EmaConfig::new(5);
        let ema = Ema::<f64>::new(config);
        let result = ema.calculate(&ohlcv).unwrap();

        assert_series_eq_optional(result.as_slice(), expected, 1e-2, "EMA(5)");
    }
}

#[test]
fn test_ema_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = EmaConfig::new(5);

    // Batch calculation
    let batch_ema = Ema::<f64>::new(config.clone());
    let batch_result = batch_ema.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_ema = Ema::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_ema.update(&bar).unwrap();

        if let Some(val) = streaming_result {
            if !batch_result[i].is_nan() {
                assert_float_eq(
                    val,
                    batch_result[i],
                    1e-10,
                    &format!("EMA streaming[{}]", i),
                );
            }
        }
    }
}

// ============================================================================
// RSI Tests
// ============================================================================

#[test]
fn test_rsi_golden() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();

    if let Some(expected) = golden.get_expected("rsi_14") {
        let config = RsiConfig::new(14);
        let rsi = Rsi::<f64>::new(config);
        let result = rsi.calculate(&ohlcv).unwrap();

        assert_series_eq_optional(result.as_slice(), expected, 1e-2, "RSI(14)");
    }
}

#[test]
fn test_rsi_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = RsiConfig::new(5);

    // Batch calculation
    let batch_rsi = Rsi::<f64>::new(config.clone());
    let batch_result = batch_rsi.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_rsi = Rsi::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_rsi.update(&bar).unwrap();

        if let Some(val) = streaming_result {
            if !batch_result[i].is_nan() {
                assert_float_eq(
                    val,
                    batch_result[i],
                    1e-8,
                    &format!("RSI streaming[{}]", i),
                );
            }
        }
    }
}

// ============================================================================
// Bollinger Bands Tests
// ============================================================================

#[test]
fn test_bollinger_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = BollingerConfig::new(5, 2.0);

    // Batch calculation
    let batch_bb = BollingerBands::<f64>::new(config.clone());
    let batch_result = batch_bb.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_bb = BollingerBands::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_bb.update(&bar).unwrap();

        if let Some(output) = streaming_result {
            if !batch_result.middle[i].is_nan() {
                assert_float_eq(
                    output.middle,
                    batch_result.middle[i],
                    1e-10,
                    &format!("Bollinger.middle streaming[{}]", i),
                );
                assert_float_eq(
                    output.upper,
                    batch_result.upper[i],
                    1e-6,
                    &format!("Bollinger.upper streaming[{}]", i),
                );
                assert_float_eq(
                    output.lower,
                    batch_result.lower[i],
                    1e-6,
                    &format!("Bollinger.lower streaming[{}]", i),
                );
            }
        }
    }
}

#[test]
fn test_bollinger_invariant_lower_le_middle_le_upper() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = BollingerConfig::new(5, 2.0);

    let bb = BollingerBands::<f64>::new(config);
    let result = bb.calculate(&ohlcv).unwrap();

    for i in 0..result.middle.len() {
        if !result.middle[i].is_nan() {
            assert!(
                result.lower[i] <= result.middle[i],
                "lower ({}) should be <= middle ({}) at index {}",
                result.lower[i],
                result.middle[i],
                i
            );
            assert!(
                result.middle[i] <= result.upper[i],
                "middle ({}) should be <= upper ({}) at index {}",
                result.middle[i],
                result.upper[i],
                i
            );
        }
    }
}

// ============================================================================
// ATR Tests
// ============================================================================

#[test]
fn test_atr_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = AtrConfig::new(5);

    // Batch calculation
    let batch_atr = Atr::<f64>::new(config.clone());
    let batch_result = batch_atr.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_atr = Atr::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_atr.update(&bar).unwrap();

        if let Some(val) = streaming_result {
            if !batch_result[i].is_nan() {
                assert_float_eq(
                    val,
                    batch_result[i],
                    1e-10,
                    &format!("ATR streaming[{}]", i),
                );
            }
        }
    }
}

#[test]
fn test_atr_always_non_negative() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = AtrConfig::new(5);

    let atr = Atr::<f64>::new(config);
    let result = atr.calculate(&ohlcv).unwrap();

    for (i, &val) in result.as_slice().iter().enumerate() {
        if !val.is_nan() {
            assert!(
                val >= 0.0,
                "ATR should be non-negative, got {} at index {}",
                val,
                i
            );
        }
    }
}

// ============================================================================
// MACD Tests
// ============================================================================

#[test]
fn test_macd_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = MacdConfig::new(3, 5, 2);

    // Batch calculation
    let batch_macd = Macd::<f64>::new(config.clone());
    let batch_result = batch_macd.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_macd = Macd::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_macd.update(&bar).unwrap();

        if let Some(output) = streaming_result {
            if !batch_result.macd[i].is_nan() {
                assert_float_eq(
                    output.macd,
                    batch_result.macd[i],
                    1e-10,
                    &format!("MACD.macd streaming[{}]", i),
                );
            }
        }
    }
}

// ============================================================================
// OBV Tests
// ============================================================================

#[test]
fn test_obv_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = ObvConfig::default();

    // Batch calculation
    let batch_obv = Obv::<f64>::new(config.clone());
    let batch_result = batch_obv.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_obv = Obv::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_obv.update(&bar).unwrap();

        if let Some(val) = streaming_result {
            if !batch_result[i].is_nan() {
                assert_float_eq(
                    val,
                    batch_result[i],
                    1e-10,
                    &format!("OBV streaming[{}]", i),
                );
            }
        }
    }
}

// ============================================================================
// Stochastic Tests
// ============================================================================

#[test]
fn test_stochastic_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = StochasticConfig::new(5, 3, 3);

    // Batch calculation
    let batch_stoch = Stochastic::<f64>::new(config.clone());
    let batch_result = batch_stoch.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_stoch = Stochastic::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_stoch.update(&bar).unwrap();

        if let Some(output) = streaming_result {
            if !batch_result.k[i].is_nan() {
                assert_float_eq(
                    output.k,
                    batch_result.k[i],
                    1e-6,
                    &format!("Stochastic.k streaming[{}]", i),
                );
            }
        }
    }
}

#[test]
fn test_stochastic_bounds() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = StochasticConfig::new(5, 3, 3);

    let stoch = Stochastic::<f64>::new(config);
    let result = stoch.calculate(&ohlcv).unwrap();

    for (i, &k) in result.k.iter().enumerate() {
        if !k.is_nan() {
            assert!(
                k >= 0.0 && k <= 100.0,
                "Stochastic %K should be in [0, 100], got {} at index {}",
                k,
                i
            );
        }
    }
}

// ============================================================================
// Williams %R Tests
// ============================================================================

#[test]
fn test_williams_r_bounds() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = WilliamsRConfig::new(5);

    let wr = WilliamsR::<f64>::new(config);
    let result = wr.calculate(&ohlcv).unwrap();

    for (i, &val) in result.as_slice().iter().enumerate() {
        if !val.is_nan() {
            assert!(
                val >= -100.0 && val <= 0.0,
                "Williams %R should be in [-100, 0], got {} at index {}",
                val,
                i
            );
        }
    }
}

// ============================================================================
// ADX Tests
// ============================================================================

#[test]
fn test_adx_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = AdxConfig::new(5);

    // Batch calculation
    let batch_adx = Adx::<f64>::new(config.clone());
    let batch_result = batch_adx.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_adx = Adx::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_adx.update(&bar).unwrap();

        if let Some(output) = streaming_result {
            if !batch_result.plus_di[i].is_nan() {
                assert_float_eq(
                    output.plus_di,
                    batch_result.plus_di[i],
                    1e-6,
                    &format!("ADX.plus_di streaming[{}]", i),
                );
                assert_float_eq(
                    output.minus_di,
                    batch_result.minus_di[i],
                    1e-6,
                    &format!("ADX.minus_di streaming[{}]", i),
                );
            }
        }
    }
}

#[test]
fn test_adx_di_bounds() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = AdxConfig::new(5);

    let adx = Adx::<f64>::new(config);
    let result = adx.calculate(&ohlcv).unwrap();

    for i in 0..result.adx.len() {
        if !result.plus_di[i].is_nan() {
            assert!(
                result.plus_di[i] >= 0.0 && result.plus_di[i] <= 100.0,
                "+DI should be in [0, 100], got {} at index {}",
                result.plus_di[i],
                i
            );
            assert!(
                result.minus_di[i] >= 0.0 && result.minus_di[i] <= 100.0,
                "-DI should be in [0, 100], got {} at index {}",
                result.minus_di[i],
                i
            );
        }
    }
}

// ============================================================================
// Aroon Tests
// ============================================================================

#[test]
fn test_aroon_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = AroonConfig::new(5);

    // Batch calculation
    let batch_aroon = Aroon::<f64>::new(config.clone());
    let batch_result = batch_aroon.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_aroon = Aroon::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_aroon.update(&bar).unwrap();

        if let Some(output) = streaming_result {
            if !batch_result.aroon_up[i].is_nan() {
                assert_float_eq(
                    output.aroon_up,
                    batch_result.aroon_up[i],
                    1e-6,
                    &format!("Aroon.up streaming[{}]", i),
                );
                assert_float_eq(
                    output.aroon_down,
                    batch_result.aroon_down[i],
                    1e-6,
                    &format!("Aroon.down streaming[{}]", i),
                );
            }
        }
    }
}

#[test]
fn test_aroon_bounds() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = AroonConfig::new(5);

    let aroon = Aroon::<f64>::new(config);
    let result = aroon.calculate(&ohlcv).unwrap();

    for i in 0..result.aroon_up.len() {
        if !result.aroon_up[i].is_nan() {
            assert!(
                result.aroon_up[i] >= 0.0 && result.aroon_up[i] <= 100.0,
                "Aroon Up should be in [0, 100], got {} at index {}",
                result.aroon_up[i],
                i
            );
            assert!(
                result.aroon_down[i] >= 0.0 && result.aroon_down[i] <= 100.0,
                "Aroon Down should be in [0, 100], got {} at index {}",
                result.aroon_down[i],
                i
            );
        }
    }
}

// ============================================================================
// MFI Tests
// ============================================================================

#[test]
fn test_mfi_bounds() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = MfiConfig::new(5);

    let mfi = Mfi::<f64>::new(config);
    let result = mfi.calculate(&ohlcv).unwrap();

    for (i, &val) in result.as_slice().iter().enumerate() {
        if !val.is_nan() {
            assert!(
                val >= 0.0 && val <= 100.0,
                "MFI should be in [0, 100], got {} at index {}",
                val,
                i
            );
        }
    }
}

// ============================================================================
// CMF Tests
// ============================================================================

#[test]
fn test_cmf_bounds() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = CmfConfig::new(5);

    let cmf = Cmf::<f64>::new(config);
    let result = cmf.calculate(&ohlcv).unwrap();

    for (i, &val) in result.as_slice().iter().enumerate() {
        if !val.is_nan() {
            assert!(
                val >= -1.0 && val <= 1.0,
                "CMF should be in [-1, 1], got {} at index {}",
                val,
                i
            );
        }
    }
}

// ============================================================================
// Keltner Channel Tests
// ============================================================================

#[test]
fn test_keltner_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = KeltnerConfig::new(5, 5, 2.0);

    // Batch calculation
    let batch_kc = KeltnerChannel::<f64>::new(config.clone());
    let batch_result = batch_kc.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_kc = KeltnerChannel::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_kc.update(&bar).unwrap();

        if let Some(output) = streaming_result {
            if !batch_result.middle[i].is_nan() {
                assert_float_eq(
                    output.middle,
                    batch_result.middle[i],
                    1e-6,
                    &format!("Keltner.middle streaming[{}]", i),
                );
            }
        }
    }
}

#[test]
fn test_keltner_invariant() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = KeltnerConfig::new(5, 5, 2.0);

    let kc = KeltnerChannel::<f64>::new(config);
    let result = kc.calculate(&ohlcv).unwrap();

    for i in 0..result.middle.len() {
        if !result.middle[i].is_nan() {
            assert!(
                result.lower[i] <= result.middle[i],
                "lower ({}) should be <= middle ({}) at index {}",
                result.lower[i],
                result.middle[i],
                i
            );
            assert!(
                result.middle[i] <= result.upper[i],
                "middle ({}) should be <= upper ({}) at index {}",
                result.middle[i],
                result.upper[i],
                i
            );
        }
    }
}

// ============================================================================
// Donchian Channel Tests
// ============================================================================

#[test]
fn test_donchian_streaming_equals_batch() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = DonchianConfig::new(5);

    // Batch calculation
    let batch_dc = DonchianChannel::<f64>::new(config.clone());
    let batch_result = batch_dc.calculate(&ohlcv).unwrap();

    // Streaming calculation
    let mut streaming_dc = DonchianChannel::<f64>::new(config);
    for (i, bar) in ohlcv.iter().enumerate() {
        let streaming_result = streaming_dc.update(&bar).unwrap();

        if let Some(output) = streaming_result {
            if !batch_result.middle[i].is_nan() {
                assert_float_eq(
                    output.middle,
                    batch_result.middle[i],
                    1e-10,
                    &format!("Donchian.middle streaming[{}]", i),
                );
                assert_float_eq(
                    output.upper,
                    batch_result.upper[i],
                    1e-10,
                    &format!("Donchian.upper streaming[{}]", i),
                );
                assert_float_eq(
                    output.lower,
                    batch_result.lower[i],
                    1e-10,
                    &format!("Donchian.lower streaming[{}]", i),
                );
            }
        }
    }
}

#[test]
fn test_donchian_invariant() {
    let path = golden_path("sample_ohlcv.json");
    let golden = match GoldenData::load(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test - could not load golden data: {}", e);
            return;
        }
    };

    let ohlcv = golden.to_ohlcv_series();
    let config = DonchianConfig::new(5);

    let dc = DonchianChannel::<f64>::new(config);
    let result = dc.calculate(&ohlcv).unwrap();

    for i in 0..result.middle.len() {
        if !result.middle[i].is_nan() {
            assert!(
                result.lower[i] <= result.middle[i],
                "lower ({}) should be <= middle ({}) at index {}",
                result.lower[i],
                result.middle[i],
                i
            );
            assert!(
                result.middle[i] <= result.upper[i],
                "middle ({}) should be <= upper ({}) at index {}",
                result.middle[i],
                result.upper[i],
                i
            );
        }
    }
}
