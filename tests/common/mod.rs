//! Common test utilities for rust-ta.
//!
//! This module provides utilities for testing indicators including
//! float comparison, golden data loading, and synthetic data generation.

#![allow(dead_code)]

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Tiered float comparison with tolerance levels.
///
/// Uses the following tolerance levels based on magnitude:
/// - For values near zero (|expected| < 1e-10): use absolute tolerance
/// - For small values (|expected| < 1): use looser relative tolerance
/// - For larger values: use standard relative tolerance
pub fn assert_float_eq(actual: f64, expected: f64, epsilon: f64, context: &str) {
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

/// Assert two series are equal with tolerance.
pub fn assert_series_eq(actual: &[f64], expected: &[f64], epsilon: f64, name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: Length mismatch: {} vs {}",
        name,
        actual.len(),
        expected.len()
    );

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let context = format!("{}[{}]", name, i);
        assert_float_eq(a, e, epsilon, &context);
    }
}

/// Golden data entry with input OHLCV and expected outputs.
#[derive(Debug, Clone)]
pub struct GoldenData {
    /// Input close prices.
    pub close: Vec<f64>,
    /// Input high prices.
    pub high: Vec<f64>,
    /// Input low prices.
    pub low: Vec<f64>,
    /// Input open prices.
    pub open: Vec<f64>,
    /// Input volumes.
    pub volume: Vec<f64>,
    /// Expected indicator outputs (keyed by indicator name).
    pub expected: std::collections::HashMap<String, Vec<f64>>,
}

impl GoldenData {
    /// Create new empty golden data.
    pub fn new() -> Self {
        Self {
            close: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            open: Vec::new(),
            volume: Vec::new(),
            expected: std::collections::HashMap::new(),
        }
    }

    /// Load golden data from JSON file.
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data: serde_json::Value = serde_json::from_reader(reader)?;

        let mut golden = GoldenData::new();

        if let Some(ohlcv) = data.get("ohlcv") {
            golden.close = parse_array(ohlcv.get("close"))?;
            golden.high = parse_array(ohlcv.get("high"))?;
            golden.low = parse_array(ohlcv.get("low"))?;
            golden.open = parse_array(ohlcv.get("open"))?;
            golden.volume = parse_array(ohlcv.get("volume"))?;
        }

        if let Some(expected) = data.get("expected").and_then(|e| e.as_object()) {
            for (key, value) in expected {
                golden.expected.insert(key.clone(), parse_array(Some(value))?);
            }
        }

        Ok(golden)
    }
}

impl Default for GoldenData {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_array(value: Option<&serde_json::Value>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    match value {
        Some(serde_json::Value::Array(arr)) => {
            arr.iter()
                .map(|v| {
                    if v.is_null() {
                        Ok(f64::NAN)
                    } else {
                        v.as_f64()
                            .ok_or_else(|| format!("Invalid number: {:?}", v).into())
                    }
                })
                .collect()
        }
        _ => Ok(Vec::new()),
    }
}

// ============================================================================
// Synthetic Data Generators
// ============================================================================

/// Generate constant price series.
pub fn generate_constant(value: f64, len: usize) -> Vec<f64> {
    vec![value; len]
}

/// Generate linear price series.
pub fn generate_linear(start: f64, step: f64, len: usize) -> Vec<f64> {
    (0..len).map(|i| start + step * i as f64).collect()
}

/// Generate sine wave price series.
pub fn generate_sine(center: f64, amplitude: f64, period: usize, len: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    (0..len)
        .map(|i| center + amplitude * (2.0 * PI * i as f64 / period as f64).sin())
        .collect()
}

/// Generate random walk price series with deterministic seed.
pub fn generate_random_walk(start: f64, volatility: f64, len: usize, seed: u64) -> Vec<f64> {
    // Simple LCG for deterministic random numbers
    let mut rng_state = seed;
    let lcg_next = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Convert to [-1, 1]
        (*state as f64 / u64::MAX as f64) * 2.0 - 1.0
    };

    let mut prices = Vec::with_capacity(len);
    prices.push(start);

    for _ in 1..len {
        let change = lcg_next(&mut rng_state) * volatility;
        let last = *prices.last().unwrap();
        prices.push((last + change).max(0.01)); // Ensure positive price
    }

    prices
}

/// Generate OHLCV data from close prices.
pub fn generate_ohlcv_from_close(close: &[f64], range_pct: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut open = Vec::with_capacity(len);
    let mut high = Vec::with_capacity(len);
    let mut low = Vec::with_capacity(len);
    let mut volume = Vec::with_capacity(len);

    for (i, &c) in close.iter().enumerate() {
        let range = c * range_pct;
        let o = if i == 0 { c } else { close[i - 1] };
        let h = c.max(o) + range * 0.5;
        let l = c.min(o) - range * 0.5;
        let v = 1000.0 + (i as f64 * 100.0);

        open.push(o);
        high.push(h);
        low.push(l);
        volume.push(v);
    }

    (open, high, low, close.to_vec(), volume)
}

// ============================================================================
// Test Macros
// ============================================================================

/// Macro to create a test for streaming vs batch equivalence.
#[macro_export]
macro_rules! test_streaming_batch_equivalence {
    ($indicator:ty, $config:expr, $data:expr) => {{
        use ta_core::traits::{Indicator, StreamingIndicator};
        use ta_core::ohlcv::Bar;

        let config = $config;
        let (open, high, low, close, volume) = $data;

        // Create OHLCV series for batch
        let mut ohlcv = ta_core::ohlcv::OhlcvSeries::new();
        for i in 0..close.len() {
            ohlcv.push(Bar::new(open[i], high[i], low[i], close[i], volume[i]));
        }

        // Batch calculation
        let batch_indicator = <$indicator>::new(config.clone());
        let batch_result = batch_indicator.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_indicator = <$indicator>::new(config);
        for (i, (&o, &h, &l, &c, &v)) in open.iter()
            .zip(high.iter())
            .zip(low.iter())
            .zip(close.iter())
            .zip(volume.iter())
            .map(|((((o, h), l), c), v)| (o, h, l, c, v))
            .enumerate()
        {
            let bar = Bar::new(o, h, l, c, v);
            let streaming_result = streaming_indicator.update(&bar).unwrap();

            if let Some(val) = streaming_result {
                let batch_val = batch_result[i];
                if !batch_val.is_nan() {
                    $crate::common::assert_float_eq(
                        val,
                        batch_val,
                        1e-10,
                        &format!("streaming[{}]", i),
                    );
                }
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_float_eq_normal() {
        assert_float_eq(1.0, 1.0, 1e-10, "test");
        assert_float_eq(100.0, 100.0000001, 1e-6, "test");
    }

    #[test]
    fn test_assert_float_eq_nan() {
        assert_float_eq(f64::NAN, f64::NAN, 1e-10, "test");
    }

    #[test]
    #[should_panic]
    fn test_assert_float_eq_nan_mismatch() {
        assert_float_eq(1.0, f64::NAN, 1e-10, "test");
    }

    #[test]
    fn test_generate_constant() {
        let data = generate_constant(42.0, 5);
        assert_eq!(data, vec![42.0, 42.0, 42.0, 42.0, 42.0]);
    }

    #[test]
    fn test_generate_linear() {
        let data = generate_linear(100.0, 1.0, 5);
        assert_eq!(data, vec![100.0, 101.0, 102.0, 103.0, 104.0]);
    }

    #[test]
    fn test_generate_random_walk_deterministic() {
        let data1 = generate_random_walk(100.0, 1.0, 10, 12345);
        let data2 = generate_random_walk(100.0, 1.0, 10, 12345);
        assert_eq!(data1, data2);
    }
}
