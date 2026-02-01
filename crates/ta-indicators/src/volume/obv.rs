//! On Balance Volume (OBV) indicator.
//!
//! OBV is a momentum indicator that uses volume flow to predict price changes.

use ta_core::{
    error::{IndicatorError, Result},
    num::TaFloat,
    ohlcv::{Bar, OhlcvSeries},
    series::Series,
    traits::{Indicator, StreamingIndicator},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the OBV indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ObvConfig {
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for ObvConfig {
    fn default() -> Self {
        Self { fillna: false }
    }
}

impl ObvConfig {
    /// Create a new OBV configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set fillna option.
    pub fn with_fillna(mut self, fillna: bool) -> Self {
        self.fillna = fillna;
        self
    }
}

/// State for the OBV indicator (for serialization/deserialization).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct ObvState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// Current OBV value.
    pub obv_value: T,
    /// Previous close price.
    pub prev_close: T,
    /// Number of values seen.
    pub count: usize,
}

impl<T: TaFloat> Default for ObvState<T> {
    fn default() -> Self {
        Self {
            version: 1,
            obv_value: T::ZERO,
            prev_close: T::NAN,
            count: 0,
        }
    }
}

/// On Balance Volume indicator.
///
/// OBV adds volume on up days and subtracts volume on down days.
///
/// # Formula
///
/// - If close > prev_close: OBV += volume
/// - If close < prev_close: OBV -= volume
/// - If close == prev_close: OBV unchanged
///
/// # Overflow Protection
///
/// OBV is protected against overflow. If |OBV| > 10^15, an error is returned.
#[derive(Debug, Clone)]
pub struct Obv<T: TaFloat> {
    config: ObvConfig,
    /// Current OBV value
    obv_value: T,
    /// Previous close price
    prev_close: T,
    /// Number of values seen
    count: usize,
}

impl<T: TaFloat> Obv<T> {
    /// Returns the current OBV value.
    pub fn value(&self) -> T {
        self.obv_value
    }

    /// Check for overflow protection.
    fn check_overflow(&self) -> Result<()> {
        let threshold = <T as TaFloat>::from_f64_lossy(1e15);
        if self.obv_value.abs() > threshold {
            return Err(IndicatorError::NumericError(
                "OBV overflow: |OBV| > 10^15".to_string(),
            ));
        }
        Ok(())
    }
}

impl<T: TaFloat> Indicator<T> for Obv<T> {
    type Output = Series<T>;
    type Config = ObvConfig;
    type State = ObvState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            config,
            obv_value: T::ZERO,
            prev_close: T::NAN,
            count: 0,
        }
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();
        let close = data.close();
        let volume = data.volume();

        let mut result = Series::with_capacity(len);

        if len == 0 {
            return Ok(result);
        }

        let threshold = <T as TaFloat>::from_f64_lossy(1e15);
        let mut obv = T::ZERO;

        // First bar: OBV = 0 (or volume, depending on convention)
        result.push(obv);

        for i in 1..len {
            let change = close[i] - close[i - 1];
            let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

            if change > epsilon {
                obv = obv + volume[i];
            } else if change < -epsilon {
                obv = obv - volume[i];
            }
            // If close unchanged, OBV stays the same

            // Check overflow
            if obv.abs() > threshold {
                return Err(IndicatorError::NumericError(
                    "OBV overflow: |OBV| > 10^15".to_string(),
                ));
            }

            result.push(obv);
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        ObvState {
            version: 1,
            obv_value: self.obv_value,
            prev_close: self.prev_close,
            count: self.count,
        }
    }

    fn set_state(&mut self, state: Self::State) -> Result<()> {
        if state.version != 1 {
            return Err(IndicatorError::StateError(
                ta_core::error::StateRestoreError::VersionMismatch {
                    expected: "1".to_string(),
                    actual: state.version.to_string(),
                },
            ));
        }

        self.obv_value = state.obv_value;
        self.prev_close = state.prev_close;
        self.count = state.count;

        Ok(())
    }

    fn reset(&mut self) {
        self.obv_value = T::ZERO;
        self.prev_close = T::NAN;
        self.count = 0;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Obv<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        self.count += 1;

        let close = bar.close;
        let volume = bar.volume;

        // First bar: just store close
        if self.prev_close.is_nan() {
            self.prev_close = close;
            return Ok(Some(self.obv_value));
        }

        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);
        let change = close - self.prev_close;

        if change > epsilon {
            self.obv_value = self.obv_value + volume;
        } else if change < -epsilon {
            self.obv_value = self.obv_value - volume;
        }

        self.prev_close = close;

        // Check overflow
        self.check_overflow()?;

        Ok(Some(self.obv_value))
    }

    fn current(&self) -> Option<T> {
        Some(self.obv_value)
    }

    fn is_ready(&self) -> bool {
        self.count >= 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_ohlcv(data: &[(f64, f64)]) -> OhlcvSeries<f64> {
        let mut ohlcv = OhlcvSeries::new();
        for &(close, volume) in data {
            ohlcv.push(Bar::new(close, close, close, close, volume));
        }
        ohlcv
    }

    #[test]
    fn test_obv_default_config() {
        let config = ObvConfig::default();
        assert!(!config.fillna);
    }

    #[test]
    fn test_obv_calculate() {
        let config = ObvConfig::new();
        let obv = Obv::<f64>::new(config);

        // Price goes: up, up, down, unchanged, up
        let data = vec![
            (100.0, 1000.0),
            (101.0, 1500.0), // up: +1500
            (102.0, 2000.0), // up: +2000
            (100.0, 1000.0), // down: -1000
            (100.0, 500.0),  // unchanged
            (101.0, 2500.0), // up: +2500
        ];
        let ohlcv = create_test_ohlcv(&data);
        let result = obv.calculate(&ohlcv).unwrap();

        assert_eq!(result.len(), 6);
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1500.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 3500.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 2500.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 2500.0, epsilon = 1e-10);
        assert_relative_eq!(result[5], 5000.0, epsilon = 1e-10);
    }

    #[test]
    fn test_obv_streaming() {
        let config = ObvConfig::new();
        let mut obv = Obv::<f64>::new(config);

        let data = vec![
            (100.0, 1000.0),
            (101.0, 1500.0),
            (102.0, 2000.0),
            (100.0, 1000.0),
        ];

        let mut results = Vec::new();
        for &(close, volume) in &data {
            let bar = Bar::new(close, close, close, close, volume);
            let result = obv.update(&bar).unwrap();
            results.push(result.unwrap());
        }

        assert_relative_eq!(results[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(results[1], 1500.0, epsilon = 1e-10);
        assert_relative_eq!(results[2], 3500.0, epsilon = 1e-10);
        assert_relative_eq!(results[3], 2500.0, epsilon = 1e-10);
    }

    #[test]
    fn test_obv_streaming_equals_batch() {
        let config = ObvConfig::new();
        let data = vec![
            (100.0, 1000.0),
            (101.0, 1500.0),
            (102.0, 2000.0),
            (100.0, 1000.0),
            (99.0, 500.0),
            (101.0, 3000.0),
        ];
        let ohlcv = create_test_ohlcv(&data);

        // Batch calculation
        let batch_obv = Obv::<f64>::new(config.clone());
        let batch_result = batch_obv.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_obv = Obv::<f64>::new(config);
        for (i, &(close, volume)) in data.iter().enumerate() {
            let bar = Bar::new(close, close, close, close, volume);
            let streaming_result = streaming_obv.update(&bar).unwrap().unwrap();
            assert_relative_eq!(streaming_result, batch_result[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_obv_all_up() {
        // When price only goes up, OBV should be sum of all volumes
        let config = ObvConfig::new();
        let mut obv = Obv::<f64>::new(config);

        let mut total_volume = 0.0;
        for i in 0..10 {
            let close = 100.0 + i as f64;
            let volume = 1000.0;
            let bar = Bar::new(close, close, close, close, volume);
            obv.update(&bar).unwrap();

            if i > 0 {
                total_volume += volume;
            }
        }

        assert_relative_eq!(obv.value(), total_volume, epsilon = 1e-10);
    }

    #[test]
    fn test_obv_all_down() {
        // When price only goes down, OBV should be negative
        let config = ObvConfig::new();
        let mut obv = Obv::<f64>::new(config);

        let mut total_volume = 0.0;
        for i in 0..10 {
            let close = 100.0 - i as f64;
            let volume = 1000.0;
            let bar = Bar::new(close, close, close, close, volume);
            obv.update(&bar).unwrap();

            if i > 0 {
                total_volume -= volume;
            }
        }

        assert_relative_eq!(obv.value(), total_volume, epsilon = 1e-10);
    }

    #[test]
    fn test_obv_unchanged_price() {
        // When price is unchanged, OBV should stay the same
        let config = ObvConfig::new();
        let mut obv = Obv::<f64>::new(config);

        // First bar
        obv.update(&Bar::new(100.0, 100.0, 100.0, 100.0, 1000.0))
            .unwrap();
        let obv1 = obv.value();

        // Same price, different volume
        obv.update(&Bar::new(100.0, 100.0, 100.0, 100.0, 5000.0))
            .unwrap();
        let obv2 = obv.value();

        assert_relative_eq!(obv1, obv2, epsilon = 1e-10);
    }

    #[test]
    fn test_obv_state_roundtrip() {
        let config = ObvConfig::new();
        let mut obv1 = Obv::<f64>::new(config.clone());

        // Feed some data
        let data = vec![
            (100.0, 1000.0),
            (101.0, 1500.0),
            (102.0, 2000.0),
        ];
        for &(close, volume) in &data {
            obv1.update(&Bar::new(close, close, close, close, volume))
                .unwrap();
        }

        // Get state
        let state = obv1.get_state();

        // Create new indicator and restore state
        let mut obv2 = Obv::<f64>::new(config);
        obv2.set_state(state).unwrap();

        // Both should produce the same result
        let next_bar = Bar::new(103.0, 103.0, 103.0, 103.0, 1000.0);
        let result1 = obv1.update(&next_bar).unwrap().unwrap();
        let result2 = obv2.update(&next_bar).unwrap().unwrap();

        assert_relative_eq!(result1, result2, epsilon = 1e-10);
    }

    #[test]
    fn test_obv_reset() {
        let config = ObvConfig::new();
        let mut obv = Obv::<f64>::new(config);

        // Feed some data
        obv.update(&Bar::new(100.0, 100.0, 100.0, 100.0, 1000.0))
            .unwrap();
        obv.update(&Bar::new(101.0, 101.0, 101.0, 101.0, 1500.0))
            .unwrap();

        assert!(obv.is_ready());
        assert!(obv.value() != 0.0);

        // Reset
        obv.reset();

        assert!(!obv.is_ready());
        assert_relative_eq!(obv.value(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_obv_min_periods() {
        let config = ObvConfig::new();
        let obv = Obv::<f64>::new(config);

        assert_eq!(obv.min_periods(), 1);
    }
}
