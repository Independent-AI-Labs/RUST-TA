//! Rate of Change (ROC) indicator.
//!
//! ROC measures the percentage change in price from n periods ago.

use ta_core::{
    error::{IndicatorError, Result},
    num::TaFloat,
    ohlcv::{Bar, OhlcvSeries},
    series::Series,
    traits::{Indicator, StreamingIndicator},
    window::RingBuffer,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for ROC.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RocConfig {
    /// Lookback period (default: 12).
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for RocConfig {
    fn default() -> Self {
        Self {
            window: 12,
            fillna: false,
        }
    }
}

impl RocConfig {
    /// Create a new configuration.
    pub fn new(window: usize) -> Self {
        Self {
            window,
            fillna: false,
        }
    }

    /// Set fillna option.
    pub fn with_fillna(mut self, fillna: bool) -> Self {
        self.fillna = fillna;
        self
    }
}

/// State for ROC.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct RocState<T: TaFloat> {
    /// Version tag.
    pub version: u32,
    /// Price buffer.
    pub buffer: Vec<T>,
    /// Count of values seen.
    pub count: usize,
}

/// Rate of Change indicator.
///
/// # Formula
///
/// ROC = 100 * (Close - Close_n) / Close_n
///
/// Where Close_n is the close price n periods ago.
#[derive(Debug, Clone)]
pub struct Roc<T: TaFloat> {
    config: RocConfig,
    buffer: RingBuffer<T>,
    count: usize,
    current_value: Option<T>,
}

impl<T: TaFloat> Roc<T> {
    /// Get current value.
    pub fn value(&self) -> Option<T> {
        self.current_value
    }
}

impl<T: TaFloat> Indicator<T> for Roc<T> {
    type Output = Series<T>;
    type Config = RocConfig;
    type State = RocState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            buffer: RingBuffer::new(config.window + 1),
            config,
            count: 0,
            current_value: None,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.window + 1
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();
        let close = data.close();
        let window = self.config.window;

        if window == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        let mut result = Series::with_capacity(len);

        for i in 0..len {
            if i < window {
                if self.config.fillna {
                    result.push(T::ZERO);
                } else {
                    result.push(T::NAN);
                }
            } else {
                let prev = close[i - window];
                let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

                let roc = if prev.abs() < epsilon {
                    T::NAN
                } else {
                    T::HUNDRED * (close[i] - prev) / prev
                };

                if roc.is_nan() && self.config.fillna {
                    result.push(T::ZERO);
                } else {
                    result.push(roc);
                }
            }
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        RocState {
            version: 1,
            buffer: self.buffer.iter().copied().collect(),
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

        self.buffer = RingBuffer::new(self.config.window + 1);
        for v in state.buffer {
            self.buffer.push(v);
        }
        self.count = state.count;
        Ok(())
    }

    fn reset(&mut self) {
        self.buffer = RingBuffer::new(self.config.window + 1);
        self.count = 0;
        self.current_value = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Roc<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        self.count += 1;
        self.buffer.push(bar.close);

        // Check if buffer is full AFTER pushing
        // When full, oldest() gives the price from `window` periods ago
        if self.buffer.is_full() {
            if let Some(&prev) = self.buffer.oldest() {
                let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);
                let roc = if prev.abs() < epsilon {
                    T::NAN
                } else {
                    T::HUNDRED * (bar.close - prev) / prev
                };

                self.current_value = Some(roc);
                return Ok(Some(roc));
            }
        }

        Ok(None)
    }

    fn current(&self) -> Option<T> {
        self.current_value
    }

    fn is_ready(&self) -> bool {
        self.current_value.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_ohlcv(closes: &[f64]) -> OhlcvSeries<f64> {
        let mut ohlcv = OhlcvSeries::new();
        for &close in closes {
            ohlcv.push(Bar::new(close, close, close, close, 1000.0));
        }
        ohlcv
    }

    #[test]
    fn test_roc_default_config() {
        let config = RocConfig::default();
        assert_eq!(config.window, 12);
    }

    #[test]
    fn test_roc_calculate() {
        let config = RocConfig::new(3);
        let roc = Roc::<f64>::new(config);

        let closes = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let ohlcv = create_test_ohlcv(&closes);
        let result = roc.calculate(&ohlcv).unwrap();

        // First 3 should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());

        // Fourth: (106 - 100) / 100 * 100 = 6%
        assert_relative_eq!(result[3], 6.0, epsilon = 1e-8);

        // Fifth: (108 - 102) / 102 * 100 ≈ 5.88%
        assert_relative_eq!(result[4], 100.0 * (108.0 - 102.0) / 102.0, epsilon = 1e-8);
    }

    #[test]
    fn test_roc_streaming() {
        let config = RocConfig::new(3);
        let mut roc = Roc::<f64>::new(config);

        let closes = vec![100.0, 102.0, 104.0, 106.0, 108.0];

        let mut results = Vec::new();
        for &close in &closes {
            let bar = Bar::new(close, close, close, close, 1000.0);
            results.push(roc.update(&bar).unwrap());
        }

        assert!(results[0].is_none());
        assert!(results[1].is_none());
        assert!(results[2].is_none());
        assert_relative_eq!(results[3].unwrap(), 6.0, epsilon = 1e-8);
    }

    #[test]
    fn test_roc_constant_price() {
        let config = RocConfig::new(3);
        let mut roc = Roc::<f64>::new(config);

        for _ in 0..10 {
            let bar = Bar::new(100.0, 100.0, 100.0, 100.0, 1000.0);
            if let Ok(Some(value)) = roc.update(&bar) {
                assert_relative_eq!(value, 0.0, epsilon = 1e-8);
            }
        }
    }
}
