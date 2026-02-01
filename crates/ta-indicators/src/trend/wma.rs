//! Weighted Moving Average (WMA) indicator.
//!
//! WMA assigns more weight to recent prices using linear weights.

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

/// Configuration for WMA.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WmaConfig {
    /// Window size (default: 14).
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for WmaConfig {
    fn default() -> Self {
        Self {
            window: 14,
            fillna: false,
        }
    }
}

impl WmaConfig {
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

/// State for WMA.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct WmaState<T: TaFloat> {
    /// Version tag.
    pub version: u32,
    /// Buffer values.
    pub buffer: Vec<T>,
    /// Count of values seen.
    pub count: usize,
}

/// Weighted Moving Average indicator.
///
/// # Formula
///
/// WMA = (n*P_n + (n-1)*P_{n-1} + ... + 1*P_1) / (n + (n-1) + ... + 1)
///     = Sum(i * P_i) / Sum(i) for i = 1 to n
///     = Sum(i * P_i) / (n * (n+1) / 2)
#[derive(Debug, Clone)]
pub struct Wma<T: TaFloat> {
    config: WmaConfig,
    buffer: RingBuffer<T>,
    /// Denominator: n * (n+1) / 2
    weight_sum: T,
    count: usize,
    current_value: Option<T>,
}

impl<T: TaFloat> Wma<T> {
    /// Get current value.
    pub fn value(&self) -> Option<T> {
        self.current_value
    }
}

impl<T: TaFloat> Indicator<T> for Wma<T> {
    type Output = Series<T>;
    type Config = WmaConfig;
    type State = WmaState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        let n = config.window;
        let weight_sum = <T as TaFloat>::from_usize(n * (n + 1) / 2);

        Self {
            buffer: RingBuffer::new(config.window),
            config,
            weight_sum,
            count: 0,
            current_value: None,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.window
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();
        let close = data.close();
        let window = self.config.window;

        if window == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        let mut result = Series::with_capacity(len);
        let weight_sum = <T as TaFloat>::from_usize(window * (window + 1) / 2);

        for i in 0..len {
            if i + 1 < window {
                if self.config.fillna {
                    result.push(T::ZERO);
                } else {
                    result.push(T::NAN);
                }
            } else {
                let mut weighted_sum = T::ZERO;
                for j in 0..window {
                    let weight = <T as TaFloat>::from_usize(j + 1);
                    // Reorder to avoid underflow: (i + 1 + j) - window is safe when i + 1 >= window
                    weighted_sum = weighted_sum + weight * close[i + 1 + j - window];
                }
                result.push(weighted_sum / weight_sum);
            }
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        WmaState {
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

        self.buffer = RingBuffer::new(self.config.window);
        for v in state.buffer {
            self.buffer.push(v);
        }
        self.count = state.count;
        Ok(())
    }

    fn reset(&mut self) {
        self.buffer = RingBuffer::new(self.config.window);
        self.count = 0;
        self.current_value = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Wma<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        self.count += 1;
        self.buffer.push(bar.close);

        if !self.buffer.is_full() {
            return Ok(None);
        }

        let window = self.config.window;
        let mut weighted_sum = T::ZERO;

        for (j, &val) in self.buffer.iter().enumerate() {
            let weight = <T as TaFloat>::from_usize(j + 1);
            weighted_sum = weighted_sum + weight * val;
        }

        let wma = weighted_sum / self.weight_sum;
        self.current_value = Some(wma);
        Ok(Some(wma))
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
    fn test_wma_default_config() {
        let config = WmaConfig::default();
        assert_eq!(config.window, 14);
    }

    #[test]
    fn test_wma_calculate() {
        let config = WmaConfig::new(3);
        let wma = Wma::<f64>::new(config);

        let closes = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ohlcv = create_test_ohlcv(&closes);
        let result = wma.calculate(&ohlcv).unwrap();

        // First 2 should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // Third: (1*1 + 2*2 + 3*3) / 6 = 14/6 ≈ 2.333
        assert_relative_eq!(result[2], 14.0 / 6.0, epsilon = 1e-8);

        // Fourth: (1*2 + 2*3 + 3*4) / 6 = 20/6 ≈ 3.333
        assert_relative_eq!(result[3], 20.0 / 6.0, epsilon = 1e-8);
    }

    #[test]
    fn test_wma_constant_price() {
        let config = WmaConfig::new(5);
        let mut wma = Wma::<f64>::new(config);

        for _ in 0..10 {
            let bar = Bar::new(100.0, 100.0, 100.0, 100.0, 1000.0);
            if let Ok(Some(value)) = wma.update(&bar) {
                assert_relative_eq!(value, 100.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_wma_streaming_equals_batch() {
        let config = WmaConfig::new(3);
        let closes = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let ohlcv = create_test_ohlcv(&closes);

        let batch_wma = Wma::<f64>::new(config.clone());
        let batch_result = batch_wma.calculate(&ohlcv).unwrap();

        let mut streaming_wma = Wma::<f64>::new(config);
        for (i, &close) in closes.iter().enumerate() {
            let bar = Bar::new(close, close, close, close, 1000.0);
            if let Ok(Some(val)) = streaming_wma.update(&bar) {
                assert_relative_eq!(val, batch_result[i], epsilon = 1e-10);
            }
        }
    }
}
