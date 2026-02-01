//! Simple Moving Average (SMA) indicator.
//!
//! The SMA is the unweighted mean of the previous n data points.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

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

/// Configuration for the SMA indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SmaConfig {
    /// The window size for the moving average.
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for SmaConfig {
    fn default() -> Self {
        Self {
            window: 14,
            fillna: false,
        }
    }
}

impl SmaConfig {
    /// Create a new SMA configuration with the given window.
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

/// State for the SMA indicator (for serialization/deserialization).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct SmaState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// The window buffer values.
    pub buffer: Vec<T>,
    /// Running sum of values in the buffer.
    pub sum: T,
    /// Number of values seen.
    pub count: usize,
}

impl<T: TaFloat> Default for SmaState<T> {
    fn default() -> Self {
        Self {
            version: 1,
            buffer: Vec::new(),
            sum: T::ZERO,
            count: 0,
        }
    }
}

/// Simple Moving Average indicator.
///
/// The SMA is calculated as the arithmetic mean of the previous n closing prices.
///
/// # Formula
///
/// SMA = (P1 + P2 + ... + Pn) / n
///
/// where Pn is the price at period n.
#[derive(Debug, Clone)]
pub struct Sma<T: TaFloat> {
    config: SmaConfig,
    buffer: RingBuffer<T>,
    sum: T,
    count: usize,
}

impl<T: TaFloat> Sma<T> {
    /// Returns the current sum of values in the buffer.
    pub fn sum(&self) -> T {
        self.sum
    }

    /// Returns the number of values seen so far.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl<T: TaFloat> Indicator<T> for Sma<T> {
    type Output = Series<T>;
    type Config = SmaConfig;
    type State = SmaState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            // Use max(1) to prevent panic on window=0; calculate() will validate
            buffer: RingBuffer::new(config.window.max(1)),
            config,
            sum: T::ZERO,
            count: 0,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.window
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let close = data.close();
        let len = close.len();
        let window = self.config.window;

        if window == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        let mut result = Series::with_capacity(len);
        let mut sum = T::ZERO;

        for i in 0..len {
            let value = close[i];
            sum = sum + value;

            if i >= window {
                sum = sum - close[i - window];
            }

            if i + 1 >= window {
                let sma = sum / <T as TaFloat>::from_usize(window);
                result.push(sma);
            } else if self.config.fillna {
                result.push(T::ZERO);
            } else {
                result.push(T::NAN);
            }
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        SmaState {
            version: 1,
            buffer: self.buffer.iter().copied().collect(),
            sum: self.sum,
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
        for value in state.buffer {
            self.buffer.push(value);
        }
        self.sum = state.sum;
        self.count = state.count;

        Ok(())
    }

    fn reset(&mut self) {
        self.buffer = RingBuffer::new(self.config.window);
        self.sum = T::ZERO;
        self.count = 0;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Sma<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        let value = bar.close;
        self.count += 1;

        // If buffer is full, subtract the oldest value
        if self.buffer.is_full() {
            if let Some(&oldest) = self.buffer.oldest() {
                self.sum = self.sum - oldest;
            }
        }

        // Add the new value
        self.buffer.push(value);
        self.sum = self.sum + value;

        // Return SMA if we have enough data
        if self.buffer.is_full() {
            let sma = self.sum / <T as TaFloat>::from_usize(self.config.window);
            Ok(Some(sma))
        } else if self.config.fillna {
            Ok(Some(T::ZERO))
        } else {
            Ok(None)
        }
    }

    fn current(&self) -> Option<T> {
        if self.buffer.is_full() {
            Some(self.sum / <T as TaFloat>::from_usize(self.config.window))
        } else {
            None
        }
    }

    fn is_ready(&self) -> bool {
        self.buffer.is_full()
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
    fn test_sma_default_config() {
        let config = SmaConfig::default();
        assert_eq!(config.window, 14);
        assert!(!config.fillna);
    }

    #[test]
    fn test_sma_calculate() {
        let config = SmaConfig::new(3);
        let sma = Sma::<f64>::new(config);

        let closes = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ohlcv = create_test_ohlcv(&closes);
        let result = sma.calculate(&ohlcv).unwrap();

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10); // (1+2+3)/3
        assert_relative_eq!(result[3], 3.0, epsilon = 1e-10); // (2+3+4)/3
        assert_relative_eq!(result[4], 4.0, epsilon = 1e-10); // (3+4+5)/3
    }

    #[test]
    fn test_sma_with_fillna() {
        let config = SmaConfig::new(3).with_fillna(true);
        let sma = Sma::<f64>::new(config);

        let closes = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ohlcv = create_test_ohlcv(&closes);
        let result = sma.calculate(&ohlcv).unwrap();

        assert_eq!(result.len(), 5);
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sma_streaming() {
        let config = SmaConfig::new(3);
        let mut sma = Sma::<f64>::new(config);

        let result1 = sma.update(&Bar::new(1.0, 1.0, 1.0, 1.0, 1000.0)).unwrap();
        assert!(result1.is_none());

        let result2 = sma.update(&Bar::new(2.0, 2.0, 2.0, 2.0, 1000.0)).unwrap();
        assert!(result2.is_none());

        let result3 = sma.update(&Bar::new(3.0, 3.0, 3.0, 3.0, 1000.0)).unwrap();
        assert_relative_eq!(result3.unwrap(), 2.0, epsilon = 1e-10);

        let result4 = sma.update(&Bar::new(4.0, 4.0, 4.0, 4.0, 1000.0)).unwrap();
        assert_relative_eq!(result4.unwrap(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sma_streaming_equals_batch() {
        let config = SmaConfig::new(5);
        let closes = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let ohlcv = create_test_ohlcv(&closes);

        // Batch calculation
        let batch_sma = Sma::<f64>::new(config.clone());
        let batch_result = batch_sma.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_sma = Sma::<f64>::new(config);
        for (i, &close) in closes.iter().enumerate() {
            let bar = Bar::new(close, close, close, close, 1000.0);
            let streaming_result = streaming_sma.update(&bar).unwrap();

            if let Some(val) = streaming_result {
                assert_relative_eq!(val, batch_result[i], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sma_constant_input() {
        let config = SmaConfig::new(5);
        let mut sma = Sma::<f64>::new(config);

        // SMA of constant should equal the constant
        for _ in 0..10 {
            let result = sma.update(&Bar::new(42.0, 42.0, 42.0, 42.0, 1000.0)).unwrap();
            if let Some(val) = result {
                assert_relative_eq!(val, 42.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sma_invalid_window() {
        let config = SmaConfig::new(0);
        let sma = Sma::<f64>::new(config);

        let ohlcv = create_test_ohlcv(&[1.0, 2.0, 3.0]);
        let result = sma.calculate(&ohlcv);

        assert!(result.is_err());
    }

    #[test]
    fn test_sma_state_roundtrip() {
        let config = SmaConfig::new(3);
        let mut sma1 = Sma::<f64>::new(config.clone());

        // Feed some data
        sma1.update(&Bar::new(10.0, 10.0, 10.0, 10.0, 1000.0)).unwrap();
        sma1.update(&Bar::new(20.0, 20.0, 20.0, 20.0, 1000.0)).unwrap();
        sma1.update(&Bar::new(30.0, 30.0, 30.0, 30.0, 1000.0)).unwrap();

        // Get state
        let state = sma1.get_state();

        // Create new indicator and restore state
        let mut sma2 = Sma::<f64>::new(config);
        sma2.set_state(state).unwrap();

        // Both should produce the same result
        let result1 = sma1.update(&Bar::new(40.0, 40.0, 40.0, 40.0, 1000.0)).unwrap();
        let result2 = sma2.update(&Bar::new(40.0, 40.0, 40.0, 40.0, 1000.0)).unwrap();

        assert_relative_eq!(result1.unwrap(), result2.unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_sma_reset() {
        let config = SmaConfig::new(3);
        let mut sma = Sma::<f64>::new(config);

        // Feed some data
        sma.update(&Bar::new(10.0, 10.0, 10.0, 10.0, 1000.0)).unwrap();
        sma.update(&Bar::new(20.0, 20.0, 20.0, 20.0, 1000.0)).unwrap();
        sma.update(&Bar::new(30.0, 30.0, 30.0, 30.0, 1000.0)).unwrap();

        assert!(sma.is_ready());

        // Reset
        sma.reset();

        assert!(!sma.is_ready());
        assert_eq!(sma.count(), 0);
    }
}
