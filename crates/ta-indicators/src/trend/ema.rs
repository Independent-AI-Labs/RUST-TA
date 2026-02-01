//! Exponential Moving Average (EMA) indicator.
//!
//! The EMA gives more weight to recent prices using an exponential
//! smoothing factor.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use ta_core::{
    error::{IndicatorError, Result},
    num::TaFloat,
    ohlcv::{Bar, OhlcvSeries},
    series::Series,
    traits::{Indicator, StreamingIndicator},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the EMA indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EmaConfig {
    /// The window size (period) for the EMA.
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self {
            window: 14,
            fillna: false,
        }
    }
}

impl EmaConfig {
    /// Create a new EMA configuration with the given window.
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

/// State for the EMA indicator (for serialization/deserialization).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct EmaState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// Current EMA value.
    pub ema_value: T,
    /// Running sum for SMA initialization.
    pub sma_sum: T,
    /// Number of values seen.
    pub count: usize,
    /// Whether the EMA has been initialized with SMA.
    pub initialized: bool,
}

impl<T: TaFloat> Default for EmaState<T> {
    fn default() -> Self {
        Self {
            version: 1,
            ema_value: T::ZERO,
            sma_sum: T::ZERO,
            count: 0,
            initialized: false,
        }
    }
}

/// Exponential Moving Average indicator.
///
/// The EMA is calculated using an exponential smoothing factor (alpha = 2/(n+1)).
/// The first n periods use a simple moving average for initialization.
///
/// # Formula
///
/// alpha = 2 / (window + 1)
/// EMA_t = alpha * Price_t + (1 - alpha) * EMA_{t-1}
///
/// For the first window periods, SMA is used to initialize the EMA.
#[derive(Debug, Clone)]
pub struct Ema<T: TaFloat> {
    config: EmaConfig,
    /// Smoothing factor: 2 / (window + 1)
    alpha: T,
    /// Current EMA value
    ema_value: T,
    /// Running sum for SMA initialization
    sma_sum: T,
    /// Number of values seen
    count: usize,
    /// Whether initialized with SMA
    initialized: bool,
}

impl<T: TaFloat> Ema<T> {
    /// Returns the smoothing factor (alpha).
    pub fn alpha(&self) -> T {
        self.alpha
    }

    /// Returns the current EMA value if initialized.
    pub fn value(&self) -> Option<T> {
        if self.initialized {
            Some(self.ema_value)
        } else {
            None
        }
    }

    /// Returns the number of values seen.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Compute EMA from raw alpha value (for internal use).
    pub(crate) fn new_with_alpha(alpha: T, fillna: bool) -> Self {
        Self {
            config: EmaConfig {
                window: 0, // Not used when alpha is specified directly
                fillna,
            },
            alpha,
            ema_value: T::ZERO,
            sma_sum: T::ZERO,
            count: 0,
            initialized: false,
        }
    }

    /// Update with a raw value (not from a bar).
    pub fn update_value(&mut self, value: T) -> Option<T> {
        self.count += 1;

        if !self.initialized {
            self.sma_sum = self.sma_sum + value;

            if self.count >= self.config.window {
                // Initialize with SMA
                self.ema_value = self.sma_sum / <T as TaFloat>::from_usize(self.config.window);
                self.initialized = true;
                return Some(self.ema_value);
            }

            if self.config.fillna {
                return Some(T::ZERO);
            }
            return None;
        }

        // EMA update: EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}
        self.ema_value = self.alpha * value + (T::ONE - self.alpha) * self.ema_value;
        Some(self.ema_value)
    }
}

impl<T: TaFloat> Indicator<T> for Ema<T> {
    type Output = Series<T>;
    type Config = EmaConfig;
    type State = EmaState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        let alpha = T::TWO / <T as TaFloat>::from_usize(config.window + 1);
        Self {
            config,
            alpha,
            ema_value: T::ZERO,
            sma_sum: T::ZERO,
            count: 0,
            initialized: false,
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

        let alpha = T::TWO / <T as TaFloat>::from_usize(window + 1);
        let mut result = Series::with_capacity(len);

        // Calculate SMA for initialization
        let mut sma_sum = T::ZERO;
        let mut ema = T::ZERO;
        let mut initialized = false;

        for i in 0..len {
            let value = close[i];

            if !initialized {
                sma_sum = sma_sum + value;

                if i + 1 >= window {
                    // Initialize with SMA
                    ema = sma_sum / <T as TaFloat>::from_usize(window);
                    initialized = true;
                    result.push(ema);
                } else if self.config.fillna {
                    result.push(T::ZERO);
                } else {
                    result.push(T::NAN);
                }
            } else {
                // EMA update
                ema = alpha * value + (T::ONE - alpha) * ema;
                result.push(ema);
            }
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        EmaState {
            version: 1,
            ema_value: self.ema_value,
            sma_sum: self.sma_sum,
            count: self.count,
            initialized: self.initialized,
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

        self.ema_value = state.ema_value;
        self.sma_sum = state.sma_sum;
        self.count = state.count;
        self.initialized = state.initialized;

        Ok(())
    }

    fn reset(&mut self) {
        self.ema_value = T::ZERO;
        self.sma_sum = T::ZERO;
        self.count = 0;
        self.initialized = false;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Ema<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        Ok(self.update_value(bar.close))
    }

    fn current(&self) -> Option<T> {
        self.value()
    }

    fn is_ready(&self) -> bool {
        self.initialized
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
    fn test_ema_default_config() {
        let config = EmaConfig::default();
        assert_eq!(config.window, 14);
        assert!(!config.fillna);
    }

    #[test]
    fn test_ema_alpha_calculation() {
        let config = EmaConfig::new(14);
        let ema = Ema::<f64>::new(config);

        // alpha = 2 / (14 + 1) = 2/15 = 0.1333...
        assert_relative_eq!(ema.alpha(), 2.0 / 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ema_calculate() {
        let config = EmaConfig::new(3);
        let ema = Ema::<f64>::new(config);

        let closes = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ohlcv = create_test_ohlcv(&closes);
        let result = ema.calculate(&ohlcv).unwrap();

        // First 2 should be NaN (window-1)
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // Third value is SMA initialization: (1+2+3)/3 = 2.0
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10);

        // Fourth value: alpha = 2/4 = 0.5
        // EMA = 0.5 * 4 + 0.5 * 2 = 3.0
        assert_relative_eq!(result[3], 3.0, epsilon = 1e-10);

        // Fifth value: EMA = 0.5 * 5 + 0.5 * 3 = 4.0
        assert_relative_eq!(result[4], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ema_with_fillna() {
        let config = EmaConfig::new(3).with_fillna(true);
        let ema = Ema::<f64>::new(config);

        let closes = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ohlcv = create_test_ohlcv(&closes);
        let result = ema.calculate(&ohlcv).unwrap();

        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ema_streaming() {
        let config = EmaConfig::new(3);
        let mut ema = Ema::<f64>::new(config);

        let result1 = ema.update(&Bar::new(1.0, 1.0, 1.0, 1.0, 1000.0)).unwrap();
        assert!(result1.is_none());

        let result2 = ema.update(&Bar::new(2.0, 2.0, 2.0, 2.0, 1000.0)).unwrap();
        assert!(result2.is_none());

        let result3 = ema.update(&Bar::new(3.0, 3.0, 3.0, 3.0, 1000.0)).unwrap();
        assert_relative_eq!(result3.unwrap(), 2.0, epsilon = 1e-10); // SMA init

        let result4 = ema.update(&Bar::new(4.0, 4.0, 4.0, 4.0, 1000.0)).unwrap();
        assert_relative_eq!(result4.unwrap(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ema_streaming_equals_batch() {
        let config = EmaConfig::new(5);
        let closes = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let ohlcv = create_test_ohlcv(&closes);

        // Batch calculation
        let batch_ema = Ema::<f64>::new(config.clone());
        let batch_result = batch_ema.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_ema = Ema::<f64>::new(config);
        for (i, &close) in closes.iter().enumerate() {
            let bar = Bar::new(close, close, close, close, 1000.0);
            let streaming_result = streaming_ema.update(&bar).unwrap();

            if let Some(val) = streaming_result {
                assert_relative_eq!(val, batch_result[i], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_ema_converges_to_constant() {
        let config = EmaConfig::new(10);
        let mut ema = Ema::<f64>::new(config);

        // Feed constant value 100.0
        for _ in 0..100 {
            let _ = ema.update(&Bar::new(100.0, 100.0, 100.0, 100.0, 1000.0));
        }

        // Should converge to 100.0
        assert_relative_eq!(ema.value().unwrap(), 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ema_invalid_window() {
        let config = EmaConfig::new(0);
        let ema = Ema::<f64>::new(config);

        let ohlcv = create_test_ohlcv(&[1.0, 2.0, 3.0]);
        let result = ema.calculate(&ohlcv);

        assert!(result.is_err());
    }

    #[test]
    fn test_ema_state_roundtrip() {
        let config = EmaConfig::new(3);
        let mut ema1 = Ema::<f64>::new(config.clone());

        // Feed some data
        ema1.update(&Bar::new(10.0, 10.0, 10.0, 10.0, 1000.0)).unwrap();
        ema1.update(&Bar::new(20.0, 20.0, 20.0, 20.0, 1000.0)).unwrap();
        ema1.update(&Bar::new(30.0, 30.0, 30.0, 30.0, 1000.0)).unwrap();

        // Get state
        let state = ema1.get_state();

        // Create new indicator and restore state
        let mut ema2 = Ema::<f64>::new(config);
        ema2.set_state(state).unwrap();

        // Both should produce the same result
        let result1 = ema1.update(&Bar::new(40.0, 40.0, 40.0, 40.0, 1000.0)).unwrap();
        let result2 = ema2.update(&Bar::new(40.0, 40.0, 40.0, 40.0, 1000.0)).unwrap();

        assert_relative_eq!(result1.unwrap(), result2.unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_ema_reset() {
        let config = EmaConfig::new(3);
        let mut ema = Ema::<f64>::new(config);

        // Feed some data
        ema.update(&Bar::new(10.0, 10.0, 10.0, 10.0, 1000.0)).unwrap();
        ema.update(&Bar::new(20.0, 20.0, 20.0, 20.0, 1000.0)).unwrap();
        ema.update(&Bar::new(30.0, 30.0, 30.0, 30.0, 1000.0)).unwrap();

        assert!(ema.is_ready());

        // Reset
        ema.reset();

        assert!(!ema.is_ready());
        assert_eq!(ema.count(), 0);
    }
}
