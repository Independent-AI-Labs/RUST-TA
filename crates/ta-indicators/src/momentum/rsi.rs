//! Relative Strength Index (RSI) indicator.
//!
//! RSI is a momentum oscillator that measures the speed and magnitude
//! of recent price changes to evaluate overbought or oversold conditions.

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

/// Configuration for the RSI indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RsiConfig {
    /// The lookback period (default: 14).
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for RsiConfig {
    fn default() -> Self {
        Self {
            window: 14,
            fillna: false,
        }
    }
}

impl RsiConfig {
    /// Create a new RSI configuration with the given window.
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

/// State for the RSI indicator (for serialization/deserialization).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct RsiState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// Average gain (Wilder-smoothed).
    pub avg_gain: T,
    /// Average loss (Wilder-smoothed).
    pub avg_loss: T,
    /// Previous close price.
    pub prev_close: T,
    /// Number of values seen.
    pub count: usize,
    /// Sum of gains during initialization.
    pub init_gain_sum: T,
    /// Sum of losses during initialization.
    pub init_loss_sum: T,
    /// Whether initialization is complete.
    pub initialized: bool,
}

impl<T: TaFloat> Default for RsiState<T> {
    fn default() -> Self {
        Self {
            version: 1,
            avg_gain: T::ZERO,
            avg_loss: T::ZERO,
            prev_close: T::NAN,
            count: 0,
            init_gain_sum: T::ZERO,
            init_loss_sum: T::ZERO,
            initialized: false,
        }
    }
}

/// Relative Strength Index indicator.
///
/// RSI measures the magnitude of recent price changes to evaluate
/// overbought or oversold conditions.
///
/// # Formula
///
/// RS = Average Gain / Average Loss
/// RSI = 100 - (100 / (1 + RS))
///
/// Wilder's smoothing is used: alpha = 1/n (not 2/(n+1))
///
/// # Edge Cases (per Section 8.6)
///
/// - AvgGain = 0 AND AvgLoss = 0 → RSI = 50 (neutral)
/// - AvgGain > 0 AND AvgLoss = 0 → RSI = 100 (strongly overbought)
/// - AvgGain = 0 AND AvgLoss > 0 → RSI = 0 (strongly oversold)
#[derive(Debug, Clone)]
pub struct Rsi<T: TaFloat> {
    config: RsiConfig,
    /// Average gain (Wilder-smoothed)
    avg_gain: T,
    /// Average loss (Wilder-smoothed)
    avg_loss: T,
    /// Previous close price
    prev_close: T,
    /// Number of values seen
    count: usize,
    /// Sum of gains during initialization
    init_gain_sum: T,
    /// Sum of losses during initialization
    init_loss_sum: T,
    /// Whether initialization is complete
    initialized: bool,
}

impl<T: TaFloat> Rsi<T> {
    /// Returns the current RSI value if ready.
    pub fn value(&self) -> Option<T> {
        if self.initialized {
            Some(Self::compute_rsi(self.avg_gain, self.avg_loss))
        } else {
            None
        }
    }

    /// Returns the average gain.
    pub fn avg_gain(&self) -> T {
        self.avg_gain
    }

    /// Returns the average loss.
    pub fn avg_loss(&self) -> T {
        self.avg_loss
    }

    /// Compute RSI from average gain and loss.
    fn compute_rsi(avg_gain: T, avg_loss: T) -> T {
        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

        // Edge cases per Section 8.6
        if avg_gain < epsilon && avg_loss < epsilon {
            // Both zero → neutral
            T::FIFTY
        } else if avg_loss < epsilon {
            // No losses → max RSI
            T::HUNDRED
        } else if avg_gain < epsilon {
            // No gains → min RSI
            T::ZERO
        } else {
            // Normal case
            let rs = avg_gain / avg_loss;
            T::HUNDRED - (T::HUNDRED / (T::ONE + rs))
        }
    }
}

impl<T: TaFloat> Indicator<T> for Rsi<T> {
    type Output = Series<T>;
    type Config = RsiConfig;
    type State = RsiState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            config,
            avg_gain: T::ZERO,
            avg_loss: T::ZERO,
            prev_close: T::NAN,
            count: 0,
            init_gain_sum: T::ZERO,
            init_loss_sum: T::ZERO,
            initialized: false,
        }
    }

    fn min_periods(&self) -> usize {
        // Need window periods to calculate first RSI (matches python-ta)
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

        if len == 0 {
            return Ok(result);
        }

        // Calculate price changes
        let mut gains = Vec::with_capacity(len);
        let mut losses = Vec::with_capacity(len);

        gains.push(T::ZERO); // First element has no change
        losses.push(T::ZERO);

        for i in 1..len {
            let change = close[i] - close[i - 1];
            if change > T::ZERO {
                gains.push(change);
                losses.push(T::ZERO);
            } else {
                gains.push(T::ZERO);
                losses.push(-change); // abs(change)
            }
        }

        // Calculate RSI using EWM (exponential weighted moving) from the start
        // This matches python-ta's implementation which uses:
        // ewm(com=window-1, adjust=False) which gives alpha = 1/window
        // Formula: y_t = alpha * x_t + (1 - alpha) * y_{t-1}, starting with y_0 = x_0
        let alpha = T::ONE / <T as TaFloat>::from_usize(window);
        let one_minus_alpha = T::ONE - alpha;

        let mut avg_gain = gains[0]; // Start with first value (which is 0)
        let mut avg_loss = losses[0];

        for i in 0..len {
            if i > 0 {
                // EWM update: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
                avg_gain = alpha * gains[i] + one_minus_alpha * avg_gain;
                avg_loss = alpha * losses[i] + one_minus_alpha * avg_loss;
            }

            if i < window - 1 {
                // Not enough data yet (min_periods = window)
                if self.config.fillna {
                    result.push(T::ZERO);
                } else {
                    result.push(T::NAN);
                }
            } else {
                result.push(Self::compute_rsi(avg_gain, avg_loss));
            }
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        RsiState {
            version: 1,
            avg_gain: self.avg_gain,
            avg_loss: self.avg_loss,
            prev_close: self.prev_close,
            count: self.count,
            init_gain_sum: self.init_gain_sum,
            init_loss_sum: self.init_loss_sum,
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

        self.avg_gain = state.avg_gain;
        self.avg_loss = state.avg_loss;
        self.prev_close = state.prev_close;
        self.count = state.count;
        self.init_gain_sum = state.init_gain_sum;
        self.init_loss_sum = state.init_loss_sum;
        self.initialized = state.initialized;

        Ok(())
    }

    fn reset(&mut self) {
        self.avg_gain = T::ZERO;
        self.avg_loss = T::ZERO;
        self.prev_close = T::NAN;
        self.count = 0;
        self.init_gain_sum = T::ZERO;
        self.init_loss_sum = T::ZERO;
        self.initialized = false;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Rsi<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        let close = bar.close;
        self.count += 1;

        // First bar: just store close
        if self.prev_close.is_nan() {
            self.prev_close = close;
            return Ok(None);
        }

        // Calculate gain/loss
        let change = close - self.prev_close;
        let (gain, loss) = if change > T::ZERO {
            (change, T::ZERO)
        } else {
            (T::ZERO, -change)
        };

        self.prev_close = close;

        let window = self.config.window;

        // EWM update: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
        // alpha = 1/window (Wilder's smoothing)
        let alpha = T::ONE / <T as TaFloat>::from_usize(window);
        let one_minus_alpha = T::ONE - alpha;

        self.avg_gain = alpha * gain + one_minus_alpha * self.avg_gain;
        self.avg_loss = alpha * loss + one_minus_alpha * self.avg_loss;

        // First RSI at count = window (matches python-ta's min_periods)
        if self.count >= window {
            self.initialized = true;
            Ok(Some(Self::compute_rsi(self.avg_gain, self.avg_loss)))
        } else {
            Ok(None)
        }
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
    fn test_rsi_default_config() {
        let config = RsiConfig::default();
        assert_eq!(config.window, 14);
        assert!(!config.fillna);
    }

    #[test]
    fn test_rsi_calculate() {
        let config = RsiConfig::new(5);
        let rsi = Rsi::<f64>::new(config);

        let closes = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 43.75, 44.0];
        let ohlcv = create_test_ohlcv(&closes);
        let result = rsi.calculate(&ohlcv).unwrap();

        assert_eq!(result.len(), 10);

        // First 4 should be NaN (matches python-ta: first RSI at index window-1)
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert!(result[3].is_nan());

        // 5th value (index 4) should be valid (RSI(5) first value at index 4)
        assert!(!result[4].is_nan());
    }

    #[test]
    fn test_rsi_bounds() {
        // RSI should always be in [0, 100]
        let config = RsiConfig::new(5);
        let mut rsi = Rsi::<f64>::new(config);

        let closes = vec![
            100.0, 110.0, 105.0, 115.0, 110.0, 120.0, 115.0, 125.0, 120.0, 130.0, 50.0, 45.0, 40.0,
            35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0,
        ];

        for &close in &closes {
            let bar = Bar::new(close, close, close, close, 1000.0);
            if let Ok(Some(value)) = rsi.update(&bar) {
                assert!(value >= 0.0, "RSI {} should be >= 0", value);
                assert!(value <= 100.0, "RSI {} should be <= 100", value);
            }
        }
    }

    #[test]
    fn test_rsi_all_gains() {
        // When price only goes up, RSI should approach 100
        let config = RsiConfig::new(5);
        let mut rsi = Rsi::<f64>::new(config);

        for i in 1..=20 {
            let close = i as f64 * 10.0;
            let bar = Bar::new(close, close, close, close, 1000.0);
            let result = rsi.update(&bar).unwrap();

            if let Some(value) = result {
                assert_relative_eq!(value, 100.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_rsi_all_losses() {
        // When price only goes down, RSI should be 0
        let config = RsiConfig::new(5);
        let mut rsi = Rsi::<f64>::new(config);

        for i in (1..=20).rev() {
            let close = i as f64 * 10.0;
            let bar = Bar::new(close, close, close, close, 1000.0);
            let result = rsi.update(&bar).unwrap();

            if let Some(value) = result {
                assert_relative_eq!(value, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_rsi_no_change() {
        // When price doesn't change, RSI should be 50
        let config = RsiConfig::new(5);
        let mut rsi = Rsi::<f64>::new(config);

        for _ in 0..20 {
            let bar = Bar::new(100.0, 100.0, 100.0, 100.0, 1000.0);
            let result = rsi.update(&bar).unwrap();

            if let Some(value) = result {
                assert_relative_eq!(value, 50.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_rsi_streaming_equals_batch() {
        let config = RsiConfig::new(5);
        let closes = vec![
            44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 43.75, 44.0, 44.5, 44.25, 45.0,
            45.5, 45.25,
        ];
        let ohlcv = create_test_ohlcv(&closes);

        // Batch calculation
        let batch_rsi = Rsi::<f64>::new(config.clone());
        let batch_result = batch_rsi.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_rsi = Rsi::<f64>::new(config);
        for (i, &close) in closes.iter().enumerate() {
            let bar = Bar::new(close, close, close, close, 1000.0);
            let streaming_result = streaming_rsi.update(&bar).unwrap();

            if let Some(val) = streaming_result {
                assert_relative_eq!(val, batch_result[i], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_rsi_invalid_window() {
        let config = RsiConfig::new(0);
        let rsi = Rsi::<f64>::new(config);

        let ohlcv = create_test_ohlcv(&[1.0, 2.0, 3.0]);
        let result = rsi.calculate(&ohlcv);

        assert!(result.is_err());
    }

    #[test]
    fn test_rsi_state_roundtrip() {
        let config = RsiConfig::new(5);
        let mut rsi1 = Rsi::<f64>::new(config.clone());

        // Feed some data
        let closes = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5];
        for &close in &closes {
            rsi1.update(&Bar::new(close, close, close, close, 1000.0))
                .unwrap();
        }

        // Get state
        let state = rsi1.get_state();

        // Create new indicator and restore state
        let mut rsi2 = Rsi::<f64>::new(config);
        rsi2.set_state(state).unwrap();

        // Both should produce the same result
        let result1 = rsi1
            .update(&Bar::new(43.75, 43.75, 43.75, 43.75, 1000.0))
            .unwrap();
        let result2 = rsi2
            .update(&Bar::new(43.75, 43.75, 43.75, 43.75, 1000.0))
            .unwrap();

        assert_relative_eq!(result1.unwrap(), result2.unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_rsi_reset() {
        let config = RsiConfig::new(5);
        let mut rsi = Rsi::<f64>::new(config);

        // Feed some data
        let closes = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25];
        for &close in &closes {
            rsi.update(&Bar::new(close, close, close, close, 1000.0))
                .unwrap();
        }

        assert!(rsi.is_ready());

        // Reset
        rsi.reset();

        assert!(!rsi.is_ready());
        assert!(rsi.prev_close.is_nan());
    }

    #[test]
    fn test_rsi_min_periods() {
        let config = RsiConfig::new(14);
        let rsi = Rsi::<f64>::new(config);

        // Matches python-ta: min_periods = window
        assert_eq!(rsi.min_periods(), 14);
    }
}
