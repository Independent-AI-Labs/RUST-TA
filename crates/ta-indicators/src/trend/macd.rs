//! Moving Average Convergence Divergence (MACD) indicator.
//!
//! MACD is a trend-following momentum indicator that shows the relationship
//! between two exponential moving averages of prices.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use ta_core::{
    error::{IndicatorError, Result},
    num::TaFloat,
    ohlcv::{Bar, OhlcvSeries},
    series::Series,
    traits::{Indicator, StreamingIndicator},
};

use super::ema::{Ema, EmaConfig, EmaState};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the MACD indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MacdConfig {
    /// Fast EMA period (default: 12).
    pub fast: usize,
    /// Slow EMA period (default: 26).
    pub slow: usize,
    /// Signal line EMA period (default: 9).
    pub signal: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for MacdConfig {
    fn default() -> Self {
        Self {
            fast: 12,
            slow: 26,
            signal: 9,
            fillna: false,
        }
    }
}

impl MacdConfig {
    /// Create a new MACD configuration.
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast,
            slow,
            signal,
            fillna: false,
        }
    }

    /// Set fillna option.
    pub fn with_fillna(mut self, fillna: bool) -> Self {
        self.fillna = fillna;
        self
    }
}

/// Output of the MACD indicator.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct MacdOutput<T: TaFloat> {
    /// MACD line (fast EMA - slow EMA).
    pub macd: T,
    /// Signal line (EMA of MACD line).
    pub signal: T,
    /// Histogram (MACD - Signal).
    pub histogram: T,
}

impl<T: TaFloat> MacdOutput<T> {
    /// Create a new MACD output.
    pub fn new(macd: T, signal: T, histogram: T) -> Self {
        Self {
            macd,
            signal,
            histogram,
        }
    }

    /// Create an output representing NaN values.
    pub fn nan() -> Self {
        Self {
            macd: T::NAN,
            signal: T::NAN,
            histogram: T::NAN,
        }
    }

    /// Check if any component is NaN.
    pub fn is_nan(&self) -> bool {
        self.macd.is_nan() || self.signal.is_nan() || self.histogram.is_nan()
    }
}

/// State for the MACD indicator (for serialization/deserialization).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct MacdState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// Fast EMA state.
    pub fast_ema: EmaState<T>,
    /// Slow EMA state.
    pub slow_ema: EmaState<T>,
    /// Signal EMA state.
    pub signal_ema: EmaState<T>,
    /// Number of values seen.
    pub count: usize,
}

impl<T: TaFloat> Default for MacdState<T> {
    fn default() -> Self {
        Self {
            version: 1,
            fast_ema: EmaState::default(),
            slow_ema: EmaState::default(),
            signal_ema: EmaState::default(),
            count: 0,
        }
    }
}

/// MACD series output containing all three series.
#[derive(Debug, Clone)]
pub struct MacdSeries<T: TaFloat> {
    /// MACD line series.
    pub macd: Series<T>,
    /// Signal line series.
    pub signal: Series<T>,
    /// Histogram series.
    pub histogram: Series<T>,
}

/// Moving Average Convergence Divergence indicator.
///
/// MACD is calculated by subtracting the slow EMA from the fast EMA.
/// The signal line is an EMA of the MACD line.
/// The histogram is the difference between MACD and signal.
///
/// # Formula
///
/// MACD Line = Fast EMA - Slow EMA
/// Signal Line = EMA(MACD Line, signal_period)
/// Histogram = MACD Line - Signal Line
#[derive(Debug, Clone)]
pub struct Macd<T: TaFloat> {
    config: MacdConfig,
    fast_ema: Ema<T>,
    slow_ema: Ema<T>,
    signal_ema: Ema<T>,
    count: usize,
    current_output: Option<MacdOutput<T>>,
}

impl<T: TaFloat> Macd<T> {
    /// Returns the current MACD output if ready.
    pub fn output(&self) -> Option<MacdOutput<T>> {
        self.current_output
    }

    /// Returns the number of values seen.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl<T: TaFloat> Indicator<T> for Macd<T> {
    type Output = MacdSeries<T>;
    type Config = MacdConfig;
    type State = MacdState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        let fast_ema = Ema::new(EmaConfig::new(config.fast).with_fillna(config.fillna));
        let slow_ema = Ema::new(EmaConfig::new(config.slow).with_fillna(config.fillna));
        let signal_ema = Ema::new(EmaConfig::new(config.signal).with_fillna(config.fillna));

        Self {
            config,
            fast_ema,
            slow_ema,
            signal_ema,
            count: 0,
            current_output: None,
        }
    }

    fn min_periods(&self) -> usize {
        // Need slow EMA to be ready, then signal periods for signal line
        self.config.slow + self.config.signal - 1
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let close = data.close();
        let len = close.len();

        if self.config.fast == 0 || self.config.slow == 0 || self.config.signal == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        if self.config.fast >= self.config.slow {
            return Err(IndicatorError::InvalidParameter {
                name: "fast",
                value: self.config.fast.to_string(),
                expected: "fast period must be less than slow period",
            });
        }

        // Calculate fast and slow EMAs
        let fast_ema = self.fast_ema.calculate(data)?;
        let slow_ema = self.slow_ema.calculate(data)?;

        // Calculate MACD line
        let mut macd_line = Series::with_capacity(len);
        for i in 0..len {
            if fast_ema[i].is_nan() || slow_ema[i].is_nan() {
                macd_line.push(T::NAN);
            } else {
                macd_line.push(fast_ema[i] - slow_ema[i]);
            }
        }

        // Calculate signal line (EMA of MACD)
        let mut signal_line = Series::with_capacity(len);
        let mut signal_ema = Ema::<T>::new(EmaConfig::new(self.config.signal));

        // Find first valid MACD value
        let mut first_valid_idx = None;
        for (i, &val) in macd_line.iter().enumerate() {
            if !val.is_nan() {
                first_valid_idx = Some(i);
                break;
            }
        }

        // Fill NaNs before first valid MACD
        if let Some(start_idx) = first_valid_idx {
            for _ in 0..start_idx {
                if self.config.fillna {
                    signal_line.push(T::ZERO);
                } else {
                    signal_line.push(T::NAN);
                }
            }

            // Calculate signal EMA from first valid MACD
            for i in start_idx..len {
                let macd_val = macd_line[i];
                let signal_result = signal_ema.update_value(macd_val);
                if let Some(sig) = signal_result {
                    signal_line.push(sig);
                } else if self.config.fillna {
                    signal_line.push(T::ZERO);
                } else {
                    signal_line.push(T::NAN);
                }
            }
        } else {
            // All NaNs
            for _ in 0..len {
                if self.config.fillna {
                    signal_line.push(T::ZERO);
                } else {
                    signal_line.push(T::NAN);
                }
            }
        }

        // Calculate histogram
        let mut histogram = Series::with_capacity(len);
        for i in 0..len {
            if macd_line[i].is_nan() || signal_line[i].is_nan() {
                histogram.push(T::NAN);
            } else {
                histogram.push(macd_line[i] - signal_line[i]);
            }
        }

        Ok(MacdSeries {
            macd: macd_line,
            signal: signal_line,
            histogram,
        })
    }

    fn get_state(&self) -> Self::State {
        MacdState {
            version: 1,
            fast_ema: self.fast_ema.get_state(),
            slow_ema: self.slow_ema.get_state(),
            signal_ema: self.signal_ema.get_state(),
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

        self.fast_ema.set_state(state.fast_ema)?;
        self.slow_ema.set_state(state.slow_ema)?;
        self.signal_ema.set_state(state.signal_ema)?;
        self.count = state.count;

        // Reconstruct current output
        if let (Some(fast), Some(slow)) = (self.fast_ema.value(), self.slow_ema.value()) {
            let macd = fast - slow;
            if let Some(signal) = self.signal_ema.value() {
                self.current_output = Some(MacdOutput::new(macd, signal, macd - signal));
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_ema.reset();
        self.count = 0;
        self.current_output = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Macd<T> {
    type StreamingOutput = Option<MacdOutput<T>>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<MacdOutput<T>>> {
        self.count += 1;
        let value = bar.close;

        // Update fast and slow EMAs
        let fast_result = self.fast_ema.update_value(value);
        let slow_result = self.slow_ema.update_value(value);

        // Both EMAs must be ready to compute MACD
        match (fast_result, slow_result) {
            (Some(fast), Some(slow)) => {
                let macd = fast - slow;

                // Update signal EMA with MACD value
                let signal_result = self.signal_ema.update_value(macd);

                if let Some(signal) = signal_result {
                    let histogram = macd - signal;
                    let output = MacdOutput::new(macd, signal, histogram);
                    self.current_output = Some(output);
                    Ok(Some(output))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    fn current(&self) -> Option<MacdOutput<T>> {
        self.current_output
    }

    fn is_ready(&self) -> bool {
        self.current_output.is_some()
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
    fn test_macd_default_config() {
        let config = MacdConfig::default();
        assert_eq!(config.fast, 12);
        assert_eq!(config.slow, 26);
        assert_eq!(config.signal, 9);
        assert!(!config.fillna);
    }

    #[test]
    fn test_macd_invalid_fast_slow() {
        let config = MacdConfig::new(26, 12, 9); // fast > slow is invalid
        let macd = Macd::<f64>::new(config);

        let closes: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let ohlcv = create_test_ohlcv(&closes);
        let result = macd.calculate(&ohlcv);

        assert!(result.is_err());
    }

    #[test]
    fn test_macd_min_periods() {
        let config = MacdConfig::new(12, 26, 9);
        let macd = Macd::<f64>::new(config);

        // slow + signal - 1 = 26 + 9 - 1 = 34
        assert_eq!(macd.min_periods(), 34);
    }

    #[test]
    fn test_macd_calculate() {
        let config = MacdConfig::new(3, 5, 2);
        let macd = Macd::<f64>::new(config);

        let closes: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let ohlcv = create_test_ohlcv(&closes);
        let result = macd.calculate(&ohlcv).unwrap();

        assert_eq!(result.macd.len(), 20);
        assert_eq!(result.signal.len(), 20);
        assert_eq!(result.histogram.len(), 20);

        // First few values should be NaN
        assert!(result.macd[0].is_nan());
        assert!(result.signal[0].is_nan());

        // Later values should be valid
        assert!(!result.macd[10].is_nan());
        assert!(!result.signal[10].is_nan());
        assert!(!result.histogram[10].is_nan());
    }

    #[test]
    fn test_macd_streaming() {
        let config = MacdConfig::new(3, 5, 2);
        let mut macd = Macd::<f64>::new(config);

        let closes: Vec<f64> = (1..=20).map(|x| x as f64).collect();

        let mut results = Vec::new();
        for &close in &closes {
            let bar = Bar::new(close, close, close, close, 1000.0);
            let result = macd.update(&bar).unwrap();
            results.push(result);
        }

        // First several should be None
        assert!(results[0].is_none());
        assert!(results[1].is_none());
        assert!(results[2].is_none());
        assert!(results[3].is_none());

        // Later ones should have values
        let last = results.last().unwrap().unwrap();
        assert!(!last.macd.is_nan());
        assert!(!last.signal.is_nan());
        assert!(!last.histogram.is_nan());
    }

    #[test]
    fn test_macd_histogram_calculation() {
        let config = MacdConfig::new(3, 5, 2);
        let macd = Macd::<f64>::new(config);

        let closes: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let ohlcv = create_test_ohlcv(&closes);
        let result = macd.calculate(&ohlcv).unwrap();

        // Histogram should be MACD - Signal
        for i in 0..result.macd.len() {
            if !result.macd[i].is_nan() && !result.signal[i].is_nan() {
                let expected_hist = result.macd[i] - result.signal[i];
                assert_relative_eq!(result.histogram[i], expected_hist, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_macd_streaming_equals_batch() {
        let config = MacdConfig::new(3, 5, 2);
        let closes: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let ohlcv = create_test_ohlcv(&closes);

        // Batch calculation
        let batch_macd = Macd::<f64>::new(config.clone());
        let batch_result = batch_macd.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_macd = Macd::<f64>::new(config);
        for (i, &close) in closes.iter().enumerate() {
            let bar = Bar::new(close, close, close, close, 1000.0);
            let streaming_result = streaming_macd.update(&bar).unwrap();

            if let Some(output) = streaming_result {
                assert_relative_eq!(output.macd, batch_result.macd[i], epsilon = 1e-10);
                assert_relative_eq!(output.signal, batch_result.signal[i], epsilon = 1e-10);
                assert_relative_eq!(output.histogram, batch_result.histogram[i], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_macd_state_roundtrip() {
        let config = MacdConfig::new(3, 5, 2);
        let mut macd1 = Macd::<f64>::new(config.clone());

        // Feed some data
        let closes: Vec<f64> = (1..=15).map(|x| x as f64).collect();
        for &close in &closes {
            macd1.update(&Bar::new(close, close, close, close, 1000.0)).unwrap();
        }

        // Get state
        let state = macd1.get_state();

        // Create new indicator and restore state
        let mut macd2 = Macd::<f64>::new(config);
        macd2.set_state(state).unwrap();

        // Both should produce the same result
        let result1 = macd1.update(&Bar::new(16.0, 16.0, 16.0, 16.0, 1000.0)).unwrap();
        let result2 = macd2.update(&Bar::new(16.0, 16.0, 16.0, 16.0, 1000.0)).unwrap();

        let out1 = result1.unwrap();
        let out2 = result2.unwrap();
        assert_relative_eq!(out1.macd, out2.macd, epsilon = 1e-10);
        assert_relative_eq!(out1.signal, out2.signal, epsilon = 1e-10);
        assert_relative_eq!(out1.histogram, out2.histogram, epsilon = 1e-10);
    }

    #[test]
    fn test_macd_reset() {
        let config = MacdConfig::new(3, 5, 2);
        let mut macd = Macd::<f64>::new(config);

        // Feed some data
        let closes: Vec<f64> = (1..=15).map(|x| x as f64).collect();
        for &close in &closes {
            macd.update(&Bar::new(close, close, close, close, 1000.0)).unwrap();
        }

        assert!(macd.is_ready());

        // Reset
        macd.reset();

        assert!(!macd.is_ready());
        assert_eq!(macd.count(), 0);
    }
}
