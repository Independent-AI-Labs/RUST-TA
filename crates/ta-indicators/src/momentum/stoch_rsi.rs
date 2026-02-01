//! Stochastic RSI (StochRSI) indicator.
//!
//! StochRSI applies the Stochastic oscillator formula to RSI values
//! instead of price values.

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

use super::rsi::{Rsi, RsiConfig, RsiState};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the StochRSI indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StochRsiConfig {
    /// RSI period (default: 14).
    pub window: usize,
    /// Smoothing period for %K (default: 3).
    pub smooth_k: usize,
    /// Smoothing period for %D (default: 3).
    pub smooth_d: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for StochRsiConfig {
    fn default() -> Self {
        Self {
            window: 14,
            smooth_k: 3,
            smooth_d: 3,
            fillna: false,
        }
    }
}

impl StochRsiConfig {
    /// Create a new StochRSI configuration.
    pub fn new(window: usize, smooth_k: usize, smooth_d: usize) -> Self {
        Self {
            window,
            smooth_k,
            smooth_d,
            fillna: false,
        }
    }

    /// Set fillna option.
    pub fn with_fillna(mut self, fillna: bool) -> Self {
        self.fillna = fillna;
        self
    }
}

/// Output of the StochRSI indicator.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct StochRsiOutput<T: TaFloat> {
    /// Raw StochRSI value (0 to 1).
    pub stochrsi: T,
    /// %K line (smoothed StochRSI, 0 to 1).
    pub stochrsi_k: T,
    /// %D line (smoothed %K, 0 to 1).
    pub stochrsi_d: T,
}

impl<T: TaFloat> StochRsiOutput<T> {
    /// Create a new StochRSI output.
    pub fn new(stochrsi: T, stochrsi_k: T, stochrsi_d: T) -> Self {
        Self {
            stochrsi,
            stochrsi_k,
            stochrsi_d,
        }
    }

    /// Check if any component is NaN.
    pub fn is_nan(&self) -> bool {
        self.stochrsi.is_nan() || self.stochrsi_k.is_nan() || self.stochrsi_d.is_nan()
    }
}

/// State for the StochRSI indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct StochRsiState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// RSI state.
    pub rsi_state: RsiState<T>,
    /// Buffer of RSI values for StochRSI calculation.
    pub rsi_buffer: Vec<T>,
    /// Buffer of StochRSI values for %K smoothing.
    pub stochrsi_buffer: Vec<T>,
    /// Buffer of %K values for %D smoothing.
    pub k_buffer: Vec<T>,
    /// Number of values seen.
    pub count: usize,
}

impl<T: TaFloat> Default for StochRsiState<T> {
    fn default() -> Self {
        Self {
            version: 1,
            rsi_state: RsiState::default(),
            rsi_buffer: Vec::new(),
            stochrsi_buffer: Vec::new(),
            k_buffer: Vec::new(),
            count: 0,
        }
    }
}

/// StochRSI series output.
#[derive(Debug, Clone)]
pub struct StochRsiSeries<T: TaFloat> {
    /// Raw StochRSI values.
    pub stochrsi: Series<T>,
    /// %K line values.
    pub stochrsi_k: Series<T>,
    /// %D line values.
    pub stochrsi_d: Series<T>,
}

/// Stochastic RSI indicator.
///
/// StochRSI applies the Stochastic oscillator formula to RSI values:
///
/// StochRSI = (RSI - min(RSI, n)) / (max(RSI, n) - min(RSI, n))
///
/// # Edge Cases
///
/// When max(RSI) = min(RSI), StochRSI = 0.5 (neutral)
#[derive(Debug, Clone)]
pub struct StochRsi<T: TaFloat> {
    config: StochRsiConfig,
    rsi: Rsi<T>,
    /// Buffer for RSI values (for min/max calculation)
    rsi_buffer: RingBuffer<T>,
    /// Buffer for StochRSI values (for %K smoothing)
    stochrsi_buffer: RingBuffer<T>,
    /// Buffer for %K values (for %D smoothing)
    k_buffer: RingBuffer<T>,
    /// Number of values seen
    count: usize,
    /// Current output
    current_output: Option<StochRsiOutput<T>>,
}

impl<T: TaFloat> StochRsi<T> {
    /// Returns the current output if ready.
    pub fn output(&self) -> Option<StochRsiOutput<T>> {
        self.current_output
    }

    /// Calculate StochRSI from RSI and buffer.
    fn calc_stochrsi(rsi: T, min_rsi: T, max_rsi: T) -> T {
        let range = max_rsi - min_rsi;
        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

        if range < epsilon {
            // When max = min, return neutral value
            <T as TaFloat>::from_f64_lossy(0.5)
        } else {
            (rsi - min_rsi) / range
        }
    }

    /// Calculate SMA of buffer values.
    fn buffer_sma(buffer: &RingBuffer<T>) -> T {
        if buffer.is_empty() {
            return T::NAN;
        }
        buffer.sum() / <T as TaFloat>::from_usize(buffer.len())
    }
}

impl<T: TaFloat> Indicator<T> for StochRsi<T> {
    type Output = StochRsiSeries<T>;
    type Config = StochRsiConfig;
    type State = StochRsiState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        let rsi = Rsi::new(RsiConfig::new(config.window).with_fillna(config.fillna));

        Self {
            rsi,
            rsi_buffer: RingBuffer::new(config.window),
            stochrsi_buffer: RingBuffer::new(config.smooth_k),
            k_buffer: RingBuffer::new(config.smooth_d),
            config,
            count: 0,
            current_output: None,
        }
    }

    fn min_periods(&self) -> usize {
        // RSI needs window+1, then we need window more for StochRSI range
        // Plus smooth_k for %K and smooth_d for %D
        self.config.window + 1 + self.config.window + self.config.smooth_k + self.config.smooth_d
            - 2
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();
        let window = self.config.window;
        let smooth_k = self.config.smooth_k;
        let smooth_d = self.config.smooth_d;

        if window == 0 || smooth_k == 0 || smooth_d == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        // Calculate RSI first
        let rsi_series = self.rsi.calculate(data)?;

        // Calculate StochRSI
        let mut stochrsi = Series::with_capacity(len);
        let mut stochrsi_k = Series::with_capacity(len);
        let mut stochrsi_d = Series::with_capacity(len);

        let mut rsi_buffer: RingBuffer<T> = RingBuffer::new(window);
        let mut stochrsi_buf: RingBuffer<T> = RingBuffer::new(smooth_k);
        let mut k_buf: RingBuffer<T> = RingBuffer::new(smooth_d);

        for i in 0..len {
            let rsi_val = rsi_series[i];

            // StochRSI calculation
            let stoch_val = if rsi_val.is_nan() {
                T::NAN
            } else {
                rsi_buffer.push(rsi_val);
                if rsi_buffer.is_full() {
                    let min_rsi = rsi_buffer.min();
                    let max_rsi = rsi_buffer.max();
                    Self::calc_stochrsi(rsi_val, min_rsi, max_rsi)
                } else if self.config.fillna {
                    T::ZERO
                } else {
                    T::NAN
                }
            };
            stochrsi.push(stoch_val);

            // %K calculation (SMA of StochRSI)
            let k_val = if stoch_val.is_nan() {
                T::NAN
            } else {
                stochrsi_buf.push(stoch_val);
                if stochrsi_buf.is_full() {
                    Self::buffer_sma(&stochrsi_buf)
                } else if self.config.fillna {
                    T::ZERO
                } else {
                    T::NAN
                }
            };
            stochrsi_k.push(k_val);

            // %D calculation (SMA of %K)
            let d_val = if k_val.is_nan() {
                T::NAN
            } else {
                k_buf.push(k_val);
                if k_buf.is_full() {
                    Self::buffer_sma(&k_buf)
                } else if self.config.fillna {
                    T::ZERO
                } else {
                    T::NAN
                }
            };
            stochrsi_d.push(d_val);
        }

        Ok(StochRsiSeries {
            stochrsi,
            stochrsi_k,
            stochrsi_d,
        })
    }

    fn get_state(&self) -> Self::State {
        StochRsiState {
            version: 1,
            rsi_state: self.rsi.get_state(),
            rsi_buffer: self.rsi_buffer.iter().copied().collect(),
            stochrsi_buffer: self.stochrsi_buffer.iter().copied().collect(),
            k_buffer: self.k_buffer.iter().copied().collect(),
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

        self.rsi.set_state(state.rsi_state)?;

        self.rsi_buffer = RingBuffer::new(self.config.window);
        for value in state.rsi_buffer {
            self.rsi_buffer.push(value);
        }

        self.stochrsi_buffer = RingBuffer::new(self.config.smooth_k);
        for value in state.stochrsi_buffer {
            self.stochrsi_buffer.push(value);
        }

        self.k_buffer = RingBuffer::new(self.config.smooth_d);
        for value in state.k_buffer {
            self.k_buffer.push(value);
        }

        self.count = state.count;

        Ok(())
    }

    fn reset(&mut self) {
        self.rsi.reset();
        self.rsi_buffer = RingBuffer::new(self.config.window);
        self.stochrsi_buffer = RingBuffer::new(self.config.smooth_k);
        self.k_buffer = RingBuffer::new(self.config.smooth_d);
        self.count = 0;
        self.current_output = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for StochRsi<T> {
    type StreamingOutput = Option<StochRsiOutput<T>>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<StochRsiOutput<T>>> {
        self.count += 1;

        // Update RSI
        let rsi_result = self.rsi.update(bar)?;

        if let Some(rsi_val) = rsi_result {
            // Add to RSI buffer
            self.rsi_buffer.push(rsi_val);

            // Calculate StochRSI if buffer is full
            if self.rsi_buffer.is_full() {
                let min_rsi = self.rsi_buffer.min();
                let max_rsi = self.rsi_buffer.max();
                let stoch_val = Self::calc_stochrsi(rsi_val, min_rsi, max_rsi);

                // Add to StochRSI buffer for %K
                self.stochrsi_buffer.push(stoch_val);

                // Calculate %K if buffer is full
                if self.stochrsi_buffer.is_full() {
                    let k_val = Self::buffer_sma(&self.stochrsi_buffer);

                    // Add to K buffer for %D
                    self.k_buffer.push(k_val);

                    // Calculate %D if buffer is full
                    if self.k_buffer.is_full() {
                        let d_val = Self::buffer_sma(&self.k_buffer);

                        let output = StochRsiOutput::new(stoch_val, k_val, d_val);
                        self.current_output = Some(output);
                        return Ok(Some(output));
                    }
                }
            }
        }

        Ok(None)
    }

    fn current(&self) -> Option<StochRsiOutput<T>> {
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
    fn test_stochrsi_default_config() {
        let config = StochRsiConfig::default();
        assert_eq!(config.window, 14);
        assert_eq!(config.smooth_k, 3);
        assert_eq!(config.smooth_d, 3);
        assert!(!config.fillna);
    }

    #[test]
    fn test_stochrsi_bounds() {
        // StochRSI should always be in [0, 1]
        let config = StochRsiConfig::new(5, 2, 2);
        let mut stochrsi = StochRsi::<f64>::new(config);

        let closes: Vec<f64> = (1..=100)
            .map(|x| 100.0 + (x as f64 * 0.1).sin() * 10.0)
            .collect();

        for &close in &closes {
            let bar = Bar::new(close, close + 1.0, close - 1.0, close, 1000.0);
            if let Ok(Some(output)) = stochrsi.update(&bar) {
                assert!(
                    output.stochrsi >= 0.0 && output.stochrsi <= 1.0,
                    "StochRSI {} out of bounds",
                    output.stochrsi
                );
                assert!(
                    output.stochrsi_k >= 0.0 && output.stochrsi_k <= 1.0,
                    "%K {} out of bounds",
                    output.stochrsi_k
                );
                assert!(
                    output.stochrsi_d >= 0.0 && output.stochrsi_d <= 1.0,
                    "%D {} out of bounds",
                    output.stochrsi_d
                );
            }
        }
    }

    #[test]
    fn test_stochrsi_constant_rsi() {
        // When RSI is constant, StochRSI should be 0.5
        let config = StochRsiConfig::new(5, 2, 2);
        let stochrsi = StochRsi::<f64>::new(config);

        // Create data that produces constant RSI (no price change after init)
        let mut closes = vec![100.0; 50];
        let ohlcv = create_test_ohlcv(&closes);
        let result = stochrsi.calculate(&ohlcv).unwrap();

        // Find first valid StochRSI value
        for val in result.stochrsi.iter() {
            if !val.is_nan() {
                assert_relative_eq!(*val, 0.5, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_stochrsi_streaming_equals_batch() {
        let config = StochRsiConfig::new(5, 2, 2);
        let closes: Vec<f64> = (1..=50).map(|x| 100.0 + (x as f64 * 0.2).sin() * 5.0).collect();
        let ohlcv = create_test_ohlcv(&closes);

        // Batch calculation
        let batch_stochrsi = StochRsi::<f64>::new(config.clone());
        let batch_result = batch_stochrsi.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_stochrsi = StochRsi::<f64>::new(config);
        for (i, &close) in closes.iter().enumerate() {
            let bar = Bar::new(close, close, close, close, 1000.0);
            let streaming_result = streaming_stochrsi.update(&bar).unwrap();

            if let Some(output) = streaming_result {
                if !batch_result.stochrsi[i].is_nan() {
                    assert_relative_eq!(output.stochrsi, batch_result.stochrsi[i], epsilon = 1e-8);
                }
                if !batch_result.stochrsi_k[i].is_nan() {
                    assert_relative_eq!(
                        output.stochrsi_k,
                        batch_result.stochrsi_k[i],
                        epsilon = 1e-8
                    );
                }
                if !batch_result.stochrsi_d[i].is_nan() {
                    assert_relative_eq!(
                        output.stochrsi_d,
                        batch_result.stochrsi_d[i],
                        epsilon = 1e-8
                    );
                }
            }
        }
    }

    #[test]
    fn test_stochrsi_reset() {
        let config = StochRsiConfig::new(5, 2, 2);
        let mut stochrsi = StochRsi::<f64>::new(config);

        // Feed data until ready
        let closes: Vec<f64> = (1..=50).map(|x| 100.0 + (x as f64 * 0.2).sin() * 5.0).collect();
        for &close in &closes {
            stochrsi
                .update(&Bar::new(close, close, close, close, 1000.0))
                .unwrap();
        }

        assert!(stochrsi.is_ready());

        // Reset
        stochrsi.reset();

        assert!(!stochrsi.is_ready());
        assert_eq!(stochrsi.count, 0);
    }

    #[test]
    fn test_stochrsi_state_roundtrip() {
        let config = StochRsiConfig::new(5, 2, 2);
        let mut stochrsi1 = StochRsi::<f64>::new(config.clone());

        // Feed data
        let closes: Vec<f64> = (1..=40).map(|x| 100.0 + (x as f64 * 0.2).sin() * 5.0).collect();
        for &close in &closes {
            stochrsi1
                .update(&Bar::new(close, close, close, close, 1000.0))
                .unwrap();
        }

        // Get state
        let state = stochrsi1.get_state();

        // Create new indicator and restore state
        let mut stochrsi2 = StochRsi::<f64>::new(config);
        stochrsi2.set_state(state).unwrap();

        // Both should produce the same result
        let result1 = stochrsi1
            .update(&Bar::new(105.0, 105.0, 105.0, 105.0, 1000.0))
            .unwrap();
        let result2 = stochrsi2
            .update(&Bar::new(105.0, 105.0, 105.0, 105.0, 1000.0))
            .unwrap();

        match (result1, result2) {
            (Some(o1), Some(o2)) => {
                assert_relative_eq!(o1.stochrsi, o2.stochrsi, epsilon = 1e-10);
                assert_relative_eq!(o1.stochrsi_k, o2.stochrsi_k, epsilon = 1e-10);
                assert_relative_eq!(o1.stochrsi_d, o2.stochrsi_d, epsilon = 1e-10);
            }
            (None, None) => {}
            _ => panic!("Results should match"),
        }
    }
}
