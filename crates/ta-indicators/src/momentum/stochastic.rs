//! Stochastic Oscillator indicator.
//!
//! The Stochastic Oscillator is a momentum indicator comparing a particular
//! closing price to a range of prices over a certain period.

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

/// Configuration for the Stochastic Oscillator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StochasticConfig {
    /// Lookback period for %K (default: 14).
    pub k_period: usize,
    /// Smoothing period for %K (default: 3).
    pub k_smooth: usize,
    /// Smoothing period for %D (default: 3).
    pub d_period: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for StochasticConfig {
    fn default() -> Self {
        Self {
            k_period: 14,
            k_smooth: 3,
            d_period: 3,
            fillna: false,
        }
    }
}

impl StochasticConfig {
    /// Create a new configuration.
    pub fn new(k_period: usize, k_smooth: usize, d_period: usize) -> Self {
        Self {
            k_period,
            k_smooth,
            d_period,
            fillna: false,
        }
    }

    /// Set fillna option.
    pub fn with_fillna(mut self, fillna: bool) -> Self {
        self.fillna = fillna;
        self
    }
}

/// Output of the Stochastic Oscillator.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct StochasticOutput<T: TaFloat> {
    /// %K line (0-100).
    pub k: T,
    /// %D line (smoothed %K, 0-100).
    pub d: T,
}

impl<T: TaFloat> StochasticOutput<T> {
    /// Create a new output.
    pub fn new(k: T, d: T) -> Self {
        Self { k, d }
    }
}

/// State for the Stochastic Oscillator.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct StochasticState<T: TaFloat> {
    /// Version tag.
    pub version: u32,
    /// High buffer.
    pub high_buffer: Vec<T>,
    /// Low buffer.
    pub low_buffer: Vec<T>,
    /// Raw %K buffer for smoothing.
    pub raw_k_buffer: Vec<T>,
    /// %K buffer for %D smoothing.
    pub k_buffer: Vec<T>,
    /// Count of values seen.
    pub count: usize,
}

/// Stochastic Oscillator series output.
#[derive(Debug, Clone)]
pub struct StochasticSeries<T: TaFloat> {
    /// %K series.
    pub k: Series<T>,
    /// %D series.
    pub d: Series<T>,
}

/// Stochastic Oscillator indicator.
///
/// # Formula
///
/// Raw %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
/// %K = SMA(Raw %K, k_smooth)
/// %D = SMA(%K, d_period)
#[derive(Debug, Clone)]
pub struct Stochastic<T: TaFloat> {
    config: StochasticConfig,
    high_buffer: RingBuffer<T>,
    low_buffer: RingBuffer<T>,
    raw_k_buffer: RingBuffer<T>,
    k_buffer: RingBuffer<T>,
    count: usize,
    current_output: Option<StochasticOutput<T>>,
}

impl<T: TaFloat> Stochastic<T> {
    /// Get current output.
    pub fn output(&self) -> Option<StochasticOutput<T>> {
        self.current_output
    }

    fn calc_raw_k(close: T, lowest: T, highest: T) -> T {
        let range = highest - lowest;
        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);
        if range.abs() < epsilon {
            T::FIFTY
        } else {
            T::HUNDRED * (close - lowest) / range
        }
    }

    fn buffer_sma(buffer: &RingBuffer<T>) -> T {
        if buffer.is_empty() {
            return T::NAN;
        }
        buffer.sum() / <T as TaFloat>::from_usize(buffer.len())
    }
}

impl<T: TaFloat> Indicator<T> for Stochastic<T> {
    type Output = StochasticSeries<T>;
    type Config = StochasticConfig;
    type State = StochasticState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            high_buffer: RingBuffer::new(config.k_period),
            low_buffer: RingBuffer::new(config.k_period),
            raw_k_buffer: RingBuffer::new(config.k_smooth),
            k_buffer: RingBuffer::new(config.d_period),
            config,
            count: 0,
            current_output: None,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.k_period + self.config.k_smooth + self.config.d_period - 2
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();
        let high = data.high();
        let low = data.low();
        let close = data.close();

        if self.config.k_period == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        let mut k_series = Series::with_capacity(len);
        let mut d_series = Series::with_capacity(len);

        let mut high_buf: RingBuffer<T> = RingBuffer::new(self.config.k_period);
        let mut low_buf: RingBuffer<T> = RingBuffer::new(self.config.k_period);
        let mut raw_k_buf: RingBuffer<T> = RingBuffer::new(self.config.k_smooth);
        let mut k_buf: RingBuffer<T> = RingBuffer::new(self.config.d_period);

        for i in 0..len {
            high_buf.push(high[i]);
            low_buf.push(low[i]);

            if high_buf.is_full() {
                let highest = high_buf.max();
                let lowest = low_buf.min();
                let raw_k = Self::calc_raw_k(close[i], lowest, highest);

                raw_k_buf.push(raw_k);

                if raw_k_buf.is_full() {
                    let k = Self::buffer_sma(&raw_k_buf);
                    k_buf.push(k);

                    if k_buf.is_full() {
                        let d = Self::buffer_sma(&k_buf);
                        k_series.push(k);
                        d_series.push(d);
                    } else {
                        k_series.push(k);
                        if self.config.fillna {
                            d_series.push(T::ZERO);
                        } else {
                            d_series.push(T::NAN);
                        }
                    }
                } else {
                    if self.config.fillna {
                        k_series.push(T::ZERO);
                        d_series.push(T::ZERO);
                    } else {
                        k_series.push(T::NAN);
                        d_series.push(T::NAN);
                    }
                }
            } else {
                if self.config.fillna {
                    k_series.push(T::ZERO);
                    d_series.push(T::ZERO);
                } else {
                    k_series.push(T::NAN);
                    d_series.push(T::NAN);
                }
            }
        }

        Ok(StochasticSeries {
            k: k_series,
            d: d_series,
        })
    }

    fn get_state(&self) -> Self::State {
        StochasticState {
            version: 1,
            high_buffer: self.high_buffer.iter().copied().collect(),
            low_buffer: self.low_buffer.iter().copied().collect(),
            raw_k_buffer: self.raw_k_buffer.iter().copied().collect(),
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

        self.high_buffer = RingBuffer::new(self.config.k_period);
        for v in state.high_buffer {
            self.high_buffer.push(v);
        }

        self.low_buffer = RingBuffer::new(self.config.k_period);
        for v in state.low_buffer {
            self.low_buffer.push(v);
        }

        self.raw_k_buffer = RingBuffer::new(self.config.k_smooth);
        for v in state.raw_k_buffer {
            self.raw_k_buffer.push(v);
        }

        self.k_buffer = RingBuffer::new(self.config.d_period);
        for v in state.k_buffer {
            self.k_buffer.push(v);
        }

        self.count = state.count;
        Ok(())
    }

    fn reset(&mut self) {
        self.high_buffer = RingBuffer::new(self.config.k_period);
        self.low_buffer = RingBuffer::new(self.config.k_period);
        self.raw_k_buffer = RingBuffer::new(self.config.k_smooth);
        self.k_buffer = RingBuffer::new(self.config.d_period);
        self.count = 0;
        self.current_output = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Stochastic<T> {
    type StreamingOutput = Option<StochasticOutput<T>>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<StochasticOutput<T>>> {
        self.count += 1;

        self.high_buffer.push(bar.high);
        self.low_buffer.push(bar.low);

        if !self.high_buffer.is_full() {
            return Ok(None);
        }

        let highest = self.high_buffer.max();
        let lowest = self.low_buffer.min();
        let raw_k = Self::calc_raw_k(bar.close, lowest, highest);

        self.raw_k_buffer.push(raw_k);

        if !self.raw_k_buffer.is_full() {
            return Ok(None);
        }

        let k = Self::buffer_sma(&self.raw_k_buffer);
        self.k_buffer.push(k);

        if !self.k_buffer.is_full() {
            return Ok(None);
        }

        let d = Self::buffer_sma(&self.k_buffer);
        let output = StochasticOutput::new(k, d);
        self.current_output = Some(output);
        Ok(Some(output))
    }

    fn current(&self) -> Option<StochasticOutput<T>> {
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

    #[test]
    fn test_stochastic_default_config() {
        let config = StochasticConfig::default();
        assert_eq!(config.k_period, 14);
        assert_eq!(config.k_smooth, 3);
        assert_eq!(config.d_period, 3);
    }

    #[test]
    fn test_stochastic_bounds() {
        let config = StochasticConfig::new(5, 2, 2);
        let mut stoch = Stochastic::<f64>::new(config);

        for i in 0..30 {
            let price = 100.0 + (i as f64 * 0.3).sin() * 10.0;
            let bar = Bar::new(price, price + 2.0, price - 2.0, price, 1000.0);
            if let Ok(Some(output)) = stoch.update(&bar) {
                assert!(output.k >= 0.0 && output.k <= 100.0, "%K out of bounds: {}", output.k);
                assert!(output.d >= 0.0 && output.d <= 100.0, "%D out of bounds: {}", output.d);
            }
        }
    }

    #[test]
    fn test_stochastic_constant_range() {
        let config = StochasticConfig::new(3, 1, 1);
        let mut stoch = Stochastic::<f64>::new(config);

        // When close is at midpoint of range, %K should be 50
        for _ in 0..10 {
            let bar = Bar::new(100.0, 110.0, 90.0, 100.0, 1000.0);
            if let Ok(Some(output)) = stoch.update(&bar) {
                assert_relative_eq!(output.k, 50.0, epsilon = 1e-8);
            }
        }
    }
}
