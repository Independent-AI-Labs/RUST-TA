//! Williams %R indicator.
//!
//! Williams %R is a momentum indicator that measures overbought/oversold levels.

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

/// Configuration for Williams %R.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WilliamsRConfig {
    /// Lookback period (default: 14).
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for WilliamsRConfig {
    fn default() -> Self {
        Self {
            window: 14,
            fillna: false,
        }
    }
}

impl WilliamsRConfig {
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

/// State for Williams %R.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct WilliamsRState<T: TaFloat> {
    /// Version tag.
    pub version: u32,
    /// High buffer.
    pub high_buffer: Vec<T>,
    /// Low buffer.
    pub low_buffer: Vec<T>,
    /// Count of values seen.
    pub count: usize,
}

/// Williams %R indicator.
///
/// # Formula
///
/// %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
///
/// Range: [-100, 0]
/// - Values near 0: overbought
/// - Values near -100: oversold
#[derive(Debug, Clone)]
pub struct WilliamsR<T: TaFloat> {
    config: WilliamsRConfig,
    high_buffer: RingBuffer<T>,
    low_buffer: RingBuffer<T>,
    count: usize,
    current_value: Option<T>,
}

impl<T: TaFloat> WilliamsR<T> {
    /// Get current value.
    pub fn value(&self) -> Option<T> {
        self.current_value
    }
}

impl<T: TaFloat> Indicator<T> for WilliamsR<T> {
    type Output = Series<T>;
    type Config = WilliamsRConfig;
    type State = WilliamsRState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            high_buffer: RingBuffer::new(config.window),
            low_buffer: RingBuffer::new(config.window),
            config,
            count: 0,
            current_value: None,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.window
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();
        let high = data.high();
        let low = data.low();
        let close = data.close();
        let window = self.config.window;

        if window == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        let mut result = Series::with_capacity(len);
        let mut high_buf: RingBuffer<T> = RingBuffer::new(window);
        let mut low_buf: RingBuffer<T> = RingBuffer::new(window);

        for i in 0..len {
            high_buf.push(high[i]);
            low_buf.push(low[i]);

            if high_buf.is_full() {
                let highest = high_buf.max();
                let lowest = low_buf.min();
                let range = highest - lowest;
                let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

                let wr = if range.abs() < epsilon {
                    -T::FIFTY // Neutral when no range
                } else {
                    -T::HUNDRED * (highest - close[i]) / range
                };
                result.push(wr);
            } else if self.config.fillna {
                result.push(T::ZERO);
            } else {
                result.push(T::NAN);
            }
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        WilliamsRState {
            version: 1,
            high_buffer: self.high_buffer.iter().copied().collect(),
            low_buffer: self.low_buffer.iter().copied().collect(),
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

        self.high_buffer = RingBuffer::new(self.config.window);
        for v in state.high_buffer {
            self.high_buffer.push(v);
        }

        self.low_buffer = RingBuffer::new(self.config.window);
        for v in state.low_buffer {
            self.low_buffer.push(v);
        }

        self.count = state.count;
        Ok(())
    }

    fn reset(&mut self) {
        self.high_buffer = RingBuffer::new(self.config.window);
        self.low_buffer = RingBuffer::new(self.config.window);
        self.count = 0;
        self.current_value = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for WilliamsR<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        self.count += 1;

        self.high_buffer.push(bar.high);
        self.low_buffer.push(bar.low);

        if !self.high_buffer.is_full() {
            return Ok(None);
        }

        let highest = self.high_buffer.max();
        let lowest = self.low_buffer.min();
        let range = highest - lowest;
        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

        let wr = if range.abs() < epsilon {
            -T::FIFTY
        } else {
            -T::HUNDRED * (highest - bar.close) / range
        };

        self.current_value = Some(wr);
        Ok(Some(wr))
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

    #[test]
    fn test_williams_r_default_config() {
        let config = WilliamsRConfig::default();
        assert_eq!(config.window, 14);
    }

    #[test]
    fn test_williams_r_bounds() {
        let config = WilliamsRConfig::new(5);
        let mut wr = WilliamsR::<f64>::new(config);

        for i in 0..20 {
            let price = 100.0 + (i as f64 * 0.3).sin() * 10.0;
            let bar = Bar::new(price, price + 2.0, price - 2.0, price, 1000.0);
            if let Ok(Some(value)) = wr.update(&bar) {
                assert!(value >= -100.0 && value <= 0.0, "%R out of bounds: {}", value);
            }
        }
    }

    #[test]
    fn test_williams_r_at_high() {
        let config = WilliamsRConfig::new(3);
        let mut wr = WilliamsR::<f64>::new(config);

        // When close is at highest high, %R should be 0
        for _ in 0..5 {
            let bar = Bar::new(100.0, 110.0, 90.0, 110.0, 1000.0);
            if let Ok(Some(value)) = wr.update(&bar) {
                assert_relative_eq!(value, 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_williams_r_at_low() {
        let config = WilliamsRConfig::new(3);
        let mut wr = WilliamsR::<f64>::new(config);

        // When close is at lowest low, %R should be -100
        for _ in 0..5 {
            let bar = Bar::new(100.0, 110.0, 90.0, 90.0, 1000.0);
            if let Ok(Some(value)) = wr.update(&bar) {
                assert_relative_eq!(value, -100.0, epsilon = 1e-8);
            }
        }
    }
}
