//! Donchian Channel indicator.
//!
//! Donchian Channel uses highest high and lowest low over a period.

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

/// Configuration for Donchian Channel.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DonchianConfig {
    /// Window size (default: 20).
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for DonchianConfig {
    fn default() -> Self {
        Self {
            window: 20,
            fillna: false,
        }
    }
}

impl DonchianConfig {
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

/// Output of Donchian Channel.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct DonchianOutput<T: TaFloat> {
    /// Upper band (highest high).
    pub upper: T,
    /// Middle band (average of upper and lower).
    pub middle: T,
    /// Lower band (lowest low).
    pub lower: T,
    /// Channel width.
    pub width: T,
}

impl<T: TaFloat> DonchianOutput<T> {
    /// Create a new output.
    pub fn new(upper: T, lower: T) -> Self {
        let middle = (upper + lower) / T::TWO;
        let width = upper - lower;
        Self { upper, middle, lower, width }
    }
}

/// State for Donchian Channel.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct DonchianState<T: TaFloat> {
    /// Version tag.
    pub version: u32,
    /// High buffer.
    pub high_buffer: Vec<T>,
    /// Low buffer.
    pub low_buffer: Vec<T>,
    /// Count.
    pub count: usize,
}

/// Donchian Channel series output.
#[derive(Debug, Clone)]
pub struct DonchianSeries<T: TaFloat> {
    /// Upper band series.
    pub upper: Series<T>,
    /// Middle band series.
    pub middle: Series<T>,
    /// Lower band series.
    pub lower: Series<T>,
    /// Width series.
    pub width: Series<T>,
}

/// Donchian Channel indicator.
///
/// # Formula
///
/// Upper = max(High, window)
/// Lower = min(Low, window)
/// Middle = (Upper + Lower) / 2
/// Width = Upper - Lower
#[derive(Debug, Clone)]
pub struct DonchianChannel<T: TaFloat> {
    config: DonchianConfig,
    high_buffer: RingBuffer<T>,
    low_buffer: RingBuffer<T>,
    count: usize,
    current_output: Option<DonchianOutput<T>>,
}

impl<T: TaFloat> DonchianChannel<T> {
    /// Get current output.
    pub fn output(&self) -> Option<DonchianOutput<T>> {
        self.current_output
    }
}

impl<T: TaFloat> Indicator<T> for DonchianChannel<T> {
    type Output = DonchianSeries<T>;
    type Config = DonchianConfig;
    type State = DonchianState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            high_buffer: RingBuffer::new(config.window),
            low_buffer: RingBuffer::new(config.window),
            config,
            count: 0,
            current_output: None,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.window
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();
        let high = data.high();
        let low = data.low();
        let window = self.config.window;

        if window == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        let mut upper = Series::with_capacity(len);
        let mut middle = Series::with_capacity(len);
        let mut lower_series = Series::with_capacity(len);
        let mut width = Series::with_capacity(len);

        let mut high_buf: RingBuffer<T> = RingBuffer::new(window);
        let mut low_buf: RingBuffer<T> = RingBuffer::new(window);

        for i in 0..len {
            high_buf.push(high[i]);
            low_buf.push(low[i]);

            if high_buf.is_full() {
                let highest = high_buf.max();
                let lowest = low_buf.min();
                let mid = (highest + lowest) / T::TWO;
                let w = highest - lowest;

                upper.push(highest);
                lower_series.push(lowest);
                middle.push(mid);
                width.push(w);
            } else if self.config.fillna {
                upper.push(T::ZERO);
                lower_series.push(T::ZERO);
                middle.push(T::ZERO);
                width.push(T::ZERO);
            } else {
                upper.push(T::NAN);
                lower_series.push(T::NAN);
                middle.push(T::NAN);
                width.push(T::NAN);
            }
        }

        Ok(DonchianSeries {
            upper,
            middle,
            lower: lower_series,
            width,
        })
    }

    fn get_state(&self) -> Self::State {
        DonchianState {
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
        self.current_output = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for DonchianChannel<T> {
    type StreamingOutput = Option<DonchianOutput<T>>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<DonchianOutput<T>>> {
        self.count += 1;

        self.high_buffer.push(bar.high);
        self.low_buffer.push(bar.low);

        if !self.high_buffer.is_full() {
            return Ok(None);
        }

        let highest = self.high_buffer.max();
        let lowest = self.low_buffer.min();

        let output = DonchianOutput::new(highest, lowest);
        self.current_output = Some(output);
        Ok(Some(output))
    }

    fn current(&self) -> Option<DonchianOutput<T>> {
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
    fn test_donchian_default_config() {
        let config = DonchianConfig::default();
        assert_eq!(config.window, 20);
    }

    #[test]
    fn test_donchian_band_order() {
        let config = DonchianConfig::new(5);
        let mut donchian = DonchianChannel::<f64>::new(config);

        for i in 0..20 {
            let price = 100.0 + (i as f64 * 0.2).sin() * 5.0;
            let bar = Bar::new(price, price + 2.0, price - 2.0, price, 1000.0);
            if let Ok(Some(output)) = donchian.update(&bar) {
                assert!(output.lower <= output.middle,
                    "Lower {} > Middle {}", output.lower, output.middle);
                assert!(output.middle <= output.upper,
                    "Middle {} > Upper {}", output.middle, output.upper);
                assert!(output.width >= 0.0, "Width should be >= 0");
            }
        }
    }

    #[test]
    fn test_donchian_constant_range() {
        let config = DonchianConfig::new(3);
        let mut donchian = DonchianChannel::<f64>::new(config);

        for _ in 0..10 {
            let bar = Bar::new(100.0, 110.0, 90.0, 100.0, 1000.0);
            if let Ok(Some(output)) = donchian.update(&bar) {
                assert_relative_eq!(output.upper, 110.0, epsilon = 1e-8);
                assert_relative_eq!(output.lower, 90.0, epsilon = 1e-8);
                assert_relative_eq!(output.middle, 100.0, epsilon = 1e-8);
                assert_relative_eq!(output.width, 20.0, epsilon = 1e-8);
            }
        }
    }
}
