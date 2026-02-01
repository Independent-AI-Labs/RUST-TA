//! Aroon indicator.
//!
//! Aroon identifies trend changes and trend strength.

use std::collections::VecDeque;

use ta_core::{
    error::{IndicatorError, Result},
    num::TaFloat,
    ohlcv::{Bar, OhlcvSeries},
    series::Series,
    traits::{Indicator, StreamingIndicator},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for Aroon.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AroonConfig {
    /// Lookback period (default: 25).
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for AroonConfig {
    fn default() -> Self {
        Self {
            window: 25,
            fillna: false,
        }
    }
}

impl AroonConfig {
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

/// Output of Aroon indicator.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct AroonOutput<T: TaFloat> {
    /// Aroon Up (0-100).
    pub aroon_up: T,
    /// Aroon Down (0-100).
    pub aroon_down: T,
    /// Aroon Oscillator (-100 to 100).
    pub oscillator: T,
}

impl<T: TaFloat> AroonOutput<T> {
    /// Create a new output.
    pub fn new(aroon_up: T, aroon_down: T) -> Self {
        Self {
            aroon_up,
            aroon_down,
            oscillator: aroon_up - aroon_down,
        }
    }
}

/// State for Aroon.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct AroonState<T: TaFloat> {
    /// Version tag.
    pub version: u32,
    /// High buffer with indices.
    pub high_buffer: Vec<(usize, T)>,
    /// Low buffer with indices.
    pub low_buffer: Vec<(usize, T)>,
    /// Current index.
    pub index: usize,
    /// Count of values.
    pub count: usize,
}

/// Aroon series output.
#[derive(Debug, Clone)]
pub struct AroonSeries<T: TaFloat> {
    /// Aroon Up series.
    pub aroon_up: Series<T>,
    /// Aroon Down series.
    pub aroon_down: Series<T>,
    /// Aroon Oscillator series.
    pub oscillator: Series<T>,
}

/// Aroon indicator.
///
/// # Formula
///
/// Aroon Up = 100 * (window - periods since highest high) / window
/// Aroon Down = 100 * (window - periods since lowest low) / window
/// Aroon Oscillator = Aroon Up - Aroon Down
#[derive(Debug, Clone)]
pub struct Aroon<T: TaFloat> {
    config: AroonConfig,
    /// Buffer storing (index, high) pairs
    high_buffer: VecDeque<(usize, T)>,
    /// Buffer storing (index, low) pairs
    low_buffer: VecDeque<(usize, T)>,
    /// Current index
    index: usize,
    count: usize,
    current_output: Option<AroonOutput<T>>,
}

impl<T: TaFloat> Aroon<T> {
    /// Get current output.
    pub fn output(&self) -> Option<AroonOutput<T>> {
        self.current_output
    }

    fn find_highest_idx(&self) -> usize {
        let mut max_val = T::NEG_INFINITY;
        let mut max_idx = 0;
        for &(idx, val) in self.high_buffer.iter() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }
        max_idx
    }

    fn find_lowest_idx(&self) -> usize {
        let mut min_val = T::INFINITY;
        let mut min_idx = 0;
        for &(idx, val) in self.low_buffer.iter() {
            if val < min_val {
                min_val = val;
                min_idx = idx;
            }
        }
        min_idx
    }
}

impl<T: TaFloat> Indicator<T> for Aroon<T> {
    type Output = AroonSeries<T>;
    type Config = AroonConfig;
    type State = AroonState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            high_buffer: VecDeque::with_capacity(config.window + 1),
            low_buffer: VecDeque::with_capacity(config.window + 1),
            config,
            index: 0,
            count: 0,
            current_output: None,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.window + 1
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();
        let high = data.high();
        let low = data.low();
        let window = self.config.window;

        if window == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        let mut aroon_up = Series::with_capacity(len);
        let mut aroon_down = Series::with_capacity(len);
        let mut oscillator = Series::with_capacity(len);

        for i in 0..len {
            if i < window {
                if self.config.fillna {
                    aroon_up.push(T::ZERO);
                    aroon_down.push(T::ZERO);
                    oscillator.push(T::ZERO);
                } else {
                    aroon_up.push(T::NAN);
                    aroon_down.push(T::NAN);
                    oscillator.push(T::NAN);
                }
            } else {
                // Find periods since highest high and lowest low
                let start = i - window;
                let mut max_val = high[start];
                let mut max_idx = start;
                let mut min_val = low[start];
                let mut min_idx = start;

                for j in start..=i {
                    if high[j] >= max_val {
                        max_val = high[j];
                        max_idx = j;
                    }
                    if low[j] <= min_val {
                        min_val = low[j];
                        min_idx = j;
                    }
                }

                let periods_since_high = i - max_idx;
                let periods_since_low = i - min_idx;

                let n = <T as TaFloat>::from_usize(window);
                let up = T::HUNDRED * (n - <T as TaFloat>::from_usize(periods_since_high)) / n;
                let down = T::HUNDRED * (n - <T as TaFloat>::from_usize(periods_since_low)) / n;

                aroon_up.push(up);
                aroon_down.push(down);
                oscillator.push(up - down);
            }
        }

        Ok(AroonSeries {
            aroon_up,
            aroon_down,
            oscillator,
        })
    }

    fn get_state(&self) -> Self::State {
        AroonState {
            version: 1,
            high_buffer: self.high_buffer.iter().copied().collect(),
            low_buffer: self.low_buffer.iter().copied().collect(),
            index: self.index,
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

        self.high_buffer = VecDeque::with_capacity(self.config.window + 1);
        for v in state.high_buffer {
            self.high_buffer.push_back(v);
        }

        self.low_buffer = VecDeque::with_capacity(self.config.window + 1);
        for v in state.low_buffer {
            self.low_buffer.push_back(v);
        }

        self.index = state.index;
        self.count = state.count;
        Ok(())
    }

    fn reset(&mut self) {
        self.high_buffer = VecDeque::with_capacity(self.config.window + 1);
        self.low_buffer = VecDeque::with_capacity(self.config.window + 1);
        self.index = 0;
        self.count = 0;
        self.current_output = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Aroon<T> {
    type StreamingOutput = Option<AroonOutput<T>>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<AroonOutput<T>>> {
        let window_size = self.config.window + 1;

        // Remove oldest if buffer is at capacity
        if self.high_buffer.len() >= window_size {
            self.high_buffer.pop_front();
        }
        if self.low_buffer.len() >= window_size {
            self.low_buffer.pop_front();
        }

        self.high_buffer.push_back((self.index, bar.high));
        self.low_buffer.push_back((self.index, bar.low));
        self.index += 1;
        self.count += 1;

        if self.high_buffer.len() < window_size {
            return Ok(None);
        }

        let max_idx = self.find_highest_idx();
        let min_idx = self.find_lowest_idx();

        let current_idx = self.index - 1;
        let periods_since_high = current_idx - max_idx;
        let periods_since_low = current_idx - min_idx;

        let n = <T as TaFloat>::from_usize(self.config.window);
        let up = T::HUNDRED * (n - <T as TaFloat>::from_usize(periods_since_high)) / n;
        let down = T::HUNDRED * (n - <T as TaFloat>::from_usize(periods_since_low)) / n;

        let output = AroonOutput::new(up, down);
        self.current_output = Some(output);
        Ok(Some(output))
    }

    fn current(&self) -> Option<AroonOutput<T>> {
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
    fn test_aroon_default_config() {
        let config = AroonConfig::default();
        assert_eq!(config.window, 25);
    }

    #[test]
    fn test_aroon_bounds() {
        let config = AroonConfig::new(5);
        let mut aroon = Aroon::<f64>::new(config);

        for i in 0..20 {
            let price = 100.0 + (i as f64 * 0.3).sin() * 10.0;
            let bar = Bar::new(price, price + 2.0, price - 2.0, price, 1000.0);
            if let Ok(Some(output)) = aroon.update(&bar) {
                assert!(output.aroon_up >= 0.0 && output.aroon_up <= 100.0,
                    "Aroon Up out of bounds: {}", output.aroon_up);
                assert!(output.aroon_down >= 0.0 && output.aroon_down <= 100.0,
                    "Aroon Down out of bounds: {}", output.aroon_down);
                assert!(output.oscillator >= -100.0 && output.oscillator <= 100.0,
                    "Oscillator out of bounds: {}", output.oscillator);
            }
        }
    }

    #[test]
    fn test_aroon_at_high() {
        let config = AroonConfig::new(5);
        let mut aroon = Aroon::<f64>::new(config);

        // Price making new highs
        for i in 0..10 {
            let bar = Bar::new(100.0 + i as f64, 102.0 + i as f64, 98.0 + i as f64, 100.0 + i as f64, 1000.0);
            if let Ok(Some(output)) = aroon.update(&bar) {
                // Aroon Up should be 100 when at new high
                assert_relative_eq!(output.aroon_up, 100.0, epsilon = 1e-8);
            }
        }
    }
}
