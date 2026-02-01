//! Bollinger Bands indicator.
//!
//! Bollinger Bands are volatility bands placed above and below a moving average.

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

/// Configuration for Bollinger Bands.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BollingerConfig {
    /// The window size for the moving average (default: 20).
    pub window: usize,
    /// Number of standard deviations for bands (default: 2.0).
    pub num_std: f64,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for BollingerConfig {
    fn default() -> Self {
        Self {
            window: 20,
            num_std: 2.0,
            fillna: false,
        }
    }
}

impl BollingerConfig {
    /// Create a new Bollinger Bands configuration.
    pub fn new(window: usize, num_std: f64) -> Self {
        Self {
            window,
            num_std,
            fillna: false,
        }
    }

    /// Set fillna option.
    pub fn with_fillna(mut self, fillna: bool) -> Self {
        self.fillna = fillna;
        self
    }
}

/// Output of the Bollinger Bands indicator.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct BollingerOutput<T: TaFloat> {
    /// Upper band (middle + k * std).
    pub upper: T,
    /// Middle band (SMA).
    pub middle: T,
    /// Lower band (middle - k * std).
    pub lower: T,
    /// Band width: (upper - lower) / middle.
    pub width: T,
    /// %B: (price - lower) / (upper - lower).
    pub pct_b: T,
}

impl<T: TaFloat> BollingerOutput<T> {
    /// Create a new Bollinger output.
    pub fn new(upper: T, middle: T, lower: T, width: T, pct_b: T) -> Self {
        Self {
            upper,
            middle,
            lower,
            width,
            pct_b,
        }
    }

    /// Check if any component is NaN.
    pub fn is_nan(&self) -> bool {
        self.upper.is_nan()
            || self.middle.is_nan()
            || self.lower.is_nan()
            || self.width.is_nan()
            || self.pct_b.is_nan()
    }
}

/// State for Bollinger Bands (for serialization/deserialization).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct BollingerState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// Price buffer values.
    pub buffer: Vec<T>,
    /// Running sum.
    pub sum: T,
    /// Running sum of squares (for variance).
    pub sum_sq: T,
    /// Number of values seen.
    pub count: usize,
    /// Last price (for %B calculation).
    pub last_price: T,
}

impl<T: TaFloat> Default for BollingerState<T> {
    fn default() -> Self {
        Self {
            version: 1,
            buffer: Vec::new(),
            sum: T::ZERO,
            sum_sq: T::ZERO,
            count: 0,
            last_price: T::NAN,
        }
    }
}

/// Bollinger Bands series output.
#[derive(Debug, Clone)]
pub struct BollingerSeries<T: TaFloat> {
    /// Upper band series.
    pub upper: Series<T>,
    /// Middle band series.
    pub middle: Series<T>,
    /// Lower band series.
    pub lower: Series<T>,
    /// Band width series.
    pub width: Series<T>,
    /// %B series.
    pub pct_b: Series<T>,
}

/// Bollinger Bands indicator.
///
/// # Formula
///
/// Middle = SMA(Price, n)
/// StdDev = sqrt(sum((Price - SMA)^2) / (n-1))  // Sample std (ddof=1)
/// Upper = Middle + k * StdDev
/// Lower = Middle - k * StdDev
/// Width = (Upper - Lower) / Middle  (if Middle=0 → NaN)
/// %B = (Price - Lower) / (Upper - Lower)  (if Upper=Lower → 0.5)
#[derive(Debug, Clone)]
pub struct BollingerBands<T: TaFloat> {
    config: BollingerConfig,
    /// Price buffer
    buffer: RingBuffer<T>,
    /// Running sum
    sum: T,
    /// Running sum of squares
    sum_sq: T,
    /// Number of values seen
    count: usize,
    /// Last price for %B
    last_price: T,
    /// Current output
    current_output: Option<BollingerOutput<T>>,
}

impl<T: TaFloat> BollingerBands<T> {
    /// Returns the current output if ready.
    pub fn output(&self) -> Option<BollingerOutput<T>> {
        self.current_output
    }

    /// Calculate Bollinger output from buffer.
    fn calculate_output(&self, price: T) -> BollingerOutput<T> {
        let n = <T as TaFloat>::from_usize(self.config.window);
        let k = <T as TaFloat>::from_f64_lossy(self.config.num_std);
        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

        // Mean (middle band)
        let mean = self.sum / n;

        // Sample variance: sum((x - mean)^2) / (n-1)
        // = (sum_sq - n * mean^2) / (n-1)
        let variance = (self.sum_sq - n * mean * mean) / (n - T::ONE);

        // Standard deviation
        let std_dev = if variance > T::ZERO {
            variance.sqrt()
        } else {
            T::ZERO
        };

        // Upper and lower bands
        let upper = mean + k * std_dev;
        let lower = mean - k * std_dev;

        // Band width: (upper - lower) / middle
        let width = if mean.abs() > epsilon {
            (upper - lower) / mean
        } else {
            T::NAN
        };

        // %B: (price - lower) / (upper - lower)
        let band_range = upper - lower;
        let pct_b = if band_range.abs() > epsilon {
            (price - lower) / band_range
        } else {
            // When upper = lower (no volatility), return 0.5 (neutral)
            <T as TaFloat>::from_f64_lossy(0.5)
        };

        BollingerOutput::new(upper, mean, lower, width, pct_b)
    }
}

impl<T: TaFloat> Indicator<T> for BollingerBands<T> {
    type Output = BollingerSeries<T>;
    type Config = BollingerConfig;
    type State = BollingerState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            // Use max(1) to prevent panic on window=0; calculate() will validate
            buffer: RingBuffer::new(config.window.max(1)),
            config,
            sum: T::ZERO,
            sum_sq: T::ZERO,
            count: 0,
            last_price: T::NAN,
            current_output: None,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.window
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let close = data.close();
        let len = close.len();
        let window = self.config.window;
        let k = <T as TaFloat>::from_f64_lossy(self.config.num_std);

        if window == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        if self.config.num_std <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_std",
                value: self.config.num_std.to_string(),
                expected: "positive number",
            });
        }

        let mut upper = Series::with_capacity(len);
        let mut middle = Series::with_capacity(len);
        let mut lower = Series::with_capacity(len);
        let mut width = Series::with_capacity(len);
        let mut pct_b = Series::with_capacity(len);

        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);
        let n = <T as TaFloat>::from_usize(window);

        // Running sum and sum of squares for O(n) sliding window
        let mut sum = T::ZERO;
        let mut sum_sq = T::ZERO;

        for i in 0..len {
            let price = close[i];

            // Add new value to running sums
            sum = sum + price;
            sum_sq = sum_sq + price * price;

            // Remove oldest value when window is full
            if i >= window {
                let oldest = close[i - window];
                sum = sum - oldest;
                sum_sq = sum_sq - oldest * oldest;
            }

            if i + 1 < window {
                // Not enough data
                if self.config.fillna {
                    upper.push(T::ZERO);
                    middle.push(T::ZERO);
                    lower.push(T::ZERO);
                    width.push(T::ZERO);
                    pct_b.push(T::ZERO);
                } else {
                    upper.push(T::NAN);
                    middle.push(T::NAN);
                    lower.push(T::NAN);
                    width.push(T::NAN);
                    pct_b.push(T::NAN);
                }
            } else {
                let mean = sum / n;

                // Sample variance (ddof=1)
                let variance = (sum_sq - n * mean * mean) / (n - T::ONE);
                let std_dev = if variance > T::ZERO {
                    variance.sqrt()
                } else {
                    T::ZERO
                };

                let u = mean + k * std_dev;
                let l = mean - k * std_dev;

                upper.push(u);
                middle.push(mean);
                lower.push(l);

                // Width
                let w = if mean.abs() > epsilon {
                    (u - l) / mean
                } else {
                    T::NAN
                };
                width.push(w);

                // %B
                let band_range = u - l;
                let pb = if band_range.abs() > epsilon {
                    (price - l) / band_range
                } else {
                    <T as TaFloat>::from_f64_lossy(0.5)
                };
                pct_b.push(pb);
            }
        }

        Ok(BollingerSeries {
            upper,
            middle,
            lower,
            width,
            pct_b,
        })
    }

    fn get_state(&self) -> Self::State {
        BollingerState {
            version: 1,
            buffer: self.buffer.iter().copied().collect(),
            sum: self.sum,
            sum_sq: self.sum_sq,
            count: self.count,
            last_price: self.last_price,
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
        self.sum_sq = state.sum_sq;
        self.count = state.count;
        self.last_price = state.last_price;

        // Reconstruct current output
        if self.buffer.is_full() {
            self.current_output = Some(self.calculate_output(self.last_price));
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.buffer = RingBuffer::new(self.config.window);
        self.sum = T::ZERO;
        self.sum_sq = T::ZERO;
        self.count = 0;
        self.last_price = T::NAN;
        self.current_output = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for BollingerBands<T> {
    type StreamingOutput = Option<BollingerOutput<T>>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<BollingerOutput<T>>> {
        let price = bar.close;
        self.count += 1;
        self.last_price = price;

        // Remove oldest value if buffer is full
        if self.buffer.is_full() {
            if let Some(&oldest) = self.buffer.oldest() {
                self.sum = self.sum - oldest;
                self.sum_sq = self.sum_sq - oldest * oldest;
            }
        }

        // Add new value
        self.buffer.push(price);
        self.sum = self.sum + price;
        self.sum_sq = self.sum_sq + price * price;

        // Calculate output if buffer is full
        if self.buffer.is_full() {
            let output = self.calculate_output(price);
            self.current_output = Some(output);
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }

    fn current(&self) -> Option<BollingerOutput<T>> {
        self.current_output
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
    fn test_bollinger_default_config() {
        let config = BollingerConfig::default();
        assert_eq!(config.window, 20);
        assert_relative_eq!(config.num_std, 2.0, epsilon = 1e-10);
        assert!(!config.fillna);
    }

    #[test]
    fn test_bollinger_calculate() {
        let config = BollingerConfig::new(5, 2.0);
        let bb = BollingerBands::<f64>::new(config);

        let closes = vec![10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0];
        let ohlcv = create_test_ohlcv(&closes);
        let result = bb.calculate(&ohlcv).unwrap();

        assert_eq!(result.upper.len(), 10);
        assert_eq!(result.middle.len(), 10);
        assert_eq!(result.lower.len(), 10);

        // First 4 should be NaN
        assert!(result.middle[0].is_nan());
        assert!(result.middle[3].is_nan());

        // 5th value should be valid
        assert!(!result.middle[4].is_nan());
    }

    #[test]
    fn test_bollinger_band_order() {
        // Lower ≤ Middle ≤ Upper should always hold
        let config = BollingerConfig::new(5, 2.0);
        let mut bb = BollingerBands::<f64>::new(config);

        let closes = vec![10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0];

        for &close in &closes {
            let bar = Bar::new(close, close, close, close, 1000.0);
            if let Ok(Some(output)) = bb.update(&bar) {
                assert!(
                    output.lower <= output.middle,
                    "Lower {} > Middle {}",
                    output.lower,
                    output.middle
                );
                assert!(
                    output.middle <= output.upper,
                    "Middle {} > Upper {}",
                    output.middle,
                    output.upper
                );
            }
        }
    }

    #[test]
    fn test_bollinger_constant_price() {
        // With constant price, bands should collapse to the price
        let config = BollingerConfig::new(5, 2.0);
        let mut bb = BollingerBands::<f64>::new(config);

        for _ in 0..10 {
            let bar = Bar::new(100.0, 100.0, 100.0, 100.0, 1000.0);
            if let Ok(Some(output)) = bb.update(&bar) {
                assert_relative_eq!(output.middle, 100.0, epsilon = 1e-10);
                assert_relative_eq!(output.upper, 100.0, epsilon = 1e-10);
                assert_relative_eq!(output.lower, 100.0, epsilon = 1e-10);
                assert_relative_eq!(output.pct_b, 0.5, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_bollinger_streaming_equals_batch() {
        let config = BollingerConfig::new(5, 2.0);
        let closes = vec![10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0];
        let ohlcv = create_test_ohlcv(&closes);

        // Batch calculation
        let batch_bb = BollingerBands::<f64>::new(config.clone());
        let batch_result = batch_bb.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_bb = BollingerBands::<f64>::new(config);
        for (i, &close) in closes.iter().enumerate() {
            let bar = Bar::new(close, close, close, close, 1000.0);
            let streaming_result = streaming_bb.update(&bar).unwrap();

            if let Some(output) = streaming_result {
                assert_relative_eq!(output.upper, batch_result.upper[i], epsilon = 1e-8);
                assert_relative_eq!(output.middle, batch_result.middle[i], epsilon = 1e-8);
                assert_relative_eq!(output.lower, batch_result.lower[i], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_bollinger_width() {
        let config = BollingerConfig::new(5, 2.0);
        let bb = BollingerBands::<f64>::new(config);

        let closes = vec![10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0];
        let ohlcv = create_test_ohlcv(&closes);
        let result = bb.calculate(&ohlcv).unwrap();

        // Width should equal (upper - lower) / middle
        for i in 0..closes.len() {
            if !result.middle[i].is_nan() && !result.width[i].is_nan() {
                let expected_width = (result.upper[i] - result.lower[i]) / result.middle[i];
                assert_relative_eq!(result.width[i], expected_width, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_bollinger_pct_b() {
        let config = BollingerConfig::new(5, 2.0);
        let bb = BollingerBands::<f64>::new(config);

        let closes = vec![10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0];
        let ohlcv = create_test_ohlcv(&closes);
        let result = bb.calculate(&ohlcv).unwrap();

        // %B should equal (price - lower) / (upper - lower)
        for i in 0..closes.len() {
            if !result.pct_b[i].is_nan() {
                let band_range = result.upper[i] - result.lower[i];
                if band_range.abs() > 1e-10 {
                    let expected_pct_b = (closes[i] - result.lower[i]) / band_range;
                    assert_relative_eq!(result.pct_b[i], expected_pct_b, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_bollinger_state_roundtrip() {
        let config = BollingerConfig::new(5, 2.0);
        let mut bb1 = BollingerBands::<f64>::new(config.clone());

        // Feed data
        let closes = vec![10.0, 11.0, 12.0, 11.0, 10.0, 11.0];
        for &close in &closes {
            bb1.update(&Bar::new(close, close, close, close, 1000.0))
                .unwrap();
        }

        // Get state
        let state = bb1.get_state();

        // Create new indicator and restore state
        let mut bb2 = BollingerBands::<f64>::new(config);
        bb2.set_state(state).unwrap();

        // Both should produce the same result
        let result1 = bb1
            .update(&Bar::new(12.0, 12.0, 12.0, 12.0, 1000.0))
            .unwrap();
        let result2 = bb2
            .update(&Bar::new(12.0, 12.0, 12.0, 12.0, 1000.0))
            .unwrap();

        let out1 = result1.unwrap();
        let out2 = result2.unwrap();
        assert_relative_eq!(out1.upper, out2.upper, epsilon = 1e-10);
        assert_relative_eq!(out1.middle, out2.middle, epsilon = 1e-10);
        assert_relative_eq!(out1.lower, out2.lower, epsilon = 1e-10);
    }

    #[test]
    fn test_bollinger_reset() {
        let config = BollingerConfig::new(5, 2.0);
        let mut bb = BollingerBands::<f64>::new(config);

        // Feed data
        let closes = vec![10.0, 11.0, 12.0, 11.0, 10.0];
        for &close in &closes {
            bb.update(&Bar::new(close, close, close, close, 1000.0))
                .unwrap();
        }

        assert!(bb.is_ready());

        // Reset
        bb.reset();

        assert!(!bb.is_ready());
        assert_eq!(bb.count, 0);
    }

    #[test]
    fn test_bollinger_invalid_window() {
        let config = BollingerConfig::new(0, 2.0);
        let bb = BollingerBands::<f64>::new(config);

        let ohlcv = create_test_ohlcv(&[1.0, 2.0, 3.0]);
        let result = bb.calculate(&ohlcv);

        assert!(result.is_err());
    }

    #[test]
    fn test_bollinger_invalid_num_std() {
        let config = BollingerConfig::new(5, -1.0);
        let bb = BollingerBands::<f64>::new(config);

        let ohlcv = create_test_ohlcv(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = bb.calculate(&ohlcv);

        assert!(result.is_err());
    }
}
