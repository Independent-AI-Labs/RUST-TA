//! Average True Range (ATR) indicator.
//!
//! ATR measures market volatility using True Range with Wilder's smoothing.

use ta_core::{
    error::{IndicatorError, Result},
    num::TaFloat,
    ohlcv::{Bar, OhlcvSeries},
    series::Series,
    traits::{Indicator, StreamingIndicator},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the ATR indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AtrConfig {
    /// The lookback period (default: 14).
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for AtrConfig {
    fn default() -> Self {
        Self {
            window: 14,
            fillna: false,
        }
    }
}

impl AtrConfig {
    /// Create a new ATR configuration with the given window.
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

/// State for the ATR indicator (for serialization/deserialization).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct AtrState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// Current ATR value.
    pub atr_value: T,
    /// Previous close price.
    pub prev_close: T,
    /// Number of values seen.
    pub count: usize,
    /// Sum of True Ranges during initialization.
    pub init_tr_sum: T,
    /// Whether initialization is complete.
    pub initialized: bool,
}

impl<T: TaFloat> Default for AtrState<T> {
    fn default() -> Self {
        Self {
            version: 1,
            atr_value: T::ZERO,
            prev_close: T::NAN,
            count: 0,
            init_tr_sum: T::ZERO,
            initialized: false,
        }
    }
}

/// Average True Range indicator.
///
/// ATR measures volatility using True Range, which accounts for gaps.
///
/// # Formula
///
/// True Range = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
///
/// For the first bar (no previous close): TR = High - Low
///
/// ATR is the Wilder-smoothed average of True Range.
#[derive(Debug, Clone)]
pub struct Atr<T: TaFloat> {
    config: AtrConfig,
    /// Current ATR value
    atr_value: T,
    /// Previous close price
    prev_close: T,
    /// Number of values seen
    count: usize,
    /// Sum of True Ranges during initialization
    init_tr_sum: T,
    /// Whether initialization is complete
    initialized: bool,
}

impl<T: TaFloat> Atr<T> {
    /// Returns the current ATR value if ready.
    pub fn value(&self) -> Option<T> {
        if self.initialized {
            Some(self.atr_value)
        } else {
            None
        }
    }

    /// Calculate True Range.
    ///
    /// For the first bar (no previous close): TR = High - Low
    /// Otherwise: TR = max(H-L, |H-Prev Close|, |L-Prev Close|)
    pub fn true_range(high: T, low: T, prev_close: Option<T>) -> T {
        let high_low = high - low;

        match prev_close {
            Some(pc) => {
                let high_pc = (high - pc).abs();
                let low_pc = (low - pc).abs();
                high_low.max(high_pc).max(low_pc)
            }
            None => high_low,
        }
    }
}

impl<T: TaFloat> Indicator<T> for Atr<T> {
    type Output = Series<T>;
    type Config = AtrConfig;
    type State = AtrState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            config,
            atr_value: T::ZERO,
            prev_close: T::NAN,
            count: 0,
            init_tr_sum: T::ZERO,
            initialized: false,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.window + 1
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();
        let window = self.config.window;

        if window == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        let high = data.high();
        let low = data.low();
        let close = data.close();

        let mut result = Series::with_capacity(len);

        if len == 0 {
            return Ok(result);
        }

        // Calculate True Range series
        let mut tr = Vec::with_capacity(len);

        // First bar: TR = High - Low
        tr.push(high[0] - low[0]);

        for i in 1..len {
            tr.push(Self::true_range(high[i], low[i], Some(close[i - 1])));
        }

        // Calculate ATR using Wilder's smoothing
        let mut atr = T::ZERO;
        let mut initialized = false;

        for i in 0..len {
            if i + 1 < window {
                // During initialization
                if self.config.fillna {
                    result.push(T::ZERO);
                } else {
                    result.push(T::NAN);
                }
            } else if i + 1 == window {
                // First ATR: SMA of first window TR values
                let mut sum = T::ZERO;
                for j in 0..window {
                    sum = sum + tr[j];
                }
                atr = sum / <T as TaFloat>::from_usize(window);
                initialized = true;
                result.push(atr);
            } else {
                // Subsequent ATR: Wilder's smoothing
                // ATR = (prev_ATR * (n-1) + TR) / n
                atr = (atr * (<T as TaFloat>::from_usize(window) - T::ONE) + tr[i]) / <T as TaFloat>::from_usize(window);
                result.push(atr);
            }
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        AtrState {
            version: 1,
            atr_value: self.atr_value,
            prev_close: self.prev_close,
            count: self.count,
            init_tr_sum: self.init_tr_sum,
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

        self.atr_value = state.atr_value;
        self.prev_close = state.prev_close;
        self.count = state.count;
        self.init_tr_sum = state.init_tr_sum;
        self.initialized = state.initialized;

        Ok(())
    }

    fn reset(&mut self) {
        self.atr_value = T::ZERO;
        self.prev_close = T::NAN;
        self.count = 0;
        self.init_tr_sum = T::ZERO;
        self.initialized = false;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Atr<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        self.count += 1;

        // Calculate True Range
        let tr = if self.prev_close.is_nan() {
            // First bar: no previous close
            bar.high - bar.low
        } else {
            Self::true_range(bar.high, bar.low, Some(self.prev_close))
        };

        self.prev_close = bar.close;

        let window = self.config.window;

        if !self.initialized {
            // Accumulate for initialization
            self.init_tr_sum = self.init_tr_sum + tr;

            if self.count >= window {
                // Initialize with SMA
                self.atr_value = self.init_tr_sum / <T as TaFloat>::from_usize(window);
                self.initialized = true;
                return Ok(Some(self.atr_value));
            }

            return Ok(None);
        }

        // Wilder's smoothing update
        self.atr_value =
            (self.atr_value * (<T as TaFloat>::from_usize(window) - T::ONE) + tr) / <T as TaFloat>::from_usize(window);

        Ok(Some(self.atr_value))
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

    fn create_test_ohlcv(data: &[(f64, f64, f64, f64)]) -> OhlcvSeries<f64> {
        let mut ohlcv = OhlcvSeries::new();
        for &(open, high, low, close) in data {
            ohlcv.push(Bar::new(open, high, low, close, 1000.0));
        }
        ohlcv
    }

    #[test]
    fn test_atr_default_config() {
        let config = AtrConfig::default();
        assert_eq!(config.window, 14);
        assert!(!config.fillna);
    }

    #[test]
    fn test_true_range_no_gap() {
        // When there's no gap, TR = High - Low
        let tr = Atr::<f64>::true_range(105.0, 100.0, Some(102.0));
        assert_relative_eq!(tr, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_true_range_gap_up() {
        // Gap up: previous close was 100, current high is 110
        let tr = Atr::<f64>::true_range(110.0, 108.0, Some(100.0));
        assert_relative_eq!(tr, 10.0, epsilon = 1e-10); // |110-100| = 10
    }

    #[test]
    fn test_true_range_gap_down() {
        // Gap down: previous close was 110, current low is 100
        let tr = Atr::<f64>::true_range(102.0, 100.0, Some(110.0));
        assert_relative_eq!(tr, 10.0, epsilon = 1e-10); // |100-110| = 10
    }

    #[test]
    fn test_true_range_first_bar() {
        // First bar has no previous close
        let tr = Atr::<f64>::true_range(105.0, 100.0, None);
        assert_relative_eq!(tr, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_atr_calculate() {
        let config = AtrConfig::new(3);
        let atr = Atr::<f64>::new(config);

        let data = vec![
            (100.0, 105.0, 98.0, 103.0),
            (103.0, 108.0, 101.0, 106.0),
            (106.0, 110.0, 104.0, 108.0),
            (108.0, 112.0, 106.0, 110.0),
            (110.0, 115.0, 109.0, 113.0),
        ];
        let ohlcv = create_test_ohlcv(&data);
        let result = atr.calculate(&ohlcv).unwrap();

        assert_eq!(result.len(), 5);

        // First 2 should be NaN (window-1)
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // Third value should be valid
        assert!(!result[2].is_nan());
        assert!(result[2] > 0.0);

        // ATR should always be positive
        for val in result.iter().skip(2) {
            assert!(*val >= 0.0, "ATR should be non-negative");
        }
    }

    #[test]
    fn test_atr_streaming() {
        let config = AtrConfig::new(3);
        let mut atr = Atr::<f64>::new(config);

        let data = vec![
            (100.0, 105.0, 98.0, 103.0),
            (103.0, 108.0, 101.0, 106.0),
            (106.0, 110.0, 104.0, 108.0),
            (108.0, 112.0, 106.0, 110.0),
            (110.0, 115.0, 109.0, 113.0),
        ];

        let mut results = Vec::new();
        for &(open, high, low, close) in &data {
            let bar = Bar::new(open, high, low, close, 1000.0);
            let result = atr.update(&bar).unwrap();
            results.push(result);
        }

        // First 2 should be None
        assert!(results[0].is_none());
        assert!(results[1].is_none());

        // Third and later should have values
        assert!(results[2].is_some());
        assert!(results[3].is_some());
        assert!(results[4].is_some());
    }

    #[test]
    fn test_atr_streaming_equals_batch() {
        let config = AtrConfig::new(3);
        let data = vec![
            (100.0, 105.0, 98.0, 103.0),
            (103.0, 108.0, 101.0, 106.0),
            (106.0, 110.0, 104.0, 108.0),
            (108.0, 112.0, 106.0, 110.0),
            (110.0, 115.0, 109.0, 113.0),
            (113.0, 118.0, 112.0, 116.0),
            (116.0, 120.0, 114.0, 118.0),
        ];
        let ohlcv = create_test_ohlcv(&data);

        // Batch calculation
        let batch_atr = Atr::<f64>::new(config.clone());
        let batch_result = batch_atr.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_atr = Atr::<f64>::new(config);
        for (i, &(open, high, low, close)) in data.iter().enumerate() {
            let bar = Bar::new(open, high, low, close, 1000.0);
            let streaming_result = streaming_atr.update(&bar).unwrap();

            if let Some(val) = streaming_result {
                assert_relative_eq!(val, batch_result[i], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_atr_always_positive() {
        let config = AtrConfig::new(5);
        let mut atr = Atr::<f64>::new(config);

        // Various price movements
        let data = vec![
            (100.0, 105.0, 95.0, 102.0),
            (102.0, 110.0, 98.0, 105.0),
            (105.0, 108.0, 100.0, 101.0),
            (101.0, 103.0, 95.0, 96.0),
            (96.0, 100.0, 90.0, 98.0),
            (98.0, 105.0, 97.0, 104.0),
            (104.0, 110.0, 103.0, 108.0),
        ];

        for &(open, high, low, close) in &data {
            let bar = Bar::new(open, high, low, close, 1000.0);
            if let Ok(Some(value)) = atr.update(&bar) {
                assert!(value >= 0.0, "ATR {} should be non-negative", value);
            }
        }
    }

    #[test]
    fn test_atr_state_roundtrip() {
        let config = AtrConfig::new(3);
        let mut atr1 = Atr::<f64>::new(config.clone());

        // Feed some data
        let data = vec![
            (100.0, 105.0, 98.0, 103.0),
            (103.0, 108.0, 101.0, 106.0),
            (106.0, 110.0, 104.0, 108.0),
            (108.0, 112.0, 106.0, 110.0),
        ];
        for &(open, high, low, close) in &data {
            atr1.update(&Bar::new(open, high, low, close, 1000.0))
                .unwrap();
        }

        // Get state
        let state = atr1.get_state();

        // Create new indicator and restore state
        let mut atr2 = Atr::<f64>::new(config);
        atr2.set_state(state).unwrap();

        // Both should produce the same result
        let next_bar = Bar::new(110.0, 115.0, 109.0, 113.0, 1000.0);
        let result1 = atr1.update(&next_bar).unwrap();
        let result2 = atr2.update(&next_bar).unwrap();

        assert_relative_eq!(result1.unwrap(), result2.unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_atr_reset() {
        let config = AtrConfig::new(3);
        let mut atr = Atr::<f64>::new(config);

        // Feed some data
        let data = vec![
            (100.0, 105.0, 98.0, 103.0),
            (103.0, 108.0, 101.0, 106.0),
            (106.0, 110.0, 104.0, 108.0),
        ];
        for &(open, high, low, close) in &data {
            atr.update(&Bar::new(open, high, low, close, 1000.0))
                .unwrap();
        }

        assert!(atr.is_ready());

        // Reset
        atr.reset();

        assert!(!atr.is_ready());
        assert!(atr.prev_close.is_nan());
    }

    #[test]
    fn test_atr_min_periods() {
        let config = AtrConfig::new(14);
        let atr = Atr::<f64>::new(config);

        assert_eq!(atr.min_periods(), 15);
    }
}
