//! Chaikin Money Flow (CMF) indicator.
//!
//! CMF measures the amount of Money Flow Volume over a specific period.

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

/// Configuration for CMF.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CmfConfig {
    /// Lookback period (default: 20).
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for CmfConfig {
    fn default() -> Self {
        Self {
            window: 20,
            fillna: false,
        }
    }
}

impl CmfConfig {
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

/// State for CMF.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct CmfState<T: TaFloat> {
    /// Version tag.
    pub version: u32,
    /// Money Flow Volume buffer.
    pub mfv_buffer: Vec<T>,
    /// Volume buffer.
    pub vol_buffer: Vec<T>,
    /// Count.
    pub count: usize,
}

/// Chaikin Money Flow indicator.
///
/// # Formula
///
/// Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
/// Money Flow Volume = MFM * Volume
/// CMF = Sum(MFV, n) / Sum(Volume, n)
///
/// Range: [-1, 1]
/// - Positive: buying pressure
/// - Negative: selling pressure
#[derive(Debug, Clone)]
pub struct Cmf<T: TaFloat> {
    config: CmfConfig,
    mfv_buffer: RingBuffer<T>,
    vol_buffer: RingBuffer<T>,
    count: usize,
    current_value: Option<T>,
}

impl<T: TaFloat> Cmf<T> {
    /// Get current value.
    pub fn value(&self) -> Option<T> {
        self.current_value
    }

    /// Calculate Close Location Value (Money Flow Multiplier).
    /// CLV = ((Close - Low) - (High - Close)) / (High - Low)
    ///     = (2*Close - High - Low) / (High - Low)
    fn clv(high: T, low: T, close: T) -> T {
        let range = high - low;
        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

        if range.abs() < epsilon {
            T::ZERO
        } else {
            (T::TWO * close - high - low) / range
        }
    }
}

impl<T: TaFloat> Indicator<T> for Cmf<T> {
    type Output = Series<T>;
    type Config = CmfConfig;
    type State = CmfState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            mfv_buffer: RingBuffer::new(config.window),
            vol_buffer: RingBuffer::new(config.window),
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
        let volume = data.volume();
        let window = self.config.window;

        if window == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        let mut result = Series::with_capacity(len);
        let mut mfv_buf: RingBuffer<T> = RingBuffer::new(window);
        let mut vol_buf: RingBuffer<T> = RingBuffer::new(window);

        for i in 0..len {
            let clv = Self::clv(high[i], low[i], close[i]);
            let mfv = clv * volume[i];

            mfv_buf.push(mfv);
            vol_buf.push(volume[i]);

            if mfv_buf.is_full() {
                let sum_mfv = mfv_buf.sum();
                let sum_vol = vol_buf.sum();
                let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

                let cmf = if sum_vol.abs() < epsilon {
                    T::ZERO
                } else {
                    sum_mfv / sum_vol
                };

                result.push(cmf);
            } else if self.config.fillna {
                result.push(T::ZERO);
            } else {
                result.push(T::NAN);
            }
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        CmfState {
            version: 1,
            mfv_buffer: self.mfv_buffer.iter().copied().collect(),
            vol_buffer: self.vol_buffer.iter().copied().collect(),
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

        self.mfv_buffer = RingBuffer::new(self.config.window);
        for v in state.mfv_buffer {
            self.mfv_buffer.push(v);
        }

        self.vol_buffer = RingBuffer::new(self.config.window);
        for v in state.vol_buffer {
            self.vol_buffer.push(v);
        }

        self.count = state.count;
        Ok(())
    }

    fn reset(&mut self) {
        self.mfv_buffer = RingBuffer::new(self.config.window);
        self.vol_buffer = RingBuffer::new(self.config.window);
        self.count = 0;
        self.current_value = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Cmf<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        self.count += 1;

        let clv = Self::clv(bar.high, bar.low, bar.close);
        let mfv = clv * bar.volume;

        self.mfv_buffer.push(mfv);
        self.vol_buffer.push(bar.volume);

        if !self.mfv_buffer.is_full() {
            return Ok(None);
        }

        let sum_mfv = self.mfv_buffer.sum();
        let sum_vol = self.vol_buffer.sum();
        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

        let cmf = if sum_vol.abs() < epsilon {
            T::ZERO
        } else {
            sum_mfv / sum_vol
        };

        self.current_value = Some(cmf);
        Ok(Some(cmf))
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
    fn test_cmf_default_config() {
        let config = CmfConfig::default();
        assert_eq!(config.window, 20);
    }

    #[test]
    fn test_cmf_bounds() {
        let config = CmfConfig::new(5);
        let mut cmf = Cmf::<f64>::new(config);

        for i in 0..20 {
            let price = 100.0 + (i as f64 * 0.3).sin() * 10.0;
            let bar = Bar::new(price, price + 2.0, price - 2.0, price, 1000.0);
            if let Ok(Some(value)) = cmf.update(&bar) {
                assert!(value >= -1.0 && value <= 1.0, "CMF out of bounds: {}", value);
            }
        }
    }

    #[test]
    fn test_cmf_close_at_high() {
        let config = CmfConfig::new(3);
        let mut cmf = Cmf::<f64>::new(config);

        // Close at high means CLV = 1, so CMF should be 1
        for _ in 0..10 {
            let bar = Bar::new(100.0, 110.0, 90.0, 110.0, 1000.0);
            if let Ok(Some(value)) = cmf.update(&bar) {
                assert_relative_eq!(value, 1.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_cmf_close_at_low() {
        let config = CmfConfig::new(3);
        let mut cmf = Cmf::<f64>::new(config);

        // Close at low means CLV = -1, so CMF should be -1
        for _ in 0..10 {
            let bar = Bar::new(100.0, 110.0, 90.0, 90.0, 1000.0);
            if let Ok(Some(value)) = cmf.update(&bar) {
                assert_relative_eq!(value, -1.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_cmf_close_at_midpoint() {
        let config = CmfConfig::new(3);
        let mut cmf = Cmf::<f64>::new(config);

        // Close at midpoint means CLV = 0, so CMF should be 0
        for _ in 0..10 {
            let bar = Bar::new(100.0, 110.0, 90.0, 100.0, 1000.0);
            if let Ok(Some(value)) = cmf.update(&bar) {
                assert_relative_eq!(value, 0.0, epsilon = 1e-8);
            }
        }
    }
}
