//! Money Flow Index (MFI) indicator.
//!
//! MFI is a volume-weighted RSI that measures buying and selling pressure.

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

/// Configuration for MFI.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MfiConfig {
    /// Lookback period (default: 14).
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for MfiConfig {
    fn default() -> Self {
        Self {
            window: 14,
            fillna: false,
        }
    }
}

impl MfiConfig {
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

/// State for MFI.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct MfiState<T: TaFloat> {
    /// Version tag.
    pub version: u32,
    /// Positive money flow buffer.
    pub pos_mf_buffer: Vec<T>,
    /// Negative money flow buffer.
    pub neg_mf_buffer: Vec<T>,
    /// Previous typical price.
    pub prev_tp: T,
    /// Count.
    pub count: usize,
}

/// Money Flow Index indicator.
///
/// # Formula
///
/// Typical Price = (High + Low + Close) / 3
/// Money Flow = Typical Price * Volume
///
/// If TP > Prev TP: Positive Money Flow += Money Flow
/// If TP < Prev TP: Negative Money Flow += Money Flow
///
/// Money Ratio = Sum(Positive MF) / Sum(Negative MF)
/// MFI = 100 - 100 / (1 + Money Ratio)
///
/// Range: [0, 100]
#[derive(Debug, Clone)]
pub struct Mfi<T: TaFloat> {
    config: MfiConfig,
    pos_mf_buffer: RingBuffer<T>,
    neg_mf_buffer: RingBuffer<T>,
    prev_tp: T,
    count: usize,
    current_value: Option<T>,
}

impl<T: TaFloat> Mfi<T> {
    /// Get current value.
    pub fn value(&self) -> Option<T> {
        self.current_value
    }

    fn typical_price(high: T, low: T, close: T) -> T {
        (high + low + close) / <T as TaFloat>::from_f64_lossy(3.0)
    }
}

impl<T: TaFloat> Indicator<T> for Mfi<T> {
    type Output = Series<T>;
    type Config = MfiConfig;
    type State = MfiState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            pos_mf_buffer: RingBuffer::new(config.window),
            neg_mf_buffer: RingBuffer::new(config.window),
            config,
            prev_tp: T::NAN,
            count: 0,
            current_value: None,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.window + 1
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

        if len == 0 {
            return Ok(result);
        }

        // Calculate typical prices and money flows
        let mut tp_vals = Vec::with_capacity(len);
        let mut pos_mf = Vec::with_capacity(len);
        let mut neg_mf = Vec::with_capacity(len);

        for i in 0..len {
            let tp = Self::typical_price(high[i], low[i], close[i]);
            tp_vals.push(tp);

            if i == 0 {
                pos_mf.push(T::ZERO);
                neg_mf.push(T::ZERO);
            } else {
                let mf = tp * volume[i];
                if tp > tp_vals[i - 1] {
                    pos_mf.push(mf);
                    neg_mf.push(T::ZERO);
                } else if tp < tp_vals[i - 1] {
                    pos_mf.push(T::ZERO);
                    neg_mf.push(mf);
                } else {
                    pos_mf.push(T::ZERO);
                    neg_mf.push(T::ZERO);
                }
            }
        }

        // Calculate MFI
        let mut pos_buf: RingBuffer<T> = RingBuffer::new(window);
        let mut neg_buf: RingBuffer<T> = RingBuffer::new(window);

        for i in 0..len {
            pos_buf.push(pos_mf[i]);
            neg_buf.push(neg_mf[i]);

            if i < window {
                if self.config.fillna {
                    result.push(T::ZERO);
                } else {
                    result.push(T::NAN);
                }
            } else {
                let sum_pos = pos_buf.sum();
                let sum_neg = neg_buf.sum();
                let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

                let mfi = if sum_neg.abs() < epsilon && sum_pos.abs() < epsilon {
                    T::FIFTY // Neutral
                } else if sum_neg.abs() < epsilon {
                    T::HUNDRED // All positive
                } else if sum_pos.abs() < epsilon {
                    T::ZERO // All negative
                } else {
                    let ratio = sum_pos / sum_neg;
                    T::HUNDRED - T::HUNDRED / (T::ONE + ratio)
                };

                // Clamp to [0, 100] to handle floating-point precision issues
                result.push(mfi.clamp_value(T::ZERO, T::HUNDRED));
            }
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        MfiState {
            version: 1,
            pos_mf_buffer: self.pos_mf_buffer.iter().copied().collect(),
            neg_mf_buffer: self.neg_mf_buffer.iter().copied().collect(),
            prev_tp: self.prev_tp,
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

        self.pos_mf_buffer = RingBuffer::new(self.config.window);
        for v in state.pos_mf_buffer {
            self.pos_mf_buffer.push(v);
        }

        self.neg_mf_buffer = RingBuffer::new(self.config.window);
        for v in state.neg_mf_buffer {
            self.neg_mf_buffer.push(v);
        }

        self.prev_tp = state.prev_tp;
        self.count = state.count;
        Ok(())
    }

    fn reset(&mut self) {
        self.pos_mf_buffer = RingBuffer::new(self.config.window);
        self.neg_mf_buffer = RingBuffer::new(self.config.window);
        self.prev_tp = T::NAN;
        self.count = 0;
        self.current_value = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Mfi<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        self.count += 1;
        let tp = Self::typical_price(bar.high, bar.low, bar.close);

        let (pos_mf, neg_mf) = if self.prev_tp.is_nan() {
            (T::ZERO, T::ZERO)
        } else {
            let mf = tp * bar.volume;
            if tp > self.prev_tp {
                (mf, T::ZERO)
            } else if tp < self.prev_tp {
                (T::ZERO, mf)
            } else {
                (T::ZERO, T::ZERO)
            }
        };

        self.prev_tp = tp;
        self.pos_mf_buffer.push(pos_mf);
        self.neg_mf_buffer.push(neg_mf);

        if !self.pos_mf_buffer.is_full() {
            return Ok(None);
        }

        let sum_pos = self.pos_mf_buffer.sum();
        let sum_neg = self.neg_mf_buffer.sum();
        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

        let mfi = if sum_neg.abs() < epsilon && sum_pos.abs() < epsilon {
            T::FIFTY
        } else if sum_neg.abs() < epsilon {
            T::HUNDRED
        } else if sum_pos.abs() < epsilon {
            T::ZERO
        } else {
            let ratio = sum_pos / sum_neg;
            T::HUNDRED - T::HUNDRED / (T::ONE + ratio)
        };

        // Clamp to [0, 100] to handle floating-point precision issues
        let mfi = mfi.clamp_value(T::ZERO, T::HUNDRED);

        self.current_value = Some(mfi);
        Ok(Some(mfi))
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

    #[test]
    fn test_mfi_default_config() {
        let config = MfiConfig::default();
        assert_eq!(config.window, 14);
    }

    #[test]
    fn test_mfi_bounds() {
        let config = MfiConfig::new(5);
        let mut mfi = Mfi::<f64>::new(config);

        for i in 0..30 {
            let price = 100.0 + (i as f64 * 0.3).sin() * 10.0;
            let bar = Bar::new(price, price + 2.0, price - 2.0, price, 1000.0 + i as f64 * 100.0);
            if let Ok(Some(value)) = mfi.update(&bar) {
                assert!(value >= 0.0 && value <= 100.0, "MFI out of bounds: {}", value);
            }
        }
    }

    #[test]
    fn test_mfi_all_up() {
        let config = MfiConfig::new(3);
        let mut mfi = Mfi::<f64>::new(config);

        // Continuously rising prices
        for i in 0..10 {
            let price = 100.0 + i as f64;
            let bar = Bar::new(price, price + 1.0, price - 1.0, price, 1000.0);
            if let Ok(Some(value)) = mfi.update(&bar) {
                // MFI should be 100 when all money flow is positive
                assert!((value - 100.0).abs() < 1e-8);
            }
        }
    }
}
