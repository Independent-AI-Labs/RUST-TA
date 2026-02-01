//! Volume Weighted Average Price (VWAP) indicator.
//!
//! VWAP is the ratio of the cumulative typical price times volume
//! to cumulative volume.

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

/// VWAP calculation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VwapMode {
    /// Cumulative VWAP (no reset).
    Cumulative,
    /// Rolling VWAP with a fixed window.
    Rolling,
    /// Session VWAP (resets at session boundaries).
    Session,
}

impl Default for VwapMode {
    fn default() -> Self {
        Self::Cumulative
    }
}

/// Configuration for the VWAP indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VwapConfig {
    /// VWAP calculation mode.
    pub mode: VwapMode,
    /// Window size for rolling VWAP (only used in Rolling mode).
    pub window: Option<usize>,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for VwapConfig {
    fn default() -> Self {
        Self {
            mode: VwapMode::Cumulative,
            window: None,
            fillna: false,
        }
    }
}

impl VwapConfig {
    /// Create a new cumulative VWAP configuration.
    pub fn cumulative() -> Self {
        Self {
            mode: VwapMode::Cumulative,
            window: None,
            fillna: false,
        }
    }

    /// Create a new rolling VWAP configuration.
    pub fn rolling(window: usize) -> Self {
        Self {
            mode: VwapMode::Rolling,
            window: Some(window),
            fillna: false,
        }
    }

    /// Create a new session VWAP configuration.
    pub fn session() -> Self {
        Self {
            mode: VwapMode::Session,
            window: None,
            fillna: false,
        }
    }

    /// Set fillna option.
    pub fn with_fillna(mut self, fillna: bool) -> Self {
        self.fillna = fillna;
        self
    }
}

/// State for the VWAP indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct VwapState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// Cumulative (TP * Volume).
    pub cum_tp_vol: T,
    /// Cumulative volume.
    pub cum_vol: T,
    /// Previous VWAP value (for carry-forward when volume is zero).
    pub prev_vwap: T,
    /// Rolling buffer for TP*Volume (if rolling mode).
    pub tp_vol_buffer: Vec<T>,
    /// Rolling buffer for Volume (if rolling mode).
    pub vol_buffer: Vec<T>,
    /// Number of values seen.
    pub count: usize,
    /// Previous day number (for session mode).
    pub prev_day: Option<i64>,
}

impl<T: TaFloat> Default for VwapState<T> {
    fn default() -> Self {
        Self {
            version: 1,
            cum_tp_vol: T::ZERO,
            cum_vol: T::ZERO,
            prev_vwap: T::NAN,
            tp_vol_buffer: Vec::new(),
            vol_buffer: Vec::new(),
            count: 0,
            prev_day: None,
        }
    }
}

/// Volume Weighted Average Price indicator.
///
/// # Formula
///
/// Typical Price = (High + Low + Close) / 3
/// VWAP = Σ(TP * Volume) / Σ(Volume)
///
/// # Edge Cases
///
/// - If Σ(Volume) = 0, carry forward the previous VWAP value
/// - If no previous VWAP exists and volume is 0, return NaN
#[derive(Debug, Clone)]
pub struct Vwap<T: TaFloat> {
    config: VwapConfig,
    /// Cumulative (TP * Volume)
    cum_tp_vol: T,
    /// Cumulative volume
    cum_vol: T,
    /// Previous VWAP (for carry-forward)
    prev_vwap: T,
    /// Rolling buffer for TP*Volume
    tp_vol_buffer: Option<RingBuffer<T>>,
    /// Rolling buffer for Volume
    vol_buffer: Option<RingBuffer<T>>,
    /// Number of values seen
    count: usize,
    /// Previous day (for session mode)
    prev_day: Option<i64>,
}

impl<T: TaFloat> Vwap<T> {
    /// Returns the current VWAP value.
    pub fn value(&self) -> Option<T> {
        if self.prev_vwap.is_nan() {
            None
        } else {
            Some(self.prev_vwap)
        }
    }

    /// Calculate typical price.
    fn typical_price(high: T, low: T, close: T) -> T {
        (high + low + close) / <T as TaFloat>::from_f64_lossy(3.0)
    }

    /// Get day number from timestamp (seconds since epoch / 86400).
    fn day_from_timestamp(timestamp: i64) -> i64 {
        timestamp / 86400
    }
}

impl<T: TaFloat> Indicator<T> for Vwap<T> {
    type Output = Series<T>;
    type Config = VwapConfig;
    type State = VwapState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        let (tp_vol_buffer, vol_buffer) = if config.mode == VwapMode::Rolling {
            let window = config.window.unwrap_or(1);
            (Some(RingBuffer::new(window)), Some(RingBuffer::new(window)))
        } else {
            (None, None)
        };

        Self {
            config,
            cum_tp_vol: T::ZERO,
            cum_vol: T::ZERO,
            prev_vwap: T::NAN,
            tp_vol_buffer,
            vol_buffer,
            count: 0,
            prev_day: None,
        }
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();
        let high = data.high();
        let low = data.low();
        let close = data.close();
        let volume = data.volume();

        if self.config.mode == VwapMode::Rolling {
            if self.config.window.is_none() || self.config.window.unwrap() == 0 {
                return Err(IndicatorError::InvalidWindow(0));
            }
        }

        let mut result = Series::with_capacity(len);
        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

        match self.config.mode {
            VwapMode::Cumulative => {
                let mut cum_tp_vol = T::ZERO;
                let mut cum_vol = T::ZERO;
                let mut prev_vwap = T::NAN;

                for i in 0..len {
                    let tp = Self::typical_price(high[i], low[i], close[i]);
                    cum_tp_vol = cum_tp_vol + tp * volume[i];
                    cum_vol = cum_vol + volume[i];

                    let vwap = if cum_vol.abs() > epsilon {
                        cum_tp_vol / cum_vol
                    } else if !prev_vwap.is_nan() {
                        prev_vwap
                    } else if self.config.fillna {
                        T::ZERO
                    } else {
                        T::NAN
                    };

                    prev_vwap = vwap;
                    result.push(vwap);
                }
            }
            VwapMode::Rolling => {
                let window = self.config.window.unwrap();
                let mut tp_vol_buf: RingBuffer<T> = RingBuffer::new(window);
                let mut vol_buf: RingBuffer<T> = RingBuffer::new(window);
                let mut prev_vwap = T::NAN;

                for i in 0..len {
                    let tp = Self::typical_price(high[i], low[i], close[i]);
                    let tp_vol = tp * volume[i];

                    tp_vol_buf.push(tp_vol);
                    vol_buf.push(volume[i]);

                    if tp_vol_buf.is_full() {
                        let sum_tp_vol = tp_vol_buf.sum();
                        let sum_vol = vol_buf.sum();

                        let vwap = if sum_vol.abs() > epsilon {
                            sum_tp_vol / sum_vol
                        } else if !prev_vwap.is_nan() {
                            prev_vwap
                        } else if self.config.fillna {
                            T::ZERO
                        } else {
                            T::NAN
                        };

                        prev_vwap = vwap;
                        result.push(vwap);
                    } else if self.config.fillna {
                        result.push(T::ZERO);
                    } else {
                        result.push(T::NAN);
                    }
                }
            }
            VwapMode::Session => {
                let timestamps = data.timestamps();
                let mut cum_tp_vol = T::ZERO;
                let mut cum_vol = T::ZERO;
                let mut prev_vwap = T::NAN;
                let mut prev_day: Option<i64> = None;

                for i in 0..len {
                    // Check for session reset (new day)
                    if let Some(ts) = timestamps.as_ref().and_then(|t| t.get(i)) {
                        let current_day = Self::day_from_timestamp(*ts);
                        if let Some(pd) = prev_day {
                            if current_day != pd {
                                // New session - reset
                                cum_tp_vol = T::ZERO;
                                cum_vol = T::ZERO;
                            }
                        }
                        prev_day = Some(current_day);
                    }

                    let tp = Self::typical_price(high[i], low[i], close[i]);
                    cum_tp_vol = cum_tp_vol + tp * volume[i];
                    cum_vol = cum_vol + volume[i];

                    let vwap = if cum_vol.abs() > epsilon {
                        cum_tp_vol / cum_vol
                    } else if !prev_vwap.is_nan() {
                        prev_vwap
                    } else if self.config.fillna {
                        T::ZERO
                    } else {
                        T::NAN
                    };

                    prev_vwap = vwap;
                    result.push(vwap);
                }
            }
        }

        Ok(result)
    }

    fn get_state(&self) -> Self::State {
        VwapState {
            version: 1,
            cum_tp_vol: self.cum_tp_vol,
            cum_vol: self.cum_vol,
            prev_vwap: self.prev_vwap,
            tp_vol_buffer: self
                .tp_vol_buffer
                .as_ref()
                .map(|b| b.iter().copied().collect())
                .unwrap_or_default(),
            vol_buffer: self
                .vol_buffer
                .as_ref()
                .map(|b| b.iter().copied().collect())
                .unwrap_or_default(),
            count: self.count,
            prev_day: self.prev_day,
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

        self.cum_tp_vol = state.cum_tp_vol;
        self.cum_vol = state.cum_vol;
        self.prev_vwap = state.prev_vwap;
        self.count = state.count;
        self.prev_day = state.prev_day;

        if let Some(ref mut buf) = self.tp_vol_buffer {
            *buf = RingBuffer::new(self.config.window.unwrap_or(1));
            for value in state.tp_vol_buffer {
                buf.push(value);
            }
        }

        if let Some(ref mut buf) = self.vol_buffer {
            *buf = RingBuffer::new(self.config.window.unwrap_or(1));
            for value in state.vol_buffer {
                buf.push(value);
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.cum_tp_vol = T::ZERO;
        self.cum_vol = T::ZERO;
        self.prev_vwap = T::NAN;
        self.count = 0;
        self.prev_day = None;

        if let Some(ref mut buf) = self.tp_vol_buffer {
            *buf = RingBuffer::new(self.config.window.unwrap_or(1));
        }
        if let Some(ref mut buf) = self.vol_buffer {
            *buf = RingBuffer::new(self.config.window.unwrap_or(1));
        }
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Vwap<T> {
    type StreamingOutput = Option<T>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<T>> {
        self.count += 1;

        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);
        let tp = Self::typical_price(bar.high, bar.low, bar.close);
        let tp_vol = tp * bar.volume;

        match self.config.mode {
            VwapMode::Cumulative => {
                // Check for session reset if timestamp provided
                if let Some(ts) = bar.timestamp {
                    let current_day = Self::day_from_timestamp(ts);
                    if self.config.mode == VwapMode::Session {
                        if let Some(pd) = self.prev_day {
                            if current_day != pd {
                                self.cum_tp_vol = T::ZERO;
                                self.cum_vol = T::ZERO;
                            }
                        }
                        self.prev_day = Some(current_day);
                    }
                }

                self.cum_tp_vol = self.cum_tp_vol + tp_vol;
                self.cum_vol = self.cum_vol + bar.volume;

                let vwap = if self.cum_vol.abs() > epsilon {
                    self.cum_tp_vol / self.cum_vol
                } else if !self.prev_vwap.is_nan() {
                    self.prev_vwap
                } else {
                    return Ok(None);
                };

                self.prev_vwap = vwap;
                Ok(Some(vwap))
            }
            VwapMode::Rolling => {
                let tp_buf = self.tp_vol_buffer.as_mut().unwrap();
                let vol_buf = self.vol_buffer.as_mut().unwrap();

                tp_buf.push(tp_vol);
                vol_buf.push(bar.volume);

                if tp_buf.is_full() {
                    let sum_tp_vol = tp_buf.sum();
                    let sum_vol = vol_buf.sum();

                    let vwap = if sum_vol.abs() > epsilon {
                        sum_tp_vol / sum_vol
                    } else if !self.prev_vwap.is_nan() {
                        self.prev_vwap
                    } else {
                        return Ok(None);
                    };

                    self.prev_vwap = vwap;
                    Ok(Some(vwap))
                } else {
                    Ok(None)
                }
            }
            VwapMode::Session => {
                // Check for session reset
                if let Some(ts) = bar.timestamp {
                    let current_day = Self::day_from_timestamp(ts);
                    if let Some(pd) = self.prev_day {
                        if current_day != pd {
                            self.cum_tp_vol = T::ZERO;
                            self.cum_vol = T::ZERO;
                        }
                    }
                    self.prev_day = Some(current_day);
                }

                self.cum_tp_vol = self.cum_tp_vol + tp_vol;
                self.cum_vol = self.cum_vol + bar.volume;

                let vwap = if self.cum_vol.abs() > epsilon {
                    self.cum_tp_vol / self.cum_vol
                } else if !self.prev_vwap.is_nan() {
                    self.prev_vwap
                } else {
                    return Ok(None);
                };

                self.prev_vwap = vwap;
                Ok(Some(vwap))
            }
        }
    }

    fn current(&self) -> Option<T> {
        self.value()
    }

    fn is_ready(&self) -> bool {
        !self.prev_vwap.is_nan()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_ohlcv(data: &[(f64, f64, f64, f64, f64)]) -> OhlcvSeries<f64> {
        let mut ohlcv = OhlcvSeries::new();
        for &(open, high, low, close, volume) in data {
            ohlcv.push(Bar::new(open, high, low, close, volume));
        }
        ohlcv
    }

    #[test]
    fn test_vwap_default_config() {
        let config = VwapConfig::default();
        assert_eq!(config.mode, VwapMode::Cumulative);
        assert!(config.window.is_none());
        assert!(!config.fillna);
    }

    #[test]
    fn test_vwap_cumulative() {
        let config = VwapConfig::cumulative();
        let vwap = Vwap::<f64>::new(config);

        // TP = (H + L + C) / 3
        let data = vec![
            (100.0, 105.0, 95.0, 100.0, 1000.0),  // TP = 100
            (100.0, 110.0, 95.0, 105.0, 2000.0),  // TP = 103.33
            (105.0, 108.0, 100.0, 102.0, 1500.0), // TP = 103.33
        ];
        let ohlcv = create_test_ohlcv(&data);
        let result = vwap.calculate(&ohlcv).unwrap();

        assert_eq!(result.len(), 3);

        // First bar: VWAP = TP1 = 100
        assert_relative_eq!(result[0], 100.0, epsilon = 1e-10);

        // Second bar: VWAP = (100*1000 + 103.33*2000) / 3000
        let tp1 = (105.0 + 95.0 + 100.0) / 3.0;
        let tp2 = (110.0 + 95.0 + 105.0) / 3.0;
        let expected_vwap2 = (tp1 * 1000.0 + tp2 * 2000.0) / 3000.0;
        assert_relative_eq!(result[1], expected_vwap2, epsilon = 1e-8);
    }

    #[test]
    fn test_vwap_rolling() {
        let config = VwapConfig::rolling(2);
        let vwap = Vwap::<f64>::new(config);

        let data = vec![
            (100.0, 105.0, 95.0, 100.0, 1000.0),
            (100.0, 110.0, 95.0, 105.0, 2000.0),
            (105.0, 108.0, 100.0, 102.0, 1500.0),
        ];
        let ohlcv = create_test_ohlcv(&data);
        let result = vwap.calculate(&ohlcv).unwrap();

        // First value should be NaN (not enough data)
        assert!(result[0].is_nan());

        // Second value: rolling VWAP over bars 0-1
        assert!(!result[1].is_nan());

        // Third value: rolling VWAP over bars 1-2
        assert!(!result[2].is_nan());
    }

    #[test]
    fn test_vwap_streaming() {
        let config = VwapConfig::cumulative();
        let mut vwap = Vwap::<f64>::new(config);

        let data = vec![
            (100.0, 105.0, 95.0, 100.0, 1000.0),
            (100.0, 110.0, 95.0, 105.0, 2000.0),
            (105.0, 108.0, 100.0, 102.0, 1500.0),
        ];

        let mut results = Vec::new();
        for &(open, high, low, close, volume) in &data {
            let bar = Bar::new(open, high, low, close, volume);
            let result = vwap.update(&bar).unwrap();
            results.push(result.unwrap());
        }

        // First bar VWAP = TP = 100
        assert_relative_eq!(results[0], 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vwap_streaming_equals_batch() {
        let config = VwapConfig::cumulative();
        let data = vec![
            (100.0, 105.0, 95.0, 100.0, 1000.0),
            (100.0, 110.0, 95.0, 105.0, 2000.0),
            (105.0, 108.0, 100.0, 102.0, 1500.0),
            (102.0, 112.0, 98.0, 108.0, 2500.0),
        ];
        let ohlcv = create_test_ohlcv(&data);

        // Batch calculation
        let batch_vwap = Vwap::<f64>::new(config.clone());
        let batch_result = batch_vwap.calculate(&ohlcv).unwrap();

        // Streaming calculation
        let mut streaming_vwap = Vwap::<f64>::new(config);
        for (i, &(open, high, low, close, volume)) in data.iter().enumerate() {
            let bar = Bar::new(open, high, low, close, volume);
            let streaming_result = streaming_vwap.update(&bar).unwrap().unwrap();
            assert_relative_eq!(streaming_result, batch_result[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vwap_zero_volume_carryforward() {
        let config = VwapConfig::cumulative();
        let mut vwap = Vwap::<f64>::new(config);

        // First bar with volume
        vwap.update(&Bar::new(100.0, 105.0, 95.0, 100.0, 1000.0))
            .unwrap();
        let vwap1 = vwap.value().unwrap();

        // Second bar with zero volume - should carry forward
        let result = vwap
            .update(&Bar::new(100.0, 110.0, 95.0, 105.0, 0.0))
            .unwrap();

        // VWAP should be carried forward
        assert_relative_eq!(result.unwrap(), vwap1, epsilon = 1e-10);
    }

    #[test]
    fn test_vwap_state_roundtrip() {
        let config = VwapConfig::cumulative();
        let mut vwap1 = Vwap::<f64>::new(config.clone());

        // Feed some data
        let data = vec![
            (100.0, 105.0, 95.0, 100.0, 1000.0),
            (100.0, 110.0, 95.0, 105.0, 2000.0),
        ];
        for &(open, high, low, close, volume) in &data {
            vwap1
                .update(&Bar::new(open, high, low, close, volume))
                .unwrap();
        }

        // Get state
        let state = vwap1.get_state();

        // Create new indicator and restore state
        let mut vwap2 = Vwap::<f64>::new(config);
        vwap2.set_state(state).unwrap();

        // Both should produce the same result
        let next_bar = Bar::new(105.0, 108.0, 100.0, 102.0, 1500.0);
        let result1 = vwap1.update(&next_bar).unwrap().unwrap();
        let result2 = vwap2.update(&next_bar).unwrap().unwrap();

        assert_relative_eq!(result1, result2, epsilon = 1e-10);
    }

    #[test]
    fn test_vwap_reset() {
        let config = VwapConfig::cumulative();
        let mut vwap = Vwap::<f64>::new(config);

        // Feed data
        vwap.update(&Bar::new(100.0, 105.0, 95.0, 100.0, 1000.0))
            .unwrap();
        vwap.update(&Bar::new(100.0, 110.0, 95.0, 105.0, 2000.0))
            .unwrap();

        assert!(vwap.is_ready());

        // Reset
        vwap.reset();

        assert!(!vwap.is_ready());
        assert_eq!(vwap.count, 0);
    }

    #[test]
    fn test_vwap_min_periods() {
        let config = VwapConfig::cumulative();
        let vwap = Vwap::<f64>::new(config);

        assert_eq!(vwap.min_periods(), 1);
    }
}
