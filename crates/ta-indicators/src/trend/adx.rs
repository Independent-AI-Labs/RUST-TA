//! Average Directional Index (ADX) indicator.
//!
//! ADX measures the strength of a trend, regardless of direction.

use ta_core::{
    error::{IndicatorError, Result},
    num::TaFloat,
    ohlcv::{Bar, OhlcvSeries},
    series::Series,
    traits::{Indicator, StreamingIndicator},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for ADX.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdxConfig {
    /// Lookback period (default: 14).
    pub window: usize,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for AdxConfig {
    fn default() -> Self {
        Self {
            window: 14,
            fillna: false,
        }
    }
}

impl AdxConfig {
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

/// Output of ADX indicator.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct AdxOutput<T: TaFloat> {
    /// ADX value (0-100).
    pub adx: T,
    /// +DI value (0-100).
    pub plus_di: T,
    /// -DI value (0-100).
    pub minus_di: T,
}

impl<T: TaFloat> AdxOutput<T> {
    /// Create a new output.
    pub fn new(adx: T, plus_di: T, minus_di: T) -> Self {
        Self { adx, plus_di, minus_di }
    }
}

/// State for ADX.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct AdxState<T: TaFloat> {
    /// Version tag.
    pub version: u32,
    /// Previous high.
    pub prev_high: T,
    /// Previous low.
    pub prev_low: T,
    /// Previous close.
    pub prev_close: T,
    /// Smoothed +DM.
    pub smooth_plus_dm: T,
    /// Smoothed -DM.
    pub smooth_minus_dm: T,
    /// Smoothed TR.
    pub smooth_tr: T,
    /// Smoothed DX for ADX.
    pub smooth_dx: T,
    /// Count of values.
    pub count: usize,
    /// DX initialization sum.
    pub dx_sum: T,
    /// Number of DX values for initialization.
    pub dx_count: usize,
    /// Whether ADX is initialized.
    pub adx_initialized: bool,
}

/// ADX series output.
#[derive(Debug, Clone)]
pub struct AdxSeries<T: TaFloat> {
    /// ADX series.
    pub adx: Series<T>,
    /// +DI series.
    pub plus_di: Series<T>,
    /// -DI series.
    pub minus_di: Series<T>,
}

/// Average Directional Index indicator.
///
/// # Formula
///
/// +DM = High - Prev High (if > 0 and > -(Low - Prev Low), else 0)
/// -DM = Prev Low - Low (if > 0 and > +(High - Prev High), else 0)
/// TR = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
///
/// +DI = 100 * Smooth(+DM) / Smooth(TR)
/// -DI = 100 * Smooth(-DM) / Smooth(TR)
/// DX = 100 * |+DI - -DI| / (+DI + -DI)
/// ADX = Smooth(DX)
#[derive(Debug, Clone)]
pub struct Adx<T: TaFloat> {
    config: AdxConfig,
    prev_high: T,
    prev_low: T,
    prev_close: T,
    smooth_plus_dm: T,
    smooth_minus_dm: T,
    smooth_tr: T,
    smooth_dx: T,
    count: usize,
    dx_sum: T,
    dx_count: usize,
    adx_initialized: bool,
    current_output: Option<AdxOutput<T>>,
}

impl<T: TaFloat> Adx<T> {
    /// Get current output.
    pub fn output(&self) -> Option<AdxOutput<T>> {
        self.current_output
    }

    fn true_range(high: T, low: T, prev_close: T) -> T {
        let hl = high - low;
        let hpc = (high - prev_close).abs();
        let lpc = (low - prev_close).abs();
        hl.max(hpc).max(lpc)
    }

    fn calc_dm(high: T, low: T, prev_high: T, prev_low: T) -> (T, T) {
        let up_move = high - prev_high;
        let down_move = prev_low - low;

        let plus_dm = if up_move > down_move && up_move > T::ZERO {
            up_move
        } else {
            T::ZERO
        };

        let minus_dm = if down_move > up_move && down_move > T::ZERO {
            down_move
        } else {
            T::ZERO
        };

        (plus_dm, minus_dm)
    }
}

impl<T: TaFloat> Indicator<T> for Adx<T> {
    type Output = AdxSeries<T>;
    type Config = AdxConfig;
    type State = AdxState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        Self {
            config,
            prev_high: T::NAN,
            prev_low: T::NAN,
            prev_close: T::NAN,
            smooth_plus_dm: T::ZERO,
            smooth_minus_dm: T::ZERO,
            smooth_tr: T::ZERO,
            smooth_dx: T::ZERO,
            count: 0,
            dx_sum: T::ZERO,
            dx_count: 0,
            adx_initialized: false,
            current_output: None,
        }
    }

    fn min_periods(&self) -> usize {
        2 * self.config.window
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

        let mut adx_series = Series::with_capacity(len);
        let mut plus_di_series = Series::with_capacity(len);
        let mut minus_di_series = Series::with_capacity(len);

        if len == 0 {
            return Ok(AdxSeries {
                adx: adx_series,
                plus_di: plus_di_series,
                minus_di: minus_di_series,
            });
        }

        // Calculate TR, +DM, -DM for each bar
        let mut tr_vals = vec![T::ZERO; len];
        let mut plus_dm_vals = vec![T::ZERO; len];
        let mut minus_dm_vals = vec![T::ZERO; len];

        tr_vals[0] = high[0] - low[0];

        for i in 1..len {
            tr_vals[i] = Self::true_range(high[i], low[i], close[i - 1]);
            let (pdm, mdm) = Self::calc_dm(high[i], low[i], high[i - 1], low[i - 1]);
            plus_dm_vals[i] = pdm;
            minus_dm_vals[i] = mdm;
        }

        // Calculate smoothed values and DI
        let mut smooth_tr = T::ZERO;
        let mut smooth_plus_dm = T::ZERO;
        let mut smooth_minus_dm = T::ZERO;
        let mut dx_vals = Vec::with_capacity(len);

        for i in 0..len {
            if i < window {
                // Accumulate for first window
                smooth_tr = smooth_tr + tr_vals[i];
                smooth_plus_dm = smooth_plus_dm + plus_dm_vals[i];
                smooth_minus_dm = smooth_minus_dm + minus_dm_vals[i];

                if self.config.fillna {
                    plus_di_series.push(T::ZERO);
                    minus_di_series.push(T::ZERO);
                    adx_series.push(T::ZERO);
                } else {
                    plus_di_series.push(T::NAN);
                    minus_di_series.push(T::NAN);
                    adx_series.push(T::NAN);
                }
            } else if i == window {
                // First smoothed value (use SMA)
                let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);
                let plus_di = if smooth_tr.abs() > epsilon {
                    T::HUNDRED * smooth_plus_dm / smooth_tr
                } else {
                    T::ZERO
                };
                let minus_di = if smooth_tr.abs() > epsilon {
                    T::HUNDRED * smooth_minus_dm / smooth_tr
                } else {
                    T::ZERO
                };

                let di_sum = plus_di + minus_di;
                let dx = if di_sum.abs() > epsilon {
                    T::HUNDRED * (plus_di - minus_di).abs() / di_sum
                } else {
                    T::ZERO
                };

                dx_vals.push(dx);

                plus_di_series.push(plus_di);
                minus_di_series.push(minus_di);

                if self.config.fillna {
                    adx_series.push(T::ZERO);
                } else {
                    adx_series.push(T::NAN);
                }
            } else {
                // Wilder's smoothing for subsequent values
                let n = <T as TaFloat>::from_usize(window);
                smooth_tr = smooth_tr - smooth_tr / n + tr_vals[i];
                smooth_plus_dm = smooth_plus_dm - smooth_plus_dm / n + plus_dm_vals[i];
                smooth_minus_dm = smooth_minus_dm - smooth_minus_dm / n + minus_dm_vals[i];

                let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);
                let plus_di = if smooth_tr.abs() > epsilon {
                    T::HUNDRED * smooth_plus_dm / smooth_tr
                } else {
                    T::ZERO
                };
                let minus_di = if smooth_tr.abs() > epsilon {
                    T::HUNDRED * smooth_minus_dm / smooth_tr
                } else {
                    T::ZERO
                };

                let di_sum = plus_di + minus_di;
                let dx = if di_sum.abs() > epsilon {
                    T::HUNDRED * (plus_di - minus_di).abs() / di_sum
                } else {
                    T::ZERO
                };

                dx_vals.push(dx);

                plus_di_series.push(plus_di);
                minus_di_series.push(minus_di);

                // Calculate ADX
                if dx_vals.len() < window {
                    if self.config.fillna {
                        adx_series.push(T::ZERO);
                    } else {
                        adx_series.push(T::NAN);
                    }
                } else if dx_vals.len() == window {
                    // First ADX is SMA of DX
                    let adx: T = dx_vals.iter().copied().fold(T::ZERO, |a, b| a + b) / n;
                    adx_series.push(adx);
                } else {
                    // Wilder's smoothing for ADX
                    let prev_adx = adx_series[i - 1];
                    let adx = (prev_adx * (n - T::ONE) + dx) / n;
                    adx_series.push(adx);
                }
            }
        }

        Ok(AdxSeries {
            adx: adx_series,
            plus_di: plus_di_series,
            minus_di: minus_di_series,
        })
    }

    fn get_state(&self) -> Self::State {
        AdxState {
            version: 1,
            prev_high: self.prev_high,
            prev_low: self.prev_low,
            prev_close: self.prev_close,
            smooth_plus_dm: self.smooth_plus_dm,
            smooth_minus_dm: self.smooth_minus_dm,
            smooth_tr: self.smooth_tr,
            smooth_dx: self.smooth_dx,
            count: self.count,
            dx_sum: self.dx_sum,
            dx_count: self.dx_count,
            adx_initialized: self.adx_initialized,
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

        self.prev_high = state.prev_high;
        self.prev_low = state.prev_low;
        self.prev_close = state.prev_close;
        self.smooth_plus_dm = state.smooth_plus_dm;
        self.smooth_minus_dm = state.smooth_minus_dm;
        self.smooth_tr = state.smooth_tr;
        self.smooth_dx = state.smooth_dx;
        self.count = state.count;
        self.dx_sum = state.dx_sum;
        self.dx_count = state.dx_count;
        self.adx_initialized = state.adx_initialized;
        Ok(())
    }

    fn reset(&mut self) {
        self.prev_high = T::NAN;
        self.prev_low = T::NAN;
        self.prev_close = T::NAN;
        self.smooth_plus_dm = T::ZERO;
        self.smooth_minus_dm = T::ZERO;
        self.smooth_tr = T::ZERO;
        self.smooth_dx = T::ZERO;
        self.count = 0;
        self.dx_sum = T::ZERO;
        self.dx_count = 0;
        self.adx_initialized = false;
        self.current_output = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for Adx<T> {
    type StreamingOutput = Option<AdxOutput<T>>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<AdxOutput<T>>> {
        self.count += 1;
        let window = self.config.window;
        let n = <T as TaFloat>::from_usize(window);
        let epsilon = <T as TaFloat>::from_f64_lossy(1e-10);

        // First bar
        if self.prev_high.is_nan() {
            self.prev_high = bar.high;
            self.prev_low = bar.low;
            self.prev_close = bar.close;
            self.smooth_tr = bar.high - bar.low;
            return Ok(None);
        }

        // Calculate TR and DM
        let tr = Self::true_range(bar.high, bar.low, self.prev_close);
        let (plus_dm, minus_dm) = Self::calc_dm(bar.high, bar.low, self.prev_high, self.prev_low);

        self.prev_high = bar.high;
        self.prev_low = bar.low;
        self.prev_close = bar.close;

        // Batch logic at i=window uses smooth values accumulated from indices 0..window-1
        // That corresponds to count=1..window in streaming (window values)
        // First DI is returned at i=window (batch) = count=window+1 (streaming)

        if self.count < window {
            // Accumulate for initialization (count=2 to window-1)
            self.smooth_tr = self.smooth_tr + tr;
            self.smooth_plus_dm = self.smooth_plus_dm + plus_dm;
            self.smooth_minus_dm = self.smooth_minus_dm + minus_dm;
            return Ok(None);
        }

        if self.count == window {
            // Last accumulation (count=window), now we have window values
            // Batch at i=window-1 is still accumulating, batch at i=window returns first DI
            // So streaming at count=window should accumulate, and count=window+1 returns first DI
            self.smooth_tr = self.smooth_tr + tr;
            self.smooth_plus_dm = self.smooth_plus_dm + plus_dm;
            self.smooth_minus_dm = self.smooth_minus_dm + minus_dm;
            return Ok(None);
        }

        if self.count == window + 1 {
            // First DI output (matches batch at i=window)
            // DON'T apply Wilder smoothing - just use accumulated SMA values
            // Batch at i=window does NOT update smooth values
            let plus_di = if self.smooth_tr.abs() > epsilon {
                T::HUNDRED * self.smooth_plus_dm / self.smooth_tr
            } else {
                T::ZERO
            };
            let minus_di = if self.smooth_tr.abs() > epsilon {
                T::HUNDRED * self.smooth_minus_dm / self.smooth_tr
            } else {
                T::ZERO
            };

            let di_sum = plus_di + minus_di;
            let dx = if di_sum.abs() > epsilon {
                T::HUNDRED * (plus_di - minus_di).abs() / di_sum
            } else {
                T::ZERO
            };

            self.dx_sum = self.dx_sum + dx;
            self.dx_count += 1;

            // DO NOT apply Wilder here - matches batch at i=window which doesn't update

            if !self.adx_initialized && self.dx_count >= window {
                self.smooth_dx = self.dx_sum / n;
                self.adx_initialized = true;
                let output = AdxOutput::new(self.smooth_dx, plus_di, minus_di);
                self.current_output = Some(output);
                return Ok(Some(output));
            }

            let output = AdxOutput::new(T::NAN, plus_di, minus_di);
            self.current_output = Some(output);
            return Ok(Some(output));
        }

        // count > window + 1: Apply Wilder's smoothing with CURRENT bar's TR/DM
        // This matches batch at i > window which updates smooth values
        self.smooth_tr = self.smooth_tr - self.smooth_tr / n + tr;
        self.smooth_plus_dm = self.smooth_plus_dm - self.smooth_plus_dm / n + plus_dm;
        self.smooth_minus_dm = self.smooth_minus_dm - self.smooth_minus_dm / n + minus_dm;

        let plus_di = if self.smooth_tr.abs() > epsilon {
            T::HUNDRED * self.smooth_plus_dm / self.smooth_tr
        } else {
            T::ZERO
        };
        let minus_di = if self.smooth_tr.abs() > epsilon {
            T::HUNDRED * self.smooth_minus_dm / self.smooth_tr
        } else {
            T::ZERO
        };

        let di_sum = plus_di + minus_di;
        let dx = if di_sum.abs() > epsilon {
            T::HUNDRED * (plus_di - minus_di).abs() / di_sum
        } else {
            T::ZERO
        };

        if !self.adx_initialized {
            self.dx_sum = self.dx_sum + dx;
            self.dx_count += 1;

            if self.dx_count >= window {
                self.smooth_dx = self.dx_sum / n;
                self.adx_initialized = true;

                let output = AdxOutput::new(self.smooth_dx, plus_di, minus_di);
                self.current_output = Some(output);
                return Ok(Some(output));
            }
            return Ok(None);
        }

        // Wilder's smoothing for ADX
        self.smooth_dx = (self.smooth_dx * (n - T::ONE) + dx) / n;

        let output = AdxOutput::new(self.smooth_dx, plus_di, minus_di);
        self.current_output = Some(output);
        Ok(Some(output))
    }

    fn current(&self) -> Option<AdxOutput<T>> {
        self.current_output
    }

    fn is_ready(&self) -> bool {
        self.adx_initialized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adx_default_config() {
        let config = AdxConfig::default();
        assert_eq!(config.window, 14);
    }

    #[test]
    fn test_adx_bounds() {
        let config = AdxConfig::new(5);
        let mut adx = Adx::<f64>::new(config);

        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.2).sin() * 10.0;
            let bar = Bar::new(price, price + 3.0, price - 3.0, price, 1000.0);
            if let Ok(Some(output)) = adx.update(&bar) {
                // ADX can be NaN before it's fully initialized
                if !output.adx.is_nan() {
                    assert!(output.adx >= 0.0 && output.adx <= 100.0,
                        "ADX out of bounds: {}", output.adx);
                }
                assert!(output.plus_di >= 0.0 && output.plus_di <= 100.0,
                    "+DI out of bounds: {}", output.plus_di);
                assert!(output.minus_di >= 0.0 && output.minus_di <= 100.0,
                    "-DI out of bounds: {}", output.minus_di);
            }
        }
    }

    #[test]
    fn test_adx_min_periods() {
        let config = AdxConfig::new(14);
        let adx = Adx::<f64>::new(config);
        assert_eq!(adx.min_periods(), 28);
    }
}
