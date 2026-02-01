//! Keltner Channel indicator.
//!
//! Keltner Channel is a volatility-based envelope using EMA and ATR.

use ta_core::{
    error::{IndicatorError, Result},
    num::TaFloat,
    ohlcv::{Bar, OhlcvSeries},
    series::Series,
    traits::{Indicator, StreamingIndicator},
};

use crate::trend::{Ema, EmaConfig};
use crate::volatility::{Atr, AtrConfig};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for Keltner Channel.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KeltnerConfig {
    /// EMA window (default: 20).
    pub ema_window: usize,
    /// ATR window (default: 10).
    pub atr_window: usize,
    /// ATR multiplier (default: 2.0).
    pub multiplier: f64,
    /// Whether to fill NaN values with 0.
    pub fillna: bool,
}

impl Default for KeltnerConfig {
    fn default() -> Self {
        Self {
            ema_window: 20,
            atr_window: 10,
            multiplier: 2.0,
            fillna: false,
        }
    }
}

impl KeltnerConfig {
    /// Create a new configuration.
    pub fn new(ema_window: usize, atr_window: usize, multiplier: f64) -> Self {
        Self {
            ema_window,
            atr_window,
            multiplier,
            fillna: false,
        }
    }

    /// Set fillna option.
    pub fn with_fillna(mut self, fillna: bool) -> Self {
        self.fillna = fillna;
        self
    }
}

/// Output of Keltner Channel.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct KeltnerOutput<T: TaFloat> {
    /// Upper band.
    pub upper: T,
    /// Middle band (EMA).
    pub middle: T,
    /// Lower band.
    pub lower: T,
}

impl<T: TaFloat> KeltnerOutput<T> {
    /// Create a new output.
    pub fn new(upper: T, middle: T, lower: T) -> Self {
        Self { upper, middle, lower }
    }
}

/// State for Keltner Channel.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct KeltnerState<T: TaFloat> {
    /// Version tag.
    pub version: u32,
    /// EMA state (simplified).
    pub ema_value: T,
    pub ema_count: usize,
    pub ema_initialized: bool,
    /// ATR state (simplified).
    pub atr_value: T,
    pub atr_count: usize,
    pub atr_initialized: bool,
    pub prev_close: T,
}

/// Keltner Channel series output.
#[derive(Debug, Clone)]
pub struct KeltnerSeries<T: TaFloat> {
    /// Upper band series.
    pub upper: Series<T>,
    /// Middle band series.
    pub middle: Series<T>,
    /// Lower band series.
    pub lower: Series<T>,
}

/// Keltner Channel indicator.
///
/// # Formula
///
/// Middle = EMA(Close, ema_window)
/// Upper = Middle + multiplier * ATR
/// Lower = Middle - multiplier * ATR
#[derive(Debug, Clone)]
pub struct KeltnerChannel<T: TaFloat> {
    config: KeltnerConfig,
    ema: Ema<T>,
    atr: Atr<T>,
    current_output: Option<KeltnerOutput<T>>,
}

impl<T: TaFloat> KeltnerChannel<T> {
    /// Get current output.
    pub fn output(&self) -> Option<KeltnerOutput<T>> {
        self.current_output
    }
}

impl<T: TaFloat> Indicator<T> for KeltnerChannel<T> {
    type Output = KeltnerSeries<T>;
    type Config = KeltnerConfig;
    type State = KeltnerState<T>;

    fn config(&self) -> &Self::Config { &self.config }

    fn new(config: Self::Config) -> Self {
        let ema = Ema::new(EmaConfig::new(config.ema_window).with_fillna(config.fillna));
        let atr = Atr::new(AtrConfig::new(config.atr_window).with_fillna(config.fillna));

        Self {
            config,
            ema,
            atr,
            current_output: None,
        }
    }

    fn min_periods(&self) -> usize {
        self.config.ema_window.max(self.config.atr_window + 1)
    }

    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> {
        let len = data.len();

        if self.config.ema_window == 0 || self.config.atr_window == 0 {
            return Err(IndicatorError::InvalidWindow(0));
        }

        let ema_series = self.ema.calculate(data)?;
        let atr_series = self.atr.calculate(data)?;

        let multiplier = <T as TaFloat>::from_f64_lossy(self.config.multiplier);
        let mut upper = Series::with_capacity(len);
        let mut middle = Series::with_capacity(len);
        let mut lower = Series::with_capacity(len);

        for i in 0..len {
            let ema_val = ema_series[i];
            let atr_val = atr_series[i];

            if ema_val.is_nan() || atr_val.is_nan() {
                if self.config.fillna {
                    upper.push(T::ZERO);
                    middle.push(T::ZERO);
                    lower.push(T::ZERO);
                } else {
                    upper.push(T::NAN);
                    middle.push(T::NAN);
                    lower.push(T::NAN);
                }
            } else {
                middle.push(ema_val);
                upper.push(ema_val + multiplier * atr_val);
                lower.push(ema_val - multiplier * atr_val);
            }
        }

        Ok(KeltnerSeries { upper, middle, lower })
    }

    fn get_state(&self) -> Self::State {
        let ema_state = self.ema.get_state();
        let atr_state = self.atr.get_state();

        KeltnerState {
            version: 1,
            ema_value: ema_state.ema_value,
            ema_count: ema_state.count,
            ema_initialized: ema_state.initialized,
            atr_value: atr_state.atr_value,
            atr_count: atr_state.count,
            atr_initialized: atr_state.initialized,
            prev_close: atr_state.prev_close,
        }
    }

    fn set_state(&mut self, _state: Self::State) -> Result<()> {
        // Simplified state restoration - full restoration would need nested states
        Ok(())
    }

    fn reset(&mut self) {
        self.ema.reset();
        self.atr.reset();
        self.current_output = None;
    }
}

impl<T: TaFloat> StreamingIndicator<T> for KeltnerChannel<T> {
    type StreamingOutput = Option<KeltnerOutput<T>>;

    fn update(&mut self, bar: &Bar<T>) -> Result<Option<KeltnerOutput<T>>> {
        let ema_result = self.ema.update(bar)?;
        let atr_result = self.atr.update(bar)?;

        match (ema_result, atr_result) {
            (Some(ema_val), Some(atr_val)) => {
                let multiplier = <T as TaFloat>::from_f64_lossy(self.config.multiplier);
                let output = KeltnerOutput::new(
                    ema_val + multiplier * atr_val,
                    ema_val,
                    ema_val - multiplier * atr_val,
                );
                self.current_output = Some(output);
                Ok(Some(output))
            }
            _ => Ok(None),
        }
    }

    fn current(&self) -> Option<KeltnerOutput<T>> {
        self.current_output
    }

    fn is_ready(&self) -> bool {
        self.current_output.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keltner_default_config() {
        let config = KeltnerConfig::default();
        assert_eq!(config.ema_window, 20);
        assert_eq!(config.atr_window, 10);
        assert!((config.multiplier - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_keltner_band_order() {
        let config = KeltnerConfig::new(5, 5, 2.0);
        let mut keltner = KeltnerChannel::<f64>::new(config);

        for i in 0..30 {
            let price = 100.0 + (i as f64 * 0.2).sin() * 5.0;
            let bar = Bar::new(price, price + 2.0, price - 2.0, price, 1000.0);
            if let Ok(Some(output)) = keltner.update(&bar) {
                assert!(output.lower <= output.middle,
                    "Lower {} > Middle {}", output.lower, output.middle);
                assert!(output.middle <= output.upper,
                    "Middle {} > Upper {}", output.middle, output.upper);
            }
        }
    }
}
