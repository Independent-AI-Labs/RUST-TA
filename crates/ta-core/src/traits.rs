//! Core trait definitions for indicators and transforms.
//!
//! This module defines the fundamental traits that all indicators and transforms
//! must implement.

#[cfg(feature = "serde")]
use serde::{de::DeserializeOwned, Serialize};

use crate::dataframe::DataFrame;
use crate::error::{Result, TransformResult};
use crate::num::TaFloat;
use crate::ohlcv::{Bar, OhlcvSeries};

/// How to handle NaN values in input data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NanMode {
    /// Propagate NaN values through calculations (IEEE 754 compliant).
    /// This is the default behavior.
    #[default]
    Propagate,
    /// Skip NaN inputs, using the previous valid value.
    /// Useful for live trading with missing data.
    Skip,
    /// Return an error when NaN is encountered.
    /// Useful for strict validation mode.
    Error,
}

/// Configuration trait bounds for indicator configurations.
#[cfg(feature = "serde")]
pub trait IndicatorConfig: Clone + Default + Serialize + DeserializeOwned + Send + Sync {}

#[cfg(not(feature = "serde"))]
pub trait IndicatorConfig: Clone + Default + Send + Sync {}

#[cfg(feature = "serde")]
impl<T> IndicatorConfig for T where T: Clone + Default + Serialize + DeserializeOwned + Send + Sync {}

#[cfg(not(feature = "serde"))]
impl<T> IndicatorConfig for T where T: Clone + Default + Send + Sync {}

/// State trait bounds for serializable indicator state.
#[cfg(feature = "serde")]
pub trait IndicatorState: Clone + Serialize + DeserializeOwned + Send + Sync {}

#[cfg(not(feature = "serde"))]
pub trait IndicatorState: Clone + Send + Sync {}

#[cfg(feature = "serde")]
impl<T> IndicatorState for T where T: Clone + Serialize + DeserializeOwned + Send + Sync {}

#[cfg(not(feature = "serde"))]
impl<T> IndicatorState for T where T: Clone + Send + Sync {}

/// Core trait for technical indicators.
///
/// All indicators implement this trait to provide:
/// - Batch computation over a full OHLCV series
/// - State serialization for checkpointing
/// - Configuration management
///
/// # Associated Types
///
/// - `Output` - The type returned by calculations
/// - `Config` - Configuration parameters (must be serializable)
/// - `State` - Internal state (must be serializable)
///
/// # Example Implementation
///
/// ```rust,ignore
/// use ta_core::{Indicator, TaFloat, OhlcvSeries, Result};
///
/// struct Sma<T: TaFloat> {
///     config: SmaConfig,
///     // Internal state
/// }
///
/// impl<T: TaFloat> Indicator<T> for Sma<T> {
///     type Output = Series<T>;
///     type Config = SmaConfig;
///     type State = SmaState;
///
///     fn new(config: Self::Config) -> Self { /* ... */ }
///     fn min_periods(&self) -> usize { self.config.window }
///     fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output> { /* ... */ }
///     // ...
/// }
/// ```
pub trait Indicator<T: TaFloat>: Send + Sync {
    /// The output type of calculations.
    type Output;

    /// Configuration type for this indicator.
    type Config: IndicatorConfig;

    /// Serializable state type.
    type State: IndicatorState;

    /// Create a new indicator with the given configuration.
    fn new(config: Self::Config) -> Self;

    /// Returns the minimum number of data points required for a valid calculation.
    fn min_periods(&self) -> usize;

    /// Perform batch calculation on an OHLCV series.
    ///
    /// # Errors
    ///
    /// Returns an error if there isn't enough data or if computation fails.
    fn calculate(&self, data: &OhlcvSeries<T>) -> Result<Self::Output>;

    /// Export the current internal state for serialization.
    fn get_state(&self) -> Self::State;

    /// Restore internal state from a previously exported state.
    ///
    /// # Errors
    ///
    /// Returns an error if the state is invalid or incompatible.
    fn set_state(&mut self, state: Self::State) -> Result<()>;

    /// Reset the indicator to its initial state.
    fn reset(&mut self);

    /// Get a reference to the current configuration.
    fn config(&self) -> &Self::Config;
}

/// Extension trait for streaming (incremental) indicator computation.
///
/// Indicators that implement `StreamingIndicator` can be updated one bar at a time,
/// which is essential for live trading applications.
///
/// # Streaming vs Batch Equivalence
///
/// A key property of streaming indicators: the result of updating with each bar
/// sequentially should be identical to batch computation on the entire series.
///
/// # Example
///
/// ```rust,ignore
/// use ta_core::{StreamingIndicator, Bar};
///
/// let mut rsi = Rsi::new(RsiConfig::default());
///
/// // Update with each bar as it arrives
/// for bar in bars {
///     let value = rsi.update(&bar)?;
///     println!("Current RSI: {:?}", value);
/// }
///
/// // Or check current value without updating
/// if let Some(current) = rsi.current() {
///     println!("RSI: {}", current);
/// }
/// ```
pub trait StreamingIndicator<T: TaFloat>: Indicator<T> {
    /// The output type of a single streaming update.
    /// This is typically `Option<T>` for scalar indicators or
    /// `Option<OutputStruct>` for multi-value indicators.
    type StreamingOutput;

    /// Update the indicator with a new bar and return the new output value.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails (e.g., invalid input).
    fn update(&mut self, bar: &Bar<T>) -> Result<Self::StreamingOutput>;

    /// Get the current indicator value without updating.
    ///
    /// Returns `None` if not enough data has been processed.
    fn current(&self) -> Self::StreamingOutput;

    /// Check if the indicator has received enough data for valid output.
    fn is_ready(&self) -> bool;
}

/// Transform state trait bounds.
#[cfg(feature = "serde")]
pub trait TransformState: Clone + Default + Serialize + DeserializeOwned + Send + Sync {}

#[cfg(not(feature = "serde"))]
pub trait TransformState: Clone + Default + Send + Sync {}

#[cfg(feature = "serde")]
impl<T> TransformState for T where T: Clone + Default + Serialize + DeserializeOwned + Send + Sync {}

#[cfg(not(feature = "serde"))]
impl<T> TransformState for T where T: Clone + Default + Send + Sync {}

/// Core trait for data transformations.
///
/// Transforms operate on `DataFrame` objects and can be composed into pipelines.
/// They follow the scikit-learn transformer pattern with `fit`, `transform`, and
/// `fit_transform` methods.
///
/// # State Management
///
/// Many transforms need to "learn" parameters from training data (e.g., mean and
/// standard deviation for normalization). The `fit` method learns these parameters,
/// which are stored as state and can be serialized for later use.
///
/// # Example
///
/// ```rust,ignore
/// use ta_core::Transform;
///
/// let mut normalizer = NormalizationTransform::new(NormalizationConfig::default());
///
/// // Fit and transform training data
/// let train_transformed = normalizer.fit_transform(&train_df)?;
///
/// // Transform new data using learned parameters
/// let test_transformed = normalizer.transform(&test_df)?;
///
/// // Inverse transform to get original scale
/// let original = normalizer.inverse_transform(&test_transformed)?;
/// ```
pub trait Transform<T: TaFloat>: Send + Sync {
    /// Serializable state type.
    type State: TransformState;

    /// Learn parameters from the input data.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails (e.g., missing required columns).
    fn fit(&mut self, df: &DataFrame<T>) -> TransformResult<()>;

    /// Apply the transformation to input data.
    ///
    /// # Errors
    ///
    /// Returns an error if the transform hasn't been fitted or if transformation fails.
    fn transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>>;

    /// Fit the transform and apply it in one step.
    ///
    /// This is equivalent to calling `fit` followed by `transform`.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting or transformation fails.
    fn fit_transform(&mut self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>> {
        self.fit(df)?;
        self.transform(df)
    }

    /// Apply the inverse transformation to recover original data.
    ///
    /// Not all transforms are invertible. For transforms that don't support
    /// inverse transformation, this method should return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the transform is not invertible or if inversion fails.
    fn inverse_transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>>;

    /// Get the output column names for given input columns.
    ///
    /// This method helps track column lineage through pipelines.
    fn get_output_columns(&self, input_columns: &[String]) -> Vec<String>;

    /// Export the current state for serialization.
    fn get_state(&self) -> Self::State;

    /// Restore state from a previously exported state.
    ///
    /// # Errors
    ///
    /// Returns an error if the state is invalid.
    fn set_state(&mut self, state: Self::State) -> TransformResult<()>;

    /// Check if the transform has been fitted.
    fn is_fitted(&self) -> bool;

    /// Reset the transform to unfitted state.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nan_mode_default() {
        let mode = NanMode::default();
        assert_eq!(mode, NanMode::Propagate);
    }
}
