//! Transform Pipeline.
//!
//! Composes multiple transforms into a single pipeline.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

use ta_core::{
    dataframe::DataFrame,
    error::{TransformError, TransformResult},
    num::TaFloat,
    traits::Transform,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for TransformPipeline.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PipelineConfig {
    /// Names for each transform in the pipeline (for identification).
    pub names: Vec<String>,
}

impl PipelineConfig {
    /// Create a new pipeline configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a transform name.
    pub fn with_name(mut self, name: String) -> Self {
        self.names.push(name);
        self
    }
}

/// Type-erased transform trait for use in pipelines.
///
/// This trait allows storing transforms with different State types
/// in the same collection by erasing the State associated type.
pub trait ErasedTransform<T: TaFloat>: Send + Sync {
    /// Learn parameters from the input data.
    fn fit(&mut self, df: &DataFrame<T>) -> TransformResult<()>;

    /// Apply the transformation to input data.
    fn transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>>;

    /// Apply the inverse transformation.
    fn inverse_transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>>;

    /// Get the output column names for given input columns.
    fn get_output_columns(&self, input_columns: &[String]) -> Vec<String>;

    /// Check if the transform has been fitted.
    fn is_fitted(&self) -> bool;

    /// Reset the transform to unfitted state.
    fn reset(&mut self);
}

/// Blanket implementation of ErasedTransform for any Transform.
impl<T: TaFloat, Tr: Transform<T> + Send + Sync> ErasedTransform<T> for Tr {
    fn fit(&mut self, df: &DataFrame<T>) -> TransformResult<()> {
        Transform::fit(self, df)
    }

    fn transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>> {
        Transform::transform(self, df)
    }

    fn inverse_transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>> {
        Transform::inverse_transform(self, df)
    }

    fn get_output_columns(&self, input_columns: &[String]) -> Vec<String> {
        Transform::get_output_columns(self, input_columns)
    }

    fn is_fitted(&self) -> bool {
        Transform::is_fitted(self)
    }

    fn reset(&mut self) {
        Transform::reset(self)
    }
}

/// Transform Pipeline.
///
/// Chains multiple transforms together. When fit_transform is called,
/// each transform is fit and applied in order.
///
/// # Example
///
/// ```ignore
/// let mut pipeline = TransformPipeline::new()
///     .add(log_return_transform)
///     .add(normalization_transform);
///
/// let transformed = pipeline.fit_transform(&df)?;
/// ```
pub struct TransformPipeline<T: TaFloat> {
    transforms: Vec<Box<dyn ErasedTransform<T>>>,
    names: Vec<String>,
    fitted: bool,
}

impl<T: TaFloat> core::fmt::Debug for TransformPipeline<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TransformPipeline")
            .field("num_transforms", &self.transforms.len())
            .field("names", &self.names)
            .field("fitted", &self.fitted)
            .finish()
    }
}

impl<T: TaFloat> Default for TransformPipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: TaFloat> TransformPipeline<T> {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            names: Vec::new(),
            fitted: false,
        }
    }

    /// Add a transform to the pipeline.
    ///
    /// Accepts any type that implements `Transform<T>`.
    pub fn add<Tr>(mut self, transform: Tr) -> Self
    where
        Tr: Transform<T> + Send + Sync + 'static,
    {
        let name = format!("transform_{}", self.transforms.len());
        self.names.push(name);
        self.transforms.push(Box::new(transform));
        self
    }

    /// Add a transform with a custom name.
    pub fn add_named<Tr>(mut self, name: String, transform: Tr) -> Self
    where
        Tr: Transform<T> + Send + Sync + 'static,
    {
        self.names.push(name);
        self.transforms.push(Box::new(transform));
        self
    }

    /// Get the number of transforms in the pipeline.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Get transform names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Fit all transforms in order.
    pub fn fit(&mut self, df: &DataFrame<T>) -> TransformResult<()> {
        let mut current_df = df.clone();

        for transform in &mut self.transforms {
            transform.fit(&current_df)?;
            current_df = transform.transform(&current_df)?;
        }

        self.fitted = true;
        Ok(())
    }

    /// Transform data through all transforms.
    pub fn transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>> {
        if !self.fitted {
            return Err(TransformError::NotFitted);
        }

        let mut current_df = df.clone();

        for transform in &self.transforms {
            current_df = transform.transform(&current_df)?;
        }

        Ok(current_df)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>> {
        self.fit(df)?;
        // Re-transform from the original data since fit already transformed
        // Actually, during fit we already have the final result, but we need
        // to call transform again to ensure consistency
        self.transform(df)
    }

    /// Inverse transform in reverse order.
    pub fn inverse_transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>> {
        if !self.fitted {
            return Err(TransformError::NotFitted);
        }

        let mut current_df = df.clone();

        // Apply inverse transforms in reverse order
        for transform in self.transforms.iter().rev() {
            current_df = transform.inverse_transform(&current_df)?;
        }

        Ok(current_df)
    }

    /// Check if the pipeline is fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Reset the pipeline and all transforms to unfitted state.
    pub fn reset(&mut self) {
        for transform in &mut self.transforms {
            transform.reset();
        }
        self.fitted = false;
    }

    /// Get output columns after all transforms.
    pub fn get_output_columns(&self, input: &[String]) -> Vec<String> {
        let mut columns = input.to_vec();

        for transform in &self.transforms {
            columns = transform.get_output_columns(&columns);
        }

        columns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_default() {
        let pipeline = TransformPipeline::<f64>::new();
        assert!(pipeline.is_empty());
        assert!(!pipeline.is_fitted());
    }

    #[test]
    fn test_pipeline_len() {
        let pipeline = TransformPipeline::<f64>::new();
        assert_eq!(pipeline.len(), 0);
    }
}
