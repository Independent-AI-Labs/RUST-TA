//! Normalization Transform (StandardScaler).
//!
//! Standardizes features by removing the mean and scaling to unit variance.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use ta_core::{
    dataframe::DataFrame,
    error::{TransformError, TransformResult},
    num::TaFloat,
    series::Series,
    traits::Transform,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for NormalizationTransform.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NormalizationConfig {
    /// Columns to normalize (if empty, normalizes all columns).
    pub columns: Option<Vec<String>>,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self { columns: None }
    }
}

impl NormalizationConfig {
    /// Create a new configuration with specific columns.
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns: Some(columns),
        }
    }

    /// Create a configuration that normalizes all columns.
    pub fn all() -> Self {
        Self { columns: None }
    }
}

/// Statistics for a single feature.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct FeatureStats<T: TaFloat> {
    /// Mean of the feature.
    pub mean: T,
    /// Standard deviation of the feature.
    pub scale: T,
    /// Variance of the feature.
    pub var: T,
}

/// State for NormalizationTransform.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct NormalizationState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// Feature names in order.
    pub feature_names: Vec<String>,
    /// Mean per feature.
    pub means: Vec<T>,
    /// Scale (std) per feature.
    pub scales: Vec<T>,
    /// Variance per feature.
    pub variances: Vec<T>,
    /// Whether the transform is fitted.
    pub fitted: bool,
}

/// Normalization Transform (StandardScaler).
///
/// Standardizes features using the formula: z = (x - μ) / σ
///
/// # Edge Cases
///
/// - σ = 0: scale is set to 1 to avoid division by zero
#[derive(Debug, Clone)]
pub struct NormalizationTransform<T: TaFloat> {
    config: NormalizationConfig,
    state: NormalizationState<T>,
}

impl<T: TaFloat> NormalizationTransform<T> {
    /// Create a new NormalizationTransform with the given configuration.
    pub fn new(config: NormalizationConfig) -> Self {
        Self {
            config,
            state: NormalizationState::default(),
        }
    }

    /// Get the mean for a feature.
    pub fn mean(&self, feature: &str) -> Option<T> {
        self.state
            .feature_names
            .iter()
            .position(|n| n == feature)
            .map(|i| self.state.means[i])
    }

    /// Get the scale (std) for a feature.
    pub fn scale(&self, feature: &str) -> Option<T> {
        self.state
            .feature_names
            .iter()
            .position(|n| n == feature)
            .map(|i| self.state.scales[i])
    }

    /// Calculate mean of a series, ignoring NaN values.
    fn calculate_mean(series: &Series<T>) -> T {
        let mut sum = T::ZERO;
        let mut count = 0usize;

        for &val in series.iter() {
            if !val.is_nan() {
                sum = sum + val;
                count += 1;
            }
        }

        if count == 0 {
            T::NAN
        } else {
            sum / <T as TaFloat>::from_usize(count)
        }
    }

    /// Calculate variance with Bessel's correction (n-1).
    fn calculate_variance(series: &Series<T>, mean: T) -> T {
        if mean.is_nan() {
            return T::NAN;
        }

        let mut sum_sq = T::ZERO;
        let mut count = 0usize;

        for &val in series.iter() {
            if !val.is_nan() {
                let diff = val - mean;
                sum_sq = sum_sq + diff * diff;
                count += 1;
            }
        }

        if count <= 1 {
            T::ZERO
        } else {
            sum_sq / <T as TaFloat>::from_usize(count - 1)
        }
    }
}

impl<T: TaFloat> Transform<T> for NormalizationTransform<T> {
    type State = NormalizationState<T>;

    fn fit(&mut self, df: &DataFrame<T>) -> TransformResult<()> {
        let columns: Vec<String> = match &self.config.columns {
            Some(cols) => {
                // Verify columns exist
                for col in cols {
                    if df.get_column(col).is_none() {
                        return Err(TransformError::MissingColumn(col.clone()));
                    }
                }
                cols.clone()
            }
            None => df.column_names().iter().map(|s| s.to_string()).collect(),
        };

        let mut means = Vec::with_capacity(columns.len());
        let mut scales = Vec::with_capacity(columns.len());
        let mut variances = Vec::with_capacity(columns.len());

        for col_name in &columns {
            let series = df.get_column(col_name).unwrap();
            let mean = Self::calculate_mean(series);
            let var = Self::calculate_variance(series, mean);

            let scale = if var > T::ZERO {
                var.sqrt()
            } else {
                T::ONE // Avoid division by zero
            };

            means.push(mean);
            variances.push(var);
            scales.push(scale);
        }

        self.state = NormalizationState {
            version: 1,
            feature_names: columns,
            means,
            scales,
            variances,
            fitted: true,
        };

        Ok(())
    }

    fn transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>> {
        if !self.state.fitted {
            return Err(TransformError::NotFitted);
        }

        let mut result = DataFrame::new();

        for col_name in df.column_names() {
            let col_name_str = col_name.to_string();
            let series = df.get_column(&col_name_str).unwrap();

            if let Some(idx) = self.state.feature_names.iter().position(|n| n == &col_name_str) {
                let mean = self.state.means[idx];
                let scale = self.state.scales[idx];

                let transformed: Series<T> = series
                    .iter()
                    .map(|&val| {
                        if val.is_nan() {
                            T::NAN
                        } else {
                            (val - mean) / scale
                        }
                    })
                    .collect();

                result.add_column(col_name_str, transformed)?;
            } else {
                // Keep column as-is
                result.add_column(col_name_str, series.clone())?;
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>> {
        if !self.state.fitted {
            return Err(TransformError::NotFitted);
        }

        let mut result = DataFrame::new();

        for col_name in df.column_names() {
            let col_name_str = col_name.to_string();
            let series = df.get_column(&col_name_str).unwrap();

            if let Some(idx) = self.state.feature_names.iter().position(|n| n == &col_name_str) {
                let mean = self.state.means[idx];
                let scale = self.state.scales[idx];

                // Inverse: x = z * σ + μ
                let transformed: Series<T> = series
                    .iter()
                    .map(|&val| {
                        if val.is_nan() {
                            T::NAN
                        } else {
                            val * scale + mean
                        }
                    })
                    .collect();

                result.add_column(col_name_str, transformed)?;
            } else {
                result.add_column(col_name_str, series.clone())?;
            }
        }

        Ok(result)
    }

    fn get_output_columns(&self, input: &[String]) -> Vec<String> {
        input.to_vec()
    }

    fn get_state(&self) -> Self::State {
        self.state.clone()
    }

    fn set_state(&mut self, state: Self::State) -> TransformResult<()> {
        if state.version != 1 {
            return Err(TransformError::Indicator(
                ta_core::error::IndicatorError::StateError(
                    ta_core::error::StateRestoreError::VersionMismatch {
                        expected: "1".to_string(),
                        actual: state.version.to_string(),
                    },
                ),
            ));
        }
        self.state = state;
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.state.fitted
    }

    fn reset(&mut self) {
        self.state = NormalizationState::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_df() -> DataFrame<f64> {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        )
        .unwrap();
        df
    }

    #[test]
    fn test_normalization_default_config() {
        let config = NormalizationConfig::default();
        assert!(config.columns.is_none());
    }

    #[test]
    fn test_normalization_fit_transform() {
        let config = NormalizationConfig::all();
        let mut transform = NormalizationTransform::<f64>::new(config);

        let df = create_test_df();
        transform.fit(&df).unwrap();

        // Check fitted statistics
        assert_relative_eq!(transform.mean("a").unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(transform.mean("b").unwrap(), 30.0, epsilon = 1e-10);

        let result = transform.transform(&df).unwrap();

        let a = result.get_column("a").unwrap();

        // Mean of normalized data should be ~0
        let mean_a: f64 = a.iter().sum::<f64>() / a.len() as f64;
        assert_relative_eq!(mean_a, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalization_inverse() {
        let config = NormalizationConfig::all();
        let mut transform = NormalizationTransform::<f64>::new(config);

        let df = create_test_df();
        transform.fit(&df).unwrap();

        let transformed = transform.transform(&df).unwrap();
        let recovered = transform.inverse_transform(&transformed).unwrap();

        // Check roundtrip
        let orig_a = df.get_column("a").unwrap();
        let rec_a = recovered.get_column("a").unwrap();

        for i in 0..orig_a.len() {
            assert_relative_eq!(orig_a[i], rec_a[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalization_zero_variance() {
        let mut df = DataFrame::new();
        df.add_column(
            "const".to_string(),
            Series::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]),
        )
        .unwrap();

        let config = NormalizationConfig::all();
        let mut transform = NormalizationTransform::<f64>::new(config);

        transform.fit(&df).unwrap();

        // Scale should be 1 to avoid division by zero
        assert_relative_eq!(transform.scale("const").unwrap(), 1.0, epsilon = 1e-10);

        let result = transform.transform(&df).unwrap();
        let const_col = result.get_column("const").unwrap();

        // All values should be 0 (val - mean = 5 - 5 = 0)
        for &val in const_col.iter() {
            assert_relative_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalization_not_fitted() {
        let config = NormalizationConfig::all();
        let transform = NormalizationTransform::<f64>::new(config);

        let df = create_test_df();
        let result = transform.transform(&df);

        assert!(result.is_err());
    }

    #[test]
    fn test_normalization_missing_column() {
        let config = NormalizationConfig::new(vec!["nonexistent".to_string()]);
        let mut transform = NormalizationTransform::<f64>::new(config);

        let df = create_test_df();
        let result = transform.fit(&df);

        assert!(result.is_err());
    }

    #[test]
    fn test_normalization_state_roundtrip() {
        let config = NormalizationConfig::all();
        let mut transform1 = NormalizationTransform::<f64>::new(config.clone());

        let df = create_test_df();
        transform1.fit(&df).unwrap();

        // Get state
        let state = transform1.get_state();

        // Create new transform and restore state
        let mut transform2 = NormalizationTransform::<f64>::new(config);
        transform2.set_state(state).unwrap();

        assert!(transform2.is_fitted());
        assert_relative_eq!(
            transform1.mean("a").unwrap(),
            transform2.mean("a").unwrap(),
            epsilon = 1e-10
        );
    }
}
