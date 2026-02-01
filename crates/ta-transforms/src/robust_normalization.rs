//! Robust Normalization Transform (RobustScaler).
//!
//! Scales features using statistics that are robust to outliers.
//! Uses the median and interquartile range (IQR).

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

/// Configuration for RobustNormalizationTransform.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RobustNormalizationConfig {
    /// Columns to normalize (if empty, normalizes all columns).
    pub columns: Option<Vec<String>>,
    /// Quantile range for scaling (default: (0.25, 0.75) for IQR).
    pub quantile_range: (f64, f64),
}

impl Default for RobustNormalizationConfig {
    fn default() -> Self {
        Self {
            columns: None,
            quantile_range: (0.25, 0.75),
        }
    }
}

impl RobustNormalizationConfig {
    /// Create a new configuration with specific columns.
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns: Some(columns),
            quantile_range: (0.25, 0.75),
        }
    }

    /// Create a configuration that normalizes all columns.
    pub fn all() -> Self {
        Self {
            columns: None,
            quantile_range: (0.25, 0.75),
        }
    }

    /// Set custom quantile range.
    pub fn with_quantile_range(mut self, lower: f64, upper: f64) -> Self {
        self.quantile_range = (lower, upper);
        self
    }
}

/// State for RobustNormalizationTransform.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct RobustNormalizationState<T: TaFloat> {
    /// Version tag for state compatibility.
    pub version: u32,
    /// Feature names in order.
    pub feature_names: Vec<String>,
    /// Center (median) per feature.
    pub centers: Vec<T>,
    /// Scale (IQR) per feature.
    pub scales: Vec<T>,
    /// Whether the transform is fitted.
    pub fitted: bool,
}

/// Robust Normalization Transform (RobustScaler).
///
/// Scales features using: z = (x - median) / IQR
///
/// Uses linear interpolation (R-7 method) for quantile calculation,
/// matching numpy's default behavior.
///
/// # Edge Cases
///
/// - IQR = 0: scale is set to 1 to avoid division by zero
#[derive(Debug, Clone)]
pub struct RobustNormalizationTransform<T: TaFloat> {
    config: RobustNormalizationConfig,
    state: RobustNormalizationState<T>,
}

impl<T: TaFloat> RobustNormalizationTransform<T> {
    /// Create a new RobustNormalizationTransform with the given configuration.
    pub fn new(config: RobustNormalizationConfig) -> Self {
        Self {
            config,
            state: RobustNormalizationState::default(),
        }
    }

    /// Get the center (median) for a feature.
    pub fn center(&self, feature: &str) -> Option<T> {
        self.state
            .feature_names
            .iter()
            .position(|n| n == feature)
            .map(|i| self.state.centers[i])
    }

    /// Get the scale (IQR) for a feature.
    pub fn scale(&self, feature: &str) -> Option<T> {
        self.state
            .feature_names
            .iter()
            .position(|n| n == feature)
            .map(|i| self.state.scales[i])
    }

    /// Calculate quantile using R-7 method (linear interpolation).
    /// This matches numpy's default quantile calculation.
    fn calculate_quantile(sorted: &[T], q: f64) -> T {
        let n = sorted.len();
        if n == 0 {
            return T::NAN;
        }
        if n == 1 {
            return sorted[0];
        }

        // R-7 method: index = (n-1) * q
        let index = (n - 1) as f64 * q;
        let lo = index.floor() as usize;
        let hi = index.ceil() as usize;
        let frac = <T as TaFloat>::from_f64_lossy(index - lo as f64);

        if lo == hi {
            sorted[lo]
        } else {
            // Linear interpolation
            sorted[lo] * (T::ONE - frac) + sorted[hi] * frac
        }
    }

    /// Calculate median (50th percentile).
    fn calculate_median(sorted: &[T]) -> T {
        Self::calculate_quantile(sorted, 0.5)
    }
}

impl<T: TaFloat> Transform<T> for RobustNormalizationTransform<T> {
    type State = RobustNormalizationState<T>;

    fn fit(&mut self, df: &DataFrame<T>) -> TransformResult<()> {
        let columns: Vec<String> = match &self.config.columns {
            Some(cols) => {
                for col in cols {
                    if df.get_column(col).is_none() {
                        return Err(TransformError::MissingColumn(col.clone()));
                    }
                }
                cols.clone()
            }
            None => df.column_names().iter().map(|s| s.to_string()).collect(),
        };

        let mut centers = Vec::with_capacity(columns.len());
        let mut scales = Vec::with_capacity(columns.len());

        let (q_lower, q_upper) = self.config.quantile_range;

        for col_name in &columns {
            let series = df.get_column(col_name).unwrap();

            // Filter out NaN values and sort
            let mut values: Vec<T> = series.iter().copied().filter(|v| !v.is_nan()).collect();

            if values.is_empty() {
                centers.push(T::NAN);
                scales.push(T::ONE);
                continue;
            }

            // Sort using total_cmp for NaN handling
            values.sort_by(|a, b| a.total_cmp_fn(b));

            let median = Self::calculate_median(&values);
            let q1 = Self::calculate_quantile(&values, q_lower);
            let q3 = Self::calculate_quantile(&values, q_upper);
            let iqr = q3 - q1;

            let scale = if iqr > T::ZERO { iqr } else { T::ONE };

            centers.push(median);
            scales.push(scale);
        }

        self.state = RobustNormalizationState {
            version: 1,
            feature_names: columns,
            centers,
            scales,
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
                let center = self.state.centers[idx];
                let scale = self.state.scales[idx];

                let transformed: Series<T> = series
                    .iter()
                    .map(|&val| {
                        if val.is_nan() {
                            T::NAN
                        } else {
                            (val - center) / scale
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

    fn inverse_transform(&self, df: &DataFrame<T>) -> TransformResult<DataFrame<T>> {
        if !self.state.fitted {
            return Err(TransformError::NotFitted);
        }

        let mut result = DataFrame::new();

        for col_name in df.column_names() {
            let col_name_str = col_name.to_string();
            let series = df.get_column(&col_name_str).unwrap();

            if let Some(idx) = self.state.feature_names.iter().position(|n| n == &col_name_str) {
                let center = self.state.centers[idx];
                let scale = self.state.scales[idx];

                // Inverse: x = z * scale + center
                let transformed: Series<T> = series
                    .iter()
                    .map(|&val| {
                        if val.is_nan() {
                            T::NAN
                        } else {
                            val * scale + center
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
        self.state = RobustNormalizationState::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_df() -> DataFrame<f64> {
        let mut df = DataFrame::new();
        // Values: 1, 2, 3, 4, 5, 6, 7, 8, 9
        // Median = 5, Q1 = 2.5, Q3 = 7.5, IQR = 5
        df.add_column(
            "a".to_string(),
            Series::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        )
        .unwrap();
        df
    }

    #[test]
    fn test_robust_default_config() {
        let config = RobustNormalizationConfig::default();
        assert!(config.columns.is_none());
        assert_eq!(config.quantile_range, (0.25, 0.75));
    }

    #[test]
    fn test_quantile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Median (50th percentile)
        let median = RobustNormalizationTransform::<f64>::calculate_median(&values);
        assert_relative_eq!(median, 5.0, epsilon = 1e-10);

        // Q1 (25th percentile) with R-7 method
        // index = 8 * 0.25 = 2.0, so value at index 2 = 3.0
        let q1 = RobustNormalizationTransform::<f64>::calculate_quantile(&values, 0.25);
        assert_relative_eq!(q1, 3.0, epsilon = 1e-10);

        // Q3 (75th percentile) with R-7 method
        // index = 8 * 0.75 = 6.0, so value at index 6 = 7.0
        let q3 = RobustNormalizationTransform::<f64>::calculate_quantile(&values, 0.75);
        assert_relative_eq!(q3, 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_robust_fit_transform() {
        let config = RobustNormalizationConfig::all();
        let mut transform = RobustNormalizationTransform::<f64>::new(config);

        let df = create_test_df();
        transform.fit(&df).unwrap();

        // Check fitted statistics
        assert_relative_eq!(transform.center("a").unwrap(), 5.0, epsilon = 1e-10);

        let result = transform.transform(&df).unwrap();
        let a = result.get_column("a").unwrap();

        // Median value (5.0) should become 0
        assert_relative_eq!(a[4], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_robust_inverse() {
        let config = RobustNormalizationConfig::all();
        let mut transform = RobustNormalizationTransform::<f64>::new(config);

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
    fn test_robust_zero_iqr() {
        let mut df = DataFrame::new();
        df.add_column(
            "const".to_string(),
            Series::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]),
        )
        .unwrap();

        let config = RobustNormalizationConfig::all();
        let mut transform = RobustNormalizationTransform::<f64>::new(config);

        transform.fit(&df).unwrap();

        // Scale should be 1 to avoid division by zero
        assert_relative_eq!(transform.scale("const").unwrap(), 1.0, epsilon = 1e-10);

        let result = transform.transform(&df).unwrap();
        let const_col = result.get_column("const").unwrap();

        // All values should be 0
        for &val in const_col.iter() {
            assert_relative_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_robust_not_fitted() {
        let config = RobustNormalizationConfig::all();
        let transform = RobustNormalizationTransform::<f64>::new(config);

        let df = create_test_df();
        let result = transform.transform(&df);

        assert!(result.is_err());
    }

    #[test]
    fn test_robust_custom_quantile_range() {
        let config = RobustNormalizationConfig::all().with_quantile_range(0.1, 0.9);
        let mut transform = RobustNormalizationTransform::<f64>::new(config);

        let df = create_test_df();
        transform.fit(&df).unwrap();

        // Should use 10th and 90th percentiles for scaling
        assert!(transform.is_fitted());
    }

    #[test]
    fn test_robust_state_roundtrip() {
        let config = RobustNormalizationConfig::all();
        let mut transform1 = RobustNormalizationTransform::<f64>::new(config.clone());

        let df = create_test_df();
        transform1.fit(&df).unwrap();

        // Get state
        let state = transform1.get_state();

        // Create new transform and restore state
        let mut transform2 = RobustNormalizationTransform::<f64>::new(config);
        transform2.set_state(state).unwrap();

        assert!(transform2.is_fitted());
        assert_relative_eq!(
            transform1.center("a").unwrap(),
            transform2.center("a").unwrap(),
            epsilon = 1e-10
        );
    }
}
