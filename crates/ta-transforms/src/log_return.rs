//! Log Return Transform.
//!
//! Converts price series to log returns.

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

/// Configuration for LogReturnTransform.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LogReturnConfig {
    /// Columns to transform (if empty, transforms all numeric columns).
    pub columns: Vec<String>,
    /// Value to fill for NaN results (first row will always be NaN).
    pub fill_na: Option<f64>,
    /// Whether to use log1p for volume columns.
    pub use_log1p_for_volume: bool,
}

impl Default for LogReturnConfig {
    fn default() -> Self {
        Self {
            columns: Vec::new(),
            fill_na: None,
            use_log1p_for_volume: true,
        }
    }
}

impl LogReturnConfig {
    /// Create a new configuration with specific columns.
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            fill_na: None,
            use_log1p_for_volume: true,
        }
    }

    /// Set fill_na value.
    pub fn with_fill_na(mut self, value: f64) -> Self {
        self.fill_na = Some(value);
        self
    }

    /// Set log1p for volume.
    pub fn with_log1p_for_volume(mut self, use_log1p: bool) -> Self {
        self.use_log1p_for_volume = use_log1p;
        self
    }
}

/// State for LogReturnTransform.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LogReturnState {
    /// Version tag for state compatibility.
    pub version: u32,
    /// Columns that were fitted.
    pub columns: Vec<String>,
    /// Whether the transform is fitted.
    pub fitted: bool,
}

/// Log Return Transform.
///
/// Converts price columns to log returns: ln(P_t / P_{t-1})
///
/// # Edge Cases
///
/// - P_{t-1} <= 0: Returns NaN
/// - First row: Returns NaN (no previous price)
/// - Volume: Uses log1p if configured
#[derive(Debug, Clone)]
pub struct LogReturnTransform<T: TaFloat> {
    config: LogReturnConfig,
    state: LogReturnState,
    _phantom: core::marker::PhantomData<T>,
}

impl<T: TaFloat> LogReturnTransform<T> {
    /// Create a new LogReturnTransform with the given configuration.
    pub fn new(config: LogReturnConfig) -> Self {
        Self {
            config,
            state: LogReturnState::default(),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Check if a column name suggests it's a volume column.
    fn is_volume_column(name: &str) -> bool {
        name.to_lowercase().contains("volume")
            || name.to_lowercase().contains("vol")
            || name.to_lowercase() == "v"
    }

    /// Calculate log return for a series.
    fn calculate_log_return(&self, series: &Series<T>, is_volume: bool) -> Series<T> {
        let len = series.len();
        let mut result = Series::with_capacity(len);

        if len == 0 {
            return result;
        }

        // First element is always NaN (no previous value)
        if let Some(fill) = self.config.fill_na {
            result.push(<T as TaFloat>::from_f64_lossy(fill));
        } else {
            result.push(T::NAN);
        }

        for i in 1..len {
            let current = series[i];
            let prev = series[i - 1];

            let log_ret = if is_volume && self.config.use_log1p_for_volume {
                // For volume, use log1p to handle zero values
                let curr_log = if current > T::ZERO {
                    (T::ONE + current).ln()
                } else {
                    T::ZERO
                };
                let prev_log = if prev > T::ZERO {
                    (T::ONE + prev).ln()
                } else {
                    T::ZERO
                };
                curr_log - prev_log
            } else {
                // Standard log return
                if prev > T::ZERO && current > T::ZERO {
                    (current / prev).ln()
                } else {
                    T::NAN
                }
            };

            if log_ret.is_nan() {
                if let Some(fill) = self.config.fill_na {
                    result.push(<T as TaFloat>::from_f64_lossy(fill));
                } else {
                    result.push(T::NAN);
                }
            } else {
                result.push(log_ret);
            }
        }

        result
    }
}

impl<T: TaFloat> Transform<T> for LogReturnTransform<T> {
    type State = LogReturnState;

    fn fit(&mut self, df: &DataFrame<T>) -> TransformResult<()> {
        let columns = if self.config.columns.is_empty() {
            df.column_names().iter().map(|s| s.to_string()).collect()
        } else {
            // Verify columns exist
            for col in &self.config.columns {
                if df.get_column(col).is_none() {
                    return Err(TransformError::MissingColumn(col.clone()));
                }
            }
            self.config.columns.clone()
        };

        self.state = LogReturnState {
            version: 1,
            columns,
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

            if self.state.columns.contains(&col_name_str) {
                let is_volume = Self::is_volume_column(&col_name_str);
                let transformed = self.calculate_log_return(series, is_volume);
                result.add_column(col_name_str, transformed)?;
            } else {
                // Keep column as-is
                result.add_column(col_name_str, series.clone())?;
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, _df: &DataFrame<T>) -> TransformResult<DataFrame<T>> {
        // Log return is not fully invertible without initial price
        Err(TransformError::InverseNotSupported(
            "LogReturnTransform is not invertible without initial prices".to_string(),
        ))
    }

    fn get_output_columns(&self, input: &[String]) -> Vec<String> {
        // Column names stay the same, just values change
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
        self.state = LogReturnState::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_df() -> DataFrame<f64> {
        let mut df = DataFrame::new();
        df.add_column(
            "close".to_string(),
            Series::from_vec(vec![100.0, 105.0, 103.0, 108.0, 106.0]),
        )
        .unwrap();
        df.add_column(
            "volume".to_string(),
            Series::from_vec(vec![1000.0, 1500.0, 1200.0, 1800.0, 1100.0]),
        )
        .unwrap();
        df
    }

    #[test]
    fn test_log_return_default_config() {
        let config = LogReturnConfig::default();
        assert!(config.columns.is_empty());
        assert!(config.fill_na.is_none());
        assert!(config.use_log1p_for_volume);
    }

    #[test]
    fn test_log_return_transform() {
        let config = LogReturnConfig::new(vec!["close".to_string()]);
        let mut transform = LogReturnTransform::<f64>::new(config);

        let df = create_test_df();
        transform.fit(&df).unwrap();
        let result = transform.transform(&df).unwrap();

        let close = result.get_column("close").unwrap();

        // First value should be NaN
        assert!(close[0].is_nan());

        // Second value: ln(105/100) = ln(1.05)
        assert_relative_eq!(close[1], (105.0_f64 / 100.0).ln(), epsilon = 1e-10);

        // Third value: ln(103/105)
        assert_relative_eq!(close[2], (103.0_f64 / 105.0).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_log_return_with_fill_na() {
        let config = LogReturnConfig::new(vec!["close".to_string()]).with_fill_na(0.0);
        let mut transform = LogReturnTransform::<f64>::new(config);

        let df = create_test_df();
        transform.fit(&df).unwrap();
        let result = transform.transform(&df).unwrap();

        let close = result.get_column("close").unwrap();

        // First value should be 0.0 (filled)
        assert_relative_eq!(close[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_return_not_fitted() {
        let config = LogReturnConfig::default();
        let transform = LogReturnTransform::<f64>::new(config);

        let df = create_test_df();
        let result = transform.transform(&df);

        assert!(result.is_err());
    }

    #[test]
    fn test_log_return_missing_column() {
        let config = LogReturnConfig::new(vec!["nonexistent".to_string()]);
        let mut transform = LogReturnTransform::<f64>::new(config);

        let df = create_test_df();
        let result = transform.fit(&df);

        assert!(result.is_err());
    }

    #[test]
    fn test_log_return_state_roundtrip() {
        let config = LogReturnConfig::new(vec!["close".to_string()]);
        let mut transform1 = LogReturnTransform::<f64>::new(config.clone());

        let df = create_test_df();
        transform1.fit(&df).unwrap();

        // Get state
        let state = transform1.get_state();

        // Create new transform and restore state
        let mut transform2 = LogReturnTransform::<f64>::new(config);
        transform2.set_state(state).unwrap();

        assert!(transform2.is_fitted());
    }

    #[test]
    fn test_log_return_inverse_not_supported() {
        let config = LogReturnConfig::new(vec!["close".to_string()]);
        let mut transform = LogReturnTransform::<f64>::new(config);

        let df = create_test_df();
        transform.fit(&df).unwrap();
        let transformed = transform.transform(&df).unwrap();

        let result = transform.inverse_transform(&transformed);
        assert!(result.is_err());
    }
}
