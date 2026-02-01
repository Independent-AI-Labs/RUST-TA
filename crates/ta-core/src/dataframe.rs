//! Multi-column tabular data with deterministic iteration order.
//!
//! The [`DataFrame`] type provides a column-oriented data structure similar to
//! pandas DataFrame, but using `IndexMap` for deterministic iteration order.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use indexmap::IndexMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{IndicatorError, Result, TransformError};
use crate::num::TaFloat;
use crate::ohlcv::OhlcvSeries;
use crate::series::Series;

/// A multi-column tabular data structure with deterministic iteration order.
///
/// `DataFrame` uses `IndexMap` internally to guarantee that columns are always
/// iterated in insertion order. This is critical for:
/// - Reproducible pipeline serialization
/// - Deterministic test results
/// - Consistent state hashes for identical data
///
/// # Example
///
/// ```rust
/// use ta_core::{DataFrame, Series};
///
/// let mut df: DataFrame<f64> = DataFrame::new();
/// df.add_column("close".to_string(), Series::from_vec(vec![100.0, 101.0, 102.0])).unwrap();
/// df.add_column("volume".to_string(), Series::from_vec(vec![1000.0, 1100.0, 1200.0])).unwrap();
///
/// assert_eq!(df.len(), 3);
/// assert_eq!(df.column_count(), 2);
/// assert_eq!(df.column_names(), vec!["close", "volume"]);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct DataFrame<T: TaFloat> {
    columns: IndexMap<String, Series<T>>,
}

impl<T: TaFloat> Default for DataFrame<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: TaFloat> DataFrame<T> {
    /// Create a new empty DataFrame.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: IndexMap::new(),
        }
    }

    /// Create a DataFrame with pre-allocated capacity for columns.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            columns: IndexMap::with_capacity(capacity),
        }
    }

    /// Create a DataFrame from a list of (name, series) pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if columns have different lengths.
    pub fn from_columns(columns: Vec<(String, Series<T>)>) -> Result<Self> {
        if columns.is_empty() {
            return Ok(Self::new());
        }

        let expected_len = columns[0].1.len();
        for (name, series) in &columns[1..] {
            if series.len() != expected_len {
                return Err(IndicatorError::LengthMismatch {
                    expected: expected_len,
                    actual: series.len(),
                });
            }
        }

        Ok(Self {
            columns: columns.into_iter().collect(),
        })
    }

    /// Returns the number of rows in the DataFrame.
    #[must_use]
    pub fn len(&self) -> usize {
        self.columns.values().next().map_or(0, Series::len)
    }

    /// Returns `true` if the DataFrame has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of columns in the DataFrame.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns the column names in insertion order.
    #[must_use]
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(String::as_str).collect()
    }

    /// Check if a column exists.
    #[must_use]
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Get a reference to a column by name.
    #[must_use]
    pub fn get_column(&self, name: &str) -> Option<&Series<T>> {
        self.columns.get(name)
    }

    /// Get a mutable reference to a column by name.
    pub fn get_column_mut(&mut self, name: &str) -> Option<&mut Series<T>> {
        self.columns.get_mut(name)
    }

    /// Add a new column to the DataFrame.
    ///
    /// The column is added at the end (preserves insertion order).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A column with the same name already exists
    /// - The series length doesn't match existing columns
    pub fn add_column(&mut self, name: String, series: Series<T>) -> Result<()> {
        if self.columns.contains_key(&name) {
            return Err(IndicatorError::InvalidParameter {
                name: "column_name",
                value: name,
                expected: "unique column name",
            });
        }

        if !self.columns.is_empty() && series.len() != self.len() {
            return Err(IndicatorError::LengthMismatch {
                expected: self.len(),
                actual: series.len(),
            });
        }

        self.columns.insert(name, series);
        Ok(())
    }

    /// Add or replace a column in the DataFrame.
    ///
    /// # Errors
    ///
    /// Returns an error if the series length doesn't match existing columns.
    pub fn set_column(&mut self, name: String, series: Series<T>) -> Result<()> {
        if !self.columns.is_empty()
            && !self.columns.contains_key(&name)
            && series.len() != self.len()
        {
            return Err(IndicatorError::LengthMismatch {
                expected: self.len(),
                actual: series.len(),
            });
        }

        self.columns.insert(name, series);
        Ok(())
    }

    /// Remove and return a column by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the column doesn't exist.
    pub fn drop_column(&mut self, name: &str) -> core::result::Result<Series<T>, TransformError> {
        self.columns
            .swap_remove(name)
            .ok_or_else(|| TransformError::MissingColumn(name.to_string()))
    }

    /// Rename a column.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The old column doesn't exist
    /// - A column with the new name already exists
    pub fn rename_column(&mut self, old_name: &str, new_name: String) -> Result<()> {
        if !self.columns.contains_key(old_name) {
            return Err(IndicatorError::InvalidParameter {
                name: "old_name",
                value: old_name.to_string(),
                expected: "existing column name",
            });
        }

        if self.columns.contains_key(&new_name) {
            return Err(IndicatorError::InvalidParameter {
                name: "new_name",
                value: new_name,
                expected: "unique column name",
            });
        }

        // Remove and re-insert with new name (preserves order)
        if let Some((index, _, series)) = self.columns.swap_remove_full(old_name) {
            self.columns.insert(new_name, series);
            // Move to original position
            let last_index = self.columns.len() - 1;
            self.columns.swap_indices(index, last_index);
        }

        Ok(())
    }

    /// Create a new DataFrame with only the specified columns.
    ///
    /// # Errors
    ///
    /// Returns an error if any column doesn't exist.
    pub fn select(&self, columns: &[&str]) -> core::result::Result<Self, TransformError> {
        let mut result = Self::with_capacity(columns.len());

        for &name in columns {
            let series = self
                .columns
                .get(name)
                .ok_or_else(|| TransformError::MissingColumn(name.to_string()))?;
            // We can use insert directly since we've verified uniqueness
            result.columns.insert(name.to_string(), series.clone());
        }

        Ok(result)
    }

    /// Concatenate two DataFrames horizontally (add columns from other).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The DataFrames have different row counts
    /// - There are duplicate column names
    pub fn concat(&self, other: &Self) -> Result<Self> {
        if !self.is_empty() && !other.is_empty() && self.len() != other.len() {
            return Err(IndicatorError::LengthMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let mut result = self.clone();
        for (name, series) in &other.columns {
            if result.columns.contains_key(name) {
                return Err(IndicatorError::InvalidParameter {
                    name: "column_name",
                    value: name.clone(),
                    expected: "unique column name",
                });
            }
            result.columns.insert(name.clone(), series.clone());
        }

        Ok(result)
    }

    /// Create a DataFrame from an OhlcvSeries.
    #[must_use]
    pub fn from_ohlcv(ohlcv: &OhlcvSeries<T>) -> Self {
        let mut df = Self::with_capacity(5);
        df.columns
            .insert("open".to_string(), ohlcv.open().clone());
        df.columns
            .insert("high".to_string(), ohlcv.high().clone());
        df.columns.insert("low".to_string(), ohlcv.low().clone());
        df.columns
            .insert("close".to_string(), ohlcv.close().clone());
        df.columns
            .insert("volume".to_string(), ohlcv.volume().clone());
        df
    }

    /// Returns an iterator over (column_name, series) pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Series<T>)> {
        self.columns.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Clear all columns from the DataFrame.
    pub fn clear(&mut self) {
        self.columns.clear();
    }
}

impl<T: TaFloat> PartialEq for DataFrame<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.columns.len() != other.columns.len() {
            return false;
        }

        // Compare in order (IndexMap iteration order is deterministic)
        for ((k1, v1), (k2, v2)) in self.columns.iter().zip(other.columns.iter()) {
            if k1 != k2 || v1 != v2 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_dataframe() {
        let df: DataFrame<f64> = DataFrame::new();
        assert!(df.is_empty());
        assert_eq!(df.len(), 0);
        assert_eq!(df.column_count(), 0);
    }

    #[test]
    fn test_add_column() {
        let mut df: DataFrame<f64> = DataFrame::new();
        df.add_column(
            "close".to_string(),
            Series::from_vec(vec![100.0, 101.0, 102.0]),
        )
        .unwrap();

        assert_eq!(df.len(), 3);
        assert_eq!(df.column_count(), 1);
        assert!(df.has_column("close"));
        assert!(!df.has_column("volume"));
    }

    #[test]
    fn test_add_column_length_mismatch() {
        let mut df: DataFrame<f64> = DataFrame::new();
        df.add_column(
            "close".to_string(),
            Series::from_vec(vec![100.0, 101.0, 102.0]),
        )
        .unwrap();

        let result = df.add_column("volume".to_string(), Series::from_vec(vec![1000.0, 1100.0]));

        assert!(result.is_err());
    }

    #[test]
    fn test_column_names_order() {
        let mut df: DataFrame<f64> = DataFrame::new();
        df.add_column("c".to_string(), Series::from_vec(vec![1.0]))
            .unwrap();
        df.add_column("a".to_string(), Series::from_vec(vec![2.0]))
            .unwrap();
        df.add_column("b".to_string(), Series::from_vec(vec![3.0]))
            .unwrap();

        // Should be in insertion order, not alphabetical
        assert_eq!(df.column_names(), vec!["c", "a", "b"]);
    }

    #[test]
    fn test_get_column() {
        let mut df: DataFrame<f64> = DataFrame::new();
        df.add_column(
            "close".to_string(),
            Series::from_vec(vec![100.0, 101.0, 102.0]),
        )
        .unwrap();

        let close = df.get_column("close").unwrap();
        assert_eq!(close.as_slice(), &[100.0, 101.0, 102.0]);

        assert!(df.get_column("nonexistent").is_none());
    }

    #[test]
    fn test_drop_column() {
        let mut df: DataFrame<f64> = DataFrame::new();
        df.add_column("a".to_string(), Series::from_vec(vec![1.0]))
            .unwrap();
        df.add_column("b".to_string(), Series::from_vec(vec![2.0]))
            .unwrap();

        let dropped = df.drop_column("a").unwrap();
        assert_eq!(dropped[0], 1.0);
        assert_eq!(df.column_count(), 1);
        assert!(!df.has_column("a"));
    }

    #[test]
    fn test_select() {
        let mut df: DataFrame<f64> = DataFrame::new();
        df.add_column("a".to_string(), Series::from_vec(vec![1.0]))
            .unwrap();
        df.add_column("b".to_string(), Series::from_vec(vec![2.0]))
            .unwrap();
        df.add_column("c".to_string(), Series::from_vec(vec![3.0]))
            .unwrap();

        let selected = df.select(&["c", "a"]).unwrap();
        assert_eq!(selected.column_names(), vec!["c", "a"]);
    }

    #[test]
    fn test_concat() {
        let mut df1: DataFrame<f64> = DataFrame::new();
        df1.add_column("a".to_string(), Series::from_vec(vec![1.0]))
            .unwrap();

        let mut df2: DataFrame<f64> = DataFrame::new();
        df2.add_column("b".to_string(), Series::from_vec(vec![2.0]))
            .unwrap();

        let combined = df1.concat(&df2).unwrap();
        assert_eq!(combined.column_count(), 2);
        assert!(combined.has_column("a"));
        assert!(combined.has_column("b"));
    }

    #[test]
    fn test_concat_duplicate_column() {
        let mut df1: DataFrame<f64> = DataFrame::new();
        df1.add_column("a".to_string(), Series::from_vec(vec![1.0]))
            .unwrap();

        let mut df2: DataFrame<f64> = DataFrame::new();
        df2.add_column("a".to_string(), Series::from_vec(vec![2.0]))
            .unwrap();

        let result = df1.concat(&df2);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_ohlcv() {
        let mut ohlcv: OhlcvSeries<f64> = OhlcvSeries::new();
        ohlcv.push(crate::Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0));

        let df = DataFrame::from_ohlcv(&ohlcv);
        assert_eq!(df.column_count(), 5);
        assert_eq!(
            df.column_names(),
            vec!["open", "high", "low", "close", "volume"]
        );
    }

    #[test]
    fn test_iteration_order_deterministic() {
        // Create two DataFrames with same columns in same order
        let mut df1: DataFrame<f64> = DataFrame::new();
        df1.add_column("x".to_string(), Series::from_vec(vec![1.0]))
            .unwrap();
        df1.add_column("y".to_string(), Series::from_vec(vec![2.0]))
            .unwrap();
        df1.add_column("z".to_string(), Series::from_vec(vec![3.0]))
            .unwrap();

        let mut df2: DataFrame<f64> = DataFrame::new();
        df2.add_column("x".to_string(), Series::from_vec(vec![1.0]))
            .unwrap();
        df2.add_column("y".to_string(), Series::from_vec(vec![2.0]))
            .unwrap();
        df2.add_column("z".to_string(), Series::from_vec(vec![3.0]))
            .unwrap();

        // Iteration order should be identical
        let names1: Vec<_> = df1.iter().map(|(k, _)| k).collect();
        let names2: Vec<_> = df2.iter().map(|(k, _)| k).collect();
        assert_eq!(names1, names2);
        assert_eq!(names1, vec!["x", "y", "z"]);
    }

    #[test]
    fn test_equality() {
        let mut df1: DataFrame<f64> = DataFrame::new();
        df1.add_column("a".to_string(), Series::from_vec(vec![1.0, 2.0]))
            .unwrap();

        let mut df2: DataFrame<f64> = DataFrame::new();
        df2.add_column("a".to_string(), Series::from_vec(vec![1.0, 2.0]))
            .unwrap();

        assert_eq!(df1, df2);

        df2.add_column("b".to_string(), Series::from_vec(vec![3.0, 4.0]))
            .unwrap();
        assert_ne!(df1, df2);
    }
}
