//! Time series data container.
//!
//! The [`Series`] type provides a contiguous, heap-allocated container for
//! time series data with operations commonly used in technical analysis.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use core::ops::{Index, IndexMut};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::num::TaFloat;

/// A contiguous time series of floating-point values.
///
/// `Series<T>` is the fundamental data structure for storing price data,
/// indicator outputs, and intermediate calculations. It provides efficient
/// append operations and random access.
///
/// # Example
///
/// ```rust
/// use ta_core::Series;
///
/// let mut series: Series<f64> = Series::new();
/// series.push(100.0);
/// series.push(101.5);
/// series.push(99.8);
///
/// assert_eq!(series.len(), 3);
/// assert_eq!(series[0], 100.0);
/// assert_eq!(series.last(), Some(&99.8));
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct Series<T: TaFloat> {
    data: Vec<T>,
}

impl<T: TaFloat> Default for Series<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: TaFloat> Series<T> {
    /// Create a new empty series.
    #[must_use]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Create a new series with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Create a series from an existing vector.
    #[must_use]
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Create a series filled with a value.
    #[must_use]
    pub fn filled(value: T, len: usize) -> Self {
        Self {
            data: vec![value; len],
        }
    }

    /// Create a series filled with NaN values.
    #[must_use]
    pub fn nan(len: usize) -> Self {
        Self::filled(T::NAN, len)
    }

    /// Returns the number of elements in the series.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the series contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the capacity of the series.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Append a value to the end of the series.
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }

    /// Remove and return the last value, if any.
    pub fn pop(&mut self) -> Option<T> {
        self.data.pop()
    }

    /// Get a reference to the value at the given index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Get a mutable reference to the value at the given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    /// Get the first value, if any.
    #[must_use]
    pub fn first(&self) -> Option<&T> {
        self.data.first()
    }

    /// Get the last value, if any.
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        self.data.last()
    }

    /// Get the last `n` values as a slice.
    ///
    /// If `n > len()`, returns the entire series.
    #[must_use]
    pub fn tail(&self, n: usize) -> &[T] {
        let start = self.len().saturating_sub(n);
        &self.data[start..]
    }

    /// Get the first `n` values as a slice.
    ///
    /// If `n > len()`, returns the entire series.
    #[must_use]
    pub fn head(&self, n: usize) -> &[T] {
        let end = n.min(self.len());
        &self.data[..end]
    }

    /// Returns an iterator over the values.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Returns a mutable iterator over the values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    /// Returns the underlying data as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the series and returns the underlying vector.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Compute first differences: `y[i] = x[i] - x[i-1]`.
    ///
    /// The first element of the result is NaN.
    #[must_use]
    pub fn diff(&self) -> Self {
        if self.is_empty() {
            return Self::new();
        }

        let mut result = Vec::with_capacity(self.len());
        result.push(T::NAN);

        for i in 1..self.len() {
            result.push(self.data[i] - self.data[i - 1]);
        }

        Self { data: result }
    }

    /// Shift values by `n` positions, filling with the specified value.
    ///
    /// Positive `n` shifts forward (newer values at the start become fill).
    /// Negative `n` shifts backward (older values at the end become fill).
    #[must_use]
    pub fn shift(&self, n: isize, fill: T) -> Self {
        if self.is_empty() {
            return Self::new();
        }

        let len = self.len();
        let mut result = vec![fill; len];

        if n >= 0 {
            let shift = n as usize;
            if shift < len {
                result[shift..].copy_from_slice(&self.data[..len - shift]);
            }
        } else {
            let shift = (-n) as usize;
            if shift < len {
                result[..len - shift].copy_from_slice(&self.data[shift..]);
            }
        }

        Self { data: result }
    }

    /// Replace all NaN values with the specified value.
    pub fn fillna(&mut self, value: T) {
        for x in &mut self.data {
            if x.is_nan() {
                *x = value;
            }
        }
    }

    /// Return a new series with NaN values replaced.
    #[must_use]
    pub fn fillna_with(&self, value: T) -> Self {
        let data = self
            .data
            .iter()
            .map(|&x| if x.is_nan() { value } else { x })
            .collect();
        Self { data }
    }

    /// Forward-fill NaN values with the last valid observation.
    #[must_use]
    pub fn ffill(&self) -> Self {
        let mut result = Vec::with_capacity(self.len());
        let mut last_valid = T::NAN;

        for &x in &self.data {
            if x.is_nan() {
                result.push(last_valid);
            } else {
                last_valid = x;
                result.push(x);
            }
        }

        Self { data: result }
    }

    /// Clamp all values to the range [min, max].
    #[must_use]
    pub fn clamp(&self, min: T, max: T) -> Self {
        let data = self.data.iter().map(|&x| x.clamp_value(min, max)).collect();
        Self { data }
    }

    /// Apply a function to each element.
    #[must_use]
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Self { data }
    }

    /// Apply a function to each element with its index.
    #[must_use]
    pub fn map_indexed<F>(&self, f: F) -> Self
    where
        F: Fn(usize, T) -> T,
    {
        let data = self.data.iter().enumerate().map(|(i, &x)| f(i, x)).collect();
        Self { data }
    }

    /// Count the number of NaN values.
    #[must_use]
    pub fn nan_count(&self) -> usize {
        self.data.iter().filter(|x| x.is_nan()).count()
    }

    /// Check if the series contains any NaN values.
    #[must_use]
    pub fn has_nan(&self) -> bool {
        self.data.iter().any(|x| x.is_nan())
    }

    /// Compute the sum of all valid (non-NaN) values.
    #[must_use]
    pub fn sum(&self) -> T {
        self.data
            .iter()
            .filter(|x| !x.is_nan())
            .fold(T::ZERO, |acc, &x| acc + x)
    }

    /// Compute the mean of all valid (non-NaN) values.
    #[must_use]
    pub fn mean(&self) -> T {
        let valid: Vec<_> = self.data.iter().filter(|x| !x.is_nan()).collect();
        if valid.is_empty() {
            return T::NAN;
        }
        let sum: T = valid.iter().fold(T::ZERO, |acc, &&x| acc + x);
        sum / TaFloat::from_usize(valid.len())
    }

    /// Compute the minimum value (excluding NaN).
    #[must_use]
    pub fn min(&self) -> T {
        self.data
            .iter()
            .filter(|x| !x.is_nan())
            .fold(T::INFINITY, |acc, &x| if x < acc { x } else { acc })
    }

    /// Compute the maximum value (excluding NaN).
    #[must_use]
    pub fn max(&self) -> T {
        self.data
            .iter()
            .filter(|x| !x.is_nan())
            .fold(T::NEG_INFINITY, |acc, &x| if x > acc { x } else { acc })
    }

    /// Clear all values from the series.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Truncate the series to the specified length.
    pub fn truncate(&mut self, len: usize) {
        self.data.truncate(len);
    }

    /// Reserve capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }
}

impl<T: TaFloat> Index<usize> for Series<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: TaFloat> IndexMut<usize> for Series<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: TaFloat> FromIterator<T> for Series<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().collect(),
        }
    }
}

impl<T: TaFloat> IntoIterator for Series<T> {
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: TaFloat> IntoIterator for &'a Series<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<T: TaFloat> From<Vec<T>> for Series<T> {
    fn from(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T: TaFloat> From<&[T]> for Series<T> {
    fn from(data: &[T]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_push() {
        let mut series: Series<f64> = Series::new();
        assert!(series.is_empty());
        assert_eq!(series.len(), 0);

        series.push(1.0);
        series.push(2.0);
        series.push(3.0);

        assert!(!series.is_empty());
        assert_eq!(series.len(), 3);
        assert_eq!(series[0], 1.0);
        assert_eq!(series[2], 3.0);
    }

    #[test]
    fn test_from_vec() {
        let series: Series<f64> = Series::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(series.len(), 3);
        assert_eq!(series.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_first_last() {
        let series: Series<f64> = Series::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(series.first(), Some(&1.0));
        assert_eq!(series.last(), Some(&3.0));

        let empty: Series<f64> = Series::new();
        assert_eq!(empty.first(), None);
        assert_eq!(empty.last(), None);
    }

    #[test]
    fn test_head_tail() {
        let series: Series<f64> = Series::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(series.head(3), &[1.0, 2.0, 3.0]);
        assert_eq!(series.tail(3), &[3.0, 4.0, 5.0]);
        assert_eq!(series.head(10), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(series.tail(10), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_diff() {
        let series: Series<f64> = Series::from_vec(vec![1.0, 3.0, 6.0, 10.0]);
        let diff = series.diff();

        assert_eq!(diff.len(), 4);
        assert!(diff[0].is_nan());
        assert_eq!(diff[1], 2.0);
        assert_eq!(diff[2], 3.0);
        assert_eq!(diff[3], 4.0);
    }

    #[test]
    fn test_shift_forward() {
        let series: Series<f64> = Series::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let shifted = series.shift(2, 0.0);

        assert_eq!(shifted.as_slice(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_shift_backward() {
        let series: Series<f64> = Series::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let shifted = series.shift(-2, 0.0);

        assert_eq!(shifted.as_slice(), &[3.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn test_fillna() {
        let mut series: Series<f64> = Series::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN]);
        series.fillna(0.0);

        assert_eq!(series.as_slice(), &[1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_ffill() {
        let series: Series<f64> = Series::from_vec(vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN]);
        let filled = series.ffill();

        assert_eq!(filled[0], 1.0);
        assert_eq!(filled[1], 1.0);
        assert_eq!(filled[2], 1.0);
        assert_eq!(filled[3], 4.0);
        assert_eq!(filled[4], 4.0);
    }

    #[test]
    fn test_clamp() {
        let series: Series<f64> = Series::from_vec(vec![-5.0, 0.0, 50.0, 150.0]);
        let clamped = series.clamp(0.0, 100.0);

        assert_eq!(clamped.as_slice(), &[0.0, 0.0, 50.0, 100.0]);
    }

    #[test]
    fn test_map() {
        let series: Series<f64> = Series::from_vec(vec![1.0, 2.0, 3.0]);
        let doubled = series.map(|x| x * 2.0);

        assert_eq!(doubled.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_statistics() {
        let series: Series<f64> = Series::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(series.sum(), 15.0);
        assert_eq!(series.mean(), 3.0);
        assert_eq!(series.min(), 1.0);
        assert_eq!(series.max(), 5.0);
    }

    #[test]
    fn test_statistics_with_nan() {
        let series: Series<f64> = Series::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);

        assert_eq!(series.sum(), 9.0);
        assert_eq!(series.mean(), 3.0);
        assert_eq!(series.min(), 1.0);
        assert_eq!(series.max(), 5.0);
        assert_eq!(series.nan_count(), 2);
        assert!(series.has_nan());
    }

    #[test]
    fn test_from_iterator() {
        let series: Series<f64> = (1..=5).map(|x| x as f64).collect();
        assert_eq!(series.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_into_iterator() {
        let series: Series<f64> = Series::from_vec(vec![1.0, 2.0, 3.0]);
        let sum: f64 = series.into_iter().sum();
        assert_eq!(sum, 6.0);
    }
}
