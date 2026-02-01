//! Numeric type abstractions for technical analysis computations.
//!
//! This module defines the [`TaFloat`] trait which abstracts over `f32` and `f64`
//! for generic numeric operations.

use core::cmp::Ordering;
use num_traits::{Float, FromPrimitive, ToPrimitive};

#[cfg(feature = "serde")]
use serde::{de::DeserializeOwned, Serialize};

/// Trait for floating-point types used in technical analysis calculations.
///
/// This trait provides a common interface for `f32` and `f64`, enabling generic
/// implementations of indicators and transforms.
///
/// # Associated Constants
///
/// - `EPSILON` - Machine epsilon for the type
/// - `NAN` - Not-a-number value
/// - `INFINITY` - Positive infinity
/// - `NEG_INFINITY` - Negative infinity
/// - `ZERO` - Zero value
/// - `ONE` - One value
/// - `TWO` - Two value
/// - `HUNDRED` - Hundred value (useful for RSI, etc.)
///
/// # Example
///
/// ```rust
/// use ta_core::TaFloat;
///
/// fn compute_rsi<T: TaFloat>(avg_gain: T, avg_loss: T) -> T {
///     if avg_loss == T::ZERO {
///         return T::HUNDRED;
///     }
///     let rs = avg_gain / avg_loss;
///     T::HUNDRED - T::HUNDRED / (T::ONE + rs)
/// }
/// ```
#[cfg(feature = "serde")]
pub trait TaFloat:
    Float + FromPrimitive + ToPrimitive + Copy + Send + Sync + Default + Serialize + DeserializeOwned + 'static
{
    /// Machine epsilon for this type.
    const EPSILON: Self;
    /// Not-a-number value.
    const NAN: Self;
    /// Positive infinity.
    const INFINITY: Self;
    /// Negative infinity.
    const NEG_INFINITY: Self;
    /// Zero value.
    const ZERO: Self;
    /// One value.
    const ONE: Self;
    /// Two value.
    const TWO: Self;
    /// Fifty value (useful for midpoint calculations).
    const FIFTY: Self;
    /// Hundred value (useful for percentage calculations like RSI).
    const HUNDRED: Self;

    /// Convert from `f64`.
    ///
    /// # Panics
    ///
    /// May panic if the value cannot be represented in the target type.
    #[must_use]
    fn from_f64_lossy(value: f64) -> Self;

    /// Convert to `f64`.
    #[must_use]
    fn to_f64_lossy(self) -> f64;

    /// Convert from `usize`.
    #[must_use]
    fn from_usize(value: usize) -> Self;

    /// Check if the value is valid (not NaN and not infinite).
    #[must_use]
    fn is_valid(self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    /// Total comparison that handles NaN consistently.
    ///
    /// NaN values are ordered after all other values (including +infinity).
    /// This ensures deterministic sorting behavior.
    #[must_use]
    fn total_cmp_fn(&self, other: &Self) -> Ordering;

    /// Clamp value to the range [min, max].
    ///
    /// If the value is NaN, returns NaN.
    #[must_use]
    fn clamp_value(self, min: Self, max: Self) -> Self {
        if self.is_nan() {
            return self;
        }
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    /// Compute the absolute difference between two values.
    #[must_use]
    fn abs_diff(self, other: Self) -> Self {
        (self - other).abs()
    }
}

#[cfg(not(feature = "serde"))]
pub trait TaFloat:
    Float + FromPrimitive + ToPrimitive + Copy + Send + Sync + Default + 'static
{
    /// Machine epsilon for this type.
    const EPSILON: Self;
    /// Not-a-number value.
    const NAN: Self;
    /// Positive infinity.
    const INFINITY: Self;
    /// Negative infinity.
    const NEG_INFINITY: Self;
    /// Zero value.
    const ZERO: Self;
    /// One value.
    const ONE: Self;
    /// Two value.
    const TWO: Self;
    /// Fifty value (useful for midpoint calculations).
    const FIFTY: Self;
    /// Hundred value (useful for percentage calculations like RSI).
    const HUNDRED: Self;

    /// Convert from `f64`.
    ///
    /// # Panics
    ///
    /// May panic if the value cannot be represented in the target type.
    #[must_use]
    fn from_f64_lossy(value: f64) -> Self;

    /// Convert to `f64`.
    #[must_use]
    fn to_f64_lossy(self) -> f64;

    /// Convert from `usize`.
    #[must_use]
    fn from_usize(value: usize) -> Self;

    /// Check if the value is valid (not NaN and not infinite).
    #[must_use]
    fn is_valid(self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    /// Total comparison that handles NaN consistently.
    ///
    /// NaN values are ordered after all other values (including +infinity).
    /// This ensures deterministic sorting behavior.
    #[must_use]
    fn total_cmp_fn(&self, other: &Self) -> Ordering;

    /// Clamp value to the range [min, max].
    ///
    /// If the value is NaN, returns NaN.
    #[must_use]
    fn clamp_value(self, min: Self, max: Self) -> Self {
        if self.is_nan() {
            return self;
        }
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    /// Compute the absolute difference between two values.
    #[must_use]
    fn abs_diff(self, other: Self) -> Self {
        (self - other).abs()
    }
}

impl TaFloat for f32 {
    const EPSILON: Self = f32::EPSILON;
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const FIFTY: Self = 50.0;
    const HUNDRED: Self = 100.0;

    #[inline]
    fn from_f64_lossy(value: f64) -> Self {
        value as f32
    }

    #[inline]
    fn to_f64_lossy(self) -> f64 {
        f64::from(self)
    }

    #[inline]
    fn from_usize(value: usize) -> Self {
        value as f32
    }

    #[inline]
    fn total_cmp_fn(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl TaFloat for f64 {
    const EPSILON: Self = f64::EPSILON;
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const FIFTY: Self = 50.0;
    const HUNDRED: Self = 100.0;

    #[inline]
    fn from_f64_lossy(value: f64) -> Self {
        value
    }

    #[inline]
    fn to_f64_lossy(self) -> f64 {
        self
    }

    #[inline]
    fn from_usize(value: usize) -> Self {
        value as f64
    }

    #[inline]
    fn total_cmp_fn(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_f64() {
        assert!(f64::EPSILON > 0.0);
        assert!(f64::NAN.is_nan());
        assert!(f64::INFINITY.is_infinite());
        assert!(f64::NEG_INFINITY.is_infinite());
        assert_eq!(f64::ZERO, 0.0);
        assert_eq!(f64::ONE, 1.0);
        assert_eq!(f64::TWO, 2.0);
        assert_eq!(f64::HUNDRED, 100.0);
    }

    #[test]
    fn test_constants_f32() {
        assert!(f32::EPSILON > 0.0);
        assert!(f32::NAN.is_nan());
        assert!(f32::INFINITY.is_infinite());
        assert!(f32::NEG_INFINITY.is_infinite());
        assert_eq!(f32::ZERO, 0.0);
        assert_eq!(f32::ONE, 1.0);
        assert_eq!(f32::TWO, 2.0);
        assert_eq!(f32::HUNDRED, 100.0);
    }

    #[test]
    fn test_from_f64_lossy() {
        assert_eq!(f64::from_f64_lossy(42.5), 42.5);
        assert_eq!(f32::from_f64_lossy(42.5), 42.5f32);
    }

    #[test]
    fn test_to_f64_lossy() {
        assert_eq!(42.5f64.to_f64_lossy(), 42.5);
        assert_eq!(42.5f32.to_f64_lossy(), 42.5);
    }

    #[test]
    fn test_from_usize() {
        assert_eq!(<f64 as TaFloat>::from_usize(42), 42.0);
        assert_eq!(<f32 as TaFloat>::from_usize(42), 42.0f32);
    }

    #[test]
    fn test_is_valid() {
        assert!(1.0f64.is_valid());
        assert!(0.0f64.is_valid());
        assert!(!f64::NAN.is_valid());
        assert!(!f64::INFINITY.is_valid());
        assert!(!f64::NEG_INFINITY.is_valid());
    }

    #[test]
    fn test_total_cmp_nan_ordering() {
        let mut values = vec![1.0f64, f64::NAN, 2.0, f64::NAN, 0.5];
        values.sort_by(|a, b| a.total_cmp_fn(b));

        // NaN should be at the end
        assert_eq!(values[0], 0.5);
        assert_eq!(values[1], 1.0);
        assert_eq!(values[2], 2.0);
        assert!(values[3].is_nan());
        assert!(values[4].is_nan());
    }

    #[test]
    fn test_clamp_value() {
        assert_eq!(5.0f64.clamp_value(0.0, 10.0), 5.0);
        assert_eq!((-5.0f64).clamp_value(0.0, 10.0), 0.0);
        assert_eq!(15.0f64.clamp_value(0.0, 10.0), 10.0);
        assert!(f64::NAN.clamp_value(0.0, 10.0).is_nan());
    }

    #[test]
    fn test_abs_diff() {
        assert_eq!(5.0f64.abs_diff(3.0), 2.0);
        assert_eq!(3.0f64.abs_diff(5.0), 2.0);
        assert_eq!(5.0f64.abs_diff(5.0), 0.0);
    }
}
