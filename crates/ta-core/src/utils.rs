//! Utility functions for common technical analysis calculations.
//!
//! These functions provide building blocks used by indicators, including
//! moving averages, rolling statistics, and signal operations.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::num::TaFloat;
use crate::series::Series;

/// Compute Simple Moving Average (SMA).
///
/// For each point, computes the mean of the previous `window` values.
/// The first `window - 1` values are NaN.
///
/// # Arguments
///
/// * `data` - Input data slice
/// * `window` - Number of periods for the moving average
///
/// # Example
///
/// ```rust
/// use ta_core::utils::sma;
///
/// let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = sma(&data, 3);
/// assert!(result[0].is_nan());
/// assert!(result[1].is_nan());
/// assert_eq!(result[2], 2.0); // (1+2+3)/3
/// assert_eq!(result[3], 3.0); // (2+3+4)/3
/// assert_eq!(result[4], 4.0); // (3+4+5)/3
/// ```
#[must_use]
pub fn sma<T: TaFloat>(data: &[T], window: usize) -> Series<T> {
    if data.is_empty() || window == 0 {
        return Series::new();
    }

    let mut result = Vec::with_capacity(data.len());

    // Fill with NaN until we have enough data
    for _ in 0..window.saturating_sub(1).min(data.len()) {
        result.push(T::NAN);
    }

    if data.len() < window {
        return Series::from_vec(result);
    }

    // Compute initial sum
    let mut sum = T::ZERO;
    for i in 0..window {
        sum = sum + data[i];
    }
    result.push(sum / TaFloat::from_usize(window));

    // Sliding window
    for i in window..data.len() {
        sum = sum - data[i - window] + data[i];
        result.push(sum / TaFloat::from_usize(window));
    }

    Series::from_vec(result)
}

/// Compute Exponential Moving Average (EMA).
///
/// Uses SMA of the first `window` values for initialization, then applies
/// the EMA formula: `EMA(t) = α * P(t) + (1 - α) * EMA(t-1)` where `α = 2 / (window + 1)`.
///
/// # Arguments
///
/// * `data` - Input data slice
/// * `window` - Number of periods for the EMA
///
/// # Example
///
/// ```rust
/// use ta_core::utils::ema;
///
/// let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = ema(&data, 3);
/// // First 2 values are NaN, third is SMA, rest are EMA
/// ```
#[must_use]
pub fn ema<T: TaFloat>(data: &[T], window: usize) -> Series<T> {
    if data.is_empty() || window == 0 {
        return Series::new();
    }

    let mut result = Vec::with_capacity(data.len());
    let alpha = T::TWO / TaFloat::from_usize(window + 1);
    let one_minus_alpha = T::ONE - alpha;

    // Fill with NaN until we have enough data for initial SMA
    for _ in 0..window.saturating_sub(1).min(data.len()) {
        result.push(T::NAN);
    }

    if data.len() < window {
        return Series::from_vec(result);
    }

    // Initial EMA value = SMA of first window values
    let mut ema_value = T::ZERO;
    for i in 0..window {
        ema_value = ema_value + data[i];
    }
    ema_value = ema_value / TaFloat::from_usize(window);
    result.push(ema_value);

    // Apply EMA formula for remaining values
    for i in window..data.len() {
        ema_value = alpha * data[i] + one_minus_alpha * ema_value;
        result.push(ema_value);
    }

    Series::from_vec(result)
}

/// Compute Wilder's Smoothing (used in RSI, ATR).
///
/// Similar to EMA but uses `α = 1/n` instead of `α = 2/(n+1)`.
/// This is equivalent to an EMA with period `2n - 1`.
///
/// # Arguments
///
/// * `data` - Input data slice
/// * `window` - Number of periods
#[must_use]
pub fn wilder_smooth<T: TaFloat>(data: &[T], window: usize) -> Series<T> {
    if data.is_empty() || window == 0 {
        return Series::new();
    }

    let mut result = Vec::with_capacity(data.len());
    let alpha = T::ONE / TaFloat::from_usize(window);
    let one_minus_alpha = T::ONE - alpha;

    // Fill with NaN until we have enough data
    for _ in 0..window.saturating_sub(1).min(data.len()) {
        result.push(T::NAN);
    }

    if data.len() < window {
        return Series::from_vec(result);
    }

    // Initial value = SMA of first window values
    let mut smooth_value = T::ZERO;
    for i in 0..window {
        smooth_value = smooth_value + data[i];
    }
    smooth_value = smooth_value / TaFloat::from_usize(window);
    result.push(smooth_value);

    // Apply Wilder's smoothing formula
    for i in window..data.len() {
        smooth_value = alpha * data[i] + one_minus_alpha * smooth_value;
        result.push(smooth_value);
    }

    Series::from_vec(result)
}

/// Compute True Range for OHLCV data.
///
/// TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
/// First bar uses High - Low since there's no previous close.
#[must_use]
pub fn true_range<T: TaFloat>(high: &[T], low: &[T], close: &[T]) -> Series<T> {
    let len = high.len().min(low.len()).min(close.len());
    if len == 0 {
        return Series::new();
    }

    let mut result = Vec::with_capacity(len);

    // First bar: just high - low
    result.push(high[0] - low[0]);

    // Remaining bars
    for i in 1..len {
        let prev_close = close[i - 1];
        let hl = high[i] - low[i];
        let hc = (high[i] - prev_close).abs();
        let lc = (low[i] - prev_close).abs();

        let tr = hl.max(hc).max(lc);
        result.push(tr);
    }

    Series::from_vec(result)
}

/// Compute rolling sum over a window.
#[must_use]
pub fn rolling_sum<T: TaFloat>(data: &[T], window: usize) -> Series<T> {
    if data.is_empty() || window == 0 {
        return Series::new();
    }

    let mut result = Vec::with_capacity(data.len());

    // Fill with NaN until we have enough data
    for _ in 0..window.saturating_sub(1).min(data.len()) {
        result.push(T::NAN);
    }

    if data.len() < window {
        return Series::from_vec(result);
    }

    // Compute initial sum
    let mut sum = T::ZERO;
    for i in 0..window {
        sum = sum + data[i];
    }
    result.push(sum);

    // Sliding window
    for i in window..data.len() {
        sum = sum - data[i - window] + data[i];
        result.push(sum);
    }

    Series::from_vec(result)
}

/// Compute rolling mean over a window (alias for SMA).
#[must_use]
pub fn rolling_mean<T: TaFloat>(data: &[T], window: usize) -> Series<T> {
    sma(data, window)
}

/// Compute rolling standard deviation over a window.
///
/// Uses sample standard deviation (Bessel's correction with n-1 denominator).
///
/// # Arguments
///
/// * `data` - Input data slice
/// * `window` - Number of periods
/// * `ddof` - Delta degrees of freedom (typically 1 for sample std)
#[must_use]
pub fn rolling_std<T: TaFloat>(data: &[T], window: usize, ddof: usize) -> Series<T> {
    rolling_variance(data, window, ddof).map(|x| x.sqrt())
}

/// Compute rolling variance over a window.
///
/// # Arguments
///
/// * `data` - Input data slice
/// * `window` - Number of periods
/// * `ddof` - Delta degrees of freedom (typically 1 for sample variance)
#[must_use]
pub fn rolling_variance<T: TaFloat>(data: &[T], window: usize, ddof: usize) -> Series<T> {
    if data.is_empty() || window == 0 || window <= ddof {
        return Series::new();
    }

    let means = sma(data, window);
    let mut result = Vec::with_capacity(data.len());

    // Fill with NaN until we have enough data
    for _ in 0..window.saturating_sub(1).min(data.len()) {
        result.push(T::NAN);
    }

    if data.len() < window {
        return Series::from_vec(result);
    }

    let divisor = TaFloat::from_usize(window - ddof);

    for i in (window - 1)..data.len() {
        let mean = means[i];
        let mut sum_sq = T::ZERO;

        for j in (i + 1 - window)..=i {
            let diff = data[j] - mean;
            sum_sq = sum_sq + diff * diff;
        }

        result.push(sum_sq / divisor);
    }

    Series::from_vec(result)
}

/// Compute rolling maximum over a window.
#[must_use]
pub fn rolling_max<T: TaFloat>(data: &[T], window: usize) -> Series<T> {
    if data.is_empty() || window == 0 {
        return Series::new();
    }

    let mut result = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        if i + 1 < window {
            result.push(T::NAN);
        } else {
            let start = i + 1 - window;
            let mut max = data[start];
            for j in (start + 1)..=i {
                if data[j] > max {
                    max = data[j];
                }
            }
            result.push(max);
        }
    }

    Series::from_vec(result)
}

/// Compute rolling minimum over a window.
#[must_use]
pub fn rolling_min<T: TaFloat>(data: &[T], window: usize) -> Series<T> {
    if data.is_empty() || window == 0 {
        return Series::new();
    }

    let mut result = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        if i + 1 < window {
            result.push(T::NAN);
        } else {
            let start = i + 1 - window;
            let mut min = data[start];
            for j in (start + 1)..=i {
                if data[j] < min {
                    min = data[j];
                }
            }
            result.push(min);
        }
    }

    Series::from_vec(result)
}

/// Compute first differences: `y[i] = x[i] - x[i-periods]`.
#[must_use]
pub fn diff<T: TaFloat>(data: &[T], periods: usize) -> Series<T> {
    if data.is_empty() || periods == 0 {
        return Series::from_vec(data.to_vec());
    }

    let mut result = Vec::with_capacity(data.len());

    // Fill with NaN for first `periods` values
    for _ in 0..periods.min(data.len()) {
        result.push(T::NAN);
    }

    // Compute differences
    for i in periods..data.len() {
        result.push(data[i] - data[i - periods]);
    }

    Series::from_vec(result)
}

/// Shift data by n positions.
///
/// Positive n shifts forward (newer values at the start become fill).
/// Negative n shifts backward (older values at the end become fill).
#[must_use]
pub fn shift<T: TaFloat>(data: &[T], periods: isize, fill: T) -> Series<T> {
    if data.is_empty() {
        return Series::new();
    }

    let len = data.len();
    let mut result = vec![fill; len];

    if periods >= 0 {
        let p = periods as usize;
        if p < len {
            result[p..].copy_from_slice(&data[..len - p]);
        }
    } else {
        let p = (-periods) as usize;
        if p < len {
            result[..len - p].copy_from_slice(&data[p..]);
        }
    }

    Series::from_vec(result)
}

/// Compute percentage change: `(x[i] - x[i-periods]) / x[i-periods]`.
#[must_use]
pub fn pct_change<T: TaFloat>(data: &[T], periods: usize) -> Series<T> {
    if data.is_empty() || periods == 0 {
        return Series::nan(data.len());
    }

    let mut result = Vec::with_capacity(data.len());

    // Fill with NaN for first `periods` values
    for _ in 0..periods.min(data.len()) {
        result.push(T::NAN);
    }

    // Compute percentage changes
    for i in periods..data.len() {
        let prev = data[i - periods];
        if prev == T::ZERO {
            result.push(T::NAN);
        } else {
            result.push((data[i] - prev) / prev);
        }
    }

    Series::from_vec(result)
}

/// Detect crossover: returns true where `a` crosses above `b`.
#[must_use]
pub fn crossover<T: TaFloat>(a: &[T], b: &[T]) -> Vec<bool> {
    let len = a.len().min(b.len());
    let mut result = Vec::with_capacity(len);

    if len == 0 {
        return result;
    }

    result.push(false); // First value can't be a crossover

    for i in 1..len {
        let crossed = a[i - 1] <= b[i - 1] && a[i] > b[i];
        result.push(crossed);
    }

    result
}

/// Detect crossunder: returns true where `a` crosses below `b`.
#[must_use]
pub fn crossunder<T: TaFloat>(a: &[T], b: &[T]) -> Vec<bool> {
    let len = a.len().min(b.len());
    let mut result = Vec::with_capacity(len);

    if len == 0 {
        return result;
    }

    result.push(false); // First value can't be a crossunder

    for i in 1..len {
        let crossed = a[i - 1] >= b[i - 1] && a[i] < b[i];
        result.push(crossed);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_basic() {
        let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 3.0);
        assert_eq!(result[4], 4.0);
    }

    #[test]
    fn test_sma_window_equals_length() {
        let data: [f64; 3] = [1.0, 2.0, 3.0];
        let result = sma(&data, 3);

        assert_eq!(result.len(), 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0);
    }

    #[test]
    fn test_sma_window_larger_than_length() {
        let data: [f64; 2] = [1.0, 2.0];
        let result = sma(&data, 5);

        assert_eq!(result.len(), 2);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
    }

    #[test]
    fn test_ema_basic() {
        let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&data, 3);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0); // Initial SMA

        // EMA with alpha = 2/(3+1) = 0.5
        // ema[3] = 0.5 * 4 + 0.5 * 2 = 3
        assert_eq!(result[3], 3.0);
        // ema[4] = 0.5 * 5 + 0.5 * 3 = 4
        assert_eq!(result[4], 4.0);
    }

    #[test]
    fn test_wilder_smooth() {
        let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wilder_smooth(&data, 3);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0); // Initial SMA

        // Wilder's with alpha = 1/3
        // ws[3] = (1/3) * 4 + (2/3) * 2 = 4/3 + 4/3 = 8/3 ≈ 2.667
        assert!((result[3] - 8.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_true_range() {
        let high: [f64; 3] = [10.0, 12.0, 11.0];
        let low: [f64; 3] = [8.0, 9.0, 7.0];
        let close: [f64; 3] = [9.0, 11.0, 8.0];

        let result = true_range(&high, &low, &close);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 2.0); // 10 - 8

        // Bar 1: max(12-9, |12-9|, |9-9|) = max(3, 3, 0) = 3
        assert_eq!(result[1], 3.0);

        // Bar 2: max(11-7, |11-11|, |7-11|) = max(4, 0, 4) = 4
        assert_eq!(result[2], 4.0);
    }

    #[test]
    fn test_rolling_sum() {
        let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_sum(&data, 3);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 6.0);
        assert_eq!(result[3], 9.0);
        assert_eq!(result[4], 12.0);
    }

    #[test]
    fn test_rolling_std() {
        let data: [f64; 5] = [2.0, 4.0, 4.0, 4.0, 5.0];
        let result = rolling_std(&data, 3, 1);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // Values: 2, 4, 4 -> mean = 10/3
        // Variance = ((2-10/3)^2 + (4-10/3)^2 + (4-10/3)^2) / 2
        // = (16/9 + 4/9 + 4/9) / 2 = 24/9 / 2 = 4/3
        // Std = sqrt(4/3)
        let expected = (4.0f64 / 3.0).sqrt();
        assert!((result[2] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_max() {
        let data: [f64; 5] = [1.0, 3.0, 2.0, 5.0, 4.0];
        let result = rolling_max(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 5.0);
        assert_eq!(result[4], 5.0);
    }

    #[test]
    fn test_rolling_min() {
        let data: [f64; 5] = [5.0, 3.0, 4.0, 1.0, 2.0];
        let result = rolling_min(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 1.0);
    }

    #[test]
    fn test_diff() {
        let data: [f64; 4] = [1.0, 3.0, 6.0, 10.0];
        let result = diff(&data, 1);

        assert!(result[0].is_nan());
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 4.0);
    }

    #[test]
    fn test_shift_forward() {
        let data: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        let result = shift(&data, 2, 0.0);

        assert_eq!(result.as_slice(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_shift_backward() {
        let data: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        let result = shift(&data, -2, 0.0);

        assert_eq!(result.as_slice(), &[3.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn test_pct_change() {
        let data: [f64; 3] = [100.0, 110.0, 99.0];
        let result = pct_change(&data, 1);

        assert!(result[0].is_nan());
        assert_eq!(result[1], 0.1); // (110-100)/100
        assert_eq!(result[2], -0.1); // (99-110)/110
    }

    #[test]
    fn test_crossover() {
        let a = [1.0, 2.0, 3.0, 2.0, 3.0];
        let b = [2.0, 2.0, 2.0, 2.0, 2.0];

        let result = crossover(&a, &b);

        assert!(!result[0]); // First is always false
        assert!(!result[1]); // 1->2 doesn't cross above 2
        assert!(result[2]); // 2->3 crosses above 2
        assert!(!result[3]); // 3->2 doesn't cross above 2
        assert!(result[4]); // 2->3 crosses above 2
    }

    #[test]
    fn test_crossunder() {
        let a = [3.0, 2.0, 1.0, 2.0, 1.0];
        let b = [2.0, 2.0, 2.0, 2.0, 2.0];

        let result = crossunder(&a, &b);

        assert!(!result[0]);
        assert!(!result[1]); // 3->2 doesn't cross below
        assert!(result[2]); // 2->1 crosses below
        assert!(!result[3]);
        assert!(result[4]); // 2->1 crosses below
    }
}
