//! OHLCV (Open, High, Low, Close, Volume) data types.
//!
//! This module provides types for representing price bar data commonly used in
//! financial technical analysis.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::num::TaFloat;
use crate::series::Series;

/// A single OHLCV price bar.
///
/// Represents a single time period's price action with open, high, low, close
/// prices and trading volume.
///
/// # Invariants
///
/// A valid bar satisfies:
/// - `low <= open <= high`
/// - `low <= close <= high`
/// - `low <= high`
/// - All prices are positive (or zero)
/// - Volume is non-negative
///
/// # Example
///
/// ```rust
/// use ta_core::Bar;
///
/// let bar = Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0);
/// assert!(bar.is_valid());
/// assert_eq!(bar.typical_price(), (105.0 + 98.0 + 103.0) / 3.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct Bar<T: TaFloat> {
    /// Opening price for the period.
    pub open: T,
    /// Highest price during the period.
    pub high: T,
    /// Lowest price during the period.
    pub low: T,
    /// Closing price for the period.
    pub close: T,
    /// Trading volume during the period.
    pub volume: T,
    /// Optional timestamp (Unix epoch in milliseconds).
    pub timestamp: Option<i64>,
}

impl<T: TaFloat> Bar<T> {
    /// Create a new bar without a timestamp.
    #[must_use]
    pub fn new(open: T, high: T, low: T, close: T, volume: T) -> Self {
        Self {
            open,
            high,
            low,
            close,
            volume,
            timestamp: None,
        }
    }

    /// Create a new bar with a timestamp.
    #[must_use]
    pub fn with_timestamp(
        open: T,
        high: T,
        low: T,
        close: T,
        volume: T,
        timestamp: i64,
    ) -> Self {
        Self {
            open,
            high,
            low,
            close,
            volume,
            timestamp: Some(timestamp),
        }
    }

    /// Set the timestamp on an existing bar (builder pattern).
    #[must_use]
    pub fn timestamp(mut self, ts: i64) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Check if the bar satisfies OHLCV invariants.
    ///
    /// A valid bar has:
    /// - `low <= open <= high`
    /// - `low <= close <= high`
    /// - Non-negative volume
    /// - All values are valid (not NaN, not infinite)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.open.is_valid()
            && self.high.is_valid()
            && self.low.is_valid()
            && self.close.is_valid()
            && self.volume.is_valid()
            && self.low <= self.open
            && self.open <= self.high
            && self.low <= self.close
            && self.close <= self.high
            && self.volume >= T::ZERO
    }

    /// Compute typical price: (High + Low + Close) / 3.
    #[must_use]
    pub fn typical_price(&self) -> T {
        (self.high + self.low + self.close) / TaFloat::from_usize(3)
    }

    /// Compute median price: (High + Low) / 2.
    #[must_use]
    pub fn median_price(&self) -> T {
        (self.high + self.low) / T::TWO
    }

    /// Compute weighted close: (High + Low + Close + Close) / 4.
    #[must_use]
    pub fn weighted_close(&self) -> T {
        (self.high + self.low + self.close + self.close) / TaFloat::from_usize(4)
    }

    /// Compute the bar's range: High - Low.
    #[must_use]
    pub fn range(&self) -> T {
        self.high - self.low
    }

    /// Compute the bar's body: |Close - Open|.
    #[must_use]
    pub fn body(&self) -> T {
        (self.close - self.open).abs()
    }

    /// Check if the bar is bullish (close > open).
    #[must_use]
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if the bar is bearish (close < open).
    #[must_use]
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Compute the upper shadow: High - max(Open, Close).
    #[must_use]
    pub fn upper_shadow(&self) -> T {
        let body_top = if self.close > self.open {
            self.close
        } else {
            self.open
        };
        self.high - body_top
    }

    /// Compute the lower shadow: min(Open, Close) - Low.
    #[must_use]
    pub fn lower_shadow(&self) -> T {
        let body_bottom = if self.close < self.open {
            self.close
        } else {
            self.open
        };
        body_bottom - self.low
    }
}

impl<T: TaFloat> Default for Bar<T> {
    fn default() -> Self {
        Self {
            open: T::ZERO,
            high: T::ZERO,
            low: T::ZERO,
            close: T::ZERO,
            volume: T::ZERO,
            timestamp: None,
        }
    }
}

/// A columnar storage of OHLCV data.
///
/// `OhlcvSeries` stores price data in a columnar format, which is more efficient
/// for vectorized computations and reduces memory overhead compared to storing
/// individual `Bar` objects.
///
/// # Example
///
/// ```rust
/// use ta_core::{Bar, OhlcvSeries};
///
/// let mut ohlcv: OhlcvSeries<f64> = OhlcvSeries::new();
/// ohlcv.push(Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0));
/// ohlcv.push(Bar::new(103.0, 108.0, 101.0, 107.0, 1_200_000.0));
///
/// assert_eq!(ohlcv.len(), 2);
/// assert_eq!(ohlcv.close()[1], 107.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct OhlcvSeries<T: TaFloat> {
    /// Open prices.
    open: Series<T>,
    /// High prices.
    high: Series<T>,
    /// Low prices.
    low: Series<T>,
    /// Close prices.
    close: Series<T>,
    /// Volumes.
    volume: Series<T>,
    /// Optional timestamps (Unix epoch in milliseconds).
    timestamps: Option<Vec<i64>>,
}

impl<T: TaFloat> Default for OhlcvSeries<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: TaFloat> OhlcvSeries<T> {
    /// Create a new empty OHLCV series.
    #[must_use]
    pub fn new() -> Self {
        Self {
            open: Series::new(),
            high: Series::new(),
            low: Series::new(),
            close: Series::new(),
            volume: Series::new(),
            timestamps: None,
        }
    }

    /// Create a new OHLCV series with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            open: Series::with_capacity(capacity),
            high: Series::with_capacity(capacity),
            low: Series::with_capacity(capacity),
            close: Series::with_capacity(capacity),
            volume: Series::with_capacity(capacity),
            timestamps: None,
        }
    }

    /// Create OHLCV series from individual series.
    ///
    /// # Panics
    ///
    /// Panics if the series have different lengths.
    #[must_use]
    pub fn from_series(
        open: Series<T>,
        high: Series<T>,
        low: Series<T>,
        close: Series<T>,
        volume: Series<T>,
    ) -> Self {
        let len = open.len();
        assert_eq!(high.len(), len, "High series length mismatch");
        assert_eq!(low.len(), len, "Low series length mismatch");
        assert_eq!(close.len(), len, "Close series length mismatch");
        assert_eq!(volume.len(), len, "Volume series length mismatch");

        Self {
            open,
            high,
            low,
            close,
            volume,
            timestamps: None,
        }
    }

    /// Returns the number of bars in the series.
    #[must_use]
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Returns `true` if the series is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    /// Append a bar to the series.
    pub fn push(&mut self, bar: Bar<T>) {
        self.open.push(bar.open);
        self.high.push(bar.high);
        self.low.push(bar.low);
        self.close.push(bar.close);
        self.volume.push(bar.volume);

        if let Some(ts) = bar.timestamp {
            if self.timestamps.is_none() {
                // Initialize timestamps vector with NaN placeholders for previous bars
                let mut timestamps = vec![0; self.len() - 1];
                timestamps.push(ts);
                self.timestamps = Some(timestamps);
            } else {
                self.timestamps.as_mut().unwrap().push(ts);
            }
        } else if let Some(ref mut timestamps) = self.timestamps {
            timestamps.push(0); // Use 0 as placeholder for missing timestamp
        }
    }

    /// Get a bar at the specified index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Bar<T>> {
        if index >= self.len() {
            return None;
        }

        Some(Bar {
            open: self.open[index],
            high: self.high[index],
            low: self.low[index],
            close: self.close[index],
            volume: self.volume[index],
            timestamp: self.timestamps.as_ref().map(|ts| ts[index]),
        })
    }

    /// Get the last bar, if any.
    #[must_use]
    pub fn last(&self) -> Option<Bar<T>> {
        if self.is_empty() {
            None
        } else {
            self.get(self.len() - 1)
        }
    }

    /// Returns a reference to the open prices series.
    #[must_use]
    pub fn open(&self) -> &Series<T> {
        &self.open
    }

    /// Returns a reference to the high prices series.
    #[must_use]
    pub fn high(&self) -> &Series<T> {
        &self.high
    }

    /// Returns a reference to the low prices series.
    #[must_use]
    pub fn low(&self) -> &Series<T> {
        &self.low
    }

    /// Returns a reference to the close prices series.
    #[must_use]
    pub fn close(&self) -> &Series<T> {
        &self.close
    }

    /// Returns a reference to the volume series.
    #[must_use]
    pub fn volume(&self) -> &Series<T> {
        &self.volume
    }

    /// Returns a reference to the timestamps, if any.
    #[must_use]
    pub fn timestamps(&self) -> Option<&[i64]> {
        self.timestamps.as_deref()
    }

    /// Compute typical prices for all bars.
    #[must_use]
    pub fn typical_prices(&self) -> Series<T> {
        let data: Vec<T> = (0..self.len())
            .map(|i| {
                (self.high[i] + self.low[i] + self.close[i]) / TaFloat::from_usize(3)
            })
            .collect();
        Series::from_vec(data)
    }

    /// Compute median prices for all bars.
    #[must_use]
    pub fn median_prices(&self) -> Series<T> {
        let data: Vec<T> = (0..self.len())
            .map(|i| (self.high[i] + self.low[i]) / T::TWO)
            .collect();
        Series::from_vec(data)
    }

    /// Returns an iterator over the bars.
    pub fn iter(&self) -> impl Iterator<Item = Bar<T>> + '_ {
        (0..self.len()).map(move |i| self.get(i).unwrap())
    }

    /// Clear all data from the series.
    pub fn clear(&mut self) {
        self.open.clear();
        self.high.clear();
        self.low.clear();
        self.close.clear();
        self.volume.clear();
        self.timestamps = None;
    }
}

impl<T: TaFloat> FromIterator<Bar<T>> for OhlcvSeries<T> {
    fn from_iter<I: IntoIterator<Item = Bar<T>>>(iter: I) -> Self {
        let mut series = Self::new();
        for bar in iter {
            series.push(bar);
        }
        series
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bar_creation() {
        let bar = Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0);
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 105.0);
        assert_eq!(bar.low, 98.0);
        assert_eq!(bar.close, 103.0);
        assert_eq!(bar.volume, 1_000_000.0);
        assert!(bar.timestamp.is_none());
    }

    #[test]
    fn test_bar_with_timestamp() {
        let bar = Bar::with_timestamp(100.0, 105.0, 98.0, 103.0, 1_000_000.0, 1234567890);
        assert_eq!(bar.timestamp, Some(1234567890));

        let bar2 = Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0).timestamp(1234567890);
        assert_eq!(bar2.timestamp, Some(1234567890));
    }

    #[test]
    fn test_bar_is_valid() {
        let valid_bar = Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0);
        assert!(valid_bar.is_valid());

        // Invalid: high < open
        let invalid_bar = Bar::new(100.0, 95.0, 98.0, 103.0, 1_000_000.0);
        assert!(!invalid_bar.is_valid());

        // Invalid: negative volume
        let invalid_bar = Bar::new(100.0, 105.0, 98.0, 103.0, -1000.0);
        assert!(!invalid_bar.is_valid());

        // Invalid: NaN value
        let invalid_bar = Bar::new(f64::NAN, 105.0, 98.0, 103.0, 1_000_000.0);
        assert!(!invalid_bar.is_valid());
    }

    #[test]
    fn test_bar_typical_price() {
        let bar = Bar::new(100.0, 105.0, 95.0, 100.0, 1_000_000.0);
        // Typical = (105 + 95 + 100) / 3 = 100
        assert_eq!(bar.typical_price(), 100.0);
    }

    #[test]
    fn test_bar_median_price() {
        let bar = Bar::new(100.0, 106.0, 94.0, 100.0, 1_000_000.0);
        // Median = (106 + 94) / 2 = 100
        assert_eq!(bar.median_price(), 100.0);
    }

    #[test]
    fn test_bar_bullish_bearish() {
        let bullish = Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0);
        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());

        let bearish = Bar::new(103.0, 105.0, 98.0, 100.0, 1_000_000.0);
        assert!(!bearish.is_bullish());
        assert!(bearish.is_bearish());
    }

    #[test]
    fn test_ohlcv_series_push_and_get() {
        let mut ohlcv: OhlcvSeries<f64> = OhlcvSeries::new();
        assert!(ohlcv.is_empty());

        ohlcv.push(Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0));
        ohlcv.push(Bar::new(103.0, 108.0, 101.0, 107.0, 1_200_000.0));

        assert_eq!(ohlcv.len(), 2);
        assert!(!ohlcv.is_empty());

        let bar0 = ohlcv.get(0).unwrap();
        assert_eq!(bar0.close, 103.0);

        let bar1 = ohlcv.get(1).unwrap();
        assert_eq!(bar1.close, 107.0);

        assert!(ohlcv.get(2).is_none());
    }

    #[test]
    fn test_ohlcv_series_accessors() {
        let mut ohlcv: OhlcvSeries<f64> = OhlcvSeries::new();
        ohlcv.push(Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0));
        ohlcv.push(Bar::new(103.0, 108.0, 101.0, 107.0, 1_200_000.0));

        assert_eq!(ohlcv.open().as_slice(), &[100.0, 103.0]);
        assert_eq!(ohlcv.high().as_slice(), &[105.0, 108.0]);
        assert_eq!(ohlcv.low().as_slice(), &[98.0, 101.0]);
        assert_eq!(ohlcv.close().as_slice(), &[103.0, 107.0]);
        assert_eq!(ohlcv.volume().as_slice(), &[1_000_000.0, 1_200_000.0]);
    }

    #[test]
    fn test_ohlcv_series_last() {
        let mut ohlcv: OhlcvSeries<f64> = OhlcvSeries::new();
        assert!(ohlcv.last().is_none());

        ohlcv.push(Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0));
        let last = ohlcv.last().unwrap();
        assert_eq!(last.close, 103.0);
    }

    #[test]
    fn test_ohlcv_series_typical_prices() {
        let mut ohlcv: OhlcvSeries<f64> = OhlcvSeries::new();
        ohlcv.push(Bar::new(100.0, 105.0, 95.0, 100.0, 1_000_000.0));
        ohlcv.push(Bar::new(100.0, 110.0, 90.0, 100.0, 1_000_000.0));

        let typical = ohlcv.typical_prices();
        assert_eq!(typical[0], 100.0); // (105 + 95 + 100) / 3
        assert_eq!(typical[1], 100.0); // (110 + 90 + 100) / 3
    }

    #[test]
    fn test_ohlcv_series_from_iterator() {
        let bars = vec![
            Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0),
            Bar::new(103.0, 108.0, 101.0, 107.0, 1_200_000.0),
        ];
        let ohlcv: OhlcvSeries<f64> = bars.into_iter().collect();

        assert_eq!(ohlcv.len(), 2);
        assert_eq!(ohlcv.close()[0], 103.0);
        assert_eq!(ohlcv.close()[1], 107.0);
    }

    #[test]
    fn test_ohlcv_series_iter() {
        let mut ohlcv: OhlcvSeries<f64> = OhlcvSeries::new();
        ohlcv.push(Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0));
        ohlcv.push(Bar::new(103.0, 108.0, 101.0, 107.0, 1_200_000.0));

        let closes: Vec<f64> = ohlcv.iter().map(|b| b.close).collect();
        assert_eq!(closes, vec![103.0, 107.0]);
    }
}
