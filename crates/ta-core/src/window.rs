//! Circular buffer for streaming calculations.
//!
//! The [`RingBuffer`] type provides a fixed-capacity circular buffer optimized
//! for rolling window calculations in streaming indicators.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::num::TaFloat;

/// A fixed-capacity circular buffer for streaming calculations.
///
/// `RingBuffer` is optimized for rolling window operations where you need to:
/// - Maintain a fixed window of recent values
/// - Add new values with O(1) complexity
/// - Compute statistics (sum, mean) over the window
///
/// When the buffer is full, adding a new value automatically removes the oldest.
///
/// # Example
///
/// ```rust
/// use ta_core::RingBuffer;
///
/// let mut buffer: RingBuffer<f64> = RingBuffer::new(3);
///
/// buffer.push(1.0);
/// buffer.push(2.0);
/// buffer.push(3.0);
/// assert_eq!(buffer.mean(), 2.0);
///
/// buffer.push(4.0); // Removes 1.0
/// assert_eq!(buffer.mean(), 3.0);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: TaFloat"))]
pub struct RingBuffer<T: TaFloat> {
    /// Internal storage.
    buffer: Vec<T>,
    /// Index of the next write position (also the oldest element when full).
    head: usize,
    /// Number of elements currently in the buffer.
    len: usize,
    /// Maximum capacity of the buffer.
    capacity: usize,
    /// Running sum for O(1) mean calculation.
    sum: T,
}

impl<T: TaFloat> RingBuffer<T> {
    /// Create a new ring buffer with the specified capacity.
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        Self {
            buffer: vec![T::ZERO; capacity],
            head: 0,
            len: 0,
            capacity,
            sum: T::ZERO,
        }
    }

    /// Push a value into the buffer.
    ///
    /// If the buffer is full, the oldest value is replaced.
    /// Returns the replaced value if the buffer was full, None otherwise.
    pub fn push(&mut self, value: T) -> Option<T> {
        let old = if self.is_full() {
            let old_value = self.buffer[self.head];
            self.sum = self.sum - old_value + value;
            Some(old_value)
        } else {
            self.sum = self.sum + value;
            self.len += 1;
            None
        };

        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        old
    }

    /// Returns `true` if the buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Returns the number of elements in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity of the buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get a value by index (0 = oldest).
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }

        // Calculate actual index in the buffer
        let actual_index = if self.is_full() {
            (self.head + index) % self.capacity
        } else {
            index
        };

        Some(&self.buffer[actual_index])
    }

    /// Get the oldest value in the buffer.
    #[must_use]
    pub fn oldest(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        if self.is_full() {
            Some(&self.buffer[self.head])
        } else {
            Some(&self.buffer[0])
        }
    }

    /// Get the newest value in the buffer.
    #[must_use]
    pub fn newest(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        let index = if self.head == 0 {
            self.capacity - 1
        } else {
            self.head - 1
        };

        Some(&self.buffer[index])
    }

    /// Compute the sum of all values in the buffer.
    #[must_use]
    pub fn sum(&self) -> T {
        self.sum
    }

    /// Compute the mean of all values in the buffer.
    ///
    /// Returns NaN if the buffer is empty.
    #[must_use]
    pub fn mean(&self) -> T {
        if self.is_empty() {
            return T::NAN;
        }
        self.sum / TaFloat::from_usize(self.len)
    }

    /// Compute the variance of values in the buffer.
    ///
    /// Uses the two-pass algorithm for numerical stability.
    /// Returns NaN if the buffer is empty.
    #[must_use]
    pub fn variance(&self) -> T {
        if self.len < 2 {
            return T::NAN;
        }

        let mean = self.mean();
        let mut sum_sq = T::ZERO;

        for i in 0..self.len {
            let diff = *self.get(i).unwrap() - mean;
            sum_sq = sum_sq + diff * diff;
        }

        sum_sq / TaFloat::from_usize(self.len - 1) // Sample variance (Bessel's correction)
    }

    /// Compute the standard deviation of values in the buffer.
    #[must_use]
    pub fn std(&self) -> T {
        self.variance().sqrt()
    }

    /// Find the minimum value in the buffer.
    #[must_use]
    pub fn min(&self) -> T {
        if self.is_empty() {
            return T::NAN;
        }

        let mut min = T::INFINITY;
        for i in 0..self.len {
            let val = *self.get(i).unwrap();
            if val < min {
                min = val;
            }
        }
        min
    }

    /// Find the maximum value in the buffer.
    #[must_use]
    pub fn max(&self) -> T {
        if self.is_empty() {
            return T::NAN;
        }

        let mut max = T::NEG_INFINITY;
        for i in 0..self.len {
            let val = *self.get(i).unwrap();
            if val > max {
                max = val;
            }
        }
        max
    }

    /// Clear all values from the buffer.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
        self.sum = T::ZERO;
    }

    /// Returns an iterator over the values from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        RingBufferIter {
            buffer: self,
            index: 0,
        }
    }
}

/// Iterator over ring buffer values.
struct RingBufferIter<'a, T: TaFloat> {
    buffer: &'a RingBuffer<T>,
    index: usize,
}

impl<'a, T: TaFloat> Iterator for RingBufferIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.buffer.get(self.index);
        if item.is_some() {
            self.index += 1;
        }
        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buffer.len.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a, T: TaFloat> ExactSizeIterator for RingBufferIter<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer() {
        let buffer: RingBuffer<f64> = RingBuffer::new(5);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 5);
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_new_zero_capacity() {
        let _: RingBuffer<f64> = RingBuffer::new(0);
    }

    #[test]
    fn test_push_and_get() {
        let mut buffer: RingBuffer<f64> = RingBuffer::new(3);

        buffer.push(1.0);
        buffer.push(2.0);

        assert_eq!(buffer.len(), 2);
        assert!(!buffer.is_full());
        assert_eq!(buffer.get(0), Some(&1.0));
        assert_eq!(buffer.get(1), Some(&2.0));
        assert_eq!(buffer.get(2), None);
    }

    #[test]
    fn test_push_overflow() {
        let mut buffer: RingBuffer<f64> = RingBuffer::new(3);

        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);

        assert!(buffer.is_full());
        assert_eq!(buffer.len(), 3);

        // Push 4.0, should remove 1.0
        let replaced = buffer.push(4.0);
        assert_eq!(replaced, Some(1.0));

        assert_eq!(buffer.get(0), Some(&2.0)); // Oldest
        assert_eq!(buffer.get(1), Some(&3.0));
        assert_eq!(buffer.get(2), Some(&4.0)); // Newest
    }

    #[test]
    fn test_oldest_newest() {
        let mut buffer: RingBuffer<f64> = RingBuffer::new(3);

        assert!(buffer.oldest().is_none());
        assert!(buffer.newest().is_none());

        buffer.push(1.0);
        assert_eq!(buffer.oldest(), Some(&1.0));
        assert_eq!(buffer.newest(), Some(&1.0));

        buffer.push(2.0);
        buffer.push(3.0);
        assert_eq!(buffer.oldest(), Some(&1.0));
        assert_eq!(buffer.newest(), Some(&3.0));

        buffer.push(4.0);
        assert_eq!(buffer.oldest(), Some(&2.0));
        assert_eq!(buffer.newest(), Some(&4.0));
    }

    #[test]
    fn test_sum() {
        let mut buffer: RingBuffer<f64> = RingBuffer::new(3);

        buffer.push(1.0);
        assert_eq!(buffer.sum(), 1.0);

        buffer.push(2.0);
        assert_eq!(buffer.sum(), 3.0);

        buffer.push(3.0);
        assert_eq!(buffer.sum(), 6.0);

        buffer.push(4.0); // Replaces 1.0
        assert_eq!(buffer.sum(), 9.0); // 2 + 3 + 4
    }

    #[test]
    fn test_mean() {
        let mut buffer: RingBuffer<f64> = RingBuffer::new(3);

        assert!(buffer.mean().is_nan());

        buffer.push(1.0);
        assert_eq!(buffer.mean(), 1.0);

        buffer.push(2.0);
        assert_eq!(buffer.mean(), 1.5);

        buffer.push(3.0);
        assert_eq!(buffer.mean(), 2.0);

        buffer.push(4.0);
        assert_eq!(buffer.mean(), 3.0); // (2 + 3 + 4) / 3
    }

    #[test]
    fn test_min_max() {
        let mut buffer: RingBuffer<f64> = RingBuffer::new(4);

        buffer.push(3.0);
        buffer.push(1.0);
        buffer.push(4.0);
        buffer.push(2.0);

        assert_eq!(buffer.min(), 1.0);
        assert_eq!(buffer.max(), 4.0);

        buffer.push(5.0); // Replaces 3.0
        assert_eq!(buffer.min(), 1.0);
        assert_eq!(buffer.max(), 5.0);

        buffer.push(0.0); // Replaces 1.0
        assert_eq!(buffer.min(), 0.0);
        assert_eq!(buffer.max(), 5.0);
    }

    #[test]
    fn test_clear() {
        let mut buffer: RingBuffer<f64> = RingBuffer::new(3);

        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.sum(), 6.0);

        buffer.clear();

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.sum(), 0.0);
    }

    #[test]
    fn test_iter() {
        let mut buffer: RingBuffer<f64> = RingBuffer::new(3);

        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);

        let values: Vec<f64> = buffer.iter().copied().collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);

        buffer.push(4.0);

        let values: Vec<f64> = buffer.iter().copied().collect();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_wraparound_multiple_times() {
        let mut buffer: RingBuffer<f64> = RingBuffer::new(2);

        for i in 1..=10 {
            buffer.push(i as f64);
        }

        // Should contain 9.0 and 10.0
        assert_eq!(buffer.oldest(), Some(&9.0));
        assert_eq!(buffer.newest(), Some(&10.0));
        assert_eq!(buffer.sum(), 19.0);
    }

    #[test]
    fn test_variance_and_std() {
        let mut buffer: RingBuffer<f64> = RingBuffer::new(5);

        // Add values: 2, 4, 4, 4, 5, 5, 7, 9
        // For simplicity, just use first 5
        buffer.push(2.0);
        buffer.push(4.0);
        buffer.push(4.0);
        buffer.push(4.0);
        buffer.push(5.0);

        // Mean = (2+4+4+4+5)/5 = 19/5 = 3.8
        let mean = buffer.mean();
        assert!((mean - 3.8).abs() < 1e-10);

        // Variance (sample) = sum((x-mean)^2) / (n-1)
        // = ((2-3.8)^2 + (4-3.8)^2 + (4-3.8)^2 + (4-3.8)^2 + (5-3.8)^2) / 4
        // = (3.24 + 0.04 + 0.04 + 0.04 + 1.44) / 4
        // = 4.8 / 4 = 1.2
        let variance = buffer.variance();
        assert!((variance - 1.2).abs() < 1e-10);

        let std = buffer.std();
        assert!((std - 1.2f64.sqrt()).abs() < 1e-10);
    }
}
