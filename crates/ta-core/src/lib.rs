//! # ta-core
//!
//! Core types and traits for the rust-ta technical analysis library.
//!
//! This crate provides the foundational abstractions used throughout the library:
//!
//! - [`TaFloat`] - Trait for numeric types (f32/f64)
//! - [`Series`] - Time series data container
//! - [`Bar`] and [`OhlcvSeries`] - OHLCV price data types
//! - [`DataFrame`] - Multi-column tabular data with deterministic ordering
//! - [`RingBuffer`] - Circular buffer for streaming calculations
//! - [`Indicator`] and [`StreamingIndicator`] - Indicator computation traits
//! - [`Transform`] - Data transformation trait
//!
//! ## Feature Flags
//!
//! - `std` (default) - Enable standard library support
//! - `alloc` - Enable heap allocation without full std
//! - `serde` - Enable serialization/deserialization support
//!
//! ## Example
//!
//! ```rust
//! use ta_core::prelude::*;
//!
//! // Create a series of close prices
//! let closes: Series<f64> = Series::from_vec(vec![100.0, 101.5, 99.8, 102.3, 101.0]);
//!
//! // Compute simple moving average
//! let sma = rolling_mean(closes.as_slice(), 3);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod dataframe;
pub mod error;
pub mod num;
pub mod ohlcv;
pub mod prelude;
pub mod series;
pub mod traits;
pub mod utils;
pub mod window;

// Re-export core types at crate root
pub use dataframe::DataFrame;
pub use error::{IndicatorError, Result, StateRestoreError, TransformError};
pub use num::TaFloat;
pub use ohlcv::{Bar, OhlcvSeries};
pub use series::Series;
pub use traits::{Indicator, NanMode, StreamingIndicator, Transform};
pub use window::RingBuffer;
