//! Prelude module for convenient imports.
//!
//! This module re-exports the most commonly used types and traits from ta-core.
//!
//! # Example
//!
//! ```rust
//! use ta_core::prelude::*;
//!
//! // Now you have access to all common types
//! let series: Series<f64> = Series::new();
//! let bar = Bar::new(100.0, 105.0, 98.0, 103.0, 1_000_000.0);
//! ```

// Core types
pub use crate::dataframe::DataFrame;
pub use crate::num::TaFloat;
pub use crate::ohlcv::{Bar, OhlcvSeries};
pub use crate::series::Series;
pub use crate::window::RingBuffer;

// Error types
pub use crate::error::{IndicatorError, Result, StateRestoreError, TransformError};

// Traits
pub use crate::traits::{Indicator, NanMode, StreamingIndicator, Transform};

// Utility functions
pub use crate::utils::{
    crossover, crossunder, diff, ema, pct_change, rolling_max, rolling_mean, rolling_min,
    rolling_std, rolling_sum, rolling_variance, shift, sma, true_range, wilder_smooth,
};
