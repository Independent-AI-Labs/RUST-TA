//! # ta-indicators
//!
//! Technical indicators for the rust-ta library.
//!
//! This crate provides implementations of technical analysis indicators organized
//! into four categories:
//!
//! - **Momentum**: RSI, StochRSI, Stochastic, Williams %R, etc.
//! - **Trend**: SMA, EMA, MACD, ADX, Aroon, Ichimoku, etc.
//! - **Volatility**: ATR, Bollinger Bands, Keltner Channel, etc.
//! - **Volume**: OBV, VWAP, MFI, etc.
//!
//! # Example
//!
//! ```
//! use ta_indicators::prelude::*;
//! use ta_core::prelude::*;
//!
//! // Create an RSI indicator
//! let config = RsiConfig::default();
//! let mut rsi = Rsi::<f64>::new(config);
//!
//! // Process bars in streaming mode
//! let bar = Bar::new(100.0, 102.0, 99.0, 101.0, 1000.0);
//! if let Ok(value) = rsi.update(&bar) {
//!     println!("RSI: {:?}", value);
//! }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![deny(unsafe_code)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod momentum;
pub mod trend;
pub mod volatility;
pub mod volume;

pub mod prelude;

pub use prelude::*;
