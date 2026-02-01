//! Trend indicators.
//!
//! This module contains trend-following indicators:
//! - SMA (Simple Moving Average)
//! - EMA (Exponential Moving Average)
//! - WMA (Weighted Moving Average)
//! - MACD (Moving Average Convergence Divergence)
//! - ADX (Average Directional Index)
//! - Aroon

mod sma;
mod ema;
mod wma;
mod macd;
mod adx;
mod aroon;

pub use sma::{Sma, SmaConfig, SmaState};
pub use ema::{Ema, EmaConfig, EmaState};
pub use wma::{Wma, WmaConfig, WmaState};
pub use macd::{Macd, MacdConfig, MacdOutput, MacdState};
pub use adx::{Adx, AdxConfig, AdxOutput, AdxState};
pub use aroon::{Aroon, AroonConfig, AroonOutput, AroonState};
