//! # ta-transforms
//!
//! Data transformation pipeline for the rust-ta library.
//!
//! This crate provides data transformation classes matching the AMI-TRADING
//! transform system, including:
//!
//! - `LogReturnTransform`: Price-to-log-return conversion
//! - `NormalizationTransform`: StandardScaler equivalent
//! - `RobustNormalizationTransform`: Quantile-based scaling
//! - `TechnicalIndicatorsTransform`: Add technical indicator columns
//! - `TransformPipeline`: Compose multiple transforms
//!
//! # Example
//!
//! ```ignore
//! use ta_transforms::prelude::*;
//! use ta_core::prelude::*;
//!
//! // Create a normalization transform
//! let mut normalizer = NormalizationTransform::<f64>::new(NormalizationConfig::default());
//!
//! // Fit and transform data
//! let transformed = normalizer.fit_transform(&df)?;
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![deny(unsafe_code)]

#[cfg(not(feature = "std"))]
extern crate alloc;

mod log_return;
mod normalization;
mod robust_normalization;
mod pipeline;

pub mod prelude;

pub use log_return::{LogReturnTransform, LogReturnConfig, LogReturnState};
pub use normalization::{NormalizationTransform, NormalizationConfig, NormalizationState};
pub use robust_normalization::{RobustNormalizationTransform, RobustNormalizationConfig, RobustNormalizationState};
pub use pipeline::{TransformPipeline, PipelineConfig};
