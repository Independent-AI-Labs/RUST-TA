//! # ta-simd
//!
//! SIMD-optimized primitives for the rust-ta library.
//!
//! This crate provides vectorized implementations of common operations
//! using the `wide` crate for portable SIMD support.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![deny(unsafe_code)]

// Module declarations will be added as SIMD operations are implemented
// pub mod rolling;
// pub mod stats;
