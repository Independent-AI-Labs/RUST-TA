//! # ta-python
//!
//! Python bindings for the rust-ta technical analysis library.
//!
//! This crate provides PyO3-based Python bindings that expose
//! rust-ta functionality to Python, enabling drop-in replacement
//! of python-ta.

#![warn(missing_docs)]
#![deny(unsafe_code)]

use pyo3::prelude::*;

/// Python module for rust-ta.
#[pymodule]
fn rust_ta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
