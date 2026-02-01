//! Error types for technical analysis operations.
//!
//! This module provides structured error types for indicator and transform operations,
//! with full error chaining support via `thiserror`.

use thiserror::Error;

/// Result type alias for indicator operations that may fail.
pub type Result<T> = core::result::Result<T, IndicatorError>;

/// Result type alias for transform operations that may fail.
pub type TransformResult<T> = core::result::Result<T, TransformError>;

/// Errors that can occur during indicator computation.
#[derive(Debug, Error)]
pub enum IndicatorError {
    /// Not enough data points to compute the indicator.
    #[error("Insufficient data: need {required} points, got {actual}")]
    InsufficientData {
        /// Required number of data points.
        required: usize,
        /// Actual number of data points provided.
        actual: usize,
    },

    /// Invalid window size parameter.
    #[error("Invalid window size: {0} (must be > 0)")]
    InvalidWindow(usize),

    /// Invalid parameter value.
    #[error("Invalid parameter '{name}': {value} (expected {expected})")]
    InvalidParameter {
        /// Name of the parameter.
        name: &'static str,
        /// Provided value as string.
        value: String,
        /// Description of expected value.
        expected: &'static str,
    },

    /// Numeric computation error (overflow, underflow, division by zero).
    #[error("Numeric error: {0}")]
    NumericError(String),

    /// State restoration failed.
    #[error("State restoration failed")]
    StateError(#[from] StateRestoreError),

    /// Series length mismatch in computation.
    #[error("Series length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// Input contains invalid values (NaN when not allowed).
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Errors that can occur when restoring indicator state.
#[derive(Debug, Error)]
pub enum StateRestoreError {
    /// State version mismatch.
    #[error("State version mismatch: expected {expected}, got {actual}")]
    VersionMismatch {
        /// Expected version.
        expected: String,
        /// Actual version found.
        actual: String,
    },

    /// State data is corrupted or invalid.
    #[error("Invalid state data: {0}")]
    InvalidData(String),

    /// Deserialization failed.
    #[error("Deserialization failed: {0}")]
    DeserializationError(String),
}

/// Errors that can occur during data transformation.
#[derive(Debug, Error)]
pub enum TransformError {
    /// Transform was used before being fitted.
    #[error("Transform not fitted: call fit() before transform()")]
    NotFitted,

    /// Column count mismatch between expected and actual.
    #[error("Shape mismatch: expected {expected} columns, got {actual}")]
    ShapeMismatch {
        /// Expected number of columns.
        expected: usize,
        /// Actual number of columns.
        actual: usize,
    },

    /// Required column not found in DataFrame.
    #[error("Missing required column: '{0}'")]
    MissingColumn(String),

    /// Underlying indicator computation failed.
    #[error("Indicator computation failed")]
    Indicator(#[from] IndicatorError),

    /// Serialization or deserialization failed.
    #[error("Serialization failed: {context}")]
    Serialization {
        /// Context describing the serialization operation.
        context: String,
        /// Underlying serialization error message.
        source_message: String,
    },

    /// Inverse transform not supported.
    #[error("Inverse transform not supported for {0}")]
    InverseNotSupported(String),

    /// Pipeline configuration error.
    #[error("Pipeline error: {0}")]
    PipelineError(String),
}

impl TransformError {
    /// Create a serialization error with context.
    #[must_use]
    pub fn serialization(context: impl Into<String>, source: impl core::fmt::Display) -> Self {
        Self::Serialization {
            context: context.into(),
            source_message: source.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indicator_error_display() {
        let err = IndicatorError::InsufficientData {
            required: 14,
            actual: 10,
        };
        assert_eq!(
            err.to_string(),
            "Insufficient data: need 14 points, got 10"
        );

        let err = IndicatorError::InvalidWindow(0);
        assert_eq!(err.to_string(), "Invalid window size: 0 (must be > 0)");

        let err = IndicatorError::InvalidParameter {
            name: "period",
            value: "-5".to_string(),
            expected: "positive integer",
        };
        assert_eq!(
            err.to_string(),
            "Invalid parameter 'period': -5 (expected positive integer)"
        );
    }

    #[test]
    fn test_transform_error_display() {
        let err = TransformError::NotFitted;
        assert_eq!(
            err.to_string(),
            "Transform not fitted: call fit() before transform()"
        );

        let err = TransformError::MissingColumn("close".to_string());
        assert_eq!(err.to_string(), "Missing required column: 'close'");
    }

    #[test]
    fn test_error_chaining() {
        let state_err = StateRestoreError::VersionMismatch {
            expected: "2".to_string(),
            actual: "1".to_string(),
        };
        let indicator_err: IndicatorError = state_err.into();
        assert!(matches!(indicator_err, IndicatorError::StateError(_)));

        let transform_err: TransformError = IndicatorError::InvalidWindow(0).into();
        assert!(matches!(transform_err, TransformError::Indicator(_)));
    }

    #[test]
    fn test_serialization_error_helper() {
        let err = TransformError::serialization("saving RSI state", "invalid JSON");
        match err {
            TransformError::Serialization {
                context,
                source_message,
            } => {
                assert_eq!(context, "saving RSI state");
                assert_eq!(source_message, "invalid JSON");
            }
            _ => panic!("Expected Serialization error"),
        }
    }
}
