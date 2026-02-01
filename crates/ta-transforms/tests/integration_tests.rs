//! Integration tests for ta-transforms.
//!
//! These tests verify transform pipelines and interactions between transforms.

use ta_core::dataframe::DataFrame;
use ta_core::series::Series;
use ta_core::traits::Transform;

use ta_transforms::prelude::*;

// ============================================================================
// Test Utilities
// ============================================================================

/// Create a test DataFrame with OHLCV-like data.
fn create_ohlcv_df() -> DataFrame<f64> {
    let mut df = DataFrame::new();

    df.add_column(
        "open".to_string(),
        Series::from_vec(vec![
            100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 104.5, 106.0,
        ]),
    )
    .unwrap();

    df.add_column(
        "high".to_string(),
        Series::from_vec(vec![
            102.0, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 106.5, 108.0,
        ]),
    )
    .unwrap();

    df.add_column(
        "low".to_string(),
        Series::from_vec(vec![
            99.0, 100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0, 103.5, 105.0,
        ]),
    )
    .unwrap();

    df.add_column(
        "close".to_string(),
        Series::from_vec(vec![
            101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 104.5, 106.0, 105.5,
        ]),
    )
    .unwrap();

    df.add_column(
        "volume".to_string(),
        Series::from_vec(vec![
            1000.0, 1100.0, 900.0, 1200.0, 1300.0, 1000.0, 1400.0, 1100.0, 1500.0, 1200.0,
        ]),
    )
    .unwrap();

    df
}

// ============================================================================
// LogReturn Tests
// ============================================================================

#[test]
fn test_log_return_basic() {
    let mut df = DataFrame::<f64>::new();
    df.add_column(
        "close".to_string(),
        Series::from_vec(vec![100.0, 110.0, 99.0, 115.0]),
    )
    .unwrap();

    let config = LogReturnConfig::new(vec!["close".to_string()]);
    let mut transform = LogReturnTransform::<f64>::new(config);

    transform.fit(&df).unwrap();
    let result = transform.transform(&df).unwrap();

    let close = result.get_column("close").unwrap();

    // First value should be NaN (no previous)
    assert!(close[0].is_nan());

    // log(110/100) ≈ 0.0953
    assert!((close[1] - 0.0953102).abs() < 1e-4);

    // log(99/110) ≈ -0.1054
    assert!((close[2] - (-0.1053605)).abs() < 1e-4);
}

#[test]
fn test_log_return_volume() {
    let mut df = DataFrame::<f64>::new();
    df.add_column(
        "volume".to_string(),
        Series::from_vec(vec![1000.0, 1500.0, 800.0, 1200.0]),
    )
    .unwrap();

    let config = LogReturnConfig::new(vec!["volume".to_string()]).with_log1p_for_volume(true);
    let mut transform = LogReturnTransform::<f64>::new(config);

    transform.fit(&df).unwrap();
    let result = transform.transform(&df).unwrap();

    let volume = result.get_column("volume").unwrap();

    // Volume uses log1p for different behavior
    // All values should be finite
    for (i, &val) in volume.iter().enumerate() {
        if i > 0 {
            // After first, should have values
            assert!(val.is_finite(), "Volume log return should be finite at {}", i);
        }
    }
}

// ============================================================================
// Normalization Tests
// ============================================================================

#[test]
fn test_normalization_standardizes() {
    let df = create_ohlcv_df();

    let config = NormalizationConfig::new(vec!["close".to_string()]);
    let mut transform = NormalizationTransform::<f64>::new(config);

    transform.fit(&df).unwrap();
    let result = transform.transform(&df).unwrap();

    let close = result.get_column("close").unwrap();

    // Mean should be approximately 0
    let mean: f64 = close.iter().sum::<f64>() / close.len() as f64;
    assert!(mean.abs() < 1e-10, "Mean should be ~0, got {}", mean);

    // Variance should be approximately 1
    let variance: f64 = close.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (close.len() - 1) as f64;
    assert!((variance - 1.0).abs() < 1e-10, "Variance should be ~1, got {}", variance);
}

#[test]
fn test_normalization_inverse_recovers_original() {
    let df = create_ohlcv_df();

    let config = NormalizationConfig::all();
    let mut transform = NormalizationTransform::<f64>::new(config);

    transform.fit(&df).unwrap();
    let normalized = transform.transform(&df).unwrap();
    let recovered = transform.inverse_transform(&normalized).unwrap();

    // Check all columns are recovered
    for col_name in df.column_names() {
        let original = df.get_column(col_name).unwrap();
        let rec = recovered.get_column(col_name).unwrap();

        for i in 0..original.len() {
            assert!(
                (original[i] - rec[i]).abs() < 1e-10,
                "Column {} index {} not recovered: {} vs {}",
                col_name,
                i,
                original[i],
                rec[i]
            );
        }
    }
}

// ============================================================================
// RobustNormalization Tests
// ============================================================================

#[test]
fn test_robust_normalization_uses_median() {
    let mut df = DataFrame::<f64>::new();
    // Data with outlier
    df.add_column(
        "price".to_string(),
        Series::from_vec(vec![10.0, 11.0, 12.0, 13.0, 14.0, 1000.0]),  // 1000 is outlier
    )
    .unwrap();

    let config = RobustNormalizationConfig::new(vec!["price".to_string()]);
    let mut transform = RobustNormalizationTransform::<f64>::new(config);

    transform.fit(&df).unwrap();

    // Median of [10, 11, 12, 13, 14, 1000] should be around 12.5
    let center = transform.center("price").unwrap();
    assert!(center < 100.0, "Center should be median-based, got {}", center);
}

#[test]
fn test_robust_normalization_inverse_recovers() {
    let df = create_ohlcv_df();

    let config = RobustNormalizationConfig::new(vec!["close".to_string(), "volume".to_string()]);
    let mut transform = RobustNormalizationTransform::<f64>::new(config);

    transform.fit(&df).unwrap();
    let normalized = transform.transform(&df).unwrap();
    let recovered = transform.inverse_transform(&normalized).unwrap();

    let original = df.get_column("close").unwrap();
    let rec = recovered.get_column("close").unwrap();

    for i in 0..original.len() {
        assert!(
            (original[i] - rec[i]).abs() < 1e-10,
            "Close index {} not recovered: {} vs {}",
            i,
            original[i],
            rec[i]
        );
    }
}

// ============================================================================
// Pipeline Tests
// ============================================================================

#[test]
fn test_pipeline_sequential_transforms() {
    let df = create_ohlcv_df();

    // Create pipeline: LogReturn -> Normalization
    let mut pipeline = TransformPipeline::<f64>::new()
        .add(LogReturnTransform::<f64>::new(LogReturnConfig::new(vec![
            "close".to_string(),
        ])))
        .add(NormalizationTransform::<f64>::new(NormalizationConfig::new(
            vec!["close".to_string()],
        )));

    // Fit and transform
    pipeline.fit(&df).unwrap();
    let result = pipeline.transform(&df).unwrap();

    let close = result.get_column("close").unwrap();

    // After log return and normalization, values should be centered
    // First value is NaN from log return
    let non_nan: Vec<f64> = close.iter().copied().filter(|x: &f64| !x.is_nan()).collect();

    if !non_nan.is_empty() {
        let mean: f64 = non_nan.iter().sum::<f64>() / non_nan.len() as f64;
        assert!(mean.abs() < 1e-6, "Mean of transformed data should be ~0, got {}", mean);
    }
}

#[test]
fn test_pipeline_fit_transform() {
    let df = create_ohlcv_df();

    let mut pipeline = TransformPipeline::<f64>::new()
        .add(NormalizationTransform::<f64>::new(NormalizationConfig::new(
            vec!["close".to_string()],
        )));

    // Use fit_transform
    let result = pipeline.fit_transform(&df).unwrap();

    assert!(pipeline.is_fitted());

    let close = result.get_column("close").unwrap();
    assert_eq!(close.len(), 10);
}

#[test]
fn test_pipeline_preserves_columns() {
    let df = create_ohlcv_df();

    // Only transform close, others should pass through
    let mut pipeline = TransformPipeline::<f64>::new()
        .add(NormalizationTransform::<f64>::new(NormalizationConfig::new(
            vec!["close".to_string()],
        )));

    pipeline.fit(&df).unwrap();
    let result = pipeline.transform(&df).unwrap();

    // All columns should exist
    assert!(result.get_column("open").is_some());
    assert!(result.get_column("high").is_some());
    assert!(result.get_column("low").is_some());
    assert!(result.get_column("close").is_some());
    assert!(result.get_column("volume").is_some());

    // open/high/low/volume should be unchanged
    let orig_open = df.get_column("open").unwrap();
    let res_open = result.get_column("open").unwrap();
    for i in 0..orig_open.len() {
        assert_eq!(orig_open[i], res_open[i]);
    }
}

#[test]
fn test_empty_pipeline() {
    let df = create_ohlcv_df();

    let mut pipeline = TransformPipeline::<f64>::new();

    pipeline.fit(&df).unwrap();
    let result = pipeline.transform(&df).unwrap();

    // Empty pipeline should return data unchanged
    let orig_close = df.get_column("close").unwrap();
    let res_close = result.get_column("close").unwrap();

    for i in 0..orig_close.len() {
        assert_eq!(orig_close[i], res_close[i]);
    }
}

// ============================================================================
// State Persistence Tests
// ============================================================================

#[test]
fn test_normalization_state_persistence() {
    let df = create_ohlcv_df();

    let config = NormalizationConfig::all();
    let mut transform1 = NormalizationTransform::<f64>::new(config.clone());

    transform1.fit(&df).unwrap();
    let result1 = transform1.transform(&df).unwrap();

    // Get state
    let state = transform1.get_state();

    // Create new transform and restore state
    let mut transform2 = NormalizationTransform::<f64>::new(config);
    transform2.set_state(state).unwrap();

    // Should produce same results
    let result2 = transform2.transform(&df).unwrap();

    let close1 = result1.get_column("close").unwrap();
    let close2 = result2.get_column("close").unwrap();

    for i in 0..close1.len() {
        assert!(
            (close1[i] - close2[i]).abs() < 1e-10,
            "Results differ at index {}: {} vs {}",
            i,
            close1[i],
            close2[i]
        );
    }
}

#[test]
fn test_robust_normalization_state_persistence() {
    let df = create_ohlcv_df();

    let config = RobustNormalizationConfig::new(vec!["close".to_string()]);
    let mut transform1 = RobustNormalizationTransform::<f64>::new(config.clone());

    transform1.fit(&df).unwrap();
    let result1 = transform1.transform(&df).unwrap();

    // Get state
    let state = transform1.get_state();

    // Create new transform and restore state
    let mut transform2 = RobustNormalizationTransform::<f64>::new(config);
    transform2.set_state(state).unwrap();

    // Should produce same results
    let result2 = transform2.transform(&df).unwrap();

    let close1 = result1.get_column("close").unwrap();
    let close2 = result2.get_column("close").unwrap();

    for i in 0..close1.len() {
        assert!(
            (close1[i] - close2[i]).abs() < 1e-10,
            "Results differ at index {}: {} vs {}",
            i,
            close1[i],
            close2[i]
        );
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_transform_empty_dataframe() {
    let df = DataFrame::<f64>::new();

    let config = NormalizationConfig::all();
    let mut transform = NormalizationTransform::<f64>::new(config);

    // Should succeed with empty DF
    transform.fit(&df).unwrap();
}

#[test]
fn test_transform_single_row() {
    let mut df = DataFrame::<f64>::new();
    df.add_column("price".to_string(), Series::from_vec(vec![100.0]))
        .unwrap();

    let config = NormalizationConfig::all();
    let mut transform = NormalizationTransform::<f64>::new(config);

    transform.fit(&df).unwrap();
    let result = transform.transform(&df).unwrap();

    // Single value normalized should be 0 (or NaN if variance is 0)
    let price = result.get_column("price").unwrap();
    // With variance=0, scale=1, so (100-100)/1 = 0
    assert_eq!(price[0], 0.0);
}

#[test]
fn test_transform_with_nan_values() {
    let mut df = DataFrame::<f64>::new();
    df.add_column(
        "price".to_string(),
        Series::from_vec(vec![100.0, f64::NAN, 102.0, 103.0, 104.0]),
    )
    .unwrap();

    let config = NormalizationConfig::all();
    let mut transform = NormalizationTransform::<f64>::new(config);

    transform.fit(&df).unwrap();
    let result = transform.transform(&df).unwrap();

    let price = result.get_column("price").unwrap();

    // NaN should remain NaN
    assert!(price[1].is_nan());

    // Other values should be normalized
    assert!(!price[0].is_nan());
    assert!(!price[2].is_nan());
}
