#!/usr/bin/env python3
"""
Generate golden data for rust-ta tests from PostgreSQL OHLCV data.

This script:
1. Connects to PostgreSQL to fetch OHLCV data
2. Calculates indicators using python-ta (the reference implementation)
3. Saves results as JSON files for use in Rust tests

Usage:
    python generate_golden.py [--db-url postgresql://...] [--output-dir ../golden/python_outputs]
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import argparse

# Try to import required libraries
try:
    import psycopg2
except ImportError:
    print("Warning: psycopg2 not installed. Install with: pip install psycopg2-binary")
    psycopg2 = None

try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed. Install with: pip install pandas")
    pd = None

try:
    import ta
except ImportError:
    print("Warning: python-ta not installed. Install with: pip install ta")
    ta = None


def fetch_ohlcv_from_db(db_url: str, symbol: str = "BTCUSDT", limit: int = 1000) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from PostgreSQL."""
    if psycopg2 is None or pd is None:
        return None

    try:
        conn = psycopg2.connect(db_url)
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        conn.close()

        # Reverse to chronological order
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error fetching from database: {e}")
        return None


def generate_synthetic_ohlcv(length: int = 100) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    import numpy as np

    np.random.seed(42)  # Reproducible

    # Start with a base price and generate random walk
    base_price = 100.0
    returns = np.random.normal(0, 0.02, length)  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    data = {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
    }

    prev_close = base_price
    for i in range(length):
        close = prices[i]
        open_price = prev_close

        # High and low within realistic range
        daily_range = abs(close - open_price) + close * 0.01
        high = max(open_price, close) + daily_range * np.random.uniform(0.1, 0.5)
        low = min(open_price, close) - daily_range * np.random.uniform(0.1, 0.5)

        # Ensure high >= close >= low (OHLCV invariant)
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = 1000 + np.random.uniform(0, 5000)

        data["open"].append(round(open_price, 4))
        data["high"].append(round(high, 4))
        data["low"].append(round(low, 4))
        data["close"].append(round(close, 4))
        data["volume"].append(round(volume, 2))

        prev_close = close

    return pd.DataFrame(data)


def calculate_indicators(df: pd.DataFrame) -> Dict[str, List[Optional[float]]]:
    """Calculate indicators using python-ta."""
    if ta is None:
        return {}

    results = {}

    # === Trend Indicators ===

    # SMA
    for window in [5, 10, 14, 20]:
        indicator = ta.trend.SMAIndicator(close=df["close"], window=window)
        results[f"sma_{window}"] = series_to_optional_list(indicator.sma_indicator())

    # EMA
    for window in [5, 10, 12, 26]:
        indicator = ta.trend.EMAIndicator(close=df["close"], window=window)
        results[f"ema_{window}"] = series_to_optional_list(indicator.ema_indicator())

    # WMA
    for window in [5, 10, 14]:
        indicator = ta.trend.WMAIndicator(close=df["close"], window=window)
        results[f"wma_{window}"] = series_to_optional_list(indicator.wma())

    # MACD
    macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    results["macd_line"] = series_to_optional_list(macd.macd())
    results["macd_signal"] = series_to_optional_list(macd.macd_signal())
    results["macd_histogram"] = series_to_optional_list(macd.macd_diff())

    # ADX
    for window in [14]:
        adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=window)
        results[f"adx_{window}"] = series_to_optional_list(adx.adx())
        results[f"plus_di_{window}"] = series_to_optional_list(adx.adx_pos())
        results[f"minus_di_{window}"] = series_to_optional_list(adx.adx_neg())

    # Aroon
    for window in [25]:
        aroon = ta.trend.AroonIndicator(high=df["high"], low=df["low"], window=window)
        results[f"aroon_up_{window}"] = series_to_optional_list(aroon.aroon_up())
        results[f"aroon_down_{window}"] = series_to_optional_list(aroon.aroon_down())
        results[f"aroon_oscillator_{window}"] = series_to_optional_list(aroon.aroon_indicator())

    # === Momentum Indicators ===

    # RSI
    for window in [14]:
        rsi = ta.momentum.RSIIndicator(close=df["close"], window=window)
        results[f"rsi_{window}"] = series_to_optional_list(rsi.rsi())

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"],
        window=14, smooth_window=3
    )
    results["stoch_k"] = series_to_optional_list(stoch.stoch())
    results["stoch_d"] = series_to_optional_list(stoch.stoch_signal())

    # Williams %R
    for window in [14]:
        wr = ta.momentum.WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=window)
        results[f"williams_r_{window}"] = series_to_optional_list(wr.williams_r())

    # ROC
    for window in [12]:
        roc = ta.momentum.ROCIndicator(close=df["close"], window=window)
        results[f"roc_{window}"] = series_to_optional_list(roc.roc())

    # StochRSI
    stochrsi = ta.momentum.StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3)
    results["stochrsi"] = series_to_optional_list(stochrsi.stochrsi())
    results["stochrsi_k"] = series_to_optional_list(stochrsi.stochrsi_k())
    results["stochrsi_d"] = series_to_optional_list(stochrsi.stochrsi_d())

    # === Volatility Indicators ===

    # ATR
    for window in [14]:
        atr = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=window)
        results[f"atr_{window}"] = series_to_optional_list(atr.average_true_range())

    # Bollinger Bands
    for window in [20]:
        bb = ta.volatility.BollingerBands(close=df["close"], window=window, window_dev=2)
        results[f"bollinger_upper_{window}"] = series_to_optional_list(bb.bollinger_hband())
        results[f"bollinger_middle_{window}"] = series_to_optional_list(bb.bollinger_mavg())
        results[f"bollinger_lower_{window}"] = series_to_optional_list(bb.bollinger_lband())
        results[f"bollinger_width_{window}"] = series_to_optional_list(bb.bollinger_wband())
        results[f"bollinger_pct_b_{window}"] = series_to_optional_list(bb.bollinger_pband())

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(high=df["high"], low=df["low"], close=df["close"], window=20, window_atr=10)
    results["keltner_upper"] = series_to_optional_list(kc.keltner_channel_hband())
    results["keltner_middle"] = series_to_optional_list(kc.keltner_channel_mband())
    results["keltner_lower"] = series_to_optional_list(kc.keltner_channel_lband())

    # Donchian Channel
    dc = ta.volatility.DonchianChannel(high=df["high"], low=df["low"], close=df["close"], window=20)
    results["donchian_upper"] = series_to_optional_list(dc.donchian_channel_hband())
    results["donchian_middle"] = series_to_optional_list(dc.donchian_channel_mband())
    results["donchian_lower"] = series_to_optional_list(dc.donchian_channel_lband())

    # === Volume Indicators ===

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
    results["obv"] = series_to_optional_list(obv.on_balance_volume())

    # MFI
    for window in [14]:
        mfi = ta.volume.MFIIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=window)
        results[f"mfi_{window}"] = series_to_optional_list(mfi.money_flow_index())

    # CMF
    for window in [20]:
        cmf = ta.volume.ChaikinMoneyFlowIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=window)
        results[f"cmf_{window}"] = series_to_optional_list(cmf.chaikin_money_flow())

    # VWAP (note: python-ta VWAP requires specific handling)
    try:
        vwap = ta.volume.VolumeWeightedAveragePrice(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"])
        results["vwap"] = series_to_optional_list(vwap.volume_weighted_average_price())
    except:
        pass  # VWAP may fail depending on data

    return results


def series_to_optional_list(series: pd.Series) -> List[Optional[float]]:
    """Convert pandas Series to list with None for NaN values."""
    result = []
    for val in series:
        if pd.isna(val):
            result.append(None)
        else:
            result.append(round(float(val), 8))
    return result


def save_golden_data(
    df: pd.DataFrame,
    indicators: Dict[str, List[Optional[float]]],
    output_path: str,
    description: str = "Golden data for rust-ta tests"
) -> None:
    """Save golden data as JSON."""
    data = {
        "description": description,
        "source": f"python-ta v{ta.__version__}" if ta else "unknown",
        "generated_at": datetime.now().isoformat(),
        "ohlcv": {
            "open": df["open"].tolist(),
            "high": df["high"].tolist(),
            "low": df["low"].tolist(),
            "close": df["close"].tolist(),
            "volume": df["volume"].tolist(),
        },
        "expected": indicators,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved golden data to {output_path}")
    print(f"  OHLCV length: {len(df)}")
    print(f"  Indicators: {len(indicators)}")


def main():
    parser = argparse.ArgumentParser(description="Generate golden data for rust-ta tests")
    parser.add_argument("--db-url", help="PostgreSQL connection URL")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol to fetch")
    parser.add_argument("--limit", type=int, default=500, help="Number of OHLCV records")
    parser.add_argument("--output-dir", default="../golden/python_outputs", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of DB")
    args = parser.parse_args()

    # Get OHLCV data
    if args.synthetic or args.db_url is None:
        print("Generating synthetic OHLCV data...")
        df = generate_synthetic_ohlcv(args.limit)
        filename = "synthetic_ohlcv.json"
        description = "Synthetic OHLCV data for rust-ta tests"
    else:
        print(f"Fetching OHLCV data from database for {args.symbol}...")
        df = fetch_ohlcv_from_db(args.db_url, args.symbol, args.limit)
        if df is None:
            print("Failed to fetch from database, falling back to synthetic data")
            df = generate_synthetic_ohlcv(args.limit)
            filename = "synthetic_ohlcv.json"
            description = "Synthetic OHLCV data for rust-ta tests"
        else:
            filename = f"{args.symbol.lower()}_ohlcv.json"
            description = f"Real {args.symbol} OHLCV data for rust-ta tests"

    print(f"OHLCV data shape: {df.shape}")

    # Calculate indicators
    if ta is None:
        print("Error: python-ta is not installed. Cannot calculate indicators.")
        sys.exit(1)

    print("Calculating indicators...")
    indicators = calculate_indicators(df)
    print(f"Calculated {len(indicators)} indicator series")

    # Save golden data
    output_path = os.path.join(args.output_dir, filename)
    save_golden_data(df, indicators, output_path, description)

    # Also save a smaller version for quick tests
    small_df = df.head(50)
    small_indicators = {k: v[:50] for k, v in indicators.items()}
    small_output_path = os.path.join(args.output_dir, "quick_test_ohlcv.json")
    save_golden_data(small_df, small_indicators, small_output_path, "Small dataset for quick tests")


if __name__ == "__main__":
    main()
