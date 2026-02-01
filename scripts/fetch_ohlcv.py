#!/usr/bin/env python3
"""
Fetch OHLCV data from multiple sources and generate test data.

Sources:
    - Binance API (public, no auth needed for historical klines)
    - TimescaleDB (local database)
    - Synthetic generation (fallback)

Usage:
    python scripts/fetch_ohlcv.py --source binance --symbol BTCUSDT --limit 100000
    python scripts/fetch_ohlcv.py --source db --symbol BTCUSDT
    python scripts/fetch_ohlcv.py --synthetic --limit 500000
    python scripts/fetch_ohlcv.py --list-symbols
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Load .env file
def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env()

# Database config
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "database": os.environ.get("DB_DATABASE", "forecasting_db"),
    "user": os.environ.get("DB_USER", "forecasting_user"),
    "password": os.environ.get("DB_PASSWORD", "forecasting_db_password"),
}

# Binance config
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_KLINES_LIMIT = 1000  # Max per request


# =============================================================================
# Binance API
# =============================================================================

def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 1000,
                         end_time: int = None) -> list:
    """Fetch klines from Binance API."""
    try:
        import requests
    except ImportError:
        print("Error: requests not installed. Run: pip install requests")
        sys.exit(1)

    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, BINANCE_KLINES_LIMIT),
    }
    if end_time:
        params["endTime"] = end_time

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Binance API error: {response.status_code} - {response.text}")
        return []

    return response.json()


def fetch_binance_ohlcv(symbol: str, interval: str = "1h", limit: int = 10000) -> dict:
    """Fetch large OHLCV dataset from Binance, handling pagination."""
    print(f"Fetching {limit} {interval} candles for {symbol} from Binance...")

    all_klines = []
    end_time = None
    remaining = limit

    while remaining > 0:
        batch_size = min(remaining, BINANCE_KLINES_LIMIT)
        klines = fetch_binance_klines(symbol, interval, batch_size, end_time)

        if not klines:
            break

        all_klines = klines + all_klines  # Prepend (older data first)
        remaining -= len(klines)

        if len(klines) < BINANCE_KLINES_LIMIT:
            break  # No more data

        # Set end_time to 1ms before the oldest candle for next batch
        end_time = klines[0][0] - 1

        print(f"  Fetched {len(all_klines)}/{limit} candles...")
        time.sleep(0.1)  # Rate limiting

    if not all_klines:
        print(f"No data returned from Binance for {symbol}")
        return None

    # Convert to our format
    # Kline format: [open_time, open, high, low, close, volume, close_time, ...]
    data = {
        "timestamp": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
    }

    for k in all_klines:
        data["timestamp"].append(datetime.fromtimestamp(k[0] / 1000).isoformat())
        data["open"].append(float(k[1]))
        data["high"].append(float(k[2]))
        data["low"].append(float(k[3]))
        data["close"].append(float(k[4]))
        data["volume"].append(float(k[5]))

    print(f"Fetched {len(data['close'])} candles from Binance")
    return data


def list_binance_symbols() -> list:
    """List available symbols on Binance."""
    try:
        import requests
    except ImportError:
        return []

    url = f"{BINANCE_BASE_URL}/api/v3/exchangeInfo"
    response = requests.get(url)
    if response.status_code != 200:
        return []

    info = response.json()
    symbols = [s["symbol"] for s in info.get("symbols", [])
               if s.get("status") == "TRADING" and s["symbol"].endswith("USDT")]
    return sorted(symbols)[:50]  # Return top 50 USDT pairs


# =============================================================================
# Database
# =============================================================================

def get_db_connection():
    """Get database connection."""
    try:
        import psycopg2
    except ImportError:
        print("Warning: psycopg2 not installed. Database source unavailable.")
        return None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Warning: Could not connect to database: {e}")
        return None


def fetch_db_ohlcv(conn, symbol: str, limit: int = 500, table: str = "market_data") -> dict:
    """Fetch OHLCV data from database."""
    cur = conn.cursor()

    cur.execute("""
        SELECT column_name FROM information_schema.columns WHERE table_name = %s
    """, (table,))
    columns = [row[0] for row in cur.fetchall()]

    time_col = next((c for c in columns if c in ['timestamp', 'time', 'datetime', 'ts']), None)
    if not time_col:
        print(f"No timestamp column found in {table}")
        return None

    query = f"""
        SELECT {time_col}, open, high, low, close, volume
        FROM {table}
        WHERE symbol = %s
        ORDER BY {time_col} DESC
        LIMIT %s
    """

    try:
        cur.execute(query, (symbol, limit))
        rows = cur.fetchall()[::-1]  # Reverse to chronological
        cur.close()

        if not rows:
            return None

        data = {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
        for ts, o, h, l, c, v in rows:
            data["timestamp"].append(ts.isoformat() if hasattr(ts, 'isoformat') else str(ts))
            data["open"].append(float(o))
            data["high"].append(float(h))
            data["low"].append(float(l))
            data["close"].append(float(c))
            data["volume"].append(float(v))

        return data
    except Exception as e:
        print(f"Database error: {e}")
        conn.rollback()
        return None


# =============================================================================
# Synthetic Data
# =============================================================================

def generate_synthetic_ohlcv(length: int = 500, base_price: float = 50000.0) -> dict:
    """Generate realistic synthetic OHLCV data."""
    import numpy as np

    np.random.seed(42)

    # Generate random walk with realistic crypto volatility
    returns = np.random.normal(0.0001, 0.02, length)  # ~2% hourly volatility
    prices = base_price * np.exp(np.cumsum(returns))

    data = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    prev_close = base_price

    for i in range(length):
        close = prices[i]
        open_price = prev_close * (1 + np.random.normal(0, 0.001))

        # Realistic range
        range_pct = np.random.uniform(0.005, 0.03)
        high = max(open_price, close) * (1 + range_pct * np.random.uniform(0.3, 1.0))
        low = min(open_price, close) * (1 - range_pct * np.random.uniform(0.3, 1.0))

        # Ensure OHLC invariants
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume correlated with volatility
        base_vol = 1000 + np.random.exponential(5000)
        vol_mult = 1 + abs(close - open_price) / open_price * 50
        volume = base_vol * vol_mult

        data["open"].append(round(open_price, 2))
        data["high"].append(round(high, 2))
        data["low"].append(round(low, 2))
        data["close"].append(round(close, 2))
        data["volume"].append(round(volume, 4))

        prev_close = close

    return data


# =============================================================================
# Indicator Calculation
# =============================================================================

def calculate_indicators(ohlcv: dict) -> dict:
    """Calculate indicators using python-ta."""
    try:
        import pandas as pd
        import ta
    except ImportError as e:
        print(f"Warning: Could not import {e.name}. Skipping indicator calculation.")
        return {}

    df = pd.DataFrame({
        "open": ohlcv["open"],
        "high": ohlcv["high"],
        "low": ohlcv["low"],
        "close": ohlcv["close"],
        "volume": ohlcv["volume"],
    })

    results = {}

    def to_list(series):
        return [None if pd.isna(v) else round(float(v), 10) for v in series]

    # Trend
    for w in [5, 10, 14, 20, 50, 100, 200]:
        if w < len(df):
            results[f"sma_{w}"] = to_list(ta.trend.SMAIndicator(df["close"], w).sma_indicator())
            results[f"ema_{w}"] = to_list(ta.trend.EMAIndicator(df["close"], w).ema_indicator())

    # RSI
    for w in [7, 14, 21]:
        if w < len(df):
            results[f"rsi_{w}"] = to_list(ta.momentum.RSIIndicator(df["close"], w).rsi())

    # MACD
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    results["macd_line"] = to_list(macd.macd())
    results["macd_signal"] = to_list(macd.macd_signal())
    results["macd_histogram"] = to_list(macd.macd_diff())

    # Bollinger
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    results["bollinger_upper"] = to_list(bb.bollinger_hband())
    results["bollinger_middle"] = to_list(bb.bollinger_mavg())
    results["bollinger_lower"] = to_list(bb.bollinger_lband())

    # ATR
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    results["atr_14"] = to_list(atr.average_true_range())

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    results["stoch_k"] = to_list(stoch.stoch())
    results["stoch_d"] = to_list(stoch.stoch_signal())

    # ADX
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    results["adx"] = to_list(adx.adx())
    results["plus_di"] = to_list(adx.adx_pos())
    results["minus_di"] = to_list(adx.adx_neg())

    # Volume
    results["obv"] = to_list(ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume())

    mfi = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=14)
    results["mfi_14"] = to_list(mfi.money_flow_index())

    print(f"Calculated {len(results)} indicator series")
    return results


# =============================================================================
# Output
# =============================================================================

def save_data(ohlcv: dict, indicators: dict, output_path: str, symbol: str, source: str):
    """Save data to JSON file."""
    try:
        import ta
        ta_version = ta.__version__
    except:
        ta_version = "unknown"

    data = {
        "description": f"{symbol} OHLCV data for rust-ta benchmarks",
        "source": source,
        "python_ta_version": ta_version,
        "generated_at": datetime.now().isoformat(),
        "symbol": symbol,
        "num_candles": len(ohlcv["close"]),
        "ohlcv": ohlcv,
        "expected": indicators,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f)  # No indent for large files

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to {output_path} ({size_mb:.2f} MB)")
    print(f"  Candles: {len(ohlcv['close']):,}")
    print(f"  Indicators: {len(indicators)}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data for rust-ta benchmarks")
    parser.add_argument("--source", choices=["binance", "db", "synthetic"], default="binance",
                       help="Data source (default: binance)")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--interval", default="1h", help="Candle interval for Binance (default: 1h)")
    parser.add_argument("--limit", type=int, default=100000, help="Number of candles (default: 100000)")
    parser.add_argument("--table", default="market_data", help="Database table name")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--output-dir", default="data", help="Output directory (default: data)")
    parser.add_argument("--list-symbols", action="store_true", help="List available symbols")
    parser.add_argument("--no-indicators", action="store_true", help="Skip indicator calculation")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    args = parser.parse_args()

    # List symbols
    if args.list_symbols:
        print("Binance USDT pairs (top 50):")
        for s in list_binance_symbols():
            print(f"  {s}")
        return

    # Fetch data
    ohlcv = None
    source_name = args.source

    if args.synthetic:
        print(f"Generating {args.limit:,} synthetic candles...")
        ohlcv = generate_synthetic_ohlcv(args.limit)
        source_name = "synthetic"
        symbol = "SYNTHETIC"
    elif args.source == "binance":
        ohlcv = fetch_binance_ohlcv(args.symbol, args.interval, args.limit)
        source_name = f"binance_{args.interval}"
        symbol = args.symbol
    elif args.source == "db":
        conn = get_db_connection()
        if conn:
            ohlcv = fetch_db_ohlcv(conn, args.symbol, args.limit, args.table)
            conn.close()
            source_name = "timescaledb"
            symbol = args.symbol

    # Fallback to synthetic
    if ohlcv is None or len(ohlcv.get("close", [])) == 0:
        print("Primary source failed. Generating synthetic data...")
        ohlcv = generate_synthetic_ohlcv(args.limit)
        source_name = "synthetic"
        symbol = "SYNTHETIC"
    else:
        symbol = args.symbol

    print(f"Got {len(ohlcv['close']):,} candles")

    # Calculate indicators
    if args.no_indicators:
        indicators = {}
    else:
        print("Calculating indicators (this may take a while for large datasets)...")
        start = time.time()
        indicators = calculate_indicators(ohlcv)
        elapsed = time.time() - start
        print(f"Indicator calculation took {elapsed:.2f}s")

    # Save
    script_dir = Path(__file__).parent.parent
    if args.output:
        output_path = args.output
    else:
        output_path = script_dir / args.output_dir / f"{symbol.lower()}_{len(ohlcv['close'])}_{source_name}.json"

    save_data(ohlcv, indicators, str(output_path), symbol, source_name)


if __name__ == "__main__":
    main()
