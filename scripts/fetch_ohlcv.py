#!/usr/bin/env python3
"""
Fetch OHLCV data from TimescaleDB and generate golden test data.

Usage:
    python scripts/fetch_ohlcv.py                    # Use defaults
    python scripts/fetch_ohlcv.py --symbol ETHUSDT  # Specific symbol
    python scripts/fetch_ohlcv.py --limit 1000      # More data
    python scripts/fetch_ohlcv.py --list-symbols    # Show available symbols
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Database connection defaults (from AMI-TRADING docker-compose)
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "database": os.environ.get("DB_DATABASE", "forecasting_db"),
    "user": os.environ.get("DB_USER", "forecasting_user"),
    "password": os.environ.get("DB_PASSWORD", "forecasting_db_password"),
}

def get_connection():
    """Get database connection."""
    try:
        import psycopg2
    except ImportError:
        print("Error: psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print(f"Config: host={DB_CONFIG['host']}, port={DB_CONFIG['port']}, db={DB_CONFIG['database']}")
        sys.exit(1)


def list_tables(conn):
    """List all tables in the database."""
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cur.fetchall()]
    cur.close()
    return tables


def list_symbols(conn):
    """List available symbols in the ohlcv table."""
    cur = conn.cursor()

    # Try to find the right table and column
    tables = list_tables(conn)
    print(f"Available tables: {tables}")

    # Common table names for OHLCV data
    ohlcv_tables = [t for t in tables if 'ohlcv' in t.lower() or 'candle' in t.lower() or 'price' in t.lower()]

    if not ohlcv_tables:
        print("No OHLCV-like tables found. Checking all tables for symbol columns...")
        for table in tables:
            cur.execute(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
            """, (table,))
            columns = [row[0] for row in cur.fetchall()]
            print(f"  {table}: {columns}")
        return []

    symbols = []
    for table in ohlcv_tables:
        try:
            cur.execute(f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol")
            symbols.extend([row[0] for row in cur.fetchall()])
        except Exception as e:
            print(f"  Could not query {table}: {e}")
            conn.rollback()

    cur.close()
    return list(set(symbols))


def fetch_ohlcv(conn, symbol: str, limit: int = 500, table: str = "ohlcv"):
    """Fetch OHLCV data for a symbol."""
    cur = conn.cursor()

    # Try to detect the table structure
    cur.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
    """, (table,))
    columns = [row[0] for row in cur.fetchall()]
    print(f"Table '{table}' columns: {columns}")

    # Build query based on available columns
    time_col = next((c for c in columns if c in ['timestamp', 'time', 'datetime', 'date', 'ts']), None)

    if time_col is None:
        print(f"Error: No timestamp column found in {table}")
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
        rows = cur.fetchall()
        cur.close()

        if not rows:
            print(f"No data found for symbol '{symbol}'")
            return None

        # Reverse to chronological order
        rows = rows[::-1]

        data = {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }

        for row in rows:
            ts, o, h, l, c, v = row
            if hasattr(ts, 'isoformat'):
                data["timestamp"].append(ts.isoformat())
            else:
                data["timestamp"].append(str(ts))
            data["open"].append(float(o))
            data["high"].append(float(h))
            data["low"].append(float(l))
            data["close"].append(float(c))
            data["volume"].append(float(v))

        return data

    except Exception as e:
        print(f"Error fetching data: {e}")
        conn.rollback()
        return None


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

    # SMA
    for window in [5, 10, 14, 20]:
        ind = ta.trend.SMAIndicator(close=df["close"], window=window)
        results[f"sma_{window}"] = to_list(ind.sma_indicator())

    # EMA
    for window in [5, 10, 12, 26]:
        ind = ta.trend.EMAIndicator(close=df["close"], window=window)
        results[f"ema_{window}"] = to_list(ind.ema_indicator())

    # WMA
    for window in [5, 10, 14]:
        ind = ta.trend.WMAIndicator(close=df["close"], window=window)
        results[f"wma_{window}"] = to_list(ind.wma())

    # RSI
    for window in [14]:
        ind = ta.momentum.RSIIndicator(close=df["close"], window=window)
        results[f"rsi_{window}"] = to_list(ind.rsi())

    # MACD
    macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    results["macd_line"] = to_list(macd.macd())
    results["macd_signal"] = to_list(macd.macd_signal())
    results["macd_histogram"] = to_list(macd.macd_diff())

    # Bollinger Bands
    for window in [20]:
        bb = ta.volatility.BollingerBands(close=df["close"], window=window, window_dev=2)
        results[f"bollinger_upper_{window}"] = to_list(bb.bollinger_hband())
        results[f"bollinger_middle_{window}"] = to_list(bb.bollinger_mavg())
        results[f"bollinger_lower_{window}"] = to_list(bb.bollinger_lband())

    # ATR
    for window in [14]:
        atr = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=window)
        results[f"atr_{window}"] = to_list(atr.average_true_range())

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)
    results["stoch_k"] = to_list(stoch.stoch())
    results["stoch_d"] = to_list(stoch.stoch_signal())

    # Williams %R
    wr = ta.momentum.WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=14)
    results["williams_r_14"] = to_list(wr.williams_r())

    # ADX
    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    results["adx_14"] = to_list(adx.adx())
    results["plus_di_14"] = to_list(adx.adx_pos())
    results["minus_di_14"] = to_list(adx.adx_neg())

    # Aroon
    aroon = ta.trend.AroonIndicator(high=df["high"], low=df["low"], window=25)
    results["aroon_up_25"] = to_list(aroon.aroon_up())
    results["aroon_down_25"] = to_list(aroon.aroon_down())

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
    results["obv"] = to_list(obv.on_balance_volume())

    # MFI
    mfi = ta.volume.MFIIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14)
    results["mfi_14"] = to_list(mfi.money_flow_index())

    # CMF
    cmf = ta.volume.ChaikinMoneyFlowIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=20)
    results["cmf_20"] = to_list(cmf.chaikin_money_flow())

    # Keltner
    kc = ta.volatility.KeltnerChannel(high=df["high"], low=df["low"], close=df["close"], window=20, window_atr=10)
    results["keltner_upper"] = to_list(kc.keltner_channel_hband())
    results["keltner_middle"] = to_list(kc.keltner_channel_mband())
    results["keltner_lower"] = to_list(kc.keltner_channel_lband())

    # Donchian
    dc = ta.volatility.DonchianChannel(high=df["high"], low=df["low"], close=df["close"], window=20)
    results["donchian_upper"] = to_list(dc.donchian_channel_hband())
    results["donchian_middle"] = to_list(dc.donchian_channel_mband())
    results["donchian_lower"] = to_list(dc.donchian_channel_lband())

    # ROC
    roc = ta.momentum.ROCIndicator(close=df["close"], window=12)
    results["roc_12"] = to_list(roc.roc())

    print(f"Calculated {len(results)} indicator series")
    return results


def save_golden_data(ohlcv: dict, indicators: dict, output_path: str, symbol: str):
    """Save golden data as JSON."""
    try:
        import ta
        source = f"python-ta v{ta.__version__}"
    except:
        source = "python-ta"

    data = {
        "description": f"Real {symbol} OHLCV data for rust-ta golden tests",
        "source": source,
        "generated_at": datetime.now().isoformat(),
        "symbol": symbol,
        "ohlcv": {
            "open": ohlcv["open"],
            "high": ohlcv["high"],
            "low": ohlcv["low"],
            "close": ohlcv["close"],
            "volume": ohlcv["volume"],
        },
        "expected": indicators,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved golden data to {output_path}")
    print(f"  OHLCV length: {len(ohlcv['close'])}")
    print(f"  Indicators: {len(indicators)}")


def generate_synthetic_ohlcv(length: int = 500) -> dict:
    """Generate realistic synthetic OHLCV data."""
    import numpy as np

    np.random.seed(42)  # Reproducible

    # Generate a random walk with trend
    base_price = 100.0
    returns = np.random.normal(0.0002, 0.015, length)  # Slight upward drift, 1.5% daily vol
    prices = base_price * np.exp(np.cumsum(returns))

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
        open_price = prev_close * (1 + np.random.normal(0, 0.002))  # Small gap

        # Realistic intraday range
        daily_range = abs(close - open_price) + close * np.random.uniform(0.005, 0.02)
        high = max(open_price, close) + daily_range * np.random.uniform(0.2, 0.8)
        low = min(open_price, close) - daily_range * np.random.uniform(0.2, 0.8)

        # Ensure OHLC invariants
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume with some correlation to price movement
        base_volume = 10000 + np.random.uniform(0, 50000)
        volatility_factor = abs(close - open_price) / open_price * 100
        volume = base_volume * (1 + volatility_factor)

        data["open"].append(round(open_price, 4))
        data["high"].append(round(high, 4))
        data["low"].append(round(low, 4))
        data["close"].append(round(close, 4))
        data["volume"].append(round(volume, 2))

        prev_close = close

    return data


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data from TimescaleDB")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--limit", type=int, default=500, help="Number of records (default: 500)")
    parser.add_argument("--table", default="market_data", help="Table name (default: market_data)")
    parser.add_argument("--list-symbols", action="store_true", help="List available symbols")
    parser.add_argument("--list-tables", action="store_true", help="List available tables")
    parser.add_argument("--output-dir", default="golden/python_outputs", help="Output directory")
    parser.add_argument("--no-indicators", action="store_true", help="Skip indicator calculation")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    args = parser.parse_args()

    conn = get_connection()
    print(f"Connected to {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")

    if args.list_tables:
        tables = list_tables(conn)
        print("Tables:")
        for t in tables:
            print(f"  - {t}")
        conn.close()
        return

    if args.list_symbols:
        symbols = list_symbols(conn)
        if symbols:
            print("Symbols:")
            for s in symbols:
                print(f"  - {s}")
        conn.close()
        return

    # Fetch or generate OHLCV data
    ohlcv = None
    symbol = args.symbol

    if args.synthetic:
        print(f"Generating {args.limit} synthetic OHLCV records...")
        ohlcv = generate_synthetic_ohlcv(args.limit)
        symbol = "SYNTHETIC"
    else:
        print(f"Fetching {args.limit} records for {args.symbol} from {args.table}...")
        ohlcv = fetch_ohlcv(conn, args.symbol, args.limit, args.table)
        conn.close()

        if ohlcv is None or len(ohlcv.get("close", [])) == 0:
            print("No data in database. Generating synthetic data as fallback...")
            ohlcv = generate_synthetic_ohlcv(args.limit)
            symbol = "SYNTHETIC"

    print(f"Got {len(ohlcv['close'])} OHLCV records")

    # Calculate indicators
    if args.no_indicators:
        indicators = {}
    else:
        print("Calculating indicators...")
        indicators = calculate_indicators(ohlcv)

    # Save to file
    script_dir = Path(__file__).parent.parent
    output_path = script_dir / args.output_dir / f"{symbol.lower()}_ohlcv.json"
    save_golden_data(ohlcv, indicators, str(output_path), symbol)

    # Also update sample_ohlcv.json for tests
    sample_path = script_dir / args.output_dir / "sample_ohlcv.json"
    save_golden_data(ohlcv, indicators, str(sample_path), symbol)
    print(f"Also updated {sample_path}")


if __name__ == "__main__":
    main()
