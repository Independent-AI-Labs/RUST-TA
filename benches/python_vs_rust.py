#!/usr/bin/env python3
"""
Performance benchmark: python-ta vs rust-ta

Runs comprehensive benchmarks comparing indicator calculation performance
between python-ta and rust-ta on large datasets.

Usage:
    python benches/python_vs_rust.py                    # Default 100k candles
    python benches/python_vs_rust.py --candles 500000   # Custom size
    python benches/python_vs_rust.py --duration 30      # Run for 30 seconds each
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    implementation: str  # "python" or "rust"
    candles: int
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_candles_per_sec: float


def load_ohlcv_data(filepath: str) -> dict:
    """Load OHLCV data from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data["ohlcv"]


def prepare_dataframe(ohlcv: dict):
    """Convert OHLCV dict to pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame({
        "open": ohlcv["open"],
        "high": ohlcv["high"],
        "low": ohlcv["low"],
        "close": ohlcv["close"],
        "volume": ohlcv["volume"],
    })


# =============================================================================
# Python Benchmarks
# =============================================================================

def benchmark_python_indicator(name: str, func: Callable, df, iterations: int) -> BenchmarkResult:
    """Benchmark a python-ta indicator."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = func(df)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    total_time = sum(times)
    avg_time = total_time / iterations
    candles = len(df)

    return BenchmarkResult(
        name=name,
        implementation="python-ta",
        candles=candles,
        iterations=iterations,
        total_time_ms=total_time,
        avg_time_ms=avg_time,
        min_time_ms=min(times),
        max_time_ms=max(times),
        throughput_candles_per_sec=(candles / avg_time) * 1000 if avg_time > 0 else 0,
    )


def run_python_benchmarks(df, iterations: int = 10) -> list[BenchmarkResult]:
    """Run all python-ta benchmarks."""
    import ta

    results = []

    benchmarks = [
        # Trend
        ("SMA(14)", lambda d: ta.trend.SMAIndicator(d["close"], window=14).sma_indicator()),
        ("SMA(50)", lambda d: ta.trend.SMAIndicator(d["close"], window=50).sma_indicator()),
        ("SMA(200)", lambda d: ta.trend.SMAIndicator(d["close"], window=200).sma_indicator()),
        ("EMA(14)", lambda d: ta.trend.EMAIndicator(d["close"], window=14).ema_indicator()),
        ("EMA(50)", lambda d: ta.trend.EMAIndicator(d["close"], window=50).ema_indicator()),
        ("WMA(14)", lambda d: ta.trend.WMAIndicator(d["close"], window=14).wma()),
        ("MACD", lambda d: ta.trend.MACD(d["close"]).macd()),
        ("ADX(14)", lambda d: ta.trend.ADXIndicator(d["high"], d["low"], d["close"], window=14).adx()),
        ("Aroon(25)", lambda d: ta.trend.AroonIndicator(d["high"], d["low"], window=25).aroon_up()),

        # Momentum
        ("RSI(14)", lambda d: ta.momentum.RSIIndicator(d["close"], window=14).rsi()),
        ("RSI(7)", lambda d: ta.momentum.RSIIndicator(d["close"], window=7).rsi()),
        ("Stochastic", lambda d: ta.momentum.StochasticOscillator(d["high"], d["low"], d["close"]).stoch()),
        ("Williams %R", lambda d: ta.momentum.WilliamsRIndicator(d["high"], d["low"], d["close"]).williams_r()),
        ("ROC(12)", lambda d: ta.momentum.ROCIndicator(d["close"], window=12).roc()),

        # Volatility
        ("Bollinger", lambda d: ta.volatility.BollingerBands(d["close"], window=20).bollinger_mavg()),
        ("ATR(14)", lambda d: ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=14).average_true_range()),
        ("Keltner", lambda d: ta.volatility.KeltnerChannel(d["high"], d["low"], d["close"]).keltner_channel_mband()),
        ("Donchian", lambda d: ta.volatility.DonchianChannel(d["high"], d["low"], d["close"]).donchian_channel_mband()),

        # Volume
        ("OBV", lambda d: ta.volume.OnBalanceVolumeIndicator(d["close"], d["volume"]).on_balance_volume()),
        ("MFI(14)", lambda d: ta.volume.MFIIndicator(d["high"], d["low"], d["close"], d["volume"], window=14).money_flow_index()),
        ("CMF(20)", lambda d: ta.volume.ChaikinMoneyFlowIndicator(d["high"], d["low"], d["close"], d["volume"], window=20).chaikin_money_flow()),
    ]

    for name, func in benchmarks:
        print(f"  Python: {name}...", end=" ", flush=True)
        result = benchmark_python_indicator(name, func, df, iterations)
        results.append(result)
        print(f"{result.avg_time_ms:.2f}ms ({result.throughput_candles_per_sec:,.0f} candles/sec)")

    return results


# =============================================================================
# Rust Benchmarks
# =============================================================================

def run_rust_benchmarks(data_file: str, iterations: int = 10) -> list[BenchmarkResult]:
    """Run rust-ta benchmarks via criterion or custom benchmark binary."""
    # For now, we'll create a benchmark binary that outputs JSON results
    # This will be called from here

    results = []

    # Check if benchmark binary exists
    bench_bin = Path(__file__).parent.parent / "target/release/rust_ta_bench"
    if not bench_bin.exists():
        print("  Building Rust benchmark binary...")
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "rust_ta_bench"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
        )

    if not bench_bin.exists():
        print("  Warning: Rust benchmark binary not available. Skipping Rust benchmarks.")
        return results

    # Run benchmark
    try:
        proc = subprocess.run(
            [str(bench_bin), data_file, str(iterations)],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if proc.returncode == 0:
            raw_results = json.loads(proc.stdout)
            for r in raw_results:
                result = BenchmarkResult(**r)
                results.append(result)
                print(f"  Rust: {result.name}... {result.avg_time_ms:.2f}ms ({result.throughput_candles_per_sec:,.0f} candles/sec)")
        else:
            print(f"  Rust benchmark failed: {proc.stderr}")
    except Exception as e:
        print(f"  Error running Rust benchmarks: {e}")

    return results


# =============================================================================
# Reporting
# =============================================================================

def print_comparison_table(python_results: list[BenchmarkResult], rust_results: list[BenchmarkResult]):
    """Print a comparison table of benchmark results."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    # Create lookup for rust results
    rust_lookup = {r.name: r for r in rust_results}

    print(f"\n{'Indicator':<20} {'Python (ms)':<15} {'Rust (ms)':<15} {'Speedup':<12} {'Python/sec':<18} {'Rust/sec':<18}")
    print("-" * 100)

    total_python_time = 0
    total_rust_time = 0
    speedups = []

    for pr in python_results:
        total_python_time += pr.avg_time_ms

        rr = rust_lookup.get(pr.name)
        if rr:
            total_rust_time += rr.avg_time_ms
            speedup = pr.avg_time_ms / rr.avg_time_ms if rr.avg_time_ms > 0 else 0
            speedups.append(speedup)
            print(f"{pr.name:<20} {pr.avg_time_ms:<15.2f} {rr.avg_time_ms:<15.2f} {speedup:<12.1f}x {pr.throughput_candles_per_sec:<18,.0f} {rr.throughput_candles_per_sec:<18,.0f}")
        else:
            print(f"{pr.name:<20} {pr.avg_time_ms:<15.2f} {'N/A':<15} {'N/A':<12} {pr.throughput_candles_per_sec:<18,.0f} {'N/A':<18}")

    print("-" * 100)

    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"\n{'TOTAL':<20} {total_python_time:<15.2f} {total_rust_time:<15.2f} {avg_speedup:<12.1f}x avg")
    else:
        print(f"\n{'TOTAL':<20} {total_python_time:<15.2f}")

    print("=" * 100)


def generate_markdown_table(python_results: list[BenchmarkResult], rust_results: list[BenchmarkResult],
                            metadata: dict) -> str:
    """Generate a markdown table for README."""
    rust_lookup = {r.name: r for r in rust_results}

    lines = []
    lines.append("## Performance Benchmarks")
    lines.append("")
    lines.append(f"Benchmarked on {metadata['candles']:,} candles ({metadata['iterations']} iterations)")
    lines.append("")
    lines.append("| Indicator | python-ta | rust-ta | Speedup |")
    lines.append("|-----------|-----------|---------|---------|")

    total_python = 0
    total_rust = 0
    speedups = []

    for pr in python_results:
        total_python += pr.avg_time_ms
        rr = rust_lookup.get(pr.name)
        if rr:
            total_rust += rr.avg_time_ms
            speedup = pr.avg_time_ms / rr.avg_time_ms if rr.avg_time_ms > 0 else 0
            speedups.append(speedup)

            # Format times nicely
            if pr.avg_time_ms >= 1000:
                py_str = f"{pr.avg_time_ms/1000:.2f}s"
            elif pr.avg_time_ms >= 1:
                py_str = f"{pr.avg_time_ms:.2f}ms"
            else:
                py_str = f"{pr.avg_time_ms*1000:.1f}μs"

            if rr.avg_time_ms >= 1000:
                rs_str = f"{rr.avg_time_ms/1000:.2f}s"
            elif rr.avg_time_ms >= 1:
                rs_str = f"{rr.avg_time_ms:.2f}ms"
            else:
                rs_str = f"{rr.avg_time_ms*1000:.1f}μs"

            lines.append(f"| {pr.name} | {py_str} | {rs_str} | **{speedup:.0f}x** |")

    # Summary row
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0

    if total_python >= 1000:
        py_total = f"{total_python/1000:.2f}s"
    else:
        py_total = f"{total_python:.2f}ms"

    if total_rust >= 1000:
        rs_total = f"{total_rust/1000:.2f}s"
    else:
        rs_total = f"{total_rust:.2f}ms"

    lines.append(f"| **Total** | **{py_total}** | **{rs_total}** | **{avg_speedup:.0f}x avg** |")
    lines.append("")

    return "\n".join(lines)


def save_results(python_results: list[BenchmarkResult], rust_results: list[BenchmarkResult],
                 output_path: str, metadata: dict):
    """Save benchmark results to JSON."""
    data = {
        "metadata": metadata,
        "python_results": [r.__dict__ for r in python_results],
        "rust_results": [r.__dict__ for r in rust_results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark python-ta vs rust-ta")
    parser.add_argument("--data", default=None, help="Path to OHLCV JSON data file")
    parser.add_argument("--candles", type=int, default=100000, help="Number of candles to benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per benchmark")
    parser.add_argument("--duration", type=int, default=None, help="Target duration in seconds (overrides iterations)")
    parser.add_argument("--output", default=None, help="Output file for results")
    parser.add_argument("--python-only", action="store_true", help="Only run Python benchmarks")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent

    # Find or generate data file
    if args.data:
        data_file = args.data
    else:
        # Look for existing data file
        data_dir = script_dir / "data"
        data_files = list(data_dir.glob(f"*_{args.candles}_*.json")) if data_dir.exists() else []

        if data_files:
            data_file = str(data_files[0])
            print(f"Using existing data file: {data_file}")
        else:
            # Generate synthetic data
            print(f"Generating {args.candles:,} synthetic candles...")
            from fetch_ohlcv import generate_synthetic_ohlcv, save_data
            ohlcv = generate_synthetic_ohlcv(args.candles)
            data_file = str(data_dir / f"synthetic_{args.candles}.json")
            save_data(ohlcv, {}, data_file, "SYNTHETIC", "synthetic")

    # Load data
    print(f"\nLoading data from {data_file}...")
    ohlcv = load_ohlcv_data(data_file)
    df = prepare_dataframe(ohlcv)
    print(f"Loaded {len(df):,} candles")

    # Calculate iterations from duration if specified
    iterations = args.iterations
    if args.duration:
        # Do a warmup run to estimate time per iteration
        import ta
        start = time.perf_counter()
        ta.trend.SMAIndicator(df["close"], window=14).sma_indicator()
        warmup_time = time.perf_counter() - start

        # Estimate iterations needed for target duration
        # Assume ~20 indicators, each taking similar time
        estimated_total_time_per_iter = warmup_time * 20
        iterations = max(1, int(args.duration / estimated_total_time_per_iter))
        print(f"Target duration: {args.duration}s -> {iterations} iterations")

    # Run Python benchmarks
    print(f"\nRunning Python benchmarks ({iterations} iterations)...")
    python_results = run_python_benchmarks(df, iterations)

    # Run Rust benchmarks
    rust_results = []
    if not args.python_only:
        print(f"\nRunning Rust benchmarks ({iterations} iterations)...")
        rust_results = run_rust_benchmarks(data_file, iterations)

    # Print comparison
    print_comparison_table(python_results, rust_results)

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(script_dir / "benches" / f"results_{timestamp}.json")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_file": data_file,
        "candles": len(df),
        "iterations": iterations,
        "python_version": sys.version,
    }

    save_results(python_results, rust_results, output_path, metadata)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_python = sum(r.avg_time_ms for r in python_results)
    print(f"Total Python time (all indicators): {total_python:.2f}ms")
    print(f"Average throughput: {(len(df) / total_python) * 1000 * len(python_results):,.0f} candles/sec")

    if rust_results:
        total_rust = sum(r.avg_time_ms for r in rust_results)
        speedup = total_python / total_rust if total_rust > 0 else 0
        print(f"Total Rust time (all indicators): {total_rust:.2f}ms")
        print(f"Overall speedup: {speedup:.1f}x")

        # Generate and print markdown table for README
        print("\n" + "=" * 60)
        print("MARKDOWN TABLE FOR README (copy-paste below)")
        print("=" * 60 + "\n")
        markdown = generate_markdown_table(python_results, rust_results, metadata)
        print(markdown)

        # Save markdown to file
        md_path = str(script_dir / "benches" / "BENCHMARK_RESULTS.md")
        with open(md_path, "w") as f:
            f.write(markdown)
        print(f"\nMarkdown saved to: {md_path}")


if __name__ == "__main__":
    main()
