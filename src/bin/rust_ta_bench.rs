//! Benchmark binary for rust-ta performance comparison with python-ta.
//!
//! Usage:
//!     rust_ta_bench <data_file> <iterations>
//!
//! Outputs JSON array of benchmark results to stdout.

use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use ta_core::prelude::*;
use ta_indicators::prelude::*;

#[derive(Debug, Deserialize)]
struct OhlcvData {
    ohlcv: OhlcvRaw,
}

#[derive(Debug, Deserialize)]
struct OhlcvRaw {
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    name: String,
    implementation: String,
    candles: usize,
    iterations: usize,
    total_time_ms: f64,
    avg_time_ms: f64,
    min_time_ms: f64,
    max_time_ms: f64,
    throughput_candles_per_sec: f64,
}

fn load_ohlcv(path: &str) -> (OhlcvSeries<f64>, Vec<f64>) {
    let file = File::open(path).expect("Failed to open data file");
    let reader = BufReader::new(file);
    let data: OhlcvData = serde_json::from_reader(reader).expect("Failed to parse JSON");

    let mut series = OhlcvSeries::with_capacity(data.ohlcv.close.len());
    for i in 0..data.ohlcv.close.len() {
        let bar = Bar::new(
            data.ohlcv.open[i],
            data.ohlcv.high[i],
            data.ohlcv.low[i],
            data.ohlcv.close[i],
            data.ohlcv.volume[i],
        );
        series.push(bar);
    }

    (series, data.ohlcv.close)
}

fn benchmark<F>(name: &str, iterations: usize, candles: usize, mut f: F) -> BenchmarkResult
where
    F: FnMut(),
{
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed);
    }

    let total_time: f64 = times.iter().sum();
    let avg_time = total_time / iterations as f64;
    let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max_time = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let throughput = if avg_time > 0.0 {
        (candles as f64 / avg_time) * 1000.0
    } else {
        0.0
    };

    BenchmarkResult {
        name: name.to_string(),
        implementation: "rust-ta".to_string(),
        candles,
        iterations,
        total_time_ms: total_time,
        avg_time_ms: avg_time,
        min_time_ms: min_time,
        max_time_ms: max_time,
        throughput_candles_per_sec: throughput,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <data_file> <iterations>", args[0]);
        std::process::exit(1);
    }

    let data_file = &args[1];
    let iterations: usize = args[2].parse().expect("Invalid iterations");

    eprintln!("Loading data from {}...", data_file);
    let (series, close) = load_ohlcv(data_file);
    let candles = series.len();
    eprintln!("Loaded {} candles", candles);

    let mut results = Vec::new();

    // SMA benchmarks
    eprintln!("  Rust: SMA(14)...");
    results.push(benchmark("SMA(14)", iterations, candles, || {
        let mut sma = Sma::<f64>::new(SmaConfig { window: 14, fillna: false });
        let _ = sma.calculate(&series);
    }));

    eprintln!("  Rust: SMA(50)...");
    results.push(benchmark("SMA(50)", iterations, candles, || {
        let mut sma = Sma::<f64>::new(SmaConfig { window: 50, fillna: false });
        let _ = sma.calculate(&series);
    }));

    eprintln!("  Rust: SMA(200)...");
    results.push(benchmark("SMA(200)", iterations, candles, || {
        let mut sma = Sma::<f64>::new(SmaConfig { window: 200, fillna: false });
        let _ = sma.calculate(&series);
    }));

    // EMA benchmarks
    eprintln!("  Rust: EMA(14)...");
    results.push(benchmark("EMA(14)", iterations, candles, || {
        let mut ema = Ema::<f64>::new(EmaConfig { window: 14, fillna: false });
        let _ = ema.calculate(&series);
    }));

    eprintln!("  Rust: EMA(50)...");
    results.push(benchmark("EMA(50)", iterations, candles, || {
        let mut ema = Ema::<f64>::new(EmaConfig { window: 50, fillna: false });
        let _ = ema.calculate(&series);
    }));

    // WMA benchmark
    eprintln!("  Rust: WMA(14)...");
    results.push(benchmark("WMA(14)", iterations, candles, || {
        let mut wma = Wma::<f64>::new(WmaConfig { window: 14, fillna: false });
        let _ = wma.calculate(&series);
    }));

    // MACD benchmark
    eprintln!("  Rust: MACD...");
    results.push(benchmark("MACD", iterations, candles, || {
        let mut macd = Macd::<f64>::new(MacdConfig::default());
        let _ = macd.calculate(&series);
    }));

    // ADX benchmark
    eprintln!("  Rust: ADX(14)...");
    results.push(benchmark("ADX(14)", iterations, candles, || {
        let mut adx = Adx::<f64>::new(AdxConfig { window: 14, fillna: false });
        let _ = adx.calculate(&series);
    }));

    // Aroon benchmark
    eprintln!("  Rust: Aroon(25)...");
    results.push(benchmark("Aroon(25)", iterations, candles, || {
        let mut aroon = Aroon::<f64>::new(AroonConfig { window: 25, fillna: false });
        let _ = aroon.calculate(&series);
    }));

    // RSI benchmarks
    eprintln!("  Rust: RSI(14)...");
    results.push(benchmark("RSI(14)", iterations, candles, || {
        let mut rsi = Rsi::<f64>::new(RsiConfig { window: 14, fillna: false });
        let _ = rsi.calculate(&series);
    }));

    eprintln!("  Rust: RSI(7)...");
    results.push(benchmark("RSI(7)", iterations, candles, || {
        let mut rsi = Rsi::<f64>::new(RsiConfig { window: 7, fillna: false });
        let _ = rsi.calculate(&series);
    }));

    // Stochastic benchmark
    eprintln!("  Rust: Stochastic...");
    results.push(benchmark("Stochastic", iterations, candles, || {
        let mut stoch = Stochastic::<f64>::new(StochasticConfig::default());
        let _ = stoch.calculate(&series);
    }));

    // Williams %R benchmark
    eprintln!("  Rust: Williams %R...");
    results.push(benchmark("Williams %R", iterations, candles, || {
        let mut wr = WilliamsR::<f64>::new(WilliamsRConfig::default());
        let _ = wr.calculate(&series);
    }));

    // ROC benchmark
    eprintln!("  Rust: ROC(12)...");
    results.push(benchmark("ROC(12)", iterations, candles, || {
        let mut roc = Roc::<f64>::new(RocConfig { window: 12, fillna: false });
        let _ = roc.calculate(&series);
    }));

    // Bollinger benchmark
    eprintln!("  Rust: Bollinger...");
    results.push(benchmark("Bollinger", iterations, candles, || {
        let mut bb = BollingerBands::<f64>::new(BollingerConfig::default());
        let _ = bb.calculate(&series);
    }));

    // ATR benchmark
    eprintln!("  Rust: ATR(14)...");
    results.push(benchmark("ATR(14)", iterations, candles, || {
        let mut atr = Atr::<f64>::new(AtrConfig { window: 14, fillna: false });
        let _ = atr.calculate(&series);
    }));

    // Keltner benchmark
    eprintln!("  Rust: Keltner...");
    results.push(benchmark("Keltner", iterations, candles, || {
        let mut kc = KeltnerChannel::<f64>::new(KeltnerConfig::default());
        let _ = kc.calculate(&series);
    }));

    // Donchian benchmark
    eprintln!("  Rust: Donchian...");
    results.push(benchmark("Donchian", iterations, candles, || {
        let mut dc = DonchianChannel::<f64>::new(DonchianConfig::default());
        let _ = dc.calculate(&series);
    }));

    // OBV benchmark
    eprintln!("  Rust: OBV...");
    results.push(benchmark("OBV", iterations, candles, || {
        let mut obv = Obv::<f64>::new(ObvConfig::default());
        let _ = obv.calculate(&series);
    }));

    // MFI benchmark
    eprintln!("  Rust: MFI(14)...");
    results.push(benchmark("MFI(14)", iterations, candles, || {
        let mut mfi = Mfi::<f64>::new(MfiConfig { window: 14, fillna: false });
        let _ = mfi.calculate(&series);
    }));

    // CMF benchmark
    eprintln!("  Rust: CMF(20)...");
    results.push(benchmark("CMF(20)", iterations, candles, || {
        let mut cmf = Cmf::<f64>::new(CmfConfig { window: 20, fillna: false });
        let _ = cmf.calculate(&series);
    }));

    // Output results as JSON
    let json = serde_json::to_string(&results).expect("Failed to serialize results");
    println!("{}", json);
}
