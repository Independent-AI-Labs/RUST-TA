//! Prelude for ta-indicators.
//!
//! This module re-exports all commonly used types and traits.

// Momentum indicators
pub use crate::momentum::{
    Rsi, RsiConfig, RsiState,
    StochRsi, StochRsiConfig, StochRsiOutput, StochRsiState,
    Stochastic, StochasticConfig, StochasticOutput, StochasticState,
    WilliamsR, WilliamsRConfig, WilliamsRState,
    Roc, RocConfig, RocState,
};

// Trend indicators
pub use crate::trend::{
    Sma, SmaConfig, SmaState,
    Ema, EmaConfig, EmaState,
    Wma, WmaConfig, WmaState,
    Macd, MacdConfig, MacdOutput, MacdState,
    Adx, AdxConfig, AdxOutput, AdxState,
    Aroon, AroonConfig, AroonOutput, AroonState,
};

// Volatility indicators
pub use crate::volatility::{
    Atr, AtrConfig, AtrState,
    BollingerBands, BollingerConfig, BollingerOutput, BollingerState,
    KeltnerChannel, KeltnerConfig, KeltnerOutput, KeltnerState,
    DonchianChannel, DonchianConfig, DonchianOutput, DonchianState,
};

// Volume indicators
pub use crate::volume::{
    Obv, ObvConfig, ObvState,
    Vwap, VwapConfig, VwapMode, VwapState,
    Mfi, MfiConfig, MfiState,
    Cmf, CmfConfig, CmfState,
};
