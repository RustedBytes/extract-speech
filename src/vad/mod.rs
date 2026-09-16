//! Runtime-independent VAD configuration, iteration, and detector APIs.

pub mod config;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub mod detector;
pub mod iterator;
