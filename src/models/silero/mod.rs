//! Silero VAD implementations.

#[cfg(feature = "candle")]
pub mod candle;
#[cfg(feature = "onnxruntime")]
pub mod onnx;
