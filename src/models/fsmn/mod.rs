//! FSMN-VAD implementations and preprocessing.

#[cfg(feature = "candle")]
pub mod candle;
pub mod frontend;
pub mod iterator;
#[cfg(feature = "onnxruntime")]
pub mod onnx;
