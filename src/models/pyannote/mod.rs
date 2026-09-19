//! `PyAnnote` segmentation implementations and shared iterator.

#[cfg(feature = "candle")]
pub mod candle;
pub mod iterator;
#[cfg(feature = "onnxruntime")]
pub mod onnx;
