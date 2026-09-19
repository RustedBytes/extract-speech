//! Model-specific inference implementations.

#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub mod fsmn;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub mod marblenet;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub mod pulsevad;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub mod pyannote;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub mod silero;
#[cfg(feature = "onnxruntime")]
pub mod ten;
