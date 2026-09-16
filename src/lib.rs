//! Voice activity detection inference for several popular VAD model families.
//!
//! [`Detector`] accepts mono, normalized `f32` PCM sampled at
//! [`SAMPLE_RATE`] and returns speech intervals expressed as sample offsets.
//!
//! ```no_run
//! use extract_speech::{Detector, Model, Result, Runtime, VadParams};
//!
//! # fn main() -> Result<()> {
//! let mut detector = Detector::builder("models/silero-vad-v5.onnx")
//!     .model(Model::Silero)
//!     .runtime(Runtime::Candle)
//!     .parameters(VadParams {
//!         threshold: 0.7,
//!         ..VadParams::default()
//!     })
//!     .build()?;
//!
//! let samples = vec![0.0_f32; 16_000];
//! let segments = detector.detect(&samples)?;
//! # Ok(())
//! # }
//! ```

#[cfg(any(feature = "candle", feature = "onnxruntime"))]
mod detector;
pub mod utils;
pub mod vad_iter;

#[cfg(feature = "candle")]
pub mod marblenet;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub mod marblenet_frontend;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub mod marblenet_iter;
#[cfg(feature = "onnxruntime")]
pub mod marblenet_ort;

#[cfg(feature = "candle")]
pub mod pulsevad;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub mod pulsevad_frontend;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub mod pulsevad_iter;
#[cfg(feature = "onnxruntime")]
pub mod pulsevad_ort;

#[cfg(feature = "candle")]
pub mod silero_v5;
#[cfg(feature = "onnxruntime")]
pub mod silero_v5_ort;

#[cfg(feature = "onnxruntime")]
pub mod fsmn_vad_frontend;
#[cfg(feature = "onnxruntime")]
pub mod fsmn_vad_iter;
#[cfg(feature = "onnxruntime")]
pub mod fsmn_vad_ort;
#[cfg(feature = "onnxruntime")]
pub mod pyannote_vad_iter;
#[cfg(feature = "onnxruntime")]
pub mod pyannote_vad_ort;
#[cfg(feature = "onnxruntime")]
pub mod ten_vad_ort;

#[cfg(feature = "cli")]
pub mod audio;
#[cfg(feature = "cli")]
pub mod opus;
#[cfg(feature = "cli")]
pub mod resampler;

pub use anyhow::{Error, Result};
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub use detector::{Detector, DetectorBuilder, Model, Runtime};
pub use utils::{TimeStamp, VadParams, VAD_SAMPLE_RATE};
pub use utils::{TimeStamp as SpeechSegment, VAD_SAMPLE_RATE as SAMPLE_RATE};
pub use vad_iter::{VadIter, VadModel};

#[cfg(feature = "candle")]
pub use candle_core;
#[cfg(feature = "onnxruntime")]
pub use ort;

#[cfg(feature = "onnxruntime")]
pub use detector::init_onnx_runtime;
