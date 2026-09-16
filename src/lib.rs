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

#[cfg(feature = "download")]
pub mod assets;
#[cfg(feature = "cli")]
pub mod audio;
#[cfg(feature = "python")]
mod bindings;
pub mod models;
pub mod vad;

// Compatibility aliases for the pre-0.7 flat module layout.
#[cfg(feature = "download")]
pub use assets as download;
#[cfg(feature = "cli")]
pub use audio::{opus, resampler};
#[cfg(feature = "onnxruntime")]
pub use models::fsmn::{
    frontend as fsmn_vad_frontend, iterator as fsmn_vad_iter, onnx as fsmn_vad_ort,
};
#[cfg(feature = "candle")]
pub use models::marblenet::candle as marblenet;
#[cfg(feature = "onnxruntime")]
pub use models::marblenet::onnx as marblenet_ort;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub use models::marblenet::{frontend as marblenet_frontend, iterator as marblenet_iter};
#[cfg(feature = "candle")]
pub use models::pulsevad::candle as pulsevad;
#[cfg(feature = "onnxruntime")]
pub use models::pulsevad::onnx as pulsevad_ort;
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub use models::pulsevad::{frontend as pulsevad_frontend, iterator as pulsevad_iter};
#[cfg(feature = "onnxruntime")]
pub use models::pyannote::{iterator as pyannote_vad_iter, onnx as pyannote_vad_ort};
#[cfg(feature = "candle")]
pub use models::silero::candle as silero_v5;
#[cfg(feature = "onnxruntime")]
pub use models::silero::onnx as silero_v5_ort;
#[cfg(feature = "onnxruntime")]
pub use models::ten::onnx as ten_vad_ort;
pub use vad::{config as utils, iterator as vad_iter};

pub use anyhow::{Error, Result};
pub use utils::{TimeStamp, VadParams, VAD_SAMPLE_RATE};
pub use utils::{TimeStamp as SpeechSegment, VAD_SAMPLE_RATE as SAMPLE_RATE};
#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub use vad::detector::{Detector, DetectorBuilder, Model, Runtime};
pub use vad_iter::{VadIter, VadModel};

#[cfg(feature = "candle")]
pub use candle_core;
#[cfg(feature = "onnxruntime")]
pub use ort;

#[cfg(feature = "onnxruntime")]
pub use vad::detector::init_onnx_runtime;
