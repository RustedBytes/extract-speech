//! FSMN-VAD-specific streaming iterator.

use crate::{utils, vad_iter};

/// Backend contract used by the FSMN-VAD iterator.
pub trait FsmnVadModel {
    /// Computes frame-level speech probabilities for a waveform.
    ///
    /// # Errors
    ///
    /// Returns backend-specific preprocessing or inference errors.
    fn speech_probabilities(&mut self, waveform: &[f32]) -> anyhow::Result<Vec<f32>>;
}

#[cfg(feature = "candle")]
impl FsmnVadModel for crate::models::fsmn::candle::FsmnVad {
    fn speech_probabilities(&mut self, waveform: &[f32]) -> anyhow::Result<Vec<f32>> {
        Self::speech_probabilities(self, waveform)
    }
}

#[cfg(feature = "onnxruntime")]
impl FsmnVadModel for crate::models::fsmn::onnx::FsmnVad {
    fn speech_probabilities(&mut self, waveform: &[f32]) -> anyhow::Result<Vec<f32>> {
        Self::speech_probabilities(self, waveform)
    }
}

/// Runtime-independent FSMN-VAD iterator.
pub struct FsmnVadIterator<M> {
    model: M,
    params: utils::VadParams,
    speeches: Vec<utils::TimeStamp>,
}

impl<M: FsmnVadModel> FsmnVadIterator<M> {
    /// Creates an iterator around a loaded FSMN-VAD backend.
    #[must_use]
    pub fn new(model: M, mut params: utils::VadParams) -> Self {
        params.frame_size = 10;
        Self {
            model,
            params,
            speeches: Vec::new(),
        }
    }

    /// Detects speech segments in one complete waveform.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid parameters or samples, or when FSMN
    /// feature extraction or inference fails.
    pub fn process(&mut self, samples: &[f32]) -> anyhow::Result<&[utils::TimeStamp]> {
        vad_iter::validate_parameters(&self.params)?;
        vad_iter::validate_samples(samples)?;
        let probabilities = self.model.speech_probabilities(samples)?;
        self.speeches =
            vad_iter::segment_probabilities(&probabilities, samples.len(), &self.params)?;
        Ok(&self.speeches)
    }
}

/// Compatibility name for the original ONNX Runtime iterator.
#[cfg(feature = "onnxruntime")]
pub type FsmnVadIter = FsmnVadIterator<crate::models::fsmn::onnx::FsmnVad>;

/// FSMN-VAD iterator exposed by Candle-only builds.
#[cfg(all(feature = "candle", not(feature = "onnxruntime")))]
pub type FsmnVadIter = FsmnVadIterator<crate::models::fsmn::candle::FsmnVad>;
