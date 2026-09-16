//! MarbleNet-specific streaming iterator.

use crate::{marblenet_frontend::OUTPUT_FRAME_SAMPLES, utils, vad_iter};

pub trait MarbleNetModel {
    /// Computes speech probabilities for a waveform.
    ///
    /// # Errors
    ///
    /// Returns backend-specific preprocessing or inference errors.
    fn speech_probabilities(&mut self, waveform: &[f32]) -> anyhow::Result<Vec<f32>>;
}

#[cfg(feature = "candle")]
impl MarbleNetModel for crate::marblenet::MarbleNet {
    fn speech_probabilities(&mut self, waveform: &[f32]) -> anyhow::Result<Vec<f32>> {
        crate::marblenet::MarbleNet::speech_probabilities(self, waveform)
    }
}

#[cfg(feature = "onnxruntime")]
impl MarbleNetModel for crate::marblenet_ort::MarbleNet {
    fn speech_probabilities(&mut self, waveform: &[f32]) -> anyhow::Result<Vec<f32>> {
        crate::marblenet_ort::MarbleNet::speech_probabilities(self, waveform)
    }
}

pub struct MarbleNetIter<M> {
    model: M,
    params: utils::VadParams,
    speeches: Vec<utils::TimeStamp>,
}

impl<M: MarbleNetModel> MarbleNetIter<M> {
    #[must_use]
    pub fn new(model: M, mut params: utils::VadParams) -> Self {
        params.frame_size = OUTPUT_FRAME_SAMPLES
            .checked_mul(1_000)
            .and_then(|samples| samples.checked_div(params.sample_rate))
            .unwrap_or(0);
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
    /// Returns an error for invalid parameters or samples, or when probability
    /// inference fails.
    pub fn process(&mut self, samples: &[f32]) -> anyhow::Result<&[utils::TimeStamp]> {
        vad_iter::validate_parameters(&self.params)?;
        vad_iter::validate_samples(samples)?;
        let probabilities = self.model.speech_probabilities(samples)?;
        self.speeches =
            vad_iter::segment_probabilities(&probabilities, samples.len(), &self.params)?;
        Ok(&self.speeches)
    }
}
