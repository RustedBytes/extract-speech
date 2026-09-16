//! FSMN-VAD-specific streaming iterator.

use crate::{fsmn_vad_ort::FsmnVad, utils, vad_iter};

pub struct FsmnVadIter {
    model: FsmnVad,
    params: utils::VadParams,
    speeches: Vec<utils::TimeStamp>,
}

impl FsmnVadIter {
    #[must_use]
    pub fn new(model: FsmnVad, mut params: utils::VadParams) -> Self {
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
    /// Returns an error when FSMN feature extraction or inference fails.
    pub fn process(&mut self, samples: &[f32]) -> anyhow::Result<&[utils::TimeStamp]> {
        let probabilities = self.model.speech_probabilities(samples)?;
        self.speeches =
            vad_iter::segment_probabilities(&probabilities, samples.len(), self.params.clone());
        Ok(&self.speeches)
    }
}
