use crate::{fsmn_vad_ort::FsmnVad, utils, vad_iter};

pub struct FsmnVadIter {
    model: FsmnVad,
    params: utils::VadParams,
    speeches: Vec<utils::TimeStamp>,
}

impl FsmnVadIter {
    pub fn new(model: FsmnVad, mut params: utils::VadParams) -> Self {
        params.frame_size = 10;
        Self {
            model,
            params,
            speeches: Vec::new(),
        }
    }

    pub fn process(&mut self, samples: &[f32]) -> anyhow::Result<&[utils::TimeStamp]> {
        let probabilities = self.model.speech_probabilities(samples)?;
        self.speeches =
            vad_iter::segment_probabilities(&probabilities, samples.len(), self.params.clone());
        Ok(&self.speeches)
    }
}
