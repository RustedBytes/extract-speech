use crate::{marblenet_frontend::OUTPUT_FRAME_SAMPLES, utils, vad_iter};

pub trait MarbleNetModel {
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
    pub fn new(model: M, mut params: utils::VadParams) -> Self {
        params.frame_size = OUTPUT_FRAME_SAMPLES * 1_000 / params.sample_rate;
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
