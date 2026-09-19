//! Candle implementation of FSMN-VAD.

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use log::debug;

use crate::fsmn_vad_frontend::{FsmnVadFrontend, FEATURE_DIM};

const CACHE_COUNT: usize = 4;
const CACHE_DIM: usize = 128;
const CACHE_ORDER: usize = 19;
const MAX_CHUNK_FRAMES: usize = 6_000;

pub struct FsmnVad {
    model: candle_onnx::onnx::ModelProto,
    frontend: FsmnVadFrontend,
    device: Device,
}

impl FsmnVad {
    /// Loads an FSMN model and its preprocessing sidecar for Candle inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or CMVN sidecar cannot be loaded.
    pub fn new(model_path: PathBuf, device: Device, _debug: bool) -> Result<Self> {
        let frontend = FsmnVadFrontend::from_model_path(&model_path)?;
        let model = candle_onnx::read_file(model_path)?;
        Ok(Self {
            model,
            frontend,
            device,
        })
    }

    /// Computes a speech probability for every FSMN frame.
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction, inference, or tensor validation fails.
    pub fn speech_probabilities(&mut self, waveform: &[f32]) -> Result<Vec<f32>> {
        let features = self.frontend.extract(waveform)?;
        if features.is_empty() {
            return Ok(Vec::new());
        }

        let cache_shape = (1, CACHE_DIM, CACHE_ORDER, 1);
        let empty_cache = Tensor::zeros(cache_shape, candle_core::DType::F32, &self.device)?;
        let mut caches: [Tensor; CACHE_COUNT] = std::array::from_fn(|_| empty_cache.clone());
        let mut probabilities = Vec::with_capacity(features.len() / FEATURE_DIM);

        for chunk in features.chunks(MAX_CHUNK_FRAMES * FEATURE_DIM) {
            let frame_count = chunk.len() / FEATURE_DIM;
            let speech = Tensor::from_slice(chunk, (1, frame_count, FEATURE_DIM), &self.device)?;
            debug!(
                "FSMN-VAD input: {:?}, dtype: {:?}",
                speech.shape(),
                speech.dtype()
            );
            let mut inputs = HashMap::from_iter([("speech".to_string(), speech)]);
            for (index, cache) in caches.iter().enumerate() {
                inputs.insert(format!("in_cache{index}"), cache.clone());
            }

            let outputs = candle_onnx::simple_eval(&self.model, inputs)
                .context("failed to evaluate FSMN-VAD; Candle supports the FP32 model only")?;
            let logits = outputs
                .get("logits")
                .context("FSMN-VAD model did not return a 'logits' tensor")?;
            let dimensions = logits.dims();
            anyhow::ensure!(
                dimensions.len() == 3 && dimensions[0] == 1 && dimensions[1] == frame_count,
                "FSMN-VAD returned logits with shape {dimensions:?}; expected [1, {frame_count}, classes]"
            );
            let class_count = dimensions[2];
            anyhow::ensure!(class_count >= 2, "FSMN-VAD returned fewer than two classes");
            let scores = logits.flatten_all()?.to_vec1::<f32>()?;
            anyhow::ensure!(
                scores.len() == frame_count * class_count,
                "FSMN-VAD returned an inconsistent logits tensor"
            );
            probabilities.extend(
                scores
                    .chunks_exact(class_count)
                    .map(|frame_scores| (1.0 - frame_scores[0]).clamp(0.0, 1.0)),
            );

            for (index, cache) in caches.iter_mut().enumerate() {
                let name = format!("out_cache{index}");
                let next_cache = outputs
                    .get(&name)
                    .with_context(|| format!("FSMN-VAD model did not return '{name}'"))?;
                anyhow::ensure!(
                    next_cache.dims() == [1, CACHE_DIM, CACHE_ORDER, 1],
                    "FSMN-VAD returned {name} with shape {:?}; expected [1, {CACHE_DIM}, {CACHE_ORDER}, 1]",
                    next_cache.dims()
                );
                cache.clone_from(next_cache);
            }
        }

        debug!(
            "FSMN-VAD produced {} frame probabilities",
            probabilities.len()
        );
        Ok(probabilities)
    }
}
