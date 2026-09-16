//! Candle implementation of MarbleNet VAD.

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use candle_core::Tensor;
use log::debug;

use crate::marblenet_frontend::{speech_probability, MarbleNetFrontend, N_MELS};

pub struct MarbleNet {
    model: candle_onnx::onnx::ModelProto,
    frontend: MarbleNetFrontend,
    device: candle_core::Device,
    debug: bool,
}

impl MarbleNet {
    pub fn new(model_path: PathBuf, device: candle_core::Device, debug: bool) -> Result<Self> {
        let model = candle_onnx::read_file(model_path)?;
        Ok(Self {
            model,
            frontend: MarbleNetFrontend::new(),
            device,
            debug,
        })
    }

    pub fn speech_probabilities(&self, waveform: &[f32]) -> Result<Vec<f32>> {
        let (features, feature_frames) = self.frontend.extract(waveform)?;
        if feature_frames == 0 {
            return Ok(Vec::new());
        }

        let input = Tensor::from_vec(features, (1, N_MELS, feature_frames), &self.device)?;
        if self.debug {
            debug!(
                "MarbleNet input: {:?}, dtype: {:?}",
                input.shape(),
                input.dtype()
            );
        }
        let outputs = candle_onnx::simple_eval(
            &self.model,
            HashMap::from_iter([("audio_signal".to_string(), input)]),
        )
        .context("failed to evaluate MarbleNet; Candle supports the FP32 model only")?;
        let output = outputs
            .get("outputs")
            .context("MarbleNet model did not return an 'outputs' tensor")?;
        let dimensions = output.dims();
        anyhow::ensure!(
            dimensions.len() == 3 && dimensions[0] == 1 && dimensions[2] >= 2,
            "MarbleNet returned logits with shape {dimensions:?}; expected [1, frames, classes]"
        );
        let class_count = dimensions[2];
        let logits = output.flatten_all()?.to_vec1::<f32>()?;
        let probabilities = logits
            .chunks_exact(class_count)
            .map(speech_probability)
            .collect::<Result<Vec<_>>>()?;

        if self.debug {
            debug!(
                "MarbleNet produced {} frame probabilities",
                probabilities.len()
            );
        }
        Ok(probabilities)
    }
}
