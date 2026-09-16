//! Candle implementation of `PulseVAD`.

use std::{collections::HashMap, path::PathBuf};

use anyhow::Context;
use candle_core::Tensor;
use log::debug;

use crate::{
    pulsevad_frontend::{PulseVadFrontend, N_FRAMES, N_MELS},
    vad_iter::VadModel,
};

pub struct PulseVad {
    model: candle_onnx::onnx::ModelProto,
    frontend: PulseVadFrontend,
    device: candle_core::Device,
    debug: bool,
}

impl PulseVad {
    /// Loads a `PulseVAD` model for Candle inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX model cannot be read.
    pub fn new(
        model_path: PathBuf,
        device: candle_core::Device,
        debug: bool,
    ) -> anyhow::Result<Self> {
        let model = candle_onnx::read_file(model_path)?;
        Ok(Self {
            model,
            frontend: PulseVadFrontend::new(),
            device,
            debug,
        })
    }
}

impl VadModel for PulseVad {
    fn reset(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    fn probability(&mut self, audio_frame: &[f32]) -> anyhow::Result<f32> {
        let features = self.frontend.extract(audio_frame)?;
        let input = Tensor::from_vec(features, (1, N_MELS, N_FRAMES), &self.device)?;
        let outputs = candle_onnx::simple_eval(
            &self.model,
            HashMap::from_iter([("log_mel".to_string(), input)]),
        )
        .context("failed to evaluate PulseVAD; Candle supports the FP32 PulseVAD model only")?;
        let logits = outputs
            .get("logits")
            .context("PulseVAD model did not return a 'logits' tensor")?;
        anyhow::ensure!(
            matches!(logits.dims(), [2] | [1, 2]),
            "PulseVAD returned logits with shape {:?}; expected [2] or [1, 2]",
            logits.dims()
        );
        let logits = logits.flatten_all()?.to_vec1::<f32>()?;

        let probability = sigmoid(logits[1] - logits[0]);
        if self.debug {
            debug!("PulseVAD speech probability: {probability:.6}");
        }
        Ok(probability)
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Sigmoid saturation and midpoint are exact expectations.
mod tests {
    use super::*;

    #[test]
    fn sigmoid_is_stable_for_large_logits() {
        assert_eq!(sigmoid(1_000.0), 1.0);
        assert_eq!(sigmoid(-1_000.0), 0.0);
        assert_eq!(sigmoid(0.0), 0.5);
    }
}
