//! Candle implementation of Silero VAD v5.

use std::{collections::HashMap, path::PathBuf};

use anyhow::Context;
use candle_core::{DType, Tensor};
use log::debug;

use crate::{utils, vad_iter::VadModel};

#[derive(Debug)]
struct State {
    frame_size_samples: usize,
    recurrent: Tensor,
    context: Tensor,
}

#[derive(Debug)]
pub struct Silero {
    vad_params: utils::VadParams,
    model: candle_onnx::onnx::ModelProto,
    sample_rate: Tensor,
    context_size: usize,
    state: State,
    device: candle_core::Device,
}

impl Silero {
    /// Loads a Silero model for Candle inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or its initial tensors cannot be created.
    pub fn new(
        vad_params: utils::VadParams,
        model_path: PathBuf,
        device: candle_core::Device,
    ) -> Result<Self, anyhow::Error> {
        let model: candle_onnx::onnx::ModelProto = candle_onnx::read_file(model_path)?;

        let sr_per_ms = vad_params.sample_rate / 1000;
        let frame_size_samples = vad_params
            .frame_size
            .checked_mul(sr_per_ms)
            .context("Silero frame size exceeds usize")?;

        let context_size: usize = if vad_params.sample_rate == 16_000 {
            64
        } else {
            32
        };
        anyhow::ensure!(
            frame_size_samples >= context_size,
            "Silero frame must contain at least {context_size} samples"
        );

        let sample_rate = Tensor::new(
            i64::try_from(vad_params.sample_rate).context("sample rate exceeds i64")?,
            &device,
        )?;

        let state = State {
            frame_size_samples,
            recurrent: Tensor::zeros((2, 1, 128), DType::F32, &device)?,
            context: Tensor::zeros((1, context_size), DType::F32, &device)?,
        };

        Ok(Self {
            vad_params,
            model,
            sample_rate,
            context_size,
            state,
            device,
        })
    }
}

impl VadModel for Silero {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.state.recurrent = Tensor::zeros((2, 1, 128), DType::F32, &self.device)?;
        self.state.context = Tensor::zeros((1, self.context_size), DType::F32, &self.device)?;
        Ok(())
    }

    fn probability(&mut self, audio_frame: &[f32]) -> anyhow::Result<f32> {
        anyhow::ensure!(
            audio_frame.len() == self.state.frame_size_samples,
            "expected {} audio samples, received {}",
            self.state.frame_size_samples,
            audio_frame.len()
        );

        let next_data = audio_frame[self.state.frame_size_samples - self.context_size..].to_vec();
        let next_context = Tensor::from_vec(next_data, (1, self.context_size), &self.device)?;

        let context_data = self.state.context.squeeze(0)?.to_vec1::<f32>()?;
        let mut input_data = Vec::with_capacity(self.context_size + audio_frame.len());
        input_data.extend_from_slice(&context_data);
        input_data.extend_from_slice(audio_frame);
        let input = Tensor::from_vec(
            input_data,
            (1, self.context_size + audio_frame.len()),
            &self.device,
        )?;

        let inputs = HashMap::from_iter([
            ("input".to_string(), input),
            ("state".to_string(), self.state.recurrent.clone()),
            ("sr".to_string(), self.sample_rate.clone()),
        ]);

        if self.vad_params.debug {
            for (k, v) in &inputs {
                debug!(
                    "{} - {:?}: dtype: {:?}, len: {:?}",
                    k,
                    v.shape(),
                    v.dtype(),
                    v.elem_count()
                );
            }
        }

        let outputs = candle_onnx::simple_eval(&self.model, inputs)?;

        let output = outputs
            .get("output")
            .context("Silero model did not return an 'output' tensor")?;
        let recurrent = outputs
            .get("stateN")
            .context("Silero model did not return a 'stateN' tensor")?;
        anyhow::ensure!(
            output.dims() == [1, 1],
            "Silero returned output with shape {:?}; expected [1, 1]",
            output.dims()
        );
        anyhow::ensure!(
            recurrent.dims() == [2, 1, 128],
            "Silero returned state with shape {:?}; expected [2, 1, 128]",
            recurrent.dims()
        );

        let output = output.flatten_all()?.to_vec1::<f32>()?;
        let prediction = output
            .first()
            .copied()
            .context("Silero model returned an empty output tensor")?;

        self.state.context = next_context;
        self.state.recurrent = recurrent.clone();

        Ok(prediction)
    }
}
