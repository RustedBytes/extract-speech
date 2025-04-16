use std::{collections::HashMap, path::PathBuf};

use candle_core::{DType, Tensor};
use log::debug;

use crate::utils;

#[derive(Debug)]
struct State {
    frame_size_samples: usize,
    state: *mut Tensor,
    context: *mut Tensor,
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
    pub fn new(
        vad_params: utils::VadParams,
        model_path: PathBuf,
        device: candle_core::Device,
    ) -> Result<Self, anyhow::Error> {
        let model: candle_onnx::onnx::ModelProto = candle_onnx::read_file(model_path)?;

        let sr_per_ms = vad_params.sample_rate / 1000;
        let frame_size_samples = vad_params.frame_size * sr_per_ms;

        let context_size: usize = if vad_params.sample_rate == 16_000 {
            64
        } else {
            32
        };

        let sample_rate = Tensor::new(vad_params.sample_rate as i64, &device)?;

        let init_state = Tensor::zeros((2, 1, 128), DType::F32, &device)?;
        let init_context = Tensor::zeros((1, context_size), DType::F32, &device)?;

        let state = State {
            frame_size_samples,
            state: Box::into_raw(Box::new(init_state)),
            context: Box::into_raw(Box::new(init_context)),
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

    pub fn reset(&mut self) {}

    pub fn probability(&mut self, audio_frame: Vec<f32>) -> Result<f32, anyhow::Error> {
        let state = &mut self.state;

        let next_data = audio_frame[state.frame_size_samples - self.context_size..].to_vec();
        let next_context = Tensor::from_vec(next_data, (1, self.context_size), &self.device)?;

        let context_tensor = unsafe { state.context.as_ref().unwrap() };
        let context_data = context_tensor.squeeze(0).unwrap().to_vec1::<f32>()?;
        let input = Tensor::from_vec(
            [context_data, audio_frame].concat(),
            (1, self.context_size + state.frame_size_samples),
            &self.device,
        )?;

        let inputs = HashMap::from_iter([
            ("input".to_string(), input),
            (
                "state".to_string(),
                unsafe { state.state.as_ref().unwrap() }.clone(),
            ),
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

        let out = candle_onnx::simple_eval(&self.model, inputs).unwrap();

        let output = &out["output"];
        let state_n = &out["stateN"];

        let output = output.flatten_all()?.to_vec1::<f32>()?;

        // assert_eq!(output.len(), 1);
        // assert_eq!(state_n.dims(), &[2, 1, 128]);

        unsafe {
            state.context.replace(next_context);
            state.state.replace(state_n.clone());
        };

        let prediction = output[0];

        drop(output);

        Ok(prediction)
    }
}
