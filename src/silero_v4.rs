use std::{collections::HashMap, path::PathBuf};

use candle_core::{DType, Tensor};

use crate::utils;

#[derive(Debug)]
struct State {
    h: Tensor,
    c: Tensor,
}

#[derive(Debug)]
pub struct Silero {
    model: candle_onnx::onnx::ModelProto,
    sample_rate: Tensor,
    state: State,
    chunk_size: usize,
    device: candle_core::Device,
}

impl Silero {
    pub fn new(
        vad_params: utils::VadParams,
        model_path: PathBuf,
        device: candle_core::Device,
    ) -> Result<Self, anyhow::Error> {
        let model: candle_onnx::onnx::ModelProto = candle_onnx::read_file(model_path)?;

        // Model state
        let state = State {
            h: Tensor::zeros((2, 1, 64), DType::F32, &device)?,
            c: Tensor::zeros((2, 1, 64), DType::F32, &device)?,
        };

        let sr_per_ms = vad_params.sample_rate / 1000;
        let chunk_size = vad_params.frame_size * sr_per_ms;
        let sample_rate = Tensor::new(vad_params.sample_rate as i64, &device)?;

        Ok(Self {
            model,
            sample_rate,
            state,
            chunk_size,
            device,
        })
    }

    pub fn reset(&mut self) {
        self.state = State {
            h: Tensor::zeros((2, 1, 64), DType::F32, &self.device).unwrap(),
            c: Tensor::zeros((2, 1, 64), DType::F32, &self.device).unwrap(),
        };
    }

    pub fn probability(&mut self, audio_frame: Vec<f32>) -> Result<f32, anyhow::Error> {
        let chunk = Tensor::from_vec(audio_frame, (1, self.chunk_size), &self.device)?;

        let inputs = HashMap::from_iter([
            ("input".to_string(), chunk.clone()),
            ("sr".to_string(), self.sample_rate.clone()),
            ("h".to_string(), self.state.h.clone()),
            ("c".to_string(), self.state.c.clone()),
        ]);

        for (k, v) in &inputs {
            println!(
                "{} - {:?}: dtype: {:?}, len: {:?}",
                k,
                v.shape(),
                v.dtype(),
                v.elem_count()
            );
        }

        let outputs = candle_onnx::simple_eval(&self.model, inputs).unwrap();

        let output = outputs["output"].clone();
        let hn = outputs["hn"].clone();
        let cn = outputs["cn"].clone();

        let output = output.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(output.len(), 1);
        assert_eq!(hn.dims(), &[2, 1, 64]);
        assert_eq!(cn.dims(), &[2, 1, 64]);

        // Add new state
        self.state = State { h: hn, c: cn };

        Ok(output[0])
    }
}
