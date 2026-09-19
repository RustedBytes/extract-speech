//! Candle implementation of TEN VAD.

use std::{collections::HashMap, path::PathBuf};

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use log::debug;

use crate::{models::ten::frontend, vad_iter::VadModel};

const INPUT_NAMES: [&str; 5] = ["input_1", "input_2", "input_3", "input_6", "input_7"];
const OUTPUT_NAMES: [&str; 5] = ["output_1", "output_2", "output_3", "output_6", "output_7"];

/// Number of 16 kHz samples consumed by each inference call.
pub const FRAME_SAMPLES: usize = frontend::FRAME_SAMPLES;

pub struct TenVad {
    model: candle_onnx::onnx::ModelProto,
    frontend: frontend::TenVadFrontend,
    hidden: [Tensor; frontend::HIDDEN_COUNT],
    device: Device,
}

impl TenVad {
    /// Loads a TEN VAD model for Candle inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX model or initial state tensors cannot be loaded.
    pub fn new(model_path: PathBuf, device: Device, _debug: bool) -> anyhow::Result<Self> {
        let mut model = candle_onnx::read_file(model_path)?;
        preserve_lstm_sequence_axis(&mut model)?;
        let empty = Tensor::zeros((1, frontend::HIDDEN_DIM), DType::F32, &device)?;
        let hidden = std::array::from_fn(|_| empty.clone());
        Ok(Self {
            model,
            frontend: frontend::TenVadFrontend::new(),
            hidden,
            device,
        })
    }
}

fn preserve_lstm_sequence_axis(model: &mut candle_onnx::onnx::ModelProto) -> anyhow::Result<()> {
    const NODE_NAME: &str = "Squeeze__104";
    const SHAPE_NAME: &str = "new_shape__176";

    let graph = model.graph.as_mut().context("TEN VAD model has no graph")?;
    anyhow::ensure!(
        graph
            .initializer
            .iter()
            .any(|initializer| initializer.name == SHAPE_NAME),
        "TEN VAD model is missing its LSTM output shape"
    );
    let node = graph
        .node
        .iter_mut()
        .find(|node| node.name == NODE_NAME)
        .context("TEN VAD model is missing its second LSTM output reshape")?;
    anyhow::ensure!(
        node.op_type == "Squeeze"
            && node.input.len() == 1
            && node
                .attribute
                .iter()
                .any(|attribute| attribute.name == "axes" && attribute.ints == [1]),
        "TEN VAD second LSTM output has an unexpected graph structure"
    );

    // This model uses the pre-opset-13 `axes` attribute. Candle's ONNX
    // evaluator currently ignores that form and removes both singleton axes,
    // losing the sequence dimension required by the following transpose.
    // The model already contains the equivalent [1, -1, 64] reshape shape.
    node.op_type = "Reshape".to_string();
    node.input.push(SHAPE_NAME.to_string());
    node.attribute.clear();
    Ok(())
}

impl VadModel for TenVad {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.frontend.reset();
        let empty = Tensor::zeros((1, frontend::HIDDEN_DIM), DType::F32, &self.device)?;
        self.hidden = std::array::from_fn(|_| empty.clone());
        Ok(())
    }

    fn probability(&mut self, audio_frame: &[f32]) -> anyhow::Result<f32> {
        let features = self.frontend.extract(audio_frame)?;
        let features = Tensor::from_vec(
            features,
            (1, frontend::CONTEXT_FRAMES, frontend::FEATURE_DIM),
            &self.device,
        )?;
        let mut inputs = HashMap::from_iter([(INPUT_NAMES[0].to_string(), features)]);
        for (name, hidden) in INPUT_NAMES[1..].iter().zip(&self.hidden) {
            inputs.insert((*name).to_string(), hidden.clone());
        }
        let outputs = candle_onnx::simple_eval(&self.model, inputs)
            .context("failed to evaluate TEN VAD with Candle")?;
        let score = outputs
            .get(OUTPUT_NAMES[0])
            .context("TEN VAD model did not return 'output_1'")?;
        anyhow::ensure!(
            score.dims() == [1, 1, 1],
            "TEN VAD returned output_1 with shape {:?}; expected [1, 1, 1]",
            score.dims()
        );
        let score = score.flatten_all()?.to_vec1::<f32>()?;
        for (index, hidden) in self.hidden.iter_mut().enumerate() {
            let name = OUTPUT_NAMES[index + 1];
            let next = outputs
                .get(name)
                .with_context(|| format!("TEN VAD model did not return '{name}'"))?;
            anyhow::ensure!(
                next.dims() == [1, frontend::HIDDEN_DIM],
                "TEN VAD returned {name} with shape {:?}; expected [1, {}]",
                next.dims(),
                frontend::HIDDEN_DIM
            );
            hidden.clone_from(next);
        }
        anyhow::ensure!(score[0].is_finite(), "TEN VAD returned a non-finite score");
        let probability = score[0].clamp(0.0, 1.0);
        debug!("TEN VAD speech probability: {probability:.6}");
        Ok(probability)
    }
}
