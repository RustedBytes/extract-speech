//! ONNX Runtime implementation of TEN VAD.

use std::path::Path;

use anyhow::Context;
use log::debug;
use ndarray::{Array, Array2};
use ort::{
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session},
    value::Value,
};

use super::frontend::{self, TenVadFrontend};
use crate::vad_iter::VadModel;

const OUTPUT_NAMES: [&str; 5] = ["output_1", "output_2", "output_3", "output_6", "output_7"];

/// Number of 16 kHz samples consumed by each inference call.
pub const FRAME_SAMPLES: usize = frontend::FRAME_SAMPLES;

/// TEN VAD backed by ONNX Runtime.
pub struct TenVad {
    session: Session,
    frontend: TenVadFrontend,
    hidden: [Array2<f32>; frontend::HIDDEN_COUNT],
}

impl TenVad {
    /// Loads a TEN VAD model.
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX model cannot be loaded.
    pub fn new(model_path: impl AsRef<Path>, _debug: bool) -> anyhow::Result<Self> {
        let builder =
            |result: ort::session::builder::BuilderResult| -> anyhow::Result<SessionBuilder> {
                result.map_err(|error| anyhow::anyhow!(error.to_string()))
            };
        let session_builder = Session::builder()?;
        let session_builder =
            builder(session_builder.with_optimization_level(GraphOptimizationLevel::Level3))?;
        let session_builder = builder(session_builder.with_intra_threads(1))?;
        let mut session_builder = builder(session_builder.with_inter_threads(1))?;
        let session = session_builder
            .commit_from_file(model_path)
            .context("failed to load TEN VAD ONNX model")?;

        Ok(Self {
            session,
            frontend: TenVadFrontend::new(),
            hidden: std::array::from_fn(|_| Array2::zeros((1, frontend::HIDDEN_DIM))),
        })
    }
}

impl VadModel for TenVad {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.frontend.reset();
        for hidden in &mut self.hidden {
            hidden.fill(0.0);
        }
        Ok(())
    }

    fn probability(&mut self, audio_frame: &[f32]) -> anyhow::Result<f32> {
        let features = Array::from_shape_vec(
            [1, frontend::CONTEXT_FRAMES, frontend::FEATURE_DIM],
            self.frontend.extract(audio_frame)?,
        )
        .context("failed to shape TEN VAD features")?;

        let outputs = self.session.run(ort::inputs![
            Value::from_array(features)?,
            Value::from_array(self.hidden[0].clone())?,
            Value::from_array(self.hidden[1].clone())?,
            Value::from_array(self.hidden[2].clone())?,
            Value::from_array(self.hidden[3].clone())?,
        ])?;

        let score = outputs
            .get(OUTPUT_NAMES[0])
            .context("TEN VAD model did not return 'output_1'")?
            .try_extract_array::<f32>()
            .context("TEN VAD score output is not an f32 tensor")?;
        anyhow::ensure!(
            score.shape() == [1, 1, 1],
            "TEN VAD score output must have shape [1, 1, 1], received {:?}",
            score.shape()
        );
        let probability = score
            .iter()
            .copied()
            .next()
            .context("TEN VAD score output is empty")?;

        for (index, name) in OUTPUT_NAMES[1..].iter().enumerate() {
            let hidden = outputs
                .get(*name)
                .with_context(|| format!("TEN VAD model did not return '{name}'"))?
                .try_extract_array::<f32>()
                .with_context(|| format!("TEN VAD hidden output {name} is not an f32 tensor"))?;
            anyhow::ensure!(
                hidden.shape() == [1, frontend::HIDDEN_DIM],
                "TEN VAD hidden output {name} must have shape [1, {}], received {:?}",
                frontend::HIDDEN_DIM,
                hidden.shape()
            );
            self.hidden[index].assign(&hidden);
        }

        anyhow::ensure!(
            probability.is_finite(),
            "TEN VAD returned a non-finite probability"
        );
        let probability = probability.clamp(0.0, 1.0);
        debug!("TEN VAD speech probability: {probability:.6}");
        Ok(probability)
    }
}
