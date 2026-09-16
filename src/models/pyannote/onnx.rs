//! ONNX Runtime implementation of `PyAnnote` segmentation.

use std::path::PathBuf;

use anyhow::Context;
use log::debug;
use ndarray::{Array, ArrayD};
use ort::value::Value;
use ort::{
    ep::ExecutionProviderDispatch,
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session, SessionInputs},
};

use crate::utils;

#[derive(Debug)]
pub struct PyAnnote {
    vad_params: utils::VadParams,
    session: Session,
}

impl PyAnnote {
    /// Loads a `PyAnnote` ONNX Runtime session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be configured or the model cannot be loaded.
    pub fn new(
        vad_params: utils::VadParams,
        execution_providers: Vec<ExecutionProviderDispatch>,
        model_path: PathBuf,
    ) -> Result<Self, anyhow::Error> {
        let builder =
            |result: ort::session::builder::BuilderResult| -> anyhow::Result<SessionBuilder> {
                result.map_err(|error| anyhow::anyhow!(error.to_string()))
            };
        let session_builder = Session::builder()?;
        let session_builder =
            builder(session_builder.with_optimization_level(GraphOptimizationLevel::Level3))?;
        let session_builder = builder(session_builder.with_intra_threads(1))?;
        let session_builder = builder(session_builder.with_inter_threads(1))?;
        let session_builder = builder(session_builder.with_parallel_execution(false))?;
        let mut session_builder =
            builder(session_builder.with_execution_providers(execution_providers))?;
        let session = session_builder.commit_from_file(model_path)?;

        Ok(Self {
            vad_params,
            session,
        })
    }

    pub fn reset(&mut self) {
        // PyAnnote doesn't maintain state between calls
    }

    /// Computes frame-level segmentation logits.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails or the output tensor is invalid.
    pub fn get_frame_probabilities(
        &mut self,
        audio_samples: &[f32],
    ) -> Result<ArrayD<f32>, anyhow::Error> {
        // PyAnnote expects input shape: [batch_size, num_channels, samples]
        let input_shape = [1, 1, audio_samples.len()];
        let input_tensor = Array::from_shape_vec(input_shape, audio_samples.to_vec())?;

        let values = ort::inputs![Value::from_array(input_tensor.into_dyn())?];

        if self.vad_params.debug {
            debug!(
                "PyAnnote input: {:?}, dtype: {:?}",
                values[0].shape(),
                values[0].dtype()
            );
        }

        let inputs = SessionInputs::ValueSlice::<1>(&values);
        let outputs = self.session.run(inputs)?;

        let logits = outputs
            .get("logits")
            .context("PyAnnote model did not return a 'logits' tensor")?;
        let (shape, data) = logits.try_extract_tensor()?.to_owned();

        if self.vad_params.debug {
            debug!("PyAnnote output shape: {shape:?}");
        }

        let shape_usize: Vec<usize> = shape
            .iter()
            .map(|&dimension| usize::try_from(dimension).context("negative tensor dimension"))
            .collect::<anyhow::Result<_>>()?;
        let probabilities = Array::from_shape_vec(shape_usize, data.to_vec())?;

        Ok(probabilities)
    }
}
