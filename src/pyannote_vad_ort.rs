use std::path::PathBuf;

use log::debug;
use ndarray::{Array, Array2, ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};
use ort::value::Value;
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{builder::GraphOptimizationLevel, Session, SessionInputs},
};

use crate::utils;

#[derive(Debug)]
pub struct PyAnnote {
    vad_params: utils::VadParams,
    session: Session,
}

impl PyAnnote {
    pub fn new(
        vad_params: utils::VadParams,
        execution_providers: Vec<ExecutionProviderDispatch>,
        model_path: PathBuf,
    ) -> Result<Self, anyhow::Error> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_parallel_execution(false)?
            .with_execution_providers(execution_providers)?
            .commit_from_file(model_path)?;

        Ok(Self {
            vad_params,
            session,
        })
    }

    pub fn reset(&mut self) {
        // PyAnnote doesn't maintain state between calls
    }

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

        let logits = &outputs["logits"];
        let probabilities: ArrayD<f32> = logits.try_extract_tensor()?.to_owned();

        if self.vad_params.debug {
            debug!("PyAnnote output shape: {:?}", probabilities.shape());
        }

        Ok(probabilities)
    }
}
