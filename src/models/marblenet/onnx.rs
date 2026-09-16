//! ONNX Runtime implementation of MarbleNet VAD.

use std::path::PathBuf;

use anyhow::{Context, Result};
use log::debug;
use ndarray::Array;
use ort::value::Value;
use ort::{
    ep::ExecutionProviderDispatch,
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session, SessionInputs},
};

use crate::marblenet_frontend::{speech_probability, MarbleNetFrontend, N_MELS};

pub struct MarbleNet {
    session: Session,
    frontend: MarbleNetFrontend,
    debug: bool,
}

impl MarbleNet {
    pub fn new(
        execution_providers: Vec<ExecutionProviderDispatch>,
        model_path: PathBuf,
        debug: bool,
    ) -> Result<Self> {
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
            session,
            frontend: MarbleNetFrontend::new(),
            debug,
        })
    }

    pub fn speech_probabilities(&mut self, waveform: &[f32]) -> Result<Vec<f32>> {
        let (features, feature_frames) = self.frontend.extract(waveform)?;
        if feature_frames == 0 {
            return Ok(Vec::new());
        }

        let input = Array::from_shape_vec([1, N_MELS, feature_frames], features)?;
        let values = ort::inputs![Value::from_array(input)?];
        if self.debug {
            debug!(
                "MarbleNet input: {:?}, dtype: {:?}",
                values[0].shape(),
                values[0].dtype()
            );
        }

        let outputs = self.session.run(SessionInputs::ValueSlice::<1>(&values))?;
        let (shape, logits) = outputs
            .get("outputs")
            .context("MarbleNet model did not return an 'outputs' tensor")?
            .try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.len() == 3 && shape[0] == 1 && shape[2] >= 2,
            "MarbleNet returned logits with shape {shape:?}; expected [1, frames, classes]"
        );
        let frame_count = usize::try_from(shape[1]).context("invalid MarbleNet frame count")?;
        let class_count = usize::try_from(shape[2]).context("invalid MarbleNet class count")?;
        anyhow::ensure!(
            logits.len() == frame_count * class_count,
            "MarbleNet returned an inconsistent logits tensor"
        );
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
