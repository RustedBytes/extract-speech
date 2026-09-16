//! ONNX Runtime implementation of `PulseVAD`.

use std::path::PathBuf;

use anyhow::Context;
use log::debug;
use ndarray::Array;
use ort::value::Value;
use ort::{
    ep::ExecutionProviderDispatch,
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session, SessionInputs},
};

use crate::{
    pulsevad_frontend::{PulseVadFrontend, N_FRAMES, N_MELS},
    vad_iter::VadModel,
};

pub struct PulseVad {
    session: Session,
    frontend: PulseVadFrontend,
    debug: bool,
}

impl PulseVad {
    /// Loads a `PulseVAD` ONNX Runtime session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be configured or the model cannot be loaded.
    pub fn new(
        execution_providers: Vec<ExecutionProviderDispatch>,
        model_path: PathBuf,
        debug: bool,
    ) -> anyhow::Result<Self> {
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
            frontend: PulseVadFrontend::new(),
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
        let input = Array::from_shape_vec([1, N_MELS, N_FRAMES], features)?;
        let values = ort::inputs![Value::from_array(input)?];

        if self.debug {
            debug!(
                "PulseVAD input: {:?}, dtype: {:?}",
                values[0].shape(),
                values[0].dtype()
            );
        }

        let outputs = self.session.run(SessionInputs::ValueSlice::<1>(&values))?;
        let logits = outputs
            .get("logits")
            .context("PulseVAD model did not return a 'logits' tensor")?
            .try_extract_tensor::<f32>()?
            .1;
        anyhow::ensure!(
            logits.len() >= 2,
            "PulseVAD returned {} logits; expected at least 2",
            logits.len()
        );

        let difference = logits[1] - logits[0];
        let probability = if difference >= 0.0 {
            1.0 / (1.0 + (-difference).exp())
        } else {
            let exponential = difference.exp();
            exponential / (1.0 + exponential)
        };
        if self.debug {
            debug!("PulseVAD speech probability: {probability:.6}");
        }
        Ok(probability)
    }
}
