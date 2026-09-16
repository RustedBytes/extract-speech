//! ONNX Runtime implementation of FSMN-VAD.

use std::path::PathBuf;

use anyhow::{Context, Result};
use log::debug;
use ndarray::{Array, ArrayD};
use ort::value::Value;
use ort::{
    ep::ExecutionProviderDispatch,
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session, SessionInputs},
};

use crate::fsmn_vad_frontend::{FsmnVadFrontend, FEATURE_DIM};

const CACHE_COUNT: usize = 4;
const CACHE_DIM: usize = 128;
const CACHE_ORDER: usize = 19;
const MAX_CHUNK_FRAMES: usize = 6_000;

pub struct FsmnVad {
    session: Session,
    frontend: FsmnVadFrontend,
}

impl FsmnVad {
    /// Loads an FSMN model and its preprocessing sidecars.
    ///
    /// # Errors
    ///
    /// Returns an error if the model, sidecars, or ONNX session cannot be initialized.
    pub fn new(
        execution_providers: Vec<ExecutionProviderDispatch>,
        model_path: PathBuf,
        _debug: bool,
    ) -> Result<Self> {
        let frontend = FsmnVadFrontend::from_model_path(&model_path)?;
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

        Ok(Self { session, frontend })
    }

    /// Computes a speech probability for every FSMN frame.
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction, inference, or tensor validation fails.
    pub fn speech_probabilities(&mut self, waveform: &[f32]) -> Result<Vec<f32>> {
        let features = self.frontend.extract(waveform)?;
        if features.is_empty() {
            return Ok(Vec::new());
        }

        let mut caches: [ArrayD<f32>; CACHE_COUNT] =
            std::array::from_fn(|_| ArrayD::zeros([1, CACHE_DIM, CACHE_ORDER, 1].as_slice()));
        let mut probabilities = Vec::with_capacity(features.len() / FEATURE_DIM);

        for chunk in features.chunks(MAX_CHUNK_FRAMES * FEATURE_DIM) {
            let frame_count = chunk.len() / FEATURE_DIM;
            let speech = Array::from_shape_vec([1, frame_count, FEATURE_DIM], chunk.to_vec())?;
            let values = ort::inputs![
                Value::from_array(speech)?,
                Value::from_array(caches[0].clone())?,
                Value::from_array(caches[1].clone())?,
                Value::from_array(caches[2].clone())?,
                Value::from_array(caches[3].clone())?,
            ];

            if log::log_enabled!(log::Level::Debug) {
                debug!(
                    "FSMN-VAD input: {:?}, dtype: {:?}",
                    values[0].shape(),
                    values[0].dtype()
                );
            }

            let outputs = self.session.run(SessionInputs::ValueSlice::<5>(&values))?;
            let (shape, scores) = outputs
                .get("logits")
                .context("FSMN-VAD model did not return a 'logits' tensor")?
                .try_extract_tensor::<f32>()?;
            let expected_frame_count =
                i64::try_from(frame_count).context("FSMN frame count exceeds i64")?;
            anyhow::ensure!(
                shape.len() == 3 && shape[0] == 1 && shape[1] == expected_frame_count,
                "FSMN-VAD returned logits with shape {shape:?}; expected [1, {frame_count}, classes]"
            );
            let class_count = usize::try_from(shape[2]).context("invalid FSMN-VAD class count")?;
            anyhow::ensure!(class_count >= 2, "FSMN-VAD returned fewer than two classes");
            anyhow::ensure!(
                scores.len() == frame_count * class_count,
                "FSMN-VAD returned an inconsistent logits tensor"
            );
            probabilities.extend(
                scores
                    .chunks_exact(class_count)
                    // FunASR reserves class zero for silence; all remaining
                    // monophone classes represent speech.
                    .map(|frame_scores| (1.0 - frame_scores[0]).clamp(0.0, 1.0)),
            );

            for (index, cache) in caches.iter_mut().enumerate() {
                let name = format!("out_cache{index}");
                let next_cache = outputs
                    .get(&name)
                    .with_context(|| format!("FSMN-VAD model did not return '{name}'"))?
                    .try_extract_array::<f32>()?
                    .to_owned();
                anyhow::ensure!(
                    next_cache.shape() == [1, CACHE_DIM, CACHE_ORDER, 1],
                    "FSMN-VAD returned {name} with shape {:?}; expected [1, {CACHE_DIM}, {CACHE_ORDER}, 1]",
                    next_cache.shape()
                );
                *cache = next_cache;
            }
        }

        debug!(
            "FSMN-VAD produced {} frame probabilities",
            probabilities.len()
        );
        Ok(probabilities)
    }
}
