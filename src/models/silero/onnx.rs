//! ONNX Runtime implementation of Silero VAD v5.

use std::path::PathBuf;

use anyhow::Context;
use log::debug;
use ndarray::{Array, Array2, ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};
use ort::value::Value;
use ort::{
    ep::ExecutionProviderDispatch,
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session, SessionInputs},
};

use crate::{utils, vad_iter::VadModel};

#[derive(Debug)]
pub struct Silero {
    vad_params: utils::VadParams,
    frame_size_samples: usize,
    session: Session,
    sample_rate: ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>,
    state: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    context: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
}

impl Silero {
    /// Loads a Silero ONNX Runtime session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session or its initial state cannot be created.
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

        let sample_rate = Array::from_vec(vec![
            i64::try_from(vad_params.sample_rate).context("sample rate exceeds i64")?
        ]);

        let state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        let context = ArrayD::<f32>::zeros([1, context_size].as_slice());

        Ok(Self {
            vad_params,
            frame_size_samples,
            session,
            sample_rate,
            state,
            context,
        })
    }
}

impl VadModel for Silero {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        self.context = ArrayD::<f32>::zeros([1, self.context.len()].as_slice());
        Ok(())
    }

    fn probability(&mut self, audio_frame: &[f32]) -> anyhow::Result<f32> {
        anyhow::ensure!(
            audio_frame.len() == self.frame_size_samples,
            "expected {} audio samples, received {}",
            self.frame_size_samples,
            audio_frame.len()
        );

        let next_data = audio_frame[self.frame_size_samples - self.context.len()..].to_vec();
        let next_context = Array2::<f32>::from_shape_vec([1, self.context.len()], next_data)?;

        let context_data = self.context.clone();

        let audio_frame_vec =
            Array2::<f32>::from_shape_vec([1, audio_frame.len()], audio_frame.to_vec())?.into_dyn();
        let input_data = ndarray::concatenate(
            ndarray::Axis(1),
            &[context_data.view(), audio_frame_vec.view()],
        )?
        .into_dyn();

        let values = ort::inputs![
            Value::from_array(input_data)?,
            Value::from_array(self.state.clone())?,
            Value::from_array(self.sample_rate.clone())?,
        ];

        if self.vad_params.debug {
            debug!(
                "input: {:?}, dtype: {:?}",
                values[0].shape(),
                values[0].dtype()
            );
            debug!(
                "state: {:?}, dtype: {:?}",
                values[1].shape(),
                values[1].dtype()
            );
            debug!(
                "sample_rate: {:?}, dtype: {:?}",
                values[2].shape(),
                values[2].dtype()
            );
        }

        let inputs = SessionInputs::ValueSlice::<3>(&values);
        let outputs = self.session.run(inputs)?;

        let (state_shape, state_data) = outputs
            .get("stateN")
            .context("Silero model did not return a 'stateN' tensor")?
            .try_extract_tensor::<f32>()
            .context("Silero 'stateN' output was not an f32 tensor")?;
        anyhow::ensure!(
            **state_shape == [2, 1, 128],
            "Silero returned state with shape {state_shape:?}; expected [2, 1, 128]"
        );
        self.state = ArrayD::from_shape_vec([2, 1, 128].as_slice(), state_data.to_vec())?;
        self.context = next_context.into_dyn();

        let (output_shape, output_data) = outputs
            .get("output")
            .context("Silero model did not return an 'output' tensor")?
            .try_extract_tensor::<f32>()
            .context("Silero 'output' was not an f32 tensor")?;
        anyhow::ensure!(
            **output_shape == [1, 1],
            "Silero returned output with shape {output_shape:?}; expected [1, 1]"
        );
        let prediction = *output_data
            .first()
            .context("Silero model returned an empty output tensor")?;

        Ok(prediction)
    }
}
