use std::path::PathBuf;

use log::debug;
use ndarray::{Array, Array2, ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{builder::GraphOptimizationLevel, Session, SessionInputs},
};

use crate::utils;

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

        let sr_per_ms = vad_params.sample_rate / 1000;
        let frame_size_samples = vad_params.frame_size * sr_per_ms;

        let context_size: usize = if vad_params.sample_rate == 16_000 {
            64
        } else {
            32
        };

        let sample_rate = Array::from_shape_vec([1], vec![vad_params.sample_rate as i64]).unwrap();

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

    pub fn reset(&mut self) {
        self.state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        self.context = ArrayD::<f32>::zeros([1, self.context.len()].as_slice());
    }

    pub fn probability(&mut self, audio_frame: Vec<f32>) -> Result<f32, anyhow::Error> {
        let next_data = audio_frame[self.frame_size_samples - self.context.len()..].to_vec();
        let next_context =
            Array2::<f32>::from_shape_vec([1, self.context.len()], next_data).unwrap();

        let context_data = self.context.clone();

        let audio_frame_vec = Array2::<f32>::from_shape_vec([1, audio_frame.len()], audio_frame)
            .unwrap()
            .into_dyn();
        let input_data = ndarray::concatenate(
            ndarray::Axis(1),
            &[context_data.view(), audio_frame_vec.view()],
        )?
        .into_dyn();

        let values = ort::inputs![
            input_data,
            std::mem::take(&mut self.state),
            self.sample_rate.clone(),
        ]?;

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

        self.state = outputs["stateN"].try_extract_tensor().unwrap().to_owned();
        self.context = next_context.into_dyn();

        let prediction = *outputs["output"]
            .try_extract_raw_tensor::<f32>()
            .unwrap()
            .1
            .first()
            .unwrap();

        Ok(prediction)
    }
}
