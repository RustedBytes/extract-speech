//! PyAnnote-specific streaming iterator.

use log::debug;
use ndarray::{ArrayView1, Axis, Ix3};

use crate::{pyannote_vad_ort::PyAnnote, utils, vad_iter};

const FRAME_STEP_SAMPLES: usize = 270;

#[derive(Debug)]
pub struct PyAnnoteVadIter {
    pyannote: PyAnnote,
    params: utils::VadParams,
    speeches: Vec<utils::TimeStamp>,
}

impl PyAnnoteVadIter {
    #[must_use]
    pub fn new(pyannote: PyAnnote, params: utils::VadParams) -> Self {
        if params.debug {
            debug!("PyAnnote vad_params: {params:?}");
        }

        Self {
            pyannote,
            params,
            speeches: Vec::new(),
        }
    }

    /// Detects speech segments in one complete waveform.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid parameters or samples, or when inference
    /// or logits post-processing fails.
    pub fn process(&mut self, samples: &[f32]) -> Result<&[utils::TimeStamp], anyhow::Error> {
        vad_iter::validate_parameters(&self.params)?;
        vad_iter::validate_samples(samples)?;
        self.reset_states();

        // Get frame probabilities from PyAnnote model
        let logits = self.pyannote.get_frame_probabilities(samples)?;

        // Convert logits to speech timestamps
        self.speeches = self.post_process_vad(&logits, samples.len())?;

        Ok(&self.speeches)
    }

    fn reset_states(&mut self) {
        self.pyannote.reset();
        self.speeches.clear();
    }

    fn post_process_vad(
        &self,
        logits_tensor: &ndarray::ArrayD<f32>,
        num_samples: usize,
    ) -> Result<Vec<utils::TimeStamp>, anyhow::Error> {
        let logits_3d = logits_tensor.view().into_dimensionality::<Ix3>()?;
        let shape = logits_3d.shape();
        anyhow::ensure!(
            shape[0] == 1 && shape[2] >= 2,
            "PyAnnote returned logits with shape {shape:?}; expected [1, frames, classes >= 2]"
        );

        let batch = logits_3d.index_axis(Axis(0), 0);
        let probabilities = batch
            .axis_iter(Axis(0))
            .map(|frame| {
                let classes = softmax(&frame)?;
                Ok((1.0 - classes[0]).clamp(0.0, 1.0))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        vad_iter::segment_probabilities_with_frame_size(
            &probabilities,
            num_samples,
            &self.params,
            FRAME_STEP_SAMPLES,
        )
    }
}

fn softmax(x: &ArrayView1<'_, f32>) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(!x.is_empty(), "PyAnnote returned an empty class dimension");
    anyhow::ensure!(
        x.iter().all(|value| value.is_finite()),
        "PyAnnote returned non-finite logits"
    );
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut softmax_array: Vec<f32> = x.iter().map(|value| (value - max).exp()).collect();

    let sum: f32 = softmax_array.iter().sum();
    anyhow::ensure!(sum.is_finite() && sum > 0.0, "PyAnnote softmax is invalid");

    for value in &mut softmax_array {
        *value /= sum;
    }

    Ok(softmax_array)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn softmax_handles_large_logits() {
        let logits = array![1_000.0, 1_001.0, 999.0];
        let probabilities = softmax(&logits.view()).unwrap();

        assert!(probabilities.iter().all(|value| value.is_finite()));
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probabilities[1] > probabilities[0]);
    }

    #[test]
    fn softmax_rejects_non_finite_logits() {
        let logits = array![0.0, f32::NAN];
        assert!(softmax(&logits.view()).is_err());
    }
}
