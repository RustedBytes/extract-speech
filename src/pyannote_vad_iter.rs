use log::debug;
use ndarray::{ArrayView1, Axis, Ix3};

use crate::{pyannote_vad_ort::PyAnnote, utils};

#[derive(Debug)]
pub struct PyAnnoteVadIter {
    pyannote: PyAnnote,
    params: utils::VadParams,
    config: DiarizationConfig,
    speeches: Vec<utils::TimeStamp>,
}

#[derive(Debug, Clone)]
struct DiarizationConfig {
    offset: f32,
    step: f32,
    sampling_rate: f32,
}

#[derive(Debug, Clone)]
struct SegmentInternal {
    start: usize, // Start frame index
    end: usize,   // End frame index (exclusive)
    score: f32,   // Accumulated probability score
}

impl PyAnnoteVadIter {
    pub fn new(pyannote: PyAnnote, params: utils::VadParams) -> Self {
        let config = DiarizationConfig {
            offset: 990.0,
            step: 270.0,
            sampling_rate: params.sample_rate as f32,
        };

        if params.debug {
            debug!("PyAnnote vad_params: {:?}", params);
            debug!("PyAnnote config: {:?}", config);
        }

        Self {
            pyannote,
            params,
            config,
            speeches: Vec::new(),
        }
    }

    pub fn process(&mut self, samples: &[f32]) -> Result<&[utils::TimeStamp], anyhow::Error> {
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

        let num_frames_float = (num_samples as f32 - self.config.offset) / self.config.step;
        if num_frames_float <= 0.0 {
            return Err(anyhow::anyhow!(
                "Calculated number of frames is zero or negative."
            ));
        }
        let ratio = (num_samples as f32 / num_frames_float) / self.config.sampling_rate;

        let mut speeches: Vec<utils::TimeStamp> = Vec::new();

        // Process first batch (we expect batch size of 1)
        if let Some(batch_item_logits) = logits_3d.axis_iter(Axis(0)).next() {
            let mut accumulated_segments: Vec<SegmentInternal> = Vec::new();
            let mut in_speech = false;

            for (i, frame_scores_view) in batch_item_logits.axis_iter(Axis(0)).enumerate() {
                let probabilities = softmax(frame_scores_view);

                // Find the class with maximum probability
                let (score, class_id) = find_max(&probabilities);

                // For VAD, we typically look at whether it's speech (class > 0) or not (class 0)
                // Adjust threshold based on the probability
                let is_speech = class_id > 0 && score > self.params.threshold;

                let start_frame = i;
                let end_frame = i + 1;

                if is_speech {
                    if !in_speech {
                        // Start of new speech segment
                        in_speech = true;
                        accumulated_segments.push(SegmentInternal {
                            start: start_frame,
                            end: end_frame,
                            score,
                        });
                    } else {
                        // Continue current speech segment
                        if let Some(last_segment) = accumulated_segments.last_mut() {
                            last_segment.end = end_frame;
                            last_segment.score += score;
                        }
                    }
                } else if in_speech {
                    // End of speech segment
                    in_speech = false;
                }
            }

            // Convert segments to timestamps
            for seg in accumulated_segments {
                let num_frames_in_segment = seg.end - seg.start;
                if num_frames_in_segment == 0 {
                    continue;
                }

                let start_time = seg.start as f32 * ratio;
                let end_time = seg.end as f32 * ratio;

                // Filter out segments that are too short (< 0.1s)
                if end_time - start_time < 0.1 {
                    continue;
                }

                let start_sample = (start_time * self.config.sampling_rate) as usize;
                let end_sample = ((end_time * self.config.sampling_rate) as usize).min(num_samples);

                speeches.push(utils::TimeStamp {
                    start: start_sample,
                    end: end_sample,
                });
            }
        }

        Ok(speeches)
    }
}

// Softmax implementation for a 1D ArrayView
fn softmax(x: ArrayView1<'_, f32>) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut softmax_array: Vec<f32> = x.iter().map(|value| (value - max).exp()).collect();

    let sum: f32 = softmax_array.iter().sum();

    for value in &mut softmax_array {
        *value /= sum;
    }

    softmax_array
}

// Find the maximum value and its index in a slice
fn find_max(probs: &[f32]) -> (f32, usize) {
    probs.iter().enumerate().fold(
        (f32::NEG_INFINITY, 0),
        |(max_prob, max_idx), (idx, &prob)| {
            if prob > max_prob {
                (prob, idx)
            } else {
                (max_prob, max_idx)
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn softmax_handles_large_logits() {
        let logits = array![1_000.0, 1_001.0, 999.0];
        let probabilities = softmax(logits.view());

        assert!(probabilities.iter().all(|value| value.is_finite()));
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(find_max(&probabilities).1, 1);
    }
}
