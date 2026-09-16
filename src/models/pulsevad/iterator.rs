//! PulseVAD-specific streaming iterator.

use log::debug;

use crate::{
    pulsevad_frontend::{HOP_SAMPLES, WINDOW_SAMPLES},
    utils,
    vad_iter::VadModel,
};

pub struct PulseVadIter<M> {
    model: M,
    params: utils::VadParams,
    speeches: Vec<utils::TimeStamp>,
}

impl<M: VadModel> PulseVadIter<M> {
    pub fn new(model: M, params: utils::VadParams) -> Self {
        Self {
            model,
            params,
            speeches: Vec::new(),
        }
    }

    pub fn process(&mut self, samples: &[f32]) -> anyhow::Result<&[utils::TimeStamp]> {
        self.model.reset()?;
        self.speeches.clear();

        if samples.len() < WINDOW_SAMPLES {
            let mut padded = vec![0.0; WINDOW_SAMPLES];
            padded[..samples.len()].copy_from_slice(samples);
            if self.model.probability(&padded)? >= self.params.threshold {
                self.speeches.push(utils::TimeStamp {
                    start: 0,
                    end: samples.len(),
                });
            }
            return Ok(&self.speeches);
        }

        let min_speech_samples =
            self.params.sample_rate * self.params.min_speech_duration_ms / 1_000;
        let min_silence_samples =
            self.params.sample_rate * self.params.min_silence_duration_ms / 1_000;
        let mut current_start = None;
        let mut last_speech_end = 0;

        for start in (0..=samples.len() - WINDOW_SAMPLES).step_by(HOP_SAMPLES) {
            let probability = self
                .model
                .probability(&samples[start..start + WINDOW_SAMPLES])?;
            let is_speech = probability >= self.params.threshold;

            if self.params.debug {
                debug!(
                    "[PulseVAD: {:.3} s ({probability:.3}) {}]",
                    start as f32 / self.params.sample_rate as f32,
                    if is_speech { "speech" } else { "silence" }
                );
            }

            if is_speech {
                current_start.get_or_insert(start);
                last_speech_end = start + WINDOW_SAMPLES;
            } else if let Some(segment_start) = current_start {
                if start.saturating_sub(last_speech_end) >= min_silence_samples {
                    self.push_if_long_enough(
                        segment_start,
                        last_speech_end.min(samples.len()),
                        min_speech_samples,
                    );
                    current_start = None;
                }
            }
        }

        if let Some(segment_start) = current_start {
            self.push_if_long_enough(
                segment_start,
                last_speech_end.min(samples.len()),
                min_speech_samples,
            );
        }

        Ok(&self.speeches)
    }

    fn push_if_long_enough(&mut self, start: usize, end: usize, minimum: usize) {
        if end.saturating_sub(start) >= minimum {
            self.speeches.push(utils::TimeStamp { start, end });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    struct FakeModel {
        probabilities: VecDeque<f32>,
        frames_seen: usize,
    }

    impl VadModel for FakeModel {
        fn reset(&mut self) -> anyhow::Result<()> {
            self.frames_seen = 0;
            Ok(())
        }

        fn probability(&mut self, audio_frame: &[f32]) -> anyhow::Result<f32> {
            assert_eq!(audio_frame.len(), WINDOW_SAMPLES);
            self.frames_seen += 1;
            Ok(self.probabilities.pop_front().unwrap_or(0.0))
        }
    }

    fn iterator(probabilities: &[f32]) -> PulseVadIter<FakeModel> {
        let params = utils::VadParams {
            threshold: 0.5,
            min_speech_duration_ms: 100,
            min_silence_duration_ms: 100,
            ..Default::default()
        };
        PulseVadIter::new(
            FakeModel {
                probabilities: probabilities.iter().copied().collect(),
                frames_seen: 99,
            },
            params,
        )
    }

    #[test]
    fn aggregates_overlapping_speech_windows() {
        let mut iter = iterator(&[0.9, 0.8, 0.1, 0.1, 0.1]);
        let samples = vec![0.0; WINDOW_SAMPLES + 4 * HOP_SAMPLES];
        let speeches = iter.process(&samples).unwrap();

        assert_eq!(
            speeches,
            &[utils::TimeStamp {
                start: 0,
                end: WINDOW_SAMPLES + HOP_SAMPLES
            }]
        );
    }

    #[test]
    fn pads_short_audio_for_inference() {
        let mut iter = iterator(&[0.9]);
        let speeches = iter.process(&[0.0; 800]).unwrap();

        assert_eq!(speeches, &[utils::TimeStamp { start: 0, end: 800 }]);
    }
}
