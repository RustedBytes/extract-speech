//! PulseVAD-specific streaming iterator.

// Segment weights are ratios of bounded window lengths.
#![allow(clippy::cast_precision_loss)]

use log::debug;

use crate::{
    pulsevad_frontend::{HOP_SAMPLES, WINDOW_SAMPLES},
    utils,
    vad_iter::{self, VadModel},
};

pub struct PulseVadIter<M> {
    model: M,
    params: utils::VadParams,
    speeches: Vec<utils::TimeStamp>,
}

impl<M: VadModel> PulseVadIter<M> {
    #[must_use]
    pub fn new(model: M, params: utils::VadParams) -> Self {
        Self {
            model,
            params,
            speeches: Vec::new(),
        }
    }

    /// Detects speech segments using overlapping `PulseVAD` windows.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid parameters or samples, duration arithmetic
    /// overflow, or when model reset or probability inference fails.
    pub fn process(&mut self, samples: &[f32]) -> anyhow::Result<&[utils::TimeStamp]> {
        vad_iter::validate_parameters(&self.params)?;
        vad_iter::validate_samples(samples)?;
        self.model.reset()?;
        self.speeches.clear();

        let min_speech_samples = vad_iter::milliseconds_to_samples(
            self.params.min_speech_duration_ms,
            self.params.sample_rate,
            "minimum speech duration",
        )?;
        let min_silence_samples = vad_iter::milliseconds_to_samples(
            self.params.min_silence_duration_ms,
            self.params.sample_rate,
            "minimum silence duration",
        )?;
        let speech_pad_samples = vad_iter::milliseconds_to_samples(
            self.params.speech_pad_ms,
            self.params.sample_rate,
            "speech padding",
        )?;
        let max_speech_samples = if self.params.max_speech_duration_s.is_infinite() {
            None
        } else {
            let total = vad_iter::seconds_to_samples(
                self.params.max_speech_duration_s,
                self.params.sample_rate,
            )?;
            let padding = speech_pad_samples
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("speech padding exceeds usize"))?;
            let maximum = total.checked_sub(padding).ok_or_else(|| {
                anyhow::anyhow!("maximum speech duration is shorter than its padding")
            })?;
            anyhow::ensure!(
                maximum >= WINDOW_SAMPLES,
                "PulseVAD maximum speech duration must contain at least one model window"
            );
            Some(maximum)
        };

        if samples.len() < WINDOW_SAMPLES {
            let mut padded = vec![0.0; WINDOW_SAMPLES];
            padded[..samples.len()].copy_from_slice(samples);
            if self.model.probability(&padded)? >= self.params.threshold {
                self.push_if_long_enough(0, samples.len(), min_speech_samples);
            }
            vad_iter::apply_speech_padding(&mut self.speeches, speech_pad_samples, samples.len());
            return Ok(&self.speeches);
        }

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
                let segment_start = *current_start.get_or_insert(start);
                last_speech_end = start + WINDOW_SAMPLES;
                if let Some(maximum) = max_speech_samples {
                    if last_speech_end.saturating_sub(segment_start) > maximum {
                        let segment_end = segment_start.saturating_add(maximum);
                        self.push_if_long_enough(segment_start, segment_end, min_speech_samples);
                        current_start = Some(segment_end);
                    }
                }
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

        vad_iter::apply_speech_padding(&mut self.speeches, speech_pad_samples, samples.len());

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
                end: WINDOW_SAMPLES + HOP_SAMPLES + 480
            }]
        );
    }

    #[test]
    fn short_audio_respects_minimum_speech_duration() {
        let mut iter = iterator(&[0.9]);
        let speeches = iter.process(&[0.0; 800]).unwrap();

        assert!(speeches.is_empty());
    }

    #[test]
    fn continuous_speech_respects_maximum_duration() {
        let params = utils::VadParams {
            threshold: 0.5,
            min_speech_duration_ms: 100,
            speech_pad_ms: 30,
            max_speech_duration_s: 0.3,
            ..Default::default()
        };
        let mut iter = PulseVadIter::new(
            FakeModel {
                probabilities: [0.9; 5].into_iter().collect(),
                frames_seen: 0,
            },
            params,
        );
        let samples = vec![0.0; WINDOW_SAMPLES + 4 * HOP_SAMPLES];

        let speeches = iter.process(&samples).unwrap();

        assert_eq!(speeches.len(), 3);
        assert!(speeches
            .iter()
            .all(|speech| speech.end - speech.start <= 4_800));
    }
}
