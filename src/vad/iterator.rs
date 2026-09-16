//! Generic probability-to-speech-segment iterator.

// Duration thresholds are expressed in seconds and converted from bounded sample counts.
#![allow(clippy::cast_precision_loss)]

use log::debug;

use crate::utils;

pub trait VadModel {
    /// Resets model state before processing an independent audio stream.
    ///
    /// # Errors
    ///
    /// Returns a backend-specific reset error.
    fn reset(&mut self) -> anyhow::Result<()>;

    /// Computes the speech probability for one audio frame.
    ///
    /// # Errors
    ///
    /// Returns a backend-specific inference or input-validation error.
    fn probability(&mut self, audio_frame: &[f32]) -> anyhow::Result<f32>;
}

#[derive(Debug)]
pub struct VadIter<M> {
    model: M,
    params: Params,
    state: State,
}

impl<M: VadModel> VadIter<M> {
    #[must_use]
    pub fn new(model: M, params: utils::VadParams) -> Self {
        let params = Params::from(params);

        if params.debug {
            debug!("vad_params: {params:?}");
        }

        Self {
            model,
            params,
            state: State::default(),
        }
    }

    /// Detects speech segments in one complete waveform.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be reset or evaluated.
    pub fn process(&mut self, samples: &[f32]) -> anyhow::Result<&[utils::TimeStamp]> {
        self.reset_states()?;

        for audio_frame in samples.chunks_exact(self.params.frame_size_samples) {
            let speech_probability = self.model.probability(audio_frame)?;
            self.state.update(&self.params, speech_probability);
        }

        self.state.finish(samples.len(), &self.params);
        apply_speech_padding(
            &mut self.state.speeches,
            self.params.speech_pad_samples,
            samples.len(),
        );

        Ok(&self.state.speeches)
    }

    fn reset_states(&mut self) -> anyhow::Result<()> {
        self.state = State::default();
        self.model.reset()
    }
}

#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub(crate) fn segment_probabilities(
    probabilities: &[f32],
    total_samples: usize,
    params: utils::VadParams,
) -> Vec<utils::TimeStamp> {
    let params = Params::from(params);
    let mut state = State::default();
    for &probability in probabilities {
        state.update(&params, probability);
    }
    state.finish(total_samples, &params);
    apply_speech_padding(
        &mut state.speeches,
        params.speech_pad_samples,
        total_samples,
    );
    state.speeches
}

#[derive(Debug)]
struct Params {
    threshold: f32,
    sample_rate: usize,
    frame_size_samples: usize,
    min_speech_samples: usize,
    speech_pad_samples: usize,
    max_speech_samples: f32,
    min_silence_samples: usize,
    min_silence_samples_at_max_speech: usize,
    debug: bool,
}

impl From<utils::VadParams> for Params {
    fn from(value: utils::VadParams) -> Self {
        let samples_per_ms = value.sample_rate / 1000;
        let frame_size_samples = value.frame_size * samples_per_ms;
        let speech_pad_samples = samples_per_ms * value.speech_pad_ms;

        Self {
            threshold: value.threshold,
            sample_rate: value.sample_rate,
            frame_size_samples,
            min_speech_samples: samples_per_ms * value.min_speech_duration_ms,
            speech_pad_samples,
            max_speech_samples: value.sample_rate as f32 * value.max_speech_duration_s
                - frame_size_samples as f32
                - 2.0 * speech_pad_samples as f32,
            min_silence_samples: samples_per_ms * value.min_silence_duration_ms,
            min_silence_samples_at_max_speech: samples_per_ms * 98,
            debug: value.debug,
        }
    }
}

#[derive(Debug, Default)]
struct State {
    current_sample: usize,
    temp_end: usize,
    next_start: usize,
    prev_end: usize,
    triggered: bool,
    current_speech: utils::TimeStamp,
    speeches: Vec<utils::TimeStamp>,
}

impl State {
    fn update(&mut self, params: &Params, speech_probability: f32) {
        self.current_sample += params.frame_size_samples;

        if speech_probability > params.threshold {
            if self.temp_end != 0 {
                self.temp_end = 0;
                if self.next_start < self.prev_end {
                    self.next_start = self
                        .current_sample
                        .saturating_sub(params.frame_size_samples);
                }
            }

            if !self.triggered {
                self.log_transition(speech_probability, params, "start");
                self.triggered = true;
                self.current_speech.start = self
                    .current_sample
                    .saturating_sub(params.frame_size_samples);
            }
            return;
        }

        if self.triggered
            && self
                .current_sample
                .saturating_sub(self.current_speech.start) as f32
                > params.max_speech_samples
        {
            if self.prev_end > 0 {
                self.current_speech.end = self.prev_end;
                self.take_speech();
                if self.next_start < self.prev_end {
                    self.triggered = false;
                } else {
                    self.current_speech.start = self.next_start;
                }
            } else {
                self.current_speech.end = self.current_sample;
                self.take_speech();
                self.triggered = false;
            }
            self.clear_temporary_boundaries();
            return;
        }

        if speech_probability >= params.threshold - 0.15 && speech_probability < params.threshold {
            let state = if self.triggered {
                "speaking"
            } else {
                "silence"
            };
            self.log_transition(speech_probability, params, state);
        }

        if self.triggered && speech_probability < params.threshold - 0.15 {
            self.log_transition(speech_probability, params, "end");
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample.saturating_sub(self.temp_end)
                > params.min_silence_samples_at_max_speech
            {
                self.prev_end = self.temp_end;
            }
            if self.current_sample.saturating_sub(self.temp_end) >= params.min_silence_samples {
                self.finish_current_speech(self.temp_end, params.min_speech_samples);
            }
        }
    }

    fn finish(&mut self, last_sample: usize, params: &Params) {
        if self.triggered {
            self.finish_current_speech(last_sample, params.min_speech_samples);
        }
    }

    fn finish_current_speech(&mut self, end: usize, min_speech_samples: usize) {
        self.current_speech.end = end;
        if self
            .current_speech
            .end
            .saturating_sub(self.current_speech.start)
            > min_speech_samples
        {
            self.take_speech();
        } else {
            self.current_speech = utils::TimeStamp::default();
        }
        self.triggered = false;
        self.clear_temporary_boundaries();
    }

    fn take_speech(&mut self) {
        self.speeches.push(std::mem::take(&mut self.current_speech));
    }

    fn clear_temporary_boundaries(&mut self) {
        self.prev_end = 0;
        self.next_start = 0;
        self.temp_end = 0;
    }

    fn log_transition(&self, speech_probability: f32, params: &Params, title: &str) {
        if params.debug {
            let sample = self
                .current_sample
                .saturating_sub(params.frame_size_samples);
            debug!(
                "[{:10}: {:.3} s ({:.3}) {:8}]",
                title,
                sample as f32 / params.sample_rate as f32,
                speech_probability,
                sample,
            );
        }
    }
}

fn apply_speech_padding(
    speeches: &mut [utils::TimeStamp],
    speech_pad_samples: usize,
    total_samples: usize,
) {
    if speeches.is_empty() {
        return;
    }

    speeches[0].start = speeches[0].start.saturating_sub(speech_pad_samples);

    for index in 0..speeches.len().saturating_sub(1) {
        let (left, right) = speeches.split_at_mut(index + 1);
        let current = &mut left[index];
        let next = &mut right[0];
        let silence = next.start.saturating_sub(current.end);

        if silence < 2 * speech_pad_samples {
            let half_silence = silence / 2;
            current.end = (current.end + half_silence).min(total_samples);
            next.start = next.start.saturating_sub(silence - half_silence);
        } else {
            current.end = (current.end + speech_pad_samples).min(total_samples);
            next.start = next.start.saturating_sub(speech_pad_samples);
        }
    }

    if let Some(last) = speeches.last_mut() {
        last.end = (last.end + speech_pad_samples).min(total_samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params::from(utils::VadParams::default())
    }

    #[test]
    fn trailing_speech_that_starts_at_zero_is_kept() {
        let params = params();
        let mut state = State::default();
        for _ in 0..10 {
            state.update(&params, 1.0);
        }

        state.finish(state.current_sample, &params);

        assert_eq!(state.speeches.len(), 1);
        assert_eq!(state.speeches[0].start, 0);
        assert_eq!(state.speeches[0].end, 10 * params.frame_size_samples);
    }

    #[test]
    fn short_speech_is_discarded_and_state_is_reset() {
        let params = params();
        let mut state = State::default();
        state.update(&params, 1.0);
        for _ in 0..5 {
            state.update(&params, 0.0);
        }

        assert!(!state.triggered);
        assert!(state.speeches.is_empty());
        assert_eq!(state.current_speech, utils::TimeStamp::default());
    }

    #[test]
    fn padding_is_shared_between_close_segments() {
        let mut speeches = vec![
            utils::TimeStamp {
                start: 100,
                end: 200,
            },
            utils::TimeStamp {
                start: 220,
                end: 300,
            },
        ];

        apply_speech_padding(&mut speeches, 30, 400);

        assert_eq!(
            speeches[0],
            utils::TimeStamp {
                start: 70,
                end: 210
            }
        );
        assert_eq!(
            speeches[1],
            utils::TimeStamp {
                start: 210,
                end: 330
            }
        );
    }
}
