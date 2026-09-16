//! Generic probability-to-speech-segment iterator.

// Duration thresholds are expressed in seconds and converted from bounded sample counts.
#![allow(clippy::cast_precision_loss)]

use log::debug;

use crate::{utils, SAMPLE_RATE};

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
    params: utils::VadParams,
    state: State,
}

impl<M: VadModel> VadIter<M> {
    #[must_use]
    pub fn new(model: M, params: utils::VadParams) -> Self {
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
    /// Returns an error if parameters or samples are invalid, arithmetic
    /// overflows, or the model cannot be reset or evaluated.
    pub fn process(&mut self, samples: &[f32]) -> anyhow::Result<&[utils::TimeStamp]> {
        let params = Params::try_from(self.params.clone())?;
        validate_samples(samples)?;
        self.reset_states()?;

        for audio_frame in samples.chunks_exact(params.frame_size_samples) {
            let speech_probability = self.model.probability(audio_frame)?;
            self.state.update(&params, speech_probability)?;
        }

        self.state.finish(samples.len(), &params);
        apply_speech_padding(
            &mut self.state.speeches,
            params.speech_pad_samples,
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
    params: &utils::VadParams,
) -> anyhow::Result<Vec<utils::TimeStamp>> {
    let frame_size_samples = params
        .frame_size
        .checked_mul(params.sample_rate / 1_000)
        .ok_or_else(|| anyhow::anyhow!("VAD frame size exceeds usize"))?;
    segment_probabilities_with_frame_size(probabilities, total_samples, params, frame_size_samples)
}

#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub(crate) fn segment_probabilities_with_frame_size(
    probabilities: &[f32],
    total_samples: usize,
    params: &utils::VadParams,
    frame_size_samples: usize,
) -> anyhow::Result<Vec<utils::TimeStamp>> {
    let params = Params::try_from_with_frame_size(params, frame_size_samples)?;
    let mut state = State::default();
    for &probability in probabilities {
        state.update(&params, probability)?;
    }
    state.finish(total_samples, &params);
    apply_speech_padding(
        &mut state.speeches,
        params.speech_pad_samples,
        total_samples,
    );
    Ok(state.speeches)
}

#[derive(Debug)]
struct Params {
    threshold: f32,
    sample_rate: usize,
    frame_size_samples: usize,
    min_speech_samples: usize,
    speech_pad_samples: usize,
    max_speech_samples: Option<usize>,
    min_silence_samples: usize,
    min_silence_samples_at_max_speech: usize,
    debug: bool,
}

impl TryFrom<utils::VadParams> for Params {
    type Error = anyhow::Error;

    fn try_from(value: utils::VadParams) -> anyhow::Result<Self> {
        validate_parameters(&value)?;
        let samples_per_ms = value.sample_rate / 1000;
        let frame_size_samples = value
            .frame_size
            .checked_mul(samples_per_ms)
            .ok_or_else(|| anyhow::anyhow!("VAD frame size exceeds usize"))?;
        Self::try_from_with_frame_size(&value, frame_size_samples)
    }
}

impl Params {
    fn try_from_with_frame_size(
        value: &utils::VadParams,
        frame_size_samples: usize,
    ) -> anyhow::Result<Self> {
        validate_parameters(value)?;
        anyhow::ensure!(
            frame_size_samples > 0,
            "VAD frame size must be greater than zero"
        );

        let samples_per_ms = value.sample_rate / 1_000;
        let speech_pad_samples = samples_per_ms
            .checked_mul(value.speech_pad_ms)
            .ok_or_else(|| anyhow::anyhow!("speech padding exceeds usize"))?;
        let min_speech_samples = samples_per_ms
            .checked_mul(value.min_speech_duration_ms)
            .ok_or_else(|| anyhow::anyhow!("minimum speech duration exceeds usize"))?;
        let min_silence_samples = samples_per_ms
            .checked_mul(value.min_silence_duration_ms)
            .ok_or_else(|| anyhow::anyhow!("minimum silence duration exceeds usize"))?;
        let min_silence_samples_at_max_speech = samples_per_ms
            .checked_mul(98)
            .ok_or_else(|| anyhow::anyhow!("maximum-speech silence duration exceeds usize"))?;

        let max_speech_samples = if value.max_speech_duration_s.is_infinite() {
            None
        } else {
            let total = seconds_to_samples(value.max_speech_duration_s, value.sample_rate)?;
            let reserved = speech_pad_samples
                .checked_mul(2)
                .and_then(|padding| frame_size_samples.checked_add(padding))
                .ok_or_else(|| anyhow::anyhow!("maximum speech duration adjustment overflowed"))?;
            anyhow::ensure!(
                total > reserved,
                "maximum speech duration must exceed one frame plus twice the speech padding"
            );
            Some(total - reserved)
        };

        Ok(Self {
            threshold: value.threshold,
            sample_rate: value.sample_rate,
            frame_size_samples,
            min_speech_samples,
            speech_pad_samples,
            max_speech_samples,
            min_silence_samples,
            min_silence_samples_at_max_speech,
            debug: value.debug,
        })
    }
}

pub(crate) fn validate_parameters(params: &utils::VadParams) -> anyhow::Result<()> {
    anyhow::ensure!(
        params.sample_rate == SAMPLE_RATE,
        "VAD inference requires {SAMPLE_RATE} Hz samples"
    );
    anyhow::ensure!(
        params.frame_size > 0,
        "frame size must be greater than zero"
    );
    anyhow::ensure!(
        params.threshold.is_finite() && (0.0..=1.0).contains(&params.threshold),
        "threshold must be between 0 and 1"
    );
    anyhow::ensure!(
        params.max_speech_duration_s > 0.0,
        "maximum speech duration must be greater than zero"
    );

    let samples_per_ms = params.sample_rate / 1_000;
    let frame_size_samples = params
        .frame_size
        .checked_mul(samples_per_ms)
        .ok_or_else(|| anyhow::anyhow!("VAD frame size exceeds usize"))?;
    params
        .min_speech_duration_ms
        .checked_mul(samples_per_ms)
        .ok_or_else(|| anyhow::anyhow!("minimum speech duration exceeds usize"))?;
    params
        .min_silence_duration_ms
        .checked_mul(samples_per_ms)
        .ok_or_else(|| anyhow::anyhow!("minimum silence duration exceeds usize"))?;
    let speech_pad_samples = params
        .speech_pad_ms
        .checked_mul(samples_per_ms)
        .and_then(|samples| samples.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("speech padding exceeds usize"))?;
    if params.max_speech_duration_s.is_finite() {
        let total = seconds_to_samples(params.max_speech_duration_s, params.sample_rate)?;
        let reserved = frame_size_samples
            .checked_add(speech_pad_samples)
            .ok_or_else(|| anyhow::anyhow!("maximum speech duration adjustment overflowed"))?;
        anyhow::ensure!(
            total > reserved,
            "maximum speech duration must exceed one frame plus twice the speech padding"
        );
    }
    Ok(())
}

pub(crate) fn validate_samples(samples: &[f32]) -> anyhow::Result<()> {
    anyhow::ensure!(
        samples
            .iter()
            .all(|sample| sample.is_finite() && (-1.0..=1.0).contains(sample)),
        "VAD samples must be finite and normalized to -1.0..=1.0"
    );
    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // The value is finite, positive, and range-checked before conversion.
pub(crate) fn seconds_to_samples(seconds: f32, sample_rate: usize) -> anyhow::Result<usize> {
    let samples = f64::from(seconds) * sample_rate as f64;
    anyhow::ensure!(samples.is_finite(), "sample duration must be finite");
    anyhow::ensure!(
        samples <= usize::MAX as f64,
        "sample duration exceeds usize"
    );
    Ok(samples as usize)
}

#[cfg(any(feature = "candle", feature = "onnxruntime"))]
pub(crate) fn milliseconds_to_samples(
    milliseconds: usize,
    sample_rate: usize,
    label: &str,
) -> anyhow::Result<usize> {
    milliseconds
        .checked_mul(sample_rate)
        .map(|samples| samples / 1_000)
        .ok_or_else(|| anyhow::anyhow!("{label} exceeds usize"))
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
    fn update(&mut self, params: &Params, speech_probability: f32) -> anyhow::Result<()> {
        anyhow::ensure!(
            speech_probability.is_finite(),
            "VAD model returned a non-finite speech probability"
        );
        self.current_sample = self
            .current_sample
            .checked_add(params.frame_size_samples)
            .ok_or_else(|| anyhow::anyhow!("VAD sample position overflowed"))?;

        let is_speech = speech_probability > params.threshold;
        if is_speech {
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
        }

        if self.triggered
            && params.max_speech_samples.is_some_and(|maximum| {
                self.current_sample
                    .saturating_sub(self.current_speech.start)
                    > maximum
            })
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
            return Ok(());
        }

        if is_speech {
            return Ok(());
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
        Ok(())
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
            >= min_speech_samples
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

pub(crate) fn apply_speech_padding(
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

        if silence < speech_pad_samples.saturating_mul(2) {
            let half_silence = silence / 2;
            current.end = current.end.saturating_add(half_silence).min(total_samples);
            next.start = next.start.saturating_sub(silence - half_silence);
        } else {
            current.end = current
                .end
                .saturating_add(speech_pad_samples)
                .min(total_samples);
            next.start = next.start.saturating_sub(speech_pad_samples);
        }
    }

    if let Some(last) = speeches.last_mut() {
        last.end = last
            .end
            .saturating_add(speech_pad_samples)
            .min(total_samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params::try_from(utils::VadParams::default()).unwrap()
    }

    #[test]
    fn trailing_speech_that_starts_at_zero_is_kept() {
        let params = params();
        let mut state = State::default();
        for _ in 0..10 {
            state.update(&params, 1.0).unwrap();
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
        state.update(&params, 1.0).unwrap();
        for _ in 0..5 {
            state.update(&params, 0.0).unwrap();
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

    #[test]
    fn continuous_speech_is_split_at_the_maximum_duration() {
        let mut params = Params::try_from(utils::VadParams {
            min_speech_duration_ms: 0,
            speech_pad_ms: 0,
            max_speech_duration_s: 0.1,
            ..utils::VadParams::default()
        })
        .unwrap();
        params.min_silence_samples_at_max_speech = 0;
        let mut state = State::default();

        for _ in 0..6 {
            state.update(&params, 1.0).unwrap();
        }
        state.finish(state.current_sample, &params);

        assert_eq!(state.speeches.len(), 2);
        assert!(state.speeches.iter().all(|speech| {
            speech.end - speech.start <= seconds_to_samples(0.1, SAMPLE_RATE).unwrap()
        }));
    }

    struct ConstantModel;

    impl VadModel for ConstantModel {
        fn reset(&mut self) -> anyhow::Result<()> {
            Ok(())
        }

        fn probability(&mut self, _audio_frame: &[f32]) -> anyhow::Result<f32> {
            Ok(1.0)
        }
    }

    #[test]
    fn public_iterator_rejects_a_zero_frame_without_panicking() {
        let mut iterator = VadIter::new(
            ConstantModel,
            utils::VadParams {
                frame_size: 0,
                ..utils::VadParams::default()
            },
        );

        assert!(iterator.process(&[0.0; 512]).is_err());
    }

    #[test]
    fn parameter_validation_rejects_sample_count_overflow() {
        let params = utils::VadParams {
            speech_pad_ms: usize::MAX,
            ..utils::VadParams::default()
        };

        assert!(validate_parameters(&params).is_err());
    }
}
