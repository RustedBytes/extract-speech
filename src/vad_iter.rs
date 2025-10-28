use log::debug;

use crate::{silero_v5::Silero, utils};

#[derive(Debug)]
pub struct VadIter {
    silero: Silero,
    params: Params,
    state: State,
}

impl VadIter {
    pub fn new(silero: Silero, params: utils::VadParams) -> Self {
        let params_mixed = Params::from(params.clone());

        if params.debug {
            debug!("vad_params: {:?}", params);
            debug!("params_mixed: {:?}", params_mixed);
        }

        Self {
            silero,
            params: params_mixed,
            state: State::new(),
        }
    }

    pub fn process(&mut self, samples: Vec<f32>) -> Result<&[utils::TimeStamp], anyhow::Error> {
        self.reset_states();

        let mut speech_probs: Vec<f32> = Vec::new();

        for audio_frame in samples.chunks_exact(self.params.frame_size_samples) {
            if audio_frame.len() < self.params.frame_size_samples {
                continue;
            }

            let speech_prob: f32 = self.silero.probability(audio_frame.to_vec())?;

            speech_probs.push(speech_prob);

            self.state.update(&self.params, speech_prob);
        }

        self.state.check_for_last_speech(samples.len());

        Ok(&self.state.speeches)
    }
}

impl VadIter {
    pub fn reset_states(&mut self) {
        self.state = State::new();
        self.silero.reset();
    }
}

#[allow(unused)]
#[derive(Debug)]
struct Params {
    frame_size: usize,
    threshold: f32,
    min_silence_duration_ms: usize,
    speech_pad_ms: usize,
    min_speech_duration_ms: usize,
    max_speech_duration_s: f32,
    sample_rate: usize,
    sr_per_ms: usize,
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
        let debug = value.debug;
        let frame_size = value.frame_size;
        let threshold = value.threshold;
        let min_silence_duration_ms = value.min_silence_duration_ms;
        let speech_pad_ms = value.speech_pad_ms;
        let min_speech_duration_ms = value.min_speech_duration_ms;
        let max_speech_duration_s = value.max_speech_duration_s;
        let sample_rate = value.sample_rate;
        let sr_per_ms = sample_rate / 1000;
        let frame_size_samples = frame_size * sr_per_ms;
        let min_speech_samples = sr_per_ms * min_speech_duration_ms;
        let speech_pad_samples = sr_per_ms * speech_pad_ms;
        let max_speech_samples = sample_rate as f32 * max_speech_duration_s
            - frame_size_samples as f32
            - 2.0 * speech_pad_samples as f32;
        let min_silence_samples = sr_per_ms * min_silence_duration_ms;
        let min_silence_samples_at_max_speech = sr_per_ms * 98;

        Self {
            frame_size,
            threshold,
            min_silence_duration_ms,
            speech_pad_ms,
            min_speech_duration_ms,
            max_speech_duration_s,
            sample_rate,
            sr_per_ms,
            frame_size_samples,
            min_speech_samples,
            speech_pad_samples,
            max_speech_samples,
            min_silence_samples,
            min_silence_samples_at_max_speech,
            debug,
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
    fn new() -> Self {
        Default::default()
    }

    fn update(&mut self, params: &Params, speech_prob: f32) {
        self.current_sample += params.frame_size_samples;
        if speech_prob > params.threshold {
            if self.temp_end != 0 {
                self.temp_end = 0;
                if self.next_start < self.prev_end {
                    self.next_start = self
                        .current_sample
                        .saturating_sub(params.frame_size_samples)
                }
            }
            if !self.triggered {
                self.debug(speech_prob, params, "start");
                self.triggered = true;
                self.current_speech.start =
                    self.current_sample as i64 - params.frame_size_samples as i64;
            }
            return;
        }
        if self.triggered
            && (self.current_sample as i64 - self.current_speech.start) as f32
                > params.max_speech_samples
        {
            if self.prev_end > 0 {
                self.current_speech.end = self.prev_end as _;
                self.take_speech();
                if self.next_start < self.prev_end {
                    self.triggered = false
                } else {
                    self.current_speech.start = self.next_start as _;
                }
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
            } else {
                self.current_speech.end = self.current_sample as _;
                self.take_speech();
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                self.triggered = false;
            }
            return;
        }

        if speech_prob >= (params.threshold - 0.15) && (speech_prob < params.threshold) {
            if self.triggered {
                self.debug(speech_prob, params, "speaking")
            } else {
                self.debug(speech_prob, params, "silence")
            }
        }

        if self.triggered && speech_prob < (params.threshold - 0.15) {
            self.debug(speech_prob, params, "end");
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample.saturating_sub(self.temp_end)
                > params.min_silence_samples_at_max_speech
            {
                self.prev_end = self.temp_end;
            }
            if self.current_sample.saturating_sub(self.temp_end) >= params.min_silence_samples {
                self.current_speech.end = self.temp_end as _;
                if self.current_speech.end - self.current_speech.start
                    > params.min_speech_samples as _
                {
                    self.take_speech();
                    self.prev_end = 0;
                    self.next_start = 0;
                    self.temp_end = 0;
                    self.triggered = false;
                }
            }
        }
    }

    fn take_speech(&mut self) {
        self.speeches.push(std::mem::take(&mut self.current_speech)); // current speech becomes TimeStamp::default() due to take()
    }

    fn check_for_last_speech(&mut self, last_sample: usize) {
        if self.current_speech.start > 0 {
            self.current_speech.end = last_sample as _;
            self.take_speech();
            self.prev_end = 0;
            self.next_start = 0;
            self.temp_end = 0;
            self.triggered = false;
        }
    }

    fn debug(&self, speech_prob: f32, params: &Params, title: &str) {
        if params.debug {
            let speech = self.current_sample as f32
                - params.frame_size_samples as f32
                - if title == "end" {
                    params.speech_pad_samples
                } else {
                    0
                } as f32; // minus window_size_samples to get precise start time point.
            debug!(
                "[{:10}: {:.3} s ({:.3}) {:8}]",
                title,
                speech / params.sample_rate as f32,
                speech_prob,
                self.current_sample - params.frame_size_samples,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_from_vad_params_default() {
        let vad_params = utils::VadParams::default();
        let params = Params::from(vad_params.clone());

        assert_eq!(params.frame_size, 32);
        assert_eq!(params.threshold, 0.5);
        assert_eq!(params.min_silence_duration_ms, 100);
        assert_eq!(params.speech_pad_ms, 30);
        assert_eq!(params.min_speech_duration_ms, 250);
        assert_eq!(params.sample_rate, 16_000);
        assert!(!params.debug);

        // Check calculated values
        assert_eq!(params.sr_per_ms, 16); // 16000 / 1000
        assert_eq!(params.frame_size_samples, 512); // 32 * 16
        assert_eq!(params.min_speech_samples, 4000); // 16 * 250
        assert_eq!(params.speech_pad_samples, 480); // 16 * 30
        assert_eq!(params.min_silence_samples, 1600); // 16 * 100
        assert_eq!(params.min_silence_samples_at_max_speech, 1568); // 16 * 98
    }

    #[test]
    fn test_params_from_vad_params_custom() {
        let vad_params = utils::VadParams {
            frame_size: 64,
            threshold: 0.7,
            min_silence_duration_ms: 200,
            speech_pad_ms: 50,
            min_speech_duration_ms: 300,
            max_speech_duration_s: 30.0,
            sample_rate: 8_000,
            debug: true,
        };

        let params = Params::from(vad_params);

        assert_eq!(params.frame_size, 64);
        assert_eq!(params.threshold, 0.7);
        assert_eq!(params.sample_rate, 8_000);
        assert!(params.debug);

        // Check calculated values for 8kHz
        assert_eq!(params.sr_per_ms, 8); // 8000 / 1000
        assert_eq!(params.frame_size_samples, 512); // 64 * 8
        assert_eq!(params.min_speech_samples, 2400); // 8 * 300
        assert_eq!(params.speech_pad_samples, 400); // 8 * 50
        assert_eq!(params.min_silence_samples, 1600); // 8 * 200
    }

    #[test]
    fn test_params_max_speech_samples_calculation() {
        let vad_params = utils::VadParams {
            frame_size: 32,
            threshold: 0.5,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            min_speech_duration_ms: 250,
            max_speech_duration_s: 10.0,
            sample_rate: 16_000,
            debug: false,
        };

        let params = Params::from(vad_params);

        // max_speech_samples = sample_rate * max_speech_duration_s - frame_size_samples - 2 * speech_pad_samples
        // = 16000 * 10.0 - 512 - 2 * 480
        // = 160000 - 512 - 960
        // = 158528
        let expected = 16_000.0 * 10.0 - 512.0 - 2.0 * 480.0;
        assert_eq!(params.max_speech_samples, expected);
    }

    #[test]
    fn test_params_infinite_max_speech_duration() {
        let vad_params = utils::VadParams {
            frame_size: 32,
            threshold: 0.5,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            min_speech_duration_ms: 250,
            max_speech_duration_s: f32::INFINITY,
            sample_rate: 16_000,
            debug: false,
        };

        let params = Params::from(vad_params);

        // When max_speech_duration_s is infinite, max_speech_samples should also be infinite
        assert!(params.max_speech_samples.is_infinite());
    }

    #[test]
    fn test_state_new() {
        let state = State::new();

        assert_eq!(state.current_sample, 0);
        assert_eq!(state.temp_end, 0);
        assert_eq!(state.next_start, 0);
        assert_eq!(state.prev_end, 0);
        assert!(!state.triggered);
        assert_eq!(state.current_speech.start, 0);
        assert_eq!(state.current_speech.end, 0);
        assert_eq!(state.speeches.len(), 0);
    }

    #[test]
    fn test_state_default() {
        let state = State::default();

        assert_eq!(state.current_sample, 0);
        assert_eq!(state.temp_end, 0);
        assert_eq!(state.next_start, 0);
        assert_eq!(state.prev_end, 0);
        assert!(!state.triggered);
        assert_eq!(state.speeches.len(), 0);
    }

    #[test]
    fn test_params_different_sample_rates() {
        // Test with 8kHz
        let vad_params_8k = utils::VadParams {
            frame_size: 32,
            sample_rate: 8_000,
            ..Default::default()
        };
        let params_8k = Params::from(vad_params_8k);
        assert_eq!(params_8k.sr_per_ms, 8);
        assert_eq!(params_8k.frame_size_samples, 256); // 32 * 8

        // Test with 16kHz
        let vad_params_16k = utils::VadParams {
            frame_size: 32,
            sample_rate: 16_000,
            ..Default::default()
        };
        let params_16k = Params::from(vad_params_16k);
        assert_eq!(params_16k.sr_per_ms, 16);
        assert_eq!(params_16k.frame_size_samples, 512); // 32 * 16

        // Test with 48kHz
        let vad_params_48k = utils::VadParams {
            frame_size: 32,
            sample_rate: 48_000,
            ..Default::default()
        };
        let params_48k = Params::from(vad_params_48k);
        assert_eq!(params_48k.sr_per_ms, 48);
        assert_eq!(params_48k.frame_size_samples, 1536); // 32 * 48
    }
}
