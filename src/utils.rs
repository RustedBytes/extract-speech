pub const VAD_SAMPLE_RATE: usize = 16_000;

#[derive(Debug, Clone)]
pub struct VadParams {
    pub frame_size: usize,
    pub threshold: f32,
    pub min_silence_duration_ms: usize,
    pub speech_pad_ms: usize,
    pub min_speech_duration_ms: usize,
    pub max_speech_duration_s: f32,
    pub sample_rate: usize,
    pub debug: bool,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            frame_size: 32,
            threshold: 0.5,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            min_speech_duration_ms: 250,
            max_speech_duration_s: f32::INFINITY,
            sample_rate: VAD_SAMPLE_RATE,
            debug: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimeStamp {
    pub start: usize,
    pub end: usize,
}

impl std::fmt::Display for TimeStamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[start:{:08}, end:{:08}]", self.start, self.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_params_default() {
        let params = VadParams::default();

        assert_eq!(params.frame_size, 32);
        assert_eq!(params.threshold, 0.5);
        assert_eq!(params.min_silence_duration_ms, 100);
        assert_eq!(params.speech_pad_ms, 30);
        assert_eq!(params.min_speech_duration_ms, 250);
        assert_eq!(params.max_speech_duration_s, f32::INFINITY);
        assert_eq!(params.sample_rate, 16_000);
        assert!(!params.debug);
    }

    #[test]
    fn test_vad_params_clone() {
        let params1 = VadParams {
            frame_size: 64,
            threshold: 0.7,
            min_silence_duration_ms: 200,
            speech_pad_ms: 50,
            min_speech_duration_ms: 300,
            max_speech_duration_s: 30.0,
            sample_rate: VAD_SAMPLE_RATE,
            debug: true,
        };

        let params2 = params1.clone();

        assert_eq!(params1.frame_size, params2.frame_size);
        assert_eq!(params1.threshold, params2.threshold);
        assert_eq!(
            params1.min_silence_duration_ms,
            params2.min_silence_duration_ms
        );
        assert_eq!(params1.speech_pad_ms, params2.speech_pad_ms);
        assert_eq!(
            params1.min_speech_duration_ms,
            params2.min_speech_duration_ms
        );
        assert_eq!(params1.max_speech_duration_s, params2.max_speech_duration_s);
        assert_eq!(params1.sample_rate, params2.sample_rate);
        assert_eq!(params1.debug, params2.debug);
    }

    #[test]
    fn test_timestamp_default() {
        let ts = TimeStamp::default();

        assert_eq!(ts.start, 0);
        assert_eq!(ts.end, 0);
    }

    #[test]
    fn test_timestamp_display() {
        let ts = TimeStamp {
            start: 1000,
            end: 2000,
        };

        let display_string = format!("{}", ts);
        assert_eq!(display_string, "[start:00001000, end:00002000]");
    }

    #[test]
    fn test_timestamp_display_padding() {
        let ts = TimeStamp {
            start: 42,
            end: 123456789,
        };

        let display_string = format!("{}", ts);
        assert_eq!(display_string, "[start:00000042, end:123456789]");
    }

    #[test]
    fn test_vad_params_custom_values() {
        let params = VadParams {
            frame_size: 16,
            threshold: 0.8,
            min_silence_duration_ms: 50,
            speech_pad_ms: 20,
            min_speech_duration_ms: 150,
            max_speech_duration_s: 60.0,
            sample_rate: 8_000,
            debug: false,
        };

        assert_eq!(params.frame_size, 16);
        assert_eq!(params.threshold, 0.8);
        assert_eq!(params.min_silence_duration_ms, 50);
        assert_eq!(params.speech_pad_ms, 20);
        assert_eq!(params.min_speech_duration_ms, 150);
        assert_eq!(params.max_speech_duration_s, 60.0);
        assert_eq!(params.sample_rate, 8_000);
        assert!(!params.debug);
    }
}
