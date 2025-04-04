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
            sample_rate: 16_000,
            debug: false,
        }
    }
}

#[derive(Debug, Default)]
pub struct TimeStamp {
    pub start: i64,
    pub end: i64,
}

impl std::fmt::Display for TimeStamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[start:{:08}, end:{:08}]", self.start, self.end)
    }
}
