//! Runtime-independent TEN VAD feature extraction.

use std::{f32::consts::PI, sync::Arc};

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use super::pitch_est::PitchEstimator;

pub const FRAME_SAMPLES: usize = 256;
pub const FEATURE_DIM: usize = 41;
pub const CONTEXT_FRAMES: usize = 3;
pub const HIDDEN_DIM: usize = 64;
pub const HIDDEN_COUNT: usize = 4;

const FFT_SIZE: usize = 1_024;
const WINDOW_SIZE: usize = 768;
const MEL_COUNT: usize = 40;
const EPSILON: f32 = 1e-20;
const PRE_EMPHASIS: f32 = 0.97;

const FEATURE_MEANS: [f32; FEATURE_DIM] = [
    -8.198_236,
    -6.265_716_6,
    -5.483_818_5,
    -4.758_691_3,
    -4.417_089,
    -4.142_893,
    -3.912_850_4,
    -3.845_928,
    -3.657_090_4,
    -3.723_418_7,
    -3.876_134_2,
    -3.843_891,
    -3.690_405_1,
    -3.756_065_8,
    -3.698_696_1,
    -3.650_463,
    -3.700_468_8,
    -3.567_321_3,
    -3.498_900_2,
    -3.477_807,
    -3.458_816,
    -3.444_923_9,
    -3.401_328_6,
    -3.306_261_3,
    -3.278_556_8,
    -3.233_250_9,
    -3.198_616,
    -3.204_526_4,
    -3.208_798_6,
    -3.257_838,
    -3.381_376_7,
    -3.534_021_4,
    -3.640_868,
    -3.726_858_9,
    -3.773_731,
    -3.804_667_2,
    -3.832_901,
    -3.871_120_5,
    -3.990_593,
    -4.480_289_5,
    92.356_9,
];

const FEATURE_STDS: [f32; FEATURE_DIM] = [
    5.166_064,
    4.977_209_6,
    4.698_896,
    4.630_621_4,
    4.634_348,
    4.641_156,
    4.640_676_5,
    4.666_367,
    4.650_534_6,
    4.640_021,
    4.637_4,
    4.620_099,
    4.596_316_3,
    4.562_655,
    4.554_361,
    4.566_91,
    4.562_49,
    4.562_412_7,
    4.585_299_5,
    4.600_179_7,
    4.592_845_4,
    4.585_922_7,
    4.583_496_6,
    4.626_092_4,
    4.626_958,
    4.626_29,
    4.637_006,
    4.683_016,
    4.726_814,
    4.734_289_6,
    4.753_227,
    4.849_722_4,
    4.869_435,
    4.884_483,
    4.921_327,
    4.959_212_3,
    4.996_619,
    5.044_823_6,
    5.072_216,
    5.096_439_4,
    115.213_69,
];

pub struct TenVadFrontend {
    feature_buffer: [[f32; FEATURE_DIM]; CONTEXT_FRAMES],
    pre_emphasis_previous: f32,
    mel_filters: Vec<Vec<f32>>,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    fft_buffer: Vec<Complex32>,
    stft_queue: Vec<f32>,
    pitch: PitchEstimator,
}

impl TenVadFrontend {
    #[must_use]
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        Self {
            feature_buffer: [[0.0; FEATURE_DIM]; CONTEXT_FRAMES],
            pre_emphasis_previous: 0.0,
            mel_filters: mel_filters(),
            window: (0..WINDOW_SIZE)
                .map(|index| {
                    #[allow(clippy::cast_precision_loss)] // Bounded DSP window index.
                    let phase = 2.0 * PI * index as f32 / WINDOW_SIZE as f32;
                    0.5 - 0.5 * phase.cos()
                })
                .collect(),
            fft: planner.plan_fft_forward(FFT_SIZE),
            fft_buffer: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            stft_queue: vec![0.0; WINDOW_SIZE],
            pitch: PitchEstimator::new(),
        }
    }

    pub fn reset(&mut self) {
        self.feature_buffer = [[0.0; FEATURE_DIM]; CONTEXT_FRAMES];
        self.pre_emphasis_previous = 0.0;
        self.fft_buffer.fill(Complex32::new(0.0, 0.0));
        self.stft_queue.fill(0.0);
        self.pitch.reset();
    }

    /// Produces the three-frame normalized model input.
    ///
    /// # Errors
    ///
    /// Returns an error unless `audio_frame` contains exactly 256 normalized samples.
    pub fn extract(&mut self, audio_frame: &[f32]) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            audio_frame.len() == FRAME_SAMPLES,
            "TEN VAD expects exactly {FRAME_SAMPLES} audio samples, received {}",
            audio_frame.len()
        );
        let pcm: Vec<f32> = audio_frame
            .iter()
            .copied()
            .map(sample_to_i16)
            .map(f32::from)
            .collect();
        let features = self.extract_frame(&pcm);
        self.feature_buffer.rotate_left(1);
        self.feature_buffer[CONTEXT_FRAMES - 1] = features;
        Ok(self.feature_buffer.concat())
    }

    fn extract_frame(&mut self, pcm: &[f32]) -> [f32; FEATURE_DIM] {
        let mut emphasized = [0.0; FRAME_SAMPLES];
        emphasized[0] = pcm[0] - PRE_EMPHASIS * self.pre_emphasis_previous;
        for index in 1..FRAME_SAMPLES {
            emphasized[index] = pcm[index] - PRE_EMPHASIS * pcm[index - 1];
        }
        self.pre_emphasis_previous = pcm[FRAME_SAMPLES - 1];

        self.stft_queue.copy_within(FRAME_SAMPLES.., 0);
        self.stft_queue[WINDOW_SIZE - FRAME_SAMPLES..].copy_from_slice(&emphasized);
        self.fft_buffer.fill(Complex32::new(0.0, 0.0));
        for (output, (sample, window)) in self
            .fft_buffer
            .iter_mut()
            .zip(self.stft_queue.iter().zip(&self.window))
        {
            output.re = sample * window;
        }
        self.fft.process(&mut self.fft_buffer);

        let mut power: Vec<f32> = self.fft_buffer[..=FFT_SIZE / 2]
            .iter()
            .map(Complex32::norm_sqr)
            .collect();
        let pitch = self.pitch.process(pcm, &power);
        let normalization = 32_768.0_f32.powi(2);
        for value in &mut power {
            *value /= normalization;
        }

        let mut features = [0.0; FEATURE_DIM];
        for (mel_index, filter) in self.mel_filters.iter().enumerate() {
            features[mel_index] = (filter
                .iter()
                .zip(&power)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                + EPSILON)
                .ln();
        }
        features[MEL_COUNT] = pitch;
        for (index, feature) in features.iter_mut().enumerate() {
            *feature = (*feature - FEATURE_MEANS[index]) / (FEATURE_STDS[index] + EPSILON);
        }
        features
    }
}

impl Default for TenVadFrontend {
    fn default() -> Self {
        Self::new()
    }
}

fn mel_filters() -> Vec<Vec<f32>> {
    let bins = FFT_SIZE / 2 + 1;
    let low_mel = 2_595.0_f32 * (1.0_f32 + 0.0_f32 / 700.0_f32).log10();
    let high_mel = 2_595.0_f32 * (1.0_f32 + 8_000.0_f32 / 700.0_f32).log10();
    let points: Vec<usize> = (0..=MEL_COUNT + 1)
        .map(|index| {
            #[allow(clippy::cast_precision_loss)] // Bounded mel filter index.
            let mel = low_mel + (high_mel - low_mel) * index as f32 / (MEL_COUNT + 1) as f32;
            let hz = 700.0 * (10.0_f32.powf(mel / 2_595.0) - 1.0);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss,
                clippy::cast_sign_loss
            )] // Fixed FFT size and non-negative frequency bound the conversion.
            let bin = ((FFT_SIZE + 1) as f32 * hz / 16_000.0) as usize;
            bin
        })
        .collect();
    let mut filters = vec![vec![0.0; bins]; MEL_COUNT];
    for index in 0..MEL_COUNT {
        for (bin, value) in filters[index]
            .iter_mut()
            .enumerate()
            .take(points[index + 1])
            .skip(points[index])
        {
            #[allow(clippy::cast_precision_loss)] // Bounded mel-bin interpolation.
            let weight = (bin - points[index]) as f32 / (points[index + 1] - points[index]) as f32;
            *value = weight;
        }
        for (bin, value) in filters[index]
            .iter_mut()
            .enumerate()
            .take(points[index + 2])
            .skip(points[index + 1])
        {
            #[allow(clippy::cast_precision_loss)] // Bounded mel-bin interpolation.
            let weight =
                (points[index + 2] - bin) as f32 / (points[index + 2] - points[index + 1]) as f32;
            *value = weight;
        }
    }
    filters
}

#[allow(clippy::cast_possible_truncation)]
fn sample_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * 32_768.0)
        .round()
        .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_normalized_samples_to_pcm() {
        assert_eq!(sample_to_i16(-1.0), i16::MIN);
        assert_eq!(sample_to_i16(0.0), 0);
        assert_eq!(sample_to_i16(1.0), i16::MAX);
        assert_eq!(sample_to_i16(f32::INFINITY), i16::MAX);
        assert_eq!(sample_to_i16(f32::NEG_INFINITY), i16::MIN);
    }

    #[test]
    fn produces_finite_context_features() {
        let mut frontend = TenVadFrontend::new();
        let features = frontend.extract(&[0.0; FRAME_SAMPLES]).unwrap();
        assert_eq!(features.len(), CONTEXT_FRAMES * FEATURE_DIM);
        assert!(features.iter().all(|value| value.is_finite()));
    }
}
