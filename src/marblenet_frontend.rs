use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

pub const N_MELS: usize = 80;
pub const OUTPUT_FRAME_SAMPLES: usize = 320;

const SAMPLE_RATE: f64 = 16_000.0;
const N_FFT: usize = 512;
const FFT_BINS: usize = N_FFT / 2 + 1;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const PREEMPHASIS_ALPHA: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8;

pub struct MarbleNetFrontend {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    mel_filterbank: Vec<f32>,
}

impl MarbleNetFrontend {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        Self {
            fft: planner.plan_fft_forward(N_FFT),
            window: make_window(),
            mel_filterbank: make_mel_filterbank(),
        }
    }

    pub fn extract(&self, waveform: &[f32]) -> anyhow::Result<(Vec<f32>, usize)> {
        if waveform.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let mut emphasized = Vec::with_capacity(waveform.len());
        emphasized.push(waveform[0]);
        emphasized.extend(
            waveform[1..]
                .iter()
                .zip(waveform)
                .map(|(&sample, &previous)| sample - PREEMPHASIS_ALPHA * previous),
        );

        // NeMo uses a centered STFT, so a signal of N samples produces
        // floor(N / hop) + 1 feature frames.
        let frame_count = waveform.len() / HOP_LENGTH + 1;
        let mut features = vec![0.0; N_MELS * frame_count];
        let mut fft_buffer = vec![Complex32::new(0.0, 0.0); N_FFT];
        let window_offset = (N_FFT - WIN_LENGTH) / 2;

        for frame in 0..frame_count {
            fft_buffer.fill(Complex32::new(0.0, 0.0));
            let padded_start = frame * HOP_LENGTH;
            for window_index in 0..WIN_LENGTH {
                let padded_index = padded_start + window_offset + window_index;
                let source_index = reflected_index(
                    padded_index as isize - (N_FFT / 2) as isize,
                    emphasized.len(),
                );
                fft_buffer[window_offset + window_index] =
                    Complex32::new(emphasized[source_index] * self.window[window_index], 0.0);
            }

            self.fft.process(&mut fft_buffer);
            for mel in 0..N_MELS {
                let mut energy = 0.0_f32;
                for (frequency, bin) in fft_buffer.iter().take(FFT_BINS).enumerate() {
                    energy += bin.norm_sqr() * self.mel_filterbank[mel * FFT_BINS + frequency];
                }
                features[mel * frame_count + frame] = (energy + LOG_GUARD).ln();
            }
        }

        anyhow::ensure!(
            features.iter().all(|value| value.is_finite()),
            "MarbleNet frontend produced non-finite features"
        );
        Ok((features, frame_count))
    }
}

fn reflected_index(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }

    let period = 2 * (len - 1);
    let wrapped = index.rem_euclid(period as isize) as usize;
    if wrapped < len {
        wrapped
    } else {
        period - wrapped
    }
}

fn make_window() -> Vec<f32> {
    (0..WIN_LENGTH)
        .map(|index| {
            let phase = 2.0 * std::f32::consts::PI * index as f32 / (WIN_LENGTH - 1) as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

fn make_mel_filterbank() -> Vec<f32> {
    let min_mel = hz_to_slaney_mel(0.0);
    let max_mel = hz_to_slaney_mel(SAMPLE_RATE / 2.0);
    let mel_frequencies: Vec<f64> = (0..N_MELS + 2)
        .map(|index| {
            let mel = min_mel + (max_mel - min_mel) * index as f64 / (N_MELS + 1) as f64;
            slaney_mel_to_hz(mel)
        })
        .collect();

    let mut filterbank = vec![0.0; N_MELS * FFT_BINS];
    for mel in 0..N_MELS {
        let lower = mel_frequencies[mel];
        let center = mel_frequencies[mel + 1];
        let upper = mel_frequencies[mel + 2];
        let normalization = 2.0 / (upper - lower);

        for frequency in 0..FFT_BINS {
            let hz = SAMPLE_RATE * frequency as f64 / N_FFT as f64;
            let rising = (hz - lower) / (center - lower);
            let falling = (upper - hz) / (upper - center);
            filterbank[mel * FFT_BINS + frequency] =
                (rising.min(falling).max(0.0) * normalization) as f32;
        }
    }
    filterbank
}

fn hz_to_slaney_mel(hz: f64) -> f64 {
    const LINEAR_SCALE: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / LINEAR_SCALE;
    let log_step = 6.4_f64.ln() / 27.0;

    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / log_step
    } else {
        hz / LINEAR_SCALE
    }
}

fn slaney_mel_to_hz(mel: f64) -> f64 {
    const LINEAR_SCALE: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / LINEAR_SCALE;
    let log_step = 6.4_f64.ln() / 27.0;

    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (log_step * (mel - MIN_LOG_MEL)).exp()
    } else {
        LINEAR_SCALE * mel
    }
}

pub fn speech_probability(logits: &[f32]) -> anyhow::Result<f32> {
    anyhow::ensure!(
        logits.len() >= 2,
        "MarbleNet returned fewer than two classes"
    );
    let difference = logits[1] - logits[0];
    Ok(if difference >= 0.0 {
        1.0 / (1.0 + (-difference).exp())
    } else {
        let exponential = difference.exp();
        exponential / (1.0 + exponential)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_matches_centered_stft_padding() {
        let indexes: Vec<usize> = (-4..8).map(|index| reflected_index(index, 4)).collect();
        assert_eq!(indexes, [2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]);
    }

    #[test]
    fn window_matches_nemo_non_periodic_hann() {
        let window = make_window();
        assert_eq!(window.len(), WIN_LENGTH);
        assert_eq!(window[0], 0.0);
        assert_eq!(window[WIN_LENGTH - 1], 0.0);
        assert!((window[1] - 0.000_061_988_83).abs() < 1e-9);
        assert!(window[WIN_LENGTH / 2] > 0.999);
    }

    #[test]
    fn mel_filterbank_matches_nemo_reference_values() {
        let filterbank = make_mel_filterbank();
        assert_eq!(filterbank.len(), N_MELS * FFT_BINS);
        assert!((filterbank[1] - 0.022_534_56).abs() < 1e-7);
        assert!((filterbank[2] - 0.008_637_710_5).abs() < 1e-7);
    }

    #[test]
    fn frontend_produces_finite_feature_frames() {
        let frontend = MarbleNetFrontend::new();
        let waveform: Vec<f32> = (0..16_000)
            .map(|index| (index as f32 * 0.031).sin())
            .collect();
        let (features, frame_count) = frontend.extract(&waveform).unwrap();

        assert_eq!(frame_count, 101);
        assert_eq!(features.len(), N_MELS * frame_count);
        assert!(features.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        assert_eq!(speech_probability(&[-1_000.0, 1_000.0]).unwrap(), 1.0);
        assert_eq!(speech_probability(&[1_000.0, -1_000.0]).unwrap(), 0.0);
        assert_eq!(speech_probability(&[0.0, 0.0]).unwrap(), 0.5);
    }
}
