use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

pub const WINDOW_SAMPLES: usize = 3_200;
pub const HOP_SAMPLES: usize = 1_600;
pub const N_MELS: usize = 64;
pub const N_FRAMES: usize = 21;

const SAMPLE_RATE: f64 = 16_000.0;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const FFT_BINS: usize = N_FFT / 2 + 1;
const PREEMPHASIS_ALPHA: f32 = 0.97;
const EPS: f32 = 1e-5;

pub struct PulseVadFrontend {
    fft: Arc<dyn Fft<f32>>,
    hann_window: Vec<f32>,
    mel_filterbank: Vec<f32>,
}

impl PulseVadFrontend {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let hann_window = (0..WIN_LENGTH)
            .map(|index| {
                let phase = 2.0 * std::f32::consts::PI * index as f32 / WIN_LENGTH as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect();

        Self {
            fft,
            hann_window,
            mel_filterbank: make_mel_filterbank(),
        }
    }

    pub fn extract(&self, waveform: &[f32]) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            waveform.len() == WINDOW_SAMPLES,
            "PulseVAD expects exactly {WINDOW_SAMPLES} audio samples, received {}",
            waveform.len()
        );

        let mut emphasized = vec![0.0; WINDOW_SAMPLES];
        emphasized[0] = waveform[0];
        for index in 1..WINDOW_SAMPLES {
            emphasized[index] = waveform[index] - PREEMPHASIS_ALPHA * waveform[index - 1];
        }
        normalize(&mut emphasized);

        let mut power_spectra = vec![0.0; N_FRAMES * FFT_BINS];
        let mut fft_buffer = vec![Complex32::new(0.0, 0.0); N_FFT];
        let window_offset = (N_FFT - WIN_LENGTH) / 2;

        for frame in 0..N_FRAMES {
            fft_buffer.fill(Complex32::new(0.0, 0.0));
            let frame_start = frame * 160;
            for window_index in 0..WIN_LENGTH {
                let padded_index = frame_start + window_offset + window_index;
                let sample = reflected_sample(&emphasized, padded_index);
                fft_buffer[window_offset + window_index] =
                    Complex32::new(sample * self.hann_window[window_index], 0.0);
            }

            self.fft.process(&mut fft_buffer);
            for frequency in 0..FFT_BINS {
                power_spectra[frame * FFT_BINS + frequency] = fft_buffer[frequency].norm_sqr();
            }
        }

        let mut features = vec![0.0; N_MELS * N_FRAMES];
        for mel in 0..N_MELS {
            for frame in 0..N_FRAMES {
                let mut energy = 0.0_f32;
                for frequency in 0..FFT_BINS {
                    energy += power_spectra[frame * FFT_BINS + frequency]
                        * self.mel_filterbank[frequency * N_MELS + mel];
                }
                features[mel * N_FRAMES + frame] = (energy + EPS).ln();
            }
            normalize(&mut features[mel * N_FRAMES..(mel + 1) * N_FRAMES]);
        }

        Ok(features)
    }
}

impl Default for PulseVadFrontend {
    fn default() -> Self {
        Self::new()
    }
}

fn normalize(values: &mut [f32]) {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32;
    let denominator = variance.sqrt() + EPS;
    for value in values {
        *value = (*value - mean) / denominator;
    }
}

fn reflected_sample(samples: &[f32], padded_index: usize) -> f32 {
    let padding = N_FFT / 2;
    if padded_index < padding {
        samples[padding - padded_index]
    } else {
        let source_index = padded_index - padding;
        if source_index < samples.len() {
            samples[source_index]
        } else {
            samples[2 * (samples.len() - 1) - source_index]
        }
    }
}

fn make_mel_filterbank() -> Vec<f32> {
    let hz_to_mel = |hz: f64| 2_595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10_f64.powf(mel / 2_595.0) - 1.0);
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(SAMPLE_RATE / 2.0);
    let mel_points: Vec<f64> = (0..N_MELS + 2)
        .map(|index| {
            let mel = min_mel + (max_mel - min_mel) * index as f64 / (N_MELS + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    let mut filterbank = vec![0.0; FFT_BINS * N_MELS];
    for frequency in 0..FFT_BINS {
        let hz = (SAMPLE_RATE / 2.0) * frequency as f64 / (FFT_BINS - 1) as f64;
        for mel in 0..N_MELS {
            let down = (hz - mel_points[mel]) / (mel_points[mel + 1] - mel_points[mel]);
            let up = (mel_points[mel + 2] - hz) / (mel_points[mel + 2] - mel_points[mel + 1]);
            let triangle = down.min(up).max(0.0);
            let slaney_norm = 2.0 / (mel_points[mel + 2] - mel_points[mel]);
            filterbank[frequency * N_MELS + mel] = (triangle * slaney_norm) as f32;
        }
    }
    filterbank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_rejects_an_invalid_window_size() {
        let frontend = PulseVadFrontend::new();
        let error = frontend.extract(&[0.0; 100]).unwrap_err();
        assert!(error.to_string().contains("exactly 3200"));
    }

    #[test]
    fn silent_input_produces_finite_features() {
        let frontend = PulseVadFrontend::new();
        let features = frontend.extract(&[0.0; WINDOW_SAMPLES]).unwrap();

        assert_eq!(features.len(), N_MELS * N_FRAMES);
        assert!(features.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn each_mel_bin_is_normalized() {
        let frontend = PulseVadFrontend::new();
        let waveform: Vec<f32> = (0..WINDOW_SAMPLES)
            .map(|index| (index as f32 * 0.037).sin())
            .collect();
        let features = frontend.extract(&waveform).unwrap();

        for mel_bin in features.as_chunks::<N_FRAMES>().0 {
            let mean = mel_bin.iter().sum::<f32>() / N_FRAMES as f32;
            assert!(mean.abs() < 1e-4);
        }
    }

    #[test]
    fn mel_filterbank_matches_the_reference_values() {
        let filterbank = make_mel_filterbank();

        assert_eq!(filterbank.len(), FFT_BINS * N_MELS);
        assert!((filterbank[6 * N_MELS + 5] - 0.025_704_478).abs() < 1e-6);
    }
}
