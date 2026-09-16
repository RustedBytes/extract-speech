//! FSMN-VAD feature extraction.

use std::{fs, path::Path};

use anyhow::{Context, Result};
use kaldi_native_fbank::{
    fbank::{FbankComputer, FbankOptions},
    online::{FeatureComputer, OnlineFeature},
};

pub const FEATURE_DIM: usize = 400;
const N_MELS: usize = 80;
const LFR_M: usize = 5;
const SAMPLE_RATE: f32 = 16_000.0;

pub struct FsmnVadFrontend {
    means: Vec<f32>,
    scales: Vec<f32>,
}

impl FsmnVadFrontend {
    pub fn from_model_path(model_path: &Path) -> Result<Self> {
        let model_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
        let cmvn_path = ["vad.mvn", "am.mvn"]
            .into_iter()
            .map(|name| model_dir.join(name))
            .find(|path| path.is_file())
            .with_context(|| {
                format!(
                    "FSMN-VAD requires vad.mvn (or am.mvn) next to {}",
                    model_path.display()
                )
            })?;
        let contents = fs::read_to_string(&cmvn_path)
            .with_context(|| format!("failed to read CMVN file {}", cmvn_path.display()))?;
        let means = parse_coefficients(&contents, "<AddShift>")?;
        let scales = parse_coefficients(&contents, "<Rescale>")?;
        anyhow::ensure!(
            means.len() == FEATURE_DIM && scales.len() == FEATURE_DIM,
            "FSMN-VAD CMVN file {} contains {} means and {} scales; expected {FEATURE_DIM} of each",
            cmvn_path.display(),
            means.len(),
            scales.len()
        );

        Ok(Self { means, scales })
    }

    pub fn extract(&self, waveform: &[f32]) -> Result<Vec<f32>> {
        let mut options = FbankOptions::default();
        options.frame_opts.samp_freq = SAMPLE_RATE;
        options.frame_opts.dither = 0.0;
        options.frame_opts.window_type = "hamming".to_owned();
        options.frame_opts.frame_shift_ms = 10.0;
        options.frame_opts.frame_length_ms = 25.0;
        options.frame_opts.snip_edges = true;
        options.mel_opts.num_bins = N_MELS;
        options.mel_opts.debug_mel = false;
        options.use_energy = false;
        options.energy_floor = 0.0;

        let computer = FbankComputer::new(options).map_err(anyhow::Error::msg)?;
        let mut fbank = OnlineFeature::new(FeatureComputer::Fbank(computer));
        let scaled_waveform: Vec<f32> = waveform.iter().map(|sample| sample * 32_768.0).collect();
        fbank.accept_waveform(SAMPLE_RATE, &scaled_waveform);

        if fbank.features.is_empty() {
            return Ok(Vec::new());
        }

        let mut features = apply_lfr(&fbank.features);
        for frame in features.as_chunks_mut::<FEATURE_DIM>().0 {
            for (index, value) in frame.iter_mut().enumerate() {
                *value = (*value + self.means[index]) * self.scales[index];
            }
        }
        Ok(features)
    }
}

fn parse_coefficients(contents: &str, marker: &str) -> Result<Vec<f32>> {
    let mut lines = contents.lines();
    while let Some(line) = lines.next() {
        if line.split_whitespace().next() != Some(marker) {
            continue;
        }

        let coefficients = lines
            .next()
            .with_context(|| format!("CMVN marker {marker} has no coefficient line"))?;
        let mut values = Vec::new();
        let mut inside_brackets = false;
        for token in coefficients.split_whitespace() {
            if token == "[" {
                inside_brackets = true;
            } else if token == "]" {
                break;
            } else if inside_brackets {
                values.push(token.parse::<f32>().with_context(|| {
                    format!("invalid CMVN coefficient {token:?} after {marker}")
                })?);
            }
        }
        anyhow::ensure!(
            !values.is_empty(),
            "CMVN marker {marker} has no coefficients"
        );
        return Ok(values);
    }

    Err(anyhow::anyhow!("CMVN file does not contain {marker}"))
}

fn apply_lfr(inputs: &[Vec<f32>]) -> Vec<f32> {
    let frame_count = inputs.len();
    let mut output = Vec::with_capacity(frame_count * FEATURE_DIM);

    for output_frame in 0..frame_count {
        for lfr_index in 0..LFR_M {
            let source_index = output_frame
                .saturating_add(lfr_index)
                .saturating_sub((LFR_M - 1) / 2)
                .min(frame_count - 1);
            output.extend_from_slice(&inputs[source_index]);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kaldi_cmvn_coefficients() {
        let cmvn = "<AddShift> 2 2\n<LearnRateCoef> 0 [ -1.5 2.25 ]\n<Rescale> 2 2\n<LearnRateCoef> 0 [ 0.5 4 ]\n";

        assert_eq!(
            parse_coefficients(cmvn, "<AddShift>").unwrap(),
            [-1.5, 2.25]
        );
        assert_eq!(parse_coefficients(cmvn, "<Rescale>").unwrap(), [0.5, 4.0]);
    }

    #[test]
    fn lfr_repeats_edge_frames() {
        let inputs = vec![vec![1.0; N_MELS], vec![2.0; N_MELS], vec![3.0; N_MELS]];
        let output = apply_lfr(&inputs);

        assert_eq!(output.len(), inputs.len() * FEATURE_DIM);
        assert_eq!(output[0], 1.0);
        assert_eq!(output[N_MELS], 1.0);
        assert_eq!(output[2 * N_MELS], 1.0);
        assert_eq!(output[3 * N_MELS], 2.0);
        assert_eq!(output[4 * N_MELS], 3.0);
        assert_eq!(output[FEATURE_DIM + 4 * N_MELS], 3.0);
    }
}
