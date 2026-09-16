//! TEN VAD adapter.

use std::path::Path;

use anyhow::Context;
use log::debug;

use crate::vad_iter::VadModel;

pub const FRAME_SAMPLES: usize = 256;

pub struct TenVad {
    model: ten_vad_rs::TenVad,
}

impl TenVad {
    /// Loads a TEN VAD model.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is not UTF-8 or model initialization fails.
    pub fn new(model_path: impl AsRef<Path>, _debug: bool) -> anyhow::Result<Self> {
        let model_path = model_path
            .as_ref()
            .to_str()
            .context("TEN VAD model path is not valid UTF-8")?;
        let model = ten_vad_rs::TenVad::new(model_path, ten_vad_rs::TARGET_SAMPLE_RATE)
            .context("failed to initialize TEN VAD")?;

        Ok(Self { model })
    }
}

impl VadModel for TenVad {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.model.reset();
        Ok(())
    }

    fn probability(&mut self, audio_frame: &[f32]) -> anyhow::Result<f32> {
        anyhow::ensure!(
            audio_frame.len() == FRAME_SAMPLES,
            "TEN VAD expects exactly {FRAME_SAMPLES} audio samples, received {}",
            audio_frame.len()
        );

        let pcm: Vec<i16> = audio_frame.iter().copied().map(sample_to_i16).collect();
        let probability = self.model.process_frame(&pcm)?;
        anyhow::ensure!(
            probability.is_finite(),
            "TEN VAD returned a non-finite probability"
        );
        let probability = probability.clamp(0.0, 1.0);

        debug!("TEN VAD speech probability: {probability:.6}");
        Ok(probability)
    }
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
}
