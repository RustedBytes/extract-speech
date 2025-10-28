use std::num::NonZeroUsize;

use log::info;

pub fn resample(in_samples: &[f32], sr_in: usize, sr_out: usize) -> anyhow::Result<Vec<f32>> {
    info!("Resampling from {} to {}", sr_in, sr_out);

    let quality = fixed_resample::ResampleQuality::High;

    let mut resampler = fixed_resample::FixedResampler::<f32, 1>::new(
        NonZeroUsize::new(1).unwrap(),
        sr_in as u32,
        sr_out as u32,
        quality,
        true, // interleaved
    );

    let output_frames = resampler.out_alloc_frames(in_samples.len() as u64);
    let mut out_samples: Vec<f32> = Vec::with_capacity(output_frames as usize);

    resampler.process_interleaved(
        in_samples,
        // This method gets called whenever there is new resampled data.
        |data| {
            out_samples.extend_from_slice(data);
        },
        // Whether or not this is the last (or only) packet of data that
        // will be resampled. This ensures that any leftover samples in
        // the internal resampler are flushed to the output.
        Some(fixed_resample::LastPacketInfo {
            // Let the resampler know that we want an exact number of output
            // frames. Otherwise the resampler may add extra padded zeros
            // to the end.
            desired_output_frames: Some(output_frames),
        }),
        // Trim the padded zeros at the beginning introduced by the internal
        // resampler.
        true, // trim_delay
    );

    Ok(out_samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_upsample() {
        // Test upsampling from 8kHz to 16kHz
        let input: Vec<f32> = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        let sr_in = 8000;
        let sr_out = 16000;

        let result = resample(&input, sr_in, sr_out);
        assert!(result.is_ok());

        let output = result.unwrap();
        // When upsampling 2x, we should get approximately 2x the samples
        assert!(output.len() >= input.len() * 2 - 2);
        assert!(output.len() <= input.len() * 2 + 2);
    }

    #[test]
    fn test_resample_downsample() {
        // Test downsampling from 48kHz to 16kHz
        let input: Vec<f32> = (0..480).map(|i| (i as f32 * 0.01).sin()).collect();
        let sr_in = 48000;
        let sr_out = 16000;

        let result = resample(&input, sr_in, sr_out);
        assert!(result.is_ok());

        let output = result.unwrap();
        // When downsampling 3x, we should get approximately 1/3 the samples
        assert!(output.len() >= input.len() / 3 - 2);
        assert!(output.len() <= input.len() / 3 + 2);
    }

    #[test]
    fn test_resample_same_rate() {
        // Test "resampling" at the same rate (should still work)
        let input: Vec<f32> = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let sr = 16000;

        let result = resample(&input, sr, sr);
        assert!(result.is_ok());

        let output = result.unwrap();
        // Output should be approximately the same length
        assert!((output.len() as i32 - input.len() as i32).abs() <= 2);
    }

    #[test]
    fn test_resample_sine_wave() {
        // Test with a simple sine wave pattern
        let sample_rate_in = 16000;
        let sample_rate_out = 8000;
        let duration_secs = 0.1;
        let frequency = 440.0; // A4 note

        let num_samples = (sample_rate_in as f32 * duration_secs) as usize;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate_in as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let result = resample(&input, sample_rate_in, sample_rate_out);
        assert!(result.is_ok());

        let output = result.unwrap();
        // Check output is approximately the right length
        let expected_len = (sample_rate_out as f32 * duration_secs) as usize;
        assert!((output.len() as i32 - expected_len as i32).abs() <= 5);
    }

    #[test]
    fn test_resample_empty_input() {
        let input: Vec<f32> = vec![];
        let result = resample(&input, 16000, 8000);

        assert!(result.is_ok());
        let output = result.unwrap();
        // Empty input may produce a small number of samples due to resampler padding
        assert!(
            output.len() <= 2,
            "Empty input should produce minimal or no output"
        );
    }

    #[test]
    fn test_resample_single_sample() {
        let input: Vec<f32> = vec![0.5];
        let result = resample(&input, 16000, 8000);

        assert!(result.is_ok());
        let output = result.unwrap();
        // Should produce at least one sample
        assert!(!output.is_empty());
    }

    #[test]
    fn test_resample_preserves_dc_offset() {
        // A DC signal (constant value) should remain approximately constant after resampling
        let input: Vec<f32> = vec![1.0; 100];
        let result = resample(&input, 16000, 8000);

        assert!(result.is_ok());
        let output = result.unwrap();

        // Calculate average to check if DC component is preserved
        let avg: f32 = output.iter().sum::<f32>() / output.len() as f32;
        assert!(
            (avg - 1.0).abs() < 0.15,
            "Average {} differs too much from 1.0",
            avg
        );

        // Check that most values are reasonably close to 1.0
        let close_count = output.iter().filter(|&&s| (s - 1.0).abs() < 0.3).count();
        let close_ratio = close_count as f32 / output.len() as f32;
        assert!(
            close_ratio > 0.8,
            "Only {:.1}% of samples are close to 1.0",
            close_ratio * 100.0
        );
    }

    #[test]
    fn test_resample_44100_to_16000() {
        // Common use case: CD quality to speech recognition rate
        let input: Vec<f32> = (0..4410).map(|i| (i as f32 / 100.0).sin()).collect();
        let result = resample(&input, 44100, 16000);

        assert!(result.is_ok());
        let output = result.unwrap();

        // 44100 -> 16000 is approximately a 2.75625x reduction
        let expected_len = (input.len() as f32 * 16000.0 / 44100.0) as usize;
        assert!((output.len() as i32 - expected_len as i32).abs() <= 10);
    }
}
