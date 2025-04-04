use std::num::NonZeroUsize;

use fixed_resample;

pub fn resample(in_samples: &[f32], sr_in: usize, sr_out: usize) -> anyhow::Result<Vec<f32>> {
    println!("Resampling from {} to {}", sr_in, sr_out);

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
        &in_samples,
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
            desired_output_frames: Some(output_frames as u64),
        }),
        // Trim the padded zeros at the beginning introduced by the internal
        // resampler.
        true, // trim_delay
    );

    Ok(out_samples)
}
