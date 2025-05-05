use std::io::ErrorKind::UnexpectedEof;
use std::{fs::File, path::PathBuf};

// use multiversion::multiversion;

use log::{info, warn};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, io::MediaSourceStream, probe::Hint,
};

use crate::resampler::resample;

// #[multiversion(targets("x86_64+avx", "aarch64+neon"))]
pub fn load_samples_from_audio_file(path: PathBuf) -> Result<Vec<f32>, anyhow::Error> {
    let file = Box::new(File::open(path.clone()).unwrap());
    let mss = MediaSourceStream::new(file, Default::default());

    let hint = Hint::new();

    // Use the default options when reading and decoding.
    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: DecoderOptions = Default::default();

    // Probe the media source stream for a format.
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .unwrap();

    // Get the format reader yielded by the probe operation.
    let mut format = probed.format;

    // Get the default track.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    // Get the sample_rate of the track.
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);

    // Check if the track has stereo channels.
    match track.codec_params.channels {
        Some(channels) => {
            if channels.count() > 1 {
                warn!("Stereo channels detected, will be converted to mono");
            }

            if channels.count() > 2 {
                return Err(anyhow::anyhow!(
                    "Unsupported number of channels: {}. Only mono and stereo are supported.",
                    channels.count()
                ));
            }
        }
        None => {
            return Err(anyhow::anyhow!("No channel information available"));
        }
    }

    let is_stereo = track
        .codec_params
        .channels
        .is_some_and(|channels| channels.count() > 1);

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .expect("unsupported codec");

    // Store the track identifier, we'll use it to filter packets.
    let track_id = track.id;

    let mut sample_buf = None;
    let mut samples: Vec<f32> = Vec::new();

    loop {
        // Get the next packet from the format reader.
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(err) => {
                if let symphonia::core::errors::Error::IoError(io_err) = &err {
                    if io_err.kind() == UnexpectedEof {
                        break;
                    }
                }

                info!("Error loading next packet: {}", err);

                break;
            }
        };

        // If the packet does not belong to the selected track, skip it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples, ignoring any decode errors.
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                // The decoded audio samples may now be accessed via the audio buffer if per-channel
                // slices of samples in their native decoded format is desired. Use-cases where
                // the samples need to be accessed in an interleaved order or converted into
                // another sample format, or a byte buffer is required, are covered by copying the
                // audio buffer into a sample buffer or raw sample buffer, respectively. In the
                // example below, we will copy the audio buffer into a sample buffer in an
                // interleaved order while also converting to a f32 sample format.

                // If this is the *first* decoded packet, create a sample buffer matching the
                // decoded audio buffer format.
                if sample_buf.is_none() {
                    // Get the audio buffer specification.
                    let spec = *audio_buf.spec();

                    // Get the capacity of the decoded buffer. Note: This is capacity, not length!
                    let duration = audio_buf.capacity() as u64;

                    // Create the f32 sample buffer.
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                // Copy the decoded audio buffer into the sample buffer in an interleaved format.
                if let Some(buf) = &mut sample_buf {
                    buf.copy_interleaved_ref(audio_buf);

                    // Append the samples to the samples variable
                    samples.extend_from_slice(buf.samples());
                }
            }
            Err(Error::DecodeError(_)) => (),
            Err(_) => break,
        }
    }

    if samples.is_empty() {
        return Err(anyhow::anyhow!("No samples found in the audio file"));
    }

    // If it's strereo, convert to mono
    if is_stereo {
        samples = samples
            .chunks_exact(2)
            .map(|chunk| chunk.iter().sum::<f32>() / 2_f32)
            .collect::<Vec<_>>();
    }

    const REQUIRED_SAMPLE_RATE: u32 = 16_000;
    if sample_rate != REQUIRED_SAMPLE_RATE {
        info!(
            "Sample rate mismatch: expected {}, got {}, resampling...",
            REQUIRED_SAMPLE_RATE, sample_rate
        );

        samples = resample(
            &samples,
            sample_rate as usize,
            REQUIRED_SAMPLE_RATE as usize,
        )?;
    }

    Ok(samples)
}
