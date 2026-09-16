use std::io::ErrorKind::UnexpectedEof;
use std::{fs::File, path::Path};

use anyhow::Context;
use log::{info, warn};
use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::{resampler::resample, utils::VAD_SAMPLE_RATE};

pub fn load_samples_from_audio_file(path: impl AsRef<Path>) -> anyhow::Result<Vec<f32>> {
    let path = path.as_ref();
    let file = Box::new(
        File::open(path)
            .with_context(|| format!("failed to open audio file {}", path.display()))?,
    );
    let mss = MediaSourceStream::new(file, Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }

    // Use the default options when reading and decoding.
    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: AudioDecoderOptions = Default::default();

    // Probe the media source stream for a format.
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, format_opts, metadata_opts)
        .with_context(|| format!("failed to detect audio format for {}", path.display()))?;

    // Get the default track.
    let track = format
        .default_track(TrackType::Audio)
        .context("audio file contains no supported audio track")?;
    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .context("audio track has no audio codec parameters")?
        .clone();
    let track_id = track.id;

    // Get the sample_rate of the track.
    let sample_rate = codec_params
        .sample_rate
        .context("audio track has no sample-rate information")?;

    // Check if the track has stereo channels.
    let channels = codec_params
        .channels
        .as_ref()
        .context("audio track has no channel information")?;
    if channels.count() > 1 {
        warn!("Stereo channels detected, will be converted to mono");
    }
    if channels.count() > 2 {
        return Err(anyhow::anyhow!(
            "unsupported channel count {}; only mono and stereo are supported",
            channels.count()
        ));
    }
    let is_stereo = channels.count() == 2;

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&codec_params, &decoder_opts)
        .context("unsupported audio codec")?;
    let mut samples: Vec<f32> = Vec::new();

    loop {
        // Get the next packet from the format reader.
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(err) => {
                if let symphonia::core::errors::Error::IoError(io_err) = &err {
                    if io_err.kind() == UnexpectedEof {
                        break;
                    }
                }

                return Err(err).context("failed to read the next audio packet");
            }
        };

        // If the packet does not belong to the selected track, skip it.
        if packet.track_id != track_id {
            continue;
        }

        // Decode the packet into audio samples, ignoring any decode errors.
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let old_len = samples.len();
                samples.resize(old_len + audio_buf.samples_interleaved(), f32::MID);
                audio_buf.copy_to_slice_interleaved(&mut samples[old_len..]);
            }
            Err(Error::DecodeError(error)) => {
                warn!("Skipping undecodable audio packet: {error}");
            }
            Err(error) => return Err(error).context("failed to decode audio packet"),
        }
    }

    if samples.is_empty() {
        return Err(anyhow::anyhow!("No samples found in the audio file"));
    }

    // If it's stereo, convert to mono.
    if is_stereo {
        samples = samples
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| chunk.iter().sum::<f32>() / 2_f32)
            .collect::<Vec<_>>();
    }

    if sample_rate as usize != VAD_SAMPLE_RATE {
        info!(
            "Sample rate mismatch: expected {}, got {}, resampling...",
            VAD_SAMPLE_RATE, sample_rate
        );

        samples = resample(&samples, sample_rate as usize, VAD_SAMPLE_RATE)?;
    }

    Ok(samples)
}
