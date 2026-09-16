use std::io::ErrorKind::UnexpectedEof;
use std::{fs::File, path::PathBuf};

// use multiversion::multiversion;

use log::{info, warn};
use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::resampler::resample;

// #[multiversion(targets("x86_64+avx", "aarch64+neon"))]
pub fn load_samples_from_audio_file(path: PathBuf) -> Result<Vec<f32>, anyhow::Error> {
    let file = Box::new(File::open(path.clone()).unwrap());
    let mss = MediaSourceStream::new(file, Default::default());

    let hint = Hint::new();

    // Use the default options when reading and decoding.
    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: AudioDecoderOptions = Default::default();

    // Probe the media source stream for a format.
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, format_opts, metadata_opts)
        .unwrap();

    // Get the default track.
    let track = format
        .default_track(TrackType::Audio)
        .expect("no supported audio tracks");
    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .expect("audio track has no codec parameters")
        .clone();
    let track_id = track.id;

    // Get the sample_rate of the track.
    let sample_rate = codec_params.sample_rate.unwrap_or(0);

    // Check if the track has stereo channels.
    match &codec_params.channels {
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

    let is_stereo = codec_params
        .channels
        .as_ref()
        .is_some_and(|channels| channels.count() > 1);

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&codec_params, &decoder_opts)
        .expect("unsupported codec");
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

                info!("Error loading next packet: {}", err);

                break;
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
            .as_chunks::<2>()
            .0
            .iter()
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
