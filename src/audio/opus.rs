//! Opus/Ogg encoding for CLI output.

use std::path::Path;

use anyhow::{Context, Result};

use crate::resampler::resample;

// This must be an allowed value among 120, 240, 480, 960, 1920, and 2880.
// Using a different value would result in a BadArg "invalid argument" error when calling encode.
// https://opus-codec.org/docs/opus_api-1.2/group__opus__encoder.html#ga4ae9905859cd241ef4bb5c59cd5e5309
const OPUS_ENCODER_FRAME_SIZE: usize = 960;
const OPUS_SAMPLE_RATE: u32 = 48000;
// const OPUS_ALLOWED_FRAME_SIZES: [usize; 6] = [120, 240, 480, 960, 1920, 2880];

/// See <https://www.opus-codec.org/docs/opusfile_api-0.4/structOpusHead.html>.
#[allow(unused)]
#[derive(Debug)]
struct OpusHeader {
    version: u8,
    channel_count: u8,
    pre_skip: u16,

    /// The sampling rate of the original input.
    ///
    /// All Opus audio is coded at 48 kHz, and should also be decoded at 48 kHz for playback (unless
    /// the target hardware does not support this sampling rate). However, this field may be used to
    /// resample the audio back to the original sampling rate, for example, when saving the output
    /// to a file.
    input_sample_rate: u32,
    output_gain: i16,
    mapping_family: u8,
}

fn write_opus_header<W: std::io::Write>(
    w: &mut W,
    channels: u8,
    sample_rate: u32,
    pre_skip: u16,
) -> std::io::Result<()> {
    use byteorder::WriteBytesExt;

    // https://wiki.xiph.org/OggOpus#ID_Header
    w.write_all(b"OpusHead")?;
    w.write_u8(1)?; // version
    w.write_u8(channels)?; // channel count
    w.write_u16::<byteorder::LittleEndian>(pre_skip)?; // pre-skip
    w.write_u32::<byteorder::LittleEndian>(sample_rate)?; //  sample-rate in Hz
    w.write_i16::<byteorder::LittleEndian>(0)?; // output gain Q7.8 in dB
    w.write_u8(0)?; // channel map
    Ok(())
}

fn write_opus_tags<W: std::io::Write>(w: &mut W) -> std::io::Result<()> {
    use byteorder::WriteBytesExt;

    // https://wiki.xiph.org/OggOpus#Comment_Header
    let vendor = "rust";
    w.write_all(b"OpusTags")?;
    w.write_u32::<byteorder::LittleEndian>(
        u32::try_from(vendor.len()).expect("static Opus vendor fits in u32"),
    )?; // vendor string length
    w.write_all(vendor.as_bytes())?; // vendor string, UTF8 encoded
    w.write_u32::<byteorder::LittleEndian>(0u32)?; // number of tags
    Ok(())
}

// Opus audio is always encoded at 48kHz, this function assumes that it is the case. The
// input_sample_rate is only indicative of the sample rate of the original source (which appears in
// the opus header).
fn write_ogg_48khz<W: std::io::Write>(
    w: &mut W,
    pcm: &[f32],
    input_sample_rate: u32,
    stereo: bool,
) -> Result<()> {
    let mut pw = ogg::PacketWriter::new(w);
    let channels = if stereo { 2 } else { 1 };

    anyhow::ensure!(
        pcm.len().is_multiple_of(channels),
        "PCM sample count must be divisible by the channel count"
    );

    let mut encoder = {
        let channels = if stereo {
            opus::Channels::Stereo
        } else {
            opus::Channels::Mono
        };
        opus::Encoder::new(OPUS_SAMPLE_RATE, channels, opus::Application::Voip)?
    };
    let pre_skip = u16::try_from(encoder.get_lookahead()?)
        .context("Opus encoder returned an invalid lookahead")?;

    // Write the opus headers and tags
    let mut head = Vec::new();
    write_opus_header(
        &mut head,
        u8::try_from(channels).expect("Opus supports at most two channels"),
        input_sample_rate,
        pre_skip,
    )?;
    pw.write_packet(head, 42, ogg::PacketWriteEndInfo::EndPage, 0)?;
    let mut tags = Vec::new();
    write_opus_tags(&mut tags)?;
    pw.write_packet(tags, 42, ogg::PacketWriteEndInfo::EndPage, 0)?;

    // Write the actual pcm data
    let mut out_encoded = vec![0u8; 50_000];

    let input_frames = pcm.len() / channels;
    let frames_to_encode = input_frames + pre_skip as usize;
    let encoded_frames = frames_to_encode.div_ceil(OPUS_ENCODER_FRAME_SIZE);
    let mut padded_pcm = vec![0.0; encoded_frames * OPUS_ENCODER_FRAME_SIZE * channels];
    padded_pcm[..pcm.len()].copy_from_slice(pcm);

    for (frame_index, frame) in padded_pcm
        .chunks_exact(OPUS_ENCODER_FRAME_SIZE * channels)
        .enumerate()
    {
        let size = encoder.encode_float(frame, &mut out_encoded)?;
        let msg = out_encoded[..size].to_vec();
        let is_last = frame_index + 1 == encoded_frames;
        let end_info = if is_last {
            ogg::PacketWriteEndInfo::EndStream
        } else {
            ogg::PacketWriteEndInfo::NormalPacket
        };
        let granule_position = if is_last {
            u64::try_from(input_frames).context("input contains too many frames")?
                + u64::from(pre_skip)
        } else {
            ((frame_index + 1) * OPUS_ENCODER_FRAME_SIZE) as u64
        };
        pw.write_packet(msg, 42, end_info, granule_position)?;
    }

    Ok(())
}

/// Encodes mono PCM samples as an Ogg Opus stream.
///
/// # Errors
///
/// Returns an error if resampling, Opus encoding, or writing fails.
pub fn write_ogg_mono<W: std::io::Write>(w: &mut W, pcm: &[f32], sample_rate: u32) -> Result<()> {
    if sample_rate == OPUS_SAMPLE_RATE {
        write_ogg_48khz(w, pcm, sample_rate, false)
    } else {
        let pcm = resample(pcm, sample_rate as usize, OPUS_SAMPLE_RATE as usize)?;
        write_ogg_48khz(w, &pcm, sample_rate, false)
    }
}

/// Writes mono PCM samples to an Ogg Opus file.
///
/// # Errors
///
/// Returns an error if the file cannot be created or encoding fails.
pub fn write_opus(filename: impl AsRef<Path>, data: &[f32], sample_rate: u32) -> Result<()> {
    let w = std::fs::File::create(filename.as_ref())?;

    let mut w = std::io::BufWriter::new(w);

    write_ogg_mono(&mut w, data, sample_rate)?;

    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Test signal generation uses small, bounded values.
mod tests {
    use super::*;

    #[test]
    fn test_write_opus_header() {
        let mut buffer = Vec::new();
        let result = write_opus_header(&mut buffer, 1, 16000, 312);

        assert!(result.is_ok());
        assert_eq!(&buffer[0..8], b"OpusHead");
        assert_eq!(buffer[8], 1); // version
        assert_eq!(buffer[9], 1); // channel count
    }

    #[test]
    fn test_write_opus_header_stereo() {
        let mut buffer = Vec::new();
        let result = write_opus_header(&mut buffer, 2, 48000, 312);

        assert!(result.is_ok());
        assert_eq!(&buffer[0..8], b"OpusHead");
        assert_eq!(buffer[8], 1); // version
        assert_eq!(buffer[9], 2); // stereo channel count
    }

    #[test]
    fn test_write_opus_tags() {
        let mut buffer = Vec::new();
        let result = write_opus_tags(&mut buffer);

        assert!(result.is_ok());
        assert_eq!(&buffer[0..8], b"OpusTags");
        // Check vendor string length is correct (4 bytes for "rust")
        assert_eq!(buffer[8], 4);
        assert_eq!(buffer[9], 0);
        assert_eq!(buffer[10], 0);
        assert_eq!(buffer[11], 0);
        // Check vendor string
        assert_eq!(&buffer[12..16], b"rust");
    }

    #[test]
    fn test_opus_constants() {
        // Verify that the constants are set to expected values
        assert_eq!(OPUS_ENCODER_FRAME_SIZE, 960);
        assert_eq!(OPUS_SAMPLE_RATE, 48000);
    }

    #[test]
    fn test_write_ogg_mono_with_silence() {
        // Test writing a small amount of silence
        let sample_rate = 48000;
        let duration_frames = 10;
        let pcm: Vec<f32> = vec![0.0; OPUS_ENCODER_FRAME_SIZE * duration_frames];

        let mut buffer = Vec::new();
        let result = write_ogg_mono(&mut buffer, &pcm, sample_rate);

        assert!(result.is_ok());
        assert!(!buffer.is_empty(), "Output buffer should contain data");
    }

    #[test]
    fn test_write_ogg_mono_with_tone() {
        // Test writing a simple sine wave
        let sample_rate = 48000;
        let duration_frames = 5;
        let frequency = 440.0;

        let pcm: Vec<f32> = (0..OPUS_ENCODER_FRAME_SIZE * duration_frames)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        let mut buffer = Vec::new();
        let result = write_ogg_mono(&mut buffer, &pcm, sample_rate);

        assert!(result.is_ok());
        assert!(!buffer.is_empty(), "Output buffer should contain data");
    }

    #[test]
    fn test_write_ogg_mono_pads_short_input_and_ends_stream() {
        let pcm = vec![0.25; 100];
        let mut buffer = Vec::new();
        write_ogg_mono(&mut buffer, &pcm, OPUS_SAMPLE_RATE).unwrap();

        let mut reader = ogg::PacketReader::new(std::io::Cursor::new(buffer));
        let mut packets = Vec::new();
        while let Some(packet) = reader.read_packet().unwrap() {
            packets.push(packet);
        }

        assert!(packets.len() >= 3, "expected headers and an audio packet");
        assert!(packets.last().unwrap().last_in_stream());
        let pre_skip = u16::from_le_bytes([packets[0].data[10], packets[0].data[11]]);
        assert_eq!(
            packets.last().unwrap().absgp_page(),
            pcm.len() as u64 + u64::from(pre_skip)
        );
    }

    #[test]
    fn test_write_ogg_mono_requires_resampling() {
        // Test with a sample rate that requires resampling
        let sample_rate = 16000;
        let duration_frames = 5;

        // Note: At 16kHz, we need proportionally fewer samples
        let num_samples = (sample_rate as f32 / OPUS_SAMPLE_RATE as f32
            * OPUS_ENCODER_FRAME_SIZE as f32) as usize
            * duration_frames;
        let pcm: Vec<f32> = vec![0.5; num_samples];

        let mut buffer = Vec::new();
        let result = write_ogg_mono(&mut buffer, &pcm, sample_rate);

        assert!(result.is_ok());
        assert!(
            !buffer.is_empty(),
            "Output buffer should contain data after resampling"
        );
    }

    #[test]
    fn test_opus_header_structure() {
        let mut buffer = Vec::new();
        write_opus_header(&mut buffer, 1, 16000, 312).unwrap();

        // Verify the structure matches the Opus specification
        assert_eq!(buffer.len(), 19); // OpusHead packet should be 19 bytes

        // Magic signature
        assert_eq!(&buffer[0..8], b"OpusHead");

        // Version (1 byte)
        assert_eq!(buffer[8], 1);

        // Channel count (1 byte)
        assert_eq!(buffer[9], 1);

        // Pre-skip (2 bytes, little-endian)
        let pre_skip = u16::from_le_bytes([buffer[10], buffer[11]]);
        assert_eq!(pre_skip, 312);

        // Sample rate (4 bytes, little-endian)
        let sample_rate = u32::from_le_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]);
        assert_eq!(sample_rate, 16000);

        // Output gain (2 bytes, little-endian)
        let output_gain = i16::from_le_bytes([buffer[16], buffer[17]]);
        assert_eq!(output_gain, 0);

        // Channel mapping family (1 byte)
        assert_eq!(buffer[18], 0);
    }

    #[test]
    fn test_opus_tags_structure() {
        let mut buffer = Vec::new();
        write_opus_tags(&mut buffer).unwrap();

        // Magic signature
        assert_eq!(&buffer[0..8], b"OpusTags");

        // Vendor string length (4 bytes, little-endian)
        let vendor_length = u32::from_le_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]);
        assert_eq!(vendor_length, 4);

        // Vendor string
        assert_eq!(&buffer[12..16], b"rust");

        // Number of user comments (4 bytes, little-endian)
        let num_comments = u32::from_le_bytes([buffer[16], buffer[17], buffer[18], buffer[19]]);
        assert_eq!(num_comments, 0);
    }
}
