use anyhow::Result;

use crate::audio::resampler::resample;

// This must be an allowed value among 120, 240, 480, 960, 1920, and 2880.
// Using a different value would result in a BadArg "invalid argument" error when calling encode.
// https://opus-codec.org/docs/opus_api-1.2/group__opus__encoder.html#ga4ae9905859cd241ef4bb5c59cd5e5309
const OPUS_ENCODER_FRAME_SIZE: usize = 960;
const OPUS_SAMPLE_RATE: u32 = 48000;
// const OPUS_ALLOWED_FRAME_SIZES: [usize; 6] = [120, 240, 480, 960, 1920, 2880];

/// See https://www.opus-codec.org/docs/opusfile_api-0.4/structOpusHead.html
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
) -> std::io::Result<()> {
    use byteorder::WriteBytesExt;

    // https://wiki.xiph.org/OggOpus#ID_Header
    w.write_all(b"OpusHead")?;
    w.write_u8(1)?; // version
    w.write_u8(channels)?; // channel count
    w.write_u16::<byteorder::LittleEndian>(3840)?; // pre-skip
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
    w.write_u32::<byteorder::LittleEndian>(vendor.len() as u32)?; // vendor string length
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

    // Write the opus headers and tags
    let mut head = Vec::new();
    write_opus_header(&mut head, channels as u8, input_sample_rate)?;
    pw.write_packet(head, 42, ogg::PacketWriteEndInfo::EndPage, 0)?;
    let mut tags = Vec::new();
    write_opus_tags(&mut tags)?;
    pw.write_packet(tags, 42, ogg::PacketWriteEndInfo::EndPage, 0)?;

    // Write the actual pcm data
    let mut encoder = {
        let channels = if stereo {
            opus::Channels::Stereo
        } else {
            opus::Channels::Mono
        };
        opus::Encoder::new(OPUS_SAMPLE_RATE, channels, opus::Application::Voip)?
    };
    let mut out_encoded = vec![0u8; 50_000];

    let mut total_data = 0;
    let n_frames = pcm.len() / (channels * OPUS_ENCODER_FRAME_SIZE);
    for (frame_idx, pcm) in pcm
        .chunks_exact(OPUS_ENCODER_FRAME_SIZE * channels)
        .enumerate()
    {
        total_data += (pcm.len() / channels) as u64;
        let size = encoder.encode_float(pcm, &mut out_encoded)?;
        let msg = out_encoded[..size].to_vec();
        let inf = if frame_idx + 1 == n_frames {
            ogg::PacketWriteEndInfo::EndPage
        } else {
            ogg::PacketWriteEndInfo::NormalPacket
        };
        pw.write_packet(msg, 42, inf, total_data)?;
    }

    Ok(())
}

pub fn write_ogg_mono<W: std::io::Write>(w: &mut W, pcm: &[f32], sample_rate: u32) -> Result<()> {
    if sample_rate == OPUS_SAMPLE_RATE {
        write_ogg_48khz(w, pcm, sample_rate, false)
    } else {
        let pcm = resample(pcm, sample_rate as usize, OPUS_SAMPLE_RATE as usize)?;
        write_ogg_48khz(w, &pcm, sample_rate, false)
    }
}

pub fn write_opus(
    filename: std::path::PathBuf,
    data: Vec<f32>,
    sample_rate: usize,
) -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    let w = std::fs::File::create(&filename)?;

    let mut w = std::io::BufWriter::new(w);

    write_ogg_mono(&mut w, &data, sample_rate as u32)?;

    Ok(())
}
