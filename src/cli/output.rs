use std::path::Path;

use anyhow::{Context, Result};
use chrono::prelude::*;
use extract_speech::{
    audio::opus::write_opus, audio::resampler::resample, SpeechSegment, SAMPLE_RATE,
};
use log::info;
use serde::Serialize;

use super::args::{Args, OutputFormat, OutputType};

#[derive(Debug, Serialize)]
struct IntervalMetadata {
    filename: String,
    duration: String,
}

#[derive(Debug, Serialize)]
struct VadMetadata {
    intervals: Vec<IntervalMetadata>,
    total_seconds: String,
    compute_seconds: String,
}

pub(super) fn write_results(
    args: &Args,
    speeches: &[SpeechSegment],
    samples: &[f32],
    compute_seconds: f64,
) -> Result<()> {
    info!("Speeches: {}", speeches.len());
    prepare_output_path(&args.output, args.output_type)?;

    let output_sample_rate = u32::try_from(args.sample_rate).context("sample rate exceeds u32")?;
    let intervals = match args.output_format {
        OutputFormat::Wav => write_wav_results(args, speeches, samples, output_sample_rate)?,
        OutputFormat::Opus | OutputFormat::Ogg => {
            write_opus_results(args, speeches, samples, output_sample_rate)?
        }
    };

    if let Some(metadata_path) = &args.metadata {
        write_metadata(metadata_path, intervals, compute_seconds)?;
    }
    Ok(())
}

fn prepare_output_path(path: &Path, output_type: OutputType) -> Result<()> {
    match output_type {
        OutputType::Files => std::fs::create_dir_all(path)?,
        OutputType::Concatenated => {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
        }
    }
    Ok(())
}

fn write_wav_results(
    args: &Args,
    speeches: &[SpeechSegment],
    samples: &[f32],
    sample_rate: u32,
) -> Result<Vec<IntervalMetadata>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut intervals = Vec::new();

    match args.output_type {
        OutputType::Files => {
            for (index, speech) in speeches.iter().enumerate() {
                let filename = format!("{}_{}.wav", Utc::now().timestamp_millis(), index);
                let output_samples = segment_output_samples(samples, speech, args.sample_rate)?;
                write_wav(&args.output.join(&filename), &output_samples, spec)?;
                intervals.push(interval_metadata(
                    filename,
                    output_samples.len(),
                    sample_rate,
                ));
            }
        }
        OutputType::Concatenated => {
            let gathered = gather_speeches(samples, speeches)?;
            let output_samples = prepare_output_samples(&gathered, args.sample_rate)?;
            write_wav(&args.output, &output_samples, spec)?;
            intervals.push(interval_metadata(
                output_filename(&args.output, "output.wav"),
                output_samples.len(),
                sample_rate,
            ));
        }
    }
    info!("Saved to WAV.");
    Ok(intervals)
}

fn write_opus_results(
    args: &Args,
    speeches: &[SpeechSegment],
    samples: &[f32],
    sample_rate: u32,
) -> Result<Vec<IntervalMetadata>> {
    let mut intervals = Vec::new();
    match args.output_type {
        OutputType::Files => {
            for (index, speech) in speeches.iter().enumerate() {
                let filename = format!(
                    "{}_{}.{}",
                    Utc::now().timestamp_millis(),
                    index,
                    args.output_format.extension()
                );
                let output_samples = segment_output_samples(samples, speech, args.sample_rate)?;
                write_opus(args.output.join(&filename), &output_samples, sample_rate)?;
                intervals.push(interval_metadata(
                    filename,
                    output_samples.len(),
                    sample_rate,
                ));
            }
        }
        OutputType::Concatenated => {
            let gathered = gather_speeches(samples, speeches)?;
            let output_samples = prepare_output_samples(&gathered, args.sample_rate)?;
            write_opus(&args.output, &output_samples, sample_rate)?;
            intervals.push(interval_metadata(
                output_filename(&args.output, "output.ogg"),
                output_samples.len(),
                sample_rate,
            ));
        }
    }
    info!("Saved to OPUS.");
    Ok(intervals)
}

fn segment_output_samples(
    samples: &[f32],
    speech: &SpeechSegment,
    output_sample_rate: usize,
) -> Result<Vec<f32>> {
    prepare_output_samples(checked_segment(samples, speech)?, output_sample_rate)
}

#[allow(clippy::cast_precision_loss)] // Durations intentionally expose fractional seconds.
fn interval_metadata(filename: String, sample_count: usize, sample_rate: u32) -> IntervalMetadata {
    IntervalMetadata {
        filename,
        duration: format!("{:.6}", sample_count as f64 / f64::from(sample_rate)),
    }
}

fn output_filename(path: &Path, fallback: &str) -> String {
    path.file_name()
        .and_then(|filename| filename.to_str())
        .unwrap_or(fallback)
        .to_owned()
}

fn write_metadata(
    path: &Path,
    intervals: Vec<IntervalMetadata>,
    compute_seconds: f64,
) -> Result<()> {
    let total_seconds: f64 = intervals
        .iter()
        .filter_map(|interval| interval.duration.parse::<f64>().ok())
        .sum();
    let metadata = VadMetadata {
        intervals,
        total_seconds: format!("{total_seconds:.6}"),
        compute_seconds: format!("{compute_seconds:.6}"),
    };
    std::fs::write(path, serde_json::to_string_pretty(&metadata)?)?;
    info!("Metadata saved.");
    Ok(())
}

fn checked_segment<'a>(samples: &'a [f32], timestamp: &SpeechSegment) -> Result<&'a [f32]> {
    anyhow::ensure!(
        timestamp.start <= timestamp.end,
        "invalid speech interval: start {} is after end {}",
        timestamp.start,
        timestamp.end
    );
    samples
        .get(timestamp.start..timestamp.end)
        .with_context(|| {
            format!(
                "speech interval {}..{} exceeds the {} available samples",
                timestamp.start,
                timestamp.end,
                samples.len()
            )
        })
}

fn gather_speeches(samples: &[f32], speeches: &[SpeechSegment]) -> Result<Vec<f32>> {
    let capacity = speeches
        .iter()
        .map(|speech| speech.end.saturating_sub(speech.start))
        .sum();
    let mut gathered = Vec::with_capacity(capacity);
    for speech in speeches {
        gathered.extend_from_slice(checked_segment(samples, speech)?);
    }
    Ok(gathered)
}

fn prepare_output_samples(samples: &[f32], output_sample_rate: usize) -> Result<Vec<f32>> {
    if output_sample_rate == SAMPLE_RATE {
        Ok(samples.to_vec())
    } else {
        resample(samples, SAMPLE_RATE, output_sample_rate)
    }
}

#[allow(clippy::cast_possible_truncation)] // Samples are clamped to the i16 PCM range first.
fn write_wav(path: &Path, samples: &[f32], spec: hound::WavSpec) -> Result<()> {
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_speech_interval_is_rejected() {
        let samples = vec![0.0; 10];
        let timestamp = SpeechSegment { start: 8, end: 11 };
        assert!(checked_segment(&samples, &timestamp).is_err());
    }

    #[test]
    fn output_samples_are_resampled_to_requested_rate() {
        let samples = vec![0.0; SAMPLE_RATE];
        let output = prepare_output_samples(&samples, 8_000).unwrap();
        assert!((output.len().cast_signed() - 8_000).abs() <= 2);
    }
}
