#[cfg(feature = "accelerate-src")]
extern crate accelerate_src;

use std::path::PathBuf;

use anyhow::Result;
use chrono::prelude::*;
use clap::{Parser, ValueEnum};
use rayon::prelude::*;

mod audio;
mod opus;
mod silero_v5;
mod silero_v5_ort;
pub(crate) mod utils;
mod vad_iter;
mod vad_iter_ort;
use hound;

use crate::audio::load_samples_from_audio_file;
use crate::opus::write_opus;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    WAV,
    OPUS,
    OGG,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum ModelInfo {
    GRAPH,
    NODES,
    IO,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Runtime {
    CANDLE,
    ONNXRUNTIME,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum OutputType {
    FILES,
    CONCATENATED,
}

#[derive(Parser, Debug)]
#[command(version, long_about = None)]
struct Args {
    /// Print the model info
    #[arg(long)]
    print_model_info: Option<ModelInfo>,

    /// A path to VAD model
    #[arg(long)]
    model_path: PathBuf,

    /// Runtime variant
    #[arg(long)]
    #[clap(value_enum, default_value_t = Runtime::CANDLE)]
    runtime: Runtime,

    /// The source audio file path
    #[arg(long)]
    source_audio: Option<PathBuf>,

    /// The audio file path to process in VAD stage, it can be denoised signal (if source_audio is provided then final samples will be taken from source_audio)
    #[arg(long)]
    process_audio: PathBuf,

    /// The path of a final result file or directory
    #[arg(long)]
    output: Option<PathBuf>,

    /// The output type
    #[arg(long)]
    #[clap(value_enum, default_value_t = OutputType::FILES)]
    output_type: OutputType,

    /// The result format
    #[arg(long)]
    #[clap(value_enum, default_value_t = OutputFormat::WAV)]
    output_format: OutputFormat,

    /// VAD threshold
    #[arg(long, default_value = "0.7")]
    threshold: f32,

    /// Sample rate of final output
    #[arg(long, default_value = "16000")]
    sample_rate: usize,

    /// Debug mode
    #[arg(long, default_value = "false")]
    debug: bool,
}

fn print_model_info(model_path: PathBuf, info: ModelInfo) -> Result<()> {
    let model = candle_onnx::read_file(model_path)?;

    let graph = model.clone().graph.unwrap();

    match info {
        ModelInfo::GRAPH => {
            println!("{model:#?}");
        }
        ModelInfo::NODES => {
            for node in graph.node.iter() {
                println!("{node:#?}");
            }
        }
        ModelInfo::IO => {
            for input in graph.input.iter() {
                println!("input: {input:#?}");
            }
            for output in graph.output.iter() {
                println!("output: {output:#?}");
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    // Parse the arguments
    let args = Args::parse();

    // Print the model info
    if args.print_model_info.is_some() {
        print_model_info(args.model_path, args.print_model_info.unwrap())?;
        return Ok(());
    }

    // Load the audio files
    let start = std::time::Instant::now();
    let process_audio_path = args.process_audio.clone();
    let process_samples = load_samples_from_audio_file(process_audio_path)?;
    println!(
        "Number of samples (process_audio): {:?}",
        process_samples.len()
    );

    let mut source_samples = process_samples.clone();

    if args.source_audio.is_some() {
        let source_audio_path = args.source_audio.clone().unwrap();

        source_samples = load_samples_from_audio_file(source_audio_path)?;

        println!(
            "Number of samples (source_audio): {:?}",
            source_samples.len()
        );

        if process_samples.len() != source_samples.len() {
            return Err(anyhow::anyhow!(
                "The number of samples in the source and process audio files should be the same."
            ));
        }
    }

    println!("Retrieved audio files in: {:?}", start.elapsed());

    // Create the VAD params
    let vad_params = utils::VadParams {
        sample_rate: 16_000,
        threshold: args.threshold,
        debug: args.debug,
        ..Default::default()
    };

    match args.runtime {
        Runtime::CANDLE => {
            // Platform specific optimizations
            println!(
                "avx: {}, neon: {}, simd128: {}, f16c: {}",
                candle_core::utils::with_avx(),
                candle_core::utils::with_neon(),
                candle_core::utils::with_simd128(),
                candle_core::utils::with_f16c()
            );

            // Set the device
            let device = candle_core::Device::Cpu;

            // Create the VAD model
            let start: std::time::Instant = std::time::Instant::now();
            let silero =
                silero_v5::Silero::new(vad_params.clone(), args.model_path.clone(), device)?;
            println!("Loaded the model in: {:?}", start.elapsed());

            // Do inference
            let start = std::time::Instant::now();
            let mut vad_iterator = vad_iter::VadIter::new(silero, vad_params);
            let speeches_result = vad_iterator.process(process_samples.to_vec())?;
            println!("Inference time: {:?}", start.elapsed());

            // Write the output
            write_results(args, &speeches_result, source_samples)
        }
        Runtime::ONNXRUNTIME => {
            // Create the VAD model
            let start: std::time::Instant = std::time::Instant::now();
            let silero = silero_v5_ort::Silero::new(vad_params.clone(), args.model_path.clone())?;
            println!("Loaded the model in: {:?}", start.elapsed());

            // Do inference
            let start = std::time::Instant::now();
            let mut vad_iterator_ort = vad_iter_ort::VadIter::new(silero, vad_params);
            let speeches_result = vad_iterator_ort.process(process_samples.to_vec())?;
            println!("Inference time: {:?}", start.elapsed());

            // Write the output
            write_results(args, &speeches_result, source_samples)
        }
    }
}

fn write_results(args: Args, speeches: &[utils::TimeStamp], samples: Vec<f32>) -> Result<()> {
    println!("Speeches: {}", speeches.len());

    let output_path = args.output.unwrap();

    // Create the output directory if it doesn't have extension
    if output_path.extension().is_none() {
        std::fs::create_dir_all(output_path.clone())?;
    }
    let directory = output_path.display();

    // Write the output
    match args.output_format {
        OutputFormat::WAV => {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };

            match args.output_type {
                OutputType::FILES => {
                    speeches
                        .par_iter()
                        .enumerate()
                        .try_for_each(|(idx, speech)| -> Result<()> {
                            let current_ts = Utc::now().timestamp_millis();
                            let filename = format!("{}/{}_{}.wav", directory, current_ts, idx);
                            let mut writer = hound::WavWriter::create(filename, spec)?;

                            let samples =
                                samples[speech.start as usize..speech.end as usize].to_vec();
                            for sample in samples {
                                let x = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                                writer.write_sample(x)?;
                            }

                            writer.finalize()?;

                            Ok(())
                        })?;
                }
                OutputType::CONCATENATED => {
                    let gathered_speeches = speeches
                        .iter()
                        .flat_map(|timestamp| {
                            &samples[timestamp.start as usize..timestamp.end as usize]
                        })
                        .cloned()
                        .collect::<Vec<f32>>();

                    let mut writer = hound::WavWriter::create(output_path, spec)?;

                    for sample in gathered_speeches {
                        let x = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        writer.write_sample(x)?;
                    }

                    writer.finalize()?;
                }
            }

            println!("Saved to WAV.");
        }
        OutputFormat::OPUS | OutputFormat::OGG => {
            match args.output_type {
                OutputType::FILES => {
                    speeches
                        .par_iter()
                        .enumerate()
                        .try_for_each(|(idx, speech)| -> Result<()> {
                            let current_ts = Utc::now().timestamp_millis();
                            let filename =
                                PathBuf::from(format!("{}/{}_{}.ogg", directory, current_ts, idx));
                            let process_samples =
                                samples[speech.start as usize..speech.end as usize].to_vec();

                            write_opus(filename, process_samples, args.sample_rate)
                                .map_err(|e| anyhow::anyhow!(e))?;

                            Ok(())
                        })?;
                }
                OutputType::CONCATENATED => {
                    let gathered_speeches = speeches
                        .iter()
                        .flat_map(|timestamp| {
                            &samples[timestamp.start as usize..timestamp.end as usize]
                        })
                        .cloned()
                        .collect::<Vec<f32>>();

                    write_opus(output_path, gathered_speeches, args.sample_rate)
                        .map_err(|e| anyhow::anyhow!(e))?;
                }
            }

            println!("Saved to OPUS.");
        }
    }

    Ok(())
}
