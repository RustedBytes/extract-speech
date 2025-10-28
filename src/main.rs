#[cfg(feature = "accelerate-src")]
extern crate accelerate_src;

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Result;
use chrono::prelude::*;
use clap::{Parser, ValueEnum};
use log::{debug, info};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    ExecutionProviderDispatch, TensorRTExecutionProvider,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod audio;
mod opus;
mod pyannote_vad_iter;
mod pyannote_vad_ort;
mod resampler;
mod silero_v5;
mod silero_v5_ort;
pub(crate) mod utils;
mod vad_iter;
mod vad_iter_ort;

use crate::audio::load_samples_from_audio_file;
use crate::opus::write_opus;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Wav,
    Opus,
    Ogg,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum ModelInfo {
    Graph,
    Nodes,
    IO,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Runtime {
    Candle,
    Onnxruntime,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum VadModel {
    Silero,
    Pyannote,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum OutputType {
    Files,
    Concatenated,
}

#[derive(Debug, Serialize, Deserialize)]
struct IntervalMetadata {
    filename: String,
    duration: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct VadMetadata {
    intervals: Vec<IntervalMetadata>,
    total_seconds: String,
    compute_seconds: String,
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
    #[clap(value_enum, default_value_t = Runtime::Candle)]
    runtime: Runtime,

    /// VAD model type
    #[arg(long)]
    #[clap(value_enum, default_value_t = VadModel::Silero)]
    vad_model: VadModel,

    /// Path to the ONNX runtime dynamic library
    #[arg(long)]
    dylib_path: Option<PathBuf>,

    /// The source audio file path
    #[arg(long)]
    source_audio: Option<PathBuf>,

    /// The audio file path to process in VAD stage, it can be denoised signal (if source_audio is provided then final samples will be taken from source_audio)
    #[arg(long)]
    process_audio: PathBuf,

    /// The path of a final result file or directory
    #[arg(long)]
    output: Option<PathBuf>,

    /// Path to write metadata JSON file
    #[arg(long)]
    metadata: Option<PathBuf>,

    /// The output type
    #[arg(long)]
    #[clap(value_enum, default_value_t = OutputType::Files)]
    output_type: OutputType,

    /// The result format
    #[arg(long)]
    #[clap(value_enum, default_value_t = OutputFormat::Wav)]
    output_format: OutputFormat,

    /// VAD threshold
    #[arg(long, default_value = "0.7")]
    threshold: f32,

    /// Sample rate of final output
    #[arg(long, default_value = "16000")]
    sample_rate: usize,

    /// Enable TensorRT
    #[arg(long, default_value_t = false)]
    trt: bool,

    /// Enable CUDA
    #[arg(long, default_value_t = false)]
    cuda: bool,

    /// Enable CoreML
    #[arg(long, default_value_t = false)]
    coreml: bool,

    /// Debug mode
    #[arg(long, default_value = "false")]
    debug: bool,
}

fn print_model_info(model_path: PathBuf, info: ModelInfo) -> Result<()> {
    let model = candle_onnx::read_file(model_path)?;

    let graph = model.clone().graph.unwrap();

    match info {
        ModelInfo::Graph => {
            debug!("{model:#?}");
        }
        ModelInfo::Nodes => {
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

    tracing_subscriber::fmt::init();

    let mut execution_providers: Vec<ExecutionProviderDispatch> =
        vec![CPUExecutionProvider::default().build()];

    if args.cuda {
        execution_providers.insert(0, CUDAExecutionProvider::default().build());
    }

    if args.coreml {
        execution_providers.insert(0, CoreMLExecutionProvider::default().build());
    }

    if args.trt {
        execution_providers.insert(0, TensorRTExecutionProvider::default().build());
    }

    // Print the model info
    if args.print_model_info.is_some() {
        print_model_info(args.model_path, args.print_model_info.unwrap())?;
        return Ok(());
    }

    // Load the audio files
    let start = std::time::Instant::now();
    let process_audio_path = args.process_audio.clone();
    let process_samples = load_samples_from_audio_file(process_audio_path)?;
    info!(
        "Number of samples (process_audio): {:?}",
        process_samples.len()
    );

    let mut source_samples = process_samples.clone();

    if args.source_audio.is_some() {
        let source_audio_path = args.source_audio.clone().unwrap();

        source_samples = load_samples_from_audio_file(source_audio_path)?;

        info!(
            "Number of samples (source_audio): {:?}",
            source_samples.len()
        );

        if process_samples.len() != source_samples.len() {
            return Err(anyhow::anyhow!(
                "The number of samples in the source and process audio files should be the same."
            ));
        }
    }

    info!("Retrieved audio files in: {:?}", start.elapsed());

    // Create the VAD params
    let vad_params = utils::VadParams {
        sample_rate: 16_000,
        threshold: args.threshold,
        debug: args.debug,
        ..Default::default()
    };

    match args.runtime {
        Runtime::Candle => {
            // Platform specific optimizations
            debug!(
                "avx: {}, neon: {}, simd128: {}, f16c: {}",
                candle_core::utils::with_avx(),
                candle_core::utils::with_neon(),
                candle_core::utils::with_simd128(),
                candle_core::utils::with_f16c()
            );

            // Set the device
            let device = candle_core::Device::Cpu;

            match args.vad_model {
                VadModel::Silero => {
                    // Create the VAD model
                    let start: std::time::Instant = std::time::Instant::now();
                    let silero = silero_v5::Silero::new(
                        vad_params.clone(),
                        args.model_path.clone(),
                        device,
                    )?;
                    info!("Loaded the model in: {:?}", start.elapsed());

                    // Do inference
                    let start = std::time::Instant::now();
                    let mut vad_iterator = vad_iter::VadIter::new(silero, vad_params);
                    let speeches_result = vad_iterator.process(process_samples.to_vec())?;
                    let compute_time = start.elapsed();
                    info!("Inference time: {:?}", compute_time);

                    // Write the output
                    write_results(
                        args,
                        speeches_result,
                        source_samples,
                        compute_time.as_secs_f64(),
                    )
                }
                VadModel::Pyannote => {
                    return Err(anyhow::anyhow!(
                        "PyAnnote model is only supported with ONNX Runtime. Use --runtime onnxruntime"
                    ));
                }
            }
        }
        Runtime::Onnxruntime => {
            let dylib_path = args.dylib_path.clone().unwrap();
            ort::init_from(
                dylib_path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Invalid path: dylib_path"))?,
            )
            .commit()?;

            match args.vad_model {
                VadModel::Silero => {
                    // Create the VAD model
                    let start: std::time::Instant = std::time::Instant::now();
                    let silero = silero_v5_ort::Silero::new(
                        vad_params.clone(),
                        execution_providers,
                        args.model_path.clone(),
                    )?;
                    info!("Loaded the model in: {:?}", start.elapsed());

                    // Do inference
                    let start = std::time::Instant::now();
                    let mut vad_iterator_ort = vad_iter_ort::VadIter::new(silero, vad_params);
                    let speeches_result = vad_iterator_ort.process(process_samples.to_vec())?;
                    let compute_time = start.elapsed();
                    info!("Inference time: {:?}", compute_time);

                    // Write the output
                    write_results(
                        args,
                        speeches_result,
                        source_samples,
                        compute_time.as_secs_f64(),
                    )
                }
                VadModel::Pyannote => {
                    // Create the VAD model
                    let start: std::time::Instant = std::time::Instant::now();
                    let pyannote = pyannote_vad_ort::PyAnnote::new(
                        vad_params.clone(),
                        execution_providers,
                        args.model_path.clone(),
                    )?;
                    info!("Loaded the model in: {:?}", start.elapsed());

                    // Do inference
                    let start = std::time::Instant::now();
                    let mut pyannote_vad_iterator =
                        pyannote_vad_iter::PyAnnoteVadIter::new(pyannote, vad_params);
                    let speeches_result =
                        pyannote_vad_iterator.process(process_samples.to_vec())?;
                    let compute_time = start.elapsed();
                    info!("Inference time: {:?}", compute_time);

                    // Write the output
                    write_results(
                        args,
                        speeches_result,
                        source_samples,
                        compute_time.as_secs_f64(),
                    )
                }
            }
        }
    }
}

fn write_results(
    args: Args,
    speeches: &[utils::TimeStamp],
    samples: Vec<f32>,
    compute_seconds: f64,
) -> Result<()> {
    info!("Speeches: {}", speeches.len());

    let output_path = args.output.unwrap();

    // Create the output directory if it doesn't have extension
    if output_path.extension().is_none() {
        std::fs::create_dir_all(output_path.clone())?;
    }
    let directory = output_path.display();

    // Collect metadata
    let mut intervals: Vec<IntervalMetadata> = Vec::new();
    let sample_rate = 16_000.0;

    // Write the output
    match args.output_format {
        OutputFormat::Wav => {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };

            match args.output_type {
                OutputType::Files => {
                    // Parallel processing to write files concurrently
                    let intervals_mutex = Mutex::new(Vec::new());
                    let base_ts = Utc::now().timestamp_millis();

                    let result: Result<()> =
                        speeches
                            .par_iter()
                            .enumerate()
                            .try_for_each(|(idx, speech)| {
                                let filename = format!("{}_{}.wav", base_ts, idx);
                                let filepath = format!("{}/{}", directory, filename);
                                let mut writer = hound::WavWriter::create(&filepath, spec)
                                    .map_err(|e| anyhow::anyhow!(e))?;

                                let segment_samples =
                                    samples[speech.start as usize..speech.end as usize].to_vec();
                                for sample in &segment_samples {
                                    let x = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                                    writer.write_sample(x).map_err(|e| anyhow::anyhow!(e))?;
                                }

                                writer.finalize().map_err(|e| anyhow::anyhow!(e))?;

                                // Calculate duration in seconds
                                let duration_seconds = segment_samples.len() as f64 / sample_rate;

                                // Collect metadata in thread-safe manner
                                intervals_mutex
                                    .lock()
                                    .map_err(|e| anyhow::anyhow!("Mutex poisoned: {:?}", e))?
                                    .push((
                                        idx,
                                        IntervalMetadata {
                                            filename,
                                            duration: format!("{:.6}", duration_seconds),
                                        },
                                    ));

                                Ok(())
                            });

                    result?;

                    // Sort intervals by index to maintain order
                    let mut intervals_with_idx = intervals_mutex
                        .into_inner()
                        .map_err(|e| anyhow::anyhow!("Mutex poisoned: {:?}", e))?;
                    intervals_with_idx.sort_by_key(|(idx, _)| *idx);
                    intervals.extend(intervals_with_idx.into_iter().map(|(_, meta)| meta));
                }
                OutputType::Concatenated => {
                    let gathered_speeches = speeches
                        .iter()
                        .flat_map(|timestamp| {
                            &samples[timestamp.start as usize..timestamp.end as usize]
                        })
                        .cloned()
                        .collect::<Vec<f32>>();

                    let mut writer = hound::WavWriter::create(&output_path, spec)?;

                    for sample in &gathered_speeches {
                        let x = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        writer.write_sample(x)?;
                    }

                    writer.finalize()?;

                    // For concatenated, we have a single output file
                    let duration_seconds = gathered_speeches.len() as f64 / sample_rate;
                    let filename = output_path
                        .file_name()
                        .and_then(|f| f.to_str())
                        .unwrap_or("output.wav")
                        .to_string();
                    intervals.push(IntervalMetadata {
                        filename,
                        duration: format!("{:.6}", duration_seconds),
                    });
                }
            }

            info!("Saved to WAV.");
        }
        OutputFormat::Opus | OutputFormat::Ogg => {
            match args.output_type {
                OutputType::Files => {
                    // Parallel processing to write files concurrently
                    let intervals_mutex = Mutex::new(Vec::new());
                    let base_ts = Utc::now().timestamp_millis();
                    let sample_rate_val = args.sample_rate;

                    let result: Result<()> =
                        speeches
                            .par_iter()
                            .enumerate()
                            .try_for_each(|(idx, speech)| {
                                let filename = format!("{}_{}.ogg", base_ts, idx);
                                let filepath = PathBuf::from(format!("{}/{}", directory, filename));
                                let process_samples =
                                    samples[speech.start as usize..speech.end as usize].to_vec();

                                write_opus(filepath, process_samples.clone(), sample_rate_val)
                                    .map_err(|e| anyhow::anyhow!(e))?;

                                // Calculate duration in seconds
                                let duration_seconds = process_samples.len() as f64 / sample_rate;

                                // Collect metadata in thread-safe manner
                                intervals_mutex
                                    .lock()
                                    .map_err(|e| anyhow::anyhow!("Mutex poisoned: {:?}", e))?
                                    .push((
                                        idx,
                                        IntervalMetadata {
                                            filename,
                                            duration: format!("{:.6}", duration_seconds),
                                        },
                                    ));

                                Ok(())
                            });

                    result?;

                    // Sort intervals by index to maintain order
                    let mut intervals_with_idx = intervals_mutex
                        .into_inner()
                        .map_err(|e| anyhow::anyhow!("Mutex poisoned: {:?}", e))?;
                    intervals_with_idx.sort_by_key(|(idx, _)| *idx);
                    intervals.extend(intervals_with_idx.into_iter().map(|(_, meta)| meta));
                }
                OutputType::Concatenated => {
                    let gathered_speeches = speeches
                        .iter()
                        .flat_map(|timestamp| {
                            &samples[timestamp.start as usize..timestamp.end as usize]
                        })
                        .cloned()
                        .collect::<Vec<f32>>();

                    write_opus(
                        output_path.clone(),
                        gathered_speeches.clone(),
                        args.sample_rate,
                    )
                    .map_err(|e| anyhow::anyhow!(e))?;

                    // For concatenated, we have a single output file
                    let duration_seconds = gathered_speeches.len() as f64 / sample_rate;
                    let filename = output_path
                        .file_name()
                        .and_then(|f| f.to_str())
                        .unwrap_or("output.ogg")
                        .to_string();
                    intervals.push(IntervalMetadata {
                        filename,
                        duration: format!("{:.6}", duration_seconds),
                    });
                }
            }

            info!("Saved to OPUS.");
        }
    }

    // Write metadata if requested
    if let Some(metadata_path) = args.metadata {
        let total_seconds: f64 = intervals
            .iter()
            .map(|i| i.duration.parse::<f64>().unwrap_or(0.0))
            .sum();

        let metadata = VadMetadata {
            intervals,
            total_seconds: format!("{:.6}", total_seconds),
            compute_seconds: format!("{:.6}", compute_seconds),
        };

        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(metadata_path, metadata_json)?;
        info!("Metadata saved.");
    }

    Ok(())
}
