#[cfg(all(feature = "accelerate-src", target_vendor = "apple"))]
extern crate accelerate_src;

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::prelude::*;
use clap::{Parser, ValueEnum};
use log::{debug, info};
use ort::ep::{CoreML, ExecutionProviderDispatch, TensorRT, CPU, CUDA};
use rayon::prelude::*;
use serde::Serialize;

mod audio;
mod opus;
mod pulsevad;
mod pulsevad_frontend;
mod pulsevad_iter;
mod pulsevad_ort;
mod pyannote_vad_iter;
mod pyannote_vad_ort;
mod resampler;
mod silero_v5;
mod silero_v5_ort;
pub(crate) mod utils;
mod vad_iter;

use crate::audio::load_samples_from_audio_file;
use crate::opus::write_opus;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Wav,
    Opus,
    Ogg,
}

impl OutputFormat {
    fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Opus => "opus",
            Self::Ogg => "ogg",
        }
    }
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
    #[value(name = "pulsevad", alias = "pulse-vad")]
    PulseVad,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum OutputType {
    Files,
    Concatenated,
}

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

#[derive(Parser, Debug, Clone)]
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
    #[arg(long, required_if_eq("runtime", "onnxruntime"))]
    dylib_path: Option<PathBuf>,

    /// The source audio file path
    #[arg(long, conflicts_with = "process_folder")]
    source_audio: Option<PathBuf>,

    /// The audio file path to process in VAD stage, it can be denoised signal (if source_audio is provided then final samples will be taken from source_audio)
    #[arg(long, conflicts_with = "process_folder")]
    process_audio: Option<PathBuf>,

    /// The folder path containing audio files to process in VAD stage
    #[arg(long, conflicts_with = "process_audio")]
    process_folder: Option<PathBuf>,

    /// The path of a final result file or directory
    #[arg(long, default_value = "output")]
    output: PathBuf,

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

    let graph = model
        .graph
        .as_ref()
        .context("ONNX model contains no graph")?;

    match info {
        ModelInfo::Graph => {
            println!("{model:#?}");
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

fn collect_audio_files(folder_path: &std::path::Path) -> Result<Vec<PathBuf>> {
    let audio_extensions = ["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac"];
    let mut audio_files = Vec::new();

    for entry in std::fs::read_dir(folder_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                if let Some(ext_str) = extension.to_str() {
                    if audio_extensions.contains(&ext_str.to_lowercase().as_str()) {
                        audio_files.push(path);
                    }
                }
            }
        }
    }

    if audio_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No audio files found in folder: {}",
            folder_path.display()
        ));
    }

    audio_files.sort();
    Ok(audio_files)
}

fn make_vad_params(args: &Args) -> utils::VadParams {
    utils::VadParams {
        sample_rate: utils::VAD_SAMPLE_RATE,
        threshold: args.threshold,
        min_speech_duration_ms: if args.vad_model == VadModel::PulseVad {
            100
        } else {
            utils::VadParams::default().min_speech_duration_ms
        },
        debug: args.debug,
        ..Default::default()
    }
}

fn process_single_file(
    args: Args,
    process_audio_path: PathBuf,
    execution_providers: Vec<ExecutionProviderDispatch>,
) -> Result<()> {
    // Load the audio files
    let start = std::time::Instant::now();
    let process_samples = load_samples_from_audio_file(process_audio_path)?;
    info!(
        "Number of samples (process_audio): {:?}",
        process_samples.len()
    );

    let source_samples = if let Some(source_audio_path) = args.source_audio.as_ref() {
        let source_samples = load_samples_from_audio_file(source_audio_path)?;

        info!(
            "Number of samples (source_audio): {:?}",
            source_samples.len()
        );

        if process_samples.len() != source_samples.len() {
            return Err(anyhow::anyhow!(
                "The number of samples in the source and process audio files should be the same."
            ));
        }

        Some(source_samples)
    } else {
        None
    };
    let output_samples = source_samples.as_deref().unwrap_or(&process_samples);

    info!("Retrieved audio files in: {:?}", start.elapsed());

    // Create the VAD params
    let vad_params = make_vad_params(&args);

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
                    let speeches_result = vad_iterator.process(&process_samples)?;
                    let compute_time = start.elapsed();
                    info!("Inference time: {:?}", compute_time);

                    // Write the output
                    write_results(
                        args,
                        speeches_result,
                        output_samples,
                        compute_time.as_secs_f64(),
                    )
                }
                VadModel::Pyannote => Err(anyhow::anyhow!(
                    "PyAnnote model is only supported with ONNX Runtime. Use --runtime onnxruntime"
                )),
                VadModel::PulseVad => {
                    let start = std::time::Instant::now();
                    let pulsevad =
                        pulsevad::PulseVad::new(args.model_path.clone(), device, args.debug)?;
                    info!("Loaded the model in: {:?}", start.elapsed());

                    let start = std::time::Instant::now();
                    let mut vad_iterator = pulsevad_iter::PulseVadIter::new(pulsevad, vad_params);
                    let speeches_result = vad_iterator.process(&process_samples)?;
                    let compute_time = start.elapsed();
                    info!("Inference time: {:?}", compute_time);

                    write_results(
                        args,
                        speeches_result,
                        output_samples,
                        compute_time.as_secs_f64(),
                    )
                }
            }
        }
        Runtime::Onnxruntime => {
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
                    let mut vad_iterator = vad_iter::VadIter::new(silero, vad_params);
                    let speeches_result = vad_iterator.process(&process_samples)?;
                    let compute_time = start.elapsed();
                    info!("Inference time: {:?}", compute_time);

                    // Write the output
                    write_results(
                        args,
                        speeches_result,
                        output_samples,
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
                    let speeches_result = pyannote_vad_iterator.process(&process_samples)?;
                    let compute_time = start.elapsed();
                    info!("Inference time: {:?}", compute_time);

                    // Write the output
                    write_results(
                        args,
                        speeches_result,
                        output_samples,
                        compute_time.as_secs_f64(),
                    )
                }
                VadModel::PulseVad => {
                    let start = std::time::Instant::now();
                    let pulsevad = pulsevad_ort::PulseVad::new(
                        execution_providers,
                        args.model_path.clone(),
                        args.debug,
                    )?;
                    info!("Loaded the model in: {:?}", start.elapsed());

                    let start = std::time::Instant::now();
                    let mut vad_iterator = pulsevad_iter::PulseVadIter::new(pulsevad, vad_params);
                    let speeches_result = vad_iterator.process(&process_samples)?;
                    let compute_time = start.elapsed();
                    info!("Inference time: {:?}", compute_time);

                    write_results(
                        args,
                        speeches_result,
                        output_samples,
                        compute_time.as_secs_f64(),
                    )
                }
            }
        }
    }
}

fn process_folder(
    args: Args,
    process_folder_path: PathBuf,
    execution_providers: Vec<ExecutionProviderDispatch>,
) -> Result<()> {
    info!("Processing folder: {}", process_folder_path.display());

    // Collect all audio files from the folder
    let audio_files = collect_audio_files(&process_folder_path)?;
    info!("Found {} audio files to process", audio_files.len());

    // Ensure output directory exists
    let output_base = args.output.clone();
    std::fs::create_dir_all(&output_base)?;

    // Create the VAD params
    let vad_params = make_vad_params(&args);

    // Process files in parallel using rayon
    let results: Vec<Result<()>> = audio_files
        .par_iter()
        .map(|audio_path| {
            let file_stem = audio_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            info!("Processing: {}", audio_path.display());

            // Load audio samples
            let process_samples = load_samples_from_audio_file(audio_path.clone())?;

            let file_output_path = match args.output_type {
                OutputType::Files => {
                    let directory = output_base.join(file_stem);
                    std::fs::create_dir_all(&directory)?;
                    directory
                }
                OutputType::Concatenated => {
                    output_base.join(format!("{}.{}", file_stem, args.output_format.extension()))
                }
            };

            // Process based on runtime
            match args.runtime {
                Runtime::Candle => {
                    let device = candle_core::Device::Cpu;
                    match args.vad_model {
                        VadModel::Silero => {
                            let silero = silero_v5::Silero::new(
                                vad_params.clone(),
                                args.model_path.clone(),
                                device,
                            )?;

                            let start = std::time::Instant::now();
                            let mut vad_iterator =
                                vad_iter::VadIter::new(silero, vad_params.clone());
                            let speeches_result = vad_iterator.process(&process_samples)?;
                            let compute_time = start.elapsed();

                            // Create args for this file
                            let mut file_args = args.clone();
                            file_args.output = file_output_path.clone();
                            file_args.metadata = args.metadata.as_ref().map(|m| {
                                let metadata_stem =
                                    m.file_stem().and_then(|s| s.to_str()).unwrap_or("metadata");
                                let metadata_ext =
                                    m.extension().and_then(|e| e.to_str()).unwrap_or("json");
                                output_base.join(format!(
                                    "{}_{}.{}",
                                    metadata_stem, file_stem, metadata_ext
                                ))
                            });

                            write_results(
                                file_args,
                                speeches_result,
                                &process_samples,
                                compute_time.as_secs_f64(),
                            )?;

                            info!("Completed: {} in {:?}", file_stem, compute_time);
                            Ok(())
                        }
                        VadModel::Pyannote => Err(anyhow::anyhow!(
                            "PyAnnote model is only supported with ONNX Runtime"
                        )),
                        VadModel::PulseVad => {
                            let pulsevad = pulsevad::PulseVad::new(
                                args.model_path.clone(),
                                device,
                                args.debug,
                            )?;

                            let start = std::time::Instant::now();
                            let mut vad_iterator =
                                pulsevad_iter::PulseVadIter::new(pulsevad, vad_params.clone());
                            let speeches_result = vad_iterator.process(&process_samples)?;
                            let compute_time = start.elapsed();

                            let mut file_args = args.clone();
                            file_args.output = file_output_path.clone();
                            file_args.metadata = args.metadata.as_ref().map(|m| {
                                let metadata_stem =
                                    m.file_stem().and_then(|s| s.to_str()).unwrap_or("metadata");
                                let metadata_ext =
                                    m.extension().and_then(|e| e.to_str()).unwrap_or("json");
                                output_base.join(format!(
                                    "{}_{}.{}",
                                    metadata_stem, file_stem, metadata_ext
                                ))
                            });

                            write_results(
                                file_args,
                                speeches_result,
                                &process_samples,
                                compute_time.as_secs_f64(),
                            )?;

                            info!("Completed: {} in {:?}", file_stem, compute_time);
                            Ok(())
                        }
                    }
                }
                Runtime::Onnxruntime => {
                    // Note: ONNX Runtime initialization is not thread-safe when done multiple times
                    // We assume it's already initialized in the main thread
                    match args.vad_model {
                        VadModel::Silero => {
                            let silero = silero_v5_ort::Silero::new(
                                vad_params.clone(),
                                execution_providers.clone(),
                                args.model_path.clone(),
                            )?;

                            let start = std::time::Instant::now();
                            let mut vad_iterator =
                                vad_iter::VadIter::new(silero, vad_params.clone());
                            let speeches_result = vad_iterator.process(&process_samples)?;
                            let compute_time = start.elapsed();

                            // Create args for this file
                            let mut file_args = args.clone();
                            file_args.output = file_output_path.clone();
                            file_args.metadata = args.metadata.as_ref().map(|m| {
                                let metadata_stem =
                                    m.file_stem().and_then(|s| s.to_str()).unwrap_or("metadata");
                                let metadata_ext =
                                    m.extension().and_then(|e| e.to_str()).unwrap_or("json");
                                output_base.join(format!(
                                    "{}_{}.{}",
                                    metadata_stem, file_stem, metadata_ext
                                ))
                            });

                            write_results(
                                file_args,
                                speeches_result,
                                &process_samples,
                                compute_time.as_secs_f64(),
                            )?;

                            info!("Completed: {} in {:?}", file_stem, compute_time);
                            Ok(())
                        }
                        VadModel::Pyannote => {
                            let pyannote = pyannote_vad_ort::PyAnnote::new(
                                vad_params.clone(),
                                execution_providers.clone(),
                                args.model_path.clone(),
                            )?;

                            let start = std::time::Instant::now();
                            let mut pyannote_vad_iterator = pyannote_vad_iter::PyAnnoteVadIter::new(
                                pyannote,
                                vad_params.clone(),
                            );
                            let speeches_result =
                                pyannote_vad_iterator.process(&process_samples)?;
                            let compute_time = start.elapsed();

                            // Create args for this file
                            let mut file_args = args.clone();
                            file_args.output = file_output_path.clone();
                            file_args.metadata = args.metadata.as_ref().map(|m| {
                                let metadata_stem =
                                    m.file_stem().and_then(|s| s.to_str()).unwrap_or("metadata");
                                let metadata_ext =
                                    m.extension().and_then(|e| e.to_str()).unwrap_or("json");
                                output_base.join(format!(
                                    "{}_{}.{}",
                                    metadata_stem, file_stem, metadata_ext
                                ))
                            });

                            write_results(
                                file_args,
                                speeches_result,
                                &process_samples,
                                compute_time.as_secs_f64(),
                            )?;

                            info!("Completed: {} in {:?}", file_stem, compute_time);
                            Ok(())
                        }
                        VadModel::PulseVad => {
                            let pulsevad = pulsevad_ort::PulseVad::new(
                                execution_providers.clone(),
                                args.model_path.clone(),
                                args.debug,
                            )?;

                            let start = std::time::Instant::now();
                            let mut vad_iterator =
                                pulsevad_iter::PulseVadIter::new(pulsevad, vad_params.clone());
                            let speeches_result = vad_iterator.process(&process_samples)?;
                            let compute_time = start.elapsed();

                            let mut file_args = args.clone();
                            file_args.output = file_output_path.clone();
                            file_args.metadata = args.metadata.as_ref().map(|m| {
                                let metadata_stem =
                                    m.file_stem().and_then(|s| s.to_str()).unwrap_or("metadata");
                                let metadata_ext =
                                    m.extension().and_then(|e| e.to_str()).unwrap_or("json");
                                output_base.join(format!(
                                    "{}_{}.{}",
                                    metadata_stem, file_stem, metadata_ext
                                ))
                            });

                            write_results(
                                file_args,
                                speeches_result,
                                &process_samples,
                                compute_time.as_secs_f64(),
                            )?;

                            info!("Completed: {} in {:?}", file_stem, compute_time);
                            Ok(())
                        }
                    }
                }
            }
        })
        .collect();

    // Check for errors
    let mut error_count = 0;
    for result in results {
        if let Err(e) = result {
            log::error!("Error processing file: {}", e);
            error_count += 1;
        }
    }

    if error_count > 0 {
        return Err(anyhow::anyhow!("Failed to process {} files", error_count));
    }

    info!("Successfully processed all {} files", audio_files.len());
    Ok(())
}

fn main() -> Result<()> {
    // Parse the arguments
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    if let Some(info) = args.print_model_info {
        return print_model_info(args.model_path.clone(), info);
    }

    // Validate that at least one input method is provided
    if args.process_audio.is_none() && args.process_folder.is_none() {
        return Err(anyhow::anyhow!(
            "Either --process-audio or --process-folder must be provided"
        ));
    }
    anyhow::ensure!(
        args.threshold.is_finite() && (0.0..=1.0).contains(&args.threshold),
        "threshold must be between 0 and 1"
    );
    anyhow::ensure!(
        args.sample_rate > 0,
        "sample rate must be greater than zero"
    );
    u32::try_from(args.sample_rate).context("sample rate exceeds u32")?;

    if args.runtime == Runtime::Onnxruntime {
        let dylib_path = args
            .dylib_path
            .as_deref()
            .context("--dylib-path is required for ONNX Runtime")?;
        let dylib_path = dylib_path
            .to_str()
            .context("ONNX Runtime library path is not valid UTF-8")?;
        ort::init_from(dylib_path)?.commit();
    }

    let mut execution_providers: Vec<ExecutionProviderDispatch> = vec![CPU::default().build()];

    if args.cuda {
        execution_providers.insert(0, CUDA::default().build());
    }

    if args.coreml {
        execution_providers.insert(0, CoreML::default().build());
    }

    if args.trt {
        execution_providers.insert(0, TensorRT::default().build());
    }

    // Process single file or folder
    if let Some(process_audio_path) = args.process_audio.clone() {
        // Single file processing
        process_single_file(args, process_audio_path, execution_providers)
    } else if let Some(process_folder_path) = args.process_folder.clone() {
        // Folder processing with parallelization
        process_folder(args, process_folder_path, execution_providers)
    } else {
        Err(anyhow::anyhow!(
            "Either --process-audio or --process-folder must be provided"
        ))
    }
}

fn write_results(
    args: Args,
    speeches: &[utils::TimeStamp],
    samples: &[f32],
    compute_seconds: f64,
) -> Result<()> {
    info!("Speeches: {}", speeches.len());

    let output_path = args.output;
    match args.output_type {
        OutputType::Files => std::fs::create_dir_all(&output_path)?,
        OutputType::Concatenated => {
            if let Some(parent) = output_path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
        }
    }

    let mut intervals: Vec<IntervalMetadata> = Vec::new();
    let output_sample_rate = u32::try_from(args.sample_rate).context("sample rate exceeds u32")?;

    match args.output_format {
        OutputFormat::Wav => {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: output_sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };

            match args.output_type {
                OutputType::Files => {
                    for (idx, speech) in speeches.iter().enumerate() {
                        let current_ts = Utc::now().timestamp_millis();
                        let filename = format!("{}_{}.wav", current_ts, idx);
                        let filepath = output_path.join(&filename);
                        let segment = checked_segment(samples, speech)?;
                        let output_samples = prepare_output_samples(segment, args.sample_rate)?;
                        write_wav(&filepath, &output_samples, spec)?;

                        let duration_seconds =
                            output_samples.len() as f64 / output_sample_rate as f64;
                        intervals.push(IntervalMetadata {
                            filename,
                            duration: format!("{:.6}", duration_seconds),
                        });
                    }
                }
                OutputType::Concatenated => {
                    let gathered_speeches = gather_speeches(samples, speeches)?;
                    let output_samples =
                        prepare_output_samples(&gathered_speeches, args.sample_rate)?;
                    write_wav(&output_path, &output_samples, spec)?;

                    let duration_seconds = output_samples.len() as f64 / output_sample_rate as f64;
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
            let extension = args.output_format.extension();
            match args.output_type {
                OutputType::Files => {
                    for (idx, speech) in speeches.iter().enumerate() {
                        let current_ts = Utc::now().timestamp_millis();
                        let filename = format!("{}_{}.{}", current_ts, idx, extension);
                        let filepath = output_path.join(&filename);
                        let segment = checked_segment(samples, speech)?;
                        let output_samples = prepare_output_samples(segment, args.sample_rate)?;

                        write_opus(filepath, &output_samples, output_sample_rate)?;

                        let duration_seconds =
                            output_samples.len() as f64 / output_sample_rate as f64;
                        intervals.push(IntervalMetadata {
                            filename,
                            duration: format!("{:.6}", duration_seconds),
                        });
                    }
                }
                OutputType::Concatenated => {
                    let gathered_speeches = gather_speeches(samples, speeches)?;
                    let output_samples =
                        prepare_output_samples(&gathered_speeches, args.sample_rate)?;

                    write_opus(&output_path, &output_samples, output_sample_rate)?;

                    let duration_seconds = output_samples.len() as f64 / output_sample_rate as f64;
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

fn checked_segment<'a>(samples: &'a [f32], timestamp: &utils::TimeStamp) -> Result<&'a [f32]> {
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

fn gather_speeches(samples: &[f32], speeches: &[utils::TimeStamp]) -> Result<Vec<f32>> {
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
    if output_sample_rate == utils::VAD_SAMPLE_RATE {
        Ok(samples.to_vec())
    } else {
        resampler::resample(samples, utils::VAD_SAMPLE_RATE, output_sample_rate)
    }
}

fn write_wav(path: &std::path::Path, samples: &[f32], spec: hound::WavSpec) -> Result<()> {
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(value)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_runtime_requires_a_dynamic_library_path() {
        let result = Args::try_parse_from([
            "extract-speech",
            "--runtime",
            "onnxruntime",
            "--model-path",
            "model.onnx",
            "--process-audio",
            "audio.wav",
        ]);

        assert!(result.is_err());
    }

    #[test]
    fn output_defaults_to_output_directory() {
        let args = Args::try_parse_from([
            "extract-speech",
            "--model-path",
            "model.onnx",
            "--process-audio",
            "audio.wav",
        ])
        .unwrap();

        assert_eq!(args.output, PathBuf::from("output"));
    }

    #[test]
    fn pulsevad_is_an_accepted_model_name() {
        let args = Args::try_parse_from([
            "extract-speech",
            "--vad-model",
            "pulsevad",
            "--model-path",
            "model.onnx",
            "--process-audio",
            "audio.wav",
        ])
        .unwrap();

        assert_eq!(args.vad_model, VadModel::PulseVad);
        assert_eq!(make_vad_params(&args).min_speech_duration_ms, 100);
    }

    #[test]
    fn invalid_speech_interval_is_rejected() {
        let samples = vec![0.0; 10];
        let timestamp = utils::TimeStamp { start: 8, end: 11 };

        assert!(checked_segment(&samples, &timestamp).is_err());
    }

    #[test]
    fn output_samples_are_resampled_to_requested_rate() {
        let samples = vec![0.0; utils::VAD_SAMPLE_RATE];
        let output = prepare_output_samples(&samples, 8_000).unwrap();

        assert!((output.len() as isize - 8_000).abs() <= 2);
    }
}
