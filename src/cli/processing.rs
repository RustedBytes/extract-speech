use std::{path::Path, path::PathBuf, time::Instant};

use anyhow::Result;
use extract_speech::{audio::load_samples_from_audio_file, Detector};
use log::{debug, info};
use ort::ep::ExecutionProviderDispatch;
use rayon::prelude::*;

use super::{
    args::{Args, OutputType, Runtime},
    output::write_results,
};

pub(super) fn process_single_file(
    args: &Args,
    process_audio_path: &Path,
    execution_providers: &[ExecutionProviderDispatch],
) -> Result<()> {
    let start = Instant::now();
    let process_samples = load_samples_from_audio_file(process_audio_path)?;
    info!(
        "Number of samples (process_audio): {}",
        process_samples.len()
    );

    let source_samples = args
        .source_audio
        .as_deref()
        .map(load_samples_from_audio_file)
        .transpose()?;
    if let Some(source_samples) = &source_samples {
        info!("Number of samples (source_audio): {}", source_samples.len());
        anyhow::ensure!(
            process_samples.len() == source_samples.len(),
            "The number of samples in the source and process audio files should be the same."
        );
    }
    let output_samples = source_samples.as_deref().unwrap_or(&process_samples);
    info!("Retrieved audio files in: {:?}", start.elapsed());

    let (speeches, compute_seconds) = detect(args, &process_samples, execution_providers)?;
    write_results(args, &speeches, output_samples, compute_seconds)
}

pub(super) fn process_folder(
    args: &Args,
    process_folder_path: &Path,
    execution_providers: &[ExecutionProviderDispatch],
) -> Result<()> {
    info!("Processing folder: {}", process_folder_path.display());
    let audio_files = collect_audio_files(process_folder_path)?;
    info!("Found {} audio files to process", audio_files.len());
    std::fs::create_dir_all(&args.output)?;

    let results: Vec<Result<()>> = audio_files
        .par_iter()
        .map(|audio_path| {
            let file_stem = audio_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("unknown");
            info!("Processing: {}", audio_path.display());

            let samples = load_samples_from_audio_file(audio_path)?;
            let mut file_args = args.clone();
            file_args.output = match args.output_type {
                OutputType::Files => {
                    let directory = args.output.join(file_stem);
                    std::fs::create_dir_all(&directory)?;
                    directory
                }
                OutputType::Concatenated => {
                    args.output
                        .join(format!("{}.{}", file_stem, args.output_format.extension()))
                }
            };
            file_args.metadata = metadata_path(args, file_stem);

            let (speeches, compute_seconds) = detect(&file_args, &samples, execution_providers)?;
            write_results(&file_args, &speeches, &samples, compute_seconds)?;
            info!("Completed: {file_stem} in {compute_seconds:.3}s");
            Ok(())
        })
        .collect();

    let errors: Vec<_> = results.into_iter().filter_map(Result::err).collect();
    for error in &errors {
        log::error!("Error processing file: {error}");
    }
    anyhow::ensure!(
        errors.is_empty(),
        "Failed to process {} files",
        errors.len()
    );
    info!("Successfully processed all {} files", audio_files.len());
    Ok(())
}

fn detect(
    args: &Args,
    samples: &[f32],
    execution_providers: &[ExecutionProviderDispatch],
) -> Result<(Vec<extract_speech::SpeechSegment>, f64)> {
    if args.runtime == Runtime::Candle {
        debug!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
    }

    let load_start = Instant::now();
    let builder = Detector::builder(args.model_path()?)
        .model(args.vad_model.library_model())
        .runtime(args.runtime.library_runtime())
        .parameters(args.vad_params())
        .candle_device(candle_core::Device::Cpu)
        .execution_providers(execution_providers.to_vec());
    let mut detector = builder.build()?;
    info!("Loaded the model in: {:?}", load_start.elapsed());

    let inference_start = Instant::now();
    let speeches = detector.detect(samples)?;
    let compute_time = inference_start.elapsed();
    info!("Inference time: {compute_time:?}");
    Ok((speeches, compute_time.as_secs_f64()))
}

fn collect_audio_files(folder_path: &Path) -> Result<Vec<PathBuf>> {
    const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac"];
    let mut audio_files = Vec::new();
    for entry in std::fs::read_dir(folder_path)? {
        let path = entry?.path();
        let is_audio = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| AUDIO_EXTENSIONS.contains(&extension.to_lowercase().as_str()));
        if path.is_file() && is_audio {
            audio_files.push(path);
        }
    }
    anyhow::ensure!(
        !audio_files.is_empty(),
        "No audio files found in folder: {}",
        folder_path.display()
    );
    audio_files.sort();
    Ok(audio_files)
}

fn metadata_path(args: &Args, file_stem: &str) -> Option<PathBuf> {
    args.metadata.as_ref().map(|path| {
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("metadata");
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json");
        args.output.join(format!("{stem}_{file_stem}.{extension}"))
    })
}
