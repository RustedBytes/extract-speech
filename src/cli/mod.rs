mod args;
mod output;
mod processing;

use anyhow::{Context, Result};
use clap::Parser;
use ort::ep::{CoreML, ExecutionProviderDispatch, TensorRT, CPU, CUDA};
use tracing_subscriber::EnvFilter;

use args::{Args, ModelInfo, Runtime};
use processing::{process_folder, process_single_file};

pub(crate) fn run() -> Result<()> {
    let args = Args::parse();
    init_logging(args.debug)?;

    if let Some(info) = args.print_model_info {
        return print_model_info(&args.model_path, info);
    }
    args.validate()?;

    let execution_providers = execution_providers(&args);
    if args.runtime == Runtime::Onnxruntime {
        let dylib_path = args
            .dylib_path
            .as_deref()
            .context("--dylib-path is required for ONNX Runtime")?;
        extract_speech::init_onnx_runtime(dylib_path, execution_providers.clone())?;
    }

    if let Some(process_audio_path) = &args.process_audio {
        process_single_file(&args, process_audio_path, &execution_providers)
    } else if let Some(process_folder_path) = &args.process_folder {
        process_folder(&args, process_folder_path, &execution_providers)
    } else {
        unreachable!("validated input source")
    }
}

fn init_logging(debug: bool) -> Result<()> {
    let default_filter = if debug { "debug" } else { "info" };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()
        .map_err(|error| anyhow::anyhow!("failed to initialize logging: {error}"))
}

fn execution_providers(args: &Args) -> Vec<ExecutionProviderDispatch> {
    let mut providers = vec![CPU::default().build()];
    if args.cuda {
        providers.insert(0, CUDA::default().build());
    }
    if args.coreml {
        providers.insert(0, CoreML::default().build());
    }
    if args.trt {
        providers.insert(0, TensorRT::default().build());
    }
    providers
}

fn print_model_info(model_path: &std::path::Path, info: ModelInfo) -> Result<()> {
    let model = candle_onnx::read_file(model_path)?;
    let graph = model
        .graph
        .as_ref()
        .context("ONNX model contains no graph")?;
    match info {
        ModelInfo::Graph => println!("{model:#?}"),
        ModelInfo::Nodes => {
            for node in &graph.node {
                println!("{node:#?}");
            }
        }
        ModelInfo::IO => {
            for input in &graph.input {
                println!("input: {input:#?}");
            }
            for output in &graph.output {
                println!("output: {output:#?}");
            }
        }
    }
    Ok(())
}
