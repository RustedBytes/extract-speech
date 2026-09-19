use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use extract_speech::{
    download::ModelAsset, Model, Runtime as LibraryRuntime, VadParams, VAD_SAMPLE_RATE,
};

#[derive(Clone, Debug, PartialEq, Eq, Subcommand)]
pub(super) enum Command {
    /// Download a checksum-verified model bundle or ONNX Runtime
    Download(DownloadArgs),
}

#[derive(Clone, Debug, PartialEq, Eq, clap::Args)]
pub(super) struct DownloadArgs {
    /// Asset to download
    #[arg(value_enum)]
    pub(super) asset: DownloadAsset,

    /// Cache directory (defaults to `EXTRACT_SPEECH_CACHE_DIR` or the platform cache)
    #[arg(long)]
    pub(super) cache_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum DownloadAsset {
    /// Every supported model bundle
    #[value(alias = "all-models")]
    All,
    /// Compatible CPU ONNX Runtime distribution
    #[value(name = "onnxruntime", alias = "onnx-runtime", alias = "ort")]
    OnnxRuntime,
    #[value(name = "silero-v6", alias = "silero")]
    SileroV6,
    #[value(name = "silero-v5")]
    SileroV5,
    #[value(name = "pyannote-segmentation", alias = "pyannote")]
    PyAnnoteSegmentation,
    #[value(name = "pulsevad-fp32", alias = "pulsevad")]
    PulseVadFp32,
    #[value(name = "pulsevad-int8")]
    PulseVadInt8,
    #[value(name = "fsmn-vad-fp32", alias = "fsmn")]
    FsmnVadFp32,
    #[value(name = "fsmn-vad-int8", alias = "fsmn-int8")]
    FsmnVadInt8,
    #[value(name = "ten-vad", alias = "ten")]
    TenVad,
    #[value(name = "marblenet-fp32", alias = "marblenet", alias = "marble-net")]
    MarbleNetFp32,
    #[value(name = "marblenet-int8", alias = "marble-net-int8")]
    MarbleNetInt8,
}

impl DownloadAsset {
    pub(super) fn model_asset(self) -> Option<ModelAsset> {
        match self {
            Self::All | Self::OnnxRuntime => None,
            Self::SileroV6 => Some(ModelAsset::SileroV6),
            Self::SileroV5 => Some(ModelAsset::SileroV5),
            Self::PyAnnoteSegmentation => Some(ModelAsset::PyAnnoteSegmentation),
            Self::PulseVadFp32 => Some(ModelAsset::PulseVadFp32),
            Self::PulseVadInt8 => Some(ModelAsset::PulseVadInt8),
            Self::FsmnVadFp32 => Some(ModelAsset::FsmnVadFp32),
            Self::FsmnVadInt8 => Some(ModelAsset::FsmnVadInt8),
            Self::TenVad => Some(ModelAsset::TenVad),
            Self::MarbleNetFp32 => Some(ModelAsset::MarbleNetFp32),
            Self::MarbleNetInt8 => Some(ModelAsset::MarbleNetInt8),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum OutputFormat {
    Wav,
    Opus,
    Ogg,
}

impl OutputFormat {
    pub(super) fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Opus => "opus",
            Self::Ogg => "ogg",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum ModelInfo {
    Graph,
    Nodes,
    IO,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum Runtime {
    Candle,
    Onnxruntime,
}

impl Runtime {
    pub(super) fn library_runtime(self) -> LibraryRuntime {
        match self {
            Self::Candle => LibraryRuntime::Candle,
            Self::Onnxruntime => LibraryRuntime::OnnxRuntime,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum VadModel {
    Silero,
    Pyannote,
    #[value(name = "pulsevad", alias = "pulse-vad")]
    PulseVad,
    #[value(name = "fsmn", alias = "fsmn-vad")]
    Fsmn,
    #[value(name = "ten", alias = "ten-vad")]
    Ten,
    #[value(name = "marblenet", alias = "marble-net")]
    MarbleNet,
}

impl VadModel {
    pub(super) fn library_model(self) -> Model {
        match self {
            Self::Silero => Model::Silero,
            Self::Pyannote => Model::PyAnnote,
            Self::PulseVad => Model::PulseVad,
            Self::Fsmn => Model::Fsmn,
            Self::Ten => Model::Ten,
            Self::MarbleNet => Model::MarbleNet,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum OutputType {
    Files,
    Concatenated,
}

#[derive(Parser, Debug, Clone)]
#[command(version, long_about = None)]
#[allow(clippy::struct_excessive_bools)] // Independent CLI switches map directly to runtime flags.
pub(super) struct Args {
    #[command(subcommand)]
    pub(super) command: Option<Command>,

    /// Print the model info
    #[arg(long)]
    pub(super) print_model_info: Option<ModelInfo>,

    /// A path to VAD model
    #[arg(long)]
    pub(super) model_path: Option<PathBuf>,

    /// Runtime variant
    #[arg(long, value_enum, default_value_t = Runtime::Candle)]
    pub(super) runtime: Runtime,

    /// VAD model type
    #[arg(long, value_enum, default_value_t = VadModel::Silero)]
    pub(super) vad_model: VadModel,

    /// Path to the ONNX runtime dynamic library
    #[arg(long, required_if_eq("runtime", "onnxruntime"))]
    pub(super) dylib_path: Option<PathBuf>,

    /// The source audio file path
    #[arg(long, conflicts_with = "process_folder")]
    pub(super) source_audio: Option<PathBuf>,

    /// The audio file path to process in VAD stage
    #[arg(long, conflicts_with = "process_folder")]
    pub(super) process_audio: Option<PathBuf>,

    /// The folder path containing audio files to process in VAD stage
    #[arg(long, conflicts_with = "process_audio")]
    pub(super) process_folder: Option<PathBuf>,

    /// The path of a final result file or directory
    #[arg(long, default_value = "output")]
    pub(super) output: PathBuf,

    /// Path to write metadata JSON file
    #[arg(long)]
    pub(super) metadata: Option<PathBuf>,

    /// The output type
    #[arg(long, value_enum, default_value_t = OutputType::Files)]
    pub(super) output_type: OutputType,

    /// The result format
    #[arg(long, value_enum, default_value_t = OutputFormat::Wav)]
    pub(super) output_format: OutputFormat,

    /// VAD threshold
    #[arg(long, default_value = "0.7")]
    pub(super) threshold: f32,

    /// Sample rate of final output
    #[arg(long, default_value = "16000")]
    pub(super) sample_rate: usize,

    /// Enable `TensorRT`
    #[arg(long, default_value_t = false)]
    pub(super) trt: bool,

    /// Enable CUDA
    #[arg(long, default_value_t = false)]
    pub(super) cuda: bool,

    /// Enable `CoreML`
    #[arg(long, default_value_t = false)]
    pub(super) coreml: bool,

    /// Use debug-level logging when `RUST_LOG` is unset
    #[arg(long, default_value_t = false)]
    pub(super) debug: bool,
}

impl Args {
    pub(super) fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.model_path.is_some(), "--model-path must be provided");
        anyhow::ensure!(
            self.process_audio.is_some() || self.process_folder.is_some(),
            "Either --process-audio or --process-folder must be provided"
        );
        anyhow::ensure!(
            self.threshold.is_finite() && (0.0..=1.0).contains(&self.threshold),
            "threshold must be between 0 and 1"
        );
        anyhow::ensure!(
            self.sample_rate > 0,
            "sample rate must be greater than zero"
        );
        u32::try_from(self.sample_rate).context("sample rate exceeds u32")?;
        Ok(())
    }

    pub(super) fn model_path(&self) -> Result<&std::path::Path> {
        self.model_path
            .as_deref()
            .context("--model-path must be provided")
    }

    pub(super) fn vad_params(&self) -> VadParams {
        VadParams {
            frame_size: if self.vad_model == VadModel::Ten {
                16
            } else {
                VadParams::default().frame_size
            },
            sample_rate: VAD_SAMPLE_RATE,
            threshold: self.threshold,
            min_speech_duration_ms: if self.vad_model == VadModel::PulseVad {
                100
            } else {
                VadParams::default().min_speech_duration_ms
            },
            debug: self.debug,
            ..Default::default()
        }
    }
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
    fn download_does_not_require_processing_arguments() {
        let args = Args::try_parse_from(["extract-speech", "download", "silero"])
            .expect("download subcommand should parse without inference arguments");
        assert_eq!(
            args.command,
            Some(Command::Download(DownloadArgs {
                asset: DownloadAsset::SileroV6,
                cache_dir: None,
            }))
        );
    }

    #[test]
    fn download_accepts_a_cache_directory_and_asset_aliases() {
        let args =
            Args::try_parse_from(["extract-speech", "download", "ort", "--cache-dir", "cache"])
                .expect("download aliases and options should parse");
        assert_eq!(
            args.command,
            Some(Command::Download(DownloadArgs {
                asset: DownloadAsset::OnnxRuntime,
                cache_dir: Some(PathBuf::from("cache")),
            }))
        );
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
    fn model_aliases_are_accepted() {
        for (name, expected) in [
            ("pulsevad", VadModel::PulseVad),
            ("fsmn-vad", VadModel::Fsmn),
            ("ten-vad", VadModel::Ten),
            ("marble-net", VadModel::MarbleNet),
        ] {
            let args = Args::try_parse_from([
                "extract-speech",
                "--vad-model",
                name,
                "--model-path",
                "model.onnx",
                "--process-audio",
                "audio.wav",
            ])
            .unwrap();
            assert_eq!(args.vad_model, expected);
        }
    }

    #[test]
    fn model_specific_parameters_are_preserved() {
        let mut args = Args::try_parse_from([
            "extract-speech",
            "--model-path",
            "model.onnx",
            "--process-audio",
            "audio.wav",
        ])
        .unwrap();
        args.vad_model = VadModel::PulseVad;
        assert_eq!(args.vad_params().min_speech_duration_ms, 100);
        args.vad_model = VadModel::Ten;
        assert_eq!(args.vad_params().frame_size, 16);
    }
}
