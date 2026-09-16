//! Python bindings exposed through `PyO3`.

// Sample offsets are intentionally converted to seconds for the Python API.
#![allow(clippy::cast_precision_loss)]

use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

use anyhow::{bail, Context};
use pyo3::{
    buffer::PyBuffer,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyModule},
};

use crate::{
    download::{self, AssetManager, ModelAsset},
    init_onnx_runtime,
    ort::ep::{ExecutionProviderDispatch, CPU},
    Detector as RustDetector, Model, Runtime, SpeechSegment, VadParams, SAMPLE_RATE,
};

static ONNX_RUNTIME_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

#[pyclass(name = "SpeechSegment", frozen, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct PySpeechSegment {
    #[pyo3(get)]
    start: usize,
    #[pyo3(get)]
    end: usize,
}

#[pymethods]
impl PySpeechSegment {
    #[getter]
    fn duration_samples(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    #[getter]
    fn start_seconds(&self) -> f64 {
        self.start as f64 / SAMPLE_RATE as f64
    }

    #[getter]
    fn end_seconds(&self) -> f64 {
        self.end as f64 / SAMPLE_RATE as f64
    }

    #[getter]
    fn duration_seconds(&self) -> f64 {
        self.duration_samples() as f64 / SAMPLE_RATE as f64
    }

    fn __repr__(&self) -> String {
        format!("SpeechSegment(start={}, end={})", self.start, self.end)
    }
}

impl From<SpeechSegment> for PySpeechSegment {
    fn from(value: SpeechSegment) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

#[derive(Clone, Copy)]
struct ParsedModel {
    model: Model,
    asset: ModelAsset,
    name: &'static str,
}

#[derive(Clone, Copy)]
struct DetectorOptions {
    threshold: f32,
    min_silence_duration_ms: usize,
    speech_pad_ms: usize,
    min_speech_duration_ms: usize,
    max_speech_duration_s: Option<f32>,
    debug: bool,
}

impl DetectorOptions {
    fn parameters(self) -> VadParams {
        VadParams {
            threshold: self.threshold,
            min_silence_duration_ms: self.min_silence_duration_ms,
            speech_pad_ms: self.speech_pad_ms,
            min_speech_duration_ms: self.min_speech_duration_ms,
            max_speech_duration_s: self.max_speech_duration_s.unwrap_or(f32::INFINITY),
            debug: self.debug,
            ..VadParams::default()
        }
    }
}

#[pyclass(name = "Detector")]
struct PyDetector {
    inner: Mutex<RustDetector>,
    model: &'static str,
    runtime: &'static str,
}

#[pymethods]
impl PyDetector {
    #[new]
    #[pyo3(signature = (
        model_path,
        *,
        model = "silero",
        runtime = "candle",
        onnx_runtime_path = None,
        threshold = 0.5,
        min_silence_duration_ms = 100,
        speech_pad_ms = 30,
        min_speech_duration_ms = 250,
        max_speech_duration_s = None,
        debug = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model_path: PathBuf,
        model: &str,
        runtime: &str,
        onnx_runtime_path: Option<PathBuf>,
        threshold: f32,
        min_silence_duration_ms: usize,
        speech_pad_ms: usize,
        min_speech_duration_ms: usize,
        max_speech_duration_s: Option<f32>,
        debug: bool,
    ) -> PyResult<Self> {
        let parsed_model = parse_model(model, false)?;
        let parsed_runtime = parse_runtime(runtime)?;
        validate_runtime(parsed_model, parsed_runtime, false)?;
        if parsed_runtime == Runtime::OnnxRuntime && onnx_runtime_path.is_none() {
            return Err(PyValueError::new_err(
                "onnx_runtime_path is required when runtime='onnxruntime'",
            ));
        }
        let options = DetectorOptions {
            threshold,
            min_silence_duration_ms,
            speech_pad_ms,
            min_speech_duration_ms,
            max_speech_duration_s,
            debug,
        };

        let detector = py
            .detach(move || {
                if let Some(path) = onnx_runtime_path {
                    ensure_onnx_runtime(&path)?;
                }
                build_detector(model_path, parsed_model.model, parsed_runtime, options)
            })
            .map_err(runtime_error)?;

        Ok(Self {
            inner: Mutex::new(detector),
            model: parsed_model.name,
            runtime: runtime_name(parsed_runtime),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (
        model = "silero",
        *,
        runtime = "candle",
        quantized = false,
        cache_dir = None,
        threshold = 0.5,
        min_silence_duration_ms = 100,
        speech_pad_ms = 30,
        min_speech_duration_ms = 250,
        max_speech_duration_s = None,
        debug = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_pretrained(
        py: Python<'_>,
        model: &str,
        runtime: &str,
        quantized: bool,
        cache_dir: Option<PathBuf>,
        threshold: f32,
        min_silence_duration_ms: usize,
        speech_pad_ms: usize,
        min_speech_duration_ms: usize,
        max_speech_duration_s: Option<f32>,
        debug: bool,
    ) -> PyResult<Self> {
        let parsed_model = parse_model(model, quantized)?;
        let parsed_runtime = parse_runtime(runtime)?;
        validate_runtime(parsed_model, parsed_runtime, quantized)?;
        let options = DetectorOptions {
            threshold,
            min_silence_duration_ms,
            speech_pad_ms,
            min_speech_duration_ms,
            max_speech_duration_s,
            debug,
        };

        let detector = py
            .detach(move || {
                let assets = asset_manager(cache_dir)?;
                let model_path = match parsed_runtime {
                    Runtime::Candle => assets.model(parsed_model.asset)?.model_path().to_owned(),
                    Runtime::OnnxRuntime => {
                        let bundle = assets.onnx_bundle(parsed_model.asset)?;
                        ensure_onnx_runtime(bundle.runtime().library_path())?;
                        bundle.model().model_path().to_owned()
                    }
                };
                build_detector(model_path, parsed_model.model, parsed_runtime, options)
            })
            .map_err(runtime_error)?;

        Ok(Self {
            inner: Mutex::new(detector),
            model: parsed_model.name,
            runtime: runtime_name(parsed_runtime),
        })
    }

    /// Detect speech in mono, normalized float32 PCM sampled at 16 kHz.
    fn detect(&self, py: Python<'_>, samples: &Bound<'_, PyAny>) -> PyResult<Vec<PySpeechSegment>> {
        let samples = extract_samples(py, samples)?;
        let segments = py
            .detach(|| {
                let mut detector = self
                    .inner
                    .lock()
                    .map_err(|_| anyhow::anyhow!("detector lock is poisoned"))?;
                detector.detect(&samples)
            })
            .map_err(runtime_error)?;
        Ok(segments.into_iter().map(Into::into).collect())
    }

    #[getter]
    fn model(&self) -> &'static str {
        self.model
    }

    #[getter]
    fn runtime(&self) -> &'static str {
        self.runtime
    }

    #[classattr]
    const SAMPLE_RATE: usize = SAMPLE_RATE;

    fn __repr__(&self) -> String {
        format!(
            "Detector(model='{}', runtime='{}')",
            self.model, self.runtime
        )
    }
}

#[pyfunction]
#[pyo3(signature = (model, *, quantized = false, cache_dir = None))]
fn download_model(
    py: Python<'_>,
    model: &str,
    quantized: bool,
    cache_dir: Option<PathBuf>,
) -> PyResult<PathBuf> {
    let parsed = parse_model(model, quantized)?;
    py.detach(move || {
        Ok(asset_manager(cache_dir)?
            .model(parsed.asset)?
            .model_path()
            .to_owned())
    })
    .map_err(runtime_error)
}

#[pyfunction]
#[pyo3(signature = (*, cache_dir = None))]
fn download_all_models(
    py: Python<'_>,
    cache_dir: Option<PathBuf>,
) -> PyResult<Vec<(String, PathBuf)>> {
    py.detach(move || {
        asset_manager(cache_dir)?
            .all_models()?
            .into_iter()
            .map(|files| {
                Ok((
                    asset_name(files.asset()).to_owned(),
                    files.model_path().to_owned(),
                ))
            })
            .collect::<anyhow::Result<_>>()
    })
    .map_err(runtime_error)
}

#[pyfunction]
#[pyo3(signature = (*, cache_dir = None))]
fn download_onnx_runtime(py: Python<'_>, cache_dir: Option<PathBuf>) -> PyResult<PathBuf> {
    py.detach(move || {
        Ok(asset_manager(cache_dir)?
            .onnx_runtime()?
            .library_path()
            .to_owned())
    })
    .map_err(runtime_error)
}

#[pyfunction(name = "initialize_onnx_runtime")]
fn py_initialize_onnx_runtime(py: Python<'_>, path: PathBuf) -> PyResult<()> {
    py.detach(move || ensure_onnx_runtime(&path))
        .map_err(runtime_error)
}

#[pyfunction]
fn default_cache_dir() -> PyResult<PathBuf> {
    download::default_cache_dir().map_err(runtime_error)
}

#[pymodule]
fn extract_speech(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDetector>()?;
    module.add_class::<PySpeechSegment>()?;
    module.add_function(wrap_pyfunction!(download_model, module)?)?;
    module.add_function(wrap_pyfunction!(download_all_models, module)?)?;
    module.add_function(wrap_pyfunction!(download_onnx_runtime, module)?)?;
    module.add_function(wrap_pyfunction!(py_initialize_onnx_runtime, module)?)?;
    module.add_function(wrap_pyfunction!(default_cache_dir, module)?)?;
    module.add("SAMPLE_RATE", SAMPLE_RATE)?;
    module.add("ONNX_RUNTIME_VERSION", download::ONNX_RUNTIME_VERSION)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

fn build_detector(
    model_path: PathBuf,
    model: Model,
    runtime: Runtime,
    options: DetectorOptions,
) -> anyhow::Result<RustDetector> {
    RustDetector::builder(model_path)
        .model(model)
        .runtime(runtime)
        .parameters(options.parameters())
        .build()
}

fn asset_manager(cache_dir: Option<PathBuf>) -> anyhow::Result<AssetManager> {
    cache_dir.map_or_else(AssetManager::default_cache, |path| {
        Ok(AssetManager::new(path))
    })
}

fn ensure_onnx_runtime(path: &Path) -> anyhow::Result<()> {
    let path = path
        .canonicalize()
        .with_context(|| format!("failed to resolve ONNX Runtime library {}", path.display()))?;
    let mut initialized = ONNX_RUNTIME_PATH
        .lock()
        .map_err(|_| anyhow::anyhow!("ONNX Runtime initialization lock is poisoned"))?;
    if let Some(existing) = initialized.as_ref() {
        if existing == &path {
            return Ok(());
        }
        bail!(
            "ONNX Runtime is already initialized from {}; cannot switch to {} in the same Python process",
            existing.display(),
            path.display()
        );
    }

    let providers: Vec<ExecutionProviderDispatch> = vec![CPU::default().build()];
    init_onnx_runtime(&path, providers)?;
    *initialized = Some(path);
    Ok(())
}

fn extract_samples(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let samples = if let Ok(buffer) = PyBuffer::<f32>::get(value) {
        let mut samples = vec![0.0; buffer.item_count()];
        buffer.copy_to_slice(py, &mut samples)?;
        samples
    } else {
        value.extract::<Vec<f32>>().map_err(|_| {
            PyTypeError::new_err(
                "samples must be a sequence of floats or a contiguous float32 buffer",
            )
        })?
    };

    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(PyValueError::new_err(
            "samples must contain only finite values",
        ));
    }
    Ok(samples)
}

fn parse_model(value: &str, quantized: bool) -> PyResult<ParsedModel> {
    let normalized = normalize_name(value);
    let parsed = match normalized.as_str() {
        "silero" | "silero-v5" if !quantized => ParsedModel {
            model: Model::Silero,
            asset: ModelAsset::SileroV5,
            name: "silero",
        },
        "pyannote" | "pyannote-segmentation" if !quantized => ParsedModel {
            model: Model::PyAnnote,
            asset: ModelAsset::PyAnnoteSegmentation,
            name: "pyannote",
        },
        "pulsevad" | "pulse-vad" => ParsedModel {
            model: Model::PulseVad,
            asset: if quantized {
                ModelAsset::PulseVadInt8
            } else {
                ModelAsset::PulseVadFp32
            },
            name: "pulsevad",
        },
        "fsmn" | "fsmn-vad" => ParsedModel {
            model: Model::Fsmn,
            asset: if quantized {
                ModelAsset::FsmnVadInt8
            } else {
                ModelAsset::FsmnVadFp32
            },
            name: "fsmn",
        },
        "ten" | "ten-vad" if !quantized => ParsedModel {
            model: Model::Ten,
            asset: ModelAsset::TenVad,
            name: "ten",
        },
        "marblenet" | "marble-net" => ParsedModel {
            model: Model::MarbleNet,
            asset: if quantized {
                ModelAsset::MarbleNetInt8
            } else {
                ModelAsset::MarbleNetFp32
            },
            name: "marblenet",
        },
        _ if quantized => {
            return Err(PyValueError::new_err(format!(
                "model '{value}' has no supported quantized variant"
            )))
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported model '{value}'; expected silero, pyannote, pulsevad, fsmn, ten, or marblenet"
            )))
        }
    };
    Ok(parsed)
}

fn parse_runtime(value: &str) -> PyResult<Runtime> {
    match normalize_name(value).as_str() {
        "candle" => Ok(Runtime::Candle),
        "onnxruntime" | "onnx-runtime" | "ort" => Ok(Runtime::OnnxRuntime),
        _ => Err(PyValueError::new_err(format!(
            "unsupported runtime '{value}'; expected candle or onnxruntime"
        ))),
    }
}

fn validate_runtime(model: ParsedModel, runtime: Runtime, quantized: bool) -> PyResult<()> {
    if runtime == Runtime::Candle
        && (matches!(model.model, Model::PyAnnote | Model::Fsmn | Model::Ten) || quantized)
    {
        return Err(PyValueError::new_err(format!(
            "{}{} is only supported with ONNX Runtime",
            model.name,
            if quantized { " INT8" } else { "" }
        )));
    }
    Ok(())
}

fn normalize_name(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

const fn runtime_name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Candle => "candle",
        Runtime::OnnxRuntime => "onnxruntime",
    }
}

const fn asset_name(asset: ModelAsset) -> &'static str {
    match asset {
        ModelAsset::SileroV5 => "silero-v5",
        ModelAsset::PyAnnoteSegmentation => "pyannote-segmentation",
        ModelAsset::PulseVadFp32 => "pulsevad-fp32",
        ModelAsset::PulseVadInt8 => "pulsevad-int8",
        ModelAsset::FsmnVadFp32 => "fsmn-vad-fp32",
        ModelAsset::FsmnVadInt8 => "fsmn-vad-int8",
        ModelAsset::TenVad => "ten-vad",
        ModelAsset::MarbleNetFp32 => "marblenet-fp32",
        ModelAsset::MarbleNetInt8 => "marblenet-int8",
    }
}

#[allow(clippy::needless_pass_by_value)]
fn runtime_error(error: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{error:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_python_model_aliases() {
        assert_eq!(
            parse_model("pulse_vad", false).unwrap().model,
            Model::PulseVad
        );
        assert_eq!(
            parse_model("FSMN-VAD", true).unwrap().asset,
            ModelAsset::FsmnVadInt8
        );
        assert_eq!(
            parse_model("marble-net", false).unwrap().model,
            Model::MarbleNet
        );
    }

    #[test]
    fn rejects_unsupported_quantized_models() {
        assert!(parse_model("silero", true).is_err());
        assert!(parse_model("ten", true).is_err());
    }

    #[test]
    fn rejects_incompatible_candle_backends() {
        let fsmn = parse_model("fsmn", false).unwrap();
        assert!(validate_runtime(fsmn, Runtime::Candle, false).is_err());
        let pulsevad = parse_model("pulsevad", true).unwrap();
        assert!(validate_runtime(pulsevad, Runtime::Candle, true).is_err());
    }
}
