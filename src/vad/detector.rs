//! High-level, runtime-independent detector API.

use std::path::PathBuf;

use anyhow::{bail, ensure, Result};

use crate::{utils::VAD_SAMPLE_RATE, SpeechSegment, VadParams};

/// A VAD model family supported by [`Detector`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Model {
    #[default]
    Silero,
    PyAnnote,
    PulseVad,
    Fsmn,
    Ten,
    MarbleNet,
}

/// The inference engine used to execute a model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Runtime {
    #[default]
    Candle,
    OnnxRuntime,
}

/// Builder for a reusable [`Detector`].
pub struct DetectorBuilder {
    model_path: PathBuf,
    model: Model,
    runtime: Runtime,
    params: VadParams,
    #[cfg(feature = "candle")]
    candle_device: candle_core::Device,
    #[cfg(feature = "onnxruntime")]
    execution_providers: Vec<ort::ep::ExecutionProviderDispatch>,
}

impl DetectorBuilder {
    /// Starts configuring a detector which will load `model_path`.
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            model: Model::default(),
            runtime: Runtime::default(),
            params: VadParams::default(),
            #[cfg(feature = "candle")]
            candle_device: candle_core::Device::Cpu,
            #[cfg(feature = "onnxruntime")]
            execution_providers: vec![ort::ep::CPU::default().build()],
        }
    }

    #[must_use]
    pub fn model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    #[must_use]
    pub fn runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = runtime;
        self
    }

    #[must_use]
    pub fn parameters(mut self, params: VadParams) -> Self {
        self.params = params;
        self
    }

    #[cfg(feature = "candle")]
    #[must_use]
    pub fn candle_device(mut self, device: candle_core::Device) -> Self {
        self.candle_device = device;
        self
    }

    #[cfg(feature = "onnxruntime")]
    #[must_use]
    pub fn execution_providers(
        mut self,
        execution_providers: Vec<ort::ep::ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    /// Loads the model and creates a stateful detector.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid parameters, unsupported model/runtime
    /// combinations, or backend initialization failures.
    pub fn build(mut self) -> Result<Detector> {
        validate_parameters(&self.params)?;

        if self.model == Model::Ten {
            self.params.frame_size = ten_frame_size_ms();
        }

        match self.runtime {
            Runtime::Candle => self.build_candle(),
            Runtime::OnnxRuntime => self.build_onnxruntime(),
        }
    }

    #[cfg(feature = "candle")]
    fn build_candle(self) -> Result<Detector> {
        let backend = match self.model {
            Model::Silero => Backend::SileroCandle(crate::vad_iter::VadIter::new(
                crate::silero_v5::Silero::new(
                    self.params.clone(),
                    self.model_path,
                    self.candle_device,
                )?,
                self.params,
            )),
            Model::PulseVad => Backend::PulseVadCandle(crate::pulsevad_iter::PulseVadIter::new(
                crate::pulsevad::PulseVad::new(
                    self.model_path,
                    self.candle_device,
                    self.params.debug,
                )?,
                self.params,
            )),
            Model::MarbleNet => {
                Backend::MarbleNetCandle(crate::marblenet_iter::MarbleNetIter::new(
                    crate::marblenet::MarbleNet::new(
                        self.model_path,
                        self.candle_device,
                        self.params.debug,
                    )?,
                    self.params,
                ))
            }
            Model::PyAnnote | Model::Fsmn | Model::Ten => {
                bail!("{:?} is only supported with ONNX Runtime", self.model)
            }
        };
        Ok(Detector { backend })
    }

    #[cfg(not(feature = "candle"))]
    fn build_candle(self) -> Result<Detector> {
        bail!("Candle support is disabled; enable the `candle` Cargo feature")
    }

    #[cfg(feature = "onnxruntime")]
    fn build_onnxruntime(self) -> Result<Detector> {
        let backend = match self.model {
            Model::Silero => Backend::SileroOnnx(Box::new(crate::vad_iter::VadIter::new(
                crate::silero_v5_ort::Silero::new(
                    self.params.clone(),
                    self.execution_providers,
                    self.model_path,
                )?,
                self.params,
            ))),
            Model::PyAnnote => {
                Backend::PyAnnoteOnnx(crate::pyannote_vad_iter::PyAnnoteVadIter::new(
                    crate::pyannote_vad_ort::PyAnnote::new(
                        self.params.clone(),
                        self.execution_providers,
                        self.model_path,
                    )?,
                    self.params,
                ))
            }
            Model::PulseVad => Backend::PulseVadOnnx(crate::pulsevad_iter::PulseVadIter::new(
                crate::pulsevad_ort::PulseVad::new(
                    self.execution_providers,
                    self.model_path,
                    self.params.debug,
                )?,
                self.params,
            )),
            Model::Fsmn => Backend::FsmnOnnx(crate::fsmn_vad_iter::FsmnVadIter::new(
                crate::fsmn_vad_ort::FsmnVad::new(
                    self.execution_providers,
                    self.model_path,
                    self.params.debug,
                )?,
                self.params,
            )),
            Model::Ten => Backend::TenOnnx(Box::new(crate::vad_iter::VadIter::new(
                crate::ten_vad_ort::TenVad::new(self.model_path, self.params.debug)?,
                self.params,
            ))),
            Model::MarbleNet => Backend::MarbleNetOnnx(crate::marblenet_iter::MarbleNetIter::new(
                crate::marblenet_ort::MarbleNet::new(
                    self.execution_providers,
                    self.model_path,
                    self.params.debug,
                )?,
                self.params,
            )),
        };
        Ok(Detector { backend })
    }

    #[cfg(not(feature = "onnxruntime"))]
    fn build_onnxruntime(self) -> Result<Detector> {
        bail!("ONNX Runtime support is disabled; enable the `onnxruntime` Cargo feature")
    }
}

/// A loaded, reusable voice activity detector.
///
/// Call [`Detector::detect`] once per independent audio stream. Stateful model
/// data is reset by the underlying implementation between calls.
pub struct Detector {
    backend: Backend,
}

impl Detector {
    #[must_use]
    pub fn builder(model_path: impl Into<PathBuf>) -> DetectorBuilder {
        DetectorBuilder::new(model_path)
    }

    /// Detects speech in mono, normalized 16 kHz PCM samples.
    ///
    /// # Errors
    ///
    /// Returns an error if model reset, preprocessing, or inference fails.
    pub fn detect(&mut self, samples: &[f32]) -> Result<Vec<SpeechSegment>> {
        let segments: &[SpeechSegment] = match &mut self.backend {
            #[cfg(feature = "candle")]
            Backend::SileroCandle(iter) => iter.process(samples)?,
            #[cfg(feature = "candle")]
            Backend::PulseVadCandle(iter) => iter.process(samples)?,
            #[cfg(feature = "candle")]
            Backend::MarbleNetCandle(iter) => iter.process(samples)?,
            #[cfg(feature = "onnxruntime")]
            Backend::SileroOnnx(iter) => iter.process(samples)?,
            #[cfg(feature = "onnxruntime")]
            Backend::PyAnnoteOnnx(iter) => iter.process(samples)?,
            #[cfg(feature = "onnxruntime")]
            Backend::PulseVadOnnx(iter) => iter.process(samples)?,
            #[cfg(feature = "onnxruntime")]
            Backend::FsmnOnnx(iter) => iter.process(samples)?,
            #[cfg(feature = "onnxruntime")]
            Backend::TenOnnx(iter) => iter.process(samples)?,
            #[cfg(feature = "onnxruntime")]
            Backend::MarbleNetOnnx(iter) => iter.process(samples)?,
            #[cfg(not(any(feature = "candle", feature = "onnxruntime")))]
            Backend::Disabled => {
                bail!("no inference runtime is enabled; enable `candle` or `onnxruntime`")
            }
        };
        Ok(segments.to_vec())
    }
}

// Runtime suffixes keep variants unambiguous when both feature sets are enabled.
#[allow(clippy::enum_variant_names)]
enum Backend {
    #[cfg(feature = "candle")]
    SileroCandle(crate::vad_iter::VadIter<crate::silero_v5::Silero>),
    #[cfg(feature = "candle")]
    PulseVadCandle(crate::pulsevad_iter::PulseVadIter<crate::pulsevad::PulseVad>),
    #[cfg(feature = "candle")]
    MarbleNetCandle(crate::marblenet_iter::MarbleNetIter<crate::marblenet::MarbleNet>),
    #[cfg(feature = "onnxruntime")]
    SileroOnnx(Box<crate::vad_iter::VadIter<crate::silero_v5_ort::Silero>>),
    #[cfg(feature = "onnxruntime")]
    PyAnnoteOnnx(crate::pyannote_vad_iter::PyAnnoteVadIter),
    #[cfg(feature = "onnxruntime")]
    PulseVadOnnx(crate::pulsevad_iter::PulseVadIter<crate::pulsevad_ort::PulseVad>),
    #[cfg(feature = "onnxruntime")]
    FsmnOnnx(crate::fsmn_vad_iter::FsmnVadIter),
    #[cfg(feature = "onnxruntime")]
    TenOnnx(Box<crate::vad_iter::VadIter<crate::ten_vad_ort::TenVad>>),
    #[cfg(feature = "onnxruntime")]
    MarbleNetOnnx(crate::marblenet_iter::MarbleNetIter<crate::marblenet_ort::MarbleNet>),
    #[cfg(not(any(feature = "candle", feature = "onnxruntime")))]
    Disabled,
}

fn validate_parameters(params: &VadParams) -> Result<()> {
    ensure!(
        params.sample_rate == VAD_SAMPLE_RATE,
        "VAD inference requires {VAD_SAMPLE_RATE} Hz samples"
    );
    ensure!(
        params.frame_size > 0,
        "frame size must be greater than zero"
    );
    ensure!(
        params.threshold.is_finite() && (0.0..=1.0).contains(&params.threshold),
        "threshold must be between 0 and 1"
    );
    ensure!(
        params.max_speech_duration_s > 0.0,
        "maximum speech duration must be greater than zero"
    );
    Ok(())
}

#[cfg(feature = "onnxruntime")]
/// Initializes the dynamically loaded ONNX Runtime environment.
///
/// # Errors
///
/// Returns an error if the path is not UTF-8 or the runtime cannot be loaded.
pub fn init_onnx_runtime(
    dynamic_library_path: impl AsRef<std::path::Path>,
    execution_providers: Vec<ort::ep::ExecutionProviderDispatch>,
) -> Result<()> {
    let path = dynamic_library_path
        .as_ref()
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("ONNX Runtime library path is not valid UTF-8"))?;
    ort::init_from(path)?
        .with_execution_providers(execution_providers)
        .commit();
    Ok(())
}

#[cfg(feature = "onnxruntime")]
const fn ten_frame_size_ms() -> usize {
    crate::ten_vad_ort::FRAME_SAMPLES * 1_000 / VAD_SAMPLE_RATE
}

#[cfg(not(feature = "onnxruntime"))]
const fn ten_frame_size_ms() -> usize {
    16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_vad_sample_rate() {
        let params = VadParams {
            sample_rate: 8_000,
            ..VadParams::default()
        };
        assert!(validate_parameters(&params).is_err());
    }

    #[test]
    fn rejects_invalid_threshold() {
        let params = VadParams {
            threshold: f32::NAN,
            ..VadParams::default()
        };
        assert!(validate_parameters(&params).is_err());
    }
}
