# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.1] - 2026-09-23

### Added

- CPython 3.15 release wheels and installation smoke tests.
- PyPI publishing for wheels built by tagged release workflows.

## [0.9.0] - 2026-09-19

### Added

- PyAnnote segmentation inference through Candle, including compatibility rewrites for ONNX operators not natively implemented by Candle.
- FunASR FSMN-VAD FP32 and dequantized INT8 inference through Candle using the existing Kaldi-compatible frontend and recurrent cache processing.
- PulseVAD INT8 QDQ model compatibility through Candle by dequantizing constant weights and bypassing unsupported activation QDQ operators.
- TEN VAD inference through Candle using shared reference-compatible feature extraction and recurrent state processing.
- NVIDIA Frame-VAD MarbleNet INT8 inference through Candle by dequantizing constant weights and bypassing unsupported dynamic quantization operators.

### Changed

- Made PulseVAD the default model in the Rust API, CLI, and Python bindings.

## [0.8.0] - 2026-09-19

### Added

- Silero VAD v6 support through Candle and ONNX Runtime, including a revision-pinned,
  checksum-verified official model download while retaining Silero VAD v5 compatibility.

## [0.7.2] - 2026-09-16

### Added

- Exposed checksum-verified model and ONNX Runtime downloads through the CLI `download` subcommand.

## [0.7.1] - 2026-09-16

### Changed

- Route model diagnostics through the `log` facade and use logger level filters instead of per-model debug switches while preserving the existing compatibility parameters.

### Fixed

- Anchor crates.io package include patterns to the repository root so ignored virtual-environment files are never packaged.
- Install the OpenSSL headers required by ONNX Runtime, use the container's older curl compatibly, and run Maturin Action on Node.js 24 for manylinux wheel builds.
- Ignore tag-dependent changelog links during link checks so release preparation does not fail before tags exist.

## [0.7.0] - 2026-09-16

### Added

- GitHub Release wheels for CPython 3.9–3.14 on manylinux x86-64, macOS ARM64, and Windows x86-64, with uv-based installation smoke tests.
- Optional PyO3 Python bindings with automatic model/runtime downloads, cached `from_pretrained` construction, float32 buffer inference, type stubs, uv/maturin packaging, and Ruff/Pyright quality gates.
- A checksum-verified download and cache API for every supported model bundle, required FSMN sidecar files, and compatible ONNX Runtime distributions across supported platforms.
- MIT licensing metadata and license text for crates.io distribution.
- A reusable Rust library API with a model-independent detector builder, speech-segment inference, feature-gated Candle and ONNX Runtime backends, and library usage documentation.
- TEN VAD support through ONNX Runtime, including its reference feature frontend, recurrent state, documentation, and revision-pinned model integration tests.
- NVIDIA Frame-VAD MarbleNet support for FP32 models through Candle and ONNX Runtime and INT8 models through ONNX Runtime, including a NeMo-compatible frontend and revision-pinned integration tests.

### Changed

- Grouped VAD core, model backends, audio utilities, asset management, Python bindings, and CLI code into domain-focused modules while preserving the existing public module paths.
- Reworked the CLI to use the public `Detector` API instead of maintaining a separate model/runtime dispatch implementation.
- Enabled Clippy's pedantic lint group in local development and CI, with narrowly scoped exceptions for intentional DSP conversions and exact test fixtures.

### Fixed

- Validate VAD parameters, normalized samples, and model output shapes before arithmetic, slicing, or tensor indexing.
- Enforce maximum speech duration during continuous speech and consistently apply duration and padding parameters to PulseVAD and PyAnnote segmentation.
- Pin and checksum-verify model URLs shown in the CLI and model setup guides.

## [0.6.0] - 2026-09-16

### Added

- PulseVAD support with its reference log-mel frontend for Candle and ONNX Runtime, including FP32 and ONNX Runtime INT8 models.
- FunASR FSMN-VAD support for FP32 and INT8 models through ONNX Runtime, including Kaldi filterbanks, low-frame-rate stacking, CMVN, and recurrent caches.
- Revision-pinned, checksum-verified end-to-end tests for every supported model and runtime combination across mono, stereo, and resampled audio fixtures.
- Dedicated installation, usage, model compatibility, and development documentation.

### Changed

- Made the command-line application optional behind the `cli` Cargo feature so library consumers do not pull in audio file I/O and CLI dependencies.

- Migrated audio resampling to `fast-audio-resampler` and upgraded the Rust inference, audio, CLI, and serialization dependencies.
- Consolidated platform builds and releases into matrix-based GitHub Actions workflows for Linux, macOS, and Windows.
- Updated GitHub Actions to current major versions, moved Linux jobs to Ubuntu 26.04, and authenticated Protocol Buffers setup downloads.
- Reworked audio decoding, model initialization, output handling, and error reporting to provide safer validation and clearer failures.

### Fixed

- Preserve trailing and short valid speech segments while rejecting invalid interval boundaries.
- Correctly handle partial Opus frames, stream finalization, output resampling, and empty inputs.
- Improve PyAnnote score normalization and tensor-shape validation.
- Reject unsupported channel layouts and inconsistent source/process audio lengths without panicking.

## [0.5.4] - 2025-12-15

### Changed

- Updated Rust dependencies and GitHub Actions used for checkout, artifact upload, pull-request automation, and repository maintenance.

## [0.5.3] - 2025-10-28

### Added

- Parallel folder processing with per-file output and metadata handling.
- Automated Rust tests and scheduled Markdown link checking in GitHub Actions.
- Comprehensive unit coverage for audio, resampling, Opus output, VAD state, and CLI behavior.

### Changed

- Hardened workflow token permissions and made Clippy warnings fail CI.
- Improved PyAnnote softmax efficiency and numerical stability.

## [0.5.2] - 2025-10-28

### Added

- PyAnnote segmentation as an ONNX Runtime VAD backend.
- Optional JSON metadata containing output interval durations and inference time.
- Expanded usage, model setup, troubleshooting, and CLI documentation.

### Fixed

- Corrected PyAnnote output extraction and speech-probability calculation.

## [0.5.1] - 2025-10-28

### Added

- Tag-triggered GitHub Actions workflow for building and publishing release artifacts.

### Fixed

- Restored the compatible `ndarray` version after the 0.5.0 dependency update.
- Corrected release workflow configuration.

## [0.5.0] - 2025-10-15

### Changed

- Updated ONNX Runtime, audio, array, CLI, logging, serialization, and concurrency dependencies.
- Replaced GoReleaser-based packaging with platform-specific GitHub Actions release workflows.
- Added automated Rust dependency update checks.

## [0.4.0] - 2025-05-05

### Added

- Runtime loading of an external ONNX Runtime library through `--dylib-path`.
- CUDA, TensorRT, and CoreML execution-provider options.
- Mono input support and automatic stereo-to-mono conversion.
- Windows x86-64 and ARM64 release workflows.

### Changed

- Moved resampling into a dedicated module.
- Switched runtime diagnostics to structured logging.
- Removed bundled ONNX models and the bundled ONNX Runtime library; users now supply compatible artifacts explicitly.

## [0.3.0] - 2025-04-05

### Added

- Initial tagged release of the `extract-speech` command-line application.
- Silero VAD v5 inference through Candle and ONNX Runtime.
- Speech extraction to individual or concatenated WAV and Ogg Opus outputs.
- Multi-format audio decoding, sample-rate conversion, model inspection, and configurable VAD thresholds.
- Linux, macOS, and Windows build workflows.

[unreleased]: https://github.com/RustedBytes/extract-speech/compare/v0.9.1...HEAD
[0.9.1]: https://github.com/RustedBytes/extract-speech/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/RustedBytes/extract-speech/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/RustedBytes/extract-speech/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/RustedBytes/extract-speech/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/RustedBytes/extract-speech/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/RustedBytes/extract-speech/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/RustedBytes/extract-speech/compare/v0.5.4...v0.6.0
[0.5.4]: https://github.com/RustedBytes/extract-speech/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/RustedBytes/extract-speech/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/RustedBytes/extract-speech/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/RustedBytes/extract-speech/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/RustedBytes/extract-speech/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/RustedBytes/extract-speech/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/RustedBytes/extract-speech/releases/tag/v0.3.0
