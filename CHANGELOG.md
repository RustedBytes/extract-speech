# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- MIT licensing metadata and license text for crates.io distribution.
- A reusable Rust library API with a model-independent detector builder, speech-segment inference, feature-gated Candle and ONNX Runtime backends, and library usage documentation.
- TEN VAD support through ONNX Runtime, including its reference feature frontend, recurrent state, documentation, and revision-pinned model integration tests.
- NVIDIA Frame-VAD MarbleNet support for FP32 models through Candle and ONNX Runtime and INT8 models through ONNX Runtime, including a NeMo-compatible frontend and revision-pinned integration tests.

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

[unreleased]: https://github.com/RustedBytes/extract-speech/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/RustedBytes/extract-speech/compare/v0.5.4...v0.6.0
[0.5.4]: https://github.com/RustedBytes/extract-speech/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/RustedBytes/extract-speech/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/RustedBytes/extract-speech/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/RustedBytes/extract-speech/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/RustedBytes/extract-speech/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/RustedBytes/extract-speech/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/RustedBytes/extract-speech/releases/tag/v0.3.0
