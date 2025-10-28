# extract-speech

A high-performance Rust tool for extracting speech segments from audio files using Voice Activity Detection (VAD) models.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Build Status](#build-status)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Usage](#usage)
  - [Basic Examples](#basic-examples)
  - [CLI Options](#cli-options)
  - [Output Formats](#output-formats)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)

## About

`extract-speech` is a command-line tool that automatically detects and extracts speech segments from audio files. It uses Voice Activity Detection (VAD) models to identify portions of audio that contain human speech, filtering out silence and non-speech audio.

**Use Cases:**
- Preprocessing audio for speech recognition systems
- Removing silence from audio recordings
- Extracting spoken content from podcasts or interviews
- Preparing datasets for machine learning
- Audio cleanup and optimization

## Features

- 🚀 **High Performance**: Built in Rust with optimized inference
- 🎯 **Multiple Runtimes**: Support for both Candle and ONNX Runtime
- 🎵 **Format Support**: Read various audio formats via Symphonia
- 📦 **Multiple Output Formats**: WAV, Opus, and OGG
- 🔧 **Flexible Processing**: Process individual files or concatenated output
- 🎛️ **Hardware Acceleration**: Support for CUDA, TensorRT, and CoreML
- 📊 **Configurable**: Adjustable VAD threshold and sample rate
- 🔊 **Channel Support**: Handles both mono and stereo audio

## Build Status

[![build linux](https://github.com/RustedBytes/extract-speech/actions/workflows/build-linux.yml/badge.svg)](https://github.com/RustedBytes/extract-speech/actions/workflows/build-linux.yml)
[![build macos](https://github.com/RustedBytes/extract-speech/actions/workflows/build-macos.yml/badge.svg)](https://github.com/RustedBytes/extract-speech/actions/workflows/build-macos.yml)
[![build windows](https://github.com/RustedBytes/extract-speech/actions/workflows/build-win.yml/badge.svg)](https://github.com/RustedBytes/extract-speech/actions/workflows/build-win.yml)

## Prerequisites

Before building or using `extract-speech`, you need:

### System Dependencies

**Linux (Ubuntu/Debian):**
```bash
apt-get install protobuf-compiler
```

**macOS:**
```bash
brew install protobuf
```

**Windows:**
Download and install protobuf from the [official releases](https://github.com/protocolbuffers/protobuf/releases).

### Rust

Ensure you have Rust installed (version 1.70 or later):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/RustedBytes/extract-speech.git
cd extract-speech

# Build the project (requires ONNX Runtime for full functionality)
cargo build --release

# The binary will be available at ./target/release/extract-speech
```

### Binary Releases

Pre-built binaries for Linux, macOS, and Windows are available on the [releases page](https://github.com/RustedBytes/extract-speech/releases).

## Model Setup

`extract-speech` requires a VAD model file. The tool supports Silero VAD v5 models.

### Download the Model

Create a `models` directory and download the Silero VAD model:

```bash
mkdir -p models
cd models

# Download Silero VAD v5 (ONNX format)
wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad_v5.onnx

# Or for Candle runtime, use the Xenova variant
wget https://huggingface.co/Xenova/silero-vad/raw/main/onnx/model.onnx -O xenova_silero_vad_v5.onnx
```

### Download ONNX Runtime (for ONNX Runtime backend)

If you plan to use the ONNX Runtime backend, download the appropriate library:

#### Linux

```bash
wget "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz"
tar -xzf onnxruntime-linux-x64-gpu-1.22.0.tgz
rm onnxruntime-linux-x64-gpu-1.22.0.tgz
```

#### macOS

```bash
wget "https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-arm64-1.20.0.tgz"
tar -xzf onnxruntime-osx-arm64-1.20.0.tgz
rm onnxruntime-osx-arm64-1.20.0.tgz
```

**Note:** The Candle runtime doesn't require downloading ONNX Runtime libraries.

## Usage

### Basic Examples

**Using Candle Runtime (simplest, no external dependencies):**

```bash
extract-speech \
  --runtime candle \
  --model-path ./models/xenova_silero_vad_v5.onnx \
  --process-audio test-audios/test_16khz.wav \
  --output ./output \
  --output-format wav
```

**Using ONNX Runtime on Linux:**

```bash
extract-speech \
  --runtime onnxruntime \
  --dylib-path onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so.1.22.0 \
  --model-path ./models/silero_vad_v5.onnx \
  --process-audio test-audios/test_16khz.wav \
  --output ./output \
  --output-format wav
```

**Using ONNX Runtime on macOS:**

```bash
extract-speech \
  --runtime onnxruntime \
  --dylib-path onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib \
  --model-path ./models/silero_vad_v5.onnx \
  --process-audio test-audios/test_16khz.wav \
  --output ./output \
  --output-format wav
```

**Creating a single concatenated output file:**

```bash
extract-speech \
  --runtime candle \
  --model-path ./models/xenova_silero_vad_v5.onnx \
  --process-audio test-audios/test_16khz.wav \
  --output ./output.wav \
  --output-type concatenated \
  --output-format wav
```

### CLI Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--runtime` | Inference runtime: `candle` or `onnxruntime` | `candle` | No |
| `--model-path` | Path to the VAD model file | - | Yes |
| `--dylib-path` | Path to ONNX Runtime library (required for `onnxruntime`) | - | Conditional |
| `--process-audio` | Input audio file to process | - | Yes |
| `--source-audio` | Optional source audio for final extraction (if different from process-audio) | - | No |
| `--output` | Output directory or file path | - | Yes |
| `--output-type` | Output type: `files` or `concatenated` | `files` | No |
| `--output-format` | Output format: `wav`, `opus`, or `ogg` | `wav` | No |
| `--threshold` | VAD detection threshold (0.0-1.0) | `0.7` | No |
| `--sample-rate` | Output sample rate in Hz | `16000` | No |
| `--cuda` | Enable CUDA acceleration | `false` | No |
| `--trt` | Enable TensorRT acceleration | `false` | No |
| `--coreml` | Enable CoreML acceleration (macOS) | `false` | No |
| `--debug` | Enable debug output | `false` | No |
| `--print-model-info` | Print model information: `graph`, `nodes`, or `io` | - | No |

### Output Formats

#### WAV (Waveform Audio File Format)
- Uncompressed audio format
- Highest quality, larger file size
- Widely compatible

#### Opus/OGG
- Compressed audio format
- Lower file size, good quality
- Ideal for speech
- Requires `--sample-rate` specification

#### Output Types

- **files**: Creates separate files for each detected speech segment
  - Filenames: `<timestamp>_<index>.<extension>`
  - Use when you need individual speech segments

- **concatenated**: Creates a single file with all speech segments joined
  - Filename: specified by `--output` parameter
  - Use when you want continuous speech without gaps

## Advanced Usage

### Processing with Hardware Acceleration

**CUDA (NVIDIA GPUs):**
```bash
extract-speech \
  --runtime onnxruntime \
  --cuda \
  --dylib-path ./onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so.1.22.0 \
  --model-path ./models/silero_vad_v5.onnx \
  --process-audio input.wav \
  --output ./output
```

**TensorRT (NVIDIA GPUs, optimized):**
```bash
extract-speech \
  --runtime onnxruntime \
  --trt \
  --dylib-path ./onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so.1.22.0 \
  --model-path ./models/silero_vad_v5.onnx \
  --process-audio input.wav \
  --output ./output
```

**CoreML (macOS with Apple Silicon):**
```bash
extract-speech \
  --runtime onnxruntime \
  --coreml \
  --dylib-path ./onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib \
  --model-path ./models/silero_vad_v5.onnx \
  --process-audio input.wav \
  --output ./output
```

### Two-Stage Processing

You can process a denoised version of audio for VAD detection while extracting segments from the original:

```bash
extract-speech \
  --runtime candle \
  --model-path ./models/xenova_silero_vad_v5.onnx \
  --source-audio original.wav \
  --process-audio denoised.wav \
  --output ./output \
  --output-format wav
```

### Adjusting VAD Sensitivity

The `--threshold` parameter controls sensitivity (0.0 to 1.0):

- **Lower values (e.g., 0.3-0.5)**: More sensitive, catches more speech, may include some noise
- **Higher values (e.g., 0.8-0.9)**: Less sensitive, only strong speech signals, may miss quiet speech
- **Default (0.7)**: Balanced for most use cases

```bash
# More sensitive - capture more potential speech
extract-speech --threshold 0.4 ...

# Less sensitive - only clear speech
extract-speech --threshold 0.85 ...
```

### Debug Mode

Enable debug logging to see detailed processing information:

```bash
RUST_LOG=debug extract-speech \
  --runtime candle \
  --model-path ./models/xenova_silero_vad_v5.onnx \
  --process-audio input.wav \
  --output ./output \
  --debug
```

## Troubleshooting

### Build Issues

**Problem:** `protobuf-compiler` not found
```
Solution: Install protobuf compiler
  Linux: sudo apt-get install protobuf-compiler
  macOS: brew install protobuf
```

**Problem:** ONNX Runtime download fails during build
```
Solution: Use Candle runtime instead, or manually download ONNX Runtime and set ORT_LIB_LOCATION environment variable
```

### Runtime Issues

**Problem:** "Failed to load dynamic library"
```
Solution: Ensure --dylib-path points to the correct ONNX Runtime library file (.so on Linux, .dylib on macOS, .dll on Windows)
```

**Problem:** "Model file not found"
```
Solution: Verify the --model-path points to a valid .onnx model file. Download from the Model Setup section.
```

**Problem:** No speech segments detected
```
Solution: Try lowering the --threshold value (e.g., --threshold 0.5). Ensure input audio contains speech.
```

**Problem:** Too many false positives
```
Solution: Increase the --threshold value (e.g., --threshold 0.8) to make detection more conservative.
```

### Audio Format Issues

**Problem:** Unsupported audio format
```
Solution: Convert audio to a supported format (WAV, MP3, FLAC, OGG, etc.) using ffmpeg:
  ffmpeg -i input.m4a output.wav
```

**Problem:** Stereo audio not processing correctly
```
Solution: The tool automatically handles stereo audio. Ensure your source and process audio have the same channel configuration.
```

## Contributing

Contributions are welcome! Here's how you can help:

1. **Report Issues**: Found a bug? Open an issue with details and reproduction steps
2. **Suggest Features**: Have ideas for improvements? Create a feature request
3. **Submit PRs**: 
   - Fork the repository
   - Create a feature branch
   - Make your changes with tests
   - Submit a pull request

### Development Setup

```bash
# Clone and build
git clone https://github.com/RustedBytes/extract-speech.git
cd extract-speech
cargo build

# Run tests (when available)
cargo test

# Format code
cargo fmt

# Run linter
cargo clippy --all-targets
```

## Citation

If you use this tool in your research or project, please cite:

```bibtex
@software{Smoliakov_Extract_Speech_2025,
  author = {Smoliakov, Yehor},
  month = oct,
  title = {{extract-speech: Extract speech from audio files by a Voice Activity Detection models}},
  url = {https://github.com/RustedBytes/extract-speech},
  version = {0.5.1},
  year = {2025}
}
```

---

**License:** Please check the repository for license information.

**Author:** Yehor Smoliakov

**Repository:** [https://github.com/RustedBytes/extract-speech](https://github.com/RustedBytes/extract-speech)
