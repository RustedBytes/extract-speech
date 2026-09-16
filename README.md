# extract-speech

[![Test Rust](https://github.com/RustedBytes/extract-speech/actions/workflows/test-rust.yml/badge.svg)](https://github.com/RustedBytes/extract-speech/actions/workflows/test-rust.yml)
[![Build](https://github.com/RustedBytes/extract-speech/actions/workflows/build.yml/badge.svg)](https://github.com/RustedBytes/extract-speech/actions/workflows/build.yml)

`extract-speech` is a Rust command-line tool that detects speech in audio and writes the detected regions as individual clips or one concatenated file.

It supports:

- Silero VAD v5 through Candle or ONNX Runtime
- PulseVAD through Candle or ONNX Runtime
- PyAnnote segmentation models through ONNX Runtime
- WAV, MP3, FLAC, Ogg, Opus, M4A, and AAC input
- WAV and Ogg Opus output
- automatic stereo-to-mono conversion and sample-rate conversion
- parallel processing of a directory of audio files
- optional CUDA, TensorRT, and CoreML execution providers
- JSON metadata containing clip durations and inference time

## Quick start

Install Rust and Protocol Buffers first; see the [installation guide](docs/installation.md) for platform-specific instructions.

```bash
git clone https://github.com/RustedBytes/extract-speech.git
cd extract-speech
cargo build --release

mkdir -p models
curl -L \
  https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx \
  -o models/silero-vad-v5.onnx
```

Extract each detected speech region to a WAV file:

```bash
./target/release/extract-speech \
  --model-path models/silero-vad-v5.onnx \
  --process-audio input.wav \
  --output output
```

Create one file with the detected regions joined together:

```bash
./target/release/extract-speech \
  --model-path models/silero-vad-v5.onnx \
  --process-audio input.wav \
  --output-type concatenated \
  --output speech.wav
```

The default runtime is Candle, the default detection threshold is `0.7`, and the default output sample rate is 16 kHz. Run `extract-speech --help` for the complete command reference.

## Runtimes

| Runtime | Models | External runtime library | Acceleration |
| --- | --- | --- | --- |
| Candle | Silero, PulseVAD FP32 | No | CPU |
| ONNX Runtime | Silero, PulseVAD FP32/INT8, PyAnnote | Yes, supplied with `--dylib-path` | CPU, CUDA, TensorRT, CoreML |

For ONNX Runtime setup and compatible model requirements, see [Models and runtimes](docs/models-and-runtimes.md).

## Documentation

- [Installation](docs/installation.md) — prerequisites, builds, and releases
- [Usage](docs/usage.md) — inputs, outputs, CLI options, metadata, and examples
- [Models and runtimes](docs/models-and-runtimes.md) — model compatibility and hardware acceleration
- [Development](docs/development.md) — architecture, checks, and contribution workflow

## Development

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build
```

See the [development guide](docs/development.md) for the code layout and project conventions.

## Citation

```bibtex
@software{Smoliakov_Extract_Speech_2025,
  author = {Smoliakov, Yehor},
  month = oct,
  title = {{extract-speech: Extract speech from audio files using Voice Activity Detection models}},
  url = {https://github.com/RustedBytes/extract-speech},
  version = {0.5.1},
  year = {2025}
}
```
