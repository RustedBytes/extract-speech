# extract-speech

[![Test Rust](https://github.com/RustedBytes/extract-speech/actions/workflows/test-rust.yml/badge.svg)](https://github.com/RustedBytes/extract-speech/actions/workflows/test-rust.yml)
[![Build](https://github.com/RustedBytes/extract-speech/actions/workflows/build.yml/badge.svg)](https://github.com/RustedBytes/extract-speech/actions/workflows/build.yml)

`extract-speech` is a Rust library for running voice activity detection (VAD) models. It accepts normalized mono 16 kHz PCM samples and returns detected speech ranges as sample offsets. An optional command-line application decodes common audio formats and writes the detected regions as clips or one concatenated file.

It supports:

- Silero VAD v5 through Candle or ONNX Runtime
- PulseVAD through Candle or ONNX Runtime
- PyAnnote segmentation models through ONNX Runtime
- FunASR FSMN-VAD FP32 and INT8 models through ONNX Runtime
- TEN VAD through ONNX Runtime
- NVIDIA Frame-VAD MarbleNet FP32 through Candle or ONNX Runtime, and INT8 through ONNX Runtime
- WAV, MP3, FLAC, Ogg, Opus, M4A, and AAC input
- WAV and Ogg Opus output
- automatic stereo-to-mono conversion and sample-rate conversion
- parallel processing of a directory of audio files
- optional CUDA, TensorRT, and CoreML execution providers
- a reusable, model-independent Rust inference API
- optional Python bindings built with PyO3 and maturin
- JSON metadata containing clip durations and inference time

## Library quick start

Add the library to your project:

```toml
[dependencies]
extract-speech = "0.7"
```

Load a Silero model through Candle and run inference on normalized mono 16 kHz samples:

```rust,no_run
use extract_speech::{
    download::{AssetManager, ModelAsset},
    Detector, Model, Result, Runtime, VadParams,
};

fn main() -> Result<()> {
    let model = AssetManager::default_cache()?.model(ModelAsset::SileroV5)?;
    let mut detector = Detector::builder(model.model_path())
        .model(Model::Silero)
        .runtime(Runtime::Candle)
        .parameters(VadParams {
            threshold: 0.7,
            ..VadParams::default()
        })
        .build()?;

    let samples = vec![0.0_f32; 16_000];
    for segment in detector.detect(&samples)? {
        println!("speech: {}..{} samples", segment.start, segment.end);
    }

    Ok(())
}
```

The same `Detector` can process multiple independent inputs; model state is reset between calls. See the [library guide](docs/library.md) for runtime features, ONNX Runtime initialization, and API details.

## Python quick start

Use `uv` to build and install the PyO3 extension from the repository, then load a cached model with one call:

```bash
uv sync --no-dev
```

```python
import array
import extract_speech

detector = extract_speech.Detector.from_pretrained("silero")
samples = array.array("f", [0.0] * extract_speech.SAMPLE_RATE)
segments = detector.detect(samples)
```

Tagged [GitHub releases](https://github.com/RustedBytes/extract-speech/releases) also include prebuilt
wheels for CPython 3.9–3.14. See the [Python guide](docs/python.md) for supported platforms, local
model paths, ONNX Runtime, buffer types, and download helpers.

## CLI quick start

Install Rust and Protocol Buffers first; see the [installation guide](docs/installation.md) for platform-specific instructions.

```bash
git clone https://github.com/RustedBytes/extract-speech.git
cd extract-speech
cargo build --release --features cli
./target/release/extract-speech download silero --cache-dir .cache/extract-speech
```

Extract each detected speech region to a WAV file:

```bash
./target/release/extract-speech \
  --model-path .cache/extract-speech/models/silero-v5/model.onnx \
  --process-audio input.wav \
  --output output
```

Create one file with the detected regions joined together:

```bash
./target/release/extract-speech \
  --model-path .cache/extract-speech/models/silero-v5/model.onnx \
  --process-audio input.wav \
  --output-type concatenated \
  --output speech.wav
```

The default runtime is Candle, the default detection threshold is `0.7`, and the default output sample rate is 16 kHz. Run `extract-speech --help` for the complete command reference.

## Cargo features

| Feature | Default | Provides |
| --- | --- | --- |
| `candle` | Yes | Silero, PulseVAD, and MarbleNet inference through Candle |
| `onnxruntime` | Yes | All supported models through dynamically loaded ONNX Runtime |
| `download` | Yes | Checksum-verified model bundles, ONNX Runtime downloads, and persistent caching |
| `python` | No | PyO3 extension module with inference and download APIs |
| `cli` | No | The `extract-speech` executable and audio file I/O |
| `accelerate-src` | No | Apple Accelerate integration for Candle builds |

## Runtimes

| Runtime | Models | External runtime library | Acceleration |
| --- | --- | --- | --- |
| Candle | Silero, PulseVAD FP32, MarbleNet FP32 | No | CPU |
| ONNX Runtime | Silero, PulseVAD FP32/INT8, PyAnnote, FSMN-VAD FP32/INT8, TEN VAD, MarbleNet FP32/INT8 | Yes, supplied with `--dylib-path` | CPU, CUDA, TensorRT, CoreML |

For ONNX Runtime setup and compatible model requirements, see [Models and runtimes](docs/models-and-runtimes.md).

## Documentation

- [Changelog](CHANGELOG.md) — notable changes by release
- [Library](docs/library.md) — Rust API, Cargo features, and inference examples
- [Python API](docs/python.md) — installation, automatic model loading, and inference
- [Installation](docs/installation.md) — prerequisites, builds, and releases
- [Usage](docs/usage.md) — inputs, outputs, CLI options, metadata, and examples
- [Models and runtimes](docs/models-and-runtimes.md) — model compatibility and hardware acceleration
- [Development](docs/development.md) — architecture, checks, and contribution workflow

## Development

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings -W clippy::pedantic
cargo test --all-targets --all-features
cargo build --features cli
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python -m unittest discover -s tests/python -v
```

See the [development guide](docs/development.md) for the code layout and project conventions.

## Citation

```bibtex
@software{Smoliakov_Extract_Speech_2026,
  author = {Smoliakov, Yehor},
  month = sep,
  title = {{extract-speech: Extract speech from audio files using Voice Activity Detection models}},
  url = {https://github.com/RustedBytes/extract-speech},
  version = {0.7.2},
  year = {2026}
}
```

## License

This project is available under the [MIT License](LICENSE).
