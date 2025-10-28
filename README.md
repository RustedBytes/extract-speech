# extract-speech

Extract speech from audio files by a Voice Activity Detection model

## Supported VAD Models

- **Silero VAD** (v3, v4, v5): Default VAD model, supports both Candle and ONNX Runtime
- **PyAnnote**: Alternative VAD model based on [pyannote-onnx-rust](https://github.com/RustedBytes/pyannote-onnx-rust), requires ONNX Runtime

## Statuses

### Build

[![build linux](https://github.com/crs-org/extract-speech/actions/workflows/build-linux.yml/badge.svg)](https://github.com/crs-org/extract-speech/actions/workflows/build-linux.yml)
[![build macos](https://github.com/crs-org/extract-speech/actions/workflows/build-macos.yml/badge.svg)](https://github.com/crs-org/extract-speech/actions/workflows/build-macos.yml)
[![build windows](https://github.com/crs-org/extract-speech/actions/workflows/build-win.yml/badge.svg)](https://github.com/crs-org/extract-speech/actions/workflows/build-win.yml)

## Required packages

```shell
apt-get install protobuf-compiler
```

## Download onnxruntime

### Linux

```shell
wget "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz"
ouch decompress onnxruntime-linux-x64-gpu-1.22.0.tgz
rm onnxruntime-linux-x64-gpu-1.22.0.tgz
```

### MacOS

```shell
wget "https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-arm64-1.20.0.tgz"
ouch decompress onnxruntime-osx-arm64-1.20.0.tgz
rm onnxruntime-osx-arm64-1.20.0.tgz
```

## Download Models

### Silero VAD Models

The Silero VAD models can be downloaded from the [Silero VAD repository](https://github.com/snakers4/silero-vad).

### PyAnnote Models

You can export PyAnnote models to ONNX format using the [pyannote-onnx-rust](https://github.com/RustedBytes/pyannote-onnx-rust) repository or use pre-exported models from Hugging Face.

For example, to get the PyAnnote segmentation model:
```shell
# You can use pyannote.audio to export models to ONNX format
# Or download pre-converted models from appropriate sources
```

## Usage

### Linux

#### ONNX Runtime with Silero VAD
```shell
RUST_LOG=debug cargo run -- --runtime onnxruntime --vad-model silero --dylib-path onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so.1.22.0 --model-path ./models/silero_vad_v5.onnx --process-audio test-audios/test_16khz.wav --output test --output-format wav --debug

RUST_LOG=debug cargo run -- --runtime onnxruntime --vad-model silero --dylib-path onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so.1.22.0 --model-path ./models/silero_vad_v5.onnx --process-audio test-audios/test_16khz_stereo.wav --output test --output-format wav --debug
```

#### ONNX Runtime with PyAnnote VAD
```shell
RUST_LOG=debug cargo run -- --runtime onnxruntime --vad-model pyannote --dylib-path onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so.1.22.0 --model-path ./models/pyannote_segmentation.onnx --process-audio test-audios/test_16khz.wav --output test --output-format wav --debug
```

#### Candle
```shell
RUST_LOG=debug cargo run -- --runtime candle --vad-model silero --model-path ./models/xenova_silero_vad_v5.onnx --process-audio test-audios/test_16khz.wav --output test --output-format wav --debug
```

### MacOS

#### ONNX Runtime with Silero VAD
```shell
RUST_LOG=debug cargo run -- --runtime onnxruntime --vad-model silero --dylib-path onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib --model-path ./models/silero_vad_v5.onnx --process-audio test-audios/test_16khz.wav --output test --output-format wav --debug

RUST_LOG=debug cargo run -- --runtime onnxruntime --vad-model silero --dylib-path onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib --model-path ./models/silero_vad_v5.onnx --process-audio test-audios/test_16khz_stereo.wav --output test --output-format wav --debug
```

#### ONNX Runtime with PyAnnote VAD
```shell
RUST_LOG=debug cargo run -- --runtime onnxruntime --vad-model pyannote --dylib-path onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib --model-path ./models/pyannote_segmentation.onnx --process-audio test-audios/test_16khz.wav --output test --output-format wav --debug
```

## Cite

```
@software{Smoliakov_Extract_Speech_2025,
  author = {Smoliakov, Yehor},
  month = oct,
  title = {{extract-speech: Extract speech from audio files by a Voice Activity Detection models}},
  url = {https://github.com/RustedBytes/extract-speech},
  version = {0.5.0},
  year = {2025}
}
```
