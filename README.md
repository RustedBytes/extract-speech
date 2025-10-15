# extract-speech

Extract speech from audio files by a Voice Activity Detection model

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

## Usage

### Linux

#### ONNX Runtime
```shell
RUST_LOG=debug cargo run -- --runtime onnxruntime --dylib-path onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so.1.22.0 --model-path ./models/silero_vad_v5.onnx --process-audio test-audios/test_16khz.wav --output test --output-format wav --debug

RUST_LOG=debug cargo run -- --runtime onnxruntime --dylib-path onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so.1.22.0 --model-path ./models/silero_vad_v5.onnx --process-audio test-audios/test_16khz_stereo.wav --output test --output-format wav --debug
```

#### Candle

```shell
RUST_LOG=debug cargo run -- --runtime candle --model-path ./models/xenova_silero_vad_v5.onnx --process-audio test-audios/test_16khz.wav --output test --output-format wav --debug
```

### MacOS

```shell
RUST_LOG=debug cargo run -- --runtime onnxruntime --dylib-path onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib --model-path ./models/silero_vad_v5.onnx --process-audio test-audios/test_16khz.wav --output test --output-format wav --debug

RUST_LOG=debug cargo run -- --runtime onnxruntime --dylib-path onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib --model-path ./models/silero_vad_v5.onnx --process-audio test-audios/test_16khz_stereo.wav --output test --output-format wav --debug
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
