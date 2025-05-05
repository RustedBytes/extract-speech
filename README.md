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

## Download libonnxruntime

```shell
wget "https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-arm64-1.20.0.tgz"
ouch decompress onnxruntime-osx-arm64-1.20.0.tgz
rm onnxruntime-osx-arm64-1.20.0.tgz
```

## Usage

```shell
cargo run -- --runtime onnxruntime --dylib-path onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib --model-path ./models/silero_vad_v5.onnx --process-audio test-audios/test_16khz.wav --output test --output-format wav

cargo run -- --runtime onnxruntime --dylib-path onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib --model-path ./models/silero_vad_v5.onnx --process-audio test-audios/test_16khz_stereo.wav --output test --output-format wav
```
