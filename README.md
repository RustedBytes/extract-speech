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


## Build

You need: cargo, rustc, cross, podman, goreleaser.

0. build images and increase resources for podman:

```shell
podman build --platform=linux/amd64 -f dockerfiles/Dockerfile.aarch64-unknown-linux-gnu -t aarch64-unknown-linux-gnu:my-edge .
podman build --platform=linux/amd64 -f dockerfiles/Dockerfile.x86_64-unknown-linux-gnu -t x86_64-unknown-linux-gnu:my-edge .

podman machine set --cpus 4 --memory 8192
```

1. make binaries:

```shell
goreleaser build --clean --snapshot --id extract-speech --timeout 60m
```
