# Installation

## Requirements

Building `extract-speech` requires:

- a current stable [Rust toolchain](https://www.rust-lang.org/tools/install)
- a C/C++ build toolchain
- the Protocol Buffers compiler (`protoc`)

Install `protoc` on common platforms:

### Ubuntu and Debian

```bash
sudo apt-get update
sudo apt-get install protobuf-compiler
```

### macOS

```bash
brew install protobuf
```

### Windows

Download a `protoc` archive from the [Protocol Buffers releases](https://github.com/protocolbuffers/protobuf/releases), extract it, and add its `bin` directory to `PATH`.

## Build from source

```bash
git clone https://github.com/RustedBytes/extract-speech.git
cd extract-speech
cargo build --release
```

The resulting executable is:

- `target/release/extract-speech` on Linux and macOS
- `target/release/extract-speech.exe` on Windows

Use `cargo build` instead when you want a faster development build.

### macOS Accelerate

On Apple platforms, the optional Accelerate integration can be enabled with:

```bash
cargo build --release --features accelerate-src
```

## Prebuilt releases

When available, platform binaries can be downloaded from the project's [GitHub releases](https://github.com/RustedBytes/extract-speech/releases).

## Next steps

The Candle backend needs only a compatible ONNX model. The ONNX Runtime backend additionally needs a dynamic ONNX Runtime library at execution time. Continue with [Models and runtimes](models-and-runtimes.md), then see the [usage guide](usage.md).
