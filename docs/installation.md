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

After publication, install the optional command-line application from crates.io with:

```bash
cargo install extract-speech --features cli
```

To build the repository directly:

```bash
git clone https://github.com/RustedBytes/extract-speech.git
cd extract-speech
cargo build --release --features cli
```

The resulting executable is:

- `target/release/extract-speech` on Linux and macOS
- `target/release/extract-speech.exe` on Windows

Use `cargo build --features cli` instead when you want a faster development build. A plain `cargo build` builds the library only.

### macOS Accelerate

On Apple platforms, the optional Accelerate integration can be enabled with:

```bash
cargo build --release --features "cli,accelerate-src"
```

## Prebuilt releases

Platform CLI binaries and Python wheels can be downloaded from the project's
[GitHub releases](https://github.com/RustedBytes/extract-speech/releases). Python wheels are also
published to PyPI for CPython 3.9 through 3.15 on manylinux x86-64, macOS ARM64, and Windows
x86-64.

## Next steps

The Candle backend needs only a compatible ONNX model. The ONNX Runtime backend additionally needs a dynamic ONNX Runtime library at execution time. Continue with [Models and runtimes](models-and-runtimes.md), then see the [usage guide](usage.md).
