# Models and runtimes

`extract-speech` reads ONNX model files through either Candle or ONNX Runtime. Model architecture and runtime must be selected together.

## Compatibility matrix

| VAD model | Candle | ONNX Runtime |
| --- | --- | --- |
| Silero VAD v5 | Supported | Supported |
| PyAnnote segmentation | Not supported | Supported |

The internal VAD sample rate is 16 kHz for every backend.

## Silero VAD

For the default Candle backend, download the ONNX Community Silero export:

```bash
mkdir -p models
curl -L \
  https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx \
  -o models/silero-vad-v5.onnx
```

Run it with:

```bash
extract-speech \
  --runtime candle \
  --vad-model silero \
  --model-path models/silero-vad-v5.onnx \
  --process-audio input.wav
```

Compatible Silero exports must expose the expected `input`, `state`, and `sr` inputs and the `output` and `stateN` outputs.

## PyAnnote

PyAnnote is available only through ONNX Runtime:

```bash
extract-speech \
  --runtime onnxruntime \
  --vad-model pyannote \
  --dylib-path /path/to/libonnxruntime.so \
  --model-path models/pyannote-segmentation.onnx \
  --process-audio input.wav
```

The exported model must accept mono audio shaped as `[batch, channel, samples]` and expose a three-dimensional output named `logits` with the shape `[batch, frames, classes]`.

## ONNX Runtime

The ONNX Runtime backend loads its dynamic library at startup. Download a package for the target platform from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases), extract it, and pass the library itself to `--dylib-path`.

Common filenames are:

- Linux: `libonnxruntime.so` or a versioned `.so` file
- macOS: `libonnxruntime.dylib`
- Windows: `onnxruntime.dll`

Example:

```bash
extract-speech \
  --runtime onnxruntime \
  --dylib-path /opt/onnxruntime/lib/libonnxruntime.so \
  --model-path models/silero-vad-v5.onnx \
  --process-audio input.wav
```

The dynamic library version must be compatible with the `ort` crate version used by the project.

## Execution providers

ONNX Runtime starts with the CPU execution provider. Optional flags add an accelerated provider before CPU fallback:

- `--cuda` for CUDA
- `--trt` for TensorRT
- `--coreml` for CoreML

The supplied ONNX Runtime library must include the requested provider, and its platform dependencies must be installed. Provider availability is controlled by that library and the host system, not only by the CLI flag.

CUDA example:

```bash
extract-speech \
  --runtime onnxruntime \
  --cuda \
  --dylib-path /opt/onnxruntime-gpu/lib/libonnxruntime.so \
  --model-path models/silero-vad-v5.onnx \
  --process-audio input.wav
```

## Troubleshooting

### The runtime library cannot be loaded

Confirm that `--dylib-path` points to the dynamic library file rather than its directory. Also confirm that dependent CUDA, TensorRT, or system libraries are discoverable by the operating system.

### Model inputs or outputs are missing

The ONNX file is not compatible with the selected `--vad-model`. Inspect its interface with:

```bash
extract-speech --model-path model.onnx --print-model-info io
```

### No speech is detected

Verify that the model matches the selected runtime and model type, then try a lower threshold such as `--threshold 0.5`.

### Too much non-speech is detected

Raise the threshold, for example to `--threshold 0.8`.
