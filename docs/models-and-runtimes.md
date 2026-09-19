# Models and runtimes

`extract-speech` reads ONNX model files through either Candle or ONNX Runtime. Model architecture and runtime must be selected together.

## Compatibility matrix

| VAD model | Candle | ONNX Runtime |
| --- | --- | --- |
| Silero VAD v5 and v6 | Supported | Supported |
| PulseVAD FP32 | Supported | Supported |
| PulseVAD INT8 QDQ | Supported (dequantized) | Supported |
| PyAnnote segmentation | Supported | Supported |
| FunASR FSMN-VAD FP32 | Supported | Supported |
| FunASR FSMN-VAD INT8 | Supported (dequantized) | Supported |
| TEN VAD | Supported | Supported |
| NVIDIA Frame-VAD MarbleNet FP32 | Supported | Supported |
| NVIDIA Frame-VAD MarbleNet INT8 | Supported (dequantized) | Supported |

The internal VAD sample rate is 16 kHz for every backend.

## Automatic downloads

CLI users can cache any supported model or the compatible CPU ONNX Runtime distribution directly:

```bash
extract-speech download silero
extract-speech download onnxruntime
```

Use `extract-speech download all` to cache every model variant and `--cache-dir <PATH>` to override the default cache location. Each invocation prints the downloaded model or runtime library path.

Library users can download a complete, revision-pinned model bundle with `AssetManager`. Downloads are SHA-256 verified and reused from a persistent cache:

```rust,no_run
use extract_speech::{download::{AssetManager, ModelAsset}, Result};

fn main() -> Result<()> {
    let assets = AssetManager::default_cache()?;
    let files = assets.model(ModelAsset::FsmnVadFp32)?;
    println!("{}", files.model_path().display());
    Ok(())
}
```

This call downloads both `model.onnx` and the required `vad.mvn`. Other `ModelAsset` variants download their complete bundles in the same way. `AssetManager::all_models()` prepares every supported variant, `AssetManager::onnx_runtime()` prepares the compatible ONNX Runtime dynamic library, and `AssetManager::onnx_bundle()` prepares both a selected model and the runtime with one call.

## Silero VAD

The `silero` download alias selects the official Silero VAD v6 model. The older
`silero-v5` asset remains available for compatibility:

```bash
extract-speech download silero
extract-speech download silero-v5
```

To download the revision-pinned v6 graph manually:

```bash
mkdir -p models
curl -L \
  https://raw.githubusercontent.com/snakers4/silero-vad/60b7ffa243625ebdc1070275a29f18c87843786a/src/silero_vad/data/silero_vad.onnx \
  -o models/silero-vad-v6.onnx
echo '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3  models/silero-vad-v6.onnx' \
  | sha256sum --check
```

Run it with:

```bash
extract-speech \
  --runtime candle \
  --vad-model silero \
  --model-path models/silero-vad-v6.onnx \
  --process-audio input.wav
```

Both versions use the same frontend and recurrent-state contract. Compatible
Silero exports must expose the expected `input`, `state`, and `sr` inputs and
the `output` and `stateN` outputs.

## PulseVAD

[PulseVAD](https://github.com/AydinAdnan/PulseVAD) processes 200 ms windows of 16 kHz mono audio. `extract-speech` includes its required pre-emphasis and normalized 64-bin log-mel frontend, and evaluates windows with the reference 100 ms hop.

Download the FP32 model for Candle:

```bash
mkdir -p models
curl -L \
  https://raw.githubusercontent.com/AydinAdnan/PulseVAD/af25e79d66830a3fee74541812721f6158fc92b5/pulsevad/data/pulsevad_2.1k.onnx \
  -o models/pulsevad_2.1k.onnx
echo '2b8c4874fc4ecd64916fc8726e2a8281b1cb9457c21f42a23a9776a4d538c665  models/pulsevad_2.1k.onnx' \
  | sha256sum --check
```

Run it with:

```bash
extract-speech \
  --runtime candle \
  --vad-model pulsevad \
  --model-path models/pulsevad_2.1k.onnx \
  --threshold 0.5 \
  --process-audio input.wav
```

Candle and ONNX Runtime can use either the FP32 model or the QDQ-quantized INT8 model. Download the smaller INT8 graph with:

```bash
curl -L \
  https://raw.githubusercontent.com/AydinAdnan/PulseVAD/af25e79d66830a3fee74541812721f6158fc92b5/pulsevad/data/pulsevad_2.1k_int8.onnx \
  -o models/pulsevad_2.1k_int8.onnx
echo '416061347a1e723ed15163acd51006bf3c513b27bb9f57d85e2c694cc44b8389  models/pulsevad_2.1k_int8.onnx' \
  | sha256sum --check

extract-speech \
  --runtime candle \
  --vad-model pulsevad \
  --model-path models/pulsevad_2.1k_int8.onnx \
  --threshold 0.5 \
  --process-audio input.wav
```

Use `--runtime onnxruntime` and supply `--dylib-path` to execute the original QDQ graph. Candle dequantizes its constant INT8 weights at model load and bypasses activation QDQ pairs because Candle does not implement those operators. PulseVAD's reference threshold is `0.5`; pass it explicitly because the CLI-wide default remains `0.7`. Compatible PulseVAD exports must accept an input named `log_mel` shaped as `[batch, 64, 21]` and return two-class logits in an output named `logits`.

## PyAnnote

PyAnnote is available through Candle and ONNX Runtime. Run the revision-pinned FP32 export with Candle:

```bash
extract-speech \
  --runtime candle \
  --vad-model pyannote \
  --model-path models/pyannote-segmentation.onnx \
  --process-audio input.wav
```

To use ONNX Runtime instead:

```bash
extract-speech \
  --runtime onnxruntime \
  --vad-model pyannote \
  --dylib-path /path/to/libonnxruntime.so \
  --model-path models/pyannote-segmentation.onnx \
  --process-audio input.wav
```

The exported model must accept mono audio shaped as `[batch, channel, samples]` and expose a three-dimensional output named `logits` with the shape `[batch, frames, classes]`.

## FunASR FSMN-VAD

[FunASR FSMN-VAD](https://huggingface.co/funasr/fsmn-vad-onnx) FP32 and INT8 graphs are available through Candle and ONNX Runtime. Download either model graph together with its CMVN coefficients:

```bash
mkdir -p models/fsmn-vad
curl -L \
  https://huggingface.co/funasr/fsmn-vad-onnx/resolve/f6e9fbb4cefa7397216c763f21307993f147f585/model.onnx \
  -o models/fsmn-vad/model.onnx
curl -L \
  https://huggingface.co/funasr/fsmn-vad-onnx/resolve/f6e9fbb4cefa7397216c763f21307993f147f585/vad.mvn \
  -o models/fsmn-vad/vad.mvn
echo '756887ce01695a9bb00dd85ca0f743653de03b18ba54d2e9ef4f4bb9b3edbf9f  models/fsmn-vad/model.onnx' \
  | sha256sum --check
echo '6820fef9687708c4fc3fab2530179c8fcea6262daa25514380056cd8f6eb1754  models/fsmn-vad/vad.mvn' \
  | sha256sum --check
```

Run the FP32 model with Candle:

```bash
extract-speech \
  --runtime candle \
  --vad-model fsmn \
  --model-path models/fsmn-vad/model.onnx \
  --process-audio input.wav
```

To use ONNX Runtime instead:

```bash
extract-speech \
  --runtime onnxruntime \
  --vad-model fsmn \
  --dylib-path /path/to/libonnxruntime.so \
  --model-path models/fsmn-vad/model.onnx \
  --process-audio input.wav
```

The quantized `model_quant.onnx` file is used in the same way. Candle dequantizes its constant INT8 weights at model load because Candle does not implement the graph's dynamic integer matrix multiplication operators; ONNX Runtime executes the original quantized graph. Keep `vad.mvn` (or the legacy name `am.mvn`) in the same directory as the selected ONNX file. `extract-speech` implements the model's 80-bin Kaldi filterbank, five-frame LFR stacking, CMVN, and recurrent FSMN cache inputs internally.

## TEN VAD

[TEN VAD](https://huggingface.co/TEN-framework/ten-vad) is available through Candle and ONNX Runtime. Download the official ONNX model:

```bash
mkdir -p models/ten-vad
curl -L \
  https://huggingface.co/TEN-framework/ten-vad/resolve/bda8ffc78b1846c5c7cbd38f04e52deff49de707/src/onnx_model/ten-vad.onnx \
  -o models/ten-vad/ten-vad.onnx
echo 'e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4  models/ten-vad/ten-vad.onnx' \
  | sha256sum --check
```

Run it with the reference `0.5` threshold:

```bash
extract-speech \
  --runtime candle \
  --vad-model ten \
  --model-path models/ten-vad/ten-vad.onnx \
  --threshold 0.5 \
  --process-audio input.wav
```

To use ONNX Runtime instead, select `--runtime onnxruntime` and provide `--dylib-path` as shown for the other ONNX Runtime models.

TEN VAD consumes mono 16 kHz audio in 256-sample (16 ms) frames. `extract-speech` supplies its reference pre-emphasis, STFT, 40-bin mel, LPC pitch, three-frame context, and recurrent-state processing. The model is distributed under TEN VAD's license, which adds conditions to Apache 2.0; review the upstream [`LICENSE`](https://huggingface.co/TEN-framework/ten-vad/blob/main/LICENSE) before deployment.

## NVIDIA Frame-VAD MarbleNet

[Frame-VAD Multilingual MarbleNet v2.0](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0) is a compact, frame-based multilingual VAD. The FP32 and INT8 ONNX exports are available from the revision-pinned [vadonnx conversion repository](https://huggingface.co/TigreGotico/frame-vad-marblenet-onnx):

```bash
mkdir -p models/marblenet
curl -L \
  https://huggingface.co/TigreGotico/frame-vad-marblenet-onnx/resolve/e8786fe74e055954901eb553cc9c3145323981cc/marblenet.onnx \
  -o models/marblenet/marblenet.onnx
echo '4ad3364be94d462b5fd4fa39910c24967dbb9dba436e27bcff7a88359515e491  models/marblenet/marblenet.onnx' \
  | sha256sum --check
curl -L \
  https://huggingface.co/TigreGotico/frame-vad-marblenet-onnx/resolve/e8786fe74e055954901eb553cc9c3145323981cc/marblenet_int8.onnx \
  -o models/marblenet/marblenet_int8.onnx
echo '9c4462323f9b576fd5e581d3c86b9b9b513468d18a79bcdcd3a2bcbcaab02699  models/marblenet/marblenet_int8.onnx' \
  | sha256sum --check
```

Run either model with Candle:

```bash
extract-speech \
  --runtime candle \
  --vad-model marblenet \
  --model-path models/marblenet/marblenet.onnx \
  --threshold 0.5 \
  --process-audio input.wav
```

Candle and ONNX Runtime accept both `marblenet.onnx` and `marblenet_int8.onnx`. Candle dequantizes the INT8 graph's constant weights at model load and evaluates the graph in floating point. Select ONNX Runtime with `--runtime onnxruntime` and provide `--dylib-path` as shown for the other ONNX Runtime models.

The model consumes 80-bin log-mel features and produces two-class logits at a 20 ms resolution. `extract-speech` implements the NeMo pre-emphasis, centered 512-point STFT, non-periodic Hann window, Slaney-normalized mel filterbank, and stable softmax internally. The model and its exports use the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/); review it before deployment.

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
  --vad-model silero \
  --dylib-path /opt/onnxruntime/lib/libonnxruntime.so \
  --model-path models/silero-vad-v6.onnx \
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
  --vad-model silero \
  --cuda \
  --dylib-path /opt/onnxruntime-gpu/lib/libonnxruntime.so \
  --model-path models/silero-vad-v6.onnx \
  --process-audio input.wav
```

## Troubleshooting

### The runtime library cannot be loaded

Confirm that `--dylib-path` points to the dynamic library file rather than its directory. Also confirm that dependent CUDA, TensorRT, or system libraries are discoverable by the operating system.

### Model inputs or outputs are missing

The ONNX file is not compatible with the selected `--vad-model` or runtime. Inspect a model's interface with:

```bash
extract-speech --model-path model.onnx --print-model-info io
```

### No speech is detected

Verify that the model matches the selected runtime and model type, then try a lower threshold such as `--threshold 0.5`.

### Too much non-speech is detected

Raise the threshold, for example to `--threshold 0.8`.
