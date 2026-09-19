# Library

The `extract-speech` crate exposes a reusable VAD inference API. It accepts raw mono `f32` PCM normalized to `-1.0..=1.0` at 16 kHz and returns `SpeechSegment` values whose `start` and `end` fields are sample offsets into the input.

## Installation

Use the default features to enable both inference engines:

```toml
[dependencies]
extract-speech = "0.8"
```

Select one engine when a smaller dependency graph is preferred:

```toml
[dependencies]
extract-speech = { version = "0.8", default-features = false, features = ["candle"] }
```

The available features are:

| Feature | Default | Contents |
| --- | --- | --- |
| `candle` | Yes | Candle backends for Silero, PulseVAD, and MarbleNet |
| `onnxruntime` | Yes | ONNX Runtime backends for every supported model |
| `download` | Yes | Revision-pinned, checksum-verified model and ONNX Runtime downloads |
| `python` | No | PyO3 extension module, Python inference API, and download helpers |
| `cli` | No | Command-line application, audio decoding, resampling, and output encoding |
| `accelerate-src` | No | Apple Accelerate integration; also enables `candle` |

## Automatic model downloads and caching

The `download` feature provides one-call model bundles. Every download is pinned to a specific upstream revision, verified with SHA-256, written atomically, and reused from the cache on later calls. FSMN bundles include the required `vad.mvn` file next to the ONNX graph.

```rust,no_run
use extract_speech::{
    download::{AssetManager, ModelAsset},
    Detector, Model, Result, Runtime,
};

fn main() -> Result<()> {
    let assets = AssetManager::default_cache()?;
    let files = assets.model(ModelAsset::SileroV6)?;

    let mut detector = Detector::builder(files.model_path())
        .model(Model::Silero)
        .runtime(Runtime::Candle)
        .build()?;

    let segments = detector.detect(&vec![0.0; 16_000])?;
    println!("{} speech segments", segments.len());
    Ok(())
}
```

`AssetManager::all_models()` downloads all ten model variants with one call. To choose an application-specific cache location, use `AssetManager::new(path)`. Otherwise `AssetManager::default_cache()` uses:

- `EXTRACT_SPEECH_CACHE_DIR` when set;
- `%LOCALAPPDATA%/extract-speech` on Windows;
- `~/Library/Caches/extract-speech` on macOS;
- `$XDG_CACHE_HOME/extract-speech` or `~/.cache/extract-speech` on other Unix systems.

The upstream model licenses still apply to downloaded artifacts. In particular, review the TEN VAD and NVIDIA MarbleNet licensing notes in [Models and runtimes](models-and-runtimes.md).

## Candle inference

```rust,no_run
use extract_speech::{Detector, Model, Result, Runtime, VadParams, SAMPLE_RATE};

fn detect(samples: &[f32]) -> Result<()> {
    assert_eq!(SAMPLE_RATE, 16_000);

    let mut detector = Detector::builder("models/silero-vad-v6.onnx")
        .model(Model::Silero)
        .runtime(Runtime::Candle)
        .parameters(VadParams {
            threshold: 0.7,
            min_speech_duration_ms: 250,
            ..VadParams::default()
        })
        .build()?;

    for segment in detector.detect(samples)? {
        let speech_samples = &samples[segment.start..segment.end];
        println!("detected {} samples", speech_samples.len());
    }

    Ok(())
}
```

`Detector::detect` resets recurrent state before each call, so a detector can be reused for independent audio buffers. Create one detector per concurrent worker because inference sessions are mutable.

## ONNX Runtime inference

The `onnxruntime` feature uses dynamic loading. `AssetManager::onnx_bundle()` prepares the model, its sidecars, and the compatible CPU runtime with one call on Linux x86-64/ARM64, macOS ARM64, or Windows x86-64/ARM64. Initialize the returned runtime once before building any ONNX-backed detector:

```rust,no_run
use extract_speech::{
    download::{AssetManager, ModelAsset},
    init_onnx_runtime, Detector, Model, Result, Runtime,
};
use extract_speech::ort::ep::CPU;

fn main() -> Result<()> {
    let providers = vec![CPU::default().build()];
    let assets = AssetManager::default_cache()?;
    let bundle = assets.onnx_bundle(ModelAsset::SileroV6)?;
    init_onnx_runtime(bundle.runtime().library_path(), providers.clone())?;

    let mut detector = Detector::builder(bundle.model().model_path())
        .model(Model::Silero)
        .runtime(Runtime::OnnxRuntime)
        .execution_providers(providers)
        .build()?;

    let samples = vec![0.0_f32; 16_000];
    let segments = detector.detect(&samples)?;
    println!("{} speech segments", segments.len());
    Ok(())
}
```

The automatic runtime bundle is the upstream CPU distribution. For CUDA or TensorRT, install the appropriate GPU distribution yourself and pass its dynamic-library path to `init_onnx_runtime`. Execution providers that fail to initialize fall back according to ONNX Runtime's provider behavior.

## Model/runtime compatibility

| Model | Candle | ONNX Runtime |
| --- | --- | --- |
| Silero VAD v5 and v6 | Yes | Yes |
| PulseVAD | FP32 | FP32 and INT8 |
| PyAnnote segmentation | Yes | Yes |
| FunASR FSMN-VAD | FP32 and dequantized INT8 | FP32 and INT8 |
| TEN VAD | No | Yes |
| NVIDIA Frame-VAD MarbleNet | FP32 | FP32 and INT8 |

See [Models and runtimes](models-and-runtimes.md) for compatible model files, pinned downloads used by the integration tests, and model-specific licensing notes.

## Logging

The library emits diagnostics through the [`log`](https://docs.rs/log) facade
and does not install a logger or choose a level filter. Applications can use
any compatible logger implementation and enable the `Debug` level to inspect
model inputs, probabilities, and segmentation transitions.

## Lower-level API

The model implementations, frontends, iterator types, `VadModel` trait, and generic `VadIter` are public for applications that need to compose a custom inference pipeline. Most applications should use `Detector`, which provides one stable entry point across models and runtimes.
