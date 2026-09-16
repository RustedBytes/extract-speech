# Library

The `extract-speech` crate exposes a reusable VAD inference API. It accepts raw mono `f32` PCM normalized to `-1.0..=1.0` at 16 kHz and returns `SpeechSegment` values whose `start` and `end` fields are sample offsets into the input.

## Installation

Use the default features to enable both inference engines:

```toml
[dependencies]
extract-speech = "0.6"
```

Select one engine when a smaller dependency graph is preferred:

```toml
[dependencies]
extract-speech = { version = "0.6", default-features = false, features = ["candle"] }
```

The available features are:

| Feature | Default | Contents |
| --- | --- | --- |
| `candle` | Yes | Candle backends for Silero, PulseVAD, and MarbleNet |
| `onnxruntime` | Yes | ONNX Runtime backends for every supported model |
| `cli` | No | Command-line application, audio decoding, resampling, and output encoding |
| `accelerate-src` | No | Apple Accelerate integration; also enables `candle` |

## Candle inference

```rust,no_run
use extract_speech::{Detector, Model, Result, Runtime, VadParams, SAMPLE_RATE};

fn detect(samples: &[f32]) -> Result<()> {
    assert_eq!(SAMPLE_RATE, 16_000);

    let mut detector = Detector::builder("models/silero-vad-v5.onnx")
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

The `onnxruntime` feature uses dynamic loading. Initialize ONNX Runtime once, before building any ONNX-backed detector, and pass the same execution-provider configuration to the builder:

```rust,no_run
use extract_speech::{init_onnx_runtime, Detector, Model, Result, Runtime};
use extract_speech::ort::ep::{CPU, CUDA};

fn main() -> Result<()> {
    let providers = vec![CUDA::default().build(), CPU::default().build()];
    init_onnx_runtime("/path/to/libonnxruntime.so", providers.clone())?;

    let mut detector = Detector::builder("models/silero-vad-v5.onnx")
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

Use the platform-specific ONNX Runtime dynamic-library filename on macOS or Windows. Execution providers that fail to initialize fall back according to ONNX Runtime's provider behavior.

## Model/runtime compatibility

| Model | Candle | ONNX Runtime |
| --- | --- | --- |
| Silero VAD v5 | Yes | Yes |
| PulseVAD | FP32 | FP32 and INT8 |
| PyAnnote segmentation | No | Yes |
| FunASR FSMN-VAD | No | FP32 and INT8 |
| TEN VAD | No | Yes |
| NVIDIA Frame-VAD MarbleNet | FP32 | FP32 and INT8 |

See [Models and runtimes](models-and-runtimes.md) for compatible model files, pinned downloads used by the integration tests, and model-specific licensing notes.

## Lower-level API

The model implementations, frontends, iterator types, `VadModel` trait, and generic `VadIter` are public for applications that need to compose a custom inference pipeline. Most applications should use `Detector`, which provides one stable entry point across models and runtimes.
