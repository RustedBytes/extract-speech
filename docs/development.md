# Development

## Repository layout

| Path | Responsibility |
| --- | --- |
| `src/main.rs` | CLI, runtime selection, folder processing, and output orchestration |
| `src/audio.rs` | audio probing, decoding, mono conversion, and input resampling |
| `src/vad_iter.rs` | shared Silero segmentation state machine |
| `src/silero_v5.rs` | Candle implementation of Silero VAD v5 |
| `src/silero_v5_ort.rs` | ONNX Runtime implementation of Silero VAD v5 |
| `src/pulsevad_frontend.rs` | PulseVAD pre-emphasis and normalized log-mel frontend |
| `src/pulsevad.rs` | Candle implementation of PulseVAD |
| `src/pulsevad_ort.rs` | ONNX Runtime implementation of PulseVAD |
| `src/pulsevad_iter.rs` | PulseVAD overlapping-window segmentation |
| `src/pyannote_vad_ort.rs` | ONNX Runtime PyAnnote inference |
| `src/pyannote_vad_iter.rs` | PyAnnote logits-to-segments processing |
| `src/resampler.rs` | sample-rate conversion |
| `src/opus.rs` | Ogg Opus encoding |
| `src/utils.rs` | shared VAD configuration and timestamps |
| `test-audios/` | small fixtures for local testing |

## Processing flow

1. Symphonia detects and decodes the input format.
2. Stereo input is mixed to mono and input is resampled to 16 kHz.
3. Candle or ONNX Runtime evaluates the selected VAD model.
4. The model-specific iterator converts probabilities or logits into sample ranges.
5. The ranges are extracted, optionally resampled, and written as WAV or Ogg Opus.
6. Optional metadata records output names, durations, and inference time.

Directory mode performs this flow concurrently with Rayon. Each file receives its own model instance so mutable inference state is not shared between jobs.

## Local checks

Run the same core checks expected in CI before submitting a change:

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build
```

The repository also provides `just` recipes:

```bash
just fmt
just clippy
just test
just release
```

Use the sample audio fixtures for manual smoke tests. A model file is not stored in the repository, so provide one explicitly:

```bash
cargo run -- \
  --model-path models/silero-vad-v5.onnx \
  --process-audio test-audios/test_16khz.wav \
  --output target/manual-output
```

Run the end-to-end model suite with:

```bash
just test-models
# or: ./scripts/test-models.sh
```

The suite downloads checksum-verified, revision-pinned Silero, PyAnnote, and FunASR FSMN-VAD ONNX models from Hugging Face, PulseVAD models from its official repository, and ONNX Runtime for Linux x86-64. Downloads are cached under `target/model-test-cache`, and outputs are written to `target/model-test-output`. It runs every supported model/runtime combination against the mono 16 kHz, stereo 16 kHz, and mono 24 kHz fixtures, then validates each WAV output and metadata file.

## Design notes

- VAD always runs on mono, 16 kHz samples. `--sample-rate` applies to final output.
- Model state is reset for every input.
- `VadModel` in `src/vad_iter.rs` is the common probability-inference boundary used by the Silero and PulseVAD backends.
- PyAnnote and FSMN-VAD use dedicated iterators for their frame-level outputs and specialized frontends.
- Errors at file, decoder, model, resampler, and writer boundaries should include enough context to identify the failing input.
- Production code should propagate recoverable errors instead of panicking.
- Behavioral fixes should include focused unit or regression tests.

## Documentation changes

Keep the root README focused on discovery and first use. Put detailed operational guidance in `docs/`, and update both the generated CLI behavior and its documentation when adding or renaming options.

Markdown links are checked automatically by the `check-links.yml` workflow.

## Release builds

Create an optimized binary with:

```bash
cargo build --release
```

The release profile enables optimization, link-time optimization, symbol stripping, and no debug information. The matrix-based release workflow builds Linux x86-64, macOS aarch64, and Windows x86-64 artifacts on tagged pushes or manual runs. Tagged pushes also publish all three binaries to a GitHub Release.
