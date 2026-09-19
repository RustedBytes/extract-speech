# Development

## Repository layout

| Path | Responsibility |
| --- | --- |
| `src/lib.rs` | stable public library surface and compatibility re-exports |
| `src/vad/` | model-independent detector, configuration, timestamps, and shared segmentation state machine |
| `src/models/` | one submodule per model family, with runtime backends, frontends, and iterators grouped together |
| `src/assets/` | pinned model/runtime registry, SHA-256 verification, extraction, and caching |
| `src/audio/` | audio decoding, mono conversion, resampling, and Ogg Opus encoding |
| `src/bindings/` | optional foreign-language bindings, currently PyO3 |
| `src/cli/` | CLI arguments, runtime orchestration, folder processing, and output writing |
| `src/main.rs` | minimal CLI entry point |
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
cargo clippy --all-targets --all-features -- -D warnings -W clippy::pedantic
cargo test --all-targets --all-features
cargo build --features cli
```

Build and smoke-test the Python extension in uv's locked environment with:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python -m unittest discover -s tests/python -v
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
cargo run --features cli -- \
  --vad-model silero \
  --model-path models/silero-vad-v6.onnx \
  --process-audio test-audios/test_16khz.wav \
  --output target/manual-output
```

Run the end-to-end model suite with:

```bash
just test-models
# or: ./scripts/test-models.sh
```

The suite downloads checksum-verified, revision-pinned Silero v5 and v6, PyAnnote, FunASR FSMN-VAD, TEN VAD, and MarbleNet ONNX models, PulseVAD models from its official repository, and ONNX Runtime for Linux x86-64. Downloads are cached under `target/model-test-cache`, and outputs are written to `target/model-test-output`. It first exercises the public `Detector` API with a real Silero v6 model, then runs every supported CLI model/runtime combination against the mono 16 kHz, stereo 16 kHz, and mono 24 kHz fixtures and validates each WAV output and metadata file.

## Design notes

- VAD always runs on mono, 16 kHz samples. `--sample-rate` applies to final output.
- Model state is reset for every input.
- `VadModel` in `src/vad/iterator.rs` is the common probability-inference boundary used by the Silero, PulseVAD, and TEN VAD backends.
- PyAnnote, FSMN-VAD, and MarbleNet use dedicated iterators for their frame-level outputs and specialized frontends.
- Errors at file, decoder, model, resampler, and writer boundaries should include enough context to identify the failing input.
- Production code should propagate recoverable errors instead of panicking.
- Behavioral fixes should include focused unit or regression tests.

## Documentation changes

Keep the root README focused on discovery and first use. Put detailed operational guidance in `docs/`, and update both the generated CLI behavior and its documentation when adding or renaming options.

Markdown links are checked automatically by the `check-links.yml` workflow.

## Release builds

Create an optimized binary with:

```bash
cargo build --release --features cli
```

The release profile enables optimization, link-time optimization, symbol stripping, and no debug information. The matrix-based release workflow builds Linux x86-64, macOS aarch64, and Windows x86-64 CLI artifacts plus CPython 3.9–3.14 wheels. Every wheel is installed and imported through uv before upload. Tagged pushes publish the binaries and wheels together in a GitHub Release.
