# Repository guidance for Codex

## Scope and priorities

These instructions apply to the entire repository. Treat `Cargo.toml`, the public API in
`src/lib.rs`, and the behavior documented in `README.md` and `docs/` as the source of truth.

- Complete requested implementation work instead of stopping after proposing it.
- Preserve existing public Rust and Python APIs unless the task explicitly authorizes a breaking
  change.
- Prefer focused, readable changes over new abstractions. Reuse existing modules and dependencies.
- Do not add a production dependency unless it provides a clear benefit that cannot reasonably be
  achieved with the standard library or an existing dependency.
- Preserve unrelated user changes. Do not commit, push, publish, tag, or create releases unless the
  user explicitly requests that action.

## Project architecture

`extract-speech` is primarily a Rust library. The CLI and Python extension are optional consumers
of the same public detector API.

- `src/vad/`: runtime-independent detector, configuration, timestamps, and segmentation.
- `src/models/<family>/`: model-specific frontend, iterator, Candle backend, and/or ONNX Runtime
  backend. Keep model-family logic within its directory.
- `src/assets/`: revision-pinned, checksum-verified model and ONNX Runtime downloads.
- `src/audio/`: CLI-only decoding, resampling, and Ogg Opus output.
- `src/bindings/`: foreign-language bindings. Python bindings use PyO3.
- `src/cli/`: argument parsing and CLI orchestration. Use the public `Detector`; do not duplicate
  model/runtime dispatch here.
- `examples/`: compilable examples of supported public APIs.
- `scripts/test-models.sh`: networked end-to-end model suite.

The legacy flat Rust module paths are compatibility re-exports. Keep them working when moving or
renaming implementation modules.

## Inference invariants

- VAD input is mono, normalized `f32` PCM at 16 kHz. `--sample-rate` changes output encoding only.
- Each independent input must start with reset model and iterator state.
- Keep preprocessing behavior aligned between Candle and ONNX Runtime implementations of the same
  model.
- Validate tensor ranks, dimensions, and model outputs before indexing or converting them.
- Use checked integer conversions at file, tensor, duration, and sample-count boundaries.
- DSP index-to-float conversions may use a narrowly scoped Clippy allow with a comment explaining
  why the value is bounded and the conversion is intentional.
- Return contextual errors with `anyhow`; do not panic on recoverable input, filesystem, download,
  decoding, or inference failures.
- Model URLs must be revision-pinned and accompanied by verified SHA-256 values. Do not commit
  downloaded models, runtime archives, caches, or generated output.

## Rust conventions

- Write idiomatic Rust and let ownership express resource lifetime. Avoid `unsafe` unless required
  by a dependency boundary and explicitly justified.
- Keep feature gates accurate. The library must compile without default features and with each
  supported feature set independently.
- Document new public items with Rustdoc. Functions returning `Result` need an accurate `# Errors`
  section; useful returned values should use `#[must_use]` where appropriate.
- Prefer specific types and checked conversions over unchecked `as` casts at external boundaries.
- Use `rustfmt`; do not hand-format around it.
- Add focused tests for behavior changes and regressions. Do not add tests that merely duplicate the
  implementation or test compiler-enforced facts.

## Python conventions

- Use `uv` for Python environments, commands, and dependency changes. Do not invoke `pip` directly.
- Keep `extract_speech.pyi`, PyO3 exports, examples, and `docs/python.md` synchronized.
- Preserve strict Pyright compatibility and Ruff formatting/linting.
- Build Python artifacts with Maturin using the `python` feature.

## Verification

Run the smallest relevant checks while iterating, then run the required checks for the changed
surface before handing work back.

Core Rust checks:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings -W clippy::pedantic
cargo test --all-targets --all-features
```

For library modules, public APIs, or feature-gate changes, also check applicable isolated builds:

```bash
cargo check --lib --no-default-features
cargo check --lib --no-default-features --features candle
cargo check --lib --no-default-features --features onnxruntime
cargo check --lib --no-default-features --features download
cargo check --lib --no-default-features --features python
```

For Python changes:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python -m unittest discover -s tests/python -v
```

Additional checks by scope:

- Run `cargo package --allow-dirty --locked` after changing package metadata, included files, or the
  public crate layout.
- Run `actionlint` for modified GitHub Actions workflows when it is available.
- Run `just test-models` for model preprocessing/inference changes when networked model downloads
  are appropriate. It is not required for documentation-only or unrelated changes.
- Use `just fmt`, `just clippy`, and `just test` as shortcuts for the core commands.

If a required check cannot run because of the host platform, unavailable network, model size, or a
missing external runtime, report the exact skipped command and reason.

## Documentation and releases

- Keep the root README concise; put detailed operational guidance in `docs/`.
- Update examples and documentation when changing CLI flags, features, supported platforms, model
  compatibility, download behavior, or public APIs.
- Record user-visible changes under `Unreleased` in `CHANGELOG.md` using Keep a Changelog sections.
- Keep release workflow artifacts aligned: CLI archives and CPython wheels are attached to tagged
  GitHub Releases.
