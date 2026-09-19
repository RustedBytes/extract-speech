# Python API

The `python` Cargo feature builds a native Python extension with [PyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs/). It exposes the same inference implementations and checksum-verified asset cache as the Rust API.

## Build and install

Building requires Python 3.9 or newer, Rust, Protocol Buffers, and
[`uv`](https://docs.astral.sh/uv/). Create the locked environment and install the project with:

```bash
uv sync --no-dev
```

Alternatively, build a wheel:

```bash
uv build --wheel
```

Tagged GitHub releases contain prebuilt wheels for CPython 3.9 through 3.14 on
manylinux x86-64, macOS ARM64, and Windows x86-64. Download the wheel matching
your interpreter and platform, then install it into a uv environment:

```bash
uv venv
uv pip install ./extract_speech-*-cp312-cp312-manylinux_2_28_x86_64.whl
```

`uv` invokes maturin through `pyproject.toml`; maturin enables the `python` feature automatically.
Commands which import the local package should run through `uv run`, for example:

```bash
uv run python examples/python_inference.py test-audios/test_16khz.wav
```

## Automatic model loading

`Detector.from_pretrained` downloads a revision-pinned model bundle, verifies its SHA-256 checksum, and reuses it from the persistent cache on later calls:

```python
import array
import extract_speech

detector = extract_speech.Detector.from_pretrained(
    "silero",
    runtime="candle",
    threshold=0.7,
)

samples = array.array("f", [0.0] * extract_speech.SAMPLE_RATE)
segments = detector.detect(samples)

for segment in segments:
    print(segment.start, segment.end, segment.duration_seconds)
```

`detect` accepts Python sequences and contiguous `float32` buffer objects, including `array.array("f")` and NumPy `float32` arrays. Audio must be normalized mono PCM sampled at 16 kHz. Model loading, downloads, and inference release the Python GIL.

Supported model names are `silero`, `pyannote`, `pulsevad`, `fsmn`, `ten`, and `marblenet`. `silero` and `silero-v6` select Silero VAD v6; use `silero-v5` for the retained v5 model. Pass `quantized=True` for the PulseVAD, FSMN, or MarbleNet INT8 model. PyAnnote, FSMN, TEN, and all INT8 variants require `runtime="onnxruntime"`.

## Automatic ONNX Runtime loading

For ONNX Runtime, the same constructor downloads the CPU runtime and every file required by the selected model:

```python
import extract_speech

detector = extract_speech.Detector.from_pretrained(
    "fsmn",
    runtime="onnxruntime",
    quantized=True,
)
```

The automatic CPU runtime supports Linux x86-64/ARM64, macOS ARM64, and Windows x86-64/ARM64. To use a manually installed runtime, pass both paths explicitly:

```python
detector = extract_speech.Detector(
    "models/model.onnx",
    model="silero",
    runtime="onnxruntime",
    onnx_runtime_path="/opt/onnxruntime/lib/libonnxruntime.so",
)
```

Only one ONNX Runtime library can be initialized in a Python process. Reusing the same library is supported; attempting to switch library paths raises `RuntimeError`.

## Download helpers

Downloads can also be prepared without constructing a detector:

```python
from extract_speech import (
    default_cache_dir,
    download_all_models,
    download_model,
    download_onnx_runtime,
)

silero_path = download_model("silero")
fsmn_int8_path = download_model("fsmn", quantized=True)
runtime_path = download_onnx_runtime()
all_models = download_all_models()
print(default_cache_dir())
```

Set `EXTRACT_SPEECH_CACHE_DIR` or pass `cache_dir=` to override the platform cache location. The upstream licenses of downloaded models still apply.

See [`examples/python_inference.py`](../examples/python_inference.py) for a complete standard-library WAV example.

## Development checks

Install the development tools and run the Python quality gates with:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python -m unittest discover -s tests/python -v
```

Ruff checks and formats the Python example, tests, and type stub. Pyright runs in strict mode
against the public stub and the supported Python 3.9 syntax level.
