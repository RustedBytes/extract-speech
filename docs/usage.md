# Usage

Every operation requires an ONNX model through `--model-path`. Audio processing additionally requires exactly one of `--process-audio` and `--process-folder`.

## Process one file

Write one WAV file per detected speech region:

```bash
extract-speech \
  --model-path models/silero-vad-v5.onnx \
  --process-audio recording.wav \
  --output clips
```

The `clips` directory will contain names such as `1730135784123_0.wav`. Names contain the creation timestamp and segment index.

Write all detected regions into one output file:

```bash
extract-speech \
  --model-path models/silero-vad-v5.onnx \
  --process-audio recording.wav \
  --output-type concatenated \
  --output speech.wav
```

With `--output-type concatenated`, `--output` is the complete destination filename for a single input.

## Process a directory

```bash
extract-speech \
  --model-path models/silero-vad-v5.onnx \
  --process-folder recordings \
  --output output
```

Directory processing is non-recursive and processes supported files in parallel. Recognized extensions are `wav`, `mp3`, `flac`, `ogg`, `opus`, `m4a`, and `aac`.

With the default `--output-type files`, each input gets its own output directory:

```text
output/
├── interview-1/
│   ├── 1730135784123_0.wav
│   └── 1730135784124_1.wav
└── interview-2/
    └── 1730135785123_0.wav
```

With `--output-type concatenated`, `--output` remains a directory and each input produces a file named after its input stem, such as `output/interview-1.wav`.

## Output formats and sample rates

Select the format with `--output-format wav`, `opus`, or `ogg`.

- `wav` writes mono, 16-bit PCM WAV data.
- `opus` and `ogg` both write mono Opus audio in an Ogg container; the selected value determines the filename extension.
- `--sample-rate` controls the final output rate and defaults to `16000` Hz.

Input is decoded, mixed down to mono when necessary, and converted to 16 kHz for VAD. Detected samples are then converted to the requested output rate. Opus is encoded internally at 48 kHz while preserving the requested rate in its identification header.

## Tune detection

`--threshold` accepts a value from `0.0` to `1.0` and defaults to `0.7`:

- lower values detect quieter speech but may include more noise
- higher values are more selective but may miss quiet speech

For example:

```bash
extract-speech \
  --model-path models/silero-vad-v5.onnx \
  --process-audio recording.wav \
  --threshold 0.5 \
  --output clips
```

PulseVAD's recommended starting threshold is `0.5`, so specify `--threshold 0.5` when using `--vad-model pulsevad`. See [Models and runtimes](models-and-runtimes.md#pulsevad) for model downloads and runtime compatibility.

## Use separate detection and source audio

`--source-audio` lets the model detect speech in one file while extracting the matching regions from another. This is useful when detection works better on a denoised copy but output should come from the original recording.

```bash
extract-speech \
  --model-path models/silero-vad-v5.onnx \
  --process-audio denoised.wav \
  --source-audio original.wav \
  --output clips
```

Both files must have the same sample count after decoding, mono conversion, and conversion to 16 kHz. This mode is available only for single-file processing.

## Write metadata

Use `--metadata` to write a JSON summary:

```bash
extract-speech \
  --model-path models/silero-vad-v5.onnx \
  --process-audio recording.wav \
  --output clips \
  --metadata metadata.json
```

Example:

```json
{
  "intervals": [
    {
      "filename": "1730135784123_0.wav",
      "duration": "2.500000"
    }
  ],
  "total_seconds": "2.500000",
  "compute_seconds": "0.123456"
}
```

For directory processing, metadata is written once per input under the output directory. The supplied metadata stem is combined with the input stem, for example `metadata_interview-1.json`.

## Inspect a model

Model inspection does not require an audio input:

```bash
extract-speech \
  --model-path model.onnx \
  --print-model-info io
```

Accepted values are `graph`, `nodes`, and `io`.

## Logging

Set `RUST_LOG` to control log output. `--debug` also enables model-specific diagnostic details when debug logging is active.

```bash
RUST_LOG=debug extract-speech \
  --debug \
  --model-path models/silero-vad-v5.onnx \
  --process-audio recording.wav
```

## CLI reference

| Option | Meaning | Default |
| --- | --- | --- |
| `--model-path <PATH>` | ONNX model path; always required | — |
| `--runtime <RUNTIME>` | `candle` or `onnxruntime` | `candle` |
| `--vad-model <MODEL>` | `silero`, `pulsevad`, `pyannote`, or `fsmn` | `silero` |
| `--dylib-path <PATH>` | ONNX Runtime dynamic library; required for `onnxruntime` | — |
| `--process-audio <PATH>` | Single audio file used for VAD | — |
| `--process-folder <PATH>` | Directory of audio files used for VAD | — |
| `--source-audio <PATH>` | Alternate source for extracted samples | — |
| `--output <PATH>` | Output directory or concatenated file | `output` |
| `--metadata <PATH>` | Metadata JSON path or naming template | — |
| `--output-type <TYPE>` | `files` or `concatenated` | `files` |
| `--output-format <FORMAT>` | `wav`, `opus`, or `ogg` | `wav` |
| `--threshold <NUMBER>` | Detection threshold from `0.0` through `1.0` | `0.7` |
| `--sample-rate <HZ>` | Final output sample rate | `16000` |
| `--cuda` | Prefer the CUDA execution provider | off |
| `--trt` | Prefer the TensorRT execution provider | off |
| `--coreml` | Prefer the CoreML execution provider | off |
| `--debug` | Enable model diagnostic logging | off |
| `--print-model-info <KIND>` | Print `graph`, `nodes`, or `io` model details | — |

Run `extract-speech --help` to view the reference generated by the installed version.
