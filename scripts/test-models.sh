#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly CACHE_DIR="${MODEL_TEST_CACHE_DIR:-$REPO_DIR/target/model-test-cache}"
readonly OUTPUT_DIR="${MODEL_TEST_OUTPUT_DIR:-$REPO_DIR/target/model-test-output}"
readonly ONNXRUNTIME_VERSION="1.27.1"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "Model integration tests currently require Linux x86_64." >&2
    exit 1
fi

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

download_file() {
    local url="$1"
    local expected_sha256="$2"
    local destination="$3"

    if [[ -f "$destination" ]]; then
        local cached_sha256
        cached_sha256="$(sha256sum "$destination" | cut -d ' ' -f 1)"
        if [[ "$cached_sha256" == "$expected_sha256" ]]; then
            return
        fi
        mv "$destination" "${destination}.invalid.$(date +%s)"
    fi

    local temporary_file
    temporary_file="$(mktemp "${destination}.tmp.XXXXXX")"
    if ! curl --fail --location --retry 3 --retry-all-errors --silent --show-error \
        "$url" --output "$temporary_file"; then
        rm -f "$temporary_file"
        return 1
    fi

    local downloaded_sha256
    downloaded_sha256="$(sha256sum "$temporary_file" | cut -d ' ' -f 1)"
    if [[ "$downloaded_sha256" != "$expected_sha256" ]]; then
        echo "Checksum mismatch for $url" >&2
        rm -f "$temporary_file"
        return 1
    fi

    mv "$temporary_file" "$destination"
}

readonly SILERO_MODEL="$CACHE_DIR/silero-vad.onnx"
readonly PYANNOTE_MODEL="$CACHE_DIR/pyannote-segmentation-3.0.onnx"
readonly PULSEVAD_MODEL="$CACHE_DIR/pulsevad-2.1k.onnx"
readonly PULSEVAD_INT8_MODEL="$CACHE_DIR/pulsevad-2.1k-int8.onnx"
readonly ONNXRUNTIME_ARCHIVE="$CACHE_DIR/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
readonly ONNXRUNTIME_DIR="$CACHE_DIR/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"
readonly ONNXRUNTIME_LIBRARY="$ONNXRUNTIME_DIR/lib/libonnxruntime.so"

# Hugging Face downloads are revision-pinned and checksum-verified.
download_file \
    "https://huggingface.co/onnx-community/silero-vad/resolve/ddc9a7e80d6758f6fc795a1e8a04b798eb929d3a/onnx/model.onnx" \
    "a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808" \
    "$SILERO_MODEL"
download_file \
    "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/733a93b6473d019a773298e08cefa686894b1854/onnx/model.onnx" \
    "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25" \
    "$PYANNOTE_MODEL"

# PulseVAD has no Hugging Face repository, so use its official pinned release artifacts.
download_file \
    "https://raw.githubusercontent.com/AydinAdnan/PulseVAD/af25e79d66830a3fee74541812721f6158fc92b5/pulsevad/data/pulsevad_2.1k.onnx" \
    "2b8c4874fc4ecd64916fc8726e2a8281b1cb9457c21f42a23a9776a4d538c665" \
    "$PULSEVAD_MODEL"
download_file \
    "https://raw.githubusercontent.com/AydinAdnan/PulseVAD/af25e79d66830a3fee74541812721f6158fc92b5/pulsevad/data/pulsevad_2.1k_int8.onnx" \
    "416061347a1e723ed15163acd51006bf3c513b27bb9f57d85e2c694cc44b8389" \
    "$PULSEVAD_INT8_MODEL"

download_file \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
    "25b1ef1fea1acd210d63f8f24dc870ad6e077795ce1f54876252c6d3803c15af" \
    "$ONNXRUNTIME_ARCHIVE"

if [[ ! -e "$ONNXRUNTIME_LIBRARY" ]]; then
    tar -xzf "$ONNXRUNTIME_ARCHIVE" -C "$CACHE_DIR"
fi

cargo build --locked --manifest-path "$REPO_DIR/Cargo.toml"
readonly BINARY="$REPO_DIR/target/debug/extract-speech"

validate_result() {
    local output_path="$1"
    local metadata_path="$2"
    python3 - "$output_path" "$metadata_path" <<'PY'
import json
import sys
import wave

output_path, metadata_path = sys.argv[1:]
with wave.open(output_path, "rb") as audio:
    assert audio.getnchannels() == 1, "output is not mono"
    assert audio.getframerate() == 16_000, "unexpected output sample rate"
    assert audio.getnframes() > 0, "model produced no speech samples"

with open(metadata_path, encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)
assert metadata["intervals"], "metadata contains no output interval"
assert float(metadata["total_seconds"]) > 0.0, "metadata duration is zero"
assert float(metadata["compute_seconds"]) >= 0.0, "invalid compute duration"
PY
}

run_case() {
    local label="$1"
    local runtime="$2"
    local vad_model="$3"
    local model_path="$4"
    local audio_path="$5"
    local audio_name
    audio_name="$(basename "$audio_path" .wav)"
    local case_dir="$OUTPUT_DIR/${label}-${audio_name}"
    local output_path="$case_dir/speech.wav"
    local metadata_path="$case_dir/metadata.json"
    mkdir -p "$case_dir"

    local runtime_args=()
    if [[ "$runtime" == "onnxruntime" ]]; then
        runtime_args=(--runtime onnxruntime --dylib-path "$ONNXRUNTIME_LIBRARY")
    fi

    echo "Testing $label with $audio_name"
    RUST_LOG=warn "$BINARY" \
        "${runtime_args[@]}" \
        --vad-model "$vad_model" \
        --model-path "$model_path" \
        --process-audio "$audio_path" \
        --threshold 0.5 \
        --output-type concatenated \
        --output "$output_path" \
        --metadata "$metadata_path"
    validate_result "$output_path" "$metadata_path"
}

readonly TEST_AUDIO_FILES=(
    "$REPO_DIR/test-audios/test_16khz.wav"
    "$REPO_DIR/test-audios/test_16khz_stereo.wav"
    "$REPO_DIR/test-audios/test_24khz.wav"
)

for audio_file in "${TEST_AUDIO_FILES[@]}"; do
    run_case "silero-candle" "candle" "silero" "$SILERO_MODEL" "$audio_file"
    run_case "silero-onnxruntime" "onnxruntime" "silero" "$SILERO_MODEL" "$audio_file"
    run_case "pulsevad-candle-fp32" "candle" "pulsevad" "$PULSEVAD_MODEL" "$audio_file"
    run_case "pulsevad-onnxruntime-fp32" "onnxruntime" "pulsevad" "$PULSEVAD_MODEL" "$audio_file"
    run_case "pulsevad-onnxruntime-int8" "onnxruntime" "pulsevad" "$PULSEVAD_INT8_MODEL" "$audio_file"
    run_case "pyannote-onnxruntime" "onnxruntime" "pyannote" "$PYANNOTE_MODEL" "$audio_file"
done

echo "All 18 model integration cases passed."
