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

readonly SILERO_V6_MODEL="$CACHE_DIR/silero-vad-v6.onnx"
readonly SILERO_V5_MODEL="$CACHE_DIR/silero-vad-v5.onnx"
readonly PYANNOTE_MODEL="$CACHE_DIR/pyannote-segmentation-3.0.onnx"
readonly PULSEVAD_MODEL="$CACHE_DIR/pulsevad-2.1k.onnx"
readonly PULSEVAD_INT8_MODEL="$CACHE_DIR/pulsevad-2.1k-int8.onnx"
readonly FSMN_MODEL="$CACHE_DIR/model.onnx"
readonly FSMN_INT8_MODEL="$CACHE_DIR/model_quant.onnx"
readonly FSMN_CMVN="$CACHE_DIR/vad.mvn"
readonly TEN_MODEL="$CACHE_DIR/ten-vad.onnx"
readonly MARBLENET_MODEL="$CACHE_DIR/marblenet.onnx"
readonly MARBLENET_INT8_MODEL="$CACHE_DIR/marblenet_int8.onnx"
readonly ONNXRUNTIME_ARCHIVE="$CACHE_DIR/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
readonly ONNXRUNTIME_DIR="$CACHE_DIR/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"
readonly ONNXRUNTIME_LIBRARY="$ONNXRUNTIME_DIR/lib/libonnxruntime.so"

# Model downloads are revision-pinned and checksum-verified.
download_file \
    "https://raw.githubusercontent.com/snakers4/silero-vad/60b7ffa243625ebdc1070275a29f18c87843786a/src/silero_vad/data/silero_vad.onnx" \
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3" \
    "$SILERO_V6_MODEL"
download_file \
    "https://huggingface.co/onnx-community/silero-vad/resolve/ddc9a7e80d6758f6fc795a1e8a04b798eb929d3a/onnx/model.onnx" \
    "a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808" \
    "$SILERO_V5_MODEL"
download_file \
    "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/733a93b6473d019a773298e08cefa686894b1854/onnx/model.onnx" \
    "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25" \
    "$PYANNOTE_MODEL"
download_file \
    "https://huggingface.co/funasr/fsmn-vad-onnx/resolve/f6e9fbb4cefa7397216c763f21307993f147f585/model.onnx" \
    "756887ce01695a9bb00dd85ca0f743653de03b18ba54d2e9ef4f4bb9b3edbf9f" \
    "$FSMN_MODEL"
download_file \
    "https://huggingface.co/funasr/fsmn-vad-onnx/resolve/f6e9fbb4cefa7397216c763f21307993f147f585/model_quant.onnx" \
    "9b28837838fce9685503c63139fadbad35d6c8ed485485dafdbb32e725969660" \
    "$FSMN_INT8_MODEL"
download_file \
    "https://huggingface.co/funasr/fsmn-vad-onnx/resolve/f6e9fbb4cefa7397216c763f21307993f147f585/vad.mvn" \
    "6820fef9687708c4fc3fab2530179c8fcea6262daa25514380056cd8f6eb1754" \
    "$FSMN_CMVN"
download_file \
    "https://huggingface.co/TEN-framework/ten-vad/resolve/bda8ffc78b1846c5c7cbd38f04e52deff49de707/src/onnx_model/ten-vad.onnx" \
    "e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4" \
    "$TEN_MODEL"
download_file \
    "https://huggingface.co/TigreGotico/frame-vad-marblenet-onnx/resolve/e8786fe74e055954901eb553cc9c3145323981cc/marblenet.onnx" \
    "4ad3364be94d462b5fd4fa39910c24967dbb9dba436e27bcff7a88359515e491" \
    "$MARBLENET_MODEL"
download_file \
    "https://huggingface.co/TigreGotico/frame-vad-marblenet-onnx/resolve/e8786fe74e055954901eb553cc9c3145323981cc/marblenet_int8.onnx" \
    "9c4462323f9b576fd5e581d3c86b9b9b513468d18a79bcdcd3a2bcbcaab02699" \
    "$MARBLENET_INT8_MODEL"

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

cargo build --locked --features cli --manifest-path "$REPO_DIR/Cargo.toml"
cargo build --locked --features cli --example library-inference --manifest-path "$REPO_DIR/Cargo.toml"
readonly BINARY="$REPO_DIR/target/debug/extract-speech"
readonly LIBRARY_EXAMPLE="$REPO_DIR/target/debug/examples/library-inference"

validate_result() {
    local output_path="$1"
    local metadata_path="$2"
    local input_path="$3"
    python3 - "$output_path" "$metadata_path" "$input_path" <<'PY'
import json
import math
import sys
import wave

output_path, metadata_path, input_path = sys.argv[1:]
with wave.open(output_path, "rb") as audio:
    assert audio.getnchannels() == 1, "output is not mono"
    assert audio.getframerate() == 16_000, "unexpected output sample rate"
    assert audio.getnframes() > 0, "model produced no speech samples"
    output_seconds = audio.getnframes() / audio.getframerate()

with wave.open(input_path, "rb") as audio:
    input_seconds = audio.getnframes() / audio.getframerate()

with open(metadata_path, encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)
assert metadata["intervals"], "metadata contains no output interval"
metadata_seconds = float(metadata["total_seconds"])
interval_seconds = sum(float(interval["duration"]) for interval in metadata["intervals"])
assert 1.0 < output_seconds < input_seconds * 0.9, "implausible detected speech duration"
assert math.isclose(metadata_seconds, output_seconds, abs_tol=1 / 16_000), \
    "metadata duration does not match output audio"
assert math.isclose(interval_seconds, metadata_seconds, abs_tol=1e-6), \
    "metadata interval durations do not match the total"
assert math.isfinite(float(metadata["compute_seconds"])), "invalid compute duration"
assert float(metadata["compute_seconds"]) >= 0.0, "negative compute duration"
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
    validate_result "$output_path" "$metadata_path" "$audio_path"
}

readonly TEST_AUDIO_FILES=(
    "$REPO_DIR/test-audios/test_16khz.wav"
    "$REPO_DIR/test-audios/test_16khz_stereo.wav"
    "$REPO_DIR/test-audios/test_24khz.wav"
)

echo "Testing the public library API with Silero v6 and test_16khz"
"$LIBRARY_EXAMPLE" "$SILERO_V6_MODEL" "$REPO_DIR/test-audios/test_16khz.wav" >/dev/null

for audio_file in "${TEST_AUDIO_FILES[@]}"; do
    run_case "silero-v6-candle" "candle" "silero" "$SILERO_V6_MODEL" "$audio_file"
    run_case "silero-v6-onnxruntime" "onnxruntime" "silero" "$SILERO_V6_MODEL" "$audio_file"
    run_case "silero-v5-candle" "candle" "silero" "$SILERO_V5_MODEL" "$audio_file"
    run_case "silero-v5-onnxruntime" "onnxruntime" "silero" "$SILERO_V5_MODEL" "$audio_file"
    run_case "pulsevad-candle-fp32" "candle" "pulsevad" "$PULSEVAD_MODEL" "$audio_file"
    run_case "pulsevad-candle-int8" "candle" "pulsevad" "$PULSEVAD_INT8_MODEL" "$audio_file"
    run_case "pulsevad-onnxruntime-fp32" "onnxruntime" "pulsevad" "$PULSEVAD_MODEL" "$audio_file"
    run_case "pulsevad-onnxruntime-int8" "onnxruntime" "pulsevad" "$PULSEVAD_INT8_MODEL" "$audio_file"
    run_case "pyannote-candle" "candle" "pyannote" "$PYANNOTE_MODEL" "$audio_file"
    run_case "pyannote-onnxruntime" "onnxruntime" "pyannote" "$PYANNOTE_MODEL" "$audio_file"
    run_case "fsmn-candle-fp32" "candle" "fsmn" "$FSMN_MODEL" "$audio_file"
    run_case "fsmn-candle-int8" "candle" "fsmn" "$FSMN_INT8_MODEL" "$audio_file"
    run_case "fsmn-onnxruntime-fp32" "onnxruntime" "fsmn" "$FSMN_MODEL" "$audio_file"
    run_case "fsmn-onnxruntime-int8" "onnxruntime" "fsmn" "$FSMN_INT8_MODEL" "$audio_file"
    run_case "ten-candle" "candle" "ten" "$TEN_MODEL" "$audio_file"
    run_case "ten-onnxruntime" "onnxruntime" "ten" "$TEN_MODEL" "$audio_file"
    run_case "marblenet-candle-fp32" "candle" "marblenet" "$MARBLENET_MODEL" "$audio_file"
    run_case "marblenet-onnxruntime-fp32" "onnxruntime" "marblenet" "$MARBLENET_MODEL" "$audio_file"
    run_case "marblenet-onnxruntime-int8" "onnxruntime" "marblenet" "$MARBLENET_INT8_MODEL" "$audio_file"
done

echo "The library API smoke test and all 57 CLI model integration cases passed."
