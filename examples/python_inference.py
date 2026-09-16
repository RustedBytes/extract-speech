from __future__ import annotations

import array
import sys
import wave

import extract_speech


def read_mono_16khz_wav(path: str) -> array.array[float]:
    with wave.open(path, "rb") as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != extract_speech.SAMPLE_RATE:
            raise ValueError("the example expects a mono 16 kHz WAV file")
        if audio.getsampwidth() != 2:
            raise ValueError("the example expects signed 16-bit PCM")
        pcm = array.array("h", audio.readframes(audio.getnframes()))

    if sys.byteorder != "little":
        pcm.byteswap()
    return array.array("f", (sample / 32768.0 for sample in pcm))


detector = extract_speech.Detector.from_pretrained(
    "silero",
    runtime="candle",
    threshold=0.7,
)

for segment in detector.detect(read_mono_16khz_wav(sys.argv[1])):
    print(f"speech: {segment.start_seconds:.3f}s..{segment.end_seconds:.3f}s")
