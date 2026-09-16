from __future__ import annotations

import array
import math
import os
import platform
import sys
import unittest
import wave
from pathlib import Path
from typing import ClassVar

import extract_speech

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPOSITORY_ROOT / "test-audios" / "test_16khz.wav"
CACHE_PATH = Path(
    os.environ.get(
        "EXTRACT_SPEECH_TEST_CACHE_DIR",
        REPOSITORY_ROOT / "target" / "python-test-cache",
    )
)
SUPPORTED_AUTOMATIC_ONNX_HOSTS = {
    ("linux", "aarch64"),
    ("linux", "x86_64"),
    ("darwin", "aarch64"),
    ("darwin", "arm64"),
    ("windows", "amd64"),
    ("windows", "arm64"),
}
SUPPORTS_AUTOMATIC_ONNX_RUNTIME = (
    platform.system().lower(),
    platform.machine().lower(),
) in SUPPORTED_AUTOMATIC_ONNX_HOSTS


def read_fixture() -> array.array[float]:
    with wave.open(str(FIXTURE_PATH), "rb") as audio:
        if audio.getnchannels() != 1:
            raise ValueError("test fixture must be mono")
        if audio.getframerate() != extract_speech.SAMPLE_RATE:
            raise ValueError("test fixture must be sampled at 16 kHz")
        if audio.getsampwidth() != 2:
            raise ValueError("test fixture must contain signed 16-bit PCM")
        pcm = array.array("h", audio.readframes(audio.getnframes()))

    if sys.byteorder != "little":
        pcm.byteswap()
    return array.array("f", (sample / 32768.0 for sample in pcm))


class PythonBindingsTest(unittest.TestCase):
    detector: ClassVar[extract_speech.Detector]
    onnx_detector: ClassVar[extract_speech.Detector]
    samples: ClassVar[array.array[float]]

    @classmethod
    def setUpClass(cls) -> None:
        model_path = extract_speech.download_model("silero", cache_dir=CACHE_PATH)
        cls.detector = extract_speech.Detector.from_pretrained(
            "silero",
            runtime="candle",
            cache_dir=CACHE_PATH,
        )
        cls.samples = read_fixture()
        if SUPPORTS_AUTOMATIC_ONNX_RUNTIME:
            cls.onnx_detector = extract_speech.Detector.from_pretrained(
                "silero",
                runtime="onnxruntime",
                cache_dir=CACHE_PATH,
            )

        if not model_path.is_file():
            raise AssertionError("download_model did not return a model file")

    def test_module_metadata(self) -> None:
        self.assertEqual(extract_speech.SAMPLE_RATE, 16_000)
        self.assertEqual(extract_speech.Detector.SAMPLE_RATE, 16_000)
        self.assertEqual(extract_speech.__version__, "0.7.2")
        self.assertEqual(self.detector.model, "silero")
        self.assertEqual(self.detector.runtime, "candle")

    def test_detects_speech_from_float32_buffer(self) -> None:
        segments = self.detector.detect(self.samples)

        self.assertGreater(len(segments), 0)
        for segment in segments:
            self.assertGreaterEqual(segment.start, 0)
            self.assertGreater(segment.end, segment.start)
            self.assertLessEqual(segment.end, len(self.samples))
            self.assertEqual(segment.duration_samples, segment.end - segment.start)
            self.assertTrue(math.isfinite(segment.duration_seconds))

    @unittest.skipUnless(
        SUPPORTS_AUTOMATIC_ONNX_RUNTIME,
        "automatic ONNX Runtime is unavailable for this host",
    )
    def test_automatic_onnx_runtime_bundle(self) -> None:
        segments = self.onnx_detector.detect(self.samples)

        self.assertGreater(len(segments), 0)
        self.assertEqual(self.onnx_detector.runtime, "onnxruntime")

    def test_accepts_python_sequences(self) -> None:
        self.assertEqual(self.detector.detect([0.0] * extract_speech.SAMPLE_RATE), [])

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            extract_speech.download_model("unknown", cache_dir=CACHE_PATH)
        with self.assertRaises(ValueError):
            self.detector.detect([math.nan])
        with self.assertRaises(ValueError):
            self.detector.detect([1.01])
        with self.assertRaises(TypeError):
            self.detector.detect(object())


if __name__ == "__main__":
    unittest.main()
