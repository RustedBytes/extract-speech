from os import PathLike
from pathlib import Path
from typing import Any

PathInput = str | bytes | PathLike[str] | PathLike[bytes]

SAMPLE_RATE: int
ONNX_RUNTIME_VERSION: str
__version__: str

class SpeechSegment:
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def duration_samples(self) -> int: ...
    @property
    def start_seconds(self) -> float: ...
    @property
    def end_seconds(self) -> float: ...
    @property
    def duration_seconds(self) -> float: ...

class Detector:
    SAMPLE_RATE: int

    def __init__(
        self,
        model_path: PathInput,
        *,
        model: str = "silero",
        runtime: str = "candle",
        onnx_runtime_path: PathInput | None = None,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float | None = None,
        debug: bool = False,
    ) -> None: ...
    @staticmethod
    def from_pretrained(
        model: str = "silero",
        *,
        runtime: str = "candle",
        quantized: bool = False,
        cache_dir: PathInput | None = None,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float | None = None,
        debug: bool = False,
    ) -> Detector: ...
    def detect(self, samples: Any) -> list[SpeechSegment]: ...
    @property
    def model(self) -> str: ...
    @property
    def runtime(self) -> str: ...

def download_model(
    model: str,
    *,
    quantized: bool = False,
    cache_dir: PathInput | None = None,
) -> Path: ...
def download_all_models(*, cache_dir: PathInput | None = None) -> list[tuple[str, Path]]: ...
def download_onnx_runtime(*, cache_dir: PathInput | None = None) -> Path: ...
def initialize_onnx_runtime(path: PathInput) -> None: ...
def default_cache_dir() -> Path: ...
