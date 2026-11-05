from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    data: Path = ROOT / "data"
    models: Path = ROOT / "models"
    max_people: int = 3
    threads: int = 2
    frame_width: int = 960
    detector_width: int = 512
    pose_width: int = 512
    identity_interval: float = 2.0
    emotion_interval: float = 0.8
    face_threshold: float = 0.45
    identity_margin: float = 0.06
    negative_seconds: float = 20.0
    standing_seconds: float = 5.0
    emotion_confidence: float = 0.45
    phone_confidence: float = 0.35
    whisper_name: str = "zipformer-vi-30m"

    @classmethod
    def load(cls):
        return cls(
            data=Path(os.getenv("EXAMWATCH_DATA", str(ROOT / "data"))).resolve(),
            models=Path(os.getenv("EXAMWATCH_MODELS", str(ROOT / "models"))).resolve(),
            threads=max(1, min(8, int(os.getenv("EXAMWATCH_THREADS", "2")))),
            whisper_name=os.getenv("EXAMWATCH_WHISPER", "zipformer-vi-30m"),
        )
