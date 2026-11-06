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
    # Derived from the 0-100 sensitivity setting (50 keeps the values above).
    sensitivity: int = 50
    head_yaw: float = 0.30
    arm_threshold: float = 45.000001
    lean_threshold: float = 20.0
    evidence_image: bool = True
    evidence_video: bool = True
    evidence_audio: bool = True
    proctor_dwell_seconds: float = 60.0
    pose_engine: str = "auto"         # auto: YOLO pose for more than 3 people, else MediaPipe
    yolo_imgsz: int = 800
    phone_interval: float = 0.5       # seconds between phone-detector passes (boxes reused in between)
    speak_threshold: float = 2.5      # mean mouth-motion needed to attribute speech to a person
    identity_budget: int = 4          # face embeddings per frame when many people are visible
    emotion_budget: int = 6
    whisper_name: str = "zipformer-vi-30m"

    @classmethod
    def load(cls):
        return cls(
            data=Path(os.getenv("EXAMWATCH_DATA", str(ROOT / "data"))).resolve(),
            models=Path(os.getenv("EXAMWATCH_MODELS", str(ROOT / "models"))).resolve(),
            threads=max(1, min(8, int(os.getenv("EXAMWATCH_THREADS", "2")))),
            whisper_name=os.getenv("EXAMWATCH_WHISPER", "zipformer-vi-30m"),
            max_people=max(1, min(40, int(os.getenv("EXAMWATCH_MAX_PEOPLE", "25")))),
            frame_width=max(640, min(1920, int(os.getenv("EXAMWATCH_FRAME_WIDTH", "1280")))),
            detector_width=max(320, min(1920, int(os.getenv("EXAMWATCH_DETECTOR_WIDTH", "960")))),
            pose_engine=os.getenv("EXAMWATCH_POSE_ENGINE", "auto").strip().lower(),
            yolo_imgsz=max(320, min(1280, int(os.getenv("EXAMWATCH_POSE_IMGSZ", "800")))),
            phone_interval=max(0.0, min(5.0, float(os.getenv("EXAMWATCH_PHONE_INTERVAL", "0.5")))),
        )
