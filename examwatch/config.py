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
    vision_url: str = ""              # remote examwatch.vision_service (pose, blendshapes, phones)
    vision_token: str = ""
    lite: bool = False                # small hosts (e.g. 512 MB): no PyTorch models, up to 3 people
    phone_enabled: bool = True
    phone_interval: float = 0.5       # seconds between phone-detector passes (boxes reused in between)
    speak_threshold: float = 2.5      # mean mouth-motion needed to attribute speech to a person
    identity_budget: int = 4          # face embeddings per frame when many people are visible
    emotion_budget: int = 6
    whisper_name: str = "zipformer-vi-30m"
    # Fine-grained proctor settings (feedback 2026-11-08). sensitivity still sets
    # the base thresholds above; values in Settings.advanced override them.
    head_seconds: float = 1.25        # yaw beyond head_yaw for this long = turning
    lean_hold_seconds: float = 2.0
    standing_rise: float = 0.5        # head rise needed for 'standing', in face heights
    stand_seconds: float = 1.5        # 'standing' held this long (head_down/moving keep standing_seconds)
    auto_calibrate: bool = True       # take the seated reference when someone holds still ~3 s
    phone_confirm_seconds: float = 1.0
    alert_cooldown: float = 6.0
    min_face_ratio: float = 0.0       # ignore faces smaller than this x the largest one (0 = off)
    advanced: dict = None             # explicit overrides saved by the proctor
    enabled: dict = None              # alert kind -> on/off
    warmup: bool = True

    def rule_on(self, kind):
        return (self.enabled or {}).get(kind, True)

    @classmethod
    def load(cls):
        LITE = os.getenv("EXAMWATCH_LITE", "").strip() == "1"
        return cls(
            data=Path(os.getenv("EXAMWATCH_DATA", str(ROOT / "data"))).resolve(),
            models=Path(os.getenv("EXAMWATCH_MODELS", str(ROOT / "models"))).resolve(),
            threads=max(1, min(8, int(os.getenv("EXAMWATCH_THREADS", "2")))),
            whisper_name=os.getenv("EXAMWATCH_WHISPER", "zipformer-vi-30m"),
            lite=LITE,
            vision_url=os.getenv("EXAMWATCH_VISION_URL", "").strip(),
            vision_token=os.getenv("EXAMWATCH_VISION_TOKEN", "").strip(),
            phone_enabled=os.getenv("EXAMWATCH_PHONE", "0" if LITE else "1").strip() == "1",
            max_people=max(1, min(40, int(os.getenv("EXAMWATCH_MAX_PEOPLE", "3" if LITE else "25")))),
            frame_width=max(640, min(1920, int(os.getenv("EXAMWATCH_FRAME_WIDTH", "960" if LITE else "1280")))),
            detector_width=max(320, min(1920, int(os.getenv("EXAMWATCH_DETECTOR_WIDTH", "640" if LITE else "960")))),
            pose_engine=os.getenv("EXAMWATCH_POSE_ENGINE", "mediapipe" if LITE else "auto").strip().lower(),
            yolo_imgsz=max(320, min(1280, int(os.getenv("EXAMWATCH_POSE_IMGSZ", "800")))),
            phone_interval=max(0.0, min(5.0, float(os.getenv("EXAMWATCH_PHONE_INTERVAL", "0.5")))),
            warmup=os.getenv("EXAMWATCH_WARMUP", "1").strip() != "0",
        )


# Proctor-adjustable values: key -> (min, max, default, group, label, help).
# The settings page renders its form from this table (GET /api/settings/detection).
ADVANCED = {
    "phone_confidence":      (.20, .90, None, "phone", "Độ tin cậy tối thiểu",
                              "Cao hơn: ít báo nhầm (thước, máy tính cầm tay) nhưng dễ bỏ sót."),
    "phone_confirm_seconds": (0, 6, 1.0, "phone", "Phải thấy liên tục (giây)",
                              "Điện thoại phải xuất hiện liên tục chừng này mới cảnh báo."),
    "head_yaw":              (.15, .80, None, "head", "Độ lệch mặt",
                              "Mũi lệch khỏi giữa hai mắt/tai. Nhỏ hơn = nhạy hơn."),
    "head_seconds":          (.5, 10, 1.25, "head", "Giữ quay mặt (giây)", ""),
    "lean_threshold":        (8, 45, None, "body", "Góc nghiêng thân (độ)", ""),
    "lean_hold_seconds":     (.5, 10, 2.0, "body", "Giữ nghiêng (giây)", ""),
    "arm_threshold":         (20, 120, None, "body", "Góc tay so với thân (độ)", ""),
    "standing_rise":         (.2, 1.5, .5, "standing", "Độ nâng đầu (× chiều cao mặt)",
                              "0,5 = đầu nâng lên nửa khuôn mặt so với lúc ngồi."),
    "stand_seconds":         (.5, 60, 1.5, "standing", "Giữ nhổm dậy (giây)", ""),
    "negative_seconds":      (5, 300, 20.0, "other", "Cảm xúc tiêu cực kéo dài (giây)", ""),
    "alert_cooldown":        (2, 120, 6.0, "other", "Giãn cách 2 cảnh báo cùng loại (giây)", ""),
    "min_face_ratio":        (0, .9, 0.0, "other", "Bỏ qua mặt nhỏ (× mặt lớn nhất)",
                              "0 = tắt. Ví dụ 0,6: không cảnh báo người ở xa phía sau có mặt nhỏ hơn "
                              "60 % thí sinh gần camera nhất. Để 0 khi camera quay cả phòng nhiều hàng ghế."),
}
# None default = taken from the sensitivity slider.

ALERT_KINDS = {
    "phone": "Điện thoại", "head_turning": "Quay mặt / nhìn bài", "body_lean": "Nghiêng thân",
    "arm_raise": "Giơ / vươn tay", "standing": "Nhổm dậy", "head_down": "Cúi xuống",
    "moving": "Rời chỗ", "negative_emotion": "Cảm xúc tiêu cực", "tired": "Mệt mỏi",
    "proctor_dwell": "Giám thị đứng lâu cạnh bàn",
}

ADVANCED_GROUPS = {"phone": "Điện thoại", "head": "Quay mặt / nhìn bài", "body": "Nghiêng người & tay",
                   "standing": "Nhổm dậy", "other": "Khác"}


def validate_advanced(values):
    clean = {}
    for key, value in (values or {}).items():
        if key not in ADVANCED:
            raise ValueError(f"Không có thông số '{key}'.")
        low, high, _, _, label, _ = ADVANCED[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}: cần là số.")
        if not low <= float(value) <= high:
            raise ValueError(f"{label}: phải từ {low:g} đến {high:g}.")
        clean[key] = float(value)
    return clean


def validate_enabled(values):
    clean = {}
    for key, value in (values or {}).items():
        if key not in ALERT_KINDS:
            raise ValueError(f"Không có loại cảnh báo '{key}'.")
        if not isinstance(value, bool):
            raise ValueError(f"{ALERT_KINDS[key]}: cần bật/tắt.")
        clean[key] = value
    return clean
