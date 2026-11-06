"""Stand-alone vision worker: body pose, face blendshapes and phone detection.

Lets a small host (e.g. two Render free instances, 512 MB each) split the models:
the main API keeps faces, emotions and speech; this service answers one JPEG frame
at a time. Protected by a shared token in the X-Vision-Token header.

    uvicorn examwatch.vision_service:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import hmac
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request

MODELS = Path(os.getenv("EXAMWATCH_MODELS", "/opt/models"))
TOKEN = os.getenv("EXAMWATCH_VISION_TOKEN", "").strip()
MAX_PEOPLE = max(1, min(10, int(os.getenv("EXAMWATCH_MAX_PEOPLE", "3"))))
MAX_BYTES = 4_000_000

app = FastAPI(title="ExamWatch vision", docs_url=None, redoc_url=None, openapi_url=None)
_lock = threading.Lock()
_models = {}


def models():
    """Load lazily so the health check answers while models load on first use."""
    with _lock:
        if not _models:
            import mediapipe as mp
            from .onnx_detect import OnnxYolo
            vision = mp.tasks.vision
            _models["mp"] = mp
            _models["pose"] = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=(MODELS / "pose_lite.task").read_bytes()),
                running_mode=vision.RunningMode.IMAGE, num_poses=MAX_PEOPLE,
                min_pose_detection_confidence=.5, min_pose_presence_confidence=.5, output_segmentation_masks=False))
            mesh = MODELS / "face_landmarker.task"
            _models["mesh"] = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=mesh.read_bytes()),
                running_mode=vision.RunningMode.IMAGE, num_faces=MAX_PEOPLE,
                output_face_blendshapes=True, min_face_detection_confidence=.4)) if mesh.is_file() else None
            phone = MODELS / "phone_detector_v5.onnx"
            _models["phone"] = OnnxYolo(phone) if phone.is_file() else None
        return _models


@app.get("/healthz")
def healthz():
    return {"ok": True, "loaded": bool(_models)}


@app.post("/analyze")
async def analyze(request: Request, pose: int = Query(1, ge=0, le=1), mesh: int = Query(0, ge=0, le=1),
                  phone: int = Query(0, ge=0, le=1), conf: float = Query(.35, ge=.05, le=.95)):
    if not TOKEN or not hmac.compare_digest(request.headers.get("x-vision-token", ""), TOKEN):
        raise HTTPException(401, "invalid token")
    body = await request.body()
    if not body or len(body) > MAX_BYTES:
        raise HTTPException(413, "frame too large")
    frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "not a JPEG frame")
    return analyze_frame(frame, bool(pose), bool(mesh), bool(phone), conf)


def analyze_frame(frame, want_pose=True, want_mesh=False, want_phone=False, conf=.35):
    m = models()
    h, w = frame.shape[:2]
    out = {"poses": [], "faces": [], "phones": None}
    with _lock:  # MediaPipe task objects are not thread-safe
        image = None
        if want_pose or (want_mesh and m["mesh"] is not None):
            image = m["mp"].Image(image_format=m["mp"].ImageFormat.SRGB,
                                  data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if want_pose:
            result = m["pose"].detect(image)
            out["poses"] = [[[round(float(l.x), 5), round(float(l.y), 5), round(float(l.z), 4), round(float(l.visibility), 3)]
                             for l in person] for person in result.pose_landmarks]
        if want_mesh and m["mesh"] is not None:
            result = m["mesh"].detect(image)
            for points, shapes in zip(result.face_landmarks, result.face_blendshapes or []):
                xs, ys = [p.x for p in points], [p.y for p in points]
                out["faces"].append({"cx": (min(xs) + max(xs)) / 2, "cy": (min(ys) + max(ys)) / 2,
                                     "w": max(xs) - min(xs), "h": max(ys) - min(ys),
                                     "blendshapes": {c.category_name: round(float(c.score), 3) for c in shapes
                                                     if c.category_name in {"eyeBlinkLeft", "eyeBlinkRight", "jawOpen"}}})
    if want_phone and m["phone"] is not None:
        out["phones"] = [[(x1 + x2) / (2 * w), (y1 + y2) / (2 * h), c]
                         for x1, y1, x2, y2, c, _ in m["phone"].predict(frame, conf=conf)]
    return out
