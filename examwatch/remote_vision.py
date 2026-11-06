"""Client for examwatch.vision_service (pose, face blendshapes and phones on another host)."""
from __future__ import annotations

import logging
import time
import types

import cv2

log = logging.getLogger(__name__)


class RemoteVision:
    def __init__(self, url, token, timeout=4.0, width=960):
        import requests

        self.url = url.rstrip("/") + "/analyze"
        self.session = requests.Session()
        self.session.headers.update({"X-Vision-Token": token, "Content-Type": "image/jpeg"})
        self.timeout, self.width = timeout, width
        self.error, self.failed_at = None, 0.0

    def analyze(self, frame, phone=False, mesh=False, conf=.35):
        """Remote analysis, or None when the vision service is unreachable."""
        if self.failed_at and time.monotonic() - self.failed_at < 5:
            return None  # back off briefly instead of stalling every frame
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, round(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        try:
            response = self.session.post(self.url, data=jpeg.tobytes(), timeout=self.timeout,
                                         params={"pose": 1, "mesh": int(mesh), "phone": int(phone), "conf": conf})
            response.raise_for_status()
            self.error, self.failed_at = None, 0.0
            return response.json()
        except Exception as exc:
            if self.error is None:
                log.warning("Vision service unavailable: %s", exc)
            self.error, self.failed_at = str(exc), time.monotonic()
            return None

    @staticmethod
    def blendshapes(face):
        return [types.SimpleNamespace(category_name=k, score=v) for k, v in face.get("blendshapes", {}).items()]

    def close(self):
        self.session.close()
