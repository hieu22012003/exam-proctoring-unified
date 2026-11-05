from pathlib import Path
import os
import logging
import queue
import threading
import time
import uuid
import cv2
import numpy as np
from .store import Store
from .clips import EvidenceClips
from .rules import body_angles

logger = logging.getLogger(__name__)


class Runtime:
    def __init__(self, settings, vision_factory=None, speech_factory=None):
        self.settings = settings
        self.store = Store(settings.data)
        self.clips = EvidenceClips(settings.data / 'evidence')
        saved_seconds=self.store.setting('standing_seconds',settings.standing_seconds)
        if type(saved_seconds) is int and 1<=saved_seconds<=300:
            self.settings.standing_seconds=saved_seconds
        self.vision_factory, self.speech_factory = vision_factory, speech_factory
        self.vision = self.speech = None
        self.session = None
        self.owner = None
        self.started = 0
        self.lock = threading.RLock()
        self.audio = queue.Queue(maxsize=6)
        self.shutdown = threading.Event()
        self.worker = None
        self.audio_busy = False
        self.audio_error = None
        self.audio_rejected = 0
        self.inference_ms = 0
        self.last_frame = 0
        self.visible = []
        self.capture_id = None
        self.last_audio_end = 0
        self.gallery = []

    def model_status(self):
        paths = {"face_detector": "yunet.onnx", "identity": "sface.onnx",
                 "emotion": "emotion.onnx", "pose": "pose_lite.task",
                 "phone_detector": "phone_detector_v5.pt",
                 "voice": f"whisper-{self.settings.whisper_name}/model.bin"}
        status = {k: (self.settings.models / v).is_file() for k, v in paths.items()}
        if self.settings.whisper_name in {'zipformer-vi-30m','zipformer-vi-30m-lora','zipformer-vi-30m-lora-low-lr20'}:
            status['voice'] = all((self.settings.models / self.settings.whisper_name / name).is_file()
                                  for name in ['encoder.int8.onnx','decoder.onnx','joiner.int8.onnx','tokens.txt'])
        return status

    def _download_phone_model(self):
        phone_path = self.settings.models / "phone_detector_v5.pt"
        gcs_path = os.getenv("EXAMWATCH_PHONE_MODEL_GCS_PATH", "").strip()
        if not phone_path.is_file() and gcs_path:
            self.store.cloud.download_model(gcs_path, phone_path)

    def configure_standing(self, seconds):
        if type(seconds) is not int or not 1<=seconds<=300:
            raise ValueError('Số giây cảnh báo phải là số nguyên từ 1 đến 300.')
        with self.lock:
            if seconds != self.settings.standing_seconds:
                self.store.set_setting('standing_seconds',seconds)
                self.settings.standing_seconds=seconds
                targets=list(self.visible)
                if self.vision and hasattr(self.vision,'tracker'):
                    targets+=list(self.vision.tracker.tracks.values())
                for track in {id(t):t for t in targets}.values():
                    track.standing_rule.seconds=seconds
                    track.standing_rule.reset()
            return {'standing_threshold_seconds':self.settings.standing_seconds}

    def load_vision(self):
        if self.vision is None:
            self._download_phone_model()
            if self.vision_factory:
                self.vision = self.vision_factory(self.settings)
            else:
                if not all(v for k, v in self.model_status().items() if k != "voice"):
                    missing = [name for name, ready in self.model_status().items()
                               if name != "voice" and not ready]
                    raise ValueError("Thiếu model hình ảnh: " + ", ".join(missing) + ".")
                from .vision import Vision
                self.vision = Vision(self.settings)
        return self.vision

    def start(self, title, owner):
        with self.lock:
            if self.session:
                raise ValueError("Đang có phiên thi hoạt động. Hãy kết thúc phiên hiện tại trước.")
            self._download_phone_model()
            if not self.vision_factory and not all(self.model_status().values()):
                raise ValueError("Thiếu model. Chạy .\\prepare.ps1 trước khi bắt đầu phiên offline.")
            self.load_vision()
            if self.speech is None:
                if self.speech_factory:
                    self.speech = self.speech_factory(self.settings)
                else:
                    from .speech import Speech
                    self.speech = Speech(self.settings)
            if self.worker is None:
                self.worker = threading.Thread(target=self._audio_loop, name="examwatch-asr", daemon=True)
                self.worker.start()
            self.vision.reset()
            self.clips.reset()
            # Enrollment is immutable during a session: avoid SQLite and JSON
            # parsing on every camera frame.
            self.gallery = self.store.students(with_embeddings=True)
            self.session = self.store.new_session(title)
            self.owner, self.started = owner, time.monotonic()
            self.capture_id = uuid.uuid4().hex
            self.last_audio_end = 0
            self.visible = []
            self.audio_error = None
            self.last_frame = 0
            return {**self.session, "capture_id": self.capture_id}

    def require_owner(self, owner, capture_id=None):
        if not self.session:
            raise ValueError("Chưa có phiên thi hoạt động.")
        if owner != self.owner or (capture_id is not None and capture_id != self.capture_id):
            raise PermissionError("Phiên thu đang được điều khiển bởi cửa sổ khác hoặc đã thay đổi.")

    def stop(self, session_id):
        with self.lock:
            previous = self.store.session(session_id)
            if previous and previous["ended_at"]:
                return previous  # A repeated stop never affects a newer session.
            if not self.session or self.session["id"] != session_id:
                raise PermissionError("Phiên đang hiển thị đã thay đổi. Hãy tải lại dữ liệu phiên.")
            self.store.end_session(session_id)
            self.clips.reset()
            row = self.store.session(session_id)
            self.session, self.owner, self.capture_id = None, None, None
            self.visible = []
            self.vision.reset()
            return row

    def calibrate(self, owner):
        with self.lock:
            self.require_owner(owner)
            if time.monotonic() - self.last_frame > 2:
                raise ValueError("Camera chưa có dữ liệu mới để hiệu chuẩn.")
            count = 0
            for track in self.visible:
                if track.signal:
                    track.baseline = track.signal.copy()
                    if track.angles.get('lean_origin'):
                        track.baseline['abdomen_anchor'] = track.angles['lean_origin'][:]
                    track.baseline_owner = track.student_id
                    track.standing_rule.reset()
                    track.posture = "seated"
                    count += 1
            if not count:
                raise ValueError("Chưa thấy rõ đầu và hai vai. Hãy chỉnh camera rồi thử lại.")
            return count

    def process(self, frame, owner, capture_id):
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Đang xử lý ảnh trước.")
        try:
            self.require_owner(owner, capture_id)
            started = time.monotonic()
            tracks, overflow = self.vision.process(frame, self.gallery)
            now = time.monotonic()
            offset = now - self.started
            self.clips.push(frame, now)
            for track in tracks:
                events = []
                if track.has_phone and now - track.phone_alert_at >= 6:
                    track.phone_alert_at = now
                    events.append(("phone", f"Phát hiện điện thoại gần Track {track.id}; cần giám thị kiểm tra."))
                if track.head_turning and now - track.head_alert_at >= 6:
                    track.head_alert_at = now
                    events.append(("head_turning", "Hướng mặt lệch liên tục ít nhất 1,25 giây; cần giám thị kiểm tra."))
                h, w = frame.shape[:2]
                anchor = track.baseline.get('abdomen_anchor') if track.baseline else None
                angles = body_angles(track.landmarks, w, h, anchor) if anchor else body_angles(track.landmarks, w, h)
                if track.angles.get('lean_source') != angles['lean_source']:
                    # Changing geometry reference must not look like angle growth.
                    track.left_arm_rule.reset()
                    track.right_arm_rule.reset()
                    track.lean_rule.reset()
                track.angles = angles
                for key, rule in [('left_arm',track.left_arm_rule),('right_arm',track.right_arm_rule),('lean',track.lean_rule)]:
                    if rule.update(angles[key], now):
                        label = {'left_arm':'Tay trái','right_arm':'Tay phải','lean':'Nghiêng thân'}[key]
                        kind = 'body_lean' if key=='lean' else 'arm_raise'
                        events.append((kind, f'{label}: góc {rule.value:.1f}°; dấu hiệu tư thế cần giám thị kiểm tra.'))
                if track.standing_rule.update(track.posture == "standing", now):
                    events.append(("standing", f"Dấu hiệu nhổm dậy kéo dài ít nhất {self.settings.standing_seconds:g} giây; cần giám thị kiểm tra."))
                negative = (track.emotion in {"sad", "angry", "fear"}
                            and track.emotion_score >= self.settings.emotion_confidence)
                if track.emotion_rule.update(negative, now):
                    events.append(("negative_emotion", "Biểu cảm sad/angry/fear kéo dài ít nhất 20 giây; kiểm tra tình trạng thí sinh."))
                for kind, detail in events:
                    if kind in {'standing','arm_raise','body_lean','phone','head_turning'}:
                        filename = uuid.uuid4().hex + '.webm'
                        self.store.add_alert(self.session['id'], track, kind, detail, offset, filename)
                        self.clips.trigger(filename, now, f'Track {track.id} | {kind} | session +{offset:.1f}s')
                        continue
                    filename = uuid.uuid4().hex + ".jpg"
                    evidence = frame.copy()
                    h, w = frame.shape[:2]
                    x, y, bw, bh = track.box
                    cv2.rectangle(evidence, (int(x * w), int(y * h)),
                                  (int((x + bw) * w), int((y + bh) * h)), (50, 80, 255), 2)
                    cv2.putText(evidence, f"Track {track.id} | {kind} | +{offset:.1f}s", (12, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 80, 255), 2)
                    encoded, jpeg = cv2.imencode('.jpg', evidence)
                    if not encoded:
                        raise OSError("Không lưu được ảnh bằng chứng.")
                    (self.settings.data / "evidence" / filename).write_bytes(jpeg.tobytes())
                    self.store.add_alert(self.session["id"], track, kind, detail, offset, filename)
                if now - track.observed_at >= 1.0:
                    self.store.observe(self.session["id"], track, offset)
                    track.observed_at = now
            self.visible, self.last_frame = tracks, now
            self.inference_ms = round((now - started) * 1000)
            return {"tracks": [self.serialize(t, now) for t in tracks], "overflow": overflow,
                    "inference_ms": self.inference_ms, "offset_seconds": offset}
        finally:
            self.lock.release()

    @staticmethod
    def serialize(t, now):
        return {"id": t.id, "bbox": t.box, "student_id": t.student_id, "name": t.name,
                "identity_score": round(t.identity_score, 3), "emotion": t.emotion,
                "emotion_score": round(t.emotion_score, 3), "posture": t.posture,
                "has_phone": t.has_phone, "phone_confidence": round(t.phone_confidence, 3),
                "head_turning": t.head_turning, "head_yaw": round(t.head_yaw, 3),
                "calibrated": t.baseline is not None, "landmarks": t.landmarks,
                "mouth_points": t.mouth_points,
                "pose_angles": {key: round(rule.value,1) if rule.value is not None else None
                    for key,rule in [('left_arm',t.left_arm_rule),('right_arm',t.right_arm_rule),('lean',t.lean_rule)]},
                "lean_source": t.angles.get('lean_source'),
                "lean_origin": t.angles.get('lean_origin'),
                "lean_target": t.angles.get('lean_target'),
                "pose_flags": [key for key,rule in [('left_arm',t.left_arm_rule),('right_arm',t.right_arm_rule),('lean',t.lean_rule)] if rule.flagged],
                "standing_seconds": round(t.standing_rule.elapsed(now), 1),
                "negative_seconds": round(t.emotion_rule.elapsed(now), 1)}

    def queue_audio(self, samples, rate, offset, owner, capture_id):
        with self.lock:
            self.require_owner(owner, capture_id)
            duration = len(samples) / rate
            if offset < self.last_audio_end - 0.05:
                raise ValueError("Đoạn âm thanh bị trùng hoặc sai thứ tự.")
            if offset + duration > time.monotonic() - self.started + 3:
                raise ValueError("Mốc thời gian âm thanh vượt quá thời gian phiên thi.")
            try:
                self.audio.put_nowait((self.session["id"], samples, rate, offset))
            except queue.Full:
                self.audio_rejected += 1
                raise BlockingIOError("Hàng đợi chép lời đầy. Hệ thống không nhận đoạn này; hãy giảm tải hoặc tạm dừng thu.")
            self.last_audio_end = offset + duration

    def _audio_loop(self):
        while not self.shutdown.is_set() or not self.audio.empty():
            try:
                session_id, samples, rate, offset = self.audio.get(timeout=0.3)
            except queue.Empty:
                continue
            self.audio_busy = True
            try:
                for segment in self.speech.transcribe(samples, rate):
                    self.store.add_transcript(session_id, offset + segment["start"],
                                              offset + segment["end"], segment["text"])
                self.audio_error = None
            except Exception as exc:
                logger.exception("Audio transcription failed")
                self.audio_error = str(exc)
            finally:
                self.audio_busy = False
                self.audio.task_done()

    def status(self):
        return {"vision_revision": "adaptive-parallel-20261103", "pose_revision": "abdomen-20-20261103", "models": self.model_status(), "session": self.session,
                "owner": self.owner, "capture_id": self.capture_id,
                "audio_pending": self.audio.qsize(), "audio_busy": self.audio_busy,
                "audio_error": self.audio_error, "audio_rejected": self.audio_rejected,
                "voice_model": self.settings.whisper_name,
                "standing_threshold_seconds": self.settings.standing_seconds,
                "audio_quality_warning": getattr(self.speech, "last_warning", None),
                "firebase_sync": self.store.cloud.status(),
                "inference_ms": self.inference_ms,
                "camera_fresh": bool(self.session and time.monotonic() - self.last_frame < 3),
                "max_people": self.settings.max_people}

    def close(self):
        self.shutdown.set()
        self.clips.close()
        if self.worker:
            self.worker.join(timeout=20)
        with self.lock:
            if self.session:
                self.store.end_session(self.session["id"])
            if self.vision:
                self.vision.close()
        self.store.cloud.close()
