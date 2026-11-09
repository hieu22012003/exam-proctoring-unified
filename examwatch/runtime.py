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
from .rules import body_angles, Sustained, sensitivity_thresholds
from .tracking import apply_thresholds
from .config import ADVANCED, ADVANCED_GROUPS, ALERT_KINDS, validate_advanced, validate_enabled
from .rules import FACING_SECONDS


def side(sign):
    return 'trái' if sign < 0 else 'phải'

logger = logging.getLogger(__name__)


class Runtime:
    def __init__(self, settings, vision_factory=None, speech_factory=None):
        self.settings = settings
        self.store = Store(settings.data)
        self.clips = EvidenceClips(settings.data / 'evidence')
        saved_seconds=self.store.setting('standing_seconds',settings.standing_seconds)
        if type(saved_seconds) is int and 1<=saved_seconds<=300:
            self.settings.standing_seconds=saved_seconds
        self._apply_detection(self.store.setting('detection', {}) or {})
        self.dwell = {}
        from collections import deque
        self.speaking = deque(maxlen=20000)  # (offset, track id, student id, name, role, mouth activity)
        self.vision_factory, self.speech_factory = vision_factory, speech_factory
        self.vision = self.speech = None
        self.session = None
        self.owner = None
        self.started = 0
        self.lock = threading.RLock()
        self.audio = queue.Queue(maxsize=20)  # ~2 min of speech backlog on a slow host
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
        self.warm_state = 'idle'

    def model_status(self):
        paths = {"face_detector": "yunet.onnx", "identity": "sface.onnx",
                 "emotion": "emotion.onnx", "pose": "pose_lite.task",
                 "phone_detector": "phone_detector_v5.pt",
                 "voice": f"whisper-{self.settings.whisper_name}/model.bin"}
        if not self.settings.phone_enabled:
            paths.pop("phone_detector")
        status = {k: (self.settings.models / v).is_file() for k, v in paths.items()}
        if self.settings.vision_url:  # served by the remote vision service
            remote = getattr(self.vision, "remote", None) if self.vision else None
            reachable = remote is None or remote.error is None
            status["pose"] = reachable
            if "phone_detector" in status:
                status["phone_detector"] = reachable
        if self.settings.whisper_name in {'zipformer-vi-30m','zipformer-vi-30m-lora','zipformer-vi-30m-lora-low-lr20'}:
            status['voice'] = all((self.settings.models / self.settings.whisper_name / name).is_file()
                                  for name in ['encoder.int8.onnx','decoder.onnx','joiner.int8.onnx','tokens.txt'])
        return status

    def _download_phone_model(self):
        if not self.settings.phone_enabled or self.settings.vision_url:
            return
        phone_path = self.settings.models / "phone_detector_v5.pt"
        if phone_path.is_file():
            return
        # The checkpoint ships in the image (models/ in the repo); copy it into the
        # persistent model directory, falling back to Firebase Storage if absent.
        from .config import ROOT
        bundled = ROOT / "models" / "phone_detector_v5.pt"
        if bundled.is_file() and bundled.resolve() != phone_path.resolve():
            import shutil
            phone_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = phone_path.with_name(phone_path.name + ".copy")
            shutil.copyfile(bundled, temporary)
            temporary.replace(phone_path)
            return
        gcs_path = os.getenv("EXAMWATCH_PHONE_MODEL_GCS_PATH", "").strip()
        if gcs_path:
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
                    apply_thresholds(track, self.settings)
                    track.standing_rule.reset()
                    track.head_down_rule.reset()
            return {'standing_threshold_seconds':self.settings.standing_seconds}

    def _apply_detection(self, saved):
        s = self.settings
        if isinstance(saved.get('sensitivity'), int) and 0 <= saved['sensitivity'] <= 100:
            s.sensitivity = saved['sensitivity']
        for key, value in sensitivity_thresholds(s.sensitivity).items():
            setattr(s, key, value)
        if isinstance(saved.get('advanced'), dict):
            try:
                s.advanced = validate_advanced(saved['advanced'])
            except ValueError:
                logger.warning("Ignoring invalid saved advanced settings: %s", saved['advanced'])
        for key, (_, _, default, *_rest) in ADVANCED.items():
            if key in (s.advanced or {}):
                setattr(s, key, (s.advanced or {})[key])
            elif default is not None:
                setattr(s, key, default)
        if isinstance(saved.get('enabled'), dict):
            try:
                s.enabled = validate_enabled(saved['enabled'])
            except ValueError:
                logger.warning("Ignoring invalid saved alert toggles: %s", saved['enabled'])
        if isinstance(saved.get('auto_calibrate'), bool):
            s.auto_calibrate = saved['auto_calibrate']
        evidence = saved.get('evidence') or {}
        for kind in ('image', 'video', 'audio'):
            if isinstance(evidence.get(kind), bool):
                setattr(s, f'evidence_{kind}', evidence[kind])
        dwell = saved.get('proctor_dwell_seconds')
        if isinstance(dwell, int) and 10 <= dwell <= 1800:
            s.proctor_dwell_seconds = dwell

    def detection_settings(self):
        s = self.settings
        return {'standing_threshold_seconds': s.standing_seconds, 'sensitivity': s.sensitivity,
                'evidence': {'image': s.evidence_image, 'video': s.evidence_video, 'audio': s.evidence_audio},
                'proctor_dwell_seconds': s.proctor_dwell_seconds,
                'thresholds': sensitivity_thresholds(s.sensitivity), 'max_people': s.max_people,
                'advanced': {k: round(float(getattr(s, k)), 3) for k in ADVANCED},
                'overridden': sorted((s.advanced or {}).keys()),
                'enabled': {k: s.rule_on(k) for k in ALERT_KINDS},
                'auto_calibrate': s.auto_calibrate,
                'schema': [{'key': k, 'min': lo, 'max': hi, 'group': g, 'label': label, 'help': help_text}
                           for k, (lo, hi, _, g, label, help_text) in ADVANCED.items()],
                'groups': ADVANCED_GROUPS, 'kinds': ALERT_KINDS}

    def configure_detection(self, sensitivity, evidence, proctor_dwell_seconds, standing_seconds,
                            advanced=None, enabled=None, auto_calibrate=None):
        """advanced: explicit overrides ({} clears them so the slider decides again);
        None keeps what was saved. Same for enabled / auto_calibrate."""
        if type(sensitivity) is not int or not 0 <= sensitivity <= 100:
            raise ValueError('Độ nhạy phải là số nguyên từ 0 đến 100.')
        if type(proctor_dwell_seconds) is not int or not 10 <= proctor_dwell_seconds <= 1800:
            raise ValueError('Thời gian nhắc giám thị phải từ 10 đến 1800 giây.')
        if not all(isinstance(evidence.get(k), bool) for k in ('image', 'video', 'audio')):
            raise ValueError('Tham số bằng chứng không hợp lệ.')
        advanced = validate_advanced(advanced) if advanced is not None else None
        enabled = validate_enabled(enabled) if enabled is not None else None
        if auto_calibrate is not None and not isinstance(auto_calibrate, bool):
            raise ValueError('Tự hiệu chuẩn: cần bật/tắt.')
        with self.lock:
            previous = self.store.setting('detection', {}) or {}
            saved = {'sensitivity': sensitivity, 'evidence': {k: evidence[k] for k in ('image', 'video', 'audio')},
                     'proctor_dwell_seconds': proctor_dwell_seconds,
                     'advanced': advanced if advanced is not None else previous.get('advanced', {}),
                     'enabled': enabled if enabled is not None else previous.get('enabled', {}),
                     'auto_calibrate': auto_calibrate if auto_calibrate is not None
                     else previous.get('auto_calibrate', True)}
            self.settings.advanced, self.settings.enabled = {}, {}
            self.store.set_setting('detection', saved)
            self._apply_detection(saved)
        self.configure_standing(standing_seconds)
        with self.lock:
            targets = list(self.visible)
            if self.vision and hasattr(self.vision, 'tracker'):
                targets += list(self.vision.tracker.tracks.values())
            for track in {id(t): t for t in targets}.values():
                apply_thresholds(track, self.settings)
            if self.vision is not None and hasattr(self.vision, 'phone_at'):
                self.vision.phone_at = -1e9  # rescan with the new confidence
        return self.detection_settings()

    def warmup(self):
        """Load the face/emotion models (and speech) at boot, so 'Bắt đầu phiên'
        does not wait for them on a cold Render instance."""
        self.warm_state = 'loading'
        try:
            with self.lock:  # never race start()/enrollment into a second model copy
                if self.session:
                    self.warm_state = 'ready'
                    return
                vision = self.load_vision()
                if hasattr(vision, 'warmup'):
                    vision.warmup()
                if self.speech is None and self.speech_factory is None:
                    from .speech import Speech
                    self.speech = Speech(self.settings)
            self.warm_state = 'ready'
        except Exception as exc:
            logger.exception("Model warm-up failed (models load at session start instead)")
            self.warm_state = f'error: {exc}'

    def _pose_events(self, track, frame, now):
        """Phone, head turning, lean and arm events for one person."""
        s, events = self.settings, []
        cooldown = s.alert_cooldown
        if track.has_phone and now - track.phone_alert_at >= cooldown:
            track.phone_alert_at = now
            events.append(("phone", f"Phát hiện điện thoại gần {track.name} ({track.phone_confidence:.0%}), "
                                    f"thấy liên tục ≥ {s.phone_confirm_seconds:g} giây; cần giám thị kiểm tra."))
        h, w = frame.shape[:2]
        anchor = track.baseline.get('abdomen_anchor') if track.baseline else None
        angles = body_angles(track.landmarks, w, h, anchor) if anchor else body_angles(track.landmarks, w, h)
        if track.angles.get('lean_source') != angles['lean_source']:
            # Changing geometry reference must not look like angle growth.
            track.left_arm_rule.reset()
            track.right_arm_rule.reset()
            track.lean_rule.reset()
        track.angles = angles
        # An automatic seat reference also needs the lean pivot, otherwise lean
        # stays unmeasured until the proctor presses 'Hiệu chuẩn'.
        if (track.baseline is not None and 'abdomen_anchor' not in track.baseline
                and angles.get('lean_origin') and track.posture == 'seated'):
            track.baseline['abdomen_anchor'] = angles['lean_origin'][:]
        head_on = s.rule_on('head_turning')
        looking_aside = (head_on and track.head_source is not None and abs(track.head_yaw) >= .7 * s.head_yaw
                         and track.facing_seconds >= FACING_SECONDS)
        head_dir = side(track.head_yaw)
        if head_on and track.head_turning and now - track.head_alert_at >= cooldown:
            track.head_alert_at = now
            events.append(("head_turning", f"Quay mặt sang {head_dir} khung hình liên tục ≥ {s.head_seconds:g} giây "
                                           "(có thể nhìn bài bên cạnh); cần giám thị kiểm tra."))
        for key, rule in [('left_arm', track.left_arm_rule), ('right_arm', track.right_arm_rule), ('lean', track.lean_rule)]:
            if not rule.update(angles[key], now):
                continue
            if key == 'lean':
                if looking_aside:
                    # Turning to read a neighbour's paper tilts the torso too: report
                    # one 'looking aside' event instead of a separate body lean.
                    if now - track.head_alert_at >= cooldown:
                        track.head_alert_at = now
                        events.append(("head_turning", f"Quay mặt sang {head_dir} kèm nghiêng người {rule.value:.0f}° "
                                                       "(có thể nhìn bài bên cạnh); cần giám thị kiểm tra."))
                elif now - track.lean_alert_at >= cooldown:
                    track.lean_alert_at = now
                    events.append(('body_lean', f'Nghiêng thân sang {side(angles.get("lean_side", 1))} '
                                                f'{rule.value:.1f}°; dấu hiệu tư thế cần giám thị kiểm tra.'))
            elif now - track.arm_alert_at >= cooldown:
                track.arm_alert_at = now
                label = 'Tay trái' if key == 'left_arm' else 'Tay phải'
                events.append(('arm_raise', f'{label}: góc {rule.value:.1f}°; dấu hiệu tư thế cần giám thị kiểm tra.'))
        return events

    def load_vision(self):
        if self.vision is None:
            self._download_phone_model()
            if self.vision_factory:
                self.vision = self.vision_factory(self.settings)
            else:
                # Enrollment only needs the face/pose models; the phone detector is
                # optional here (Vision skips it when absent) and enforced in start().
                missing = [name for name, ready in self.model_status().items()
                           if name not in {"voice", "phone_detector"} and not ready]
                if missing:
                    raise ValueError("Thiếu model hình ảnh: " + ", ".join(missing) + ".")
                from .vision import Vision
                self.vision = Vision(self.settings)
        return self.vision

    def start(self, title, owner):
        with self.lock:
            if self.session:
                raise ValueError("Đang có phiên thi hoạt động. Hãy kết thúc phiên hiện tại trước.")
            if self.settings.phone_enabled:
                self._download_phone_model()
            if not self.vision_factory and not all(self.model_status().values()):
                missing = [name for name, ready in self.model_status().items() if not ready]
                hint = ""
                if "phone_detector" in missing:
                    cloud_error = getattr(self.store.cloud, "error", None)
                    hint = (" Upload phone_detector_v5.pt lên Firebase Storage tại "
                            f"'{os.getenv('EXAMWATCH_PHONE_MODEL_GCS_PATH', '').strip() or 'models/phone_detector_v5.pt'}'."
                            + (f" Lỗi: {cloud_error}" if cloud_error else ""))
                raise ValueError("Thiếu model: " + ", ".join(missing) + "." + hint)
            if not self.vision_factory and self.vision is not None \
                    and getattr(self.vision, "phone_detector", True) is None \
                    and (self.settings.models / "phone_detector_v5.pt").is_file():
                # Vision was loaded for enrollment before the phone model arrived; reload it.
                self.vision.close()
                self.vision = None
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
            roster = [{"id": p["id"], "name": p["name"]} for p in self.gallery if p.get("role", "student") != "proctor"]
            self.session = self.store.new_session(title, roster)
            self.dwell = {}
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
            result = self.vision.process(frame, self.gallery)
            tracks, overflow = result[0], result[1]
            phones = result[2] if len(result) > 2 else []
            now = time.monotonic()
            offset = now - self.started
            self.clips.push(frame, now)
            ratio = float(getattr(self.settings, 'min_face_ratio', 0) or 0)
            largest = max((t.box[3] for t in tracks if t.role != "proctor"), default=0)
            for track in tracks:
                events = self._pose_events(track, frame, now)
                if track.standing_rule.update(track.posture == "standing", now):
                    events.append(("standing", f"Nhổm dậy khỏi ghế ≥ {self.settings.stand_seconds:g} giây; cần giám thị kiểm tra."))
                if track.head_down_rule.update(track.posture == "head_down", now):
                    events.append(("head_down", f"Cúi xuống kéo dài ít nhất {self.settings.standing_seconds:g} giây; cần giám thị kiểm tra."))
                if track.moving_rule.update(track.posture == "moving", now):
                    events.append(("moving", "Thí sinh rời khỏi vị trí ngồi đã hiệu chuẩn; cần giám thị kiểm tra."))
                # Bowed head held for a long time with a flat/sad face reads as tiredness.
                # Tired: eyes closed a large share of the last minute / repeated yawns
                # (face blendshapes), or the head bowed for a long time with a flat face.
                bowed = track.tired_rule.update(track.posture == "head_down" and track.emotion in {"neutral", "sad", "unknown"}, now) \
                    or (track.tired and track.posture == "head_down")
                track.tired = bool(track.drowsy or bowed)
                if track.tired and not getattr(track, 'tired_alerted', False):
                    track.tired_alerted = True
                    reason = (f"mắt nhắm {track.perclos * 100:.0f}% thời gian trong 1 phút" if track.perclos >= .35 else
                              f"ngáp {track.yawns} lần" if track.yawns >= 2 else "cúi gục kéo dài")
                    events.append(("tired", f"Dấu hiệu mệt mỏi ({reason}); cân nhắc hỏi thăm, hỗ trợ thí sinh."))
                elif not track.tired:
                    track.tired_alerted = False
                negative = (track.emotion in {"sad", "angry", "fear"}
                            and track.emotion_score >= self.settings.emotion_confidence)
                if track.emotion_rule.update(negative, now):
                    events.append(("negative_emotion", f"Biểu cảm buồn/lo lắng kéo dài ít nhất {self.settings.negative_seconds:g} giây; kiểm tra tình trạng thí sinh."))
                events = [e for e in events if self.settings.rule_on(e[0])]
                if track.role == "proctor":
                    events = []  # proctors are never reported as candidates
                if ratio and track.box[3] < ratio * largest:
                    events = []  # a small face far behind the candidates (proctor setting)
                for kind, detail in events:
                    self._record_alert(track, kind, detail, offset, now, frame)
                if track.role != "proctor" and now - track.observed_at >= 1.0:
                    self.store.observe(self.session["id"], track, offset)
                    track.observed_at = now
            self._proctor_dwell(tracks, offset, now, frame)
            for t in tracks:
                self.speaking.append((offset, t.id, t.student_id, t.name, t.role, t.mouth_activity))
            self.visible, self.last_frame = tracks, now
            self.inference_ms = round((now - started) * 1000)
            return {"tracks": [self.serialize(t, now) for t in tracks], "overflow": overflow, "phones": phones,
                    "inference_ms": self.inference_ms, "offset_seconds": offset}
        finally:
            self.lock.release()

    def _snapshot(self, frame, track, label):
        evidence = frame.copy()
        h, w = frame.shape[:2]
        x, y, bw, bh = track.box
        cv2.rectangle(evidence, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), (50, 80, 255), 2)
        cv2.putText(evidence, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 80, 255), 2)
        encoded, jpeg = cv2.imencode('.jpg', evidence)
        if not encoded:
            raise OSError("Không lưu được ảnh bằng chứng.")
        filename = uuid.uuid4().hex + ".jpg"
        (self.settings.data / "evidence" / filename).write_bytes(jpeg.tobytes())
        return filename

    def _record_alert(self, track, kind, detail, offset, now, frame):
        label = f"Track {track.id} | {kind} | +{offset:.1f}s"
        video_kind = kind in {'standing', 'arm_raise', 'body_lean', 'phone', 'head_turning', 'head_down', 'moving', 'proctor_dwell'}
        if video_kind and self.settings.evidence_video:
            filename = uuid.uuid4().hex + '.webm'
            thumb = self._snapshot(frame, track, label)
            self.store.add_alert(self.session['id'], track, kind, detail, offset, filename, thumbnail=thumb)
            self.clips.trigger(filename, now, label)
        elif self.settings.evidence_image:
            filename = self._snapshot(frame, track, label)
            self.store.add_alert(self.session["id"], track, kind, detail, offset, filename, thumbnail=filename)
        else:
            self.store.add_alert(self.session["id"], track, kind, detail, offset, "")

    def _proctor_dwell(self, tracks, offset, now, frame):
        """Remind a registered proctor who stays beside one candidate's desk too long."""
        proctors = [t for t in tracks if t.role == "proctor"]
        # A proctor bending over a desk often hides that candidate's face; keep
        # candidates seen in the last few seconds so the timer does not restart.
        nearby = list(tracks)
        tracker = getattr(getattr(self, 'vision', None), 'tracker', None)
        if tracker is not None:
            nearby += [t for t in tracker.tracks.values() if t not in tracks and now - t.last_seen < 5]
        students = [t for t in nearby if t.role != "proctor"]
        active = set()
        for proctor in proctors:
            px, py = proctor.box[0] + proctor.box[2] / 2, proctor.box[1] + proctor.box[3] / 2
            best = None
            for student in students:
                sx, sy = student.box[0] + student.box[2] / 2, student.box[1] + student.box[3] / 2
                reach = max(0.08, student.box[2] * 2.5)
                d = abs(px - sx)
                if d < reach and abs(py - sy) < 0.35 and (best is None or d < best[0]):
                    best = (d, student)
            if not best:
                continue
            student = best[1]
            key = (proctor.id, student.id)
            active.add(key)
            rule = self.dwell.setdefault(key, Sustained(self.settings.proctor_dwell_seconds, max_gap=4))
            rule.seconds = self.settings.proctor_dwell_seconds
            proctor.dwell_seconds, proctor.dwell_student = round(rule.elapsed(now), 1), student.name
            if rule.update(True, now) and self.settings.rule_on('proctor_dwell'):
                class Marker:  # alert row describes the proctor at the candidate's desk
                    id, box, student_id = proctor.id, proctor.box, student.student_id
                    name = f"{proctor.name} · bàn {student.name}"
                self._record_alert(Marker, "proctor_dwell",
                                   f"Giám thị {proctor.name} đứng cạnh bàn {student.name} quá {self.settings.proctor_dwell_seconds:g} giây.",
                                   offset, now, frame)
        for key in [k for k in self.dwell if k not in active]:
            if now - (self.dwell[key].last or now) > 4:
                del self.dwell[key]
        for proctor in proctors:
            if not any(k[0] == proctor.id for k in active):
                proctor.dwell_seconds, proctor.dwell_student = 0, None

    @staticmethod
    def serialize(t, now):
        return {"id": t.id, "bbox": t.box, "student_id": t.student_id, "name": t.name,
                "identity_score": round(t.identity_score, 3), "emotion": t.emotion,
                "emotion_score": round(t.emotion_score, 3), "posture": t.posture,
                "has_phone": t.has_phone, "phone_seen": t.phone_seen, "phone_confidence": round(t.phone_confidence, 3),
                "head_turning": t.head_turning, "head_yaw": round(t.head_yaw, 3), "head_source": t.head_source,
                "head_seconds": round(t.head_rule.elapsed(now), 1),
                "facing_ready": t.facing_seconds >= FACING_SECONDS,
                "calibrated": t.baseline is not None, "calibration_auto": t.baseline_auto, "landmarks": t.landmarks,
                "lean_threshold": round(t.lean_rule.threshold, 1),
                "mouth_points": t.mouth_points,
                "pose_angles": {key: round(rule.value,1) if rule.value is not None else None
                    for key,rule in [('left_arm',t.left_arm_rule),('right_arm',t.right_arm_rule),('lean',t.lean_rule)]},
                "lean_source": t.angles.get('lean_source'),
                "lean_origin": t.angles.get('lean_origin'),
                "lean_target": t.angles.get('lean_target'),
                "pose_flags": [key for key,rule in [('left_arm',t.left_arm_rule),('right_arm',t.right_arm_rule),('lean',t.lean_rule)] if rule.flagged],
                "standing_seconds": round(t.standing_rule.elapsed(now), 1),
                "role": t.role, "tired": t.tired, "drowsy": t.drowsy, "perclos": round(t.perclos, 2),
                "mouth_activity": round(t.mouth_activity, 2),
                "dwell_seconds": getattr(t, "dwell_seconds", 0), "dwell_student": getattr(t, "dwell_student", None),
                "negative_seconds": round(t.emotion_rule.elapsed(now), 1)}

    def queue_audio(self, samples, rate, offset, owner, capture_id):
        with self.lock:
            self.require_owner(owner, capture_id)
            duration = len(samples) / rate
            # Offsets come from the browser. A clock that runs ahead or a late
            # first chunk used to reject every chunk of the session; place the
            # chunk on the session timeline instead.
            elapsed = time.monotonic() - self.started
            if offset + duration > elapsed + 3:
                offset = max(self.last_audio_end, elapsed - duration)
            offset = max(offset, self.last_audio_end)
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
                    audio = self._save_clip(samples, rate, segment) if self.settings.evidence_audio else None
                    start, end = offset + segment["start"], offset + segment["end"]
                    speaker = self.attribute_speaker(start, end)
                    self.store.add_transcript(session_id, start, end, segment["text"], audio, speaker)
                self.audio_error = None
            except Exception as exc:
                logger.exception("Audio transcription failed")
                self.audio_error = str(exc)
            finally:
                self.audio_busy = False
                self.audio.task_done()

    def attribute_speaker(self, start, end):
        """Person whose mouth moved most while the segment was spoken, if clearly ahead."""
        with self.lock:
            # Frames arrive about once a second on a small host: a +/-0.4 s window
            # usually held no frame at all, so nobody could be named.
            window = [e for e in self.speaking if start - 1.0 <= e[0] <= end + 1.0]
        per = {}
        for _, track_id, student_id, name, role, activity in window:
            item = per.setdefault(track_id, [0.0, 0, student_id, name, role])
            item[0] += activity; item[1] += 1
            if student_id: item[2], item[3] = student_id, name
        ranked = sorted(((total / count, sid, name, role) for total, count, sid, name, role in per.values() if count),
                        key=lambda r: r[0], reverse=True)
        if not ranked or ranked[0][0] < self.settings.speak_threshold:
            return None
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        if second and ranked[0][0] < second * 1.4:
            return None
        best = ranked[0]
        return {"id": best[1], "name": ("Giám thị " if best[3] == "proctor" else "") + best[2],
                "score": round(best[0] / (best[0] + second), 2) if best[0] + second else 1.0}

    def _save_clip(self, samples, rate, segment):
        import wave
        a = max(0, int((segment["start"] - 0.2) * rate))
        b = min(len(samples), int((segment["end"] + 0.3) * rate)) or len(samples)
        pcm = (np.clip(samples[a:b], -1, 1) * 32767).astype("<i2").tobytes()
        if not pcm:
            return None
        filename = uuid.uuid4().hex + ".wav"
        with wave.open(str(self.settings.data / "evidence" / filename), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(int(rate)); wav.writeframes(pcm)
        return filename

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
                "max_people": self.settings.max_people, "frame_width": self.settings.frame_width,
                "detection": self.detection_settings(), "warmup": self.warm_state}

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
