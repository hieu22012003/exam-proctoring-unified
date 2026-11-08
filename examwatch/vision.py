"""CPU-only adapters. All weights are explicit local paths; no runtime downloads."""
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .rules import posture_signal, classify_posture, pose_head_yaw
from .tracking import Tracker, identify

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
# COCO-17 keypoints (YOLO pose) -> MediaPipe-33 indices used by the posture rules.
COCO_TO_MP = {0: (0,), 1: (1, 2, 3), 2: (4, 5, 6), 3: (7,), 4: (8,), 5: (11,), 6: (12,), 7: (13,), 8: (14,),
              9: (15,), 10: (16,), 11: (23,), 12: (24,), 13: (25,), 14: (26,), 15: (27,), 16: (28,)}


def coco_to_mediapipe(keypoints, width, height):
    """One YOLO person (17 x [x, y, conf] in pixels) as 33 normalized [x, y, z, visibility]."""
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
    for coco, targets in COCO_TO_MP.items():
        x, y, conf = (float(v) for v in keypoints[coco][:3])
        for t in targets:
            out[t] = [x / width, y / height, 0.0, conf]
    return out


def region(gray, cx, cy, half_w, half_h, size=(32, 20)):
    """Fixed-size grayscale patch around a point; None when it leaves the frame."""
    h, w = gray.shape[:2]
    x1, x2 = int(cx - half_w), int(cx + half_w)
    y1, y2 = int(cy - half_h), int(cy + half_h)
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 - x1 < 6 or y2 - y1 < 4:
        return None
    return cv2.resize(gray[y1:y2, x1:x2], size, interpolation=cv2.INTER_AREA).astype(np.float32)


def mouth_motion(track, gray, face):
    """Frame-to-frame motion of the mouth minus the eye band (head motion, lighting).

    A cheap per-frame speaking cue: lips move while talking, the eye band mostly does not."""
    eye_l, eye_r = (float(face[4]), float(face[5])), (float(face[6]), float(face[7]))
    m_l, m_r = (float(face[10]), float(face[11])), (float(face[12]), float(face[13]))
    span = max(4.0, abs(m_r[0] - m_l[0]))
    mouth = region(gray, (m_l[0] + m_r[0]) / 2, (m_l[1] + m_r[1]) / 2 + span * .12, span * .85, span * .55)
    eyes = region(gray, (eye_l[0] + eye_r[0]) / 2, (eye_l[1] + eye_r[1]) / 2, span * 1.1, span * .4)
    prev = getattr(track, 'mouth_patch', None)
    track.mouth_patch = (mouth, eyes)
    if mouth is None or eyes is None or prev is None or prev[0] is None or prev[1] is None:
        return 0.0
    activity = max(0.0, float(np.mean(np.abs(mouth - prev[0]))) - .8 * float(np.mean(np.abs(eyes - prev[1]))))
    track.mouth_activity = .5 * activity + .5 * getattr(track, 'mouth_activity', 0.0)
    return track.mouth_activity


def update_drowsiness(track, blendshapes, now, window=60.0):
    """PERCLOS (share of samples with eyes closed) and yawns from face blendshapes."""
    scores = {c.category_name: float(c.score) for c in blendshapes}
    eyes = (scores.get('eyeBlinkLeft', 0) + scores.get('eyeBlinkRight', 0)) / 2
    jaw = scores.get('jawOpen', 0)
    samples = getattr(track, 'eye_samples', None)
    if samples is None:
        from collections import deque
        samples = track.eye_samples = deque(maxlen=120)
    samples.append((now, eyes >= .55, jaw >= .6))
    while samples and now - samples[0][0] > window * 2:
        samples.popleft()
    recent = [s for s in samples if now - s[0] <= window]
    track.perclos = sum(s[1] for s in recent) / len(recent) if len(recent) >= 5 else 0.0
    track.yawns = sum(s[2] for s in samples)
    track.eyes_closed, track.jaw_open = eyes, jaw
    track.drowsy = len(recent) >= 5 and (track.perclos >= .35 or track.yawns >= 2)
    return track.drowsy


def due(tracks, stamp, interval, now, budget):
    """Tracks whose identity/emotion refresh is due, oldest first, capped per frame."""
    ready = [t for t in tracks if now - getattr(t, stamp) >= interval]
    if budget and len(tracks) > 3:
        ready = sorted(ready, key=lambda t: getattr(t, stamp))[:budget]
    return {id(t) for t in ready}


# Real phones are small. On the repo's demo recordings the phone model returned
# boxes 37-61 % of the frame width (screens, panels, sheets) at up to 0.60
# confidence; those are rejected here instead of becoming phone alerts.
PHONE_MAX_WIDTH, PHONE_MAX_HEIGHT, PHONE_MAX_FACE_AREA = .30, .45, 4.0


def plausible_phone(bw, bh, owner=None):
    """False for boxes far too large to be a phone (bw, bh normalized)."""
    if bw > PHONE_MAX_WIDTH or bh > PHONE_MAX_HEIGHT:
        return False
    if owner is not None and len(getattr(owner, 'box', ())) == 4:
        face_area = max(owner.box[2] * owner.box[3], 1e-4)
        if bw * bh > PHONE_MAX_FACE_AREA * face_area:
            return False
    return True


def assign_phone(cx, cy, tracks):
    """The person a phone box most plausibly belongs to, or None.

    Region = the visible body extended down to the bottom of the frame (lap, under
    the desk) and half a shoulder width sideways. When regions overlap, the nearest
    wrist wins. The old rule required the box inside the landmark bounding box and
    dropped it whenever two people qualified."""
    best = None
    for track in tracks:
        if getattr(track, 'role', 'student') == 'proctor':
            continue
        lm = track.landmarks
        points = [p for p in lm if len(p) >= 4 and p[3] >= .25]
        if len(points) >= 4 and len(lm) > 16:
            px, py = [p[0] for p in points], [p[1] for p in points]
            sw = abs(lm[11][0] - lm[12][0]) if lm[11][3] >= .25 and lm[12][3] >= .25 else .1
            margin = max(.05, sw * .5)
            if not (min(px) - margin <= cx <= max(px) + margin and cy >= min(py) - .05):
                continue
            anchors = [lm[i][:2] for i in (15, 16, 17, 18, 19, 20) if len(lm) > i and lm[i][3] >= .3]
            if not anchors:
                anchors = [[(lm[11][0] + lm[12][0]) / 2, (lm[11][1] + lm[12][1]) / 2 + sw]]
        else:
            bx, by, bw, bh = track.box
            if not (bx - bw <= cx <= bx + 2 * bw and cy >= by):
                continue
            anchors = [[bx + bw / 2, by + bh * 2.5]]
        distance = min((cx - a[0]) ** 2 + (cy - a[1]) ** 2 for a in anchors)
        if best is None or distance < best[0]:
            best = (distance, track)
    return best[1] if best else None


def mouth_points(face, width, height):
    """YuNet mouth corners plus their geometric midpoint, not a lip-contour model."""
    corners=sorted([[float(face[10]/width),float(face[11]/height)],
                    [float(face[12]/width),float(face[13]/height)]],key=lambda p:p[0])
    if not np.isfinite(corners).all() or any(not 0<=v<=1 for p in corners for v in p):
        return []
    left,right=corners
    return [left,[(left[0]+right[0])/2,(left[1]+right[1])/2],right]


class Vision:
    def __init__(self, settings):
        import onnxruntime as ort
        self.remote = None
        if settings.vision_url:
            # Pose, blendshapes and phones run on examwatch.vision_service; MediaPipe is
            # never imported here, which keeps this process small.
            from .remote_vision import RemoteVision
            self.remote = RemoteVision(settings.vision_url, settings.vision_token, width=settings.frame_width)
            self.remote_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='examwatch-remote')
            self.mesh_at = -1e9
            mp = None
        else:
            import mediapipe as mp
        self.settings, self.mp = settings, mp
        cv2.setNumThreads(settings.threads)
        root = settings.models
        # OpenCV's filename overload does not handle all Unicode Windows paths.
        self.detector = cv2.FaceDetectorYN.create(
            "onnx", np.frombuffer((root / "yunet.onnx").read_bytes(),dtype=np.uint8),
            np.empty(0,dtype=np.uint8), (320, 320), score_threshold=0.8,
            nms_threshold=0.3, top_k=100)
        self.recognizer = cv2.FaceRecognizerSF.create("onnx",
            np.frombuffer((root / "sface.onnx").read_bytes(),dtype=np.uint8),np.empty(0,dtype=np.uint8))
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.threads
        options.inter_op_num_threads = 1
        self.emotion = ort.InferenceSession(str(root / "emotion.onnx"),
                                          sess_options=options, providers=["CPUExecutionProvider"])
        self.emotion_input = self.emotion.get_inputs()[0]
        # This adapter intentionally accepts the DeepFace NHWC grayscale model only.
        if self.emotion_input.shape[-3:] != [48, 48, 1]:
            raise ValueError(f"Emotion ONNX shape không phù hợp: {self.emotion_input.shape}")
        self.pose = self.make_pose(1) if self.remote is None else None
        self.multi_pose = None
        self.yolo_pose = None
        engine = settings.pose_engine if settings.pose_engine in {"yolo", "mediapipe"} else (
            "yolo" if settings.max_people > 3 else "mediapipe")
        if engine == "yolo" and self.remote is None:
            # One pass returns every person's body: needed for a full classroom.
            try:
                from ultralytics import YOLO
                path = root / "yolo11n-pose.pt"
                self.yolo_pose = YOLO(str(path) if path.is_file() else "yolo11n-pose.pt")
            except Exception:
                import logging
                logging.getLogger(__name__).exception("YOLO pose unavailable; falling back to MediaPipe")
                self.yolo_pose = None
        self.face_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='examwatch-face')
        self.previous_face_count = 0
        self.multi_until = 0
        self.last_timestamp = 0
        self.tracker = Tracker(settings)
        self.face_mesh = None
        mesh_path = root / "face_landmarker.task"
        if mesh_path.is_file() and self.remote is None:
            try:  # eye closure / yawning for the "tired" state; optional
                self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(
                    mp.tasks.vision.FaceLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_buffer=mesh_path.read_bytes()),
                        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1,
                        output_face_blendshapes=True, min_face_detection_confidence=.4))
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Face landmarker unavailable")
                self.face_mesh = None
        self.phone_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='examwatch-phone')
        self.phone_boxes, self.phone_at = [], -1e9
        self.phone_detector = None
        phone_path = root / "phone_detector_v5.pt"
        if settings.phone_enabled and phone_path.is_file() and self.remote is None:
            try:
                from ultralytics import YOLO
                self.phone_detector = YOLO(str(phone_path))
            except Exception as exc:
                raise RuntimeError(f"Không tải được model phát hiện điện thoại: {exc}") from exc

    def make_pose(self, count):
        mp = self.mp
        return mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=(self.settings.models / "pose_lite.task").read_bytes()),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=count,
                min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5, output_segmentation_masks=False))

    def reset(self):
        self.tracker = Tracker(self.settings)
        self.previous_face_count = 0
        self.multi_until = 0

    def close(self):
        if hasattr(self, 'face_worker'):
            self.face_worker.shutdown(wait=True)
        if hasattr(self, 'phone_worker'):
            self.phone_worker.shutdown(wait=True)
        if getattr(self, 'remote', None) is not None:
            self.remote_worker.shutdown(wait=True)
            self.remote.close()
        if getattr(self, 'face_mesh', None) is not None:
            self.face_mesh.close()
        if getattr(self, 'pose', None) is not None:
            self.pose.close()
        if getattr(self, 'multi_pose', None) is not None:
            self.multi_pose.close()

    def faces(self, frame, enrollment=False):
        h, w = frame.shape[:2]
        # Small detector input, restore original pixel coordinates including 5 face landmarks.
        scale = min(1.0, (640 if enrollment else self.settings.detector_width) / w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        if faces is None:
            return []
        faces[:, :14] /= scale
        return sorted(faces, key=lambda f: float(f[2] * f[3]), reverse=True)

    def embedding(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        vector = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-9)

    def enrollment(self, frame):
        faces = self.faces(frame, enrollment=True)
        if len(faces) != 1:
            raise ValueError("Ảnh đăng ký cần có đúng một khuôn mặt.")
        if min(faces[0][2:4]) < 65:
            raise ValueError("Khuôn mặt quá nhỏ. Hãy đến gần camera hơn.")
        return self.embedding(frame, faces[0]).tolist()

    def predict_emotion(self, frame, face):
        h, w = frame.shape[:2]
        x, y, fw, fh = face[:4]
        x1, y1, x2, y2 = max(0, int(x)), max(0, int(y)), min(w, int(x + fw)), min(h, int(y + fh))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        crop = frame[y1:y2, x1:x2]
        # Match demography.analyze -> preprocessing.resize_image -> Emotion.predict:
        # preserve aspect ratio, pad to 224, normalize to [0,1], grayscale -> 48x48 NHWC.
        scale = min(224 / crop.shape[0], 224 / crop.shape[1])
        resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                                    max(1, int(crop.shape[0] * scale))))
        dy, dx = 224 - resized.shape[0], 224 - resized.shape[1]
        padded = cv2.copyMakeBorder(resized, dy // 2, dy - dy // 2,
                                   dx // 2, dx - dx // 2, cv2.BORDER_CONSTANT, value=0)
        normalized = padded.astype(np.float32) / 255.0
        gray = cv2.resize(cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY), (48, 48))
        tensor = gray[None, :, :, None]
        prediction = self.emotion.run(None, {self.emotion_input.name: tensor})[0].reshape(-1)
        if prediction.size != 7 or not np.isfinite(prediction).all():
            return None
        if prediction.min() < 0:
            prediction = np.exp(prediction - prediction.max())
        return prediction / max(float(prediction.sum()), 1e-9)

    def detect_phones(self, frame):
        h, w = frame.shape[:2]
        # 640 = the size the model was trained/exported at (same as the Render ONNX).
        result = self.phone_detector.predict(frame, conf=self.settings.phone_confidence, imgsz=640, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().tolist()]
            boxes.append(((x1 + x2) / (2 * w), (y1 + y2) / (2 * h), float(box.conf[0].cpu()),
                          (x2 - x1) / w, (y2 - y1) / h))
        return boxes

    def warmup(self):
        """Run the local models once at server start so the first session is not slow."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[120:360, 240:400] = 128
        self.faces(frame)
        self.predict_emotion(frame, np.array([240, 120, 160, 240] + [0] * 11, dtype=np.float32))
        if getattr(self, 'phone_detector', None) is not None:
            self.detect_phones(frame)
        self.reset()

    def _update_head(self, track, face_yaw, now):
        """Head yaw from YuNet and from the pose ears; the larger one wins."""
        pose_yaw = pose_head_yaw(track.landmarks)
        if pose_yaw is not None and (face_yaw is None or abs(pose_yaw) > abs(face_yaw)):
            yaw, source = pose_yaw, 'pose'
        else:
            yaw, source = face_yaw, ('face' if face_yaw is not None else None)
        track.head_yaw, track.head_source = (float(yaw) if yaw is not None else 0.0), source
        active = yaw is not None and abs(yaw) >= self.settings.head_yaw
        track.head_rule.update(active, now)
        track.head_turning = active and track.head_rule.elapsed(now) >= track.head_rule.seconds

    def _update_posture(self, track, now, signal=None):
        track.signal = signal if signal is not None else posture_signal(track.landmarks, track.box)
        if track.baseline is None and getattr(self.settings, 'auto_calibrate', True):
            base = track.auto_baseline.update(track.signal, now)
            if base is not None:
                track.baseline, track.baseline_auto = base, True
                track.baseline_owner = track.student_id
                track.standing_rule.reset()
        track.posture = classify_posture(track.baseline, track.signal, getattr(self.settings, 'standing_rise', .5))

    def blendshapes(self, frame, face):
        h, w = frame.shape[:2]
        x, y, fw, fh = (float(v) for v in face[:4])
        pad = .35 * max(fw, fh)
        x1, y1, x2, y2 = max(0, int(x - pad)), max(0, int(y - pad)), min(w, int(x + fw + pad)), min(h, int(y + fh + pad))
        if x2 - x1 < 40 or y2 - y1 < 40:
            return None
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        scale = 256 / max(crop.shape[:2])
        if scale < 1:
            crop = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)))
        result = self.face_mesh.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
        return result.face_blendshapes[0] if result.face_blendshapes else None

    def process(self, frame, gallery, now=None):
        now = time.monotonic() if now is None else now
        h, w = frame.shape[:2]
        # The phone detector is the slowest model: run it beside face/pose work and
        # at most every phone_interval seconds, reusing the last boxes in between.
        phone_job = None
        if getattr(self, 'phone_detector', None) is not None and now - getattr(self, 'phone_at', -1e9) >= self.settings.phone_interval:
            worker = getattr(self, 'phone_worker', None)
            phone_job = worker.submit(self.detect_phones, frame) if worker else None
            if worker is None:
                self.phone_boxes, self.phone_at = self.detect_phones(frame), now
        remote_job = None
        if getattr(self, 'remote', None) is not None:
            want_phone = self.settings.phone_enabled and now - getattr(self, 'phone_at', -1e9) >= self.settings.phone_interval
            want_mesh = now - self.mesh_at >= 1.0
            remote_job = self.remote_worker.submit(self.remote.analyze, frame, want_phone, want_mesh, self.settings.phone_confidence)
            if want_mesh:
                self.mesh_at = now
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # DNN detection and pose consume the same immutable frame concurrently.
        face_job = self.face_worker.submit(self.faces, frame) if hasattr(self,'face_worker') else None
        timestamp = max(self.last_timestamp + 1, int(now * 1000))
        self.last_timestamp = timestamp
        poses = []
        all_faces = None if face_job else self.faces(frame)
        need_pose = bool(self.tracker.tracks or getattr(self,'previous_face_count',0) or (all_faces is not None and len(all_faces)))
        if not need_pose and face_job:
            all_faces = face_job.result()
            need_pose = bool(len(all_faces))
        image = None
        used_multi = False
        remote = None
        if remote_job is not None:
            remote = remote_job.result() or {}
            poses = remote.get("poses") or []
            used_multi = True
            if remote.get("phones") is not None:
                self.phone_boxes, self.phone_at = [tuple(p) for p in remote["phones"]], now
        elif need_pose and getattr(self, 'yolo_pose', None) is not None:
            result = self.yolo_pose.predict(frame, imgsz=self.settings.yolo_imgsz, conf=0.3, verbose=False,
                                            max_det=self.settings.max_people + 5)[0]
            keypoints = getattr(result, 'keypoints', None)
            data = getattr(keypoints, 'data', None) if keypoints is not None else None
            if data is not None and len(data):
                data = data.cpu().numpy() if hasattr(data, 'cpu') else np.asarray(data)
                poses = [coco_to_mediapipe(person, w, h) for person in data]
            used_multi = True
        elif need_pose:
            scale = min(1.0, self.settings.pose_width / w)
            pose_frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                  data=np.ascontiguousarray(cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)))
            # Dormant duplicate tracks from a one-person identity switch must
            # not force the slower multi-person engine. Hold only confirmed
            # multiple faces/bodies across temporary face occlusion.
            used_multi = getattr(self,'previous_face_count',0)>1 or now<getattr(self,'multi_until',0)
            if used_multi and getattr(self,'multi_pose',None) is None:
                self.multi_pose = self.make_pose(self.settings.max_people)
            engine = self.multi_pose if used_multi else self.pose
            result = engine.detect_for_video(image, timestamp)
            poses = [[[float(l.x), float(l.y), float(l.z), float(l.visibility)] for l in p]
                     for p in result.pose_landmarks]
        if all_faces is None:
            all_faces = face_job.result()
        faces = all_faces[:self.settings.max_people]
        # A newly entered second/third person gets multi-person inference on
        # this frame, rather than waiting for the next detection cycle.
        if len(faces)>1 and not used_multi and image is not None and getattr(self, 'yolo_pose', None) is None and remote is None:
            if getattr(self,'multi_pose',None) is None:
                self.multi_pose = self.make_pose(self.settings.max_people)
            result = self.multi_pose.detect_for_video(image,timestamp)
            poses = [[[float(l.x),float(l.y),float(l.z),float(l.visibility)] for l in p] for p in result.pose_landmarks]
        self.previous_face_count = len(faces)
        if len(faces)>1 or len(poses)>1:
            self.multi_until = now+2
        boxes = [[float(f[0]/w),float(f[1]/h),float(f[2]/w),float(f[3]/h)] for f in faces]
        tracks = self.tracker.update(boxes, now)
        assigned_poses = set()
        identity_due = due(tracks, 'identity_at', self.settings.identity_interval, now, self.settings.identity_budget)
        emotion_due = due(tracks, 'emotion_at', self.settings.emotion_interval, now, self.settings.emotion_budget)
        for track, face in zip(tracks, faces):
            eye_mid = (float(face[4]) + float(face[6])) / 2
            eye_span = max(1.0, abs(float(face[4]) - float(face[6])))
            face_yaw = (float(face[8]) - eye_mid) / eye_span
            track.mouth_points=mouth_points(face,w,h)
            mouth_motion(track, gray, face)
            if id(track) in identity_due:
                person, score = identify(self.embedding(frame, face) if gallery else None, gallery,
                                         self.settings.face_threshold, self.settings.identity_margin)
                new_id = person["id"] if person else None
                if track.student_id != new_id:
                    # A failed face match must not erase the seat calibration.
                    # A positive different identity starts with its own baseline.
                    keep = new_id is None or track.baseline_owner in (None, new_id)
                    track.clear_identity(preserve_baseline=keep)
                    if new_id in self.tracker.calibrations:
                        track.baseline = self.tracker.calibrations[new_id].copy()
                        track.baseline_owner = new_id
                track.student_id = new_id
                track.name = person["name"] if person else "Chưa xác định"
                track.role = person.get("role", "student") if person else "student"
                track.identity_score, track.identity_at = score, now
                if new_id and track.baseline:
                    track.baseline_owner = new_id
            if remote and remote.get("faces"):
                fx, fy = (float(face[0]) + float(face[2]) / 2) / w, (float(face[1]) + float(face[3]) / 2) / h
                near = min(remote["faces"], key=lambda m: (m["cx"] - fx) ** 2 + (m["cy"] - fy) ** 2)
                if abs(near["cx"] - fx) < float(face[2]) / w and abs(near["cy"] - fy) < float(face[3]) / h:
                    update_drowsiness(track, self.remote.blendshapes(near), now)
            elif id(track) in emotion_due and getattr(self, 'face_mesh', None) is not None:
                try:
                    shapes = self.blendshapes(frame, face)
                    if shapes:
                        update_drowsiness(track, shapes, now)
                except Exception:
                    pass
            if id(track) in emotion_due:
                probabilities = self.predict_emotion(frame, face)
                if probabilities is not None:
                    track.probabilities = (probabilities if track.probabilities is None else
                                           0.55 * probabilities + 0.45 * track.probabilities)
                    idx = int(np.argmax(track.probabilities))
                    track.emotion_score = float(track.probabilities[idx])
                    track.emotion = EMOTIONS[idx] if track.emotion_score >= self.settings.emotion_confidence else "unknown"
                else:
                    track.emotion, track.emotion_score = "unknown", 0
                    track.probabilities = None
                track.emotion_at = now
            candidates = []
            bx, by, bw, bh = track.box
            for i, pose in enumerate(poses):
                if i in assigned_poses:
                    continue
                nx, ny = pose[0][:2]
                if bx - bw * 0.25 <= nx <= bx + bw * 1.25 and by - bh * 0.25 <= ny <= by + bh * 1.25:
                    candidates.append(((nx - bx - bw / 2) ** 2 + (ny - by - bh / 2) ** 2, i))
            track.landmarks = []
            if candidates:
                _, index = min(candidates)
                assigned_poses.add(index)
                track.landmarks = poses[index]
            self._update_head(track, face_yaw, now)
            self._update_posture(track, now)
        # Recover briefly occluded faces using the previous head position and
        # visible shoulders. Never assign a pose twice or resolve ambiguous pairs.
        lost = [t for t in self.tracker.tracks.values() if t not in tracks and t.signal]
        proposals = []
        for i, pose in enumerate(poses):
            if i in assigned_poses:
                continue
            candidates = []
            for track in lost:
                signal = posture_signal(pose, track.box)
                if signal is None:
                    continue
                dx = signal['shoulder_x'] - track.signal['shoulder_x']
                dy = signal['nose_y'] - track.signal['nose_y']
                ratio = signal['shoulder_width'] / track.signal['shoulder_width']
                if abs(dx) < .10 and abs(dy) < .25 and .65 < ratio < 1.5:
                    candidates.append((dx*dx + dy*dy, track.id, signal))
            candidates.sort(key=lambda item: item[0])
            if candidates and (len(candidates) == 1 or candidates[1][0]-candidates[0][0] > .01):
                proposals.append((candidates[0][0], i, candidates[0][1], candidates[0][2]))
        recovered = set()
        for _, i, track_id, signal in sorted(proposals):
            if track_id in recovered:
                continue
            track = self.tracker.tracks[track_id]
            dx = signal['shoulder_x'] - track.signal['shoulder_x']
            dy = signal['nose_y'] - track.signal['nose_y']
            track.box = [track.box[0]+dx, track.box[1]+dy, *track.box[2:]]
            track.landmarks = poses[i]
            track.last_seen = now  # A fresh, unambiguous body observation keeps the track alive.
            track.mouth_points = []
            track.emotion, track.emotion_score = 'unknown', 0
            track.probabilities = None
            # The face is often lost *because* the head turned away (looking at a
            # neighbour's paper): keep measuring yaw from the pose ears.
            self._update_head(track, None, now)
            self._update_posture(track, now, signal)
            tracks.append(track)
            recovered.add(track_id)
        for track in self.tracker.tracks.values():
            if track not in tracks:
                track.head_rule.reset()
                track.auto_baseline.reset()
                track.head_turning = False
                track.standing_rule.reset()
                track.head_down_rule.reset()
                track.moving_rule.reset()
                track.tired_rule.reset()
                track.emotion_rule.reset()
                track.left_arm_rule.reset()
                track.right_arm_rule.reset()
                track.lean_rule.reset()
                track.angles = {}
        # Do not attach the same registered name to two visible people.
        for track in tracks:
            if track.student_id and sum(t.student_id == track.student_id for t in tracks) > 1:
                duplicate_id = track.student_id
                for other in tracks:
                    if other.student_id == duplicate_id:
                        other.clear_identity()
        fresh = phone_job is not None or (remote is not None and remote.get("phones") is not None)
        if phone_job is not None:
            try:
                self.phone_boxes, self.phone_at = phone_job.result(), now
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Phone detection failed")
                self.phone_boxes, self.phone_at = [], now
        if not self.settings.phone_enabled:
            self.phone_boxes = []
        for track in tracks:
            track.phone_seen = False
            track.phone_confidence = 0.0
        phones = []
        for box in getattr(self, 'phone_boxes', []):
            cx, cy, confidence = (float(v) for v in box[:3])
            sized = len(box) >= 5  # older vision services send centre + confidence only
            bw, bh = (float(box[3]), float(box[4])) if sized else (.05, .08)
            if sized and not plausible_phone(bw, bh):
                continue
            owner = assign_phone(cx, cy, tracks)
            if sized and owner is not None and not plausible_phone(bw, bh, owner):
                continue
            phones.append({"bbox": [cx - bw / 2, cy - bh / 2, bw, bh], "confidence": round(confidence, 3),
                           "track_id": owner.id if owner else None})
            if owner is not None and confidence > owner.phone_confidence:
                owner.phone_seen, owner.phone_confidence = True, confidence
        # One box is not enough: the phone must keep being detected for
        # phone_confirm_seconds (boxes are reused between scans, so count scans only).
        for track in tracks:
            if fresh and track.phone_seen:
                track.phone_rule.hit(now)
            track.has_phone = track.phone_rule.confirmed(now)
        return tracks, len(all_faces) > self.settings.max_people, phones
