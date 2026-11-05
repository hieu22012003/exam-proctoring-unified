"""CPU-only adapters. All weights are explicit local paths; no runtime downloads."""
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .rules import posture_signal, classify_posture
from .tracking import Tracker, identify

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


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
        self.pose = self.make_pose(1)
        self.multi_pose = None
        self.face_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='examwatch-face')
        self.previous_face_count = 0
        self.multi_until = 0
        self.last_timestamp = 0
        self.tracker = Tracker(settings)
        self.phone_detector = None
        phone_path = root / "phone_detector_v5.pt"
        if phone_path.is_file():
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

    def process(self, frame, gallery, now=None):
        now = time.monotonic() if now is None else now
        h, w = frame.shape[:2]
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
        if need_pose:
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
        if len(faces)>1 and not used_multi and image is not None:
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
        for track, face in zip(tracks, faces):
            eye_mid = (float(face[4]) + float(face[6])) / 2
            eye_span = max(1.0, abs(float(face[4]) - float(face[6])))
            track.head_yaw = (float(face[8]) - eye_mid) / eye_span
            turning = abs(track.head_yaw) >= 0.30
            if turning:
                if now - track.head_suspicion_last > 0.75:
                    track.head_suspicion_since = now
                    track.head_suspicion_samples = 0
                track.head_suspicion_last = now
                track.head_suspicion_samples += 1
            else:
                track.head_suspicion_since = None
                track.head_suspicion_samples = 0
            track.head_turning = bool(turning and track.head_suspicion_since is not None
                                      and now - track.head_suspicion_since >= 1.25
                                      and track.head_suspicion_samples >= 3)
            track.mouth_points=mouth_points(face,w,h)
            if now - track.identity_at >= self.settings.identity_interval:
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
                track.identity_score, track.identity_at = score, now
                if new_id and track.baseline:
                    track.baseline_owner = new_id
            if now - track.emotion_at >= self.settings.emotion_interval:
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
            track.signal = posture_signal(track.landmarks, track.box)
            track.posture = classify_posture(track.baseline, track.signal)
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
            track.signal, track.landmarks = signal, poses[i]
            track.last_seen = now  # A fresh, unambiguous body observation keeps the track alive.
            track.mouth_points = []
            track.emotion, track.emotion_score = 'unknown', 0
            track.probabilities = None
            track.posture = classify_posture(track.baseline, signal)
            tracks.append(track)
            recovered.add(track_id)
        for track in self.tracker.tracks.values():
            if track not in tracks:
                track.head_suspicion_since = None
                track.head_suspicion_samples = 0
                track.head_turning = False
                track.standing_rule.reset()
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
        for track in tracks:
            track.has_phone = False
            track.phone_confidence = 0.0
        if self.phone_detector is not None:
            result = self.phone_detector.predict(frame, conf=self.settings.phone_confidence,
                                                 imgsz=960, verbose=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().tolist()]
                cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
                confidence = float(box.conf[0].cpu())
                candidates = []
                for track in tracks:
                    points = [p for p in track.landmarks if len(p) >= 4 and p[3] >= .25]
                    if len(points) < 4:
                        continue
                    px, py = [p[0] for p in points], [p[1] for p in points]
                    if min(px) - .06 <= cx <= max(px) + .06 and min(py) - .04 <= cy <= max(py) + .04:
                        candidates.append(track)
                if len(candidates) == 1 and confidence > candidates[0].phone_confidence:
                    candidates[0].has_phone = True
                    candidates[0].phone_confidence = confidence
        return tracks, len(all_faces) > self.settings.max_people
