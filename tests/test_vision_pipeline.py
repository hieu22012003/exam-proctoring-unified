"""Vision.process end to end in remote mode (as on Render), with fake detectors."""
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examwatch.config import Settings  # noqa: E402
from examwatch.rules import sensitivity_thresholds  # noqa: E402
from examwatch.tracking import Tracker  # noqa: E402
from examwatch.vision import Vision  # noqa: E402
from examwatch.runtime import Runtime  # noqa: E402
from test_feedback_fixes import body  # noqa: E402

W, H = 640, 480


def face_row(cx=.5, cy=.30, size=.12, yaw=0.0):
    """One YuNet row: x, y, w, h, 5 landmarks (eyes, nose, mouth corners), score."""
    fw, fh = size * W, size * H * 1.3
    x, y = cx * W - fw / 2, cy * H - fh / 2
    ex = fw * .18
    eye_y = y + fh * .4
    nose_x = cx * W + yaw * 2 * ex
    return np.array([x, y, fw, fh, cx * W - ex, eye_y, cx * W + ex, eye_y, nose_x, y + fh * .6,
                     cx * W - ex, y + fh * .8, cx * W + ex, y + fh * .8, .99], dtype=np.float32)


class FakeRemote:
    def __init__(self):
        self.poses, self.phones, self.error = [], None, None

    def analyze(self, frame, phone=False, mesh=False, conf=.35):
        return {"poses": self.poses, "faces": [], "phones": self.phones if phone else None}

    @staticmethod
    def blendshapes(face):
        return []


def make_vision(**overrides):
    s = Settings()
    for k, val in sensitivity_thresholds(s.sensitivity).items():
        setattr(s, k, val)
    s.phone_interval = .5
    for k, val in overrides.items():
        setattr(s, k, val)
    v = Vision.__new__(Vision)
    v.settings, v.mp, v.remote = s, None, FakeRemote()
    v.remote_worker = ThreadPoolExecutor(1)
    v.face_worker = ThreadPoolExecutor(1)
    v.mesh_at, v.previous_face_count, v.multi_until, v.last_timestamp = -1e9, 0, 0, 0
    v.tracker, v.phone_boxes, v.phone_at, v.phone_detector, v.face_mesh = Tracker(s), [], -1e9, None, None
    v.yolo_pose = None
    v.face_rows = [face_row()]
    v.faces = lambda frame, enrollment=False: list(v.face_rows)
    v.embedding = lambda frame, face: np.ones(128, np.float32) / np.sqrt(128)
    v.predict_emotion = lambda frame, face: None
    return v


FRAME = np.zeros((H, W, 3), np.uint8)


def run(v, seconds, fps=1.1, start=0.0):
    out, now = None, start
    while now < start + seconds:
        out = v.process(FRAME, [], now=now)
        now += 1 / fps
    return out, now


def test_pipeline_auto_calibrates_then_detects_standing():
    v = make_vision()
    v.remote.poses = [body()]
    (tracks, _, _), now = run(v, 4)
    assert tracks[0].baseline is not None and tracks[0].baseline_auto
    v.remote.poses = [body(nose=(.5, .22), shoulders_y=.44)]
    v.face_rows = [face_row(cy=.22)]
    (tracks, _, _), now = run(v, 2.5, start=now)
    assert tracks[0].posture == 'standing'  # the hold-time rule runs in Runtime.process


def test_pipeline_head_turn_survives_face_loss_at_1fps():
    v = make_vision()
    v.remote.poses = [body()]
    _, now = run(v, 4)
    # Head turns to profile: YuNet loses the face, pose ears still see it.
    v.face_rows = []
    v.remote.poses = [body(nose=(.56, .30), ears=((.47, .29, .9), (.55, .29, .05)))]
    (tracks, _, _), now = run(v, 3, start=now)
    assert tracks and tracks[0].head_source == 'pose' and tracks[0].head_turning


def test_pipeline_phone_confirmed_only_after_persisting_and_big_boxes_ignored():
    v = make_vision()
    v.remote.poses = [body(wrists=((.45, .75), (.55, .75)))]
    _, now = run(v, 2)
    v.remote.phones = [[.5, .85, .7, .05, .09], [.6, .5, .9, .45, .6]]  # phone in lap + a "screen"
    (tracks, _, phones), now = run(v, .5, fps=2, start=now)
    assert [p['track_id'] for p in phones] == [tracks[0].id]   # the screen-sized box is gone
    assert tracks[0].phone_seen and not tracks[0].has_phone      # seen once: not yet an alert
    (tracks, _, phones), now = run(v, 1.6, fps=2, start=now)
    assert tracks[0].has_phone


def test_pipeline_accepts_old_vision_service_format():
    v = make_vision()
    v.remote.poses = [body(wrists=((.45, .75), (.55, .75)))]
    _, now = run(v, 1)
    v.remote.phones = [[.5, .85, .7]]  # service not redeployed yet
    (tracks, _, phones), _ = run(v, .5, fps=2, start=now)
    assert phones and phones[0]['track_id'] == tracks[0].id


class _Store:
    def observe(self, *a):
        pass

    def add_alert(self, *a, **k):
        self.alerts = getattr(self, 'alerts', []) + [a[2]]


class _Clips:
    def push(self, *a):
        pass

    def trigger(self, *a):
        pass


def test_runtime_process_returns_phones_and_alerts():
    v = make_vision()
    v.remote.poses = [body(wrists=((.45, .75), (.55, .75)))]
    r = Runtime.__new__(Runtime)
    r.settings, r.vision, r.store, r.clips = v.settings, v, _Store(), _Clips()
    r.lock, r.session, r.owner, r.capture_id = threading.RLock(), {"id": "s"}, "o" * 8, "c"
    r.started, r.gallery, r.dwell, r.speaking = 0, [], {}, []
    r._snapshot = lambda frame, track, label: "x.jpg"
    r.settings.evidence_video = False
    import time
    t0 = time.monotonic()
    v.remote.phones = [[.5, .85, .7, .05, .09]]
    out = None
    for i in range(6):
        v.phone_at = -1e9
        out = r.process(FRAME, "o" * 8, "c")
        time.sleep(.25)
    assert out["phones"] and 'phone' in getattr(r.store, 'alerts', [])
    assert time.monotonic() - t0 < 10
