"""Standing up during the exam (customer report 10/11: "đang thi thì đứng dậy" not detected).

The customer's camera is close: the face box is ~40 % of the frame height and the
server analyses ~0.7 frames per second. Standing up takes the head out of the
top of the frame, usually between two analysed frames.
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examwatch.main import parse_rises  # noqa: E402
from test_feedback_fixes import body  # noqa: E402
from test_vision_pipeline import FRAME, _Clips, _Store, face_row, make_vision  # noqa: E402

FPS = .7
SEATED_FACE = dict(cx=.5, cy=.45, size=.31)           # box top ~.25, height ~.40
SEATED_BODY = dict(nose=(.5, .45), shoulders_y=.85, sw=.5, ears=((.41, .43, .9), (.59, .43, .9)))


def seated(v):
    v.face_rows = [face_row(**SEATED_FACE)]
    v.remote.poses = [body(**SEATED_BODY)]


def empty(v):
    v.face_rows, v.remote.poses = [], []


def step(v, now, alerts, rises=None):
    """One analysed frame plus the hold-time rule exactly as Runtime.process applies it."""
    tracks, _, _ = v.process(FRAME, [], now=now, rises=rises)
    for t in tracks:
        if t.standing_rule.update(t.posture == "standing", now):
            alerts.append((t.id, now))
    return tracks


def calibrated(v, alerts):
    seated(v)
    now = 0.0
    while now < 5:
        tracks = step(v, now, alerts)
        now += 1 / FPS
    assert tracks[0].baseline is not None and tracks[0].posture == "seated"
    return tracks[0], now


def test_head_rising_out_of_the_top_is_standing():
    v, alerts = make_vision(), []
    person, now = calibrated(v, alerts)
    # The analysed frame happens to catch the rise: head and shoulders up.
    v.face_rows = [face_row(cx=.5, cy=.33, size=.31)]
    v.remote.poses = [body(nose=(.5, .33), shoulders_y=.76, sw=.5, ears=((.41, .31, .9), (.59, .31, .9)))]
    tracks = step(v, now, alerts)
    assert tracks[0] is person
    empty(v)                                 # then the head is above the frame
    for _ in range(3):
        now += 1 / FPS
        tracks = step(v, now, alerts)
    assert [t.id for t in tracks] == [person.id] and person.posture == "standing" and person.exit_up_at
    assert alerts == [(person.id, alerts[0][1])]


def test_browser_report_catches_a_stand_between_two_analysed_frames():
    v, alerts = make_vision(), []
    person, now = calibrated(v, alerts)
    empty(v)
    now += 1 / FPS
    step(v, now, alerts, rises=[(.51, .4)])
    now += 1 / FPS
    step(v, now, alerts, rises=[(.51, 1.8)])
    now += 1 / FPS
    step(v, now, alerts)
    assert len(alerts) == 1 and person.posture == "standing"


def test_close_camera_empty_seat_is_standing_even_4s_apart():
    """Customer test 10/11 18:50: frames ~4 s apart, the rise was never in a frame."""
    v, alerts = make_vision(), []
    person, now = calibrated(v, alerts)
    empty(v)
    for _ in range(3):
        now += 4.0
        tracks = step(v, now, alerts)
    assert [t.id for t in tracks] == [person.id] and person.exit_up_at
    assert len(alerts) == 1


def test_one_missed_frame_on_a_close_camera_is_not_an_alert():
    v, alerts = make_vision(), []
    person, now = calibrated(v, alerts)
    empty(v)
    now += 4.0
    step(v, now, alerts)
    seated(v)
    for _ in range(3):
        now += 4.0
        step(v, now, alerts)
    assert alerts == [] and person.posture == "seated"


def far_seat(v):
    v.face_rows = [face_row(cx=.3, cy=.5, size=.06)]
    v.remote.poses = [body(cx=.3, nose=(.3, .5), shoulders_y=.58, sw=.1, ears=((.28, .5, .9), (.32, .5, .9)))]


def test_far_camera_face_lost_without_a_rise_is_not_standing():
    """Room camera: small faces are missed by the detector now and then."""
    v, alerts = make_vision(), []
    far_seat(v)
    now = 0.0
    while now < 5:
        step(v, now, alerts)
        now += 1 / FPS
    empty(v)
    for _ in range(4):
        now += 1 / FPS
        tracks = step(v, now, alerts)
    assert alerts == [] and tracks == []


def test_report_from_another_seat_is_ignored():
    v, alerts = make_vision(), []
    far_seat(v)
    now = 0.0
    while now < 5:
        step(v, now, alerts)
        now += 1 / FPS
    empty(v)
    for _ in range(4):
        now += 1 / FPS
        step(v, now, alerts, rises=[(.9, .5)])
    assert alerts == []


def test_far_camera_browser_report_is_standing():
    v, alerts = make_vision(), []
    far_seat(v)
    now = 0.0
    while now < 5:
        step(v, now, alerts)
        now += 1 / FPS
    empty(v)
    for _ in range(3):
        now += 4.0
        step(v, now, alerts, rises=[(.3, 2.0)])
    assert len(alerts) == 1


def test_leaning_into_the_camera_is_not_standing():
    v, alerts = make_vision(), []
    person, now = calibrated(v, alerts)
    v.face_rows = [face_row(cx=.5, cy=.35, size=.5)]   # much larger face reaching the top edge
    v.remote.poses = [body(nose=(.5, .38), shoulders_y=.95, sw=.8, ears=((.36, .36, .9), (.64, .36, .9)))]
    step(v, now, alerts)
    empty(v)
    for _ in range(4):
        now += 1 / FPS
        step(v, now, alerts)
    assert alerts == []


def test_sitting_back_down_returns_to_seated_with_the_same_track():
    v, alerts = make_vision(), []
    person, now = calibrated(v, alerts)
    person.student_id, person.name = "TS1", "nguyen van a"
    empty(v)
    now += 1 / FPS
    step(v, now, alerts, rises=[(.5, .3)])
    now += 1 / FPS
    step(v, now, alerts)
    seated(v)
    for _ in range(2):
        now += 1 / FPS
        tracks = step(v, now, alerts)
    assert tracks == [person] and person.posture == "seated" and person.exit_up_at is None
    assert person.name == "nguyen van a"


def test_far_camera_rise_keeps_the_track_and_is_standing():
    """Room camera, small faces: a full rise moves the face ~2 face heights in one frame."""
    v, alerts = make_vision(), []
    v.face_rows = [face_row(cx=.5, cy=.5, size=.06)]
    v.remote.poses = [body(nose=(.5, .5), shoulders_y=.58, sw=.1, ears=((.48, .5, .9), (.52, .5, .9)))]
    now = 0.0
    while now < 5:
        tracks = step(v, now, alerts)
        now += 1 / FPS
    person = tracks[0]
    assert person.baseline
    v.face_rows = [face_row(cx=.5, cy=.34, size=.06)]
    v.remote.poses = [body(nose=(.5, .34), shoulders_y=.43, sw=.1, ears=((.48, .34, .9), (.52, .34, .9)))]
    for _ in range(3):
        tracks = step(v, now, alerts)
        now += 1 / FPS
    assert tracks == [person] and person.posture == "standing" and len(alerts) == 1


def test_standing_is_confirmed_on_a_slow_host():
    """Frames 2.8 s apart (5 people on a 0.15-CPU host) used to restart the timer."""
    v, alerts = make_vision(), []
    v.face_rows = [face_row(cx=.5, cy=.5, size=.06)]
    v.remote.poses = [body(nose=(.5, .5), shoulders_y=.58, sw=.1, ears=((.48, .5, .9), (.52, .5, .9)))]
    now = 0.0
    while now < 5:
        step(v, now, alerts)
        now += 1.0
    v.face_rows = [face_row(cx=.5, cy=.42, size=.06)]
    v.remote.poses = [body(nose=(.5, .42), shoulders_y=.51, sw=.1, ears=((.48, .42, .9), (.52, .42, .9)))]
    for _ in range(2):
        step(v, now, alerts)
        now += 2.8
    assert len(alerts) == 1


def test_runtime_passes_the_browser_report_and_alerts(monkeypatch):
    from examwatch.runtime import Runtime
    clock = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    v = make_vision()
    r = Runtime.__new__(Runtime)
    r.settings, r.vision, r.store, r.clips = v.settings, v, _Store(), _Clips()
    r.lock, r.session, r.owner, r.capture_id = threading.RLock(), {"id": "s"}, "o" * 8, "c"
    r.started, r.gallery, r.dwell, r.speaking = clock["t"], [], {}, []
    r._snapshot = lambda frame, track, label: "x.jpg"
    r.settings.evidence_video = False
    seated(v)
    for _ in range(5):
        out = r.process(FRAME, "o" * 8, "c")
        clock["t"] += 1 / FPS
    empty(v)
    for _ in range(3):
        out = r.process(FRAME, "o" * 8, "c", rises=[(.5, .5)])
        clock["t"] += 1 / FPS
    assert "standing" in getattr(r.store, "alerts", [])
    assert out["tracks"][0]["exited"] and out["tracks"][0]["posture"] == "standing"


def test_parse_rises_rejects_garbage():
    assert parse_rises("0.5:1.2,abc,2:1,0.3:-1,0.4:0.5") == [(.5, 1.2), (.4, .5)]
    assert parse_rises("") == []
