"""Candidate/proctor verification, proctor dwell and audio stability (feedback 09/11)."""
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examwatch.runtime import Runtime  # noqa: E402
from examwatch.speech import MIN_VOICED_SECONDS, voiced_seconds  # noqa: E402
from examwatch.tracking import Track, Tracker  # noqa: E402
from test_feedback_fixes import body  # noqa: E402
from test_vision_pipeline import FRAME, face_row, make_vision  # noqa: E402

A = {"id": "TS1", "name": "nguyen van a", "role": "student"}
B = {"id": "TS2", "name": "tran thi b", "role": "student"}


def _identify_from(answers, monkeypatch):
    import examwatch.vision as vision_module
    monkeypatch.setattr(vision_module, "identify", lambda vec, gallery, thr, margin: answers["value"])


def test_one_frame_matching_a_neighbour_does_not_rename(monkeypatch):
    answers = {"value": (A, .8)}
    _identify_from(answers, monkeypatch)
    v = make_vision()
    v.settings.identity_interval = 0.0
    v.remote.poses = [body()]
    for i in range(3):
        tracks, _, _ = v.process(FRAME, [A, B], now=i * .9)
    assert tracks[0].student_id == "TS1"
    answers["value"] = (B, .5)            # a single blurred frame
    tracks, _, _ = v.process(FRAME, [A, B], now=3 * .9)
    assert tracks[0].student_id == "TS1"
    answers["value"] = (A, .8)
    tracks, _, _ = v.process(FRAME, [A, B], now=4 * .9)
    assert tracks[0].student_id == "TS1" and tracks[0].pending_id is None


def test_two_matches_in_a_row_switch_the_name(monkeypatch):
    answers = {"value": (A, .8)}
    _identify_from(answers, monkeypatch)
    v = make_vision()
    v.settings.identity_interval = 0.0
    v.remote.poses = [body()]
    for i in range(3):
        tracks, _, _ = v.process(FRAME, [A, B], now=i * .9)
    answers["value"] = (B, .8)            # the seat really changed hands
    names = []
    for i in range(3, 6):
        tracks, _, _ = v.process(FRAME, [A, B], now=i * .9)
        names.append(tracks[0].student_id)
    assert names[0] == "TS1" and names[-1] == "TS2"


def test_same_name_on_two_people_stays_on_the_best_match(monkeypatch):
    import examwatch.vision as vision_module
    v = make_vision()
    v.settings.identity_interval = 0.0
    xs = [.3, .7]
    v.face_rows = [face_row(cx=x, size=.1) for x in xs]
    v.remote.poses = [body(cx=x, nose=(x, .3), sw=.14, ears=((x - .03, .29, .9), (x + .03, .29, .9))) for x in xs]
    scores = iter([.9, .6] * 20)
    monkeypatch.setattr(vision_module, "identify", lambda vec, gallery, thr, margin: (A, next(scores)))
    for i in range(4):
        tracks, _, _ = v.process(FRAME, [A], now=i * .9)
    named = [t for t in tracks if t.student_id == "TS1"]
    assert len(named) == 1
    # The other person keeps the seat calibration even though the name was removed.
    assert all(t.baseline for t in tracks)


def test_duplicate_enrollment_finds_the_same_face_under_another_code():
    from examwatch.main import duplicate_enrollment
    rng = np.random.default_rng(0)
    face = rng.normal(size=128).astype(np.float32)
    face /= np.linalg.norm(face)
    other = rng.normal(size=128).astype(np.float32)
    other /= np.linalg.norm(other)
    enrolled = [{"id": "GT1", "name": "giam thi", "embeddings": [face.tolist()]},
                {"id": "TS9", "name": "khac", "embeddings": [other.tolist()]}]
    sample = face + rng.normal(scale=.02, size=128).astype(np.float32)
    sample /= np.linalg.norm(sample)
    assert duplicate_enrollment([sample], enrolled, "TS1", .45)["id"] == "GT1"
    assert duplicate_enrollment([sample], enrolled, "GT1", .45) is None   # re-enrolling oneself
    assert duplicate_enrollment([other * -1], enrolled, "TS1", .45) is None


def _dwell_runtime(seconds=10):
    r = Runtime.__new__(Runtime)
    v = make_vision()
    v.settings.proctor_dwell_seconds = seconds
    r.settings, r.vision, r.dwell = v.settings, v, {}
    r.alerts = []
    r._record_alert = lambda track, kind, detail, offset, now, frame: r.alerts.append((track.name, kind))
    return r


def _person(track_id, x, role, name, now):
    t = Track(track_id, [x - .05, .25, .1, .3], now)
    t.role, t.name, t.student_id = role, name, name
    return t


def test_proctor_dwell_alerts_once_even_when_the_candidate_is_hidden():
    r = _dwell_runtime(10)
    student = _person(1, .40, "student", "hs1", 0)
    proctor = _person(2, .52, "proctor", "gt1", 0)
    r.vision.tracker.tracks = {1: student, 2: proctor}
    now = 0.0
    while now < 12:
        student.last_seen = now if now < 4 or now > 7 else 3.9   # hidden 4-7 s by the proctor
        visible = [proctor] + ([student] if student.last_seen == now else [])
        r._proctor_dwell(visible, now, now, FRAME)
        now += 1.0
    assert r.alerts == [("gt1 · bàn hs1", "proctor_dwell")]
    assert proctor.dwell_student == "hs1"


def test_proctor_walking_past_does_not_alert():
    r = _dwell_runtime(10)
    students = [_person(i, x, "student", f"hs{i}", 0) for i, x in enumerate([.15, .5, .85], 1)]
    proctor = _person(9, 0, "proctor", "gt1", 0)
    r.vision.tracker.tracks = {t.id: t for t in students + [proctor]}
    for step in range(20):
        now = step * 1.0
        proctor.box[0] = .05 + step * .045 - .05
        for t in students:
            t.last_seen = now
        r._proctor_dwell(students + [proctor], now, now, FRAME)
    assert r.alerts == []


def test_voiced_seconds_ignores_steady_noise_and_keeps_speech():
    rate = 16000
    rng = np.random.default_rng(1)
    noise = (rng.normal(scale=.004, size=rate * 2)).astype(np.float32)
    assert voiced_seconds(noise, rate) < MIN_VOICED_SECONDS
    t = np.arange(rate) / rate
    speech = np.concatenate([noise[:rate // 2], .1 * np.sin(2 * np.pi * 220 * t).astype(np.float32), noise[:rate // 2]])
    assert voiced_seconds(speech, rate) > .8


def _audio_runtime():
    r = Runtime.__new__(Runtime)
    r.lock = threading.RLock()
    r.session = {"id": "s"}
    r.started = time.monotonic() - 10
    r.last_audio_end = 0.0
    r.audio = queue.Queue(maxsize=20)
    r.audio_rejected = 0
    r.require_owner = lambda owner, capture_id: None
    return r


def test_queue_audio_places_chunks_from_a_fast_browser_clock():
    r = _audio_runtime()
    samples = np.zeros(16000 * 2, np.float32)
    r.queue_audio(samples, 16000, 500.0, "o", "c")         # browser clock far ahead
    _, _, _, offset = r.audio.get_nowait()
    assert 7.5 <= offset <= 8.5
    r.queue_audio(samples, 16000, 3.0, "o", "c")           # overlapping: moved after the last one
    _, _, _, second = r.audio.get_nowait()
    assert second >= offset + 2 - 1e-6
    assert r.audio_rejected == 0
