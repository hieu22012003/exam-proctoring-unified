"""Regression tests for the customer feedback of 2026-11-08 (no model files needed).

    python -m pytest tests -q
"""
import threading
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examwatch.config import Settings, ADVANCED  # noqa: E402
from examwatch.rules import (pose_head_yaw, classify_posture, posture_signal,  # noqa: E402
                             AutoBaseline, Persistence, sensitivity_thresholds, FACING_SECONDS)
from examwatch.tracking import Track, apply_thresholds  # noqa: E402
from examwatch.vision import Vision, assign_phone, plausible_phone  # noqa: E402
from examwatch.runtime import Runtime  # noqa: E402


def body(nose=(.5, .30), shoulders_y=.50, sw=.20, cx=.5, ears=((.44, .29, .9), (.56, .29, .9)), wrists=None):
    """33 MediaPipe-style landmarks [x, y, z, visibility]."""
    lm = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
    lm[0] = [nose[0], nose[1], 0, .95]
    lm[2] = [nose[0] - .025, nose[1] - .02, 0, .9]
    lm[5] = [nose[0] + .025, nose[1] - .02, 0, .9]
    lm[7] = [ears[0][0], ears[0][1], 0, ears[0][2]]
    lm[8] = [ears[1][0], ears[1][1], 0, ears[1][2]]
    lm[11] = [cx - sw / 2, shoulders_y, 0, .95]
    lm[12] = [cx + sw / 2, shoulders_y, 0, .95]
    lm[13] = [cx - sw / 2 - .02, shoulders_y + .15, 0, .9]
    lm[14] = [cx + sw / 2 + .02, shoulders_y + .15, 0, .9]
    if wrists:
        lm[15] = [*wrists[0], 0, .9]
        lm[16] = [*wrists[1], 0, .9]
    return lm


def settings(**overrides):
    s = Settings()
    for key, value in sensitivity_thresholds(s.sensitivity).items():
        setattr(s, key, value)
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


def track(s, box=(.45, .2, .1, .12)):
    t = Track(1, list(box), 0)
    apply_thresholds(t, s)
    return t


class FakeVision:
    """Vision's per-track helpers without loading any model."""
    _update_head = Vision._update_head
    _update_posture = Vision._update_posture

    def __init__(self, s):
        self.settings = s


# ---------------------------------------------------------------- 4. quay mặt nhìn bài
def test_head_turn_confirmed_at_one_fps():
    """Old counter restarted when frames were > 0.75 s apart: never fired on Render."""
    s = settings()
    v, t = FakeVision(s), track(s)
    t.landmarks = body()
    for i in range(4):  # first seen facing the camera (~1.1 FPS)
        v._update_head(t, 0.0, i * .9)
    states = []
    for i in range(4, 9):  # then turns: yaw 0.5 > threshold 0.30
        v._update_head(t, .5, i * .9)
        states.append(t.head_turning)
    assert states[:2] == [False, False] and any(states), states


def test_head_yaw_from_pose_when_face_is_lost():
    lm = body(nose=(.58, .30), ears=((.48, .29, .9), (.55, .29, .05)))  # right ear hidden
    assert pose_head_yaw(lm) == 1.0
    assert abs(pose_head_yaw(body())) < .05
    s = settings()
    v, t = FakeVision(s), track(s)
    t.landmarks = lm
    v._update_head(t, None, 0)
    assert t.head_source == 'pose' and t.head_yaw == 1.0


def _runtime(s):
    r = Runtime.__new__(Runtime)
    r.settings = s
    return r


def _lean_events(rt, t, yaw, source, facing=FACING_SECONDS):
    frame = np.zeros((480, 640, 3), np.uint8)
    t.facing_seconds = facing  # seen facing the camera before this
    t.baseline = {"nose_y": .3, "shoulder_y": .5, "shoulder_x": .5, "scale": .12,
                  "shoulder_width": .2, "abdomen_anchor": [.5, .70]}
    t.posture, kinds, now = 'seated', [], 0.0
    for shift in [0, .02, .06, .10, .12, .12, .12, .12]:
        t.landmarks = body(cx=.5 + shift, nose=(.5 + shift, .3))
        t.head_yaw, t.head_source = yaw, source
        kinds += [k for k, _ in rt._pose_events(t, frame, now)]
        now += .8
    return kinds


def test_turning_to_neighbour_is_reported_as_looking_aside_not_lean():
    s = settings()
    kinds = _lean_events(_runtime(s), track(s), .45, 'pose')
    assert 'head_turning' in kinds and 'body_lean' not in kinds, kinds


def test_plain_lean_is_still_a_lean():
    s = settings()
    kinds = _lean_events(_runtime(s), track(s), 0.0, 'face')
    assert 'body_lean' in kinds and 'head_turning' not in kinds, kinds


# ---------------------------------------------------------------- 2. nhổm dậy
def test_half_rise_is_standing_with_new_rule_but_not_old():
    face_h = .12
    base = posture_signal(body(), [.45, .24, .1, face_h])
    risen = posture_signal(body(nose=(.5, .23), shoulders_y=.45), [.45, .17, .1, face_h])
    assert classify_posture(base, risen, .5) == 'standing'
    # Old rule: both head and shoulders above max(0.07, 0.75 x face height).
    assert not (base['nose_y'] - risen['nose_y'] > .09 and base['shoulder_y'] - risen['shoulder_y'] > .09)


def test_standing_alert_without_pressing_calibrate():
    s = settings()
    v, t = FakeVision(s), track(s)
    t.landmarks, now = body(), 0.0
    for _ in range(7):  # sits still ~3 s at ~2 FPS
        v._update_posture(t, now)
        now += .5
    assert t.baseline is not None and t.baseline_auto
    t.landmarks, fired = body(nose=(.5, .22), shoulders_y=.44), False
    for _ in range(4):
        v._update_posture(t, now)
        fired = t.standing_rule.update(t.posture == 'standing', now) or fired
        now += .6
    assert t.posture == 'standing' and fired


def test_auto_calibration_waits_for_stillness_and_can_be_disabled():
    ab = AutoBaseline()
    moving = [posture_signal(body(nose=(.5, .30 + .06 * (i % 2))), [.45, .24, .1, .12]) for i in range(8)]
    assert all(ab.update(sig, i * .5) is None for i, sig in enumerate(moving))
    s = settings(auto_calibrate=False)
    v, t = FakeVision(s), track(s)
    t.landmarks = body()
    for i in range(8):
        v._update_posture(t, i * .5)
    assert t.baseline is None and t.posture == 'uncalibrated'


# ---------------------------------------------------------------- 3. điện thoại
def test_phone_in_lap_is_attributed():
    t = Track(1, [.45, .2, .1, .12], 0)
    t.landmarks = body(wrists=((.45, .75), (.55, .75)))
    assert assign_phone(.5, .88, [t]) is t          # below the old landmark box
    assert assign_phone(.95, .88, [t]) is None


def test_phone_between_two_people_goes_to_nearest_wrist():
    a, b = Track(1, [.25, .2, .1, .12], 0), Track(2, [.57, .2, .1, .12], 0)
    a.landmarks = body(cx=.30, nose=(.30, .3), wrists=((.25, .7), (.38, .7)))
    b.landmarks = body(cx=.62, nose=(.62, .3), wrists=((.55, .7), (.70, .7)))
    assert assign_phone(.40, .72, [a, b]) is a        # old code: 2 candidates -> dropped


def test_proctor_never_gets_the_phone():
    p = Track(1, [.45, .2, .1, .12], 0)
    p.role, p.landmarks = 'proctor', body(wrists=((.45, .7), (.55, .7)))
    assert assign_phone(.5, .7, [p]) is None


def test_large_boxes_are_not_phones():
    face = Track(1, [.45, .2, .1, .12], 0)
    assert plausible_phone(.06, .10, face)
    assert not plausible_phone(.37, .60)               # a screen/panel, as on the demo videos
    assert not plausible_phone(.20, .30, face)         # 5x the face area


def test_phone_needs_to_persist():
    p = Persistence(1.0, max_gap=1.6)
    p.hit(0)
    assert not p.confirmed(0)
    p.hit(.5); p.hit(1.0)
    assert p.confirmed(1.0) and not p.confirmed(3.0)


# ---------------------------------------------------------------- 5. thông số giám thị
class FakeStore:
    def __init__(self):
        self.saved = {}

    def setting(self, key, default=None):
        return self.saved.get(key, default)

    def set_setting(self, key, value):
        self.saved[key] = value


def _configurable_runtime():
    r = Runtime.__new__(Runtime)
    r.settings, r.store, r.lock = Settings(), FakeStore(), threading.RLock()
    r.visible, r.vision = [], None
    r._apply_detection({})
    return r


def test_advanced_settings_override_and_reach_live_tracks():
    r = _configurable_runtime()
    t = track(r.settings)
    r.visible = [t]
    d = r.configure_detection(50, {'image': True, 'video': True, 'audio': True}, 60, 5,
                              advanced={'lean_threshold': 12, 'head_seconds': 3, 'phone_confidence': .6},
                              enabled={'arm_raise': False}, auto_calibrate=False)
    assert t.lean_rule.threshold == 12 and t.head_rule.seconds == 3
    assert r.settings.phone_confidence == .6 and not r.settings.rule_on('arm_raise')
    assert d['overridden'] == ['head_seconds', 'lean_threshold', 'phone_confidence']
    assert d['auto_calibrate'] is False and len(d['schema']) == len(ADVANCED)
    # Omitted = keep; {} = back to the sensitivity slider.
    r.configure_detection(70, {'image': True, 'video': True, 'audio': True}, 60, 5)
    assert r.settings.lean_threshold == 12
    r.configure_detection(70, {'image': True, 'video': True, 'audio': True}, 60, 5, advanced={})
    assert r.settings.lean_threshold == pytest.approx(sensitivity_thresholds(70)['lean_threshold'])


def test_out_of_range_rejected():
    r = _configurable_runtime()
    with pytest.raises(ValueError):
        r.configure_detection(50, {'image': True, 'video': True, 'audio': True}, 60, 5, advanced={'lean_threshold': 200})
    with pytest.raises(ValueError):
        r.configure_detection(50, {'image': True, 'video': True, 'audio': True}, 60, 5, enabled={'bogus': True})


def test_disabled_rule_is_silent():
    s = settings(enabled={'body_lean': False})
    r = _runtime(s)
    kinds = [k for k in _lean_events(r, track(s), 0.0, 'face') if s.rule_on(k)]
    assert 'body_lean' not in kinds


# ---------------------------------------------------------------- feedback 09/11 (video thử)
def test_person_in_profile_from_the_start_is_never_flagged():
    """Video 09/11, giây 30: người ngồi máy tính phía sau, luôn nghiêng mặt."""
    s = settings()
    v, t = FakeVision(s), track(s)
    t.landmarks = body(nose=(.58, .30), ears=((.48, .29, .9), (.55, .29, .05)))
    for i in range(12):
        v._update_head(t, None, i * .9)
        assert not t.head_turning
    assert t.facing_seconds == 0


def test_lean_of_a_never_facing_person_is_not_turned_into_looking_aside():
    s = settings()
    kinds = _lean_events(_runtime(s), track(s), .45, 'pose', facing=0.0)
    assert 'head_turning' not in kinds


def test_turn_is_measured_from_the_resting_yaw_of_an_angled_seat():
    s = settings()
    v, t = FakeVision(s), track(s)
    t.landmarks = body()
    t.baseline = {"nose_y": .3, "shoulder_y": .5, "shoulder_x": .5, "scale": .12,
                  "shoulder_width": .2, "yaw": .25}  # seat at an angle to the camera
    for i in range(8):
        v._update_head(t, .25, i * .9)  # sitting normally: raw yaw .25
    assert not t.head_turning and t.facing_seconds >= FACING_SECONDS
    for i in range(8, 13):
        v._update_head(t, .70, i * .9)  # turns a further .45
    assert t.head_turning


def test_resting_yaw_cannot_hide_a_large_turn():
    s = settings()
    v, t = FakeVision(s), track(s)
    t.baseline = {"yaw": 1.0}  # reference taken while already looking aside
    t.facing_seconds = FACING_SECONDS
    t.landmarks = body()
    for i in range(5):
        v._update_head(t, 1.0, i * .9)
    assert t.head_turning  # offset capped at REST_YAW_LIMIT


def test_auto_baseline_records_the_resting_yaw():
    ab = AutoBaseline()
    got = None
    for i in range(8):
        sig = dict(posture_signal(body(), [.45, .24, .1, .12]), yaw=.2 + .01 * (i % 2))
        got = ab.update(sig, i * .5) or got
    assert got is not None and abs(got['yaw'] - .205) < .01


def test_turned_face_reattaches_to_the_known_person_at_the_same_seat():
    from examwatch.tracking import Tracker
    tr = Tracker(settings())
    [t] = tr.update([[.40, .20, .12, .16]], 0.0)
    t.student_id, t.name = 'TS1', 'nguyen van a'
    # Profile: box shifts sideways and narrows, IoU with the old box ~0.
    [again] = tr.update([[.50, .21, .07, .15]], 1.0)
    assert again is t and again.name == 'nguyen van a'
    # Someone appearing far away is a different person.
    tr2 = Tracker(settings())
    [a] = tr2.update([[.10, .20, .12, .16]], 0.0)
    a.student_id = 'TS1'
    [b] = tr2.update([[.70, .20, .12, .16]], 1.0)
    assert b is not a and b.student_id is None
