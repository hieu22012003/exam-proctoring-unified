"""Temporal rules use monotonic time; missing observations break continuity."""
from dataclasses import dataclass
import math


def body_angles(landmarks, width, height, abdomen_anchor=None):
    """Image-plane angles in pixels; never use hidden hips as reliable points."""
    result = {'left_arm': None, 'right_arm': None, 'lean': None, 'lean_source': None}
    def point(index):
        if index >= len(landmarks):
            return None
        p = landmarks[index]
        if len(p) < 4 or not all(math.isfinite(v) for v in p[:4]) or p[3] < .6:
            return None
        if not (0 <= p[0] <= 1 and 0 <= p[1] <= 1):
            return None
        return (p[0]*width, p[1]*height)
    left, right = point(11), point(12)
    if left is None or right is None:
        return result
    shoulder = (right[0]-left[0],right[1]-left[1])
    length = math.hypot(*shoulder)
    if length < 15:
        return result
    # Downward normal to the shoulder line also works with desk-occluded hips.
    down = (-shoulder[1],shoulder[0])
    if down[1] < 0:
        down = (-down[0],-down[1])
    hips = [point(23),point(24)]
    if all(p is not None for p in hips):
        center = ((left[0]+right[0])/2,(left[1]+right[1])/2)
        torso = ((hips[0][0]+hips[1][0])/2-center[0],(hips[0][1]+hips[1][1])/2-center[1])
        if torso[1] > 10 and math.hypot(*torso) > 20:
            down = torso
            result['lean_source'] = 'shoulders_hips'
    if result['lean_source'] is None:
        result['lean_source'] = 'shoulder_normal'
    result['lean'] = math.degrees(math.atan2(abs(down[0]),down[1]))
    for name, origin, index in [('left_arm',left,13),('right_arm',right,14)]:
        elbow = point(index)
        if elbow is None:
            continue
        arm = (elbow[0]-origin[0],elbow[1]-origin[1])
        norm = math.hypot(*arm)*math.hypot(*down)
        if math.hypot(*arm) < 15 or norm < 1e-6:
            continue
        result[name] = math.degrees(math.acos(max(-1,min(1,(arm[0]*down[0]+arm[1]*down[1])/norm))))
    # Lean uses an abdominal pivot; arm angles above retain their original reference.
    center = ((left[0]+right[0])/2,(left[1]+right[1])/2)
    if abdomen_anchor is not None and len(abdomen_anchor)==2 and all(math.isfinite(v) for v in abdomen_anchor):
        # A calibrated seat pivot must not jump to an intermittently visible hip.
        anchor = (abdomen_anchor[0]*width,abdomen_anchor[1]*height)
        result['lean_source'] = 'abdomen_calibrated'
    elif result['lean_source'] == 'shoulders_hips':
        anchor = (center[0]+.75*down[0],center[1]+.75*down[1])
        result['lean_source'] = 'abdomen_hips'
    else:
        # Seed only for seated calibration. Do not infer lean from a pivot that
        # moves with the same shoulders we are trying to measure.
        anchor = (center[0],center[1]+length)
        result['lean_source'] = 'abdomen_uncalibrated'
    result['lean_origin'] = [anchor[0]/width,anchor[1]/height]
    result['lean_target'] = [center[0]/width,center[1]/height]
    up = (center[0]-anchor[0],anchor[1]-center[1])
    result['lean'] = (math.degrees(math.atan2(abs(up[0]),up[1]))
                      if up[1]>15 and result['lean_source']!='abdomen_uncalibrated' else None)
    return result


@dataclass
class AngleWatch:
    threshold: float
    hold_seconds: float | None = None
    value: float | None = None
    since: float | None = None
    initial: float | None = None
    last: float | None = None
    emitted: bool = False
    held_since: float | None = None

    def reset(self):
        self.value = self.since = self.initial = self.last = self.held_since = None
        self.emitted = False

    @property
    def flagged(self):
        return self.since is not None

    def update(self, angle, now):
        if angle is None or not math.isfinite(angle):
            self.reset()
            return False
        if self.last is not None and (now-self.last > 2.5 or now < self.last):
            self.reset()
        self.last = now
        self.value = angle if self.value is None else .5*angle+.5*self.value
        if self.value >= self.threshold:
            if self.held_since is None:
                self.held_since = now
        else:
            self.held_since = None
        # Five-degree release hysteresis avoids repeated alerts at the boundary.
        if self.value < self.threshold-5:
            self.since = self.initial = None
            self.emitted = False
        if self.value >= self.threshold and self.since is None:
            self.since, self.initial = now, self.value
        if self.since is None or self.emitted:
            return False
        elapsed = now-self.since
        grew = self.value >= self.threshold and self.value >= self.initial+5 and elapsed >= .5
        held = (self.hold_seconds is not None and self.held_since is not None
                and now-self.held_since >= self.hold_seconds)
        if grew or held:
            self.emitted = True
            return True
        return False


@dataclass
class Sustained:
    seconds: float
    max_gap: float = 2.5
    since: float | None = None
    last: float | None = None
    emitted: bool = False

    def reset(self):
        self.since = self.last = None
        self.emitted = False

    def update(self, active: bool, now: float):
        if self.last is not None and (now - self.last > self.max_gap or now < self.last):
            self.reset()
        self.last = now
        if not active:
            self.since = None
            self.emitted = False
            return False
        if self.since is None:
            self.since = now
        if not self.emitted and now - self.since >= self.seconds:
            self.emitted = True
            return True
        return False

    def elapsed(self, now):
        return max(0, now - self.since) if self.since is not None else 0


def posture_signal(landmarks, face_box):
    """Head/shoulder position in image coordinates. Never invent occluded hips/legs."""
    if not landmarks or len(landmarks) < 25:
        return None
    nose, left, right = [landmarks[i] for i in (0, 11, 12)]
    if min(p[3] for p in (nose, left, right)) < 0.6:
        return None
    if abs(left[0] - right[0]) < 0.035:
        return None
    return {"nose_y": nose[1], "shoulder_y": (left[1] + right[1]) / 2,
            "shoulder_x": (left[0] + right[0]) / 2,
            "scale": max(face_box[3], 0.04), "shoulder_width": abs(left[0] - right[0])}


def classify_posture(baseline, current):
    if baseline is None:
        return "uncalibrated"
    if current is None:
        return "insufficient"
    # Large lateral/scale changes invalidate a seat-relative comparison.
    ratio = current["shoulder_width"] / baseline["shoulder_width"]
    if not 0.65 < ratio < 1.5 or abs(current["shoulder_x"] - baseline["shoulder_x"]) > 0.18:
        return "insufficient"
    rise_head = baseline["nose_y"] - current["nose_y"]
    rise_shoulders = baseline["shoulder_y"] - current["shoulder_y"]
    threshold = max(0.07, baseline["scale"] * 0.75)
    if rise_head > threshold and rise_shoulders > threshold:
        return "standing"
    return "seated"
