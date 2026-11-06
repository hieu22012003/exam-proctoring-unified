from dataclasses import dataclass, field
import numpy as np
from .rules import Sustained, AngleWatch


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    w = max(0, min(a[0] + a[2], b[0] + b[2]) - x)
    h = max(0, min(a[1] + a[3], b[1] + b[3]) - y)
    intersection = w * h
    return intersection / max(1e-9, a[2] * a[3] + b[2] * b[3] - intersection)


def identify(vector, gallery, threshold, margin):
    scores = []
    for person in gallery:
        score = max(float(np.dot(vector, sample)) for sample in person["embeddings"])
        scores.append((score, person))
    scores.sort(key=lambda r: r[0], reverse=True)
    if not scores or scores[0][0] < threshold:
        return None, 0.0
    if len(scores) > 1 and scores[0][0] - scores[1][0] < margin:
        return None, scores[0][0]
    return scores[0][1], scores[0][0]


@dataclass
class Track:
    id: int
    box: list
    last_seen: float
    student_id: str | None = None
    name: str = "Chưa xác định"
    identity_score: float = 0
    identity_at: float = -1e9
    emotion_at: float = -1e9
    emotion: str = "unknown"
    emotion_score: float = 0
    probabilities: object = None
    posture: str = "uncalibrated"
    baseline: dict | None = None
    baseline_owner: str | None = None
    signal: dict | None = None
    landmarks: list = field(default_factory=list)
    mouth_points: list = field(default_factory=list)
    standing_rule: Sustained = field(default_factory=lambda: Sustained(5))
    emotion_rule: Sustained = field(default_factory=lambda: Sustained(20))
    observed_at: float = -1e9
    left_arm_rule: AngleWatch = field(default_factory=lambda: AngleWatch(45.000001))
    right_arm_rule: AngleWatch = field(default_factory=lambda: AngleWatch(45.000001))
    lean_rule: AngleWatch = field(default_factory=lambda: AngleWatch(20, 2))
    angles: dict = field(default_factory=dict)
    has_phone: bool = False
    phone_confidence: float = 0.0
    phone_alert_at: float = -1e9
    head_turning: bool = False
    head_yaw: float = 0.0
    head_suspicion_since: float | None = None
    head_suspicion_last: float = -1e9
    head_suspicion_samples: int = 0
    head_alert_at: float = -1e9
    role: str = "student"
    tired: bool = False
    head_down_rule: Sustained = field(default_factory=lambda: Sustained(5))
    moving_rule: Sustained = field(default_factory=lambda: Sustained(3))
    tired_rule: Sustained = field(default_factory=lambda: Sustained(20))

    def clear_identity(self, preserve_baseline=False):
        self.student_id, self.name = None, "Chưa xác định"
        self.identity_score = 0
        self.identity_at = -1e9
        if not preserve_baseline:
            self.baseline = None
            self.baseline_owner = None
        self.probabilities = None
        self.emotion, self.emotion_score = "unknown", 0
        self.emotion_rule.reset()
        if not preserve_baseline:
            self.standing_rule.reset()
            self.left_arm_rule.reset()
            self.right_arm_rule.reset()
            self.lean_rule.reset()


def apply_thresholds(track, settings):
    track.standing_rule.seconds = settings.standing_seconds
    track.head_down_rule.seconds = settings.standing_seconds
    track.moving_rule.seconds = max(2.0, settings.standing_seconds / 2)
    track.emotion_rule.seconds = settings.negative_seconds
    track.left_arm_rule.threshold = track.right_arm_rule.threshold = settings.arm_threshold
    track.lean_rule.threshold = settings.lean_threshold


class Tracker:
    def __init__(self, settings):
        self.settings = settings
        self.tracks = {}
        self.counter = 0
        self.calibrations = {}

    def remember_calibrations(self):
        for track in self.tracks.values():
            if track.student_id and track.baseline and track.baseline_owner in (None, track.student_id):
                track.baseline_owner = track.student_id
                self.calibrations[track.student_id] = track.baseline.copy()

    def update(self, boxes, now):
        self.remember_calibrations()
        self.tracks = {k: t for k, t in self.tracks.items() if now - t.last_seen < 8.0}
        # Conservative overlap matching. Ambiguous matches become new unknown tracks.
        pairs = []
        for index, box in enumerate(boxes):
            candidates = sorted([(iou(box, t.box), t.id) for t in self.tracks.values()], reverse=True)
            if not candidates or candidates[0][0] <= .18:
                # A rising head can leave its previous bbox entirely. Use a
                # tightly gated centre/scale association for a calibrated track.
                candidates = []
                for track in self.tracks.values():
                    old = track.box
                    dx = abs(box[0]+box[2]/2-old[0]-old[2]/2)
                    dy = abs(box[1]+box[3]/2-old[1]-old[3]/2)
                    if (track.baseline and dx < .08 and dy < .25
                            and .65 < box[2]/max(old[2],1e-9) < 1.5
                            and .65 < box[3]/max(old[3],1e-9) < 1.5):
                        candidates.append((1-dx/.08*.5-dy/.25*.5,track.id))
                candidates.sort(reverse=True)
            if candidates and candidates[0][0] > 0.18:
                if len(candidates) == 1 or candidates[0][0] - candidates[1][0] > 0.12:
                    pairs.append((candidates[0][0], index, candidates[0][1]))
        matches, used = {}, set()
        for _, index, track_id in sorted(pairs, reverse=True):
            if track_id not in used:
                matches[index] = self.tracks[track_id]
                used.add(track_id)
        active = []
        for index, box in enumerate(boxes):
            if index in matches:
                track = matches[index]
            else:
                self.counter += 1
                track = Track(self.counter, box, now)
                apply_thresholds(track, self.settings)
                self.tracks[track.id] = track
            track.box, track.last_seen = box, now
            active.append(track)
        for t in self.tracks.values():
            if t not in active:
                t.emotion_rule.reset()
        return active
