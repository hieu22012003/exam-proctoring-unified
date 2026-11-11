"""Customer test 2026-11-11 (video testAIexam.mp4).

1. Boxes and skeleton looked jittery -> One Euro filter + eased drawing (web/live-pose.js).
2. After one real phone the app kept "inventing" phones: boxes on the shirt front
   (38-50 %) and on a poster (39 %) were confirmed and re-reported every few seconds.
"""
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examwatch.rules import Persistence  # noqa: E402
from examwatch.runtime import Runtime  # noqa: E402
from examwatch.tracking import Track  # noqa: E402
from examwatch.vision import hand_near  # noqa: E402
from test_feedback_fixes import body  # noqa: E402
from test_vision_pipeline import FRAME, _Clips, _Store, make_vision, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Measured on the video: the false box on the shirt front and the real phone in the hand.
SHIRT = [.37, .73, .45, .12, .27]
HELD = [.31, .81, .40, .14, .15]


def person(wrists=None):
    t = Track(1, [.42, .15, .16, .35], 0)
    t.landmarks = body(wrists=wrists)
    return t


def test_box_on_the_shirt_has_no_hand():
    assert not hand_near(*SHIRT[:2], *SHIRT[3:], person())                       # arms out of the picture
    assert not hand_near(*SHIRT[:2], *SHIRT[3:], person(wrists=((.05, .98), (.95, .98))))
    off_frame = person(wrists=((.36, 1.2), (.6, 1.2)))                           # extrapolated below the image
    assert not hand_near(*SHIRT[:2], *SHIRT[3:], off_frame)


def test_phone_in_the_hand_counts():
    assert hand_near(*HELD[:2], *HELD[3:], person(wrists=((.26, .99), (.7, .9))))
    low = person(wrists=((.26, .99), (.7, .9)))
    low.landmarks[15][3] = .2                                                    # wrist not actually seen
    low.landmarks[16][3] = .2
    assert not hand_near(*HELD[:2], *HELD[3:], low)


def test_pipeline_ignores_the_shirt_but_reports_the_held_phone():
    v = make_vision()
    v.remote.poses = [body()]
    _, now = run(v, 2)
    v.remote.phones = [SHIRT]
    (tracks, _, phones), now = run(v, 6, fps=2, start=now)
    assert phones and phones[0]['counted'] is False
    assert not tracks[0].phone_seen and not tracks[0].has_phone
    v.remote.poses = [body(wrists=((.26, .99), (.7, .9)))]
    v.remote.phones = [HELD]
    (tracks, _, phones), now = run(v, 2.5, fps=2, start=now)
    assert phones[0]['counted'] and phones[0]['held'] and tracks[0].has_phone


def test_a_very_confident_box_counts_without_a_hand():
    v = make_vision()
    v.remote.poses = [body()]
    _, now = run(v, 2)
    v.remote.phones = [[.5, .8, .72, .06, .10]]          # e.g. a phone lying on the desk
    (tracks, _, phones), _ = run(v, 2.5, fps=2, start=now)
    assert phones[0]['counted'] and not phones[0]['held'] and tracks[0].has_phone


def test_scattered_boxes_do_not_keep_a_phone_alive():
    p = Persistence(1.0, max_gap=1.6)
    box = (.3, .8, .14, .15)
    for t in (0, .5, 1.0):
        p.hit(t, box)
    assert p.confirmed(1.0)
    first = p.run
    # Old rule: one box every 1.5 s kept "phone" confirmed for ever.
    p.miss(1.5); p.miss(2.0)
    assert not p.confirmed(2.0)
    p.hit(2.5, box)
    assert not p.confirmed(2.5) and p.run == first      # same place, short dropout: same episode
    p.miss(3.0); p.miss(3.5); p.hit(4.0, (.7, .3, .1, .1))
    assert not p.confirmed(4.0) and p.run == first + 1  # somewhere else: a different object
    p.hit(4.5, (.7, .3, .1, .1)); p.miss(5.0); p.hit(5.5, (.7, .3, .1, .1))
    assert p.confirmed(5.5)                              # one missed scan is tolerated


def test_slow_host_can_still_confirm_a_phone():
    """On Render frames (= scans) can be ~4 s apart; a fixed 1.6 s gap never confirmed."""
    v = make_vision()
    v.remote.poses = [body(wrists=((.45, .75), (.55, .75)))]
    _, now = run(v, 9, fps=.25)
    v.remote.phones = [[.5, .85, .5, .05, .09]]
    (tracks, _, _), _ = run(v, 9, fps=.25, start=now)
    assert tracks[0].has_phone


def test_one_alert_per_phone_episode():
    v = make_vision()
    v.remote.poses = [body(wrists=((.45, .75), (.55, .75)))]
    r = Runtime.__new__(Runtime)
    r.settings, r.vision, r.store, r.clips = v.settings, v, _Store(), _Clips()
    r.lock, r.session, r.owner, r.capture_id = threading.RLock(), {"id": "s"}, "o" * 8, "c"
    r.started, r.gallery, r.dwell, r.speaking = 0, [], {}, []
    r._snapshot = lambda frame, track, label: "x.jpg"
    r.settings.evidence_video = False
    r.settings.alert_cooldown = .3                       # old code: a new alert every cooldown
    v.remote.phones = [[.5, .85, .5, .05, .09]]
    for _ in range(12):
        v.phone_at = -1e9
        r.process(FRAME, "o" * 8, "c")
        time.sleep(.2)
    assert getattr(r.store, 'alerts', []).count('phone') == 1


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
def test_overlay_is_steady_and_keeps_up(tmp_path):
    module = tmp_path / 'live-pose.mjs'
    module.write_text((ROOT / 'web' / 'live-pose.js').read_text(encoding='utf-8'), encoding='utf-8')
    script = tmp_path / 'sim.mjs'
    script.write_text(r'''
const LP = (await import('./live-pose.mjs')).default;
let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
const PX = 1280, NOISE = 4 / PX;
const pose = (cx, noise) => Array.from({length: 33}, (_, i) => [cx + (i % 5) * .02 + gauss() * noise, .3 + (i % 7) * .03 + gauss() * noise, 0, .95]);
function scenario(speed) {
  LP.unsettle();
  const track = {id: 1, bbox: [.35, .2, .1, .2], landmarks: pose(.4, 0), mouth_points: [], lean_target: null};
  let rawPrev, raw = [], drawn = [], lag = [], prev, live = null;
  for (let ms = 0; ms < 8000; ms += 1000 / 60) {
    const now = 100000 + ms, truth = .4 + speed * ms / 1000;
    if (!live || ms - live.ms >= 66) {
      const p = pose(truth, NOISE);
      if (rawPrev !== undefined) raw.push((p[0][0] - rawPrev) * PX - speed * .066 * PX);
      rawPrev = p[0][0];
      live = {ms, pose: LP.smooth([p], now)[0]};
    }
    const d = LP.shift(track.landmarks, live.pose);
    const out = LP.settle({...track, bbox: [.35 + d[0], .2 + d[1], .1, .2], landmarks: live.pose, live: true}, now);
    if (ms > 1500) { if (prev !== undefined) drawn.push((out.bbox[0] - prev) * PX - speed / 60 * PX); lag.push((truth - out.landmarks[0][0]) * PX); }
    prev = out.bbox[0];
  }
  return {raw: rms(raw), drawn: rms(drawn), lagMs: speed ? lag.reduce((a, b) => a + b, 0) / lag.length / (speed * PX) * 1000 : 0};
}
// Server-only overlay: 3.5 box updates per second with 3 px of detector noise.
LP.unsettle(); let box = null, last, step = 0;
for (let ms = 0; ms < 8000; ms += 1000 / 60) {
  if (!box || ms - box.ms >= 285) box = {ms, b: [.35 + gauss() * 3 / PX, .2 + gauss() * 3 / PX, .1 + gauss() * 3 / PX, .2 + gauss() * 3 / PX]};
  const out = LP.settle({id: 2, bbox: box.b, landmarks: [], mouth_points: []}, 500000 + ms);
  if (last !== undefined) step = Math.max(step, Math.abs(out.bbox[0] - last) * PX);
  last = out.bbox[0];
}
// A head that turns: the nose swings 0.05 to the side, eyes and ears barely move.
const front = pose(.4, 0), turned = front.map((p) => [...p]);
turned[0][0] += .05;
console.log(JSON.stringify({still: scenario(0), fast: scenario(.5), serverStep: step, turn: LP.shift(front, turned)[0]}));
''', encoding='utf-8')
    result = json.loads(subprocess.run(['node', str(script)], capture_output=True, text=True, check=True, cwd=tmp_path).stdout)
    assert result['still']['raw'] > 4                    # the input really shakes (px per sample)
    assert result['still']['drawn'] < 0.6                # ...the drawing barely moves (px per frame)
    assert result['fast']['lagMs'] < 130                 # and still follows a fast move
    assert result['serverStep'] < 2                      # 3-4 FPS box updates glide, max px per frame
    assert result['turn'] < .015                         # old: the box slid by the full 0.05
