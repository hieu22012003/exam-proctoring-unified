/* Live body pose in the browser, so boxes and skeletons follow the live video.
 *
 * The server analyses about one frame per second on a small host, so its boxes
 * lag the live picture. MediaPipe Pose Lite runs here at up to 15 FPS on the
 * camera stream; app.js moves each server box with the matching person's nose
 * and draws the live skeleton. Labels and alerts still come from the server.
 * Any failure (no WebGL, CDN blocked, old browser) leaves `ready` false and the
 * dashboard keeps drawing the server results as before. */
const VERSION = '0.10.21';
const BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/vision_bundle.mjs`;
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const MIN_INTERVAL_MS = 66; // ~15 FPS is plenty for drawing and keeps laptops cool

const state = { ready: false, failed: null, loading: null, landmarker: null, poses: [], at: 0, lastTs: -1, ms: 0,
  chains: [], exits: [], slots: [], nextSlot: 1 };

/* Steady overlay (customer test 2026-11-11: boxes and skeleton looked jittery).
 * Pose Lite landmarks shake by a few pixels from frame to frame even when the
 * person sits still. Each person's landmarks go through a One Euro filter: strong
 * smoothing while still, almost none during a fast move, so the skeleton neither
 * shimmers nor trails behind. app.js then eases what it draws on every screen
 * refresh (settle), which also turns the server's 3-4 box updates per second
 * into a glide instead of jumps. */
const EURO = { minCutoff: 1.2, beta: 6.0, dCutoff: 1.0 };
const SLOT_GATE = 0.15, SLOT_KEEP_MS = 500, VIS_ALPHA = 0.35;

const lowpass = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
function euro(f, value, dt) {
  if (f.x === undefined) { f.x = value; f.v = 0; return value; }
  f.v += lowpass(EURO.dCutoff, dt) * ((value - f.x) / dt - f.v);
  f.x += lowpass(EURO.minCutoff + EURO.beta * Math.abs(f.v), dt) * (value - f.x);
  return f.x;
}

/** One Euro smoothing per person; each returned pose carries a stable `slot` id. */
function smooth(poses, now) {
  state.slots = state.slots.filter((s) => now - s.at <= SLOT_KEEP_MS);
  const pairs = [];
  poses.forEach((p, pi) => state.slots.forEach((s, si) => {
    const d = Math.hypot(p[0][0] - s.x, p[0][1] - s.y);
    if (d < SLOT_GATE) pairs.push([d, pi, si]);
  }));
  pairs.sort((a, b) => a[0] - b[0]);
  const slotOf = new Map(), taken = new Set();
  for (const [, pi, si] of pairs) {
    if (slotOf.has(pi) || taken.has(si)) continue;
    slotOf.set(pi, state.slots[si]); taken.add(si);
  }
  return poses.map((pose, pi) => {
    let slot = slotOf.get(pi);
    if (!slot) { slot = { id: state.nextSlot++, filters: [], vis: [], at: now }; state.slots.push(slot); }
    const dt = Math.min(0.25, Math.max(0.001, (now - slot.at) / 1000));
    slot.x = pose[0][0]; slot.y = pose[0][1]; slot.at = now;
    const out = pose.map((l, i) => {
      const f = slot.filters[i] || (slot.filters[i] = [{}, {}]);
      // A point coming back into view starts from where it is, not from where the
      // filter last saw it.
      if ((slot.vis[i] ?? 0) < 0.3) { f[0] = {}; f[1] = {}; }
      slot.vis[i] = slot.vis[i] === undefined ? l[3] : slot.vis[i] + VIS_ALPHA * (l[3] - slot.vis[i]);
      return [euro(f[0], l[0], dt), euro(f[1], l[1], dt), l[2], slot.vis[i]];
    });
    out.slot = slot.id;
    return out;
  });
}

const FACE_POINTS = [0, 2, 5, 7, 8]; // nose, eyes, ears

/** How far the head moved between the analysed frame (server landmarks) and the
 * live pose. The mean of the face points seen in both is used: the nose alone
 * swings sideways when the head turns, which slid the box off the face. */
function shift(serverLandmarks, pose) {
  let dx = 0, dy = 0, n = 0;
  for (const i of FACE_POINTS) {
    const a = serverLandmarks?.[i], b = pose?.[i];
    if (!a || !b || a[3] < 0.5 || b[3] < 0.5) continue;
    dx += b[0] - a[0]; dy += b[1] - a[1]; n += 1;
  }
  return n ? [dx / n, dy / n] : null;
}

const shown = new Map(); // track id -> what is on screen now
const TAU = { live: { pos: 0.06, size: 0.25, points: 0.04 }, server: { pos: 0.14, size: 0.25, points: 0.14 } };
const SIZE_DEADBAND = 0.004, SNAP = 0.25;

function easePoints(prev, next, k) {
  if (!prev || prev.length !== next.length) return next;
  return next.map((p, i) => {
    const q = prev[i];
    if (!q || (q[3] ?? 1) < 0.3 || Math.abs(p[0] - q[0]) > SNAP || Math.abs(p[1] - q[1]) > SNAP) return p;
    return [q[0] + k * (p[0] - q[0]), q[1] + k * (p[1] - q[1]), ...p.slice(2)];
  });
}

/** The track as it should be drawn now: box, skeleton, mouth and lean line glide
 * towards the latest result instead of jumping to it. */
function settle(track, now = performance.now()) {
  const tau = track.live ? TAU.live : TAU.server;
  const old = shown.get(track.id);
  const [x, y, w, h] = track.bbox;
  if (!old || now - old.at > 600 || now < old.at || Math.abs(x - old.bbox[0]) > SNAP || Math.abs(y - old.bbox[1]) > SNAP) {
    const first = { at: now, bbox: [x, y, w, h], landmarks: track.landmarks, mouth: track.mouth_points, lean: track.lean_target };
    shown.set(track.id, first);
    return track;
  }
  const dt = Math.min(0.25, (now - old.at) / 1000);
  const k = (t) => 1 - Math.exp(-dt / t);
  const b = old.bbox, kp = k(tau.pos), ks = k(tau.size);
  // The centre glides; the size only follows real changes (face-detector noise
  // made the frame "breathe").
  const nw = Math.abs(w - b[2]) < SIZE_DEADBAND ? b[2] : b[2] + ks * (w - b[2]);
  const nh = Math.abs(h - b[3]) < SIZE_DEADBAND ? b[3] : b[3] + ks * (h - b[3]);
  const cx = b[0] + b[2] / 2 + kp * (x + w / 2 - b[0] - b[2] / 2);
  const cy = b[1] + b[3] / 2 + kp * (y + h / 2 - b[1] - b[3] / 2);
  const next = {
    at: now, bbox: [cx - nw / 2, cy - nh / 2, nw, nh],
    landmarks: easePoints(old.landmarks, track.landmarks || [], k(tau.points)),
    mouth: easePoints(old.mouth, track.mouth_points || [], kp),
    lean: track.lean_target && old.lean ? easePoints([old.lean], [track.lean_target], k(tau.points))[0] : track.lean_target,
  };
  shown.set(track.id, next);
  if (shown.size > 64) for (const [id, s] of shown) if (now - s.at > 2000) shown.delete(id);
  return { ...track, bbox: next.bbox, landmarks: next.landmarks, mouth_points: next.mouth, lean_target: next.lean };
}
function unsettle() { shown.clear(); }

/* Heads that rose and left through the top of the frame.
 * Standing up in front of a close webcam takes the head out of the picture in
 * about a second; the server, at ~1 analysed frame per second, usually sees a
 * seated frame and then an empty seat. Following noses here at ~15 FPS shows
 * the rise itself. Leaning towards the camera also lifts the nose but widens
 * the shoulders, so that is not counted. */
const EXIT_TOP = 0.12, EXIT_RISE = 0.12, CHAIN_GATE = 0.15, KEEP_EXITS_S = 6;

function risen(c) {
  const last = c.path[c.path.length - 1];
  const low = c.path.reduce((a, q) => (q.y > a.y ? q : a), last);
  const widened = low.sw > 0 && last.sw > 1.4 * low.sw;
  return low.y - last.y >= EXIT_RISE && !widened;
}

function follow(poses, now) {
  const cur = poses.filter((p) => p[0] && p[0][3] >= 0.5).map((p) => ({
    x: p[0][0], y: p[0][1], sw: Math.abs((p[11]?.[0] ?? 0) - (p[12]?.[0] ?? 0)),
  }));
  const pairs = [];
  state.chains.forEach((c, ci) => cur.forEach((n, ni) => {
    const d = Math.hypot(n.x - c.x, n.y - c.y);
    if (d < CHAIN_GATE) pairs.push([d, ci, ni]);
  }));
  pairs.sort((a, b) => a[0] - b[0]);
  const usedC = new Set(), usedN = new Set(), next = [];
  for (const [, ci, ni] of pairs) {
    if (usedC.has(ci) || usedN.has(ni)) continue;
    usedC.add(ci); usedN.add(ni);
    const c = state.chains[ci], n = cur[ni];
    c.x = n.x; c.y = n.y;
    c.path = c.path.filter((q) => now - q.at <= 2500).concat([{ at: now, y: n.y, sw: n.sw }]);
    // Pose tracking can keep extrapolating a nose above the image for a moment.
    if (n.y < 0 && !c.exited && risen(c)) { c.exited = true; state.exits.push({ x: c.x, at: now }); }
    next.push(c);
  }
  state.chains.forEach((c, ci) => {
    if (usedC.has(ci)) return;
    const last = c.path[c.path.length - 1];
    if (now - last.at > 400) return;            // lost for a while: forget it
    if (last.y > EXIT_TOP) { next.push(c); return; }  // briefly missed, keep for a moment
    if (!c.exited && risen(c)) state.exits.push({ x: c.x, at: now });
  });
  cur.forEach((n, ni) => { if (!usedN.has(ni)) next.push({ x: n.x, y: n.y, path: [{ at: now, y: n.y, sw: n.sw }] }); });
  state.chains = next;
  state.exits = state.exits.filter((e) => now - e.at <= KEEP_EXITS_S * 1000);
}

/** Recent upward exits as [{x, age}] (age in seconds) for the server. */
function exits(now = performance.now()) {
  return state.exits.map((e) => ({ x: e.x, age: (now - e.at) / 1000 })).filter((e) => e.age <= KEEP_EXITS_S);
}

async function create(vision, files, numPoses, delegate) {
  return vision.PoseLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: 'VIDEO', numPoses,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
}

function init(numPoses = 5) {
  if (state.loading) return state.loading;
  state.loading = (async () => {
    try {
      const vision = await import(BUNDLE);
      const files = await vision.FilesetResolver.forVisionTasks(WASM);
      try { state.landmarker = await create(vision, files, numPoses, 'GPU'); }
      catch { state.landmarker = await create(vision, files, numPoses, 'CPU'); }
      // The first call compiles GPU shaders (~3 s); do it now on a blank frame
      // instead of freezing the first live frame.
      try {
        const blank = document.createElement('canvas');
        blank.width = blank.height = 64;
        blank.getContext('2d').fillRect(0, 0, 64, 64);
        state.landmarker.detectForVideo(blank, 0);
        state.lastTs = 0;
      } catch { /* warm-up is best effort */ }
      state.ready = true;
    } catch (error) {
      state.failed = String(error?.message || error);
      console.warn('[ExamWatch] live pose unavailable, using server overlay:', state.failed);
    }
    return state.ready;
  })();
  return state.loading;
}

/** Latest live poses as [[x, y, z, visibility] x 33] in video-normalised coordinates. */
function detect(video, now = performance.now()) {
  const width = video?.videoWidth ?? video?.width;
  if (!state.ready || !video || ('readyState' in video && video.readyState < 2) || !width) return [];
  if (now - state.at < MIN_INTERVAL_MS) return state.poses;
  const ts = Math.max(state.lastTs + 1, Math.round(now));
  let raw = [];
  try {
    const started = performance.now();
    const result = state.landmarker.detectForVideo(video, ts);
    state.ms = performance.now() - started;
    raw = (result.landmarks || []).map((pose) => pose.map((l) => [l.x, l.y, l.z ?? 0, l.visibility ?? 1]));
  } catch (error) {
    raw = [];
  }
  follow(raw, now);              // standing-up exits use the unfiltered positions
  state.poses = smooth(raw, now);
  state.lastTs = ts; state.at = now;
  return state.poses;
}

/** Pair server tracks with live poses by nose position (each pose used once). */
function match(tracks, poses, maxAge = 1.0) {
  const pairs = [];
  tracks.forEach((t, ti) => {
    const nose = t.landmarks?.[0];
    if (!nose || nose[3] < 0.5) return;
    const gate = Math.max(0.12, (t.bbox?.[2] || 0.1) * 1.6) * (1 + maxAge);
    poses.forEach((p, pi) => {
      if (!p[0] || p[0][3] < 0.5) return;
      const d = Math.hypot(p[0][0] - nose[0], p[0][1] - nose[1]);
      if (d < gate) pairs.push([d, ti, pi]);
    });
  });
  pairs.sort((a, b) => a[0] - b[0]);
  const byTrack = new Map(), used = new Set();
  for (const [, ti, pi] of pairs) {
    if (byTrack.has(ti) || used.has(pi)) continue;
    byTrack.set(ti, pi); used.add(pi);
  }
  return { byTrack, unused: poses.map((_, i) => i).filter((i) => !used.has(i)) };
}

const api = { init, detect, match, exits, follow, smooth, shift, settle, unsettle, get ready() { return state.ready; }, get failed() { return state.failed; }, get ms() { return state.ms; } };
if (typeof window !== 'undefined') window.ExamWatchLivePose = api;
export default api;
