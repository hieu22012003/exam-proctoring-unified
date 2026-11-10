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
  chains: [], exits: [] };

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
  try {
    const started = performance.now();
    const result = state.landmarker.detectForVideo(video, ts);
    state.ms = performance.now() - started;
    state.poses = (result.landmarks || []).map((pose) => pose.map((l) => [l.x, l.y, l.z ?? 0, l.visibility ?? 1]));
  } catch (error) {
    state.poses = [];
  }
  follow(state.poses, now);
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

const api = { init, detect, match, exits, follow, get ready() { return state.ready; }, get failed() { return state.failed; }, get ms() { return state.ms; } };
if (typeof window !== 'undefined') window.ExamWatchLivePose = api;
export default api;
