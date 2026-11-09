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

const state = { ready: false, failed: null, loading: null, landmarker: null, poses: [], at: 0, lastTs: -1, ms: 0 };

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

const api = { init, detect, match, get ready() { return state.ready; }, get failed() { return state.failed; }, get ms() { return state.ms; } };
if (typeof window !== 'undefined') window.ExamWatchLivePose = api;
export default api;
