# Vercel + Railway deployment

The web client is static and deploys to Vercel. FastAPI, model inference, SQLite, and evidence files run in one always-on Railway service. Keep one replica: the active capture session lives in process memory, and SQLite/evidence live on the mounted volume.

## 1. Prepare Firebase

Follow [FIREBASE_SETUP.md](FIREBASE_SETUP.md). Before deployment, have these ready:

- Firebase Web API key, with Email/Password enabled in Firebase Authentication.
- Service-account JSON with the required Firestore and Storage access.
- Upload `models/phone_detector_v5.pt` to Firebase Storage as `models/phone_detector_v5.pt`.

The service-account JSON is a Railway secret variable, never a repository file.

## 2. Create the Railway backend

Create a service from this GitHub repository. Railway detects the root `Dockerfile`. Attach one persistent volume mounted at `/data`, then set:

| Variable | Value |
| --- | --- |
| `EXAMWATCH_DATA` | `/data` |
| `EXAMWATCH_MODELS` | `/data/models` |
| `EXAMWATCH_WHISPER` | `zipformer-vi-30m` |
| `EXAMWATCH_THREADS` | `2` |
| `EXAMWATCH_COOKIE_SECURE` | `1` |
| `FIREBASE_PROJECT_ID` | `exam-proctoring-unified` |
| `FIREBASE_STORAGE_BUCKET` | `exam-proctoring-unified.firebasestorage.app` |
| `FIREBASE_WEB_API_KEY` | Firebase Web API key |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Full service-account JSON as a secret |
| `EXAMWATCH_PHONE_MODEL_GCS_PATH` | `models/phone_detector_v5.pt` |
| `EXAMWATCH_ALLOWED_ORIGINS` | Exact Vercel production origin, added after step 3 |
| `EXAMWATCH_TRUSTED_HOSTS` | `*.up.railway.app,*.vercel.app` |
| `EXAMWATCH_2FA_EMAIL` | `1` to require a 6-digit email code after every sign-in (password or Google) |
| `EXAMWATCH_MAIL_WEBHOOK_URL` | Gmail relay: `/exec` URL of `scripts/gmail_relay.gs` deployed as an Apps Script web app (see the setup notes in that file). Railway blocks outbound SMTP on Trial/Hobby, so mail goes over HTTPS |
| `EXAMWATCH_MAIL_WEBHOOK_SECRET` | Secret printed by the script's `setup` function |
| `BREVO_API_KEY` | Alternative to the Gmail relay: Brevo transactional email API key (secret) |
| `EXAMWATCH_MAIL_FROM` | With Brevo: sender address verified in Brevo (Senders & IP → Senders) |
| `EXAMWATCH_MAIL_NAME` | Optional sender name, default `ExamWatch` |

| `EXAMWATCH_MAX_PEOPLE` | Optional, default `25`: people tracked per frame |
| `EXAMWATCH_POSE_ENGINE` | Optional, default `auto` (YOLO pose when more than 3 people, else MediaPipe); `yolo` or `mediapipe` to force |
| `EXAMWATCH_POSE_IMGSZ` | Optional, default `800`: YOLO pose input size; lower (640) is faster, higher sees farther rows |
| `EXAMWATCH_FRAME_WIDTH` / `EXAMWATCH_DETECTOR_WIDTH` | Optional, defaults `1280` / `960`: camera frame and face-detector width for a full classroom |

| `EXAMWATCH_PHONE_INTERVAL` | Optional, default `0.5`: seconds between phone-detector passes (runs beside face/pose work; boxes reused in between) |

Speaker attribution: each transcript line is assigned to the person whose mouth region moved most (relative to their eye band) while it was spoken, when that person is clearly ahead of the next; otherwise it stays "not identified". Tiredness uses MediaPipe face blendshapes (`face_landmarker.task`, fetched at startup): eyes closed in at least 35 % of samples over a minute (PERCLOS) or two yawns, or a long bowed head.

Classroom scale: with 25 people the backend runs one YOLO pose pass for every body and refreshes at most 4 face identities and 6 emotions per frame (round robin), so a 2 vCPU service analyses roughly 1–2 frames per second. Register proctors on the Students page with the role "Giám thị" so they are excluded from alerts and attendance and reminded after standing beside one desk longer than the configured time.

Google sign-in: enable the Google provider in Firebase Authentication → Sign-in method, and add every Vercel domain the app is served from to Authentication → Settings → Authorized domains.

Set the Railway healthcheck path to `/healthz`; leave the service on one replica. The container prepares the public face, pose, and speech models into `/data/models` during startup. It listens on Railway's injected `PORT`.

After enabling a Railway public domain, keep its HTTPS URL for the next step. Confirm that `https://<railway-domain>/healthz` responds with `{"status":"ok"}`.

## 3. Configure and deploy Vercel

From the repository root, generate the static-site and API proxy config with the Railway HTTPS origin:

```powershell
python scripts/configure_vercel.py https://<railway-domain>
```

Commit and push the generated `vercel.json`. Import `hieu22012003/exam-proctoring-unified` into Vercel and deploy the `main` branch. The config publishes `web/`, rewrites `/static/*` to the static files, and proxies `/api/*` to Railway while keeping the browser on one origin so the secure session cookie works.

Copy the resulting Vercel production URL into Railway's `EXAMWATCH_ALLOWED_ORIGINS` (include `https://`, no trailing slash), then redeploy the backend. Add any custom production domain to that variable too.

## Costs and storage

Railway Hobby is a $5/month minimum that includes $5 of resource usage; it is not a usage cap. Railway meters memory, CPU, volume storage, and outbound traffic. At current rates, one continuously used GB of memory is about $10.14 per 30-day month before CPU, storage, and egress; two GB is about $20.27. The app's actual use depends on inference workload and traffic. Hobby volumes can hold up to 5 GB, so monitor evidence growth and usage in Railway.

Vercel serves only the static client and proxies API requests; its rewrite feature keeps the browser URL unchanged. SQLite and evidence must remain on Railway's mounted volume.

## Updates

Push code changes to GitHub. Railway rebuilds the backend from the Dockerfile and Vercel rebuilds the static client from `main`. When the Railway domain changes, rerun `scripts/configure_vercel.py` and update both `EXAMWATCH_ALLOWED_ORIGINS` and `EXAMWATCH_TRUSTED_HOSTS`.


## Render free (two services)

A single 512 MB free instance cannot hold every model, so `render.yaml` defines two free services:

| Service | Dockerfile | Holds | Measured peak |
|---|---|---|---|
| `examwatch-api` | `Dockerfile.render` | API, login/2FA, data, face detection + recognition, emotions, speech | ~400 MB |
| `examwatch-vision` | `Dockerfile.vision` | MediaPipe pose, face blendshapes, int8 ONNX phone detector (`models/phone_detector_v5.onnx`, 640 px) | ~400 MB + blendshapes |

No PyTorch is installed on either. The API sends each frame to the vision service (`EXAMWATCH_VISION_URL`, shared `EXAMWATCH_VISION_TOKEN`); if it is unreachable, faces, attendance and emotions keep working without poses. Lite mode (`EXAMWATCH_LITE=1`) tracks up to 3 people.

Free services have no disk: on every start the API rebuilds its database from Firestore (`EXAMWATCH_RESTORE_FROM_FIREBASE=1`, default). Evidence images/videos are not restored unless Firebase Storage is enabled. Free services sleep after 15 minutes without traffic (about a minute to wake), have 0.1 CPU, and share 750 instance hours per month per workspace.

Deploy: Render Dashboard -> New -> Blueprint -> this repository; enter the secrets; after `examwatch-vision` is live, set `EXAMWATCH_VISION_URL` on `examwatch-api` to its public URL. Then point `vercel.json` at the API's `onrender.com` URL (`python scripts/configure_vercel.py https://examwatch-api.onrender.com`).
