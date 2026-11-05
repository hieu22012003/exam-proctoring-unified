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
| `FIREBASE_STORAGE_BUCKET` | `exam-proctoring-unified.appspot.com` |
| `FIREBASE_WEB_API_KEY` | Firebase Web API key |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Full service-account JSON as a secret |
| `EXAMWATCH_PHONE_MODEL_GCS_PATH` | `models/phone_detector_v5.pt` |
| `EXAMWATCH_ALLOWED_ORIGINS` | Exact Vercel production origin, added after step 3 |
| `EXAMWATCH_TRUSTED_HOSTS` | `*.up.railway.app,*.vercel.app` |

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
