# Firebase sync setup

Project ID: `exam-proctoring-unified`. The app uses Firebase Authentication (email/password) for staff sign-in, Firebase Admin Python SDK to validate sessions, and mirrors application records to Cloud Firestore/upload evidence to Cloud Storage. SQLite remains the local source of truth; cloud sync runs in the background.

## Collections and bucket paths

- `students`: ID, name, face embeddings and enrollment time.
- `sessions`: title and start/end timestamps.
- `alerts`: type, track, details, time, review state and Storage object path.
- `transcripts`: recognized text and time offsets.
- `observations`: track, display name, emotion, posture and time.
- `settings`: saved app settings.
- `evidence/{session_id}/{filename}` in Storage: JPG snapshots and WebM clips.

On startup, existing SQLite rows are queued for initial backfill. New writes are mirrored in the background. Evidence uploads wait until the local file is ready.

## Console setup

1. Open the [Firebase project](https://console.firebase.google.com/project/exam-proctoring-unified/overview).
2. Create a Cloud Firestore database and enable Cloud Storage.
3. Copy the exact Storage bucket name from Firebase Console > Storage.
4. In Project settings > Service accounts, create/download a service-account JSON key for local development. Use a dedicated account with only the required Firestore/Storage permissions.
5. Put the key at `secrets/firebase-service-account.json` (git-ignored), or configure its actual path below.
6. Go to Authentication > Sign-in method and enable **Email/Password**.
7. In Project settings > General > Your apps, register a Web app if needed and copy its Web API key.

## Local configuration

Set these PowerShell variables before starting the backend, replacing the bucket name and key path:

```powershell
$env:FIREBASE_PROJECT_ID = 'exam-proctoring-unified'
$env:FIREBASE_STORAGE_BUCKET = 'exam-proctoring-unified.appspot.com'
$env:FIREBASE_WEB_API_KEY = 'your-firebase-web-api-key'
$env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\path\to\firebase-service-account.json'
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

Set the same variables in the PowerShell window before starting the app. `/api/auth/config` reports whether sign-in is ready. Firebase Authentication creates accounts and checks passwords; ExamWatch keeps only an 8-hour, HttpOnly session cookie and does not store passwords. Protected API requests require that signed session. Without the Web API key or service account, the login screen explains which setup is missing. If Firebase sync is unavailable after sign-in, local writes continue and status reports `degraded` with pending work/error details.

## Data handling

Enabling sync sends face embeddings, student names/IDs, observations, alerts, transcript text, settings and evidence media to this Firebase project. Admin SDK calls bypass client Security Rules, so protect the service-account key and restrict IAM roles. Never commit the JSON key.
