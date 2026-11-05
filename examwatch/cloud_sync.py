"""Best-effort Firebase mirror; SQLite remains the local source of truth."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
from .auth import firebase_admin_app

logger = logging.getLogger(__name__)


class FirebaseSync:
    def __init__(self, evidence_dir: Path):
        self.evidence_dir = evidence_dir
        self.project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
        self.bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "").strip()
        self.db = self.bucket = None
        self.error = None
        self.queue = queue.Queue(maxsize=10000)
        self.stop_event = threading.Event()
        configured = bool(self.project_id and self.bucket_name and (
            os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()))
        if not configured:
            self.state = "not_configured"
            return
        try:
            from firebase_admin import firestore, storage

            app = firebase_admin_app()
            if app is None:
                raise RuntimeError("Firebase Admin credentials are not configured.")
            self.db = firestore.client(app=app)
            self.bucket = storage.bucket(app=app)
            self.state = "connected"
            self.worker = threading.Thread(target=self._run, name="firebase-sync", daemon=True)
            self.worker.start()
        except Exception as exc:
            self.state = "error"
            self.error = str(exc)
            logger.exception("Firebase initialization failed")

    def download_model(self, object_path: str, destination: Path):
        """Fetch one explicitly configured private model into the model directory."""
        from pathlib import PurePosixPath

        parts = PurePosixPath(object_path).parts
        if (not self.bucket or not object_path or object_path.startswith("/")
                or ".." in parts or "\\" in object_path):
            return False
        try:
            blob = self.bucket.blob(object_path)
            if not blob.exists(timeout=30):
                self.error = f"Firebase Storage model was not found: {object_path}"
                return False
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".download")
            blob.download_to_filename(str(temporary), timeout=300)
            if temporary.stat().st_size < 1_000_000 or temporary.stat().st_size > 500_000_000:
                temporary.unlink(missing_ok=True)
                self.error = "Firebase Storage model size is invalid."
                return False
            temporary.replace(destination)
            self.error = None
            return True
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Firebase model download failed")
            return False

    def enqueue(self, action, collection, document_id, payload=None):
        if self.state not in {"connected", "degraded"}:
            return
        try:
            self.queue.put_nowait((action, collection, str(document_id), payload, 0))
        except queue.Full:
            self.error = "Firebase sync queue is full."
            self.state = "degraded"
            logger.error(self.error)

    def upload_evidence(self, session_id, filename, alert_id):
        if self.state == "connected":
            self.enqueue("upload", "alerts", alert_id,
                         {"session_id": session_id, "filename": filename})

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                action, collection, doc_id, payload, attempt = self.queue.get(timeout=.5)
            except queue.Empty:
                continue
            try:
                if action == "set":
                    self.db.collection(collection).document(doc_id).set(payload, merge=True)
                elif action == "delete":
                    self.db.collection(collection).document(doc_id).delete()
                elif action == "upload":
                    path = self.evidence_dir / payload["filename"]
                    if not path.is_file():
                        if attempt < 24 and not path.with_suffix(".failed").exists():
                            time.sleep(.75)
                            self.queue.put((action, collection, doc_id, payload, attempt + 1))
                        continue
                    object_path = f"evidence/{payload['session_id']}/{path.name}"
                    blob = self.bucket.blob(object_path)
                    blob.upload_from_filename(path, content_type=(
                        "video/webm" if path.suffix.lower() == ".webm" else "image/jpeg"))
                    self.db.collection(collection).document(doc_id).set(
                        {"storage_path": object_path, "storage_sync": "uploaded"}, merge=True)
                self.state = "connected"
                self.error = None
            except Exception as exc:
                self.error = str(exc)
                self.state = "degraded"
                logger.exception("Firebase sync operation failed")
                if attempt < 3:
                    time.sleep(min(2 ** attempt, 8))
                    self.queue.put((action, collection, doc_id, payload, attempt + 1))
            finally:
                self.queue.task_done()

    def status(self):
        return {"state": self.state, "pending": self.queue.qsize(), "error": self.error}

    def close(self):
        if hasattr(self, "worker"):
            self.stop_event.set()
            self.worker.join(timeout=10)

