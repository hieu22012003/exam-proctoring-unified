"""Firebase Authentication configuration shared with the cloud sync client."""
from __future__ import annotations

import os
from pathlib import Path
import json


def firebase_admin_credential():
    """Load Admin credentials from a secret env var or a local ignored file."""
    from firebase_admin import credentials

    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        try:
            account = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON.") from exc
        if not isinstance(account, dict):
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON must contain a JSON object.")
        return credentials.Certificate(account)

    credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if credential_path and Path(credential_path).is_file():
        return credentials.Certificate(credential_path)
    return None


def firebase_admin_app():
    project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
    if not project_id:
        return None
    import firebase_admin
    credential = firebase_admin_credential()
    if credential is None:
        return None
    name = f"exam-proctoring-{project_id}"
    try:
        return firebase_admin.get_app(name)
    except ValueError:
        return firebase_admin.initialize_app(
            credential,
            {"projectId": project_id,
             **({"storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "").strip()}
                if os.getenv("FIREBASE_STORAGE_BUCKET", "").strip() else {})},
            name=name,
        )


def auth_configuration():
    if os.getenv("EXAMWATCH_DEMO_MODE", "").strip() == "1":
        return {"enabled": True, "demo": True, "api_key": "", "project_id": "local-demo"}
    api_key = os.getenv("FIREBASE_WEB_API_KEY", "").strip()
    project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
    admin_ready = False
    try:
        admin_ready = firebase_admin_app() is not None
    except Exception:
        admin_ready = False
    return {"enabled": bool(api_key and project_id and admin_ready),
            "api_key": api_key, "project_id": project_id}
