"""Firebase Authentication configuration shared with the cloud sync client."""
from __future__ import annotations

import os
from pathlib import Path


def firebase_admin_app():
    project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
    credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not project_id or not credential_path or not Path(credential_path).is_file():
        return None
    import firebase_admin
    name = f"exam-proctoring-{project_id}"
    try:
        return firebase_admin.get_app(name)
    except ValueError:
        from firebase_admin import credentials
        return firebase_admin.initialize_app(
            credentials.Certificate(credential_path), {"projectId": project_id}, name=name
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
