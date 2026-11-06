from contextlib import asynccontextmanager
from pathlib import Path
import base64
import json
import os
import sqlite3
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .config import Settings, ROOT
from .runtime import Runtime
from .auth import firebase_admin_app, auth_configuration


class StartSession(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    owner: str = Field(min_length=8, max_length=80)


class Owner(BaseModel):
    owner: str = Field(min_length=8, max_length=80)


class StopSession(BaseModel):
    # A local dashboard controls the displayed session; owner is a capture lease,
    # not authentication. Naming the session prevents a stale tab stopping a new one.
    session_id: str = Field(min_length=1, max_length=80)


class Enrollment(BaseModel):
    student_id: str = Field(min_length=1, max_length=40, pattern=r"^[\w-]+$")
    name: str = Field(min_length=1, max_length=100)
    images: list[str] = Field(min_length=3, max_length=8)


class AlertSettings(BaseModel):
    standing_threshold_seconds: int = Field(strict=True,ge=1,le=300)


class FirebaseToken(BaseModel):
    id_token: str = Field(min_length=20, max_length=10000)


def decode_frame(data, width):
    if len(data) > 1_500_000:
        raise ValueError("Ảnh vượt quá 1.5 MB.")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Ảnh không hợp lệ.")
    h, w = frame.shape[:2]
    if w > 4096 or h > 2160:
        raise ValueError("Ảnh đầu vào quá lớn.")
    if w > width:
        frame = cv2.resize(frame, (width, round(h * width / w)))
    return frame


def create_app(settings=None, runtime=None):
    settings = settings or Settings.load()
    runtime = runtime or Runtime(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        runtime.close()

    app = FastAPI(title="ExamWatch Local", lifespan=lifespan)
    app.state.runtime = runtime
    trusted_hosts = {"127.0.0.1", "localhost", "testserver"}
    trusted_hosts.update(host.strip() for host in os.getenv("EXAMWATCH_TRUSTED_HOSTS", "").split(",")
                         if host.strip())
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=sorted(trusted_hosts))

    @app.middleware("http")
    async def require_firebase_session(request: Request, call_next):
        path = request.url.path
        public = {"/api/auth/config", "/api/auth/session", "/api/auth/demo-session", "/api/auth/logout"}
        if path.startswith("/api/") and path not in public:
            cookie = request.cookies.get("examwatch_session")
            if not cookie:
                return JSONResponse({"detail": "Vui lòng đăng nhập để tiếp tục."}, status_code=401)
            if os.getenv("EXAMWATCH_DEMO_MODE", "").strip() == "1" and cookie == "local-demo-session":
                request.state.user = {"uid": "local-demo", "email": "demo@localhost"}
                return await call_next(request)
            try:
                from firebase_admin import auth
                firebase_app = firebase_admin_app()
                if firebase_app is None:
                    raise RuntimeError("Firebase Authentication chưa được cấu hình.")
                request.state.user = auth.verify_session_cookie(cookie, app=firebase_app, check_revoked=True)
            except Exception:
                return JSONResponse({"detail": "Phiên đăng nhập hết hạn. Vui lòng đăng nhập lại."}, status_code=401)
        return await call_next(request)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # Reject cross-site mutations, while allowing the production Vercel proxy origin.
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            request_origin = f"{request.url.scheme}://{request.url.netloc}".lower().rstrip("/")
            allowed_origins = {request_origin}
            allowed_origins.update(value.strip().lower().rstrip("/")
                                   for value in os.getenv("EXAMWATCH_ALLOWED_ORIGINS", "").split(",")
                                   if value.strip())
            if origin and origin.lower().rstrip("/") not in allowed_origins:
                return Response("Cross-origin request rejected", status_code=403)
            size = request.headers.get("content-length", "0")
            if not size.isdigit() or int(size) > 13_000_000:
                return Response("Request too large", status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(PermissionError)
    async def forbidden(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(BlockingIOError)
    async def busy(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=429)

    @app.get("/api/status")
    def status():
        return runtime.status()

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/api/auth/config")
    def auth_config():
        return auth_configuration()

    @app.post("/api/auth/demo-session")
    def create_demo_session(response: Response):
        if os.getenv("EXAMWATCH_DEMO_MODE", "").strip() != "1":
            raise HTTPException(404, "Chế độ mô phỏng chưa bật.")
        response.set_cookie("examwatch_session", "local-demo-session", max_age=2 * 60 * 60,
                            httponly=True, secure=os.getenv("EXAMWATCH_COOKIE_SECURE", "") == "1",
                            samesite="strict", path="/")
        return {"ok": True, "demo": True}

    @app.post("/api/auth/session")
    def create_auth_session(body: FirebaseToken, response: Response):
        from datetime import timedelta
        from firebase_admin import auth
        firebase_app = firebase_admin_app()
        if not firebase_app or not auth_configuration()["enabled"]:
            raise HTTPException(503, "Firebase Authentication chưa được cấu hình.")
        try:
            auth.verify_id_token(body.id_token, app=firebase_app, check_revoked=True)
            session_cookie = auth.create_session_cookie(body.id_token, expires_in=timedelta(hours=8), app=firebase_app)
        except Exception as exc:
            raise HTTPException(401, "Không xác thực được tài khoản Firebase.") from exc
        response.set_cookie("examwatch_session", session_cookie, max_age=8 * 60 * 60,
                            httponly=True, secure=os.getenv("EXAMWATCH_COOKIE_SECURE", "") == "1",
                            samesite="strict", path="/")
        return {"ok": True}

    @app.get("/api/auth/me")
    def auth_me(request: Request):
        user = request.state.user
        return {"uid": user.get("uid"), "email": user.get("email")}

    @app.post("/api/auth/logout")
    def auth_logout(response: Response):
        response.delete_cookie("examwatch_session", path="/", httponly=True,
                               secure=os.getenv("EXAMWATCH_COOKIE_SECURE", "") == "1",
                               samesite="strict")
        return {"ok": True}

    @app.post('/api/settings/alerts')
    def alert_settings(body: AlertSettings):
        return runtime.configure_standing(body.standing_threshold_seconds)

    @app.get("/api/students")
    def students():
        return runtime.store.students()

    @app.post("/api/students", status_code=201)
    def enroll(body: Enrollment):
        with runtime.lock:
            if runtime.session:
                raise ValueError("Hãy kết thúc phiên thi trước khi đăng ký thí sinh.")
            if not body.name.strip():
                raise ValueError("Tên thí sinh không được để trống.")
            vision = runtime.load_vision()
            vectors = []
            for raw in body.images:
                if len(raw) > 2_000_000:
                    raise ValueError("Ảnh đăng ký quá lớn.")
                try:
                    data = base64.b64decode(raw.split(",")[-1], validate=True)
                except Exception as exc:
                    raise ValueError("Ảnh đăng ký không hợp lệ.") from exc
                vectors.append(vision.enrollment(decode_frame(data, settings.frame_width)))
            # Reject enrollment samples that plainly belong to different people.
            similarities = np.asarray(vectors) @ np.asarray(vectors).T
            if float(similarities.min()) < settings.face_threshold:
                raise ValueError("Ảnh đăng ký không nhất quán. Hãy chụp lại cùng một người.")
            try:
                runtime.store.add_student(body.student_id, body.name.strip(), vectors)
            except sqlite3.IntegrityError as exc:
                raise HTTPException(409, "Mã thí sinh đã tồn tại.") from exc
            return {"id": body.student_id, "name": body.name.strip(), "samples": len(vectors)}

    @app.delete("/api/students/{student_id}")
    def delete_student(student_id: str):
        with runtime.lock:
            if runtime.session:
                raise ValueError("Hãy kết thúc phiên thi trước khi xóa thí sinh.")
            if not runtime.store.delete_student(student_id):
                raise HTTPException(404, "Không tìm thấy thí sinh.")
            return {"ok": True}

    @app.post("/api/sessions", status_code=201)
    def start(body: StartSession):
        if not body.title.strip():
            raise ValueError("Tên phiên thi không được để trống.")
        return runtime.start(body.title.strip(), body.owner)

    @app.post("/api/sessions/stop")
    def stop(body: StopSession):
        return runtime.stop(body.session_id)

    @app.post("/api/calibrate")
    def calibrate(body: Owner):
        return {"calibrated": runtime.calibrate(body.owner)}

    @app.get("/api/sessions")
    def sessions():
        return runtime.store.sessions()

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str):
        row = runtime.store.session(session_id)
        if not row:
            raise HTTPException(404, "Không tìm thấy phiên thi.")
        return {"session": row, "alerts": runtime.store.records(session_id, "alerts"),
                "transcripts": runtime.store.records(session_id, "transcripts"),
                "observations": runtime.store.records(session_id, "observations")}

    @app.get("/api/sessions/{session_id}/export")
    def export(session_id: str):
        row = runtime.store.session(session_id)
        if not row:
            raise HTTPException(404, "Không tìm thấy phiên thi.")
        payload = {"session": row}
        for table in ("alerts", "transcripts", "observations"):
            payload[table] = runtime.store.records(session_id, table, limit=10_000_000)
        return Response(json.dumps(payload, ensure_ascii=False), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="exam-{session_id}.json"'})

    @app.post("/api/frames")
    async def frame(request: Request, owner: str, capture_id: str):
        from starlette.concurrency import run_in_threadpool
        data = await request.body()
        image = decode_frame(data, settings.frame_width)
        return await run_in_threadpool(runtime.process, image, owner, capture_id)

    @app.post("/api/audio", status_code=202)
    async def audio(request: Request, owner: str, capture_id: str,
                    rate: int = Query(16000, ge=8000, le=96000),
                    offset: float = Query(0, ge=0, le=86400)):
        data = await request.body()
        if len(data) % 4 or not 0 < len(data) <= rate * 4 * 10:
            raise ValueError("Đoạn âm thanh PCM float32 cần dài từ 0 đến 10 giây.")
        samples = np.frombuffer(data, dtype="<f4").copy()
        if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1.01:
            raise ValueError("Dữ liệu âm thanh không hợp lệ.")
        runtime.queue_audio(samples, rate, offset, owner, capture_id)
        return {"queued": True}

    @app.post("/api/alerts/{alert_id}/acknowledge")
    def acknowledge(alert_id: str):
        if not runtime.store.acknowledge(alert_id):
            raise HTTPException(404, "Không tìm thấy cảnh báo.")
        return {"ok": True}

    @app.get("/api/alerts/{alert_id}/image")
    def image(alert_id: str):
        path = runtime.store.evidence(alert_id)
        if path is None or path.suffix != '.jpg' or not path.is_file():
            raise HTTPException(404, "Không tìm thấy ảnh.")
        return FileResponse(path, media_type="image/jpeg")

    @app.get('/api/alerts/{alert_id}/video')
    def video(alert_id: str):
        path = runtime.store.evidence(alert_id)
        if path is None or path.suffix != '.webm' or not path.is_file():
            raise HTTPException(404, 'Video chưa sẵn sàng hoặc không tồn tại.')
        return FileResponse(path, media_type='video/webm')

    static_dir = ROOT / "web"
    try:
        static_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    except Exception:
        pass
    
    @app.get("/")
    def index():
        return FileResponse(ROOT / "web" / "index.html")

    return app
