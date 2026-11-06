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
from .otp import OtpStore, PENDING_COOKIE, CODE_TTL_SECONDS, two_factor_enabled, mail_configured, mask_email


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
    role: str = Field("student", pattern=r"^(student|proctor)$")


class EvidenceSettings(BaseModel):
    image: bool = True
    video: bool = True
    audio: bool = True


class DetectionSettings(BaseModel):
    sensitivity: int = Field(strict=True, ge=0, le=100)
    standing_threshold_seconds: int = Field(strict=True, ge=1, le=300)
    proctor_dwell_seconds: int = Field(60, strict=True, ge=10, le=1800)
    evidence: EvidenceSettings = EvidenceSettings()


class Review(BaseModel):
    category: str = Field(min_length=1, max_length=60)
    decision: str = Field(pattern=r"^(violation|warning|dismissed)$")
    note: str = Field("", max_length=2000)
    reviewer: str = Field(min_length=1, max_length=100)
    signature: str | None = Field(None, max_length=400_000)


class AlertSettings(BaseModel):
    standing_threshold_seconds: int = Field(strict=True,ge=1,le=300)


class FirebaseToken(BaseModel):
    id_token: str = Field(min_length=20, max_length=10000)


class OtpCode(BaseModel):
    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


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


def create_app(settings=None, runtime=None, otp_store=None):
    settings = settings or Settings.load()
    runtime = runtime or Runtime(settings)
    otp_store = otp_store or OtpStore()
    secure_cookie = lambda: os.getenv("EXAMWATCH_COOKIE_SECURE", "") == "1"

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
        public = {"/api/auth/config", "/api/auth/session", "/api/auth/demo-session", "/api/auth/logout",
                  "/api/auth/otp/verify", "/api/auth/otp/resend"}
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
        config = auth_configuration()
        config["two_factor"] = two_factor_enabled() and not config.get("demo")
        return config

    def set_session_cookie(response: Response, session_cookie: str):
        response.set_cookie("examwatch_session", session_cookie, max_age=8 * 60 * 60,
                            httponly=True, secure=secure_cookie(), samesite="strict", path="/")

    def clear_pending_cookie(response: Response):
        response.delete_cookie(PENDING_COOKIE, path="/api/auth/otp", httponly=True,
                               secure=secure_cookie(), samesite="strict")

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
            claims = auth.verify_id_token(body.id_token, app=firebase_app, check_revoked=True)
            session_cookie = auth.create_session_cookie(body.id_token, expires_in=timedelta(hours=8), app=firebase_app)
        except Exception as exc:
            raise HTTPException(401, "Không xác thực được tài khoản Firebase.") from exc
        if not two_factor_enabled():
            set_session_cookie(response, session_cookie)
            return {"ok": True}
        email = (claims.get("email") or "").strip()
        if not email:
            raise HTTPException(400, "Tài khoản không có email để nhận mã xác thực.")
        if not mail_configured():
            raise HTTPException(503, "Xác thực 2 lớp đã bật nhưng chưa cấu hình dịch vụ gửi email.")
        try:
            token = otp_store.start(claims.get("uid", ""), email, session_cookie)
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc
        response.delete_cookie("examwatch_session", path="/")
        response.set_cookie(PENDING_COOKIE, token, max_age=CODE_TTL_SECONDS, httponly=True,
                            secure=secure_cookie(), samesite="strict", path="/api/auth/otp")
        return {"ok": True, "otp_required": True, "email": mask_email(email)}

    @app.post("/api/auth/otp/verify")
    def verify_otp(body: OtpCode, request: Request, response: Response):
        token = request.cookies.get(PENDING_COOKIE)
        try:
            session_cookie = otp_store.verify(token, body.code)
        except PermissionError as exc:
            failed = JSONResponse({"detail": str(exc), "restart": True}, status_code=401)
            clear_pending_cookie(failed)
            return failed
        set_session_cookie(response, session_cookie)
        clear_pending_cookie(response)
        return {"ok": True}

    @app.post("/api/auth/otp/resend")
    def resend_otp(request: Request):
        try:
            item = otp_store.resend(request.cookies.get(PENDING_COOKIE))
        except PermissionError as exc:
            failed = JSONResponse({"detail": str(exc), "restart": True}, status_code=401)
            clear_pending_cookie(failed)
            return failed
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"ok": True, "email": mask_email(item.email)}

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
                runtime.store.add_student(body.student_id, body.name.strip(), vectors, body.role)
            except sqlite3.IntegrityError as exc:
                raise HTTPException(409, "Mã thí sinh đã tồn tại.") from exc
            return {"id": body.student_id, "name": body.name.strip(), "samples": len(vectors), "role": body.role}

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
        return {"session": row, "attendance": runtime.store.attendance(session_id),
                "alerts": runtime.store.records(session_id, "alerts"),
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

    @app.get("/api/settings/detection")
    def detection_settings():
        return runtime.detection_settings()

    @app.post("/api/settings/detection")
    def save_detection(body: DetectionSettings):
        return runtime.configure_detection(body.sensitivity, body.evidence.model_dump(),
                                           body.proctor_dwell_seconds, body.standing_threshold_seconds)

    @app.get("/api/alerts")
    def search_alerts(q: str = "", kind: str = "", status: str = "", session_id: str = "",
                      date_from: str = "", date_to: str = "", limit: int = Query(500, ge=1, le=5000)):
        return runtime.store.search_alerts(q.strip()[:100], kind, status, session_id, date_from, date_to, limit)

    @app.get("/api/alerts/{alert_id}")
    def alert_detail(alert_id: str):
        row = runtime.store.alert(alert_id)
        if not row:
            raise HTTPException(404, "Không tìm thấy cảnh báo.")
        session_row = runtime.store.session(row["session_id"]) or {}
        return {**runtime.store._evidence_fields(row), "session_title": session_row.get("title", "")}

    @app.post("/api/alerts/{alert_id}/review")
    def review(alert_id: str, body: Review, request: Request):
        signature = None
        if body.signature:
            prefix = "data:image/png;base64,"
            if not body.signature.startswith(prefix):
                raise ValueError("Chữ ký không hợp lệ.")
            try:
                png = base64.b64decode(body.signature[len(prefix):], validate=True)
            except Exception as exc:
                raise ValueError("Chữ ký không hợp lệ.") from exc
            if not png.startswith(b"\x89PNG"):
                raise ValueError("Chữ ký không hợp lệ.")
            import uuid
            signature = uuid.uuid4().hex + ".png"
            (settings.data / "evidence" / signature).write_bytes(png)
        account = getattr(request.state, "user", {}) or {}
        reviewer = body.reviewer.strip()
        if account.get("email") and account["email"] not in reviewer:
            reviewer = f"{reviewer} ({account['email']})"
        row = runtime.store.review(alert_id, body.category.strip(), body.decision, body.note.strip(), reviewer, signature)
        if not row:
            raise HTTPException(404, "Không tìm thấy cảnh báo.")
        return runtime.store._evidence_fields(row)

    def evidence_response(name, media):
        path = runtime.store.evidence_file(name)
        if path is None or not path.is_file():
            raise HTTPException(404, "Không tìm thấy tệp bằng chứng.")
        return FileResponse(path, media_type=media)

    @app.get("/api/alerts/{alert_id}/thumb")
    def thumb(alert_id: str):
        row = runtime.store.alert(alert_id) or {}
        name = row.get("thumbnail") or (row.get("image") if str(row.get("image", "")).endswith(".jpg") else None)
        return evidence_response(name, "image/jpeg")

    @app.get("/api/alerts/{alert_id}/signature")
    def signature(alert_id: str):
        return evidence_response((runtime.store.alert(alert_id) or {}).get("signature"), "image/png")

    @app.get("/api/transcripts")
    def transcripts(q: str = "", session_id: str = "", speaker: str = "", limit: int = Query(500, ge=1, le=5000)):
        return runtime.store.search_transcripts(q.strip()[:100], session_id, limit, speaker)

    @app.get("/api/speakers")
    def speakers():
        return runtime.store.speakers()

    @app.get("/api/transcripts/{transcript_id}/audio")
    def transcript_audio(transcript_id: str):
        path = runtime.store.transcript_audio(transcript_id)
        return evidence_response(path.name if path else None, "audio/wav")

    @app.get("/api/sessions/{session_id}/emotions")
    def emotions(session_id: str, student_id: str = "", bucket: int = Query(10, ge=1, le=600)):
        if not runtime.store.session(session_id):
            raise HTTPException(404, "Không tìm thấy phiên thi.")
        return runtime.store.mood_series(session_id, student_id, bucket)

    @app.get("/api/sessions/{session_id}/report")
    def report(session_id: str):
        row = runtime.store.session(session_id)
        if not row:
            raise HTTPException(404, "Không tìm thấy phiên thi.")
        alerts = runtime.store.search_alerts(session_id=session_id, limit=5000)
        by_kind = {}
        for a in alerts:
            by_kind[a["kind"]] = by_kind.get(a["kind"], 0) + 1
        confirmed = sum(a.get("decision") == "violation" for a in alerts)
        open_count = sum(not a["acknowledged"] for a in alerts)
        status = "violation" if confirmed else "pending" if open_count else "normal"
        return {"session": row, "attendance": runtime.store.attendance(session_id),
                "alerts": {"total": len(alerts), "open": open_count, "reviewed": len(alerts) - open_count,
                           "confirmed": confirmed, "by_kind": by_kind},
                "incidents": [a for a in alerts if a["acknowledged"]],
                "status": status}

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
