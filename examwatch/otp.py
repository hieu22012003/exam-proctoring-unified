"""Email one-time codes used as the second sign-in factor.

A login is held as "pending" after Firebase verifies the password or Google
account. The Firebase session cookie is only handed to the browser once the
six-digit code sent to the account's email address has been entered.
Pending logins live in memory (the service runs a single worker); a restart
simply asks the user to sign in again.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

CODE_TTL_SECONDS = 10 * 60
RESEND_COOLDOWN_SECONDS = 60
MAX_ATTEMPTS = 5
MAX_SENDS = 5
PENDING_COOKIE = "examwatch_pending"
BREVO_URL = "https://api.brevo.com/v3/smtp/email"


def two_factor_enabled() -> bool:
    return os.getenv("EXAMWATCH_2FA_EMAIL", "").strip() == "1"


def mail_configured() -> bool:
    return bool(os.getenv("BREVO_API_KEY", "").strip() and os.getenv("EXAMWATCH_MAIL_FROM", "").strip())


def mask_email(email: str) -> str:
    name, _, domain = email.partition("@")
    if not domain:
        return "email của bạn"
    shown = name[:2] if len(name) > 2 else name[:1]
    return f"{shown}{'*' * max(len(name) - len(shown), 3)}@{domain}"


def _digest(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


@dataclass
class PendingLogin:
    uid: str
    email: str
    session_cookie: str
    code_hash: str
    expires_at: float
    sent_at: float
    attempts: int = 0
    sends: int = 1
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class OtpStore:
    def __init__(self, sender=None, clock=time.time):
        self._items: dict[str, PendingLogin] = {}
        self._lock = threading.Lock()
        self._send = sender or send_code_email
        self._clock = clock

    def _purge(self):
        now = self._clock()
        for key in [k for k, v in self._items.items() if v.expires_at < now]:
            del self._items[key]

    def start(self, uid: str, email: str, session_cookie: str) -> str:
        code = f"{secrets.randbelow(1_000_000):06d}"
        self._send(email, code)
        now = self._clock()
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._purge()
            self._items[token] = PendingLogin(uid, email, session_cookie, _digest(code),
                                              now + CODE_TTL_SECONDS, now)
        return token

    def get(self, token: str | None) -> PendingLogin | None:
        if not token:
            return None
        with self._lock:
            self._purge()
            return self._items.get(token)

    def verify(self, token: str | None, code: str) -> str:
        """Return the stored Firebase session cookie, or raise ValueError/PermissionError."""
        item = self.get(token)
        if item is None:
            raise PermissionError("Phiên xác thực đã hết hạn. Vui lòng đăng nhập lại.")
        with item.lock:
            if item.attempts >= MAX_ATTEMPTS:
                self.discard(token)
                raise PermissionError("Nhập sai quá nhiều lần. Vui lòng đăng nhập lại.")
            item.attempts += 1
            if not hmac.compare_digest(item.code_hash, _digest(code.strip())):
                left = MAX_ATTEMPTS - item.attempts
                if left <= 0:
                    self.discard(token)
                    raise PermissionError("Nhập sai quá nhiều lần. Vui lòng đăng nhập lại.")
                raise ValueError(f"Mã không đúng. Còn {left} lần thử.")
        self.discard(token)
        return item.session_cookie

    def resend(self, token: str | None) -> PendingLogin:
        item = self.get(token)
        if item is None:
            raise PermissionError("Phiên xác thực đã hết hạn. Vui lòng đăng nhập lại.")
        with item.lock:
            now = self._clock()
            wait = int(item.sent_at + RESEND_COOLDOWN_SECONDS - now)
            if wait > 0:
                raise BlockingIOError(f"Vui lòng đợi {wait} giây rồi gửi lại.")
            if item.sends >= MAX_SENDS:
                raise BlockingIOError("Đã gửi lại quá nhiều lần. Vui lòng đăng nhập lại sau.")
            code = f"{secrets.randbelow(1_000_000):06d}"
            self._send(item.email, code)
            item.code_hash, item.sent_at = _digest(code), now
            item.expires_at, item.attempts = now + CODE_TTL_SECONDS, 0
            item.sends += 1
        return item

    def discard(self, token: str | None):
        if token:
            with self._lock:
                self._items.pop(token, None)


def send_code_email(email: str, code: str):
    api_key = os.getenv("BREVO_API_KEY", "").strip()
    sender = os.getenv("EXAMWATCH_MAIL_FROM", "").strip()
    if not api_key or not sender:
        raise RuntimeError("Chưa cấu hình dịch vụ gửi email (BREVO_API_KEY, EXAMWATCH_MAIL_FROM).")
    minutes = CODE_TTL_SECONDS // 60
    body = {
        "sender": {"email": sender, "name": os.getenv("EXAMWATCH_MAIL_NAME", "ExamWatch").strip() or "ExamWatch"},
        "to": [{"email": email}],
        "subject": f"Mã đăng nhập ExamWatch: {code}",
        "htmlContent": (
            "<div style=\"font-family:Arial,sans-serif;color:#1d2b36;max-width:420px\">"
            "<p>Mã xác thực đăng nhập ExamWatch của bạn là:</p>"
            f"<p style=\"font-size:32px;font-weight:700;letter-spacing:6px;color:#127986\">{html.escape(code)}</p>"
            f"<p>Mã có hiệu lực trong {minutes} phút. Nếu bạn không đăng nhập, hãy bỏ qua email này "
            "và đổi mật khẩu.</p></div>"),
        "textContent": f"Mã đăng nhập ExamWatch: {code} (hiệu lực {minutes} phút).",
    }
    request = urllib.request.Request(
        BREVO_URL, data=json.dumps(body).encode(), method="POST",
        headers={"api-key": api_key, "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status >= 300:
                raise RuntimeError(f"Brevo trả về mã {response.status}.")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Không gửi được email xác thực ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Không kết nối được dịch vụ gửi email.") from exc
