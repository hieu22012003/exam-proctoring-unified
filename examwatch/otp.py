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


def _gmail_relay():
    url = os.getenv("EXAMWATCH_MAIL_WEBHOOK_URL", "").strip()
    secret = os.getenv("EXAMWATCH_MAIL_WEBHOOK_SECRET", "").strip()
    return (url, secret) if url and secret else None


def _brevo():
    key = os.getenv("BREVO_API_KEY", "").strip()
    sender = os.getenv("EXAMWATCH_MAIL_FROM", "").strip()
    return (key, sender) if key and sender else None


def mail_configured() -> bool:
    return bool(_gmail_relay() or _brevo())


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


def _message(code: str):
    minutes = CODE_TTL_SECONDS // 60
    subject = f"Mã đăng nhập ExamWatch: {code}"
    html_body = (
        "<div style=\"font-family:Arial,sans-serif;color:#1d2b36;max-width:420px\">"
        "<p>Mã xác thực đăng nhập ExamWatch của bạn là:</p>"
        f"<p style=\"font-size:32px;font-weight:700;letter-spacing:6px;color:#127986\">{html.escape(code)}</p>"
        f"<p>Mã có hiệu lực trong {minutes} phút. Nếu bạn không đăng nhập, hãy bỏ qua email này "
        "và đổi mật khẩu.</p></div>")
    text = f"Mã đăng nhập ExamWatch: {code} (hiệu lực {minutes} phút)."
    return subject, html_body, text


def _sender_name():
    return os.getenv("EXAMWATCH_MAIL_NAME", "ExamWatch").strip() or "ExamWatch"


def send_code_email(email: str, code: str):
    """Send the code through the Gmail Apps Script relay if configured, else Brevo."""
    if _gmail_relay():
        return _send_gmail_relay(email, code)
    if _brevo():
        return _send_brevo(email, code)
    raise RuntimeError("Chưa cấu hình dịch vụ gửi email (EXAMWATCH_MAIL_WEBHOOK_URL/SECRET hoặc BREVO_API_KEY).")


def _send_gmail_relay(email: str, code: str):
    url, secret = _gmail_relay()
    subject, html_body, text = _message(code)
    payload = {"secret": secret, "to": email, "subject": subject, "html": html_body,
               "text": text, "name": _sender_name()}
    # Apps Script answers the POST with a redirect to the result page; urllib follows it with GET.
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Không gửi được email xác thực qua Gmail ({exc.code}).") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Không kết nối được Google Apps Script gửi email.") from exc
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Apps Script trả về dữ liệu lạ. Kiểm tra lại quyền truy cập 'Anyone' khi deploy.") from exc
    if not result.get("ok"):
        raise RuntimeError(f"Gmail không gửi được email: {str(result.get('error', 'lỗi không rõ'))[:200]}")


def _send_brevo(email: str, code: str):
    api_key, sender = _brevo()
    subject, html_body, text = _message(code)
    body = {
        "sender": {"email": sender, "name": _sender_name()},
        "to": [{"email": email}],
        "subject": subject, "htmlContent": html_body, "textContent": text,
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
