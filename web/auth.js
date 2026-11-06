'use strict';
(() => {
  const gate = document.getElementById('auth-gate');
  const form = document.getElementById('auth-form');
  const message = document.getElementById('auth-message');
  const submit = document.getElementById('auth-submit');
  const toggle = document.getElementById('auth-toggle');
  const google = document.getElementById('auth-google');
  const divider = document.getElementById('auth-divider');
  const otpForm = document.getElementById('otp-form');
  const otpCode = document.getElementById('otp-code');
  const otpMessage = document.getElementById('otp-message');
  const otpSubmit = document.getElementById('otp-submit');
  const otpResend = document.getElementById('otp-resend');
  const FIREBASE_SDK = 'https://www.gstatic.com/firebasejs/10.14.1/';
  let config = null, registering = false, firebaseReady = null;

  const showMessage = (text, error = false) => { message.textContent = text; message.classList.toggle('error', error); };
  const showOtpMessage = (text, error = false) => { otpMessage.textContent = text; otpMessage.classList.toggle('error', error); };
  const ready = (demo = false) => { gate.hidden = true; document.getElementById('logout-button').hidden = false; window.dispatchEvent(new CustomEvent('examwatch-auth-ready',{detail:{demo}})); };
  const busy = (on) => { [submit, toggle, google].forEach((b) => { b.disabled = on; }); };
  const detailOf = async (response, fallback) => { try { return (await response.json()).detail || fallback; } catch { return fallback; } };
  const firebaseMessage = (code) => ({
    EMAIL_EXISTS: 'Email này đã được đăng ký.', INVALID_EMAIL: 'Email không hợp lệ.',
    INVALID_PASSWORD: 'Mật khẩu không đúng hoặc chưa đủ 6 ký tự.', WEAK_PASSWORD: 'Mật khẩu cần ít nhất 6 ký tự.',
    EMAIL_NOT_FOUND: 'Không tìm thấy tài khoản này.', INVALID_LOGIN_CREDENTIALS: 'Email hoặc mật khẩu không đúng.',
    OPERATION_NOT_ALLOWED: 'Firebase chưa bật phương thức đăng nhập Email/Mật khẩu.',
    TOO_MANY_ATTEMPTS_TRY_LATER: 'Thử quá nhiều lần. Vui lòng đợi rồi thử lại.'
  })[code] || 'Không thể xác thực. Kiểm tra cấu hình Firebase và thử lại.';
  const googleMessage = (code) => ({
    'auth/popup-closed-by-user': 'Bạn đã đóng cửa sổ đăng nhập Google.',
    'auth/cancelled-popup-request': 'Bạn đã đóng cửa sổ đăng nhập Google.',
    'auth/popup-blocked': 'Trình duyệt đã chặn cửa sổ đăng nhập. Hãy cho phép popup rồi thử lại.',
    'auth/unauthorized-domain': 'Tên miền này chưa được thêm vào Authorized domains của Firebase.',
    'auth/operation-not-allowed': 'Firebase chưa bật phương thức đăng nhập Google.',
    'auth/account-exists-with-different-credential': 'Email này đã đăng ký bằng mật khẩu. Hãy đăng nhập bằng email và mật khẩu.',
    'auth/network-request-failed': 'Lỗi mạng khi kết nối Google. Vui lòng thử lại.'
  })[code] || 'Không đăng nhập được bằng Google.';

  const showLogin = () => { otpForm.hidden = true; form.hidden = false; otpCode.value = ''; showOtpMessage(''); };
  const showOtp = (email) => {
    form.hidden = true; otpForm.hidden = false;
    document.getElementById('otp-description').textContent = `Mã 6 số đã được gửi tới ${email}. Mã có hiệu lực trong 10 phút.`;
    showOtpMessage(''); otpCode.value = ''; otpCode.focus();
  };

  // Exchange a Firebase ID token for a server session; may require an email code first.
  const completeSession = async (idToken) => {
    const session = await fetch('/api/auth/session', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id_token:idToken})});
    if (!session.ok) throw new Error(await detailOf(session, 'Không tạo được phiên đăng nhập.'));
    const result = await session.json();
    document.getElementById('auth-password').value = '';
    if (result.otp_required) { showMessage(''); showOtp(result.email); return; }
    showMessage('Đăng nhập thành công.'); ready();
  };

  const loadScript = (src) => new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src; script.onload = resolve;
    script.onerror = () => reject(new Error('Không tải được thư viện đăng nhập Google.'));
    document.head.appendChild(script);
  });
  const firebaseAuth = () => {
    firebaseReady ||= (async () => {
      await loadScript(FIREBASE_SDK + 'firebase-app-compat.js');
      await loadScript(FIREBASE_SDK + 'firebase-auth-compat.js');
      const app = window.firebase.initializeApp({apiKey: config.api_key, authDomain: `${config.project_id}.firebaseapp.com`, projectId: config.project_id});
      const auth = app.auth();
      await auth.setPersistence(window.firebase.auth.Auth.Persistence.NONE);
      return auth;
    })();
    firebaseReady.catch(() => { firebaseReady = null; });
    return firebaseReady;
  };

  toggle.addEventListener('click', () => {
    registering = !registering;
    document.getElementById('auth-title').innerHTML = registering ? 'Tạo tài khoản<span class="title-dot">.</span>' : 'Đăng nhập<span class="title-dot">.</span>';
    document.getElementById('auth-description').textContent = registering ? 'Tạo tài khoản giám thị bằng email.' : 'Đăng nhập để mở dashboard giám sát.';
    document.getElementById('auth-password').autocomplete = registering ? 'new-password' : 'current-password';
    submit.textContent = registering ? 'Đăng ký' : 'Đăng nhập';
    toggle.textContent = registering ? 'Đã có tài khoản? Đăng nhập' : 'Chưa có tài khoản? Đăng ký';
    showMessage('');
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!config?.enabled) { showMessage('Firebase chưa được cấu hình. Hãy thêm FIREBASE_WEB_API_KEY và service account rồi khởi động lại.', true); return; }
    busy(true); showMessage('Đang kết nối Firebase…');
    try {
      if (config.demo) {
        const response=await fetch('/api/auth/demo-session',{method:'POST'});
        if(!response.ok)throw new Error('Không mở được phiên mô phỏng.');
        document.getElementById('auth-password').value='';
        showMessage('Đã vào phiên mô phỏng với dữ liệu demo.'); ready(true); return;
      }
      const endpoint = registering ? 'signUp' : 'signInWithPassword';
      const response = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:${endpoint}?key=${encodeURIComponent(config.api_key)}`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({email: document.getElementById('auth-email').value.trim(), password: document.getElementById('auth-password').value, returnSecureToken: true})
      });
      const result = await response.json();
      if (!response.ok) throw new Error(firebaseMessage(result.error?.message?.split(' :')[0] || result.error?.message));
      if (config.two_factor) showMessage('Đang gửi mã xác thực tới email…');
      await completeSession(result.idToken);
    } catch (error) { showMessage(error.message, true); }
    finally { busy(false); }
  });

  google.addEventListener('click', async () => {
    if (!config?.enabled) { showMessage('Firebase chưa được cấu hình.', true); return; }
    busy(true); showMessage('Đang mở cửa sổ đăng nhập Google…');
    let auth = null;
    try {
      auth = await firebaseAuth();
      const provider = new window.firebase.auth.GoogleAuthProvider();
      provider.setCustomParameters({prompt: 'select_account'});
      const credential = await auth.signInWithPopup(provider);
      if (config.two_factor) showMessage('Đang gửi mã xác thực tới email…');
      await completeSession(await credential.user.getIdToken());
    } catch (error) {
      showMessage(error?.code ? googleMessage(error.code) : error.message, true);
    } finally {
      if (auth) auth.signOut().catch(() => {});
      busy(false);
    }
  });

  otpCode.addEventListener('input', () => {
    otpCode.value = otpCode.value.replace(/\D/g, '').slice(0, 6);
    if (otpCode.value.length === 6 && !otpSubmit.disabled) otpForm.requestSubmit();
  });

  otpForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!/^\d{6}$/.test(otpCode.value)) { showOtpMessage('Mã gồm 6 chữ số.', true); return; }
    otpSubmit.disabled = true; showOtpMessage('Đang kiểm tra mã…');
    try {
      const response = await fetch('/api/auth/otp/verify', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({code: otpCode.value})});
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        if (body.restart) { showLogin(); showMessage(body.detail, true); return; }
        throw new Error(body.detail || 'Mã không đúng.');
      }
      showOtpMessage('Xác thực thành công.'); ready();
    } catch (error) { showOtpMessage(error.message, true); otpCode.select(); }
    finally { otpSubmit.disabled = false; }
  });

  otpResend.addEventListener('click', async () => {
    otpResend.disabled = true; showOtpMessage('Đang gửi lại mã…');
    try {
      const response = await fetch('/api/auth/otp/resend', {method:'POST'});
      const body = await response.json().catch(() => ({}));
      if (body.restart) { showLogin(); showMessage(body.detail, true); return; }
      if (!response.ok) throw new Error(body.detail || 'Không gửi lại được mã.');
      showOtpMessage(`Đã gửi mã mới tới ${body.email}.`); otpCode.value = ''; otpCode.focus();
    } catch (error) { showOtpMessage(error.message, true); }
    finally { setTimeout(() => { otpResend.disabled = false; }, 3000); }
  });

  document.getElementById('otp-back').addEventListener('click', () => { showLogin(); showMessage(''); });

  document.getElementById('logout-button').addEventListener('click', async () => {
    await fetch('/api/auth/logout', {method:'POST'});
    window.location.reload();
  });

  (async () => {
    try {
      config = await (await fetch('/api/auth/config')).json();
      if (config.demo || !config.enabled) { google.hidden = true; divider.hidden = true; }
      if (config.demo) {
        registering=false;
        document.getElementById('auth-title').innerHTML='Mô phỏng ứng dụng<span class="title-dot">.</span>';
        document.getElementById('auth-description').textContent='Dữ liệu mẫu cục bộ · không dùng camera hoặc Firebase.';
        document.querySelectorAll('#auth-form .field').forEach(field=>field.hidden=true);
        toggle.hidden=true; submit.textContent='Mở dashboard mô phỏng';
        showMessage('Dữ liệu mô phỏng được đánh dấu DEMO.');
      } else if (!config.enabled) showMessage('Cần cấu hình FIREBASE_WEB_API_KEY, bật Email/Password trong Firebase Authentication và đặt service account.', true);
      if (config.demo) {
        let session=await fetch('/api/auth/me');
        if(!session.ok) session=await fetch('/api/auth/demo-session',{method:'POST'});
        if(!session.ok) throw new Error('Không thể mở phiên mô phỏng.');
        ready(true);
      } else if (config.enabled) {
        firebaseAuth().catch(() => {}); // preload so the Google popup opens straight from the click
        const session = await fetch('/api/auth/me');
        if (session.ok) { await session.json(); ready(); }
      }
    } catch { showMessage('Không thể kết nối máy chủ ExamWatch.', true); }
  })();
})();
