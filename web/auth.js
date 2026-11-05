'use strict';
(() => {
  const gate = document.getElementById('auth-gate');
  const form = document.getElementById('auth-form');
  const message = document.getElementById('auth-message');
  const submit = document.getElementById('auth-submit');
  const toggle = document.getElementById('auth-toggle');
  let config = null, registering = false;
  const showMessage = (text, error = false) => { message.textContent = text; message.classList.toggle('error', error); };
  const ready = (demo = false) => { gate.hidden = true; document.getElementById('logout-button').hidden = false; window.dispatchEvent(new CustomEvent('examwatch-auth-ready',{detail:{demo}})); };
  const firebaseMessage = (code) => ({
    EMAIL_EXISTS: 'Email này đã được đăng ký.', INVALID_EMAIL: 'Email không hợp lệ.',
    INVALID_PASSWORD: 'Mật khẩu không đúng hoặc chưa đủ 6 ký tự.', WEAK_PASSWORD: 'Mật khẩu cần ít nhất 6 ký tự.',
    EMAIL_NOT_FOUND: 'Không tìm thấy tài khoản này.', INVALID_LOGIN_CREDENTIALS: 'Email hoặc mật khẩu không đúng.',
    OPERATION_NOT_ALLOWED: 'Firebase chưa bật phương thức đăng nhập Email/Mật khẩu.',
    TOO_MANY_ATTEMPTS_TRY_LATER: 'Thử quá nhiều lần. Vui lòng đợi rồi thử lại.'
  })[code] || 'Không thể xác thực. Kiểm tra cấu hình Firebase và thử lại.';
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
    submit.disabled = true; toggle.disabled = true; showMessage('Đang kết nối Firebase…');
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
      const session = await fetch('/api/auth/session', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id_token:result.idToken})});
      if (!session.ok) throw new Error((await session.json()).detail || 'Không tạo được phiên đăng nhập.');
      document.getElementById('auth-password').value = '';
      showMessage('Đăng nhập thành công.'); ready();
    } catch (error) { showMessage(error.message, true); }
    finally { submit.disabled = false; toggle.disabled = false; }
  });
  document.getElementById('logout-button').addEventListener('click', async () => {
    await fetch('/api/auth/logout', {method:'POST'});
    window.location.reload();
  });
  (async () => {
    try {
      config = await (await fetch('/api/auth/config')).json();
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
        const session = await fetch('/api/auth/me');
        if (session.ok) { await session.json(); ready(); }
      }
    } catch { showMessage('Không thể kết nối máy chủ ExamWatch.', true); }
  })();
})();
