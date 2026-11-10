'use strict';
// Incident search & review, detection settings, emotion chart, speech search and shift reports.
(() => {
  const EVENT_NAMES = {
    phone: 'VẬT THỂ · ĐIỆN THOẠI', head_turning: 'HƯỚNG MẶT · QUAY ĐẦU KÉO DÀI', standing: 'TƯ THẾ · NHỔM DẬY',
    head_down: 'TƯ THẾ · CÚI XUỐNG', moving: 'CHUYỂN ĐỘNG · RỜI CHỖ', arm_raise: 'TƯ THẾ · GÓC TAY TĂNG',
    body_lean: 'TƯ THẾ · NGẢ NGƯỜI', negative_emotion: 'CẢM XÚC TIÊU CỰC ≥ 20 GIÂY', tired: 'HỖ TRỢ · DẤU HIỆU MỆT MỎI', proctor_dwell: 'NHẮC GIÁM THỊ · ĐỨNG LÂU CẠNH BÀN',
  };
  const SHORT = {
    phone: 'Điện thoại', head_turning: 'Quay đầu', standing: 'Đứng dậy', head_down: 'Cúi xuống', moving: 'Rời chỗ',
    arm_raise: 'Giơ tay', body_lean: 'Ngả người', negative_emotion: 'Cảm xúc tiêu cực', tired: 'Mệt mỏi', proctor_dwell: 'Giám thị đứng lâu',
  };
  const CATEGORIES = ['Sử dụng điện thoại', 'Nhìn bài / trao đổi', 'Rời khỏi chỗ', 'Nói chuyện', 'Sử dụng tài liệu',
    'Hỗ trợ thí sinh (sức khỏe, tâm lý)', 'Giám thị đứng lâu cạnh bàn', 'Khác'];
  const DEFAULT_CATEGORY = { phone: 0, head_turning: 1, body_lean: 1, arm_raise: 1, standing: 2, moving: 2, head_down: 1, negative_emotion: 5, tired: 5, proctor_dwell: 6 };
  const DECISION = { violation: 'Xác nhận vi phạm', warning: 'Nhắc nhở', dismissed: 'Không vi phạm' };
  const MOOD = { normal: ['Bình thường', '#2f9e74'], anxious: ['Lo lắng', '#e0a030'], tired: ['Mệt mỏi', '#7b6fd6'], sad: ['Buồn', '#d2554f'] };
  window.examwatchEventNames = EVENT_NAMES;
  const debounce = (fn, ms = 300) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };
  const highlight = (text, q) => {
    const safe = esc(text);
    if (!q) return safe;
    const fold = (s) => s.normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/đ/gi, 'd').toLowerCase();
    const i = fold(text).indexOf(fold(q));
    return i < 0 ? safe : esc(text.slice(0, i)) + '<mark>' + esc(text.slice(i, i + q.length)) + '</mark>' + esc(text.slice(i + q.length));
  };
  const statusBadge = (a) => a.acknowledged
    ? `<span class="status-badge decision-${esc(a.decision || 'done')}">${esc(a.decision_label || 'Đã kiểm tra')}</span>`
    : '<span class="status-badge open">Chưa xử lý</span>';

  // ---------- session selectors ----------
  async function fillSessions(select) {
    const sessions = await api('/api/sessions');
    const current = select.value;
    select.innerHTML = '<option value="">Tất cả phiên</option>' + sessions.map((s) => `<option value="${esc(s.id)}">${esc(s.title)} · ${timestamp(s.started_at)}</option>`).join('');
    select.value = sessions.some((s) => s.id === current) ? current : '';
  }

  // ---------- #2 incident history ----------
  $('inc-kind').innerHTML += Object.entries(SHORT).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
  async function loadIncidents() {
    const params = new URLSearchParams({ q: $('inc-q').value.trim(), kind: $('inc-kind').value, status: $('inc-status').value, session_id: $('inc-session').value });
    if ($('inc-from').value) params.set('date_from', new Date($('inc-from').value + 'T00:00:00').toISOString());
    if ($('inc-to').value) params.set('date_to', new Date($('inc-to').value + 'T23:59:59').toISOString());
    const rows = await api('/api/alerts?' + params);
    const q = $('inc-q').value.trim();
    $('inc-rows').innerHTML = rows.length ? rows.map((a) => `<tr data-review="${a.id}">
      <td><b>${new Date(a.created_at).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}</b><small>${new Date(a.created_at).toLocaleDateString('vi-VN')} · +${duration(a.offset_seconds)}</small></td>
      <td>${highlight(a.session_title, q)}</td><td>${highlight(a.student_name, q)}${a.student_id ? `<small>${esc(a.student_id)}</small>` : ''}</td>
      <td><span class="kind-dot kind-${esc(a.kind)}"></span>${esc(SHORT[a.kind] || a.kind)}<small>${highlight(a.detail, q)}</small></td>
      <td>${a.has_thumbnail ? `<img class="thumb" loading="lazy" src="/api/alerts/${a.id}/thumb" alt="">` : `<span class="thumb none">${a.evidence_type === 'none' ? '—' : '…'}</span>`}</td>
      <td>${statusBadge(a)}</td><td><button class="button ${a.acknowledged ? 'secondary' : 'primary'}" data-review="${a.id}">${a.acknowledged ? 'Xem' : 'Ghi nhận'}</button></td></tr>`).join('')
      : '<tr><td colspan="7"><div class="empty compact">Không có sự cố phù hợp bộ lọc.</div></td></tr>';
    const open = rows.filter((a) => !a.acknowledged).length;
    $('inc-summary').textContent = `${rows.length} sự cố · ${open} chưa xử lý · ${rows.length - open} đã xử lý`;
  }
  const reloadIncidents = debounce(() => loadIncidents().catch((e) => notice(e.message)));
  ['inc-q', 'inc-kind', 'inc-status', 'inc-from', 'inc-to', 'inc-session'].forEach((id) => $(id).addEventListener('input', reloadIncidents));

  // ---------- #3 review dialog with signature ----------
  const pad = $('signature-pad'), pctx = pad.getContext('2d');
  let drawing = false, signed = false, reviewing = null;
  const padPoint = (e) => { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) * pad.width / r.width, (e.clientY - r.top) * pad.height / r.height]; };
  const clearPad = () => { pctx.fillStyle = '#fff'; pctx.fillRect(0, 0, pad.width, pad.height); signed = false; };
  pad.addEventListener('pointerdown', (e) => { drawing = true; pad.setPointerCapture(e.pointerId); pctx.beginPath(); pctx.moveTo(...padPoint(e)); });
  pad.addEventListener('pointermove', (e) => { if (!drawing) return; pctx.lineTo(...padPoint(e)); pctx.strokeStyle = '#13324a'; pctx.lineWidth = 2.4; pctx.lineCap = 'round'; pctx.stroke(); signed = true; });
  pad.addEventListener('pointerup', () => { drawing = false; });
  $('signature-clear').addEventListener('click', clearPad);
  $('review-category').innerHTML = CATEGORIES.map((c) => `<option>${esc(c)}</option>`).join('');
  $('review-close').addEventListener('click', () => $('review-dialog').close());

  window.openReview = async (id) => {
    const a = await api(`/api/alerts/${id}`).catch((e) => { notice(e.message); return null; });
    if (!a) return;
    reviewing = a;
    const ev = a.evidence_type === 'video' && a.evidence_status === 'ready'
      ? `<video controls playsinline src="/api/alerts/${a.id}/video"></video>`
      : a.evidence_status === 'missing' ? '<div class="empty compact">Bằng chứng không còn trên máy chủ (máy chủ đã khởi động lại). Biên bản và quyết định vẫn được lưu.</div>'
      : a.evidence_type === 'image' ? `<img src="/api/alerts/${a.id}/image" alt="Ảnh bằng chứng">`
      : a.has_thumbnail ? `<img src="/api/alerts/${a.id}/thumb" alt="Ảnh bằng chứng"><p class="hint">Video đang hoàn tất…</p>`
      : '<div class="empty compact">Không lưu bằng chứng cho sự cố này (theo cấu hình).</div>';
    $('review-evidence').innerHTML = ev;
    $('review-info').innerHTML = `<span class="event-kind">${esc(EVENT_NAMES[a.kind] || a.kind)}</span><h3>${esc(a.student_name)}</h3>
      <p>${esc(a.detail)}</p><p class="hint">${esc(a.session_title || '')} · ${timestamp(a.created_at)} · mốc phiên ${duration(a.offset_seconds)}</p>
      ${a.acknowledged ? `<p class="hint">Đã ghi nhận bởi <b>${esc(a.reviewer || '—')}</b> lúc ${a.reviewed_at ? timestamp(a.reviewed_at) : '—'}</p>${a.has_signature ? `<img class="signature-img" src="/api/alerts/${a.id}/signature?${Date.now()}" alt="Chữ ký">` : ''}` : ''}`;
    $('review-category').value = a.category || CATEGORIES[DEFAULT_CATEGORY[a.kind] ?? 7];
    document.querySelectorAll('[name=decision]').forEach((r) => { r.checked = r.value === (a.decision || 'warning'); });
    $('review-note').value = a.note || '';
    $('review-reviewer').value = (a.reviewer || '').replace(/ \(.*\)$/, '') || store.get('examwatch-reviewer') || '';
    $('review-submit').textContent = a.acknowledged ? 'Cập nhật biên bản' : 'Ghi nhận';
    $('review-status').textContent = '';
    clearPad();
    $('review-dialog').showModal();
  };
  $('review-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!reviewing) return;
    const decision = document.querySelector('[name=decision]:checked')?.value || 'warning';
    const reviewer = $('review-reviewer').value.trim();
    if (!reviewer) { $('review-status').textContent = 'Nhập họ tên giám thị xác nhận.'; return; }
    $('review-submit').disabled = true;
    try {
      await api(`/api/alerts/${reviewing.id}/review`, json({ category: $('review-category').value, decision, note: $('review-note').value,
        reviewer, signature: signed ? pad.toDataURL('image/png') : null }));
      store.set('examwatch-reviewer', reviewer);
      $('review-dialog').close();
      notice(`Đã ghi nhận: ${DECISION[decision]} · ${reviewing.student_name}`);
      if (session) await loadSessionData(session.id);
      if (viewing && viewing !== session?.id) await loadSessionData(viewing);
      if (currentPage === 'incidents') await loadIncidents();
    } catch (err) { $('review-status').textContent = err.message; }
    finally { $('review-submit').disabled = false; }
  });

  // ---------- #4 detection settings ----------
  const sensitivityLabel = () => { $('sensitivity-value').textContent = `${$('sensitivity').value}%`; };
  $('sensitivity').addEventListener('input', () => { sensitivityLabel(); standingSettingsDirty = true; });
  ['proctor-dwell', 'ev-image', 'ev-video', 'ev-audio'].forEach((id) => $(id).addEventListener('input', () => { standingSettingsDirty = true; }));
  // Fine-grained values: a field the proctor edits becomes an override saved on
  // the server; untouched ones keep following the sensitivity slider / defaults.
  let detail = null, overridden = new Set(), draft = {};
  const fmt = (v) => (Math.round(Number(v) * 100) / 100).toString().replace('.', ',');
  const stepFor = (spec) => { const span = spec.max - spec.min; return span <= 3 ? 0.05 : span <= 12 ? 0.25 : 1; };
  function renderAdvanced() {
    if (!detail?.schema) return;
    $('alert-toggles').innerHTML = Object.entries(detail.kinds).map(([k, label]) =>
      `<label><input type="checkbox" data-kind="${k}" ${detail.enabled[k] !== false ? 'checked' : ''}> ${esc(label)}</label>`).join('');
    $('auto-calibrate').checked = detail.auto_calibrate !== false;
    $('advanced-settings').innerHTML = Object.entries(detail.groups).map(([gid, gname]) => {
      const rows = detail.schema.filter((x) => x.group === gid).map((spec) => {
        const v = draft[spec.key], step = stepFor(spec), touched = overridden.has(spec.key);
        return `<label class="tunable"><span>${esc(spec.label)} <b data-out="${spec.key}">${fmt(v)}</b>${touched ? '<em>đã chỉnh</em>' : ''}</span>
          <input type="range" data-key="${spec.key}" min="${spec.min}" max="${spec.max}" step="${step}" value="${v}">
          <input type="number" data-key="${spec.key}" min="${spec.min}" max="${spec.max}" step="${step}" value="${v}">
          ${spec.help ? `<small>${esc(spec.help)}</small>` : ''}</label>`;
      }).join('');
      return `<fieldset class="tunable-group"><legend>${esc(gname)}</legend>${rows}</fieldset>`;
    }).join('');
  }
  $('advanced-settings').addEventListener('input', (e) => {
    const key = e.target.dataset.key; if (!key) return;
    const value = Number(e.target.value); if (!Number.isFinite(value)) return;
    draft[key] = value; standingSettingsDirty = true;
    const fresh = !overridden.has(key); overridden.add(key);
    document.querySelectorAll(`#advanced-settings [data-key="${key}"]`).forEach((el) => { if (el !== e.target) el.value = value; });
    const out = document.querySelector(`[data-out="${key}"]`);
    if (out) { out.textContent = fmt(value); if (fresh) out.insertAdjacentHTML('afterend', '<em>đã chỉnh</em>'); }
  });
  $('alert-toggles').addEventListener('input', () => { standingSettingsDirty = true; });
  $('auto-calibrate').addEventListener('input', () => { standingSettingsDirty = true; });
  $('reset-advanced').addEventListener('click', () => {
    overridden = new Set(); standingSettingsDirty = true;
    $('standing-settings-status').textContent = 'Đã bỏ các ô đã chỉnh. Bấm Lưu cấu hình để dùng lại độ nhạy chung.';
    renderAdvanced();
  });
  function applyDetection(d) {
    detail = d; window.examwatchDetection = d;
    overridden = new Set(d.overridden || []); draft = { ...(d.advanced || {}) };
    renderAdvanced();
  }
  async function loadDetection() {
    const d = await api('/api/settings/detection');
    if (standingSettingsDirty) return;
    $('sensitivity').value = d.sensitivity; sensitivityLabel();
    $('standing-threshold').value = d.standing_threshold_seconds;
    $('proctor-dwell').value = d.proctor_dwell_seconds;
    $('ev-image').checked = d.evidence.image; $('ev-video').checked = d.evidence.video; $('ev-audio').checked = d.evidence.audio;
    applyDetection(d);
  }
  $('save-alert-settings').addEventListener('click', async () => {
    const standing = Number($('standing-threshold').value), dwell = Number($('proctor-dwell').value);
    if (!Number.isInteger(standing) || standing < 1 || standing > 300) return notice('Thời gian giám sát: số nguyên từ 1 đến 300 giây.');
    if (!Number.isInteger(dwell) || dwell < 10 || dwell > 1800) return notice('Thời gian nhắc giám thị: số nguyên từ 10 đến 1800 giây.');
    $('save-alert-settings').disabled = true;
    try {
      const advanced = Object.fromEntries([...overridden].map((k) => [k, Number(draft[k])]));
      const enabled = Object.fromEntries([...document.querySelectorAll('#alert-toggles [data-kind]')].map((el) => [el.dataset.kind, el.checked]));
      const d = await api('/api/settings/detection', json({ sensitivity: Number($('sensitivity').value), standing_threshold_seconds: standing,
        proctor_dwell_seconds: dwell, evidence: { image: $('ev-image').checked, video: $('ev-video').checked, audio: $('ev-audio').checked },
        advanced, enabled, auto_calibrate: $('auto-calibrate').checked }));
      standingSettingsDirty = false; standingThreshold = d.standing_threshold_seconds;
      applyDetection(d);
      $('standing-threshold-hint').textContent = standingThreshold;
      notice(`Đã lưu cấu hình · độ nhạy ${d.sensitivity}% · giám sát ${d.standing_threshold_seconds} giây · nhắc giám thị sau ${d.proctor_dwell_seconds} giây.`);
    } catch (e) { notice(e.message); } finally { $('save-alert-settings').disabled = false; }
  });

  // ---------- #5 emotion chart ----------
  async function renderMood(box, sessionId, studentId = '') {
    const data = await api(`/api/sessions/${sessionId}/emotions?bucket=10${studentId ? '&student_id=' + encodeURIComponent(studentId) : ''}`);
    const pts = data.series;
    const options = '<option value="">Cả phòng thi</option>' + data.students.map((p) => `<option value="${esc(p.id)}" ${p.id === studentId ? 'selected' : ''}>${esc(p.name)}</option>`).join('');
    if (pts.length < 2) { box.innerHTML = `<div class="mood-head"><select class="mood-student">${options}</select></div><div class="empty compact">Chưa đủ dữ liệu biểu cảm để vẽ biểu đồ.</div>`; }
    else {
      const W = 720, H = 220, L = 40, B = 26, T = 10, maxX = pts[pts.length - 1].offset || 1;
      const x = (o) => L + (W - L - 10) * o / maxX, y = (v) => T + (H - T - B) * (1 - v);
      const lines = Object.entries(MOOD).map(([k, [, color]]) => `<path d="${pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.offset).toFixed(1)},${y(p[k] || 0).toFixed(1)}`).join('')}" fill="none" stroke="${color}" stroke-width="2.4" stroke-linejoin="round"/>`).join('');
      const grid = [0, .25, .5, .75, 1].map((v) => `<line x1="${L}" x2="${W - 10}" y1="${y(v)}" y2="${y(v)}" stroke="currentColor" stroke-opacity=".12"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join('');
      const ticks = Array.from({ length: 5 }, (_, i) => maxX * i / 4).map((o) => `<text x="${x(o)}" y="${H - 6}" text-anchor="middle">${duration(o).slice(3)}</text>`).join('');
      box.innerHTML = `<div class="mood-head"><select class="mood-student">${options}</select><span class="mood-legend">${Object.values(MOOD).map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join('')}</span></div>
        <svg viewBox="0 0 ${W} ${H}" class="mood-svg" role="img" aria-label="Tỉ lệ biểu cảm theo thời gian">${grid}${ticks}${lines}</svg>
        <div class="mood-people">${data.students.map((p) => { const tot = (p.normal + p.anxious + p.tired + p.sad) || 1; const worst = ['sad', 'tired', 'anxious'].sort((a, b) => p[b] - p[a])[0]; return `<span class="${p[worst] / tot > .3 ? 'warn' : ''}"><b>${esc(p.name)}</b> ${Object.keys(MOOD).map((k) => `<i style="background:${MOOD[k][1]};width:${Math.round(60 * p[k] / tot)}px"></i>`).join('')}${p[worst] / tot > .3 ? ` · ${MOOD[worst][0]} ${Math.round(100 * p[worst] / tot)}%` : ''}</span>`; }).join('')}</div>`;
    }
    box.querySelector('.mood-student').addEventListener('change', (e) => renderMood(box, sessionId, e.target.value).catch((err) => notice(err.message)));
  }
  window.examwatchHistoryRendered = () => {
    const box = document.querySelector('[data-mood-session]');
    if (box && !box.dataset.loaded) { box.dataset.loaded = '1'; renderMood(box, box.dataset.moodSession).catch((e) => { box.innerHTML = `<p class="hint">${esc(e.message)}</p>`; }); }
  };

  // ---------- #6 speech search ----------
  async function loadSpeech() {
    const q = $('sp-q').value.trim();
    const rows = await api('/api/transcripts?' + new URLSearchParams({ q, session_id: $('sp-session').value, speaker: $('sp-speaker').value }));
    $('sp-rows').innerHTML = rows.length ? rows.map((t) => `<div class="speech-row"><div><time>${esc(t.session_title)} · ${duration(t.start_seconds)} — ${duration(t.end_seconds)}</time>
      <span class="speaker ${t.speaker_name ? '' : 'unknown'}">${t.speaker_name ? `${highlight(t.speaker_name, q)} <small>ước tính ${Math.round((t.speaker_score || 0) * 100)}%</small>` : 'Chưa xác định người nói'}</span><p>${highlight(t.text, q)}</p></div>
      ${t.has_audio ? `<audio controls preload="none" src="/api/transcripts/${t.id}/audio"></audio>` : '<span class="tag">Không lưu âm thanh</span>'}</div>`).join('')
      : `<div class="empty compact">${q ? 'Không tìm thấy lời nói chứa từ khóa này.' : 'Chưa có bản chép lời.'}</div>`;
  }
  const reloadSpeech = debounce(() => loadSpeech().catch((e) => notice(e.message)));
  ['sp-q', 'sp-session', 'sp-speaker'].forEach((id) => $(id).addEventListener('input', reloadSpeech));
  async function fillSpeakers() {
    const people = await api('/api/speakers');
    const current = $('sp-speaker').value;
    $('sp-speaker').innerHTML = '<option value="">Tất cả người nói</option><option value="unknown">Chưa xác định</option>' + people.map((p) => `<option value="${esc(p.id)}">${esc(p.name)} (${p.lines})</option>`).join('');
    $('sp-speaker').value = [...$('sp-speaker').options].some((o) => o.value === current) ? current : '';
  }

  // ---------- #1 shift report ----------
  $('report-close').addEventListener('click', () => $('report-dialog').close());
  $('report-print').addEventListener('click', () => window.print());
  window.openReport = async (id) => {
    const r = await api(`/api/sessions/${id}/report`).catch((e) => { notice(e.message); return null; });
    if (!r) return;
    const s = r.session, at = r.attendance || { total: 0, present: 0, absent: 0, people: [] };
    const end = s.ended_at ? new Date(s.ended_at) : new Date();
    const status = { normal: ['Bình thường', 'ok'], pending: ['Còn sự cố chưa xử lý', 'pending'], violation: ['Có vi phạm được xác nhận', 'bad'] }[r.status];
    $('report-body').innerHTML = `<div class="report">
      <div class="report-head"><div class="brand-mark">E</div><div><div class="eyebrow">EXAMWATCH · BÁO CÁO CA THI</div><h1>${esc(s.title)}</h1></div><span class="report-status ${status[1]}">${status[0]}</span></div>
      <div class="report-kpis"><div><span>Tổng số thí sinh</span><b>${at.total}</b></div><div><span>Đã điểm danh</span><b>${at.present}</b></div><div><span>Vắng</span><b>${at.absent}</b></div><div><span>Sự cố</span><b>${r.alerts.total}</b></div><div><span>Đã xử lý</span><b>${r.alerts.reviewed}</b></div></div>
      <table><tbody><tr><th>Thời gian bắt đầu</th><td>${timestamp(s.started_at)}</td><th>Thời gian kết thúc</th><td>${s.ended_at ? timestamp(s.ended_at) : 'Đang diễn ra'}</td></tr>
      <tr><th>Thời lượng</th><td>${duration((end - new Date(s.started_at)) / 1000)}</td><th>Sự cố theo loại</th><td>${Object.entries(r.alerts.by_kind).map(([k, v]) => `${esc(SHORT[k] || k)}: ${v}`).join(' · ') || 'Không có'}</td></tr></tbody></table>
      <h3>Điểm danh</h3><table><thead><tr><th>Mã</th><th>Họ và tên</th><th>Trạng thái</th><th>Xuất hiện lần đầu</th></tr></thead><tbody>
      ${at.people.map((p) => `<tr><td>${esc(p.id)}</td><td>${esc(p.name)}</td><td>${p.present ? (p.unlisted ? 'Có mặt (ngoài danh sách)' : 'Có mặt') : '<b class="absent">Vắng</b>'}</td><td>${p.first_seen != null ? duration(p.first_seen) : '—'}</td></tr>`).join('') || '<tr><td colspan="4">Chưa có danh sách thí sinh.</td></tr>'}</tbody></table>
      <h3>Biên bản sự cố đã ghi nhận</h3><table><thead><tr><th>Mốc</th><th>Thí sinh</th><th>Loại</th><th>Quyết định</th><th>Ghi chú</th><th>Giám thị</th></tr></thead><tbody>
      ${r.incidents.map((a) => `<tr><td>${duration(a.offset_seconds)}</td><td>${esc(a.student_name)}</td><td>${esc(a.category || SHORT[a.kind] || a.kind)}</td><td>${esc(a.decision_label || 'Đã kiểm tra')}</td><td>${esc(a.note || '')}</td><td>${esc(a.reviewer || '')}${a.has_signature ? `<br><img class="signature-img" src="/api/alerts/${a.id}/signature" alt="">` : ''}</td></tr>`).join('') || '<tr><td colspan="6">Chưa có sự cố được ghi nhận.</td></tr>'}</tbody></table>
      <p class="hint">Báo cáo tạo lúc ${timestamp(Date.now())}. Cảnh báo AI là tín hiệu tham khảo; quyết định cuối cùng thuộc về giám thị.</p></div>`;
    $('report-dialog').showModal();
  };

  // ---------- #8 live proctor reminder ----------
  const seen = new Set();
  let primed = false;
  window.examwatchLiveAlerts = (alerts) => {
    for (const a of alerts) {
      if (seen.has(a.id)) continue;
      seen.add(a.id);
      if (primed && a.kind === 'proctor_dwell') notice(`⚠ Nhắc giám thị: ${a.detail}`, true);
    }
    primed = true;
  };
  window.examwatchStatus = (status) => {
    window.examwatchDwell = status.detection?.proctor_dwell_seconds || 60;
    if (status.detection) window.examwatchDetection = status.detection;
  };

  // ---------- page hooks ----------
  window.examwatchPage = (page) => {
    if (page === 'incidents') fillSessions($('inc-session')).then(loadIncidents).catch((e) => notice(e.message));
    if (page === 'speech') Promise.all([fillSessions($('sp-session')), fillSpeakers()]).then(loadSpeech).catch((e) => notice(e.message));
    if (page === 'settings') loadDetection().catch((e) => notice(e.message));
  };
})();
