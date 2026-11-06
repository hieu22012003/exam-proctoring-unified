'use strict';
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const emotionNames = {angry:'Giận dữ',disgust:'Ghê sợ',fear:'Sợ hãi',happy:'Vui vẻ',sad:'Buồn',surprise:'Ngạc nhiên',neutral:'Trung tính',unknown:'Chưa rõ'};
const postureNames = {seated:'Ngồi đúng tư thế',standing:'Dấu hiệu đứng lên',head_down:'Cúi xuống',moving:'Rời chỗ / di chuyển',uncalibrated:'Chưa hiệu chuẩn',insufficient:'Không đủ dữ liệu'};
const moodNames = {normal:'Bình thường',anxious:'Lo lắng',tired:'Mệt mỏi',sad:'Buồn',unknown:'Chưa rõ'};
const moodOf = (t) => t.tired ? 'tired' : ({neutral:'normal',happy:'normal',fear:'anxious',surprise:'anxious',angry:'anxious',disgust:'anxious',sad:'sad'})[t.emotion] || 'unknown';
let maxPeople = 25, frameWidth = 1280;
function poseWatchHTML(t) {
  const labels={left_arm:'Tay trái',right_arm:'Tay phải',lean:'Nghiêng thân'};
  const values=Object.entries(t.pose_angles||{}).filter(([,v])=>v!==null)
    .map(([key,v])=>`${labels[key]} ${Math.round(v)}°${(t.pose_flags||[]).includes(key)?' ⚑':''}`);
  return values.length ? `<small style="color:${t.pose_flags?.length?'#b84945':'inherit'}">${esc(values.join(' · '))}${t.lean_source==='abdomen_calibrated'?' · gốc bụng hiệu chuẩn':''}${t.lean_source==='abdomen_uncalibrated'?' · hiệu chuẩn ngồi để đo nghiêng':''}</small>` : '<small>Góc tay/thân: không đủ landmark rõ</small>';
}
const owner = sessionStorage.getItem('examwatch-owner') || crypto.randomUUID();
sessionStorage.setItem('examwatch-owner', owner);
let camera = null, session = null, capturing = false, busy = false, tracks = [], trackAt = 0, samples = [];
let selectedAngle = 'front';
let audioContext = null, audioStream = null, audioNode = null, audioSource = null;
let audioChain = Promise.resolve(), flushDone = null, viewing = null, lastRecords = null;
let noticeTimeout, lastError = '', currentPage = 'overview';
let micSampleUrl = null, historyKey = '';
let standingThreshold = 5, standingSettingsDirty = false;
let cameraResultAt = null, cameraAnalysisFps = 0;
const frameCanvas = document.createElement('canvas');
const framePair = new ExamWatchFrames.FramePair();
$('standing-threshold').addEventListener('input',()=>{standingSettingsDirty=true;});
function clearProcessedFrame() {
  framePair.reset();
  cameraResultAt=null;cameraAnalysisFps=0;
  $('camera-stage').classList.remove('processing');
}

function notice(message, persistent = false) {
  clearTimeout(noticeTimeout);
  $('notice').textContent = message;
  $('notice').hidden = false;
  if (!persistent) noticeTimeout = setTimeout(() => { $('notice').hidden = true; }, 8000);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let body;
    try { body = await response.json(); } catch { body = {detail: `HTTP ${response.status}`}; }
    const error = new Error(typeof body.detail === 'string' ? body.detail : 'Dữ liệu gửi lên không hợp lệ.');
    error.status = response.status;
    throw error;
  }
  return response.json();
}
const json = (data) => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
const timestamp = (date) => new Date(date).toLocaleString('vi-VN');
function duration(seconds) {
  seconds = Math.max(0, Math.floor(seconds));
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60].map((s) => String(s).padStart(2,'0')).join(':');
}
function navigate(page) {
  currentPage = page;
  document.querySelectorAll('.page').forEach((e) => { e.hidden = e.id !== page; });
  document.querySelectorAll('.nav').forEach((e) => e.classList.toggle('active', e.dataset.page === page));
  $('page-label').textContent = {overview:'TỔNG QUAN',monitor:'GIÁM SÁT TRỰC TIẾP',students:'THÍ SINH',incidents:'SỰ CỐ',history:'PHIÊN THI & BÁO CÁO',speech:'LỜI NÓI',settings:'CÀI ĐẶT'}[page];
  window.examwatchPage?.(page);
  if (page === 'overview') loadOverview().catch((e)=>notice(e.message));
  if (page === 'students') loadStudents().catch((e) => notice(e.message));
  if (page === 'history') loadSessions().catch((e) => notice(e.message));
}
document.querySelectorAll('[data-page]').forEach((e) => e.addEventListener('click', () => navigate(e.dataset.page)));

async function loadOverview() {
  const [students,sessions,status]=await Promise.all([api('/api/students'),api('/api/sessions'),api('/api/status')]);
  $('overview-students').textContent=students.length;
  $('overview-sessions').textContent=sessions.length;
  $('overview-active').textContent=status.session?1:sessions.filter(s=>!s.ended_at).length;
  const cloud=status.firebase_sync?.state||'not_configured';
  $('overview-cloud').textContent={connected:'Đã kết nối',degraded:'Đang chờ',error:'Có lỗi',not_configured:'Cục bộ'}[cloud]||cloud;
  $('overview-session-list').innerHTML=sessions.length?sessions.slice(0,4).map(s=>`<button class="overview-session" data-overview-session="${esc(s.id)}"><span class="overview-session-icon">◷</span><span><b>${esc(s.title)}</b><small>${timestamp(s.started_at)}</small></span><span class="overview-session-state ${s.ended_at?'':'is-open'}">${s.ended_at?'Đã kết thúc':'Đang mở'}</span><span class="overview-chevron">›</span></button>`).join(''):'<div class="empty compact">Chưa có phiên thi. Bắt đầu phiên giám sát đầu tiên.</div>';
}
$('overview-session-list').addEventListener('click',async(e)=>{
  const row=e.target.closest('[data-overview-session]');if(!row)return;
  viewing=row.dataset.overviewSession;navigate('history');
  try{await loadSessionData(viewing);await loadSessions();}catch(err){notice(err.message);}
});

async function connectCamera() {
  if (camera?.getVideoTracks().some((t) => t.readyState === 'live')) return;
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Hãy mở bằng http://localhost:8000 hoặc http://127.0.0.1:8000 trên Chrome/Edge.');
  camera = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:15,max:20}},audio:false});
  $('video').srcObject = camera;
  await $('video').play();
  $('camera-stage').style.aspectRatio = `${$('video').videoWidth} / ${$('video').videoHeight}`;
  $('camera-placeholder').hidden = true;
  $('camera-stamp').hidden = false;
  $('camera-label').textContent = 'ĐÃ KẾT NỐI · CHỈ XEM TRƯỚC';
  $('enroll-help').hidden = true;
  camera.getVideoTracks()[0].addEventListener('ended', () => {
    $('enroll-help').hidden = false;
    $('camera-label').textContent = 'CAMERA ĐÃ NGẮT';
    $('camera-placeholder').hidden = false;
    tracks = []; clearProcessedFrame();
    notice('Camera đã ngắt. Phiên thi vẫn còn mở; hãy kết nối lại hoặc kết thúc phiên.', true);
  });
}
$('connect-camera').addEventListener('click', () => connectCamera().catch((e) => notice(`Không mở được camera: ${e.message}`, true)));
$('connect-enroll-camera').addEventListener('click', () => connectCamera().catch((e) => notice(`Không mở được camera: ${e.message}`, true)));

function captureCanvas() {
  const video = $('video');
  if (!camera || video.readyState < 2 || !video.videoWidth) throw new Error('Camera chưa sẵn sàng.');
  const width = Math.min(frameWidth, video.videoWidth);
  const height = Math.round(video.videoHeight * width / video.videoWidth);
  if(frameCanvas.width!==width||frameCanvas.height!==height){frameCanvas.width=width;frameCanvas.height=height;}
  frameCanvas.getContext('2d').drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  return frameCanvas;
}

async function startAudio(row) {
  audioStream = await navigator.mediaDevices.getUserMedia(ExamWatchMic.constraints());
  const micSettings = audioStream.getAudioTracks()[0].getSettings();
  $('mic-state').title = micSettings.autoGainControl === true
    ? 'Micro đang bật tự động cân âm'
    : 'Trình duyệt chưa xác nhận trạng thái tự động cân âm';
  audioContext = new AudioContext({sampleRate:16000});
  await audioContext.audioWorklet.addModule('/static/audio-worklet.js');
  await audioContext.resume();
  audioSource = audioContext.createMediaStreamSource(audioStream);
  audioNode = new AudioWorkletNode(audioContext, 'examwatch-capture');
  const base = Math.max(0, (Date.now() - new Date(row.started_at).getTime()) / 1000);
  audioChain = Promise.resolve();
  audioNode.port.onmessage = ({data}) => {
    if (data.done) { flushDone?.(); return; }
    if (data.level !== undefined) {
      $('audio-meter').style.width = `${Math.min(40, data.level*500)}px`;
      return;
    }
    let power = 0;
    for (const v of data.samples) power += v*v;
    $('audio-meter').style.width = `${Math.min(40, Math.sqrt(power / data.samples.length)*500)}px`;
    audioChain = audioChain.then(() => api(`/api/audio?owner=${encodeURIComponent(owner)}&capture_id=${row.capture_id}&rate=${data.rate}&offset=${(base+data.start).toFixed(4)}`, {
      method:'POST',headers:{'Content-Type':'application/octet-stream'},body:data.samples.buffer,
    })).catch((e) => notice(`Chép lời: ${e.message}`, true));
  };
  audioSource.connect(audioNode);
  audioNode.connect(audioContext.destination);
  $('mic-state').textContent = 'ĐANG THU';
  $('audio-dot').classList.add('recording');
  $('audio-status').textContent = 'Đang nghe · chép sau khoảng ngừng nói';
  audioStream.getAudioTracks()[0].addEventListener('ended', () => {
    if (capturing) notice('Micro đã ngắt. Chép lời đã dừng; camera vẫn hoạt động.', true);
  });
}
$('record-mic-sample').addEventListener('click', async () => {
  const button=$('record-mic-sample');
  button.disabled=true;
  $('mic-check-player').pause();
  $('mic-check-result').hidden=true;
  $('download-mic-sample').hidden=true;
  if(micSampleUrl){URL.revokeObjectURL(micSampleUrl);micSampleUrl=null;}
  $('mic-check-status').textContent='Đang mở micro…';
  try {
    const sample=await ExamWatchMic.capture(audioStream, seconds=>{
      $('mic-check-status').textContent=`Đang thu ${Math.min(8,Math.floor(seconds))}/8 giây · Hãy nói câu thử.`;
    });
    micSampleUrl=URL.createObjectURL(sample.wav);
    $('mic-check-player').src=micSampleUrl;
    $('download-mic-sample').href=micSampleUrl;
    $('download-mic-sample').hidden=false;
    $('mic-check-device').textContent=`Micro: ${sample.device} · ${sample.stats.seconds.toFixed(1)} giây`;
    const descriptions={quiet:'Tín hiệu nhỏ hoặc chưa có tiếng nói. Hãy nghe lại và thử đưa micro gần hơn.',
      clipped:'Tín hiệu có dấu hiệu vỡ tiếng. Hãy giảm âm lượng micro hoặc nói xa hơn.',
      usable:'Mức tín hiệu có thể dùng. Hãy nghe xem lời nói rõ hay bị lẫn tiếng khác; mức âm lượng không xác nhận độ chính xác nhận dạng.'};
    $('mic-check-quality').textContent=descriptions[sample.stats.level];
    $('mic-check-result').hidden=false;
    $('mic-check-status').textContent='Đã thu xong. Mẫu chưa được gửi đi; bạn có thể nghe lại hoặc tải WAV để kiểm tra.';
  } catch(e) {
    $('mic-check-status').textContent=`Không thu được mẫu: ${e.message}`;
  } finally {button.disabled=false;}
});
async function stopAudio() {
  if (audioNode && audioContext?.state === 'running') {
    await Promise.race([new Promise((resolve) => { flushDone = resolve; audioNode.port.postMessage('flush'); }), new Promise((r) => setTimeout(r, 1500))]);
    await audioChain;
  }
  audioNode?.disconnect();
  audioSource?.disconnect();
  audioStream?.getTracks().forEach((t) => t.stop());
  if (audioContext && audioContext.state !== 'closed') await audioContext.close();
  audioContext = audioNode = audioStream = audioSource = null;
  $('mic-state').textContent = 'MIC TẮT';
  $('audio-dot').classList.remove('recording');
  $('audio-meter').style.width = '0';
}
function updateControls() {
  $('start-session').disabled = busy || Boolean(session);
  $('stop-session').disabled = busy || !session;
  $('calibrate').disabled = busy || !capturing;
  $('session-title').disabled = Boolean(session) || busy;
  $('save-student').disabled = Boolean(session) || busy;
  $('take-sample').disabled = Boolean(session) || busy;
  $('session-state').textContent = session ? (capturing ? 'ĐANG GIÁM SÁT' : 'ĐANG XEM PHIÊN') : 'CHƯA BẮT ĐẦU';
  $('live-dot').classList.toggle('live', capturing);
}
$('start-session').addEventListener('click', async () => {
  busy = true; updateControls();
  try {
    const title = $('session-title').value.trim();
    if (!title) throw new Error('Hãy nhập tên phiên thi.');
    await connectCamera();
    notice('Đang nạp mô hình trên máy. Lần đầu có thể mất vài giây…', true);
    session = await api('/api/sessions', json({title,owner}));
    $('alert-count').textContent = '0';
    $('alert-badge').textContent = '0';
    $('alerts').innerHTML = '<div class="empty"><span class="empty-icon">✓</span><h3>Chưa có cảnh báo</h3><p>Cảnh báo tư thế lưu video, kèm thời gian để giám thị kiểm tra.</p></div>';
    $('transcripts').innerHTML = '<div class="empty compact">Đang chờ nội dung trao đổi…</div>';
    capturing = true;
    $('camera-label').textContent = 'ĐANG PHÂN TÍCH';
    $('notice').hidden = true;
    frameLoop();
    try { await startAudio(session); }
    catch (e) {
      await stopAudio();
      $('audio-status').textContent = 'Không mở được micro · chưa có bản chép lời';
      notice(`Camera đã chạy nhưng micro chưa hoạt động: ${e.message}. Kết thúc và bắt đầu lại sau khi cấp quyền micro.`, true);
    }
  } catch (e) { notice(e.message, true); }
  finally { busy = false; updateControls(); }
});
$('stop-session').addEventListener('click', async () => {
  const closingSession = session;
  if (!closingSession || busy) return;
  busy = true; capturing = false; updateControls();
  try {
    await stopAudio();
    await api('/api/sessions/stop', json({session_id: closingSession.id}));
    if (session?.id === closingSession.id) session = null;
    tracks = []; clearProcessedFrame();
    $('camera-label').textContent = camera ? 'ĐÃ KẾT THÚC · CHỈ XEM TRƯỚC' : 'CHƯA KẾT NỐI';
    $('audio-status').textContent = 'Đã ngừng thu · các đoạn đã nhận vẫn tiếp tục được chép';
    $('people').innerHTML = '<div class="empty compact">Phiên thi đã kết thúc. Xem dữ liệu tại Lịch sử phiên thi.</div>';
    notice('Đã lưu phiên thi. Các đoạn âm thanh đang chờ sẽ được lưu vào đúng phiên.');
    await loadSessions();
  } catch (e) { notice(e.message, true); }
  finally { busy = false; updateControls(); }
});
$('calibrate').addEventListener('click', async () => {
  try {
    const result = await api('/api/calibrate', json({owner}));
    notice(`Đã hiệu chuẩn tư thế ngồi cho ${result.calibrated} người. Giữ camera cố định trong phiên thi.`);
  } catch (e) { notice(e.message); }
});

async function frameLoop() {
  while (capturing && session) {
    const row = session;
    const started = performance.now();
    try {
      const blob = await new Promise((resolve) => captureCanvas().toBlob(resolve,'image/jpeg',0.8));
      if (!capturing) break;
      const result = await api(`/api/frames?owner=${encodeURIComponent(owner)}&capture_id=${row.capture_id}`, {
        method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob,
      });
      if (!capturing || session?.id !== row.id) break;
      const accepted = await framePair.accept(blob, result.tracks, (imageBlob) => createImageBitmap(imageBlob),
        () => capturing && session?.id === row.id);
      if (!accepted) break;
      tracks = framePair.current.tracks; trackAt = performance.now();
      $('camera-stage').classList.add('processing');
      $('camera-stage').style.aspectRatio = `${framePair.current.bitmap.width} / ${framePair.current.bitmap.height}`;
      const resultAt=performance.now();
      if(cameraResultAt!==null){const fps=1000/Math.max(1,resultAt-cameraResultAt);cameraAnalysisFps=cameraAnalysisFps ? .3*fps+.7*cameraAnalysisFps : fps;}
      cameraResultAt=resultAt;
      $('camera-stamp').textContent = `ẢNH ĐÃ PHÂN TÍCH · ${Math.round(resultAt-started)} ms · ${cameraAnalysisFps.toFixed(1)} FPS AI`;
      $('people-count').innerHTML = `${tracks.filter(t=>t.role!=='proctor').length} <small>/ ${maxPeople}</small>`;
      $('latency').innerHTML = `${result.inference_ms} <small>ms</small>`;
      renderPeople();
      if (result.overflow) notice('Có hơn 3 khuôn mặt. Bản demo chỉ xử lý 3 khuôn mặt lớn nhất.', true);
      lastError = '';
    } catch (e) {
      if (!capturing || session?.id !== row.id) break;
      tracks = []; clearProcessedFrame();
      if (e.status !== 429 && e.message !== lastError) { notice(`Hình ảnh: ${e.message}`, true); lastError = e.message; }
      if (e.status === 409 || e.message.includes('Chưa có phiên')) { capturing = false; updateControls(); break; }
      await new Promise((r) => setTimeout(r,500));
    }
    // One request in flight; immediately sample a fresh frame when inference is
    // slower than 10 Hz rather than adding another fixed delay or building a queue.
    await new Promise((r) => setTimeout(r,Math.max(0, 50 - (performance.now()-started))));
  }
}
function renderPeople() {
  $('people').innerHTML = tracks.length ? [...tracks].sort((a,b)=>(b.role==='proctor')-(a.role==='proctor')).map((t) => t.role==='proctor' ? `<div class="person proctor"><span class="avatar">GT</span><div class="person-text"><b>${esc(t.name)} · Giám thị</b><small>${t.dwell_student ? `Đứng cạnh bàn ${esc(t.dwell_student)} · ${Math.round(t.dwell_seconds)}s` : 'Đang di chuyển trong phòng'}</small></div><span class="tag ${t.dwell_seconds>=(window.examwatchDwell||60)*.8?'warn':''}">Giám thị</span></div>` : `<div class="person"><span class="avatar">${esc(t.student_id ? t.name.split(' ').map(s=>s[0]).slice(-2).join('') : '#'+t.id)}</span><div class="person-text"><b>${esc(t.name)}</b><small>${esc(postureNames[t.posture])}${t.mouth_activity>=2.5 ? ' · 🗣 đang nói' : ''}${t.drowsy ? ` · mắt nhắm ${Math.round(t.perclos*100)}%` : ''}${t.has_phone ? ' · Phát hiện điện thoại' : ''}${t.posture==='standing' ? ` · ${t.standing_seconds}s / ${standingThreshold}s` : ''}${t.negative_seconds ? ` · tiêu cực ${t.negative_seconds}s / 20s` : ''}</small>${poseWatchHTML(t)}</div><span class="tag ${t.has_phone||['anxious','sad','tired'].includes(moodOf(t))?'warn':''}">${t.has_phone ? 'Điện thoại · '+Math.round(t.phone_confidence*100)+'%' : `${esc(moodNames[moodOf(t)])} · ${esc(emotionNames[t.emotion])} ${t.emotion==='unknown'?'':Math.round(t.emotion_score*100)+'%'}`}</span></div>`).join('') : '<div class="empty compact">Chưa thấy khuôn mặt. Hãy kiểm tra góc camera và ánh sáng.</div>';
}
const connections = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]];
function draw() {
  const video = $('video'), canvas = $('overlay');
  if (video.videoWidth) {
    let pairedTracks = null;
    if (capturing) pairedTracks = framePair.paint(canvas);
    if (!pairedTracks) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) { canvas.width=video.videoWidth; canvas.height=video.videoHeight; }
      canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
    }
    const ctx=canvas.getContext('2d'), w=canvas.width,h=canvas.height;
    if (pairedTracks && performance.now()-trackAt < 1800) for (const t of pairedTracks) {
      const [x,y,bw,bh] = t.bbox;
      ctx.strokeStyle = t.role==='proctor' ? '#7fb7ff' : t.student_id ? '#c7e895' : '#f0cd86'; ctx.lineWidth=2;
      ctx.strokeRect(x*w,y*h,bw*w,bh*h);
      const label = t.role==='proctor' ? `${t.name} · Giám thị${t.dwell_seconds?` · ${Math.round(t.dwell_seconds)}s`:''}` : `${t.name} · ${t.has_phone ? 'Điện thoại' : t.head_turning ? 'Quay đầu' : ['standing','head_down','moving'].includes(t.posture) ? postureNames[t.posture] : moodNames[moodOf(t)]}`;
      ctx.font=`${Math.max(12,Math.round(w/65))}px Segoe UI`;
      const tw=ctx.measureText(label).width+14;
      const ly=Math.max(25,y*h-4);
      ctx.fillStyle='#173c30e8';ctx.fillRect(Math.max(0,x*w),ly-23,Math.min(tw,w-x*w),25);
      ctx.fillStyle='#e6f2c9';ctx.fillText(label,Math.max(0,x*w)+7,ly-6);
      if ($('show-pose').checked) {
        if(t.lean_origin&&t.lean_target&&t.pose_angles?.lean!=null){
          const ox=t.lean_origin[0]*w,oy=t.lean_origin[1]*h;
          const tx=t.lean_target[0]*w,ty=t.lean_target[1]*h;
          ctx.strokeStyle=t.pose_flags?.includes('lean')?'#ff935e':'#f3d77a';ctx.lineWidth=3;
          ctx.beginPath();ctx.moveTo(ox,oy);ctx.lineTo(ox,oy-Math.hypot(tx-ox,ty-oy));ctx.moveTo(ox,oy);ctx.lineTo(tx,ty);ctx.stroke();
          ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.arc(ox,oy,5,0,Math.PI*2);ctx.fill();
          ctx.fillText(`${Math.round(t.pose_angles.lean)}° / 20°`,ox+8,oy-10);
        }
        ctx.strokeStyle='#86d1de';ctx.fillStyle='#bce9e1';ctx.lineWidth=2;
        for (const [a,b] of connections) {
          const pa=t.landmarks[a], pb=t.landmarks[b];
          if (pa&&pb&&pa[3]>.6&&pb[3]>.6) { ctx.beginPath();ctx.moveTo(pa[0]*w,pa[1]*h);ctx.lineTo(pb[0]*w,pb[1]*h);ctx.stroke(); }
        }
        for (const [i,p] of t.landmarks.slice(0,25).entries()) if(i!==9&&i!==10&&p[3]>.6) {ctx.beginPath();ctx.arc(p[0]*w,p[1]*h,3,0,Math.PI*2);ctx.fill();}
        const mouth=t.mouth_points||[];
        if(mouth.length===3){
          ctx.beginPath();ctx.moveTo(mouth[0][0]*w,mouth[0][1]*h);
          for(const p of mouth.slice(1))ctx.lineTo(p[0]*w,p[1]*h);
          ctx.stroke();
          for(const p of mouth){ctx.beginPath();ctx.arc(p[0]*w,p[1]*h,3,0,Math.PI*2);ctx.fill();}
        }
      }
    }
    if (currentPage==='students' && video.readyState>=2) {
      const preview=$('enroll-preview'); preview.width=640;preview.height=Math.round(video.videoHeight*640/video.videoWidth);
      preview.getContext('2d').drawImage(video,0,0,preview.width,preview.height);
    }
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

const angleNames={front:'Chính diện',left:'Quay trái',right:'Quay phải'};
function renderSamples() {
  const complete=new Set(samples.map(s=>s.angle));
  document.querySelectorAll('.angle-step').forEach((step)=>{
    step.classList.toggle('active',step.dataset.angle===selectedAngle);
    step.classList.toggle('complete',complete.has(step.dataset.angle));
  });
  $('sample-count').textContent=`${samples.length}/${samples.length<3?3:8}`;
  $('take-sample').innerHTML=`＋ Chụp góc ${angleNames[selectedAngle]} <span id="sample-count">${samples.length}/${samples.length<3?3:8}</span>`;
  $('sample-thumbs').innerHTML=samples.map((s,i)=>`<img src="${s.data}" alt="Mẫu ${angleNames[s.angle]} ${i+1}" title="${angleNames[s.angle]}">`).join('');
  const missing=Object.keys(angleNames).filter(angle=>!complete.has(angle));
  $('angle-status').textContent=missing.length?`Còn thiếu: ${missing.map(a=>angleNames[a]).join(', ')}. Chọn góc rồi chụp mẫu.`:`Đã đủ ba góc. Bạn có thể chụp thêm mẫu ở góc đang chọn (tối đa 8).`;
}
document.querySelectorAll('.angle-step').forEach((step)=>step.addEventListener('click',()=>{selectedAngle=step.dataset.angle;renderSamples();}));
$('take-sample').addEventListener('click', () => {
  try { if(samples.length>=8) throw new Error('Đã đủ 8 mẫu.'); samples.push({data:captureCanvas().toDataURL('image/jpeg',.85),angle:selectedAngle});renderSamples(); }
  catch(e){notice(e.message);}
});
$('clear-samples').addEventListener('click',()=>{samples=[];selectedAngle='front';renderSamples();});
$('enroll-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const missing=Object.keys(angleNames).filter(angle=>!samples.some(s=>s.angle===angle));
  if(missing.length) return notice(`Cần chụp đủ ba góc: ${missing.map(a=>angleNames[a]).join(', ')}.`);
  busy=true;updateControls();
  try {
    await api('/api/students',json({student_id:$('student-id').value.trim(),name:$('student-name').value.trim(),images:samples.map(s=>s.data),role:$('student-role').value}));
    samples=[];renderSamples();$('enroll-form').reset();await loadStudents();notice('Đã lưu thông tin và đặc trưng khuôn mặt.');
  } catch(e){notice(e.message,true);}finally{busy=false;updateControls();}
});
async function loadStudents() {
  const students=await api('/api/students');
  $('student-count').textContent=`${students.filter(s=>s.role!=='proctor').length} THÍ SINH · ${students.filter(s=>s.role==='proctor').length} GIÁM THỊ`;
  $('student-list').innerHTML=students.length?students.map((s)=>`<div class="student-row"><span class="avatar">${esc(s.name.split(' ').slice(-1)[0][0])}</span><div class="person-text"><b>${esc(s.name)}</b><small>${esc(s.id)} · ${s.samples} mẫu${s.role==='proctor'?' · <b>Giám thị</b>':''}</small></div><button class="button danger" data-delete="${esc(s.id)}" ${session?'disabled':''}>Xóa</button></div>`).join(''):'<div class="empty">Chưa có thí sinh đăng ký.</div>';
}
$('student-list').addEventListener('click',async(e)=>{
  const button=e.target.closest('[data-delete]');if(!button)return;
  if(!confirm('Xóa thông tin đăng ký của thí sinh này? Các cảnh báo cũ vẫn được giữ.'))return;
  try{await api(`/api/students/${encodeURIComponent(button.dataset.delete)}`,{method:'DELETE'});await loadStudents();}catch(err){notice(err.message);}
});
function alertHTML(a) {
  const eventNames=window.examwatchEventNames;
  const video=a.evidence_type==='video';
  const evidence=a.evidence_type==='none' ? '<span class="tag">Không lưu bằng chứng</span>' : video
    ? a.evidence_status==='ready'
      ? `<button class="button secondary" data-video="${a.id}" data-caption="${esc(a.student_name+' · '+timestamp(a.created_at))}">Xem video ↗</button>`
      : `<span class="tag">${a.evidence_status==='failed'?'Không tạo được video':'Đang hoàn tất video…'}</span>`
    : `<button class="button secondary" data-image="${a.id}" data-caption="${esc(a.student_name+' · '+timestamp(a.created_at))}">Xem ảnh ↗</button>`;
  return `<div class="event"><span class="event-kind">${esc(eventNames[a.kind]||a.kind)}</span><div class="event-title"><span>${esc(a.student_name)} <span class="muted">#${a.track_id}</span></span><time>${duration(a.offset_seconds)}</time></div><p>${esc(a.detail)}</p><div class="event-actions">${evidence}${a.acknowledged?`<span class="tag decision-${esc(a.decision||'done')}">${esc(a.decision_label||'Đã kiểm tra')}</span><button class="button secondary" data-review="${a.id}">Biên bản</button>`:`<button class="button primary" data-review="${a.id}">Ghi nhận</button>`}</div></div>`;
}
const transcriptHTML=(t)=>`<div class="transcript"><time>${duration(t.start_seconds)} — ${duration(t.end_seconds)}</time>${t.speaker_name?`<span class="speaker">${esc(t.speaker_name)} <small>ước tính</small></span>`:''}<p>${esc(t.text)}</p>${t.has_audio?`<audio controls preload="none" src="/api/transcripts/${t.id}/audio"></audio>`:''}</div>`;
async function loadSessionData(id) {
  const data=await api(`/api/sessions/${id}`);
  if(session?.id===id) {
    const open=data.alerts.filter(a=>!a.acknowledged).length;
    $('alert-count').textContent=open;
    $('reviewed-count').textContent=data.alerts.length-open;
    $('alert-badge').textContent=open;
    if(data.attendance){$('roster-count').textContent=data.attendance.total;$('known-count').innerHTML=`${data.attendance.present} <small>/ ${data.attendance.total}</small>`;}
    window.examwatchLiveAlerts?.(data.alerts);
    if(data.alerts.length)$('alerts').innerHTML=data.alerts.map(alertHTML).join('');
    if(data.transcripts.length)$('transcripts').innerHTML=[...data.transcripts].reverse().map(transcriptHTML).join('');
  }
  if(viewing===id) {
    // Redraw only when the record changed: a 2 s poll must not reset the emotion
    // chart selection, scroll position or an audio clip that is playing.
    const key=JSON.stringify([id,data.session.ended_at,data.alerts.map(a=>[a.id,a.acknowledged,a.decision,a.evidence_status]),data.transcripts.length,data.attendance?.present]);
    if(key!==historyKey||!$('history-detail').querySelector('.history-content')){historyKey=key;renderHistory(data);}
  }
  lastRecords=data;
  if(viewing===id) window.examwatchHistoryRendered?.(data);
  return data;
}
async function loadSessions() {
  const sessions=await api('/api/sessions');
  $('session-list').innerHTML=sessions.length?sessions.map(s=>`<button class="session-row ${viewing===s.id?'selected':''}" data-session="${s.id}"><b>${esc(s.title)}</b><small>${timestamp(s.started_at)}</small><small>${s.ended_at?'Đã kết thúc':'Đang mở'}</small></button>`).join(''):'<div class="empty">Chưa có phiên thi.</div>';
  if(window.examWatchDemo&&!viewing&&sessions.length){viewing=sessions[0].id;await loadSessionData(viewing);await loadSessions();}
}
$('session-list').addEventListener('click',async(e)=>{
  const b=e.target.closest('[data-session]');if(!b)return;
  viewing=b.dataset.session;
  try{await loadSessionData(viewing);await loadSessions();}catch(err){notice(err.message);}
});
function renderHistory(data) {
  const row=data.session;
  $('history-detail').innerHTML=`<div class="card-heading"><h2>${esc(row.title)}</h2><span><button class="button primary" data-report="${row.id}">Báo cáo ca thi</button> <a class="button secondary" href="/api/sessions/${row.id}/export">Xuất JSON ↓</a></span></div><div class="history-content"><p>${timestamp(row.started_at)} — ${row.ended_at?timestamp(row.ended_at):'Đang hoạt động'}</p><p class="hint">Màn hình hiển thị tối đa 300 bản ghi mới nhất mỗi loại. File JSON chứa toàn bộ dữ liệu đã lưu.</p><h3>Cảnh báo</h3>${data.alerts.length?data.alerts.map(alertHTML).join(''):'<p>Không có cảnh báo.</p>'}<h3>Điểm danh</h3>${data.attendance?`<p>${data.attendance.present}/${data.attendance.total} thí sinh có mặt · vắng ${data.attendance.absent}</p>`:''}<h3>Biểu cảm theo thời gian</h3><div class="mood-chart" data-mood-session="${row.id}"><div class="empty compact">Đang tải biểu đồ…</div></div><h3>Nội dung trao đổi</h3>${data.transcripts.length?[...data.transcripts].reverse().map(transcriptHTML).join(''):'<p>Chưa có bản chép lời.</p>'}<h3>Diễn biến quan sát</h3><div class="table-wrap"><table><thead><tr><th>Mốc phiên</th><th>Thí sinh</th><th>Cảm xúc</th><th>Tư thế</th></tr></thead><tbody>${data.observations.map(o=>`<tr><td>${duration(o.offset_seconds)}</td><td>${esc(o.student_name)} #${o.track_id}</td><td>${esc(moodNames[o.mood]||'')} · ${esc(emotionNames[o.emotion])} ${Math.round(o.confidence*100)}%</td><td>${esc(postureNames[o.posture])}</td></tr>`).join('')}</tbody></table></div></div>`;
}
document.addEventListener('click',async(e)=>{
  const image=e.target.closest('[data-image]');
  if(image){$('evidence-video').hidden=true;$('evidence-image').hidden=false;$('evidence-image').src=`/api/alerts/${image.dataset.image}/image`;$('evidence-caption').textContent=image.dataset.caption;$('evidence-dialog').showModal();}
  const video=e.target.closest('[data-video]');
  if(video){$('evidence-image').hidden=true;$('evidence-video').hidden=false;$('evidence-video').src=`/api/alerts/${video.dataset.video}/video`;$('evidence-caption').textContent=video.dataset.caption;$('evidence-dialog').showModal();}
  const review=e.target.closest('[data-review]');
  if(review){window.openReview(review.dataset.review);return;}
  const report=e.target.closest('[data-report]');
  if(report){window.openReport(report.dataset.report);return;}
  const ack=e.target.closest('[data-ack]');
  if(ack)try{await api(`/api/alerts/${ack.dataset.ack}/acknowledge`,json({}));if(session)await loadSessionData(session.id);if(viewing&&viewing!==session?.id)await loadSessionData(viewing);}catch(err){notice(err.message);}
});
$('close-evidence').addEventListener('click',()=>$('evidence-dialog').close());
$('evidence-dialog').addEventListener('close',()=>{$('evidence-video').pause();$('evidence-video').removeAttribute('src');$('evidence-video').load();});

async function poll() {
  try {
    const status=await api('/api/status');
    standingThreshold=status.standing_threshold_seconds||5;
    maxPeople=status.max_people||maxPeople; frameWidth=status.frame_width||frameWidth;
    $('people-limit').textContent=`TỐI ĐA ${maxPeople} NGƯỜI`;
    window.examwatchStatus?.(status);
    $('standing-threshold-hint').textContent=standingThreshold;
    if(!standingSettingsDirty&&document.activeElement!==$('standing-threshold'))$('standing-threshold').value=standingThreshold;
    $('connection-dot').style.color='#7b9d5c';
    if ($('notice').textContent.startsWith('Mất kết nối máy chủ:')) $('notice').hidden = true;
    const voiceName = {'zipformer-vi-30m-lora-low-lr20':'Zipformer LoRA LR thấp · epoch 10','zipformer-vi-30m-lora':'Zipformer 30M LoRA · epoch 6','zipformer-vi-30m':'Zipformer 30M gốc INT8','phowhisper-base':'PhoWhisper gốc INT8','phowhisper-vietsuper':'PhoWhisper đã huấn luyện INT8','phowhisper-tiny':'PhoWhisper tiny gốc INT8','phowhisper-tiny-lora':'PhoWhisper tiny LoRA INT8'}[status.voice_model] || `Whisper ${status.voice_model || 'tiny'} INT8`;
    const names={face_detector:'YuNet',identity:'SFace',emotion:'Emotion ONNX',pose:'Pose Lite',phone_detector:'YOLO Phone',voice:voiceName};
    const firebaseState=status.firebase_sync?.state||'not_configured';
    const firebaseLabel={connected:'Firebase · Đã kết nối',degraded:`Firebase · Chờ đồng bộ (${status.firebase_sync?.pending||0})`,error:'Firebase · Lỗi cấu hình',not_configured:'Firebase · Chưa cấu hình'}[firebaseState]||`Firebase · ${firebaseState}`;
    $('models').innerHTML=Object.entries(status.models).map(([k,v])=>`<span class="${v?'':'missing'}">${v?'●':'○'} ${names[k]}</span>`).join('')+(!Object.values(status.models).every(Boolean)?'<span class="model-help missing">Thiếu model · chạy prepare.ps1</span>':'<span class="model-help">CPU · Sẵn sàng offline</span>')+`<span class="model-help ${firebaseState==='degraded'||firebaseState==='error'?'missing':''}">${firebaseLabel}</span>`;
    if(!busy) {
      if(status.session?.id !== session?.id) {
        const incoming = status.session;
        capturing=false;tracks=[];clearProcessedFrame();await stopAudio();
        session=incoming ? {...incoming,capture_id:status.capture_id} : null;
        if(session) {
          $('session-title').value=session.title;
          notice('Đang xem phiên đã mở. Bạn có thể kết thúc phiên tại dashboard này; bắt đầu phiên mới để thu camera và micro ở cửa sổ hiện tại.',true);
        }
        updateControls();
        if (currentPage === 'history') await loadSessions();
      }
    }
    if(session) {
      $('session-clock').textContent=duration((Date.now()-new Date(session.started_at).getTime())/1000);
      await loadSessionData(session.id);
    } else if(currentPage==='history'&&viewing) await loadSessionData(viewing);
    if(status.audio_error) notice(`Lỗi chép lời: ${status.audio_error}`,true);
    else if(status.audio_busy||status.audio_pending)$('audio-status').textContent=`Đang chép lời · ${status.audio_pending} đoạn chờ`;
    else if(status.audio_quality_warning)$('audio-status').textContent=status.audio_quality_warning;
    else if(capturing&&audioContext)$('audio-status').textContent='Đang nghe · chép sau khoảng ngừng nói';
    if(capturing&&!status.camera_fresh&&performance.now()-trackAt>3500){tracks=[];$('camera-label').textContent='CHƯA CÓ KẾT QUẢ MỚI';}
    else if(capturing)$('camera-label').textContent='ĐANG PHÂN TÍCH';
  } catch(e){$('connection-dot').style.color='#b84945';notice(`Mất kết nối máy chủ: ${e.message}`,true);}
  setTimeout(poll,2000);
}
window.addEventListener('beforeunload',(event)=>{if(session){event.preventDefault();event.returnValue='';}});
window.addEventListener('examwatch-auth-ready',async(event)=>{
  updateControls();
  if(event.detail?.demo){
    window.examWatchDemo=true;$('demo-badge').hidden=false;
    const requested=new URLSearchParams(location.search).get('demo_page');
    navigate(['overview','monitor','students','history','settings'].includes(requested)?requested:(requested==='evidence'?'history':'overview'));
    if(requested==='evidence'){
      await new Promise(resolve=>setTimeout(resolve,800));
      document.querySelector('[data-video]')?.click();
      try{await $('evidence-video').play();}catch{}
    }
  }
  poll();
},{once:true});
