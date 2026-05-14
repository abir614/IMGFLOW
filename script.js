// ═══════════════════════════════════════
// STATE
// ═══════════════════════════════════════
let flow = 1, queue = [], results = [];
let _previewSrc = { w: 0, h: 0 };

// ═══════════════════════════════════════
// INLINE SERVER-PROGRESS OVERLAY
// Replaces the old model-download overlay —
// now shows server processing status
// ═══════════════════════════════════════
let _inlineOverlay = null;

function showInlineProgress(title, icon = '⚙️') {
  if (_inlineOverlay) return;
  _inlineOverlay = document.createElement('div');
  _inlineOverlay.id = 'inlineOverlay';
  _inlineOverlay.innerHTML = `
    <div class="mo-card">
      <div class="mo-icon-wrap">
        <svg class="mo-spinner" width="48" height="48" viewBox="0 0 48 48" fill="none">
          <circle cx="24" cy="24" r="20" stroke="rgba(198,241,53,0.12)" stroke-width="3"/>
          <path d="M24 4 A20 20 0 0 1 44 24" stroke="var(--a)" stroke-width="3" stroke-linecap="round"/>
        </svg>
        <span class="mo-icon-inner" id="ipoIcon">${icon}</span>
      </div>
      <div class="mo-title" id="ipoTitle">${title}</div>
      <div class="mo-sub" id="ipoSub">Sending to server…</div>
      <div class="mo-track">
        <div class="mo-bar-wrap">
          <div class="mo-bar" id="ipoBar" style="background:linear-gradient(90deg,var(--a2),var(--a))"></div>
          <div class="mo-bar-glow" id="ipoBarGlow" style="background:radial-gradient(ellipse,rgba(198,241,53,.55),transparent 70%)"></div>
        </div>
        <div class="mo-pct" id="ipoPct" style="color:var(--a)">—</div>
      </div>
      <div class="mo-files" id="ipoDetail"></div>
      <div class="mo-note">Processing on server · no downloads required</div>
    </div>`;
  document.body.appendChild(_inlineOverlay);
  requestAnimationFrame(() => _inlineOverlay.classList.add('visible'));
}

function updateInlineProgress(pct, sub, detail) {
  const bar    = document.getElementById('ipoBar');
  const glow   = document.getElementById('ipoBarGlow');
  const pctEl  = document.getElementById('ipoPct');
  const subEl  = document.getElementById('ipoSub');
  const detEl  = document.getElementById('ipoDetail');
  if (bar)   { bar.style.width = pct + '%'; }
  if (glow)  { glow.style.left = pct + '%'; }
  if (pctEl) pctEl.textContent = typeof pct === 'number' ? Math.round(pct) + '%' : pct;
  if (subEl  && sub    != null) subEl.textContent  = sub;
  if (detEl  && detail != null) detEl.textContent  = detail;
}

function hideInlineProgress() {
  if (!_inlineOverlay) return;
  _inlineOverlay.classList.add('hiding');
  setTimeout(() => { _inlineOverlay?.remove(); _inlineOverlay = null; }, 500);
}

// ═══════════════════════════════════════
// FLOW UI  (unchanged from original)
// ═══════════════════════════════════════
function selectFlow(n) {
  flow = n;
  document.querySelectorAll('.ftab').forEach((b, i) => b.classList.toggle('active', i === n - 1));

  document.getElementById('bgModelGroup').classList.toggle('sg-hidden',    n !== 2);
  document.getElementById('bgSensGroup').classList.toggle('sg-hidden',     n !== 2);
  document.getElementById('bgFeatherGroup').classList.toggle('sg-hidden',  n !== 2);
  document.getElementById('bgOutputGroup').classList.toggle('sg-hidden',   n !== 2);
  document.getElementById('bgInfo').classList.toggle('sg-hidden',          n !== 2);

  document.querySelectorAll('.resize-setting').forEach(el => el.classList.toggle('sg-hidden', n !== 3));
  document.getElementById('resizeInfo').classList.toggle('sg-hidden',    n !== 3);
  document.getElementById('resizePreview').classList.toggle('sg-hidden', n !== 3);

  toggleFillColor();
  onResizeModeChange();
  renderTrack();
  if (n === 3) updateResizePreview();
}

function onResizeModeChange() {
  if (flow !== 3) return;
  const mode = document.getElementById('resizeMode')?.value;
  const blendGroup = document.getElementById('resizeBlendGroup');
  if (blendGroup) blendGroup.classList.toggle('sg-hidden', mode !== 'smart-crop-extend');
  const focusGroup = document.getElementById('resizeFocusGroup');
  if (focusGroup) focusGroup.classList.toggle('sg-hidden', mode !== 'smart-crop-extend');
  const fillEl = document.getElementById('resizeFill');
  if (!fillEl) return;
  const prevVal = fillEl.value;
  const hasBluryOption = [...fillEl.options].some(o => o.value === 'blur');
  const hasAIOption    = [...fillEl.options].some(o => o.value === 'ai-extend');
  if (mode === 'proportional') {
    if (hasAIOption) [...fillEl.options].find(o => o.value === 'ai-extend')?.remove();
    if (!hasBluryOption) {
      const opt = document.createElement('option');
      opt.value = 'blur'; opt.textContent = 'Blurred Source (cinematic)';
      fillEl.insertBefore(opt, fillEl.options[1]);
    }
    if (fillEl.value === 'ai-extend' || !fillEl.value) fillEl.value = 'extend';
  } else {
    if (hasBluryOption) [...fillEl.options].find(o => o.value === 'blur')?.remove();
    if (!hasAIOption) {
      const opt = document.createElement('option');
      opt.value = 'ai-extend'; opt.textContent = '🤖 AI Environment Fill (smart)';
      const extOpt = [...fillEl.options].find(o => o.value === 'extend');
      if (extOpt) fillEl.insertBefore(opt, extOpt.nextSibling);
      else fillEl.appendChild(opt);
    }
    if (prevVal === 'blur' || !prevVal) fillEl.value = 'ai-extend';
  }
  toggleFillColor();
  updateResizePreview();
}

function renderTrack() {
  let steps;
  if (flow === 1) steps = [['🔎','Lanczos\nUpscale'],['📐','Shopify\nResize'],['🌐','WebP\nEncode']];
  else if (flow === 2) steps = [['🧠','ISNet AI\nBG Remove'],['✨','Refine\nEdges'],['🔎','Lanczos\nUpscale'],['🌐','WebP\nEncode']];
  else steps = [['🔍','Detect\nSubject'],['🔎','Upscale'],['⚡','Smart\nResize'],['🌐','Encode']];

  document.getElementById('pipeTrack').innerHTML = steps.map((s, i) => `
    <div class="pt-step">
      <div class="pt-icon" id="ptd-${i}">${s[0]}</div>
      <div class="pt-label" id="ptl-${i}">${s[1].replace('\n','<br>')}</div>
    </div>
    ${i < steps.length - 1 ? `<div class="pt-line" id="ptln-${i}"></div>` : ''}
  `).join('');
}

function pipeStep(i, state) {
  const d = document.getElementById('ptd-' + i);
  const l = document.getElementById('ptln-' + i);
  if (d) { d.classList.remove('running','done'); if (state) d.classList.add(state); }
  if (state === 'done' && l) l.classList.add('done');
}

function resetTrack() {
  document.querySelectorAll('.pt-icon').forEach(d => d.classList.remove('running','done'));
  document.querySelectorAll('.pt-line').forEach(l => l.classList.remove('done'));
}

function toggleFillColor() {
  const mode = document.getElementById('resizeFill')?.value;
  const cg   = document.getElementById('fillColorGroup');
  if (!cg) return;
  cg.classList.toggle('sg-hidden', mode !== 'color' || flow !== 3);
}

// ─── Live resize preview (browser-only, purely cosmetic)  ────────────────
function updateResizePreview() {
  if (flow !== 3) return;
  const tw = parseInt(document.getElementById('resizeW')?.value) || 0;
  const th = parseInt(document.getElementById('resizeH')?.value) || 0;
  const mode = document.getElementById('resizeMode')?.value || 'smart-crop-extend';

  const rpSrc    = document.getElementById('rpSrc');
  const rpTarget = document.getElementById('rpTarget');
  const rpW      = document.getElementById('rpW');
  const rpH      = document.getElementById('rpH');

  if (rpTarget) rpTarget.textContent = tw && th ? `${tw} × ${th}` : '—';

  if (!_previewSrc.w || !tw || !th) {
    if (rpSrc) rpSrc.textContent = _previewSrc.w ? `${_previewSrc.w} × ${_previewSrc.h}` : '(drop image)';
    if (rpW) { rpW.textContent = 'W: —'; rpW.className = 'rp-axis'; }
    if (rpH) { rpH.textContent = 'H: —'; rpH.className = 'rp-axis'; }
    return;
  }
  if (rpSrc) rpSrc.textContent = `${_previewSrc.w} × ${_previewSrc.h}`;

  if (mode === 'proportional') {
    const ratio = Math.min(tw / _previewSrc.w, th / _previewSrc.h);
    const fw = Math.round(_previewSrc.w * ratio), fh = Math.round(_previewSrc.h * ratio);
    const padW = tw - fw, padH = th - fh;
    if (rpW) { rpW.textContent = `W: scale → ${fw}px (${padW>0?'+'+padW+' pad':'no pad'})`; rpW.className='rp-axis fit'; }
    if (rpH) { rpH.textContent = `H: scale → ${fh}px (${padH>0?'+'+padH+' pad':'no pad'})`; rpH.className='rp-axis fit'; }
  } else {
    const sW=_previewSrc.w, sH=_previewSrc.h, sAR=sW/sH, tAR=tw/th;
    let wLabel,hLabel,wCls,hCls;
    if (sW<tw&&sH<th) {
      wLabel=`W: ${sW}px → extend → ${tw}px`; wCls='extend';
      hLabel=`H: ${sH}px → extend → ${th}px`; hCls='extend';
    } else if (sW>=tw&&sH>=th) {
      if (Math.abs(sAR-tAR)<0.005) { wLabel=`W: scale → ${tw}px`; wCls='match'; hLabel=`H: scale → ${th}px`; hCls='match'; }
      else if (sAR>tAR) { wLabel=`W: crop → ${tw}px (subject-centred)`; wCls='crop'; hLabel=`H: fit ✓`; hCls='match'; }
      else { wLabel=`W: fit ✓`; wCls='match'; hLabel=`H: crop → ${th}px (subject-centred)`; hCls='crop'; }
    } else if (sW<tw) { wLabel=`W: extend → ${tw}px`; wCls='extend'; hLabel=`H: crop → ${th}px`; hCls='crop'; }
    else { wLabel=`W: crop → ${tw}px`; wCls='crop'; hLabel=`H: extend → ${th}px`; hCls='extend'; }
    if (rpW) { rpW.textContent=wLabel; rpW.className='rp-axis '+wCls; }
    if (rpH) { rpH.textContent=hLabel; rpH.className='rp-axis '+hCls; }
  }
}

// ═══════════════════════════════════════
// FILE INPUT
// ═══════════════════════════════════════
const dz = document.getElementById('dropZone');
const fi = document.getElementById('fileInput');
dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); addFiles([...e.dataTransfer.files]); });
fi.addEventListener('change', () => { addFiles([...fi.files]); fi.value = ''; });

function addFiles(files) {
  files.filter(f => f.type.startsWith('image/')).forEach(f => {
    if (queue.find(q => q.name === f.name && q.size === f.size)) return;
    const item = { id: Date.now() + Math.random(), file: f, name: f.name, size: f.size };
    queue.push(item);
    renderQI(item);
  });
  syncQ();
}

function renderQI(item) {
  document.getElementById('emptyMsg')?.remove();
  const div = document.createElement('div');
  div.className = 'qi'; div.id = 'qi-' + item.id;
  div.innerHTML = `
    <img class="qi-thumb" id="qt-${item.id}" src="">
    <div>
      <div class="qi-name">${item.name}</div>
      <div class="qi-meta"><span>${(item.size/1024).toFixed(0)} KB</span><span id="qdm-${item.id}">—</span></div>
      <div class="qpb" id="qpb-${item.id}"><div class="qpbr" id="qpbr-${item.id}"></div></div>
    </div>
    <span class="qs sw" id="qs-${item.id}">WAITING</span>`;
  document.getElementById('queueList').appendChild(div);

  const r = new FileReader();
  r.onload = e => { const el = document.getElementById('qt-' + item.id); if (el) el.src = e.target.result; };
  r.readAsDataURL(item.file);

  const img = new Image();
  img.onload = () => {
    const d = document.getElementById('qdm-' + item.id);
    if (d) d.textContent = img.width + '×' + img.height;
    _previewSrc = { w: img.width, h: img.height };
    updateResizePreview();
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(item.file);
}

function syncQ() {
  document.getElementById('qCount').textContent = queue.length + (queue.length === 1 ? ' file' : ' files');
  document.getElementById('runBtn').disabled = queue.length === 0;
}

function clearQueue() {
  queue = [];
  _previewSrc = { w: 0, h: 0 };
  document.getElementById('queueList').innerHTML = '<div class="empty" id="emptyMsg">No images queued</div>';
  syncQ();
  updateResizePreview();
}

function setQS(id, cls, txt) { const e = document.getElementById('qs-'+id); if (e) { e.className='qs '+cls; e.textContent=txt; } }
function setPB(id, show, pct) {
  const pb = document.getElementById('qpb-'+id), pbr = document.getElementById('qpbr-'+id);
  if (pb) pb.classList.toggle('on', show); if (pbr) pbr.style.width = pct+'%';
}
function log(msg, cls='') {
  const w = document.getElementById('logWrap');
  w.classList.add('on');
  const d = document.createElement('div');
  d.className='log-line '+cls; d.textContent='» '+msg;
  w.appendChild(d); w.scrollTop = w.scrollHeight;
}

// ═══════════════════════════════════════════════════════════════════
// PIPELINE RUNNER  ← Now calls the server API
// ═══════════════════════════════════════════════════════════════════

async function runPipeline() {
  if (!queue.length) return;
  results = [];
  document.getElementById('resGrid').innerHTML = '';
  document.getElementById('resultsSection').style.display = 'none';
  document.getElementById('logWrap').innerHTML = '';
  document.getElementById('logWrap').classList.remove('on');
  document.getElementById('runBtn').disabled = true;

  // ── Collect settings from the UI ──
  const resizeMode = document.getElementById('resizeMode')?.value || 'smart-crop-extend';
  const cfg = {
    flow:           flow,
    upscale_factor: parseFloat(document.getElementById('upscaleFactor').value),
    upscale_method: document.getElementById('upscaleMethod').value,
    shopify_size:   parseInt(document.getElementById('shopifySize').value),
    webp_quality:   parseInt(document.getElementById('webpQ').value),
    max_kb:         parseInt(document.getElementById('maxKB').value),
    rename_pattern: document.getElementById('renamePattern').value.trim(),
    bg_model:       document.getElementById('bgModel')?.value || 'isnet',
    alpha_threshold: parseInt(document.getElementById('bgSens')?.value) || 80,
    feather:        parseInt(document.getElementById('bgFeather')?.value) || 1,
    output_format:  document.getElementById('bgOutputFormat')?.value || 'webp',
    resize_w:       parseInt(document.getElementById('resizeW')?.value) || 1200,
    resize_h:       parseInt(document.getElementById('resizeH')?.value) || 1200,
    resize_mode:    resizeMode,
    resize_focus:   document.getElementById('resizeFocus')?.value || 'smart',
    resize_align:   document.getElementById('resizeAlign')?.value || 'center',
    resize_blend:   parseInt(document.getElementById('resizeBlend')?.value) || 40,
    resize_fill:    document.getElementById('resizeFill')?.value || 'extend',
    fill_color:     document.getElementById('fillColor')?.value || '#ffffff',
  };

  // ── Process each image ──
  for (const item of queue) {
    setQS(item.id, 'sp', 'PROCESSING');
    setPB(item.id, true, 5);
    resetTrack();
    log(`Processing: ${item.name}`);

    try {
      const flowIcons = { 1:'🔎', 2:'🧠', 3:'⚡' };
      const flowNames = { 1:`Upscaling ×${cfg.upscale_factor}`, 2:'ISNet BG Removal', 3:'Smart Resize' };
      showInlineProgress(flowNames[flow] || 'Processing…', flowIcons[flow] || '⚙️');
      updateInlineProgress(10, 'Uploading to server…', item.name);

      // ── Build FormData ──
      const fd = new FormData();
      fd.append('file', item.file);
      Object.entries(cfg).forEach(([k, v]) => fd.append(k, v));

      // Animate a fake progress bar while waiting
      let fakeP = 10;
      const fakeTick = setInterval(() => {
        fakeP = Math.min(fakeP + (85 - fakeP) * 0.06, 84);
        updateInlineProgress(fakeP, 'Server processing…', item.name);
        setPB(item.id, true, Math.round(fakeP));
      }, 400);

      // ── Send to API ──
      const response = await fetch('/api/process', { method: 'POST', body: fd });

      clearInterval(fakeTick);
      updateInlineProgress(95, 'Receiving result…', '');

      if (!response.ok) {
        let errMsg = `Server error ${response.status}`;
        try { const j = await response.json(); errMsg = j.detail || errMsg; } catch {}
        throw new Error(errMsg);
      }

      const blob = await response.blob();

      // ── Parse response headers ──
      const outName = response.headers.get('X-IMGFLOW-Name') || `output_${item.name}`;
      const dims    = response.headers.get('X-IMGFLOW-Dims') || '—';
      const srvLog  = response.headers.get('X-IMGFLOW-Log')  || '';
      const srvTime = response.headers.get('X-IMGFLOW-Time') || '';

      updateInlineProgress(100, `Done ✓  (${srvTime})`, '');
      hideInlineProgress();

      // Read as dataURL for preview
      const dataURL = await _blobToDataURL(blob);

      const out = {
        id: item.id,
        name: outName,
        blob,
        dataURL,
        orig: item.size,
        size: blob.size,
        dims,
      };
      results.push(out);
      addResult(out);
      setQS(item.id, 'sd', 'DONE');
      setPB(item.id, true, 100);

      if (srvLog) srvLog.split(' | ').forEach(l => log(`  ✓ ${l}`, 'ok'));
      log(`  ✓ ${outName} — ${(blob.size/1024).toFixed(0)} KB  ${srvTime}`, 'ok');

      // Mark all pipe steps done
      const stepCount = flow === 2 ? 4 : 3;
      for (let i = 0; i < stepCount; i++) pipeStep(i, 'done');

    } catch (err) {
      hideInlineProgress();
      setQS(item.id, 'se', 'ERROR');
      log(`  ✗ ${err.message}`, 'err');
      console.error(err);
    }

    setPB(item.id, false, 0);
    resetTrack();
  }

  document.getElementById('statsCount').textContent = results.length;
  document.getElementById('resultsSection').style.display = 'block';
  document.getElementById('resCount').textContent = results.length + ' file' + (results.length !== 1 ? 's' : '');
  document.getElementById('runBtn').disabled = false;
  log(`Done. ${results.length} file(s) ready.`, 'ok');
}

// Blob → data URL
function _blobToDataURL(blob) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload  = () => res(r.result);
    r.onerror = () => rej(new Error('FileReader failed'));
    r.readAsDataURL(blob);
  });
}

// ═══════════════════════════════════════
// RESULT CARDS
// ═══════════════════════════════════════
function addResult(r) {
  const s = Math.round((1 - r.size / r.orig) * 100);
  const card = document.createElement('div');
  card.className = 'rc';
  card.innerHTML = `
    <img class="rc-img" src="${r.dataURL}" alt="${r.name}">
    <div class="rc-body">
      <div class="rc-name">${r.name}</div>
      <div class="rc-stats">
        <span class="pill py">${(r.size/1024).toFixed(0)} KB</span>
        <span class="pill ${s>0?'pg':'pr'}">${s>0?'↓':'↑'}${Math.abs(s)}%</span>
        <span class="pill pg">${r.dims}</span>
      </div>
      <button class="dl-btn" onclick="dlOne('${r.id}')">⬇ DOWNLOAD</button>
    </div>`;
  document.getElementById('resGrid').appendChild(card);
  document.getElementById('resultsSection').style.display = 'block';
}

function dlOne(id) {
  const r = results.find(x => String(x.id) === String(id));
  if (!r) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(r.blob);
  a.download = r.name;
  a.click();
}

async function downloadAll() {
  if (!results.length) return;
  const zip = new JSZip();
  results.forEach(r => zip.file(r.name, r.blob));
  const blob = await zip.generateAsync({ type: 'blob' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'imgflow_pipeline.zip';
  a.click();
}

// ═══════════════════════════════════════
// PRESETS
// ═══════════════════════════════════════
function applyPreset(p) {
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.toggle('active', b.dataset.preset === p));
  if (p === 'quick') {
    document.getElementById('upscaleMethod').value = 'bicubic';
    document.getElementById('upscaleFactor').value = '2';
    document.getElementById('webpQ').value = 75; document.getElementById('qv').textContent = 75;
    document.getElementById('maxKB').value = 300;
  } else if (p === 'balanced') {
    document.getElementById('upscaleMethod').value = 'lanczos';
    document.getElementById('upscaleFactor').value = '2';
    document.getElementById('webpQ').value = 85; document.getElementById('qv').textContent = 85;
    document.getElementById('maxKB').value = 500;
  } else {
    document.getElementById('upscaleMethod').value = 'lanczos';
    document.getElementById('upscaleFactor').value = '3';
    document.getElementById('webpQ').value = 95; document.getElementById('qv').textContent = 95;
    document.getElementById('maxKB').value = 2000;
  }
}

// ═══════════════════════════════════════
// INIT
// ═══════════════════════════════════════
renderTrack();
