/* FvF Bellows — Bespoke Local Classifier Forge Frontend */
const ext = typeof browser !== 'undefined' ? browser : (typeof chrome !== 'undefined' ? chrome : null);

const bellowsState = {
  step: 1,
  preflight: null,
  modality: 'image',
  prompt: '',
  jobId: null,
  tier: 'balanced',
  counts: { positives: 0, hard_negatives: 0, base_negatives: 0 },
  pollTimer: null,
  finalizedCategory: false
};

const pendingBridge = new Map();

function sendBridge(action, payload = {}, timeoutMs = 15000) {
  const hasFirefoxApi =
    typeof browser !== 'undefined' &&
    browser.runtime &&
    browser.runtime.id &&
    typeof browser.runtime.sendMessage === 'function';
  if (hasFirefoxApi) {
    return new Promise((resolve) => {
      let settled = false;
      const timer = setTimeout(() => {
        if (!settled) {
          settled = true;
          resolve({ ok: false, error: 'Extension bridge timeout' });
        }
      }, timeoutMs);
      try {
        Promise.resolve(browser.runtime.sendMessage({ ...payload, type: action }))
          .then((resp) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            resolve(resp || { ok: true });
          })
          .catch((err) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            resolve({ ok: false, error: err?.message || String(err) });
          });
      } catch (err) {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve({ ok: false, error: err?.message || String(err) });
        }
      }
    });
  }

  const hasChromeApi =
    typeof chrome !== 'undefined' &&
    chrome.runtime &&
    chrome.runtime.id &&
    typeof chrome.runtime.sendMessage === 'function';
  if (hasChromeApi) {
    return new Promise((resolve) => {
      let settled = false;
      const timer = setTimeout(() => {
        if (!settled) {
          settled = true;
          resolve({ ok: false, error: 'Extension bridge timeout' });
        }
      }, timeoutMs);
      try {
        chrome.runtime.sendMessage({ ...payload, type: action }, (resp) => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          if (chrome.runtime.lastError) {
            resolve({ ok: false, error: chrome.runtime.lastError.message });
            return;
          }
          resolve(resp || { ok: true });
        });
      } catch (err) {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve({ ok: false, error: err?.message || String(err) });
        }
      }
    });
  }

  return new Promise((resolve) => {
    const requestId = `blw-${Math.random().toString(36).slice(2)}-${Date.now()}`;
    const timer = setTimeout(() => {
      if (pendingBridge.has(requestId)) {
        pendingBridge.delete(requestId);
        resolve({ ok: false, error: 'Bridge timeout' });
      }
    }, timeoutMs);
    pendingBridge.set(requestId, { resolve, timer });
    window.postMessage({ source: 'fvf-onboarding-page', action, payload, requestId }, '*');
  });
}

function handleBridgePush(msg) {
  if (!msg) return;
  if (msg.type === 'BELLOWS_PROGRESS' && msg.payload) {
    const p = msg.payload;
    if (p.stage === 'ethics') {
      const box = document.getElementById('ethics-status-box');
      if (box) {
        box.hidden = false;
        box.className = 'bellows-alert';
        box.textContent = p.message || 'Checking prompt…';
      }
    } else if (p.stage === 'base_dataset') {
      const fill = document.getElementById('base-dataset-progress-fill');
      const txt = document.getElementById('base-dataset-status-text');
      if (fill && typeof p.percent === 'number') fill.style.width = `${p.percent}%`;
      if (txt && p.message) txt.textContent = p.message;
    }
  } else if (msg.type === 'BELLOWS_COLLECTOR_COUNTS' && msg.payload) {
    if (msg.payload.counts) {
      bellowsState.counts = msg.payload.counts;
      updateDatasetCountsUI();
    }
  }
}

window.addEventListener('message', (event) => {
  if (event.source !== window || !event.data || event.data.source !== 'fvf-extension-bridge') return;
  const d = event.data;
  if (d.type === 'BELLOWS_PROGRESS' || d.type === 'BELLOWS_COLLECTOR_COUNTS') {
    handleBridgePush(d);
    return;
  }
  if (d.requestId && pendingBridge.has(d.requestId)) {
    const { resolve, timer } = pendingBridge.get(d.requestId);
    clearTimeout(timer);
    pendingBridge.delete(d.requestId);
    resolve(d.response);
  }
});

if (ext && ext.runtime && ext.runtime.onMessage) {
  try {
    ext.runtime.onMessage.addListener((msg) => handleBridgePush(msg));
  } catch (_) {}
}

function goBellowsStep(n) {
  bellowsState.step = n;
  document.querySelectorAll('.wizard-screen').forEach((el) => {
    el.classList.toggle('active', el.id === `bellows-screen-${n}`);
  });
  document.querySelectorAll('#bellows-step-pills .step-pill').forEach((el) => {
    const s = Number(el.dataset.step);
    el.classList.toggle('active', s === n);
    el.classList.toggle('done', s < n);
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (n === 4) renderTiers();
}

function fmtMinutes(min) {
  if (min < 60) return `~${min} min`;
  return `~${(min / 60).toFixed(1)} hours`;
}

async function runPreflight() {
  const res = await sendBridge('BELLOWS_PREFLIGHT', {}, 15000);
  if (!res || !res.ok) {
    const blocker = document.getElementById('preflight-blocker-msg');
    blocker.hidden = false;
    blocker.textContent = `Could not reach the Native Host (${res?.error || 'disconnected'}). Ensure Fire vs Fire Native Host is installed and running.`;
    return;
  }
  bellowsState.preflight = res;

  const pwrBox = document.getElementById('stat-power-box');
  const pwrVal = document.getElementById('stat-power-val');
  const plugged = Boolean(res.power?.is_plugged_in);
  const pct = res.power?.battery_percent != null ? `${res.power.battery_percent}%` : '';
  pwrVal.textContent = plugged ? `Plugged In (AC) ${pct}` : `On Battery (${pct || 'unplugged'})`;
  pwrBox.className = `bellows-stat-box ${plugged ? 'is-ok' : 'is-bad'}`;

  document.getElementById('stat-hw-val').textContent = `${res.hardware?.name || 'CPU'} (${res.hardware?.backend || 'CPU'})`;
  document.getElementById('stat-ram-val').textContent = `${Math.round((res.memory?.ram_avail_mb || 0) / 102.4) / 10} / ${Math.round((res.memory?.ram_total_mb || 0) / 102.4) / 10} GB free`;
  const vramT = res.memory?.vram_total_mb || 0;
  document.getElementById('stat-vram-val').textContent = vramT > 0 ? `${Math.round((res.memory?.vram_avail_mb || 0) / 102.4) / 10} / ${Math.round(vramT / 102.4) / 10} GB free` : 'Shared System RAM';

  const freeGb = Number(res.free_disk_gb || 0);
  document.getElementById('stat-disk-val').textContent = `${freeGb.toFixed(1)} GB free`;

  const estMin = res.tiers?.image?.balanced?.est_minutes || 60;
  document.getElementById('warn-hw-estimate').textContent = `${fmtMinutes(estMin)} (Balanced tier on ${res.hardware?.name || 'your hardware'})`;
  document.getElementById('warn-disk-space').textContent = `Needs ~1.2 GB free temporarily for Image training or ~0.9 GB for Text (${freeGb.toFixed(1)} GB free detected)`;

  const procEl = document.getElementById('preflight-proc-list');
  const procs = Array.isArray(res.heavy_processes) ? res.heavy_processes : [];
  procEl.innerHTML = procs.length
    ? procs.map((pr) => `<div class="bellows-proc-row"><span><strong>${pr.name}</strong> (${pr.pids} proc)</span><span>${Math.round(pr.ram_mb)} MB RAM${pr.vram_mb > 0 ? ` • ${Math.round(pr.vram_mb)} MB VRAM` : ''}</span></div>`).join('')
    : '<div class="bellows-muted">No unusually heavy user applications detected.</div>';

  const policyEl = document.getElementById('policy-rules-list');
  if (policyEl && Array.isArray(res.prohibited_policies)) {
    policyEl.innerHTML = res.prohibited_policies.map((r) => `<li><strong>${r.title}:</strong> ${r.detail}</li>`).join('');
  }

  const blocker = document.getElementById('preflight-blocker-msg');
  const proceedBtn = document.getElementById('preflight-proceed-btn');
  if (!plugged) {
    blocker.hidden = false;
    blocker.textContent = 'Plug in your computer to wall power (AC) before proceeding. Bellows cannot start training on battery.';
    proceedBtn.disabled = true;
  } else if (freeGb < 1.2) {
    blocker.hidden = false;
    blocker.textContent = `Only ${freeGb.toFixed(2)} GB free disk space detected. Free at least 1.2 GB temporarily before continuing.`;
    proceedBtn.disabled = true;
  } else {
    blocker.hidden = true;
    proceedBtn.disabled = false;
  }
}

async function handleEthicsCheck() {
  const prompt = (document.getElementById('bellows-prompt-input')?.value || '').trim();
  const box = document.getElementById('ethics-status-box');
  const btn = document.getElementById('ethics-check-btn');
  bellowsState.prompt = prompt;

  btn.disabled = true;
  box.hidden = false;
  box.className = 'bellows-alert';
  box.textContent = 'Inspecting filter target with local ethical guard…';

  const res = await sendBridge('BELLOWS_CHECK_ETHICS', { modality: bellowsState.modality, prompt }, 240000);
  btn.disabled = false;

  if (!res || !res.ok || !res.allowed) {
    box.className = 'bellows-alert is-danger';
    box.textContent = res?.reason || 'This filter target is not permitted under the FvF Bellows ethical policy.';
    return;
  }

  box.className = 'bellows-alert is-ok';
  box.textContent = `✓ Approved (${res.source || 'guard'}): ${res.reason || 'Safe to build!'}`;

  // Prepare dataset session & fetch Hugging Face base pack
  const nameInput = document.getElementById('cfg-model-name');
  const labelInput = document.getElementById('cfg-model-label');
  const cleanTitle = prompt.charAt(0).toUpperCase() + prompt.slice(1);
  if (nameInput && !nameInput.value.trim()) nameInput.value = `My ${cleanTitle} Filter`.slice(0, 60);
  if (labelInput && !labelInput.value.trim()) labelInput.value = cleanTitle.slice(0, 40);

  updateModalityDatasetUI();
  goBellowsStep(3);
  await prepareBaseDatasetSession();
}

function updateModalityDatasetUI() {
  const isText = bellowsState.modality === 'text';
  document.getElementById('byo-image-panel').hidden = isText;
  document.getElementById('byo-text-panel').hidden = !isText;
  document.getElementById('text-collector-source-row').hidden = !isText;
}

async function prepareBaseDatasetSession() {
  const fill = document.getElementById('base-dataset-progress-fill');
  const txt = document.getElementById('base-dataset-status-text');
  const badge = document.getElementById('base-dataset-badge');
  badge.textContent = 'Downloading…';
  fill.style.width = '15%';

  const res = await sendBridge('BELLOWS_PREPARE_SESSION', {
    modality: bellowsState.modality,
    prompt: bellowsState.prompt,
    jobId: bellowsState.jobId
  }, 300000);

  if (res && res.ok) {
    bellowsState.jobId = res.jobId;
    bellowsState.counts = res.counts || bellowsState.counts;
    fill.style.width = '100%';
    badge.textContent = 'Ready ✓';
    txt.textContent = `Loaded ${bellowsState.counts.base_negatives} baseline web negatives from Hugging Face (LuckyCow/fvf-bellows-base).`;
    updateDatasetCountsUI();
  } else {
    badge.textContent = 'Error';
    txt.textContent = `Failed downloading base dataset: ${res?.error || 'network error'}`;
  }
}

function updateDatasetCountsUI() {
  const pos = bellowsState.counts.positives || 0;
  const hn = bellowsState.counts.hard_negatives || 0;
  const base = bellowsState.counts.base_negatives || 0;
  const minNeed = bellowsState.modality === 'image' ? 40 : 30;
  const solidNeed = bellowsState.modality === 'image' ? 120 : 80;
  const maxGoal = bellowsState.modality === 'image' ? 300 : 150;

  document.getElementById('ds-pos-count').textContent = `${pos} Targets`;
  document.getElementById('ds-hn-count').textContent = `${hn} Hard Negatives`;
  document.getElementById('ds-base-count').textContent = `${base} Base Web Negatives`;

  const pct = Math.min(100, Math.round((pos / maxGoal) * 100));
  document.getElementById('ds-quality-fill').style.width = `${pct}%`;

  const lbl = document.getElementById('ds-quality-label');
  if (pos >= maxGoal) lbl.textContent = 'High-precision dataset ready!';
  else if (pos >= solidNeed) lbl.textContent = 'Solid dataset ready (or keep adding more)';
  else if (pos >= minNeed) lbl.textContent = 'Minimum reached — more examples improve accuracy';
  else lbl.textContent = `Add ${Math.max(0, minNeed - pos)} more targets (or ≥15 absolute minimum) to continue`;

  document.getElementById('step3-next-btn').disabled = pos < 15;
}

async function uploadImageFiles(fileList, kind) {
  if (!bellowsState.jobId) return;
  const files = Array.from(fileList || []).slice(0, 600);
  const chunkSize = 12;
  for (let i = 0; i < files.length; i += chunkSize) {
    const batch = files.slice(i, i + chunkSize);
    const dataUrls = await Promise.all(batch.map((f) => new Promise((res) => {
      const r = new FileReader();
      r.onload = () => res(r.result);
      r.onerror = () => res(null);
      r.readAsDataURL(f);
    })));
    const resp = await sendBridge('BELLOWS_ADD_SAMPLES', {
      jobId: bellowsState.jobId,
      modality: 'image',
      kind,
      samples: dataUrls.filter(Boolean)
    }, 30000);
    if (resp && resp.counts) {
      bellowsState.counts = resp.counts;
      updateDatasetCountsUI();
    }
  }
}

async function addTextSamples(rawStrings, kind) {
  if (!bellowsState.jobId || !rawStrings.length) return;
  const resp = await sendBridge('BELLOWS_ADD_SAMPLES', {
    jobId: bellowsState.jobId,
    modality: 'text',
    kind,
    samples: rawStrings.map((t) => ({ text: t }))
  }, 30000);
  if (resp && resp.counts) {
    bellowsState.counts = resp.counts;
    updateDatasetCountsUI();
  }
}

function renderTiers() {
  const tiers = bellowsState.preflight?.tiers?.[bellowsState.modality] || {};
  const list = document.getElementById('tier-picker-list');
  if (!list) return;
  list.innerHTML = '';
  ['swift', 'balanced', 'thorough'].forEach((key) => {
    const info = tiers[key];
    if (!info) return;
    const card = document.createElement('button');
    card.type = 'button';
    card.className = `bellows-tier-card ${bellowsState.tier === key ? 'active' : ''}`;
    card.innerHTML = `
      <div>
        <strong>${info.name}</strong>
        <div class="bellows-muted">${info.epochs} epochs • ~${info.disk_mb} MB temp cache</div>
      </div>
      <span class="bellows-badge">${info.est_text}</span>
    `;
    card.addEventListener('click', () => {
      bellowsState.tier = key;
      renderTiers();
    });
    list.appendChild(card);
  });
}

async function igniteTraining() {
  const errBox = document.getElementById('start-error-box');
  errBox.hidden = true;

  const modelName = (document.getElementById('cfg-model-name').value || '').trim() || 'My Bellows Filter';
  const label = (document.getElementById('cfg-model-label').value || '').trim() || 'Custom filter';
  const suggestedThreshold = parseFloat(document.getElementById('cfg-model-thresh').value || '0.75');
  const autoThreshold = Boolean(document.getElementById('cfg-auto-thresh').checked);

  const res = await sendBridge('BELLOWS_START_TRAINING', {
    jobId: bellowsState.jobId,
    modality: bellowsState.modality,
    prompt: bellowsState.prompt,
    tier: bellowsState.tier,
    modelName,
    label,
    suggestedThreshold,
    autoThreshold
  }, 20000);

  if (!res || !res.ok) {
    errBox.hidden = false;
    errBox.textContent = res?.error || 'Could not launch training worker.';
    return;
  }

  bellowsState.finalizedCategory = false;
  goBellowsStep(5);
  startStatusPolling();
}

function drawTrainingChart(history) {
  const canvas = document.getElementById('train-chart-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    const y = (h / 5) * i;
    ctx.beginPath();
    ctx.moveTo(32, y);
    ctx.lineTo(w - 16, y);
    ctx.stroke();
  }

  if (!Array.isArray(history) || history.length === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.font = '13px Inter, sans-serif';
    ctx.fillText('Metrics curve will plot as Epoch 1 finishes…', 44, h / 2);
    return;
  }

  const maxLoss = Math.max(1.0, ...history.map((d) => Math.max(d.train_loss || 0, d.val_loss || 0)));
  const xStep = (w - 60) / Math.max(1, history.length - 1);

  const plotLine = (vals, maxV, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = 36 + (history.length === 1 ? (w - 60) / 2 : i * xStep);
      const y = h - 22 - Math.min(1, Math.max(0, v / maxV)) * (h - 44);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  plotLine(history.map((d) => d.train_loss || 0), maxLoss, '#ff6218');
  plotLine(history.map((d) => d.val_loss || 0), maxLoss, '#60a5fa');
  plotLine(history.map((d) => d.val_acc || 0), 1.0, '#34d399');
}

function startStatusPolling() {
  if (bellowsState.pollTimer) clearInterval(bellowsState.pollTimer);
  const poll = async () => {
    const res = await sendBridge('BELLOWS_GET_STATUS', { jobId: bellowsState.jobId }, 8000);
    if (!res || !res.ok || !res.status) return;
    const st = res.status;

    document.getElementById('train-phase-detail').textContent = st.phase_detail || st.phase || 'Training…';
    const pwr = res.power || st.power_status || {};
    const pwrText = pwr.is_plugged_in ? 'Plugged In' : `On Battery (${pwr.battery_percent ?? '?'}%)`;
    document.getElementById('train-hw-meta').textContent = `Device: ${st.device_name || 'Detecting…'} (${st.backend || 'CPU'}) • Power: ${pwrText}`;

    const fill = document.getElementById('train-progress-fill');
    if (fill) fill.style.width = `${Math.min(100, Math.round(st.progress_pct || 0))}%`;

    const etaEl = document.getElementById('train-eta-badge');
    if (etaEl) {
      if (st.state === 'completed' || st.state === 'stopped_low_battery' || st.state === 'stopped_early') {
        etaEl.textContent = 'Finished ✓';
      } else if (typeof st.eta_seconds === 'number' && st.eta_seconds > 0) {
        const mins = Math.max(1, Math.round(st.eta_seconds / 60));
        etaEl.textContent = `ETA: ~${mins} min`;
      } else {
        etaEl.textContent = 'ETA: Measuring…';
      }
    }

    drawTrainingChart(st.history || []);

    const cm = st.confusion_matrix;
    if (cm) {
      document.getElementById('cm-tp').textContent = String(cm.tp ?? '—');
      document.getElementById('cm-fp').textContent = String(cm.fp ?? '—');
      document.getElementById('cm-fn').textContent = String(cm.fn ?? '—');
      document.getElementById('cm-tn').textContent = String(cm.tn ?? '—');
      document.getElementById('cm-summary-line').textContent = `Val Acc: ${(cm.acc * 100).toFixed(1)}% • Target Recall: ${(cm.tpr * 100).toFixed(1)}% • False Positive Rate: ${(cm.fpr * 100).toFixed(1)}% (at ${Math.round((cm.threshold || 0.75) * 100)}%)`;
    }

    if (st.state === 'completed' || st.state === 'stopped_low_battery' || st.state === 'stopped_early' || st.state === 'failed') {
      clearInterval(bellowsState.pollTimer);
      bellowsState.pollTimer = null;
      document.getElementById('train-stop-btn').hidden = true;

      if (st.state === 'failed') {
        document.getElementById('train-hero-title').textContent = 'Training Failed';
        document.getElementById('train-hero-sub').textContent = st.error || st.phase_detail;
        return;
      }

      document.getElementById('train-hero-title').textContent =
        st.state === 'stopped_low_battery' ? 'Saved Best Checkpoint (Low Battery)' : 'Classifier Ready!';
      document.getElementById('train-complete-panel').hidden = false;
      document.getElementById('complete-summary-text').textContent = `${st.model_name} (${st.classifier_id}) was exported locally and activated as a new Fire vs Fire category at ${Math.round((st.suggested_threshold || 0.75) * 100)}% threshold.`;
      document.getElementById('cleanup-size-text').textContent = `Bellows is currently using ~${res.cache_size_mb || 0} MB for its cached gatekeeper & Hugging Face base dataset (+ ~${res.jobs_size_mb || 0} MB raw training samples). Keep them or delete them now?`;

      if (!bellowsState.finalizedCategory && st.classifier_id) {
        bellowsState.finalizedCategory = true;
        await sendBridge('BELLOWS_FINALIZE_CATEGORY', {
          classifierId: st.classifier_id,
          modelName: st.model_name,
          modality: st.modality,
          threshold: st.suggested_threshold
        }, 10000);
      }
    }
  };
  poll();
  bellowsState.pollTimer = setInterval(poll, 2000);
}

function initAmbientEmbers() {
  const canvas = document.getElementById('ember-webgl-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  let w = (canvas.width = window.innerWidth);
  let h = (canvas.height = window.innerHeight);
  window.addEventListener('resize', () => {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }, { passive: true });
  const embers = Array.from({ length: 46 }, () => ({
    x: Math.random() * w,
    y: Math.random() * h,
    vx: 0.2 + Math.random() * 0.6,
    vy: -0.35 - Math.random() * 0.65,
    r: 1.2 + Math.random() * 2.5,
    a: 0.2 + Math.random() * 0.55
  }));
  const step = () => {
    ctx.clearRect(0, 0, w, h);
    for (const p of embers) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x > w + 16) p.x = -10;
      if (p.y < -16) p.y = h + 10;
      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4.5);
      grad.addColorStop(0, `rgba(110, 231, 183, ${p.a})`);
      grad.addColorStop(0.5, `rgba(16, 185, 129, ${p.a * 0.45})`);
      grad.addColorStop(1, 'rgba(16, 185, 129, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 4.5, 0, Math.PI * 2);
      ctx.fill();
    }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function initBellows() {
  initAmbientEmbers();
  const params = new URLSearchParams(window.location.search);
  const initMod = params.get('modality');
  if (initMod === 'text' || initMod === 'image') {
    bellowsState.modality = initMod;
    document.querySelectorAll('.bellows-mod-card').forEach((c) => c.classList.toggle('active', c.dataset.mod === initMod));
  }

  document.getElementById('bellows-back-hub-btn')?.addEventListener('click', () => {
    window.location.href = './index.html?screen=2';
  });
  document.getElementById('goto-hub-shelf-btn')?.addEventListener('click', () => {
    window.location.href = './index.html?screen=2';
  });

  document.getElementById('preflight-refresh-btn')?.addEventListener('click', runPreflight);
  document.getElementById('preflight-proceed-btn')?.addEventListener('click', () => goBellowsStep(2));
  document.getElementById('step2-back-btn')?.addEventListener('click', () => goBellowsStep(1));
  document.getElementById('step3-back-btn')?.addEventListener('click', () => goBellowsStep(2));
  document.getElementById('step3-next-btn')?.addEventListener('click', () => goBellowsStep(4));
  document.getElementById('step4-back-btn')?.addEventListener('click', () => goBellowsStep(3));

  document.querySelectorAll('.bellows-mod-card').forEach((card) => {
    card.addEventListener('click', () => {
      bellowsState.modality = card.dataset.mod;
      document.querySelectorAll('.bellows-mod-card').forEach((c) => c.classList.toggle('active', c === card));
    });
  });

  document.getElementById('ethics-check-btn')?.addEventListener('click', handleEthicsCheck);

  document.getElementById('launch-collector-btn')?.addEventListener('click', () => {
    const engine = document.querySelector('input[name="txt-engine"]:checked')?.value || undefined;
    sendBridge('BELLOWS_START_COLLECTOR', {
      jobId: bellowsState.jobId,
      modality: bellowsState.modality,
      prompt: bellowsState.prompt,
      counts: bellowsState.counts,
      engine
    }, 10000);
  });

  document.getElementById('byo-pos-files')?.addEventListener('change', (e) => uploadImageFiles(e.target.files, 'positive'));
  document.getElementById('byo-hn-files')?.addEventListener('change', (e) => uploadImageFiles(e.target.files, 'hard_negative'));
  document.getElementById('byo-text-files')?.addEventListener('change', async (e) => {
    const chunks = [];
    for (const f of Array.from(e.target.files || [])) {
      const txt = await f.text();
      txt.split(/\r?\n\r?\n|\r?\n/).forEach((line) => {
        if (line.trim().length >= 40) chunks.push(line.trim());
      });
    }
    await addTextSamples(chunks, 'positive');
  });
  document.getElementById('byo-paste-pos-btn')?.addEventListener('click', async () => {
    const ta = document.getElementById('byo-text-paste');
    const items = (ta.value || '').split(/\n\s*\n/).map((s) => s.trim()).filter((s) => s.length >= 40);
    await addTextSamples(items, 'positive');
    ta.value = '';
  });
  document.getElementById('byo-paste-hn-btn')?.addEventListener('click', async () => {
    const ta = document.getElementById('byo-text-paste');
    const items = (ta.value || '').split(/\n\s*\n/).map((s) => s.trim()).filter((s) => s.length >= 40);
    await addTextSamples(items, 'hard_negative');
    ta.value = '';
  });

  const threshSlider = document.getElementById('cfg-model-thresh');
  threshSlider?.addEventListener('input', () => {
    document.getElementById('cfg-thresh-val').textContent = `${Math.round(parseFloat(threshSlider.value) * 100)}%`;
  });

  document.getElementById('ignite-training-btn')?.addEventListener('click', igniteTraining);
  document.getElementById('train-stop-btn')?.addEventListener('click', () => {
    sendBridge('BELLOWS_STOP_TRAINING', { jobId: bellowsState.jobId }, 8000);
  });

  document.getElementById('keep-cache-btn')?.addEventListener('click', () => {
    document.getElementById('cleanup-prompt-box').innerHTML = '<strong style="color:#6ee7b7;">✓ Kept Bellows cache &amp; base datasets for faster future runs.</strong>';
  });
  document.getElementById('delete-cache-btn')?.addEventListener('click', async () => {
    const res = await sendBridge('BELLOWS_CLEANUP_STORAGE', { deleteCache: true, deleteDatasets: true }, 15000);
    document.getElementById('cleanup-prompt-box').innerHTML = `<strong style="color:#a7f3d0;">✓ Deleted temporary caches and raw datasets (freed ${res?.freed_mb || 0} MB). Your trained classifier remains installed!</strong>`;
  });

  runPreflight().then(() => {
    if (params.get('resume') === '1' || bellowsState.preflight?.active_job?.state === 'running') {
      bellowsState.jobId = params.get('jobId') || bellowsState.preflight?.active_job?.job_id;
      if (bellowsState.jobId) {
        goBellowsStep(5);
        startStatusPolling();
      }
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initBellows);
} else {
  initBellows();
}
