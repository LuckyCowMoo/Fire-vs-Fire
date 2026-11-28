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

const BELLOWS_PROMPT_PLACEHOLDERS = {
  image: 'e.g., spiders, clowns, deep ocean underwater, needles, gambling ads...',
  text: 'e.g., hate speech, graphic gore, spoilers for a specific show, gambling promotions...'
};

const ARCH_NEURON_INFO = {
  image: {
    swift: { arch: 'MobileNetV3-Small', neurons: '1,518,881' },
    balanced: { arch: 'EfficientNet-B0', neurons: '4,008,829' },
    thorough: { arch: 'ConvNeXt-Tiny', neurons: '27,820,897' }
  },
  text: {
    swift: { arch: 'ELECTRA-Small', neurons: '13,549,057' },
    balanced: { arch: 'MiniLM-L6', neurons: '22,713,601' },
    thorough: { arch: 'DistilRoBERTa-Base', neurons: '82,119,169' }
  }
};

function updatePromptPlaceholder() {
  const input = document.getElementById('bellows-prompt-input');
  if (!input) return;
  input.placeholder = BELLOWS_PROMPT_PLACEHOLDERS[bellowsState.modality] || BELLOWS_PROMPT_PLACEHOLDERS.image;
}

function updateEduArchitectureSummary(modality, tier) {
  const mod = modality === 'text' ? 'text' : 'image';
  const t = tier && ARCH_NEURON_INFO[mod][tier] ? tier : 'balanced';
  const info = ARCH_NEURON_INFO[mod][t];
  const archEl = document.getElementById('edu-arch-name');
  const neuronEl = document.getElementById('edu-neuron-count');
  if (archEl) archEl.textContent = info.arch;
  if (neuronEl) neuronEl.textContent = info.neurons;
}

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

let ethicsCurrentPct = 0;
let ethicsTrickleTimer = null;

function setEthicsProgress(percent, message) {
  const wrap = document.getElementById('ethics-progress-wrap');
  const fill = document.getElementById('ethics-progress-fill');
  const pctEl = document.getElementById('ethics-progress-pct');
  const labelEl = document.getElementById('ethics-progress-label');
  const clamped = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
  ethicsCurrentPct = Math.max(ethicsCurrentPct, clamped);
  if (wrap) wrap.hidden = false;
  if (fill) fill.style.width = `${ethicsCurrentPct}%`;
  if (pctEl) pctEl.textContent = `${ethicsCurrentPct}%`;
  if (labelEl && message) labelEl.textContent = message;
}

function handleBridgePush(msg) {
  if (!msg) return;
  if (msg.type === 'BELLOWS_PROGRESS' && msg.payload) {
    const p = msg.payload;
    if (p.stage === 'ethics') {
      const pct = typeof p.percent === 'number' ? p.percent : ethicsCurrentPct;
      setEthicsProgress(pct);
      const box = document.getElementById('ethics-status-box');
      if (box && !box.classList.contains('is-danger') && !box.classList.contains('is-ok')) {
        box.hidden = true;
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
  const wrap = document.getElementById('ethics-progress-wrap');
  const fill = document.getElementById('ethics-progress-fill');
  bellowsState.prompt = prompt;

  btn.disabled = true;
  box.hidden = true;
  box.className = 'bellows-alert';

  if (ethicsTrickleTimer) {
    clearInterval(ethicsTrickleTimer);
    ethicsTrickleTimer = null;
  }
  ethicsCurrentPct = 0;
  if (fill) fill.classList.add('is-pulsing');
  setEthicsProgress(8);
  ethicsTrickleTimer = setInterval(() => {
    const cap = ethicsCurrentPct < 55 ? 50 : 90;
    if (ethicsCurrentPct < cap) {
      setEthicsProgress(ethicsCurrentPct + 1);
    }
  }, 650);

  let res = null;
  try {
    res = await sendBridge('BELLOWS_CHECK_ETHICS', { modality: bellowsState.modality, prompt }, 240000);
  } finally {
    if (ethicsTrickleTimer) {
      clearInterval(ethicsTrickleTimer);
      ethicsTrickleTimer = null;
    }
    if (fill) fill.classList.remove('is-pulsing');
    btn.disabled = false;
  }

  if (!res || !res.ok || !res.allowed) {
    if (wrap) wrap.hidden = true;
    box.hidden = false;
    box.className = 'bellows-alert is-danger';
    box.textContent = res?.reason || 'This filter target is not permitted under the FvF Bellows ethical policy.';
    return;
  }

  setEthicsProgress(100);
  box.hidden = false;
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

  const promptPreview = document.getElementById('collector-prompt-preview');
  if (promptPreview) promptPreview.textContent = bellowsState.prompt || 'your topic';
  const mediaNoun = document.getElementById('collector-media-noun');
  if (mediaNoun) mediaNoun.textContent = isText ? 'examples' : 'images';

  const byoHelp = document.getElementById('byo-help-text');
  if (byoHelp) {
    byoHelp.textContent = isText
      ? 'Accepted text formats: .txt, .csv, .jsonl, .json files, or pasted paragraphs (min 40 chars each).'
      : 'Accepted image formats: JPG, PNG, WebP, GIF, BMP.';
  }
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
  const maxGoal = bellowsState.modality === 'image' ? 300 : 150;

  const posEl = document.getElementById('ds-pos-count');
  if (posEl) posEl.textContent = `${pos} Targets`;
  const hnEl = document.getElementById('ds-hn-count');
  if (hnEl) hnEl.textContent = `${hn} Hard Negatives`;

  const pct = Math.min(100, Math.round((pos / maxGoal) * 100));
  const fillEl = document.getElementById('ds-quality-fill');
  if (fillEl) fillEl.style.width = `${pct}%`;

  const nextBtn = document.getElementById('step3-next-btn');
  if (nextBtn) nextBtn.disabled = pos < 15;
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
      <div class="bellows-tier-speed-row">Est. Training Time: ${info.est_text}</div>
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
  const stopBtn = document.getElementById('train-stop-btn');
  if (stopBtn) {
    stopBtn.hidden = true;
    stopBtn.disabled = false;
    stopBtn.innerHTML = '<span>✓</span> I\'m Happy with This — Stop &amp; Use Best Model';
  }
  const epochStopBtn = document.getElementById('epoch-happy-stop-btn');
  if (epochStopBtn) {
    epochStopBtn.disabled = false;
    epochStopBtn.innerHTML = '✓ Happy with This Epoch — Finish Now';
  }
  const stoppingBanner = document.getElementById('stopping-progress-banner');
  if (stoppingBanner) stoppingBanner.hidden = true;
  const epochHappyBanner = document.getElementById('epoch-happy-banner');
  if (epochHappyBanner) epochHappyBanner.hidden = true;

  updateEduArchitectureSummary(bellowsState.modality, bellowsState.tier);
  goBellowsStep(5);
  startStatusPolling();
}

function drawTrainingChart(history, lossHistory, totalSteps, totalEpochs) {
  const canvas = document.getElementById('train-chart-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 46;
  const padR = 26;
  const padT = 28;
  const padB = 30;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Background horizontal grid lines (0%, 25%, 50%, 75%, 100%)
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;

  const numGridLines = 4;
  for (let i = 0; i <= numGridLines; i++) {
    const y = padT + (plotH / numGridLines) * i;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
  }

  const hist = Array.isArray(history) ? history : [];
  const lossHist = Array.isArray(lossHistory) ? lossHistory : [];
  const epochs = Math.max(1, totalEpochs || (hist.length > 0 ? hist[hist.length - 1].epoch : 1));
  const steps = Math.max(1, totalSteps || (lossHist.length > 0 ? lossHist[lossHist.length - 1].step : 100));

  // Vertical epoch dividers
  if (epochs > 1) {
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    for (let e = 1; e <= epochs; e++) {
      const ex = padL + (e / epochs) * plotW;
      ctx.beginPath();
      ctx.moveTo(ex, padT);
      ctx.lineTo(ex, h - padB);
      ctx.stroke();

      // Epoch label centered in slice
      const midX = padL + ((e - 0.5) / epochs) * plotW;
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`Ep ${e}`, midX, h - padB + 14);
    }
    ctx.setLineDash([]);
  } else {
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Epoch 1 Progress', padL + plotW / 2, h - padB + 14);
  }

  // Calculate maximum loss for auto-scaling
  let maxLoss = 1.0;
  for (const p of lossHist) {
    if (typeof p.loss === 'number' && p.loss > maxLoss) maxLoss = p.loss;
  }
  for (const hItem of hist) {
    if (typeof hItem.train_loss === 'number' && hItem.train_loss > maxLoss) maxLoss = hItem.train_loss;
    if (typeof hItem.val_loss === 'number' && hItem.val_loss > maxLoss) maxLoss = hItem.val_loss;
  }
  maxLoss = Math.ceil(maxLoss * 10) / 10;

  // Y-axis labels
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.font = '10px Inter, sans-serif';
  ctx.fillText(maxLoss.toFixed(1), padL - 8, padT);
  ctx.fillText((maxLoss * 0.5).toFixed(1), padL - 8, padT + plotH * 0.5);
  ctx.fillText('0.0', padL - 8, h - padB);

  // Right Y-axis labels for Accuracy
  ctx.textAlign = 'left';
  ctx.fillStyle = 'rgba(52, 211, 153, 0.55)';
  ctx.fillText('100%', w - padR + 5, padT);
  ctx.fillText('50%', w - padR + 5, padT + plotH * 0.5);
  ctx.fillText('0%', w - padR + 5, h - padB);

  // If no data points at all yet
  if (lossHist.length === 0 && hist.length === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.font = '13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Live training loss curve starts streaming in ~10s…', padL + plotW / 2, padT + plotH / 2);
    return;
  }

  // Helper to map step to X
  const getX = (step) => {
    return padL + Math.min(1.0, Math.max(0.0, step / steps)) * plotW;
  };

  // Helper to map loss to Y
  const getLossY = (loss) => {
    return (h - padB) - Math.min(1.0, Math.max(0.0, loss / maxLoss)) * plotH;
  };

  // Helper to map acc to Y (0 to 1)
  const getAccY = (acc) => {
    return (h - padB) - Math.min(1.0, Math.max(0.0, acc)) * plotH;
  };

  // 1. Draw continuous Orange Training Loss Curve (streaming every 10s)
  if (lossHist.length > 0) {
    const firstX = getX(lossHist[0].step);
    const firstY = getLossY(lossHist[0].loss);

    // Area fill under loss curve
    ctx.beginPath();
    ctx.moveTo(firstX, h - padB);
    ctx.lineTo(firstX, firstY);

    for (let i = 1; i < lossHist.length; i++) {
      const px = getX(lossHist[i].step);
      const py = getLossY(lossHist[i].loss);
      ctx.lineTo(px, py);
    }
    const lastX = getX(lossHist[lossHist.length - 1].step);
    ctx.lineTo(lastX, h - padB);
    ctx.closePath();

    const grad = ctx.createLinearGradient(0, padT, 0, h - padB);
    grad.addColorStop(0, 'rgba(255, 98, 24, 0.22)');
    grad.addColorStop(1, 'rgba(255, 98, 24, 0.0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Stroke line
    ctx.strokeStyle = '#ff6218';
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(firstX, firstY);
    for (let i = 1; i < lossHist.length; i++) {
      ctx.lineTo(getX(lossHist[i].step), getLossY(lossHist[i].loss));
    }
    ctx.stroke();

    // Pulse dot at the latest live loss point
    const latest = lossHist[lossHist.length - 1];
    const lx = getX(latest.step);
    const ly = getLossY(latest.loss);

    ctx.beginPath();
    ctx.arc(lx, ly, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 98, 24, 0.35)';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = '#ff6218';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(lx, ly, 1.5, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();

    // Live loss badge in canvas header
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ff884d';
    ctx.fillText(`Live Train Loss: ${latest.loss.toFixed(4)} (Step ${latest.step}/${steps})`, padL + 4, padT - 10);
  }

  // 2. Draw Blue Validation Loss (at epoch boundaries)
  if (hist.length > 0) {
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    hist.forEach((d, idx) => {
      const ex = padL + (d.epoch / epochs) * plotW;
      const ey = getLossY(d.val_loss || 0);
      if (idx === 0) ctx.moveTo(ex, ey);
      else ctx.lineTo(ex, ey);
    });
    ctx.stroke();

    // Validation Loss dots
    hist.forEach((d) => {
      const ex = padL + (d.epoch / epochs) * plotW;
      const ey = getLossY(d.val_loss || 0);
      ctx.beginPath();
      ctx.arc(ex, ey, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#60a5fa';
      ctx.fill();
    });

    // 3. Draw Green Validation Accuracy Line & Dots
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    hist.forEach((d, idx) => {
      const ex = padL + (d.epoch / epochs) * plotW;
      const ey = getAccY(d.val_acc || 0);
      if (idx === 0) ctx.moveTo(ex, ey);
      else ctx.lineTo(ex, ey);
    });
    ctx.stroke();

    // Validation Accuracy dots and percentage labels
    hist.forEach((d) => {
      const ex = padL + (d.epoch / epochs) * plotW;
      const ey = getAccY(d.val_acc || 0);

      ctx.beginPath();
      ctx.arc(ex, ey, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#34d399';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(ex, ey, 2, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();

      // Label above dot
      const pct = `${Math.round((d.val_acc || 0) * 100)}%`;
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.fillStyle = '#6ee7b7';
      ctx.textAlign = 'center';
      ctx.fillText(pct, ex, Math.max(padT - 6, ey - 9));
    });
  }
}

async function handleStopEarly() {
  const stopBtn = document.getElementById('train-stop-btn');
  const epochStopBtn = document.getElementById('epoch-happy-stop-btn');
  if (stopBtn) {
    stopBtn.disabled = true;
    stopBtn.innerHTML = '<span>⏳</span> Stopping &amp; Saving Best Model…';
  }
  if (epochStopBtn) {
    epochStopBtn.disabled = true;
    epochStopBtn.innerHTML = '<span>⏳</span> Stopping &amp; Saving Best Model…';
  }
  const stoppingBanner = document.getElementById('stopping-progress-banner');
  if (stoppingBanner) stoppingBanner.hidden = false;
  const phaseDetail = document.getElementById('train-phase-detail');
  if (phaseDetail) phaseDetail.textContent = 'Stopping training early — saving best model checkpoint and exporting…';

  await sendBridge('BELLOWS_STOP_TRAINING', { jobId: bellowsState.jobId }, 8000);
}

function startStatusPolling() {
  if (bellowsState.pollTimer) clearInterval(bellowsState.pollTimer);
  const poll = async () => {
    const res = await sendBridge('BELLOWS_GET_STATUS', { jobId: bellowsState.jobId }, 8000);
    if (!res || !res.ok || !res.status) return;
    const st = res.status;
    updateEduArchitectureSummary(st.modality || bellowsState.modality, st.tier || bellowsState.tier);

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

    drawTrainingChart(st.history || [], st.loss_history || [], st.total_steps || 1, st.total_epochs || 1);

    const hist = Array.isArray(st.history) ? st.history : [];
    const lastEp = hist.length > 0 ? hist[hist.length - 1] : null;
    const completedEpochs = lastEp ? Number(lastEp.epoch || hist.length) : 0;
    const rawThresh = typeof st.suggested_threshold === 'number' && Number.isFinite(st.suggested_threshold)
      ? st.suggested_threshold
      : 0.75;
    const accThreshold = Math.max(0.75, rawThresh);
    const currentAcc = lastEp && typeof lastEp.val_acc === 'number' ? lastEp.val_acc : 0;
    const canOfferEarlyStop = st.state === 'running' && completedEpochs >= 2 && currentAcc >= accThreshold;

    const stopBtn = document.getElementById('train-stop-btn');
    if (stopBtn) {
      stopBtn.hidden = !canOfferEarlyStop;
    }

    const happyBanner = document.getElementById('epoch-happy-banner');
    if (happyBanner) {
      if (canOfferEarlyStop && lastEp) {
        happyBanner.hidden = false;
        const accPct = Math.round(currentAcc * 100);
        const titleEl = document.getElementById('epoch-happy-title');
        if (titleEl) {
          titleEl.textContent = `Epoch ${lastEp.epoch}/${st.total_epochs || hist.length} Complete — ${accPct}% Validation Accuracy!`;
        }
        const descEl = document.getElementById('epoch-happy-desc');
        if (descEl) {
          descEl.textContent = `Target Recall: ${(lastEp.val_tpr * 100).toFixed(1)}% • False Alarms: ${(lastEp.val_fpr * 100).toFixed(1)}%. If you are happy that the AI has found the pattern, you can stop now and use this model immediately.`;
        }
      } else {
        happyBanner.hidden = true;
      }
    }

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
      if (happyBanner) happyBanner.hidden = true;
      const stoppingBanner = document.getElementById('stopping-progress-banner');
      if (stoppingBanner) stoppingBanner.hidden = true;

      if (st.state === 'failed') {
        document.getElementById('train-hero-title').textContent = 'Training Failed';
        document.getElementById('train-hero-sub').textContent = st.error || st.phase_detail;
        const failedActions = document.getElementById('train-failed-actions');
        if (failedActions) failedActions.hidden = false;
        bellowsState.lastStatus = st;
        return;
      }

      document.getElementById('train-hero-title').textContent =
        st.state === 'stopped_low_battery'
          ? 'Saved Best Checkpoint (Low Battery)'
          : st.state === 'stopped_early'
          ? 'Classifier Ready! (Stopped Early at Best Checkpoint)'
          : 'Classifier Ready!';
      document.getElementById('train-complete-panel').hidden = false;
      const bestAcc = st.best_metrics?.acc ? ` (Validation Accuracy: ${(st.best_metrics.acc * 100).toFixed(1)}%)` : '';
      document.getElementById('complete-summary-text').textContent = `${st.model_name} (${st.classifier_id}) was exported locally and activated as a new Fire vs Fire category at ${Math.round((st.suggested_threshold || 0.75) * 100)}% threshold${bestAcc}.`;
      document.getElementById('cleanup-size-text').textContent = `Bellows is currently using ~${res.cache_size_mb || 0} MB for the gatekeeper & Hugging Face base dataset (+ ~${res.jobs_size_mb || 0} MB raw training samples). Now that your model is trained, you can delete these, or choose to keep them here so you don't have to download again next time.`;
      window.scrollTo({ top: 0, behavior: 'smooth' });

      if (!bellowsState.finalizedCategory && st.classifier_id) {
        bellowsState.finalizedCategory = true;
        await sendBridge('BELLOWS_FINALIZE_CATEGORY', {
          classifierId: st.classifier_id,
          modelName: st.model_name,
          label: st.label,
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
  updatePromptPlaceholder();

  document.getElementById('bellows-back-hub-btn')?.addEventListener('click', () => {
    window.location.href = './index.html?screen=2';
  });
  document.getElementById('close-bellows-tab-btn')?.addEventListener('click', () => {
    if (ext && ext.tabs && typeof ext.tabs.getCurrent === 'function' && typeof ext.tabs.remove === 'function') {
      ext.tabs.getCurrent((tab) => {
        if (tab && tab.id != null) {
          ext.tabs.remove(tab.id);
        } else {
          window.close();
        }
      });
    } else {
      window.close();
    }
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
      updatePromptPlaceholder();
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
  document.getElementById('train-retry-btn')?.addEventListener('click', async () => {
    const st = bellowsState.lastStatus || {};
    const jobId = bellowsState.jobId || st.job_id;
    if (!jobId) return;
    const retryBtn = document.getElementById('train-retry-btn');
    if (retryBtn) retryBtn.disabled = true;
    const res = await sendBridge('BELLOWS_START_TRAINING', {
      jobId,
      modality: st.modality || bellowsState.modality,
      prompt: st.prompt || bellowsState.prompt,
      tier: st.tier || bellowsState.tier,
      modelName: st.model_name,
      label: st.label,
      suggestedThreshold: st.suggested_threshold,
      autoThreshold: st.auto_threshold
    }, 20000);
    if (retryBtn) retryBtn.disabled = false;
    if (!res || !res.ok) {
      document.getElementById('train-hero-sub').textContent = res?.error || 'Could not restart training worker.';
      return;
    }
    const failedActions = document.getElementById('train-failed-actions');
    if (failedActions) failedActions.hidden = true;
    const stopBtn = document.getElementById('train-stop-btn');
    if (stopBtn) {
      stopBtn.hidden = true;
      stopBtn.disabled = false;
      stopBtn.innerHTML = '<span>✓</span> I\'m Happy with This — Stop &amp; Use Best Model';
    }
    const epochStopBtn = document.getElementById('epoch-happy-stop-btn');
    if (epochStopBtn) {
      epochStopBtn.disabled = false;
      epochStopBtn.innerHTML = '✓ Happy with This Epoch — Finish Now';
    }
    const stoppingBanner = document.getElementById('stopping-progress-banner');
    if (stoppingBanner) stoppingBanner.hidden = true;
    const epochHappyBanner = document.getElementById('epoch-happy-banner');
    if (epochHappyBanner) epochHappyBanner.hidden = true;

    document.getElementById('train-hero-title').textContent = 'Forging Your Classifier…';
    document.getElementById('train-hero-sub').textContent =
      'Training runs in the background Native Host service—you can safely close this tab and reopen it from Fire vs Fire Options anytime.';
    bellowsState.finalizedCategory = false;
    updateEduArchitectureSummary(st.modality || bellowsState.modality, st.tier || bellowsState.tier);
    startStatusPolling();
  });
  document.getElementById('train-stop-btn')?.addEventListener('click', handleStopEarly);
  document.getElementById('epoch-happy-stop-btn')?.addEventListener('click', handleStopEarly);

  document.getElementById('keep-cache-btn')?.addEventListener('click', () => {
    document.getElementById('cleanup-prompt-box').innerHTML = '<strong style="color:#6ee7b7;">✓ Kept Bellows cache &amp; base datasets for faster future runs.</strong>';
  });
  document.getElementById('delete-cache-btn')?.addEventListener('click', async () => {
    const res = await sendBridge('BELLOWS_CLEANUP_STORAGE', { deleteCache: true, deleteDatasets: true }, 15000);
    document.getElementById('cleanup-prompt-box').innerHTML = `<strong style="color:#a7f3d0;">✓ Deleted temporary caches and raw datasets (freed ${res?.freed_mb || 0} MB). Your trained classifier remains installed!</strong>`;
  });

  runPreflight().then(() => {
    const activeJob = bellowsState.preflight?.active_job;
    const activeStates = ['initializing', 'downloading_backbone', 'unpacking_dataset', 'running', 'exporting'];
    if (params.get('resume') === '1' || (activeJob && activeStates.includes(activeJob.state))) {
      bellowsState.jobId = params.get('jobId') || activeJob?.job_id;
      if (bellowsState.jobId) {
        goBellowsStep(5);
        startStatusPolling();
      }
    } else if (activeJob && activeJob.state === 'failed' && activeJob.job_id) {
      const blocker = document.getElementById('preflight-blocker-msg');
      if (blocker && blocker.hidden) {
        blocker.hidden = false;
        blocker.innerHTML = '';
        blocker.appendChild(
          document.createTextNode(
            `Previous training job “${activeJob.model_name || activeJob.job_id}” failed with its dataset still saved. `
          )
        );
        const resumeFailedBtn = document.createElement('button');
        resumeFailedBtn.type = 'button';
        resumeFailedBtn.className = 'btn-ember-outline';
        resumeFailedBtn.style.marginLeft = '10px';
        resumeFailedBtn.textContent = 'Open & Retry Failed Job →';
        resumeFailedBtn.addEventListener('click', () => {
          bellowsState.jobId = activeJob.job_id;
          goBellowsStep(5);
          startStatusPolling();
        });
        blocker.appendChild(resumeFailedBtn);
      }
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initBellows);
} else {
  initBellows();
}
