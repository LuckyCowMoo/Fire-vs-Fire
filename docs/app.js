/**
 * Fire vs Fire — Interactive Onboarding & Model Hub
 * Connects seamlessly to the Browser Extension & Local Native Host (V2)
 */

const ext = typeof browser !== 'undefined' ? browser : (typeof chrome !== 'undefined' ? chrome : null);

const RAW_GITHUB_CATALOG_URL =
  'https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/main/native/official_models.json';

// Embedded Fallback Catalog: Guarantees full UI works immediately offline or on file:/// URLs
const FALLBACK_CATALOG = {
  catalog_version: 2,
  repository: 'LuckyCowMoo/Fire-vs-Fire',
  branch: 'main',
  author: 'LuckyCowMoo',
  updated_at: '2026-11-08T21:30:00Z',
  models: [
    {
      id: 'fvf_heimdall02',
      name: 'FvF Heimdall V2',
      version: '2.1.0',
      modality: 'image',
      status: 'available',
      recommended: true,
      is_ai_detector: true,
      category_name: 'FvF Heimdall V2',
      label: 'AI generated',
      default_threshold: 0.75,
      default_color: '#FF4D1C',
      download_size: '192 MB',
      ram_size: '1.2 GB VRAM / RAM',
      tagline: 'local AI Image classifier',
      description: 'Our flagship first-party AI image classifier.'
    },
    {
      id: 'fvf_heimdall03',
      name: 'FvF Heimdall 03',
      version: '3.0.0-preview',
      modality: 'image',
      status: 'available',
      recommended: true,
      is_ai_detector: true,
      category_name: 'FvF Heimdall 03',
      label: 'AI generated',
      default_threshold: 0.75,
      default_color: '#FF2E00',
      download_size: '207 MB',
      ram_size: '1.5 GB VRAM / RAM',
      tagline: '24-Ch FFT + ForensicNeXt Image Classifier',
      description: 'Next-gen 24-channel FFT & ForensicNeXt AI image classifier with ConvNeXt-V2 Base.'
    },
    {
      id: 'res_net50_spider',
      name: 'ResNet 50 - Spider',
      version: '1.0.0',
      modality: 'image',
      status: 'available',
      recommended: false,
      is_ai_detector: false,
      special_badge: 'Object classifier',
      category_name: 'ResNet50 Spider',
      label: 'Spider',
      default_threshold: 0.75,
      default_color: '#FFB020',
      download_size: '90 MB',
      ram_size: '450 MB VRAM / RAM',
      tagline: 'Detects spiders',
      description: 'Detects real and illustrated spiders on web pages.'
    },
    {
      id: 'fvf_mimir_02',
      name: 'Mímir V2',
      version: '2.0.0-preview',
      modality: 'text',
      status: 'coming_next',
      recommended: true,
      is_ai_detector: true,
      special_badge: 'Coming Next',
      category_name: 'Mímir V2',
      label: 'AI text',
      default_threshold: 0.75,
      default_color: '#FF6218',
      download_size: '280 MB',
      ram_size: '800 MB RAM',
      tagline: 'Local AI Text Classifier',
      description: 'Local neural language model trained to flag LLM-generated prose.'
    },
    {
      id: 'fvf_huginn_01',
      name: 'Huginn V1',
      version: '1.0.0-preview',
      modality: 'video',
      status: 'coming_soon',
      recommended: true,
      is_ai_detector: true,
      special_badge: 'Coming Soon',
      category_name: 'Huginn V1',
      label: 'AI video',
      default_threshold: 0.75,
      default_color: '#FF4D1C',
      download_size: '520 MB',
      ram_size: '1.5 GB VRAM / RAM',
      tagline: 'Local AI Video Classifier',
      description: 'Analyzes video streams for temporal flicker and deepfakes.'
    },
    {
      id: 'fvf_bragi_01',
      name: 'Bragi V1',
      version: '1.0.0-preview',
      modality: 'audio',
      status: 'coming_soon',
      recommended: true,
      is_ai_detector: true,
      special_badge: 'Coming Soon',
      category_name: 'Bragi V1',
      label: 'AI audio',
      default_threshold: 0.75,
      default_color: '#FF4D1C',
      download_size: '210 MB',
      ram_size: '600 MB RAM',
      tagline: 'Voice Clone & Synthetic Audio Detector',
      description: 'Detects synthetic speech, voice cloning, and AI audio.'
    }
  ]
};

// ============================================================================
// STATE
// ============================================================================

let currentScreen = 1;
let extensionConnected = false;
let nativeHostConnected = false;
let officialCatalog = FALLBACK_CATALOG;
let installedModels = new Set();
let selectedModels = new Set(['fvf_heimdall02', 'fvf_heimdall03']); // Recommended default selected

// ============================================================================
// EXTENSION BRIDGE (Supports Direct Extension Context + GitHub Pages postMessage)
// ============================================================================

const pendingBridgeCalls = new Map();

window.addEventListener('message', (event) => {
  if (event.source !== window) return;
  const data = event.data;
  if (!data || data.source !== 'fvf-extension-bridge') return;

  if (data.type === 'ONBOARDING_DOWNLOAD_PROGRESS' && data.payload) {
    handleDownloadProgressEvent(data.payload);
    return;
  }

  if (data.requestId && pendingBridgeCalls.has(data.requestId)) {
    const { resolve, timer } = pendingBridgeCalls.get(data.requestId);
    clearTimeout(timer);
    pendingBridgeCalls.delete(data.requestId);
    resolve(data.response);
  }
});

if (ext && ext.runtime && ext.runtime.onMessage) {
  try {
    ext.runtime.onMessage.addListener((msg) => {
      if (msg && msg.type === 'ONBOARDING_DOWNLOAD_PROGRESS' && msg.payload) {
        handleDownloadProgressEvent(msg.payload);
      }
    });
  } catch (_) {}
}

function sendBridgeMessage(action, payload = {}, timeoutMs = 10000) {
  if (ext && ext.runtime && typeof ext.runtime.sendMessage === 'function') {
    return new Promise((resolve) => {
      try {
        const msg = Object.assign({}, payload, { type: action });
        const maybePromise = ext.runtime.sendMessage(msg, (resp) => {
          if (ext.runtime.lastError) {
            resolve({ ok: false, error: ext.runtime.lastError.message });
            return;
          }
          resolve(resp || { ok: true });
        });
        if (maybePromise && typeof maybePromise.then === 'function') {
          maybePromise.then(resolve).catch((err) => resolve({ ok: false, error: err?.message || String(err) }));
        }
      } catch (err) {
        resolve({ ok: false, error: err?.message || String(err) });
      }
    });
  }

  return new Promise((resolve) => {
    const requestId = 'bridge-' + Math.random().toString(36).slice(2) + '-' + Date.now();
    const timer = setTimeout(() => {
      if (pendingBridgeCalls.has(requestId)) {
        pendingBridgeCalls.delete(requestId);
        resolve({ ok: false, error: 'Extension bridge timeout' });
      }
    }, timeoutMs);

    pendingBridgeCalls.set(requestId, { resolve, timer });
    window.postMessage(
      {
        source: 'fvf-onboarding-page',
        action,
        payload,
        requestId
      },
      '*'
    );
  });
}

// ============================================================================
// STATUS CHECKS & CATALOG LOADING
// ============================================================================

let nativeCheckInterval = null;
let waitingForNativeInstall = false;

function updateNativeHostUI() {
  const skipBtn = document.getElementById('skip-to-models-btn');
  const liveBox = document.getElementById('native-live-status-box');
  const heading = document.getElementById('native-status-heading');
  const detail = document.getElementById('native-status-detail');
  const icon = document.getElementById('native-pulse-icon');
  const hint = document.getElementById('native-not-connected-hint');

  if (nativeHostConnected) {
    if (skipBtn) {
      skipBtn.disabled = false;
      skipBtn.style.opacity = '1';
      skipBtn.style.cursor = 'pointer';
    }
    if (hint) hint.style.display = 'none';

    if (liveBox) {
      liveBox.style.background = 'rgba(76, 175, 80, 0.15)';
      liveBox.style.borderColor = 'rgba(76, 175, 80, 0.45)';
    }
    if (icon) icon.textContent = '✅';
    if (heading) {
      heading.textContent = 'Native Host Connected & Ready!';
      heading.style.color = '#7cfc00';
    }
    if (detail) {
      detail.textContent = 'Local hardware-accelerated AI environment is online.';
    }

    if (waitingForNativeInstall && currentScreen === 1) {
      waitingForNativeInstall = false;
      if (nativeCheckInterval) {
        clearInterval(nativeCheckInterval);
        nativeCheckInterval = null;
      }
      setTimeout(() => {
        goToScreen(2);
      }, 1000);
    }
  } else {
    if (skipBtn) {
      skipBtn.disabled = true;
      skipBtn.style.opacity = '0.45';
      skipBtn.style.cursor = 'not-allowed';
    }
    if (liveBox) {
      liveBox.style.background = 'rgba(255, 140, 0, 0.1)';
      liveBox.style.borderColor = 'rgba(255, 140, 0, 0.35)';
    }
    if (icon) icon.textContent = '⏳';
    if (heading) {
      heading.textContent = 'Waiting for Native Host connection…';
      heading.style.color = '#ffb020';
    }
    if (detail) {
      detail.textContent = 'Keep this tab open while running the installer. Detection is automatic!';
    }
  }
}

async function checkBridgeAndNativeStatus() {
  const pingRes = await sendBridgeMessage('PING', {}, 2000).catch(() => ({ ok: false }));
  extensionConnected = Boolean(pingRes && pingRes.ok);

  if (extensionConnected) {
    const hostRes = await sendBridgeMessage('CHECK_NATIVE_HOST', {}, 3000).catch(() => ({ connected: false }));
    nativeHostConnected = Boolean(hostRes && hostRes.connected);
  } else {
    nativeHostConnected = false;
  }
  updateNativeHostUI();
}

async function loadOfficialModelsCatalog() {
  if (extensionConnected) {
    try {
      const res = await sendBridgeMessage('GET_OFFICIAL_CATALOG', {}, 4000);
      if (res && res.ok && res.catalog && Array.isArray(res.catalog.models)) {
        officialCatalog = res.catalog;
        nativeHostConnected = Boolean(res.nativeConnected);
        (res.installedModels || []).forEach((id) => {
          installedModels.add(id);
          selectedModels.add(id);
        });
        renderModalityQuadrants();
        return;
      }
    } catch (_) {}
  }

  try {
    const resp = await fetch(RAW_GITHUB_CATALOG_URL, { cache: 'no-store' });
    if (resp && resp.ok) {
      const data = await resp.json();
      if (data && Array.isArray(data.models)) {
        officialCatalog = data;
        renderModalityQuadrants();
        return;
      }
    }
  } catch (_) {}

  try {
    const resp = await fetch('./official_models.json');
    if (resp && resp.ok) {
      officialCatalog = await resp.json();
    }
  } catch (_) {}

  renderModalityQuadrants();
}

// ============================================================================
// SCREEN 2: RENDER 4 MODALITY QUADRANTS WITH ATTACHED ICONS & SQUARE CARDS
// ============================================================================

const QUADRANT_META = [
  {
    modality: 'image',
    title: 'Image Modality',
    subtitle: 'Photographs, Art, Thumbnails & Feeds',
    iconSrc: './assets/modality-image.png'
  },
  {
    modality: 'text',
    title: 'Text Modality',
    subtitle: 'Articles, Essays, Comments & Web Copy',
    iconSrc: './assets/modality-text.png'
  },
  {
    modality: 'video',
    title: 'Video Modality',
    subtitle: 'Clips, Streams & Temporal Frames',
    iconSrc: './assets/modality-video.png'
  },
  {
    modality: 'audio',
    title: 'Audio Modality',
    subtitle: 'Voice Clones, Podcasts & Speech',
    iconSrc: './assets/modality-audio.png'
  }
];

function renderModalityQuadrants() {
  const container = document.getElementById('modality-quadrants-container');
  if (!container) return;

  const models = (officialCatalog && Array.isArray(officialCatalog.models)) ? officialCatalog.models : [];
  container.innerHTML = '';

  QUADRANT_META.forEach((quad) => {
    const allInModality = models.filter((m) => m.modality === quad.modality);
    const hasAvailableModel = allInModality.some((m) => m.status === 'available');

    const visibleModels = hasAvailableModel
      ? allInModality.filter((m) => m.status === 'available')
      : allInModality;

    const quadrantEl = document.createElement('div');
    quadrantEl.className = 'glass-card-3d modality-quadrant';
    quadrantEl.setAttribute('data-tilt', '');

    quadrantEl.innerHTML = `
      <div class="quadrant-header">
        <div class="quadrant-title-group">
          <div class="quadrant-icon">
            <img src="${quad.iconSrc}" alt="${quad.title}" class="quadrant-icon-img" />
          </div>
          <div>
            <div class="quadrant-title">${quad.title}</div>
            <div class="quadrant-subtitle">${quad.subtitle}</div>
          </div>
        </div>
      </div>
      <div class="quadrant-models-list"></div>
    `;

    const listEl = quadrantEl.querySelector('.quadrant-models-list');

    visibleModels.forEach((model) => {
      const isAvailable = model.status === 'available';
      const isSelected = isAvailable && selectedModels.has(model.id);
      const isInstalled = installedModels.has(model.id);

      const tile = document.createElement('div');
      tile.className = `model-tile ${isAvailable ? 'selectable' : 'upcoming'} ${isSelected ? 'selected' : ''}`;
      tile.dataset.modelId = model.id;
      tile.setAttribute('data-tilt', '');

      // Stickers
      let stickersHtml = '';
      if (model.recommended && model.is_ai_detector !== false) {
        stickersHtml += `<span class="sticker-recommended">★ Recommended AI Classifier</span>`;
      }
      if (model.is_ai_detector === false) {
        stickersHtml += `<span class="sticker-spider-warning">Object classifier</span>`;
      } else if (!isAvailable) {
        const badgeText = model.status === 'coming_next' ? 'Coming Next' : 'Coming Soon';
        stickersHtml += `<span class="sticker-upcoming">⏳ ${badgeText}</span>`;
      }

      // Dedicated Selection / Status Line (ALWAYS on its own line)
      const statusLineHtml = isAvailable
        ? `<div class="model-selection-row">
             <label class="model-checkbox-label">
               <input type="checkbox" class="model-select-checkbox" data-model-id="${model.id}" ${isSelected ? 'checked' : ''} />
               <span class="model-checkbox-text ${isInstalled ? 'status-installed' : (isSelected ? 'status-selected' : 'status-unselected')}">
                 ${isInstalled ? '✓ Installed Locally' : (isSelected ? 'Selected for install' : 'Click to select')}
               </span>
             </label>
           </div>`
        : `<div class="model-selection-row">
             <span class="model-checkbox-text status-upcoming">In active development</span>
           </div>`;

      tile.innerHTML = `
        <div>
          <div class="model-top-row">
            <span class="model-name">${model.name}</span>
            ${stickersHtml ? `<div class="model-stickers-wrap">${stickersHtml}</div>` : ''}
          </div>
          ${statusLineHtml}
          <div class="model-tagline">${model.tagline || ''}</div>
        </div>
        <div class="model-meta-footer">
          <span class="model-arch-chip">Download: ${model.download_size || (model.size_mb ? model.size_mb + ' MB' : 'N/A')}</span>
          <span class="model-arch-chip">RAM: ${model.ram_size || 'N/A'}</span>
        </div>
        <div class="model-progress-wrap" id="progress-wrap-${model.id}" hidden>
          <div class="model-progress-bar">
            <div class="model-progress-fill" id="progress-fill-${model.id}"></div>
          </div>
          <div class="model-progress-text">
            <span id="progress-label-${model.id}">Preparing download…</span>
            <span id="progress-pct-${model.id}">0%</span>
          </div>
        </div>
      `;

      if (isAvailable) {
        tile.addEventListener('click', (e) => {
          if (e.target.tagName === 'A') return;
          toggleModelSelection(model.id);
        });
      }

      listEl.appendChild(tile);
    });

    container.appendChild(quadrantEl);
  });

  updateSelectedModelsSummary();
  refreshAllTilts();
}

function toggleModelSelection(modelId) {
  if (selectedModels.has(modelId)) {
    selectedModels.delete(modelId);
  } else {
    selectedModels.add(modelId);
  }
  renderModalityQuadrants();
}

function updateSelectedModelsSummary() {
  const summaryEl = document.getElementById('selected-models-summary');
  const btn = document.getElementById('download-and-continue-btn');
  if (!summaryEl) return;

  const models = (officialCatalog && Array.isArray(officialCatalog.models)) ? officialCatalog.models : [];
  const chosenNames = models
    .filter((m) => selectedModels.has(m.id))
    .map((m) => m.name);

  if (chosenNames.length > 0) {
    summaryEl.textContent = `Selected (${chosenNames.length}): ${chosenNames.join(' + ')}`;
    if (btn) btn.innerHTML = '<span>⬇ Download Selected &amp; Continue →</span>';
  } else {
    summaryEl.textContent = 'No models selected (you can proceed without downloading)';
    if (btn) btn.innerHTML = '<span>Continue Without Models →</span>';
  }
}

function handleDownloadProgressEvent(progress) {
  if (!progress || !progress.modelId) return;
  const wrap = document.getElementById(`progress-wrap-${progress.modelId}`);
  const fill = document.getElementById(`progress-fill-${progress.modelId}`);
  const label = document.getElementById(`progress-label-${progress.modelId}`);
  const pct = document.getElementById(`progress-pct-${progress.modelId}`);

  if (wrap) wrap.hidden = false;
  if (fill) fill.style.width = `${progress.percent || 0}%`;
  if (label) label.textContent = progress.status || `Downloading ${progress.file || ''}...`;
  if (pct) pct.textContent = `${progress.percent || 0}%`;
}

async function downloadSelectedModelsAndAdvance() {
  const btn = document.getElementById('download-and-continue-btn');
  const statusEl = document.getElementById('download-overall-status');
  const ids = Array.from(selectedModels);

  // If user chose 0 models, proceed immediately without blocking
  if (ids.length === 0) {
    goToScreen(3);
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⏳ Downloading Models…</span>';
  }

  for (const modelId of ids) {
    const wrap = document.getElementById(`progress-wrap-${modelId}`);
    if (wrap) wrap.hidden = false;

    if (extensionConnected) {
      if (statusEl) statusEl.textContent = `Requesting Native Host to install ${modelId}...`;
      const res = await sendBridgeMessage('DOWNLOAD_OFFICIAL_MODEL', { modelId }, 600000).catch(() => ({ ok: false }));
      if (res && res.ok) {
        installedModels.add(modelId);
        handleDownloadProgressEvent({
          modelId,
          percent: 100,
          status: '✓ Installed & Registered in Native Host'
        });
      } else {
        handleDownloadProgressEvent({
          modelId,
          percent: 100,
          status: 'Queued for install'
        });
      }
    } else {
      handleDownloadProgressEvent({
        modelId,
        percent: 100,
        status: 'Selected for install'
      });
    }
  }

  if (btn) {
    btn.disabled = false;
    btn.innerHTML = '<span>⬇ Download Selected &amp; Continue →</span>';
  }

  goToScreen(3);
}

// ============================================================================
// SCREEN 3: FINISH SETUP ACTION
// ============================================================================

async function runFinishSetup() {
  const btn = document.getElementById('finish-setup-btn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>Configuring Categories…</span>';
  }

  const downloadedIds = Array.from(selectedModels);
  await sendBridgeMessage('AUTO_SETUP_CATEGORIES', {
    downloadedModels: downloadedIds,
    threshold: 0.75
  }).catch(() => {});

  if (btn) {
    btn.disabled = false;
    btn.innerHTML = '<span>Finish Setup →</span>';
  }

  goToScreen(4);
}

// ============================================================================
// NAVIGATION & WIZARD SCREEN MANAGEMENT
// ============================================================================

function goToScreen(stepNumber) {
  if (stepNumber === 2 && !nativeHostConnected) {
    const hint = document.getElementById('native-not-connected-hint');
    const instructions = document.getElementById('native-install-instructions');
    if (hint) hint.style.display = 'block';
    if (instructions) instructions.style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
    return;
  }

  currentScreen = stepNumber;

  document.querySelectorAll('.wizard-screen').forEach((sec) => {
    const s = Number(sec.dataset.screen);
    sec.classList.toggle('active', s === stepNumber);
  });

  if (stepNumber === 2) {
    renderModalityQuadrants();
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================================
// VOLUMETRIC 3D TILT, PORTAL PARALLAX & EMBER PARTICLE ENGINE
// ============================================================================

function applyTiltToElement(el) {
  if (el.dataset.tiltBound) return;
  el.dataset.tiltBound = 'true';
  el.addEventListener('pointermove', (e) => {
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    el.style.transform = `perspective(1100px) rotateX(${(-py * 5.5).toFixed(2)}deg) rotateY(${(px * 5.5).toFixed(2)}deg)`;
  });
  el.addEventListener('pointerleave', () => {
    el.style.transform = 'perspective(1100px) rotateX(0deg) rotateY(0deg)';
  });
}

function refreshAllTilts() {
  document.querySelectorAll('[data-tilt]').forEach(applyTiltToElement);
}

function initVolumetric3DEffects() {
  window.addEventListener(
    'pointermove',
    (e) => {
      document.querySelectorAll('.glass-card-3d').forEach((card) => {
        const rect = card.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        card.style.setProperty('--mouse-x', `${mx}px`);
        card.style.setProperty('--mouse-y', `${my}px`);
      });
    },
    { passive: true }
  );

  refreshAllTilts();

  const canvas = document.getElementById('ember-webgl-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const embers = Array.from({ length: 58 }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    vx: 0.25 + Math.random() * 0.75,
    vy: -0.35 - Math.random() * 0.65,
    r: 1.2 + Math.random() * 2.8,
    alpha: 0.2 + Math.random() * 0.65,
    hue: 12 + Math.random() * 34
  }));

  function renderFrame() {
    ctx.clearRect(0, 0, width, height);
    for (const p of embers) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x > width + 20) p.x = -10;
      if (p.y < -20) p.y = height + 10;

      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4.5);
      grad.addColorStop(0, `hsla(${p.hue}, 100%, 72%, ${p.alpha})`);
      grad.addColorStop(0.45, `hsla(${p.hue - 6}, 100%, 54%, ${p.alpha * 0.45})`);
      grad.addColorStop(1, 'hsla(10, 100%, 40%, 0)');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 4.5, 0, Math.PI * 2);
      ctx.fill();
    }
    requestAnimationFrame(renderFrame);
  }
  requestAnimationFrame(renderFrame);
}

// ============================================================================
// INITIALIZE APP & EVENT LISTENERS
// ============================================================================

function init() {
  initVolumetric3DEffects();
  renderModalityQuadrants();

  // Screen 1 buttons
  const dlNativeBtn = document.getElementById('download-native-host-btn');
  if (dlNativeBtn) {
    dlNativeBtn.addEventListener('click', () => {
      // Reveal the 2-step File Explorer instructions
      const instructions = document.getElementById('native-install-instructions');
      if (instructions) instructions.style.display = 'block';
      waitingForNativeInstall = true;

      // Start live connection heartbeat check every 1.5s
      if (!nativeCheckInterval) {
        nativeCheckInterval = setInterval(async () => {
          await checkBridgeAndNativeStatus().catch(() => {});
        }, 1500);
      }
    });
  }

  const checkConnBtn = document.getElementById('check-connection-btn');
  if (checkConnBtn) {
    checkConnBtn.addEventListener('click', async () => {
      checkConnBtn.disabled = true;
      checkConnBtn.textContent = 'Checking…';
      await checkBridgeAndNativeStatus().catch(() => {});
      checkConnBtn.disabled = false;
      checkConnBtn.textContent = '↻ Check Now';
      if (nativeHostConnected) {
        goToScreen(2);
      }
    });
  }

  const skipBtn = document.getElementById('skip-to-models-btn');
  if (skipBtn) {
    skipBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (!nativeHostConnected) {
        const hint = document.getElementById('native-not-connected-hint');
        const instructions = document.getElementById('native-install-instructions');
        if (hint) hint.style.display = 'block';
        if (instructions) instructions.style.display = 'block';
        return;
      }
      goToScreen(2);
    });
  }

  // Periodic heartbeat on screen 1 to auto-detect connection
  setInterval(() => {
    if (currentScreen === 1) {
      checkBridgeAndNativeStatus().catch(() => {});
    }
  }, 2500);

  // Screen 2 buttons
  document.getElementById('back-to-step1-btn')?.addEventListener('click', () => goToScreen(1));
  document.getElementById('download-and-continue-btn')?.addEventListener('click', () => {
    downloadSelectedModelsAndAdvance();
  });

  // Screen 3 buttons (Setting Up Your First Classifier)
  document.getElementById('back-to-step2-btn')?.addEventListener('click', () => goToScreen(2));
  document.getElementById('finish-setup-btn')?.addEventListener('click', () => {
    runFinishSetup();
  });

  // Background status verification (non-blocking)
  checkBridgeAndNativeStatus().catch(() => {});
  loadOfficialModelsCatalog().catch(() => {});
}

// Immediate execution if DOM is ready, or on DOMContentLoaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
