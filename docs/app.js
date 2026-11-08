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
  updated_at: '2026-11-08T18:00:00Z',
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
      size_mb: 463.4,
      architecture: '5-Crop 20-Channel SRNet + ConvNeXt-Small Fusion',
      tagline: 'Flagship AI Image Detector • 20 Forensic Channels × 5 Spatial Crops',
      description: 'Our flagship first-party AI image classifier. Extracts 20 low-level forensic signal channels (FFT spectra, DCT residuals, SRM filters, chromatic aberration, and noise variance) across 5 spatial crops (center, corners, and edges) fused with a ConvNeXt-Small semantic backbone.'
    },
    {
      id: 'res_net50_spider',
      name: 'ResNet 50 - Spider',
      version: '1.0.0',
      modality: 'image',
      status: 'available',
      recommended: false,
      is_ai_detector: false,
      special_badge: 'Detects Spiders • NOT AI',
      category_name: 'ResNet50 Spider',
      label: 'Spider',
      default_threshold: 0.75,
      default_color: '#FFB020',
      size_mb: 90.0,
      architecture: 'ResNet-50 Binary Vision Classifier',
      tagline: 'Arachnophobia Shield • Detects Spiders, NOT AI Images',
      description: 'Demonstrates that Fire vs Fire can classify any visual concept—not just AI! This model detects real and illustrated spiders on web pages so they are blurred automatically before you see them. Note: This model detects spiders, NOT AI-generated content.'
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
      size_mb: 280.0,
      architecture: 'Bi-Directional Token Burstiness & Syntactic Transformer',
      tagline: 'Next in Queue • Local AI Text & Article Classifier',
      description: 'Local neural language model trained to flag LLM-generated prose, essays, fake reviews, and synthetic comments using token burstiness variance and perplexity signatures.'
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
      size_mb: 520.0,
      architecture: 'Temporal Flow & Frame Inter-Consistency Network',
      tagline: 'Coming Soon • Local AI Video & Deepfake Detector',
      description: 'Analyzes video streams for temporal flicker, optical flow morphing discontinuities, and frame-to-frame diffusion latent shifts.'
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
      size_mb: 210.0,
      architecture: 'Phase Vocoder & Spectral Inharmonicity Discriminator',
      tagline: 'Coming Soon • Voice Clone & Synthetic Audio Detector',
      description: 'Detects synthetic speech, voice cloning, and AI audio by scanning for phase incoherence and spectral repetition in HTML5 web audio.'
    }
  ]
};

// ============================================================================
// STATE
// ============================================================================

let currentScreen = 1;
let currentTutorialStep = 1;
let extensionConnected = false;
let nativeHostConnected = false;
let officialCatalog = FALLBACK_CATALOG;
let installedModels = new Set();
let selectedModels = new Set(['fvf_heimdall02']); // Recommended default selected

// Sandbox state for Module 3 of the Tutorial
let sandboxCategories = [
  {
    id: 'auto_image_fvf_heimdall02',
    name: 'FvF Heimdall V2',
    modality: 'image',
    enabled: true,
    threshold: 0.75,
    classifiers: ['fvf_heimdall02'],
    weights: [1],
    blurMode: 'hover',
    blurAmount: 8,
    borderColor: '#FF4D1C'
  }
];
let activeSandboxCatIndex = 0;

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
  // 1. Direct extension page context (moz-extension:// or chrome-extension://)
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

  // 2. GitHub Pages / Web context via content.js window.postMessage bridge
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

async function checkBridgeAndNativeStatus() {
  const pingRes = await sendBridgeMessage('PING', {}, 2000).catch(() => ({ ok: false }));
  extensionConnected = Boolean(pingRes && pingRes.ok);

  if (extensionConnected) {
    const hostRes = await sendBridgeMessage('CHECK_NATIVE_HOST', {}, 3000).catch(() => ({ connected: false }));
    nativeHostConnected = Boolean(hostRes && hostRes.connected);
  } else {
    nativeHostConnected = false;
  }
}

async function loadOfficialModelsCatalog() {
  // 1. Try via Extension -> Native Host first (gives live local installed status)
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

  // 2. Try fetching directly from GitHub Raw
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

  // 3. Fallback to local ./official_models.json
  try {
    const resp = await fetch('./official_models.json');
    if (resp && resp.ok) {
      officialCatalog = await resp.json();
    }
  } catch (_) {}

  renderModalityQuadrants();
}

// ============================================================================
// SCREEN 2: RENDER 4 MODALITY QUADRANTS WITH DYNAMIC PLACEHOLDER LOGIC
// ============================================================================

const QUADRANT_META = [
  {
    modality: 'image',
    title: 'Image Modality',
    subtitle: 'Photographs, Art, Thumbnails & Social Media Feeds',
    icon: '🖼️'
  },
  {
    modality: 'text',
    title: 'Text Modality',
    subtitle: 'Articles, Essays, Comments & Web Copy',
    icon: '📝'
  },
  {
    modality: 'video',
    title: 'Video Modality',
    subtitle: 'Short-Form Clips, Streams & Temporal Frames',
    icon: '🎬'
  },
  {
    modality: 'audio',
    title: 'Audio Modality',
    subtitle: 'Voice Clones, Podcasts & Synthetic Speech',
    icon: '🎙️'
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

    // Dynamic Catalog Rule:
    // When at least one model is "available" for a modality, hide "coming_soon" / "coming_next"
    const visibleModels = hasAvailableModel
      ? allInModality.filter((m) => m.status === 'available')
      : allInModality;

    const quadrantEl = document.createElement('div');
    quadrantEl.className = 'glass-card-3d modality-quadrant';

    const statusLabel = hasAvailableModel
      ? `${visibleModels.length} Active Model${visibleModels.length > 1 ? 's' : ''}`
      : 'Upcoming Release';

    quadrantEl.innerHTML = `
      <div class="quadrant-header">
        <div class="quadrant-title-group">
          <div class="quadrant-icon">${quad.icon}</div>
          <div>
            <div class="quadrant-title">${quad.title}</div>
            <div class="quadrant-subtitle">${quad.subtitle}</div>
          </div>
        </div>
        <span class="brand-tag">${statusLabel}</span>
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

      // Stickers
      let stickersHtml = '';
      if (model.recommended && model.is_ai_detector !== false) {
        stickersHtml += `<span class="sticker-recommended">★ Recommended AI Classifier</span>`;
      }
      if (model.is_ai_detector === false) {
        stickersHtml += `<span class="sticker-spider-warning">🕷️ ${model.special_badge || 'Detects Spiders • NOT AI'}</span>`;
      } else if (!isAvailable) {
        const badgeText = model.status === 'coming_next' ? 'Coming Next' : 'Coming Soon';
        stickersHtml += `<span class="sticker-upcoming">⏳ ${badgeText}</span>`;
      }

      const checkboxStateHtml = isAvailable
        ? `<div style="display: flex; align-items: center; gap: 8px;">
             <input type="checkbox" class="model-select-checkbox" data-model-id="${model.id}" ${isSelected ? 'checked' : ''} style="width: 18px; height: 18px; accent-color: #ff6218; cursor: pointer;" />
             <span style="font-size: 0.8rem; font-weight: 700; color: ${isInstalled ? '#34d399' : '#ffcaa6'};" id="tile-status-${model.id}">
               ${isInstalled ? '✓ Installed Locally' : (isSelected ? 'Selected for Setup' : 'Click to Select')}
             </span>
           </div>`
        : `<span style="font-size: 0.78rem; color: var(--text-muted);">In active development by LuckyCowMoo</span>`;

      tile.innerHTML = `
        <div class="model-top-row">
          <div class="model-name-cluster">
            <span class="model-name">${model.name}</span>
            ${stickersHtml}
          </div>
          ${checkboxStateHtml}
        </div>
        <div class="model-tagline">${model.tagline || ''}</div>
        <p class="model-desc">${model.description || ''}</p>
        <div class="model-meta-footer">
          <span class="model-arch-chip">${model.architecture || ''}</span>
          <span class="model-arch-chip">${model.size_mb ? `~${model.size_mb} MB (.py + .pt)` : ''}</span>
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
}

function toggleModelSelection(modelId) {
  if (selectedModels.has(modelId)) {
    if (selectedModels.size > 1) {
      selectedModels.delete(modelId);
    }
  } else {
    selectedModels.add(modelId);
  }
  renderModalityQuadrants();
  syncSandboxWithSelection();
}

function updateSelectedModelsSummary() {
  const summaryEl = document.getElementById('selected-models-summary');
  if (!summaryEl) return;

  const models = (officialCatalog && Array.isArray(officialCatalog.models)) ? officialCatalog.models : [];
  const chosenNames = models
    .filter((m) => selectedModels.has(m.id))
    .map((m) => (m.id === 'res_net50_spider' ? `${m.name} (Spider Filter)` : m.name));

  summaryEl.textContent = chosenNames.length
    ? `Selected (${chosenNames.length}): ${chosenNames.join(' + ')}`
    : 'Select at least one classifier model above';
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

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⏳ Verifying &amp; Downloading Models…</span>';
  }

  for (const modelId of ids) {
    const wrap = document.getElementById(`progress-wrap-${modelId}`);
    if (wrap) wrap.hidden = false;

    if (extensionConnected) {
      if (statusEl) statusEl.textContent = `Requesting Native Host to install ${modelId} (.py + .pt)...`;
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
          status: 'Queued for Category Setup'
        });
      }
    } else {
      handleDownloadProgressEvent({
        modelId,
        percent: 100,
        status: 'Selected for Category Setup'
      });
    }
  }

  if (btn) {
    btn.disabled = false;
    btn.innerHTML = '<span>⬇ Download Selected &amp; Continue →</span>';
  }

  syncSandboxWithSelection();
  goToScreen(3);
}

// ============================================================================
// SCREEN 3 & 4: AUTO-SETUP ("SET UP FOR ME") & INTERACTIVE TUTORIAL SANDBOX
// ============================================================================

function syncSandboxWithSelection() {
  const newCats = [];
  if (selectedModels.has('fvf_heimdall02') || selectedModels.size === 0) {
    newCats.push({
      id: 'auto_image_fvf_heimdall02',
      name: 'FvF Heimdall V2',
      modality: 'image',
      enabled: true,
      threshold: 0.75,
      classifiers: ['fvf_heimdall02'],
      weights: [1],
      blurMode: 'hover',
      blurAmount: 8,
      borderColor: '#FF4D1C'
    });
  }
  if (selectedModels.has('res_net50_spider')) {
    newCats.push({
      id: 'auto_image_res_net50_spider',
      name: 'ResNet50 Spider',
      modality: 'image',
      enabled: true,
      threshold: 0.75,
      classifiers: ['res_net50_spider'],
      weights: [1],
      blurMode: 'hover',
      blurAmount: 8,
      borderColor: '#FFB020'
    });
  }
  if (newCats.length) {
    sandboxCategories = newCats;
    activeSandboxCatIndex = 0;
    renderSandboxUI();
  }
}

async function runAutoSetupForMe() {
  const btn = document.getElementById('btn-setup-for-me');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⚡ Configuring Categories at 75% Sensitivity…</span>';
  }

  const downloadedIds = Array.from(selectedModels);
  await sendBridgeMessage('AUTO_SETUP_CATEGORIES', {
    downloadedModels: downloadedIds,
    threshold: 0.75
  }).catch(() => {});

  const summaryEl = document.getElementById('final-ready-summary');
  const hasSpider = selectedModels.has('res_net50_spider');
  if (summaryEl) {
    summaryEl.innerHTML = hasSpider
      ? `We automatically created your <strong>FvF Heimdall V2 (AI Images)</strong> category and your <strong>ResNet50 Spider</strong> category, both set to <strong>75% sensitivity</strong> with Hover-to-Reveal Blur enabled!`
      : `We automatically created your <strong>FvF Heimdall V2 (AI Images)</strong> category set to <strong>75% sensitivity</strong> with Hover-to-Reveal Blur, Border, and Badge enabled!`;
  }

  if (btn) {
    btn.disabled = false;
    btn.innerHTML = '<span>⚡ Set up for me &amp; Finish</span>';
  }

  goToScreen(5);
}

function renderSandboxUI() {
  const strip = document.getElementById('sandbox-cat-strip');
  if (!strip) return;
  strip.innerHTML = '';

  sandboxCategories.forEach((cat, idx) => {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className = `sandbox-cat-tab ${idx === activeSandboxCatIndex ? 'active' : ''}`;
    tab.style.borderLeft = `4px solid ${cat.borderColor}`;
    tab.textContent = cat.name;
    tab.addEventListener('click', () => {
      activeSandboxCatIndex = idx;
      renderSandboxUI();
    });
    strip.appendChild(tab);
  });

  const current = sandboxCategories[activeSandboxCatIndex];
  if (!current) return;

  const threshSlider = document.getElementById('sandbox-threshold-slider');
  const threshLabel = document.getElementById('sandbox-threshold-label');
  if (threshSlider) threshSlider.value = String(current.threshold);
  if (threshLabel) threshLabel.textContent = `${Math.round(current.threshold * 100)}%`;

  const blurPxSlider = document.getElementById('sandbox-blurpx-slider');
  const blurPxLabel = document.getElementById('sandbox-blurpx-label');
  if (blurPxSlider) blurPxSlider.value = String(current.blurAmount);
  if (blurPxLabel) blurPxLabel.textContent = `${current.blurAmount} px`;

  document.querySelectorAll('[data-blurmode]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.blurmode === current.blurMode);
  });

  applySandboxVisualsToTestCards();
}

function applySandboxVisualsToTestCards(hoveredCardId = null) {
  const current = sandboxCategories[activeSandboxCatIndex];
  if (!current) return;

  const cards = [
    { id: 'sandbox-card-ai', badgeId: 'sandbox-badge-ai', score: 0.998 },
    { id: 'sandbox-card-real', badgeId: 'sandbox-badge-real', score: 0.006 }
  ];

  cards.forEach(({ id, badgeId, score }) => {
    const cardEl = document.getElementById(id);
    const badgeEl = document.getElementById(badgeId);
    if (!cardEl) return;
    const imgEl = cardEl.querySelector('img');
    const stageEl = cardEl.querySelector('.sandbox-img-stage');
    if (!imgEl || !stageEl) return;

    const isFlagged = score >= current.threshold;
    const isHovered = hoveredCardId === id;

    if (!isFlagged) {
      imgEl.style.filter = 'none';
      stageEl.style.outline = 'none';
      if (badgeEl) badgeEl.style.display = 'none';
      return;
    }

    // Apply border & badge
    stageEl.style.outline = `3px solid ${current.borderColor}`;
    stageEl.style.outlineOffset = '-3px';
    if (badgeEl) {
      badgeEl.style.display = 'inline-block';
      badgeEl.style.borderColor = current.borderColor;
      badgeEl.textContent = `${current.name} • ${Math.round(score * 100)}%`;
    }

    // Apply Blur Mode (hover / always / off)
    if (current.blurMode === 'off') {
      imgEl.style.filter = 'none';
    } else if (current.blurMode === 'always') {
      imgEl.style.filter = `blur(${current.blurAmount}px)`;
    } else {
      // 'hover' mode: blurred at rest, unblurred when hovered!
      imgEl.style.filter = isHovered ? 'none' : `blur(${current.blurAmount}px)`;
    }
  });
}

async function saveTutorialSettingsAndFinish() {
  const formattedEnsembles = sandboxCategories.map((cat) => ({
    id: cat.id,
    name: cat.name,
    modality: cat.modality || 'image',
    enabled: true,
    threshold: cat.threshold,
    classifiers: cat.classifiers,
    weights: cat.weights || [1],
    styles: {
      image: {
        applyBlur: cat.blurMode !== 'off',
        blurAmount: cat.blurAmount,
        applyBorder: true,
        borderMultiplier: 1.2,
        borderColor: cat.borderColor,
        applyBadge: true,
        blurMode: cat.blurMode,
        borderMode: 'hover',
        badgeMode: 'hover'
      }
    }
  }));

  await sendBridgeMessage('SAVE_ONBOARDING_ENSEMBLES', {
    ensembles: formattedEnsembles
  }).catch(() => {});

  goToScreen(5);
}

// ============================================================================
// NAVIGATION & WIZARD SCREEN MANAGEMENT
// ============================================================================

function goToScreen(stepNumber) {
  currentScreen = stepNumber;

  document.querySelectorAll('.wizard-screen').forEach((sec) => {
    const s = Number(sec.dataset.screen);
    sec.classList.toggle('active', s === stepNumber);
  });

  if (stepNumber === 2) {
    renderModalityQuadrants();
  }

  if (stepNumber === 3 || stepNumber === 4) {
    renderSandboxUI();
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function goToTutorialModule(tutStep) {
  currentTutorialStep = tutStep;
  document.querySelectorAll('.tut-tab-btn').forEach((btn) => {
    btn.classList.toggle('active', Number(btn.dataset.tutStep) === tutStep);
  });
  document.querySelectorAll('.tutorial-module').forEach((mod, idx) => {
    mod.classList.toggle('active', idx + 1 === tutStep);
  });
}

// ============================================================================
// BROWSER AUTO-DETECTION FOR TUTORIAL MODULE 1
// ============================================================================

const BROWSER_PIN_GUIDES = {
  firefox:
    'In <strong>Firefox</strong>: Click the Puzzle Piece (🧩) icon in the top-right toolbar, click the Gear (⚙️) next to <strong>Fire Vs Fire</strong>, and select <strong>"Pin to Toolbar"</strong>.',
  chrome:
    'In <strong>Chrome / Brave</strong>: Click the Puzzle Piece (🧩) icon to the right of your address bar, then click the <strong>Pin (📌)</strong> icon next to <strong>Fire Vs Fire</strong>.',
  edge:
    'In <strong>Microsoft Edge</strong>: Click the Puzzle Piece (🧩) icon near the top-right corner, then click the <strong>Eye / Pin icon</strong> next to <strong>Fire Vs Fire</strong> to keep it visible.',
  opera:
    'In <strong>Opera</strong>: Click the Cube (Extensions) icon in the top-right bar, then click the <strong>Pin (📌)</strong> icon beside <strong>Fire Vs Fire</strong>.'
};

function initBrowserPinSimulator() {
  const ua = navigator.userAgent.toLowerCase();
  let detected = 'firefox';
  if (ua.includes('edg/')) detected = 'edge';
  else if (ua.includes('opr/') || ua.includes('opera')) detected = 'opera';
  else if (ua.includes('chrome') && !ua.includes('firefox')) detected = 'chrome';

  const setBrowserGuide = (browserKey) => {
    document.querySelectorAll('[data-browser]').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.browser === browserKey);
    });
    const instr = document.getElementById('browser-pin-instruction');
    if (instr && BROWSER_PIN_GUIDES[browserKey]) {
      instr.innerHTML = BROWSER_PIN_GUIDES[browserKey];
    }
  };

  setBrowserGuide(detected);

  document.querySelectorAll('[data-browser]').forEach((btn) => {
    btn.addEventListener('click', () => setBrowserGuide(btn.dataset.browser));
  });

  const pinBtn = document.getElementById('sim-pin-action-btn');
  const pinnedIcon = document.getElementById('sim-pinned-fvf-icon');
  if (pinBtn && pinnedIcon) {
    pinBtn.addEventListener('click', () => {
      pinnedIcon.style.display = 'grid';
      pinBtn.innerHTML = '✓ Pinned to Browser Bar!';
    });
  }
}

// ============================================================================
// VOLUMETRIC 3D TILT, PORTAL PARALLAX & EMBER PARTICLE ENGINE
// ============================================================================

function initVolumetric3DEffects() {
  const portal = document.getElementById('portal-assembly');
  const orbRig = document.getElementById('orb-3d-rig');

  window.addEventListener(
    'pointermove',
    (e) => {
      const nx = (e.clientX / window.innerWidth - 0.5) * 2;
      const ny = (e.clientY / window.innerHeight - 0.5) * 2;

      if (portal) {
        portal.style.transform = `rotateY(${nx * 8}deg) rotateX(${-ny * 6}deg) translateZ(0)`;
      }
      if (orbRig) {
        orbRig.style.transform = `rotateY(${nx * 16}deg) rotateX(${-ny * 14}deg) translate3d(${nx * -18}px, ${ny * -14}px, 0)`;
      }

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

  document.querySelectorAll('[data-tilt]').forEach((card) => {
    card.addEventListener('pointermove', (e) => {
      const rect = card.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width - 0.5;
      const py = (e.clientY - rect.top) / rect.height - 0.5;
      card.style.transform = `perspective(1100px) rotateX(${(-py * 4.5).toFixed(2)}deg) rotateY(${(px * 4.5).toFixed(2)}deg)`;
    });
    card.addEventListener('pointerleave', () => {
      card.style.transform = 'perspective(1100px) rotateX(0deg) rotateY(0deg)';
    });
  });

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
  initBrowserPinSimulator();
  renderSandboxUI();
  renderModalityQuadrants();

  // Screen 1 buttons
  const dlNativeBtn = document.getElementById('download-native-host-btn');
  if (dlNativeBtn) {
    dlNativeBtn.addEventListener('click', () => {
      setTimeout(() => goToScreen(2), 900);
    });
  }

  const skipBtn = document.getElementById('skip-to-models-btn');
  if (skipBtn) {
    skipBtn.addEventListener('click', (e) => {
      e.preventDefault();
      goToScreen(2);
    });
  }

  // Screen 2 buttons
  document.getElementById('back-to-step1-btn')?.addEventListener('click', () => goToScreen(1));
  document.getElementById('download-and-continue-btn')?.addEventListener('click', () => {
    downloadSelectedModelsAndAdvance();
  });

  // Screen 3 buttons ("Set up for me" vs "Take a quick tutorial - 5 min")
  document.getElementById('btn-setup-for-me')?.addEventListener('click', () => {
    runAutoSetupForMe();
  });
  document.getElementById('btn-start-tutorial')?.addEventListener('click', () => {
    goToScreen(4);
    goToTutorialModule(1);
  });

  // Screen 4 Tutorial navigation
  document.querySelectorAll('.tut-tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => goToTutorialModule(Number(btn.dataset.tutStep)));
  });
  document.querySelectorAll('[data-next-tut]').forEach((btn) => {
    btn.addEventListener('click', () => goToTutorialModule(Number(btn.dataset.nextTut)));
  });
  document.querySelectorAll('[data-prev-tut]').forEach((btn) => {
    btn.addEventListener('click', () => goToTutorialModule(Number(btn.dataset.prevTut)));
  });

  // Sandbox controls in Module 3
  document.getElementById('sandbox-threshold-slider')?.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value) || 0.75;
    if (sandboxCategories[activeSandboxCatIndex]) {
      sandboxCategories[activeSandboxCatIndex].threshold = val;
      renderSandboxUI();
    }
  });

  document.getElementById('sandbox-blurpx-slider')?.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10) || 0;
    if (sandboxCategories[activeSandboxCatIndex]) {
      sandboxCategories[activeSandboxCatIndex].blurAmount = val;
      renderSandboxUI();
    }
  });

  document.querySelectorAll('[data-blurmode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (sandboxCategories[activeSandboxCatIndex]) {
        sandboxCategories[activeSandboxCatIndex].blurMode = btn.dataset.blurmode;
        renderSandboxUI();
      }
    });
  });

  document.querySelectorAll('.sandbox-color-swatch').forEach((swatch) => {
    swatch.addEventListener('click', () => {
      if (sandboxCategories[activeSandboxCatIndex]) {
        sandboxCategories[activeSandboxCatIndex].borderColor = swatch.dataset.color;
        renderSandboxUI();
      }
    });
  });

  document.getElementById('sandbox-add-cat-btn')?.addEventListener('click', () => {
    const idx = sandboxCategories.length + 1;
    sandboxCategories.push({
      id: `custom_cat_${Date.now()}`,
      name: `Custom Category ${idx}`,
      modality: 'image',
      enabled: true,
      threshold: 0.75,
      classifiers: ['fvf_heimdall02'],
      weights: [1],
      blurMode: 'hover',
      blurAmount: 8,
      borderColor: '#38BDF8'
    });
    activeSandboxCatIndex = sandboxCategories.length - 1;
    renderSandboxUI();
  });

  // Hover interactions on Sandbox Test Cards
  ['sandbox-card-ai', 'sandbox-card-real'].forEach((cardId) => {
    const el = document.getElementById(cardId);
    if (!el) return;
    el.addEventListener('pointerenter', () => applySandboxVisualsToTestCards(cardId));
    el.addEventListener('pointerleave', () => applySandboxVisualsToTestCards(null));
  });

  document.getElementById('btn-finish-tutorial')?.addEventListener('click', () => {
    saveTutorialSettingsAndFinish();
  });

  // Screen 5 buttons
  document.getElementById('btn-close-tab')?.addEventListener('click', () => {
    window.close();
  });
  document.getElementById('btn-review-models')?.addEventListener('click', () => goToScreen(2));
  document.getElementById('btn-replay-tutorial')?.addEventListener('click', () => {
    goToScreen(4);
    goToTutorialModule(1);
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
