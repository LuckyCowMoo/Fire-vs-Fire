// ============================================================================
// DYNAMIC BASE CLASSIFIER SOURCE FETCHER
// ============================================================================

async function loadBaseClassifierCode() {
  const codeEl = document.getElementById('base-classifier-code') || document.querySelector('.byo-classifier-card pre code');
  if (!codeEl) return;
  const urls = [
    'https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/dynamic-prioritisation/native/classifiers/base_classifier.py',
    'https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/FVF-clean/native/classifiers/base_classifier.py',
    'https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/main/native/classifiers/base_classifier.py',
    '../native/classifiers/base_classifier.py'
  ];
  for (const url of urls) {
    try {
      const resp = await fetch(`${url}?_t=${Date.now()}`, { cache: 'no-store' });
      if (resp && resp.ok) {
        const text = await resp.text();
        if (text && text.trim().length > 50) {
          codeEl.textContent = text;
          return;
        }
      }
    } catch (_) {}
  }
}

// Call on startup
loadBaseClassifierCode().catch(() => {});
/**
 * Fire vs Fire — Interactive Onboarding & Model Hub
 * Connects seamlessly to the Browser Extension & Local Native Host (V2)
 */

const ext = typeof browser !== 'undefined' ? browser : (typeof chrome !== 'undefined' ? chrome : null);

const RAW_GITHUB_CATALOG_URL =
  'https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/FVF-clean/native/official_models.json';

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
      recommended: false,
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
      id: 'fvf_heimdall04',
      name: 'FvF Heimdall 04',
      version: '4.0.0-preview',
      modality: 'image',
      status: 'available',
      recommended: true,
      is_ai_detector: true,
      category_name: 'FvF Heimdall 04',
      label: 'AI generated',
      default_threshold: 0.75,
      default_color: '#FF1744',
      download_size: '198 MB',
      ram_size: '1.5 GB VRAM / RAM',
      tagline: '24-Ch FFT + ForensicNeXt + Intrinsic Grouping',
      description: 'Our latest flagship AI image detector with ConvNeXt-V2 Base, 24-ch FFT, and Intrinsic Structural Grouping.'
    },
    {
      id: 'res_net50_spider',
      name: 'ResNet 50 - Spider',
      version: '1.0.0',
      modality: 'image',
      status: 'available',
      is_ai_detector: false,
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
      name: 'FvF Mímir V2',
      version: '2.2.0',
      modality: 'text',
      status: 'available',
      recommended: true,
      is_ai_detector: true,
      category_name: 'FvF Mímir V2',
      label: 'AI text',
      default_threshold: 0.75,
      default_color: '#FF6218',
      download_size: '167 MB',
      ram_size: '420 MB RAM',
      tagline: 'Hardware-Accelerated Dual-Channel AI Text Classifier',
      description: 'Dual-channel AI text detector combining a calibrated DistilRoBERTa-AFPL encoder with Causal LM perplexity curvature and length-normalized Mindner stylometrics.'
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
let activeDownloads = new Set();

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
  if (skipBtn) {
    skipBtn.addEventListener('click', (e) => {
      e.preventDefault();
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
    goToScreen(4);
  });

  // Screen 3 buttons (Forum Architecture Guide)
  document.getElementById('back-to-step2-btn')?.addEventListener('click', () => goToScreen(2));
  document.getElementById('finish-setup-btn')?.addEventListener('click', () => {
    runFinishSetup();
  });
  document.getElementById('goto-guide-btn')?.addEventListener('click', () => goToScreen(3));

  const helpfulBtn = document.getElementById('forum-helpful-btn');
  if (helpfulBtn) {
    let count = 42;
    helpfulBtn.addEventListener('click', () => {
      if (!helpfulBtn.classList.contains('voted')) {
        helpfulBtn.classList.add('voted');
        count += 1;
        helpfulBtn.innerHTML = `<span class="helpful-thumb">✓</span> <span>Helpful Guide (${count})</span>`;
      }
    });
  }

  // Bypass link for users who already installed or are testing
  const forceBtn = document.getElementById('force-proceed-models-btn');
  if (forceBtn) {
    forceBtn.addEventListener('click', (e) => {
      e.preventDefault();
      nativeHostConnected = true;
      updateNativeHostUI();
      goToScreen(2);
    });
  }

  // Handle direct navigation to screen from URL (e.g. ?screen=2 or ?screen=3)
  const urlParams = new URLSearchParams(window.location.search);
  const targetScreen = parseInt(urlParams.get('screen') || urlParams.get('step') || '0', 10);
  if (targetScreen >= 2 && targetScreen <= 4) {
    nativeHostConnected = true; // Bypass lock when navigating directly from extension options
    goToScreen(targetScreen);
  } else if (window.location.hash === '#models' || window.location.hash === '#screen-2') {
    nativeHostConnected = true;
    goToScreen(2);
  } else if (window.location.hash === '#guide' || window.location.hash === '#screen-3') {
    goToScreen(3);
  }

  // Auto-scan local classifiers every 6 seconds on Screen 2
  setInterval(() => {
    if (currentScreen === 2 && !document.hidden) {
      loadOfficialModelsCatalog(true).catch(() => {});
    }
  }, 6000);

  // Auto-refresh when tab regains focus or visibility (so dropping files updates immediately)
  window.addEventListener('focus', () => {
    if (currentScreen === 2) {
      loadOfficialModelsCatalog(true).catch(() => {});
    }
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && currentScreen === 2) {
      loadOfficialModelsCatalog(true).catch(() => {});
    }
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
