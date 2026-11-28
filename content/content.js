// Cross-browser extension API abstraction (Chrome/Firefox compatibility)
if (window.__AI_DETECTOR_BOOTSTRAPPED__) {
  console.warn('[AI Detector] Content script already initialized; skipping duplicate init');
} else {
  window.__AI_DETECTOR_BOOTSTRAPPED__ = true;

const ext = typeof browser !== 'undefined' ? browser : chrome;
const VE = globalThis.FVFVisualEffects;
const ENS = globalThis.FVFEnsembles;
const TU = globalThis.FVFTextUtils;

if (typeof window !== 'undefined' && window.location && typeof window.location.hash === 'string' && window.location.hash.includes('fvf_bellows_collector=1')) {
  window.__FVF_BELLOWS_COLLECTOR_ACTIVE = true;
}

// ============================================================================
// FvF BELLOWS — INTERACTIVE DATASET GATHERING MODE (GREEN EMBERS + 1-CLICK)
// ============================================================================
const bellowsCollector = {
  active: false,
  paused: false,
  boxMode: false,
  jobId: null,
  modality: 'image',
  prompt: '',
  counts: { positives: 0, hard_negatives: 0, base_negatives: 0 },
  elToRecord: new WeakMap(),
  trackedEls: new Set(),
  overlayRefs: new Map(),
  hudEl: null,
  emberCanvas: null,
  emberRaf: null,
  frameEl: null,
  marqueeEl: null,
  dragStart: null
};

function isBellowsCollectorActive() {
  return Boolean(window.__FVF_BELLOWS_COLLECTOR_ACTIVE || bellowsCollector.active);
}

// Unique page context identifier for this page instance (used by Native Host to drop stale items on navigation)
let currentPageId = 'page-' + Math.random().toString(16).slice(2, 10) + '-' + Date.now();

/**
 * USER SETTINGS - Loaded from extension storage
 * These values are synced from the options page
 */
let settings = {
  enabled: true,                  // Master toggle for the extension
  altTextOnly: true,              // If true, only process images with alt text (default: true)
  ignoreSmallImages: true,        // Skip physically small images such as avatars and UI icons
  investigator: true,             // Automatically locate higher-res candidates for small/compressed thumbnails
  blurAmount: 3,                  // Blur radius (px) for flagged images
  borderMultiplier: 1,            // Multiplier for border width calculation
  borderColor: '#ff0064',       // Color for image borders and text strikethrough (red)
  miniBatchSize: 16,              // Legacy setting
  classificationDelay: 10,        // Minimal micro-task yield (ms)
  imageCaptureQuality: 1,         // JPEG quality (0.2-1) for image capture
  textBlurAmount: 2,              // Blur radius (px) for flagged text
  textStrikethroughEnabled: true, // Whether to apply strikethrough on AI text
  imageAiThreshold: 0.75,          // Threshold (0-1) for flagging images as AI (0.5 = 50%)
  minImageDimension: 80,          // Minimum natural and rendered pixel size (px) for images
  textAiThreshold: 0.75,           // Threshold (0-1) for flagging text as AI
  minTextLength: 250,             // Minimum character length for text to be classified
  verboseLogs: false,             // Show detailed logging for processing details
  lazyLoad: false,                // Unload models after use (true) or keep in memory (false)
  ensembles: null                 // Ensemble configs (loaded from storage)
};

/**
 * PAGE TRACKING - Accumulating data and tracking
 */
let cachedImages = [];           // Legacy compatibility array
let cachedTextSections = [];     // Legacy compatibility array
let elementMap = new Map();      // Maps itemId to DOM elements
let elementToItemId = new WeakMap(); // Reverse lookup: DOM element -> itemId
let hoverHandlerAttached = false; // Tracks whether a global hover handler is active already
let currentHoverEl = null;       // Tracks the current hovered AI-detected element
let hoverRafPending = false;     // Throttle pointermove/scroll hover updates to one per frame
let lastPointerState = null;     // Last observed { clientX, clientY, target }
let lastHitTestCoords = null;    // Last evaluated { clientX, clientY, target }
let hoverDirty = false;          // Re-hit-test after scroll even if cursor is stationary
let hoverPriorityItemId = null;  // Pending itemId currently hovered (before/after dwell)
let hoverPrioritySentId = null;  // Pending itemId currently boosted on Native Host
let hoverPriorityTimer = null;   // Dwell timer before elevating hovered item to P0
const HOVER_PRIORITY_DWELL_MS = 120;
let scanPaused = false;            // Pause periodic scans while tab is hidden
let settingsLoaded = false;        // Guard against classifying with unloaded default ensembles
let processedImages = new WeakSet(); // WeakSet: elements automatically GC'd when removed from DOM
let activeJobIds = new Set();    // Legacy job ID tracking
let itemResults = new Map();     // itemId -> { modality, ensembles: { ensembleId: { score, label, displayLabel } } }
let pendingItemIds = new Set();  // Item IDs queued for classification (awaiting result)
let lastEnsembleSignature = null;
let lastEnsembleStyleSignature = null;
let ensembleByIdCache = null;
let seenTextContent = new Set();
let activeJobTimeouts = new Map();
let hoverPointerHandler = null;
let hoverScrollHandler = null;
let hoverMouseLeaveHandler = null;
let aiDetectedImageCount = 0;
let lastContextMenuState = null;
const JOB_TRACKING_MS = 5 * 60 * 1000;

/**
 * OBSERVERS - Monitor page for dynamic content & viewport visibility
 */
let mutationObserver = null;     // Watches for new page updates
let intersectionObserver = null; // Watches for images entering viewport (lazy load + P1 viewport priority boost)
let pendingClassification = [];   // Legacy queue
let pendingTextSections = [];    // Legacy queue
let classificationTimeout = null;
let scanIntervalId = null;       // Periodic scan interval
let seenImageUrls = new Set();   // Image URLs already queued or classified
let pendingMutations = [];       // Batched MutationObserver records
let mutationFlushTimer = null;   // Debounce timer for mutation processing
const MUTATION_FLUSH_MS = 120;
const TEXT_STREAM_SETTLE_MS = 650;
const TEXT_GROWTH_RECLASSIFY_DELTA = 40;
const MAX_ITEM_RESULTS = 3000;
const PERIODIC_SCAN_MS = 3500;

let processedTextElements = new WeakSet();
let streamingTextElements = new Map(); // Element -> { timerId, lastMutatedAt }
let elementClassifiedMeta = new WeakMap(); // Element -> { preview, length }
let activeMutationBatchBlocks = new Set(); // Temporarily blocks newly added/mutated trailing nodes during batch flush

function isElementInViewport(el) {
  if (!el || typeof el.getBoundingClientRect !== 'function') return false;
  try {
    const rect = el.getBoundingClientRect();
    const vw = window.innerWidth || document.documentElement.clientWidth || 1280;
    const vh = window.innerHeight || document.documentElement.clientHeight || 800;
    const margin = 200; // Include items just about to scroll into view
    return (
      rect.bottom >= -margin &&
      rect.right >= -margin &&
      rect.top <= vh + margin &&
      rect.left <= vw + margin
    );
  } catch (_) {
    return true;
  }
}

function getEnsembleConfigs() {
  if (!Array.isArray(settings.ensembles) || settings.ensembles.length === 0) {
    return ENS.buildDefaultEnsembles(settings);
  }
  return settings.ensembles;
}

function getEnsemblesByModality() {
  const ensembles = getEnsembleConfigs();
  const byModality = { image: [], text: [] };
  ensembles.forEach((ens) => {
    if (!ens || !ens.modality || ens.enabled === false) return;
    if (ens.modality === 'image') byModality.image.push(ens);
    if (ens.modality === 'text') byModality.text.push(ens);
  });
  return byModality;
}

function hasActiveModality(modality) {
  if (!settings.enabled) return false;
  const byMod = getEnsemblesByModality();
  return Array.isArray(byMod[modality]) && byMod[modality].length > 0;
}

function buildModelConfig() {
  const ensemblesByModality = getEnsemblesByModality();
  return {
    ensemblesByModality: {
      image: (ensemblesByModality.image || []).map((ens) => ({
        id: ens.id,
        classifiers: ens.classifiers || [],
        weights: ens.weights || null
      })),
      text: (ensemblesByModality.text || []).map((ens) => ({
        id: ens.id,
        classifiers: ens.classifiers || [],
        weights: ens.weights || null
      }))
    },
    lazyLoad: settings.lazyLoad,
    streamResults: true
  };
}

function computeEnsembleSignature(ensembles, includeStyles) {
  if (!Array.isArray(ensembles)) return '';
  const normalized = ensembles.map((ens) => {
    const base = {
      id: ens.id || '',
      modality: ens.modality || '',
      enabled: ens.enabled !== false,
      classifiers: Array.isArray(ens.classifiers) ? [...ens.classifiers].sort() : [],
    };
    if (includeStyles) {
      base.weights = Array.isArray(ens.weights) ? [...ens.weights] : null;
      base.name = typeof ens.name === 'string' ? ens.name.trim() : '';
      base.threshold = typeof ens.threshold === 'number' ? ens.threshold : 0.75;
      base.styles = ens.styles || {};
    }
    return base;
  });
  normalized.sort((a, b) => a.id.localeCompare(b.id));
  return JSON.stringify(normalized);
}

function updateEnsembleSignatures() {
  const ensembles = getEnsembleConfigs();
  lastEnsembleSignature = computeEnsembleSignature(ensembles, false);
  lastEnsembleStyleSignature = computeEnsembleSignature(ensembles, true);
  ensembleByIdCache = null;
}

function rebuildEnsembleCache() {
  ensembleByIdCache = new Map();
  getEnsembleConfigs().forEach((ens) => {
    if (ens && ens.id) ensembleByIdCache.set(ens.id, ens);
  });
}

function getEnsembleById(ensembleId) {
  if (!ensembleByIdCache) rebuildEnsembleCache();
  return ensembleByIdCache.get(ensembleId);
}

function extractCanonicalResponsiveImageKey(urlStr) {
  if (!urlStr || typeof urlStr !== 'string') return '';
  try {
    // BBC iChef responsive URLs: https://ichef.bbci.co.uk/ace/standard/<width>/cpsprodpb/...
    const bbcMatch = urlStr.match(/^(https?:\/\/ichef\.bbci\.co\.uk\/(?:ace\/(?:standard|ws)|news)\/)\d+(\/.*)$/i);
    if (bbcMatch) {
      return (bbcMatch[1] + '*' + bbcMatch[2]).replace(/\.webp($|\?)/i, '$1');
    }
  } catch (_) {}
  return '';
}

function isSameResponsiveImageAsset(urlA, urlB, el = null) {
  if (!urlA || !urlB) return false;
  if (urlA === urlB) return true;
  const keyA = extractCanonicalResponsiveImageKey(urlA);
  const keyB = extractCanonicalResponsiveImageKey(urlB);
  if (keyA && keyB && keyA === keyB) return true;
  if (el && el.tagName && el.tagName.toLowerCase() === 'img') {
    const rawSrcset = (el.getAttribute && (el.getAttribute('srcset') || el.getAttribute('data-srcset'))) || '';
    if (rawSrcset && rawSrcset.includes(urlA) && rawSrcset.includes(urlB)) {
      return true;
    }
  }
  return false;
}

function trackElement(itemId, element) {
  if (!itemId || !element) return;
  const existing = elementMap.get(itemId);
  if (!existing) {
    elementMap.set(itemId, new Set([element]));
  } else if (existing instanceof Set) {
    existing.add(element);
  } else {
    elementMap.set(itemId, new Set([existing, element]));
  }
  elementToItemId.set(element, itemId);
}

function getElementsForItem(itemId) {
  const entry = elementMap.get(itemId);
  if (!entry) return [];
  if (entry instanceof Set) {
    const alive = [];
    for (const el of entry) {
      if (el && el.isConnected) alive.push(el);
      else entry.delete(el);
    }
    if (entry.size === 0) elementMap.delete(itemId);
    return alive;
  }
  if (entry && entry.isConnected) return [entry];
  elementMap.delete(itemId);
  return [];
}

function hasResultsForElement(el) {
  if (!el) return false;
  const itemId = elementToItemId.get(el);
  return itemId != null && itemResults.has(itemId);
}

function unobserveImage(el) {
  if (!el || el.tagName?.toLowerCase() !== 'img') return;
  if (intersectionObserver) {
    try { intersectionObserver.unobserve(el); } catch (_) { /* already unobserved */ }
  }
}

function observeImage(el) {
  if (!el || el.tagName?.toLowerCase() !== 'img') return;
  if (intersectionObserver) intersectionObserver.observe(el);
}

function clearStreamingTextElement(el) {
  if (!el) return;
  const entry = streamingTextElements.get(el);
  if (entry && entry.timerId) {
    clearTimeout(entry.timerId);
  }
  streamingTextElements.delete(el);
}

function clearAllStreamingTextElements() {
  streamingTextElements.forEach((entry) => {
    if (entry && entry.timerId) clearTimeout(entry.timerId);
  });
  streamingTextElements.clear();
}

function isEditableTextElement(el) {
  if (!el || !(el instanceof Element)) return true;
  if (el.isContentEditable) return true;
  if (typeof el.closest === 'function' && el.closest('[contenteditable="true"], textarea, input')) {
    return true;
  }
  return false;
}

function findEnclosingTextBlock(node) {
  if (!node) return null;
  const el = node.nodeType === 3 ? node.parentElement : node;
  if (!el || typeof el.closest !== 'function') return null;
  const block = el.closest(TU.TEXT_SELECTORS);
  if (!block || !block.isConnected) return null;
  if (isEditableTextElement(block)) return null;
  if (TU.hasNestedTextCandidate(block)) return null;
  return block;
}

function releaseElementTracking(el) {
  if (!el || !(el instanceof Element)) return;

  clearStreamingTextElement(el);

  const itemId = elementToItemId.get(el);
  if (itemId) {
    const tracked = elementMap.get(itemId);
    if (tracked instanceof Set) {
      tracked.delete(el);
      if (tracked.size === 0) elementMap.delete(itemId);
    } else if (tracked === el) {
      elementMap.delete(itemId);
    }
    elementToItemId.delete(el);
  }

  if (el.tagName?.toLowerCase() === 'img') {
    if (el.classList.contains('ai-detected')) {
      aiDetectedImageCount = Math.max(0, aiDetectedImageCount - 1);
      VE.clearImageStyles(el);
      updateHoverHandlerAttachment();
    }
    if (el.dataset) {
      delete el.dataset.aiLoadListenerAttached;
      delete el.dataset.aiLoadListenerSrc;
    }
    unobserveImage(el);
    processedImages.delete(el);
    if (currentHoverEl === el) currentHoverEl = null;
  } else {
    processedTextElements.delete(el);
    elementClassifiedMeta.delete(el);
  }
}

function cleanupRemovedSubtree(node) {
  if (!node || node.nodeType !== 1) return;

  releaseElementTracking(node);

  if (typeof node.querySelector !== 'function') return;
  if (!node.querySelector('img') && !node.querySelector(TU.TEXT_SELECTORS)) return;

  try {
    node.querySelectorAll('img').forEach((img) => releaseElementTracking(img));
    node.querySelectorAll(TU.TEXT_SELECTORS).forEach((el) => releaseElementTracking(el));
  } catch (_) {}
}

function pruneItemResults() {
  if (itemResults.size <= MAX_ITEM_RESULTS) return;

  for (const itemId of itemResults.keys()) {
    if (!elementMap.has(itemId)) {
      itemResults.delete(itemId);
      if (itemId.startsWith('http')) seenImageUrls.delete(itemId);
      if (itemResults.size <= MAX_ITEM_RESULTS) return;
    }
  }

  while (itemResults.size > MAX_ITEM_RESULTS) {
    const oldest = itemResults.keys().next().value;
    itemResults.delete(oldest);
    if (typeof oldest === 'string' && oldest.startsWith('http')) {
      seenImageUrls.delete(oldest);
    }
  }
}

function cleanupStaleEntries() {
  for (const [itemId, entry] of elementMap) {
    if (entry instanceof Set) {
      for (const el of entry) {
        if (!el?.isConnected) entry.delete(el);
      }
      if (entry.size === 0) elementMap.delete(itemId);
    } else if (!entry?.isConnected) {
      elementMap.delete(itemId);
    }
  }

  if (currentHoverEl && !currentHoverEl.isConnected) {
    currentHoverEl = null;
  }

  for (const el of Array.from(streamingTextElements.keys())) {
    if (!el || !el.isConnected) {
      clearStreamingTextElement(el);
    }
  }

  pruneItemResults();
}

function sendHoverPriorityMessage(itemId) {
  if (itemId && settings.verboseLogs) {
    console.log(`[AI Detector] Hover priority (P0):`, String(itemId).substring(0, 80));
  }
  try {
    ext.runtime.sendMessage({
      type: 'HOVER_ITEM',
      pageId: currentPageId,
      itemId: itemId || null
    }).catch(() => {});
  } catch (_) {}
}

function clearHoverPriority() {
  if (hoverPriorityTimer) {
    clearTimeout(hoverPriorityTimer);
    hoverPriorityTimer = null;
  }
  hoverPriorityItemId = null;
  if (hoverPrioritySentId !== null) {
    hoverPrioritySentId = null;
    sendHoverPriorityMessage(null);
  }
}

function updateHoverPriority(el) {
  const rawId = el ? elementToItemId.get(el) : null;
  const id = (rawId && pendingItemIds.has(rawId)) ? rawId : null;
  if (id === hoverPriorityItemId) return;

  hoverPriorityItemId = id;
  if (hoverPriorityTimer) {
    clearTimeout(hoverPriorityTimer);
    hoverPriorityTimer = null;
  }

  if (id) {
    hoverPriorityTimer = setTimeout(() => {
      hoverPriorityTimer = null;
      if (hoverPriorityItemId === id && id !== hoverPrioritySentId && pendingItemIds.has(id)) {
        hoverPrioritySentId = id;
        sendHoverPriorityMessage(id);
      }
    }, HOVER_PRIORITY_DWELL_MS);
  } else if (hoverPrioritySentId !== null) {
    hoverPrioritySentId = null;
    sendHoverPriorityMessage(null);
  }
}

function clearVisualHover() {
  if (!currentHoverEl) return;
  currentHoverEl.classList.remove('fvf-hover-active');
  if (hasResultsForElement(currentHoverEl)) {
    if (currentHoverEl.dataset.aiImageStyle) {
      VE.applyImageInteractionState(currentHoverEl, 'rest', settings);
    } else {
      const blurAmount = parseFloat(currentHoverEl.dataset.aiBlur || settings.blurAmount || 0);
      VE.hideImageAi(currentHoverEl, blurAmount);
    }
  }
  currentHoverEl = null;
}

function hitTestPointer(x, y, target) {
  const needVisual = aiDetectedImageCount > 0;
  const needPriority = pendingItemIds.size > 0;
  let visualEl = null;
  let priorityEl = null;

  if ((!needVisual && !needPriority) || typeof x !== 'number' || typeof y !== 'number') {
    return { visualEl: null, priorityEl: null };
  }

  // 1. Check direct target and ancestors (instant, zero hit-test cost)
  if (target instanceof Element) {
    if (needVisual && typeof target.closest === 'function') {
      visualEl = target.closest('.ai-detected');
    }
    if (needPriority) {
      let cur = target;
      while (cur && cur !== document.body && cur !== document.documentElement) {
        const itemId = elementToItemId.get(cur);
        if (itemId && pendingItemIds.has(itemId)) {
          priorityEl = cur;
          break;
        }
        cur = cur.parentElement;
      }
    }
    if ((!needVisual || visualEl) && (!needPriority || priorityEl)) {
      return { visualEl, priorityEl };
    }
  }

  // 2. Penetrate transparent overlay divs/elements using a single elementsFromPoint(x, y) call
  if (typeof document.elementsFromPoint === 'function') {
    try {
      const elements = document.elementsFromPoint(x, y);
      if (elements && elements.length) {
        for (let i = 0; i < elements.length; i++) {
          const el = elements[i];
          if (!el) continue;
          if (needVisual && !visualEl) {
            if (el.classList && el.classList.contains('ai-detected')) {
              visualEl = el;
            } else if (typeof el.closest === 'function') {
              const parent = el.closest('.ai-detected');
              if (parent) visualEl = parent;
            }
          }
          if (needPriority && !priorityEl) {
            const itemId = elementToItemId.get(el);
            if (itemId && pendingItemIds.has(itemId)) {
              priorityEl = el;
            }
          }
          if ((!needVisual || visualEl) && (!needPriority || priorityEl)) {
            break;
          }
        }
      }
    } catch (_) {}
  }

  return { visualEl, priorityEl };
}

function handlePointerMoveHover(x, y, target) {
  const { visualEl, priorityEl } = hitTestPointer(x, y, target);

  if (visualEl !== currentHoverEl) {
    clearVisualHover();
    currentHoverEl = visualEl;
    if (currentHoverEl && hasResultsForElement(currentHoverEl)) {
      currentHoverEl.classList.add('fvf-hover-active');
      if (currentHoverEl.dataset.aiImageStyle) {
        VE.applyImageInteractionState(currentHoverEl, 'hover', settings);
      } else {
        VE.showImageAi(currentHoverEl);
      }
    }
  }

  updateHoverPriority(priorityEl);
}

function scheduleHoverEvaluation() {
  if (hoverRafPending) return;
  hoverRafPending = true;
  requestAnimationFrame(() => {
    hoverRafPending = false;
    if (!hoverHandlerAttached || !lastPointerState) return;
    const { clientX, clientY, target } = lastPointerState;
    const fromScroll = hoverDirty;
    hoverDirty = false;

    if (!fromScroll && lastHitTestCoords) {
      const dx = clientX - lastHitTestCoords.clientX;
      const dy = clientY - lastHitTestCoords.clientY;
      if ((dx * dx + dy * dy) < 9 && target === lastHitTestCoords.target) {
        return;
      }
    }

    const effectiveTarget = fromScroll ? null : target;
    lastHitTestCoords = { clientX, clientY, target: effectiveTarget };
    handlePointerMoveHover(clientX, clientY, effectiveTarget);
  });
}

function detachGlobalHoverHandler() {
  if (!hoverHandlerAttached) return;
  if (hoverPointerHandler) {
    document.removeEventListener('pointermove', hoverPointerHandler, { capture: true });
  }
  if (hoverScrollHandler) {
    window.removeEventListener('scroll', hoverScrollHandler, { capture: true });
  }
  if (hoverMouseLeaveHandler && document.documentElement) {
    document.documentElement.removeEventListener('mouseleave', hoverMouseLeaveHandler);
  }
  hoverHandlerAttached = false;
  hoverPointerHandler = null;
  hoverScrollHandler = null;
  hoverMouseLeaveHandler = null;
  hoverRafPending = false;
  hoverDirty = false;
  lastPointerState = null;
  lastHitTestCoords = null;
  clearVisualHover();
  clearHoverPriority();
}

function updateHoverHandlerAttachment() {
  const shouldAttach = Boolean(
    settings.enabled &&
    !document.hidden &&
    (aiDetectedImageCount > 0 || pendingItemIds.size > 0)
  );
  if (shouldAttach && !hoverHandlerAttached) {
    attachGlobalHoverHandler();
  } else if (!shouldAttach && hoverHandlerAttached) {
    detachGlobalHoverHandler();
  }
}

function attachGlobalHoverHandler() {
  if (hoverHandlerAttached || document.hidden || !settings.enabled) return;
  hoverHandlerAttached = true;
  hoverPointerHandler = (evt) => {
    lastPointerState = {
      clientX: evt.clientX,
      clientY: evt.clientY,
      target: evt.target instanceof Element ? evt.target : null
    };
    scheduleHoverEvaluation();
  };
  hoverScrollHandler = () => {
    if (!lastPointerState) return;
    hoverDirty = true;
    scheduleHoverEvaluation();
  };
  hoverMouseLeaveHandler = () => {
    lastPointerState = null;
    lastHitTestCoords = null;
    clearVisualHover();
    clearHoverPriority();
  };
  document.addEventListener('pointermove', hoverPointerHandler, { capture: true, passive: true });
  window.addEventListener('scroll', hoverScrollHandler, { capture: true, passive: true });
  if (document.documentElement) {
    document.documentElement.addEventListener('mouseleave', hoverMouseLeaveHandler);
  }
}

// ============================================================================
// Single-Item Real-Time Streaming Dispatch
// ============================================================================

function getCaptureQuality() {
  return typeof settings.imageCaptureQuality === 'number'
    ? Math.max(0.2, Math.min(1.0, settings.imageCaptureQuality))
    : 1.0;
}

// Extract image as compact base64 from DOM; returns null if tainted by cross-origin CORS
function imageToBase64(imgElement) {
  try {
    const srcForDims = imgElement.currentSrc || imgElement.src || imgElement.getAttribute('src') || '';
    const dims = getEffectiveImageDimensions(imgElement, srcForDims);
    let w = dims.width || imgElement.width || 0;
    let h = dims.height || imgElement.height || 0;
    if (w < 48 || h < 48) return null;

    const MAX_DIM = 768;
    if (w > MAX_DIM || h > MAX_DIM) {
      if (w >= h) {
        h = Math.round((h * MAX_DIM) / w);
        w = MAX_DIM;
      } else {
        w = Math.round((w * MAX_DIM) / h);
        h = MAX_DIM;
      }
    }

    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, w);
    canvas.height = Math.max(1, h);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0, w, h);

    const quality = getCaptureQuality();
    const dataUrl = canvas.toDataURL('image/jpeg', quality);
    canvas.width = 0;
    canvas.height = 0;
    return dataUrl;
  } catch (_) {
    // Tainted cross-origin canvas: background.js will capture & downscale via extension `<all_urls>` privileges
    return null;
  }
}

function dispatchSingleImageItem(imgObj, modelConfig) {
  if (!imgObj || !imgObj.element) return;
  const itemId = imgObj.src || ('img-unknown-' + Math.random());
  trackElement(itemId, imgObj.element);
  pendingItemIds.add(itemId);
  updateHoverHandlerAttachment();

  const inViewport = Boolean(imgObj.inViewport ?? isElementInViewport(imgObj.element));

  // If offscreen, keep observing so IntersectionObserver can boost its priority to P1 if user scrolls to it!
  if (!inViewport) {
    observeImage(imgObj.element);
  }

  const localBase64 = imageToBase64(imgObj.element);
  const upgradedSrc = imgObj.investigatorSrc || null;
  const quality = getCaptureQuality();

  if (settings.verboseLogs) {
    const modeLabel = upgradedSrc
      ? 'investigator-hi-res'
      : (localBase64 ? 'canvas-b64' : 'bg-cors-fetch');
    console.log(
      `[AI Detector] Streaming image (${inViewport ? 'P1 viewport' : 'P2 offscreen'}, ${modeLabel}):`,
      itemId.substring(0, 80)
    );
  }

  const messagePayload = {
    type: 'CLASSIFY_SINGLE_ITEM',
    pageId: currentPageId,
    itemId,
    modality: 'image',
    inViewport,
    data: upgradedSrc ? null : localBase64,
    src: upgradedSrc || imgObj.src,
    captureQuality: quality,
    width: imgObj.width || null,
    height: imgObj.height || null,
    model: modelConfig
  };
  if (upgradedSrc) {
    messagePayload.isInvestigatorUpgrade = true;
    messagePayload.fallbackSrc = imgObj.src;
    if (localBase64) {
      messagePayload.fallbackData = localBase64;
    }
  }

  ext.runtime.sendMessage(messagePayload).then((resp) => {
    if (resp && resp.ok === false && resp.error !== 'paused_by_power_policy') {
      pendingItemIds.delete(itemId);
      seenImageUrls.delete(imgObj.src);
      processedImages.delete(imgObj.element);
      if (imgObj.element && imgObj.element.dataset) {
        delete imgObj.element.dataset.aiLoadListenerAttached;
        delete imgObj.element.dataset.aiLoadListenerSrc;
      }
    }
  }).catch((err) => {
    pendingItemIds.delete(itemId);
    seenImageUrls.delete(imgObj.src);
    processedImages.delete(imgObj.element);
    if (settings.verboseLogs) {
      console.warn('[AI Detector] Failed to send single image item:', err);
    }
  });
}

function dispatchSingleTextItem(section, modelConfig) {
  if (!section || !section.text) return;
  const itemId = section.text.substring(0, 500);
  const elements = Array.isArray(section.elements) && section.elements.length
    ? section.elements
    : [section.element];
  elements.forEach((el) => {
    if (el) trackElement(itemId, el);
  });
  pendingItemIds.add(itemId);
  updateHoverHandlerAttachment();

  const inViewport = elements.some((el) => isElementInViewport(el));

  ext.runtime.sendMessage({
    type: 'CLASSIFY_SINGLE_ITEM',
    pageId: currentPageId,
    itemId,
    modality: 'text',
    inViewport,
    data: section.text,
    text: section.text,
    length: section.text.length,
    context: section.selector || section.tag,
    model: modelConfig
  }).catch((err) => {
    if (settings.verboseLogs) {
      console.warn('[AI Detector] Failed to send single text item:', err);
    }
  });
}

function scheduleClassification(images) {
  /**
   * Streams discovered images individually to Native Host,
   * sending in-viewport (P1) images first, yielding to the browser event loop
   * so main thread rendering is never blocked.
   */
  if (isBellowsCollectorActive() || !settings.enabled || !images || !images.length) return;
  if (!hasActiveModality('image')) return;
  const modelConfig = buildModelConfig();

  // Annotate viewport status and sort so in-viewport items are dispatched first
  images.forEach((img) => {
    img.inViewport = isElementInViewport(img.element);
  });
  const sorted = images.slice().sort((a, b) => (b.inViewport ? 1 : 0) - (a.inViewport ? 1 : 0));

  const dispatchBatch = (items, index = 0) => {
    const BATCH_SIZE = 4;
    const end = Math.min(items.length, index + BATCH_SIZE);
    for (let i = index; i < end; i++) {
      dispatchSingleImageItem(items[i], modelConfig);
    }
    if (end < items.length) {
      setTimeout(() => dispatchBatch(items, end), 16);
    }
  };

  dispatchBatch(sorted);
}

function scheduleTextClassification(textSections) {
  /**
   * Immediately streams discovered text sections individually to the Native Host.
   */
  if (isBellowsCollectorActive() || !settings.enabled || !textSections || !textSections.length) return;
  if (!hasActiveModality('text')) return;
  const modelConfig = buildModelConfig();

  textSections.forEach((section) => {
    dispatchSingleTextItem(section, modelConfig);
  });
}

function isTextElementBlocked(el) {
  return !el || processedTextElements.has(el) || streamingTextElements.has(el) || activeMutationBatchBlocks.has(el) || isEditableTextElement(el);
}

function hasTrackedTextResult(preview) {
  return Boolean(preview && (itemResults.has(preview) || pendingItemIds.has(preview)));
}

function extractSectionsFromScope(scope, outList = []) {
  if (!scope || typeof TU.extractCohesiveTextSections !== 'function') return outList;
  const candidates = TU.extractCohesiveTextSections(
    scope,
    getMinTextLength(),
    isTextElementBlocked,
    null,
    hasTrackedTextResult
  );
  candidates.forEach((sec) => {
    const accepted = acceptCohesiveTextSection(sec);
    if (accepted) outList.push(accepted);
  });
  return outList;
}

function buildSectionForCompletedElement(el) {
  if (!el || !el.isConnected || isTextElementBlocked(el)) return null;
  if (TU.hasNestedTextCandidate(el)) return null;

  const minLen = getMinTextLength();
  const tag = el.tagName ? el.tagName.toLowerCase() : '';
  if (tag === 'p' && !TU.isIsolatedFeedElement(el)) {
    // Walk back to the start of any contiguous completed <p> run (capped at 4 paragraphs)
    let runStart = el;
    let backSteps = 0;
    while (backSteps < 3 && runStart.previousElementSibling) {
      const prev = runStart.previousElementSibling;
      if (!prev.tagName || prev.tagName.toLowerCase() !== 'p') break;
      if (isTextElementBlocked(prev) || TU.hasNestedTextCandidate(prev) || TU.isIsolatedFeedElement(prev)) break;
      if (!TU.normalizeElementText(prev)) break;
      runStart = prev;
      backSteps++;
    }

    const group = [];
    const parts = [];
    let accLen = 0;
    let cur = runStart;
    while (cur && group.length < 4 && accLen < 1800) {
      if (!cur.tagName || cur.tagName.toLowerCase() !== 'p') break;
      if (isTextElementBlocked(cur) || TU.hasNestedTextCandidate(cur) || TU.isIsolatedFeedElement(cur)) break;
      const curText = TU.normalizeElementText(cur);
      if (!curText) break;
      if (cur !== runStart && hasTrackedTextResult(curText.substring(0, 500))) break;
      if (accLen + (parts.length ? 2 : 0) + curText.length > 1800 && accLen >= minLen && cur !== el && group.includes(el)) {
        break;
      }
      group.push(cur);
      parts.push(curText);
      accLen += (parts.length > 1 ? 2 : 0) + curText.length;
      if (cur === runStart && hasTrackedTextResult(curText.substring(0, 500))) break;
      cur = cur.nextElementSibling;
    }

    const joined = parts.join('\n\n');
    if (!joined || joined.length < minLen) return null;
    return TU.buildTextSection(group[0], 0, joined, group);
  }

  const text = TU.normalizeElementText(el);
  if (!text || text.length < minLen) return null;
  return TU.buildTextSection(el, 0, text, [el]);
}

function flushStreamingTextElement(el) {
  if (!streamingTextElements.has(el)) return;
  clearStreamingTextElement(el);
  if (!el || !el.isConnected || processedTextElements.has(el) || isEditableTextElement(el)) return;
  if (isBellowsCollectorActive() || !settings.enabled || scanPaused || !hasActiveModality('text')) return;

  const sec = buildSectionForCompletedElement(el);
  const accepted = sec ? acceptCohesiveTextSection(sec) : null;
  if (accepted) {
    scheduleTextClassification([accepted]);
  }
}

function flushPrecedingStreamingSiblings(node, activeInBatchSet) {
  if (!node || streamingTextElements.size === 0) return;

  const toFlush = [];
  for (const streamEl of streamingTextElements.keys()) {
    if (!streamEl || streamEl === node) continue;
    if (activeInBatchSet && activeInBatchSet.has(streamEl)) continue;
    if (!streamEl.isConnected) {
      toFlush.push(streamEl);
      continue;
    }
    if (typeof node.contains === 'function' && (node.contains(streamEl) || streamEl.contains(node))) {
      continue;
    }
    if (typeof streamEl.compareDocumentPosition === 'function') {
      const pos = streamEl.compareDocumentPosition(node);
      if ((pos & Node.DOCUMENT_POSITION_FOLLOWING) !== 0) {
        toFlush.push(streamEl);
      }
    }
  }

  toFlush.forEach((streamEl) => flushStreamingTextElement(streamEl));
}

function scheduleStreamingTextSettle(el) {
  if (!el || !el.isConnected || processedTextElements.has(el) || isEditableTextElement(el)) return;
  if (TU.hasNestedTextCandidate(el)) return;

  const existing = streamingTextElements.get(el);
  if (existing && existing.timerId) {
    clearTimeout(existing.timerId);
  }
  const timerId = setTimeout(() => {
    flushStreamingTextElement(el);
  }, TEXT_STREAM_SETTLE_MS);
  streamingTextElements.set(el, { timerId, lastMutatedAt: Date.now() });
}

function checkAlreadyProcessedTextMutation(block) {
  if (!block || !block.isConnected || !processedTextElements.has(block)) return;
  if (isEditableTextElement(block) || TU.hasNestedTextCandidate(block)) return;

  const newText = TU.normalizeElementText(block);
  const prevMeta = elementClassifiedMeta.get(block);
  if (prevMeta) {
    const newPreview = newText.substring(0, 200);
    const samePreview = newPreview === prevMeta.preview;
    const lenDelta = Math.abs(newText.length - prevMeta.length);
    if (samePreview && lenDelta < TEXT_GROWTH_RECLASSIFY_DELTA) {
      return;
    }
  }

  releaseElementTracking(block);
  VE.clearTextStyles(block);
  if (newText.length > 0) {
    scheduleStreamingTextSettle(block);
  }
}

function processMutationBatch(mutations) {
  if (!settingsLoaded) return;
  if (scanPaused && !document.hidden) scanPaused = false;
  if (isBellowsCollectorActive() || !settings.enabled || scanPaused) return;

  const hasImages = hasActiveModality('image');
  const hasText = hasActiveModality('text');
  if (!hasImages && !hasText) return;

  const newImages = [];
  const newText = [];
  const inPlaceMutatedBlocks = new Set();
  const newlyAddedNodes = [];

  mutations.forEach((mutation) => {
    if (mutation.target && typeof mutation.target.closest === 'function' && mutation.target.closest('#fvf-community-report-backdrop')) {
      return;
    }

    mutation.removedNodes.forEach((node) => cleanupRemovedSubtree(node));

    if (hasImages && mutation.type === 'attributes' && (mutation.attributeName === 'src' || mutation.attributeName === 'srcset')) {
      const el = mutation.target;
      if (el.tagName && el.tagName.toLowerCase() === 'img') {
        const newSrc = el.currentSrc || el.src || el.getAttribute('src') || '';
        const oldItemId = elementToItemId.get(el);
        if (processedImages.has(el)) {
          if (
            (oldItemId && isSameResponsiveImageAsset(oldItemId, newSrc, el)) ||
            (!oldItemId && newSrc && seenImageUrls.has(newSrc))
          ) {
            if (oldItemId && newSrc && oldItemId !== newSrc) {
              trackElement(newSrc, el);
              trackElement(oldItemId, el);
              if (itemResults.has(oldItemId) && !itemResults.has(newSrc)) {
                itemResults.set(newSrc, itemResults.get(oldItemId));
                applyStylesForItem(oldItemId);
              }
            }
            return;
          }
        }
        releaseElementTracking(el);
        const img = processImageElement(el);
        if (img) {
          newImages.push(img);
          if (settings.verboseLogs) {
            console.log(`[AI Detector] Image src changed, re-processing:`, (newSrc || '').substring(0, 80));
          }
        }
      }
      return;
    }

    if (hasText && mutation.type === 'characterData') {
      const block = findEnclosingTextBlock(mutation.target);
      if (block) inPlaceMutatedBlocks.add(block);
      return;
    }

    mutation.addedNodes.forEach((node) => {
      if (node.nodeType === 1 && (node.id === 'fvf-community-report-backdrop' || (typeof node.closest === 'function' && node.closest('#fvf-community-report-backdrop')))) {
        return;
      }

      if (node.nodeType === 3) {
        if (hasText) {
          const block = findEnclosingTextBlock(node);
          if (block) inPlaceMutatedBlocks.add(block);
        }
        return;
      }

      if (node.nodeType !== 1) return;

      if (hasImages && node.tagName && node.tagName.toLowerCase() === 'img') {
        const img = processImageElement(node);
        if (img) newImages.push(img);
        return;
      }

      if (!node.querySelectorAll) return;

      try {
        if (hasImages && node.querySelector('img')) {
          node.querySelectorAll('img').forEach((el) => {
            const img = processImageElement(el);
            if (img) newImages.push(img);
          });
        }

        if (hasText) {
          const isSelfText = Boolean(node.tagName && TU.isTextTagName(node.tagName.toLowerCase()));
          const hasDescendantText = Boolean(node.querySelector(TU.TEXT_SELECTORS));
          if (!isSelfText && !hasDescendantText) {
            const enclosingBlock = findEnclosingTextBlock(node);
            if (enclosingBlock) {
              inPlaceMutatedBlocks.add(enclosingBlock);
            }
          } else {
            newlyAddedNodes.push({ node, isSelfText });
          }
        }
      } catch (_) {}
    });
  });

  activeMutationBatchBlocks = new Set(inPlaceMutatedBlocks);
  newlyAddedNodes.forEach(({ node, isSelfText }) => {
    if (isSelfText && node && !node.nextElementSibling && !TU.hasNestedTextCandidate(node)) {
      activeMutationBatchBlocks.add(node);
    }
  });

  try {
    inPlaceMutatedBlocks.forEach((block) => {
      flushPrecedingStreamingSiblings(block, activeMutationBatchBlocks);
      if (processedTextElements.has(block)) {
        checkAlreadyProcessedTextMutation(block);
      } else {
        scheduleStreamingTextSettle(block);
      }
    });

    newlyAddedNodes.forEach(({ node, isSelfText }) => {
      if (!node || !node.isConnected) return;
      flushPrecedingStreamingSiblings(node, activeMutationBatchBlocks);

      if (isSelfText && !node.nextElementSibling && !TU.hasNestedTextCandidate(node)) {
        if (isEditableTextElement(node) || processedTextElements.has(node)) return;
        const text = TU.normalizeElementText(node);
        const preview = text.substring(0, 500);
        if (text && (itemResults.has(preview) || pendingItemIds.has(preview))) {
          const sec = TU.buildTextSection(node, 0, text, [node]);
          activeMutationBatchBlocks.delete(node);
          acceptCohesiveTextSection(sec);
        } else {
          scheduleStreamingTextSettle(node);
        }
        return;
      }

      if (isSelfText) {
        activeMutationBatchBlocks.delete(node);
        const sec = buildSectionForCompletedElement(node);
        const accepted = sec ? acceptCohesiveTextSection(sec) : null;
        if (accepted) {
          newText.push(accepted);
        }
        return;
      }

      extractSectionsFromScope(node, newText);
    });
  } finally {
    activeMutationBatchBlocks.clear();
  }

  if (newImages.length > 0) {
    if (settings.verboseLogs) {
      console.log(`[AI Detector] MutationObserver detected ${newImages.length} new images`);
    }
    scheduleClassification(newImages);
  }
  if (newText.length > 0) {
    scheduleTextClassification(newText);
  }
}

function scheduleMutationFlush() {
  if (mutationFlushTimer) return;
  mutationFlushTimer = setTimeout(() => {
    mutationFlushTimer = null;
    if (!pendingMutations.length) return;
    const batch = pendingMutations;
    pendingMutations = [];
    processMutationBatch(batch);
  }, MUTATION_FLUSH_MS);
}

function initMutationObserver() {
  if (mutationObserver) return;

  mutationObserver = new MutationObserver((mutations) => {
    pendingMutations.push(...mutations);
    scheduleMutationFlush();
  });

  mutationObserver.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['src', 'srcset'],
    characterData: true
  });
}

// Detect lazy-loaded images entering viewport AND boost already-queued offscreen images to P1
function initIntersectionObserver() {
  if (intersectionObserver) return;

  intersectionObserver = new IntersectionObserver((entries) => {
    if (isBellowsCollectorActive() || !settingsLoaded) return;
    const newImages = [];
    const boostItemIds = [];

    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      if (el.tagName && el.tagName.toLowerCase() === 'img') {
        // Case 1: Image was already queued offscreen (P2) and just scrolled into view -> boost to P1!
        if (processedImages.has(el)) {
          const trackedId = elementToItemId.get(el) || el.src;
          if (trackedId && !itemResults.has(trackedId)) {
            boostItemIds.push(trackedId);
          }
          unobserveImage(el);
          return;
        }

        // Case 2: Lazy-loaded image now has dimensions in viewport -> process & dispatch as P1
        if (el.naturalWidth > 0 && el.naturalHeight > 0) {
          const img = processImageElement(el);
          if (img) {
            img.inViewport = true;
            newImages.push(img);
            if (settings.verboseLogs) {
              console.log(`[AI Detector] IntersectionObserver: Lazy image entered viewport:`, el.src.substring(0, 80));
            }
          }
        } else {
          const img = processImageElement(el, true);
          if (img) {
            img.inViewport = true;
            newImages.push(img);
          }
        }
      }
    });

    if (boostItemIds.length > 0) {
      ext.runtime.sendMessage({
        type: 'BOOST_ITEMS',
        pageId: currentPageId,
        itemIds: boostItemIds
      }).catch(() => {});
    }

    if (newImages.length > 0) {
      scheduleClassification(newImages);
    }
  }, {
    rootMargin: '200px'
  });
}

function getMinTextLength() {
  const value = Number(settings.minTextLength);
  return Number.isFinite(value) && value >= 50 ? value : 250;
}

function getEligibleTextForElement(el) {
  if (!el || !(el instanceof Element)) return null;
  if (TU.hasNestedTextCandidate(el)) return null;

  const text = TU.normalizeElementText(el);
  if (!text || text.length < getMinTextLength()) return null;
  return text;
}

function getMinImageDimension() {
  const value = Number(settings.minImageDimension);
  return Number.isFinite(value) && value >= 16 ? value : 80;
}

function parseExplicitCssPixels(raw) {
  if (!raw || typeof raw !== 'string') return 0;
  const trimmed = raw.trim();
  if (!trimmed.endsWith('px')) return 0;
  const num = parseFloat(trimmed);
  return Number.isFinite(num) && num > 0 ? num : 0;
}

function getRenderedImageDimensions(el) {
  const rect = typeof el.getBoundingClientRect === 'function' ? el.getBoundingClientRect() : null;
  let width = rect && rect.width > 0 ? rect.width : (el.clientWidth || 0);
  let height = rect && rect.height > 0 ? rect.height : (el.clientHeight || 0);

  if (width <= 0 && el.getAttribute) {
    const attrW = parseFloat(el.getAttribute('width'));
    if (Number.isFinite(attrW) && attrW > 0) width = attrW;
  }
  if (height <= 0 && el.getAttribute) {
    const attrH = parseFloat(el.getAttribute('height'));
    if (Number.isFinite(attrH) && attrH > 0) height = attrH;
  }

  if ((width <= 0 || height <= 0) && typeof window !== 'undefined' && typeof window.getComputedStyle === 'function') {
    try {
      const style = window.getComputedStyle(el);
      if (style) {
        if (width <= 0) {
          width = parseExplicitCssPixels(style.width) || parseExplicitCssPixels(style.maxWidth);
        }
        if (height <= 0) {
          height = parseExplicitCssPixels(style.height) || parseExplicitCssPixels(style.maxHeight);
        }
      }
    } catch (_) {}
  }

  return { width, height };
}

function isYoutubeVideoThumbnailElement(el) {
  if (!el || typeof el.closest !== 'function') return false;
  if (el.closest('#avatar, #avatar-link, #author-thumbnail, yt-avatar-shape, .yt-spec-avatar-shape')) {
    return false;
  }
  return Boolean(
    el.closest(
      'ytd-thumbnail, yt-thumbnail-view-model, ytd-compact-video-renderer, yt-lockup-view-model, ytd-rich-item-renderer, ytd-video-renderer, ytd-grid-video-renderer, ytd-reel-item-renderer, ytd-playlist-panel-video-renderer'
    )
  );
}

function isPhysicallySmallImage(el, nw, nh) {
  if (!settings.ignoreSmallImages) return false;
  const minDim = getMinImageDimension();
  const shortSideFloor = Math.min(54, Math.round(minDim * 0.6));

  // Rectangular/widescreen thumbnails (e.g. 16:9 YouTube sidebar 168x94) are valid if longer side >= minDim and shorter side >= floor
  if (Math.max(nw, nh) < minDim || Math.min(nw, nh) < shortSideFloor) {
    return true;
  }

  // Known YouTube video thumbnail containers (excluding channel avatars)
  if (isYoutubeVideoThumbnailElement(el) && nw >= 100 && nh >= 54) {
    return false;
  }

  const rendered = getRenderedImageDimensions(el);
  if (rendered.width > 0 && rendered.height > 0) {
    if (Math.max(rendered.width, rendered.height) < minDim) {
      return true;
    }
    if (Math.min(rendered.width, rendered.height) < shortSideFloor) {
      return true;
    }
  }

  return false;
}

function processTextElement(el) {
  if (isBellowsCollectorActive()) return null;
  if (!el || !(el instanceof Element)) return null;
  if (processedTextElements.has(el)) return null;

  const text = getEligibleTextForElement(el);
  if (!text) return null;

  if (seenTextContent.has(text)) return null;
  seenTextContent.add(text);
  processedTextElements.add(el);

  const textPreview = text.substring(0, 500);
  if (itemResults.has(textPreview)) {
    trackElement(textPreview, el);
    applyStylesForItem(textPreview);
    return null;
  }

  return TU.buildTextSection(el, seenTextContent.size, text);
}

const INVESTIGATOR_MAX_SIDE_THRESHOLD = 640;
const INVESTIGATOR_HIGHRES_ATTRS = [
  'data-large-file',
  'data-orig-file',
  'data-full-src',
  'data-zoom-src',
  'data-hi-res-src',
  'data-original',
  'data-src'
];

function parseSrcset(raw, baseUri) {
  const results = [];
  if (!raw || typeof raw !== 'string') return results;
  const len = raw.length;
  let i = 0;

  while (i < len) {
    while (i < len && (raw[i] === ',' || /\s/.test(raw[i]))) i++;
    if (i >= len) break;

    const urlStart = i;
    while (i < len && !/\s/.test(raw[i])) i++;
    let urlToken = raw.slice(urlStart, i);
    let width = null;
    let density = null;

    if (urlToken.endsWith(',')) {
      urlToken = urlToken.replace(/,+$/, '');
      density = 1;
    } else {
      while (i < len && /\s/.test(raw[i])) i++;
      const descStart = i;
      while (i < len && raw[i] !== ',') i++;
      const descStr = raw.slice(descStart, i).trim();
      if (i < len && raw[i] === ',') i++;

      if (!descStr) {
        density = 1;
      } else {
        const firstToken = descStr.split(/\s+/)[0];
        const match = firstToken.match(/^(\d+(?:\.\d+)?)(w|x)$/i);
        if (!match) continue;
        const val = parseFloat(match[1]);
        if (!Number.isFinite(val) || val <= 0) continue;
        if (match[2].toLowerCase() === 'w') {
          width = Math.round(val);
        } else {
          density = val;
        }
      }
    }

    if (!urlToken) continue;
    try {
      const resolved = new URL(urlToken, baseUri);
      if (resolved.protocol !== 'http:' && resolved.protocol !== 'https:') continue;
      results.push({ url: resolved.href, width, density });
    } catch (_) {}
  }

  return results;
}

function collectSrcsetCandidates(el, baseUri) {
  const srcsetStrings = [];
  if (el && typeof el.getAttribute === 'function') {
    ['srcset', 'data-srcset', 'data-lazy-srcset'].forEach((attr) => {
      const v = el.getAttribute(attr);
      if (v) srcsetStrings.push(v);
    });
  }

  const parent = el && el.parentElement;
  if (parent && parent.tagName && parent.tagName.toLowerCase() === 'picture' && parent.children) {
    for (let i = 0; i < parent.children.length; i++) {
      const child = parent.children[i];
      if (!child || !child.tagName || child.tagName.toLowerCase() !== 'source') continue;
      const mimeType = typeof child.getAttribute === 'function'
        ? (child.getAttribute('type') || '').toLowerCase()
        : '';
      if (mimeType === 'image/svg+xml') continue;
      if (typeof child.getAttribute === 'function') {
        ['srcset', 'data-srcset', 'data-lazy-srcset'].forEach((attr) => {
          const v = child.getAttribute(attr);
          if (v) srcsetStrings.push(v);
        });
      }
    }
  }

  if (srcsetStrings.length === 0) return [];
  const parsedByUrl = new Map();
  srcsetStrings.forEach((raw) => {
    parseSrcset(raw, baseUri).forEach((cand) => {
      if (!parsedByUrl.has(cand.url)) {
        parsedByUrl.set(cand.url, cand);
      }
    });
  });
  return Array.from(parsedByUrl.values());
}

function getEffectiveImageDimensions(el, baseSrc) {
  const nw = (el && el.naturalWidth) || 0;
  const nh = (el && el.naturalHeight) || 0;
  if (!el || nw <= 0 || nh <= 0) {
    return { width: nw, height: nh, density: 1 };
  }

  const baseUri = (typeof document !== 'undefined' && document.baseURI) ||
    (typeof window !== 'undefined' && window.location && window.location.href) ||
    baseSrc || '';
  const candidates = collectSrcsetCandidates(el, baseUri);
  if (candidates.length > 0 && baseSrc) {
    const match = candidates.find((c) => c.url === baseSrc && c.density && c.density > 1);
    if (match) {
      return {
        width: Math.round(nw * match.density),
        height: Math.round(nh * match.density),
        density: match.density
      };
    }
  }
  return { width: nw, height: nh, density: 1 };
}

function getRegistrableDomain(hostname) {
  const host = String(hostname || '').toLowerCase().replace(/^www\./, '').replace(/\.$/, '');
  if (!host) return '';
  const parts = host.split('.');
  if (parts.length <= 2) return host;
  const twoPart = parts.slice(-2).join('.');
  if (twoPart === 'co.uk' || twoPart === 'com.au' || twoPart === 'org.uk' || twoPart === 'ac.uk') {
    return parts.slice(-3).join('.');
  }
  return twoPart;
}

function upgradeSameCdnUrl(urlStr, el = null) {
  try {
    const u = new URL(urlStr);
    const host = u.hostname.toLowerCase();

    // Pinterest small thumbnails -> 736x medium-high tier
    if (host === 'i.pinimg.com') {
      if (/\/(?:170x|236x|474x|564x)\//.test(u.pathname)) {
        u.pathname = u.pathname.replace(/\/(?:170x|236x|474x|564x)\//, '/736x/');
        return u.href;
      }
      return null;
    }

    // X / Twitter media thumbnails -> large
    if (host === 'pbs.twimg.com' && u.pathname.startsWith('/media/')) {
      const nameParam = u.searchParams.get('name');
      if (nameParam && /^(?:thumb|tiny|small|120x120|240x240|360x360)$/i.test(nameParam)) {
        u.searchParams.set('name', 'large');
        return u.href;
      }
      if (/:(?:small|thumb)$/i.test(u.pathname)) {
        u.pathname = u.pathname.replace(/:(?:small|thumb)$/i, ':large');
        return u.href;
      }
      return null;
    }

    // Wikimedia Commons thumbnails -> valid standard bucket step (960px or 500px; 800px returns HTTP 400)
    if ((host === 'upload.wikimedia.org' || host === 'thumb.wikimedia.org') && u.pathname.includes('/thumb/')) {
      const m = u.pathname.match(/\/(\d+)px-([^/]+)$/);
      if (m) {
        const currentPx = parseInt(m[1], 10);
        const origWidthAttr = el && typeof el.getAttribute === 'function'
          ? parseInt(el.getAttribute('data-file-width') || '', 10)
          : NaN;
        const maxOrigPx = Number.isFinite(origWidthAttr) && origWidthAttr > 0 ? origWidthAttr : Infinity;
        const validSteps = [500, 960];
        const targetStep = validSteps.find((s) => s > currentPx && s <= maxOrigPx && (s >= INVESTIGATOR_MAX_SIDE_THRESHOLD || maxOrigPx < 960))
          || (960 > currentPx && maxOrigPx >= 960 ? 960 : (500 > currentPx && maxOrigPx >= 500 ? 500 : null));
        if (targetStep && targetStep > currentPx) {
          u.pathname = u.pathname.replace(/\/(\d+)px-([^/]+)$/, `/${targetStep}px-` + m[2]);
          return u.href;
        }
      }
      return null;
    }

    // WordPress Photon CDN
    if (host === 'i0.wp.com' || host === 'i1.wp.com' || host === 'i2.wp.com') {
      const wParam = u.searchParams.get('w');
      if (wParam && /^\d+$/.test(wParam) && parseInt(wParam, 10) < INVESTIGATOR_MAX_SIDE_THRESHOLD) {
        u.searchParams.set('w', '768');
        return u.href;
      }
      return null;
    }

    // Shopify CDN
    if (host === 'cdn.shopify.com' || u.pathname.includes('/cdn/shop/')) {
      const widthParam = u.searchParams.get('width');
      if (widthParam && /^\d+$/.test(widthParam) && parseInt(widthParam, 10) < INVESTIGATOR_MAX_SIDE_THRESHOLD) {
        u.searchParams.set('width', '800');
        return u.href;
      }
      const m = u.pathname.match(/_(\d{2,3})x(\.(?:jpe?g|png|webp))$/i);
      if (m && parseInt(m[1], 10) < INVESTIGATOR_MAX_SIDE_THRESHOLD) {
        u.pathname = u.pathname.replace(/_(\d{2,3})x(\.(?:jpe?g|png|webp))$/i, '_800x' + m[2]);
        return u.href;
      }
      return null;
    }
  } catch (_) {}
  return null;
}

function resolveInvestigatorImageUrl(el, baseSrc, nw, nh) {
  if (!el || !baseSrc) return null;
  const baseUri = (typeof document !== 'undefined' && document.baseURI) ||
    (typeof window !== 'undefined' && window.location && window.location.href) ||
    baseSrc;

  const dims = getEffectiveImageDimensions(el, baseSrc);
  const trueWidth = dims.width || nw || 0;

  // 1. Gather srcset candidates from <img> and parent <picture><source>
  const candidates = collectSrcsetCandidates(el, baseUri);
  if (candidates.length > 0) {
    // nw is already CSS 1x width when a density-based srcset candidate is active
    const base1xWidth = nw || 240;

    const better = candidates
      .map((c) => ({
        url: c.url,
        estWidth: c.width || Math.round((c.density || 1) * base1xWidth)
      }))
      .filter((c) => c.url !== baseSrc && c.estWidth > trueWidth);

    if (better.length > 0) {
      const geThreshold = better.filter((c) => c.estWidth >= INVESTIGATOR_MAX_SIDE_THRESHOLD);
      if (geThreshold.length > 0) {
        geThreshold.sort((a, b) => a.estWidth - b.estWidth);
        return geThreshold[0].url;
      }
      better.sort((a, b) => b.estWidth - a.estWidth);
      return better[0].url;
    }
  }

  // 2. High-res data-* attributes on the <img> element
  if (typeof el.getAttribute === 'function') {
    for (let i = 0; i < INVESTIGATOR_HIGHRES_ATTRS.length; i++) {
      const val = el.getAttribute(INVESTIGATOR_HIGHRES_ATTRS[i]);
      if (!val || !val.trim()) continue;
      try {
        const resolved = new URL(val.trim(), baseUri);
        if ((resolved.protocol === 'http:' || resolved.protocol === 'https:') && resolved.href !== baseSrc) {
          return resolved.href;
        }
      } catch (_) {}
    }
  }

  // 3. Same-domain direct-image wrapping <a href> (excluding HTML file description pages such as MediaWiki /wiki/File:...)
  const anchor = typeof el.closest === 'function' ? el.closest('a[href]') : null;
  const isMediaWikiFileLink = anchor && (
    (anchor.classList && (anchor.classList.contains('mw-file-description') || anchor.classList.contains('image')))
  );
  if (anchor && !isMediaWikiFileLink && typeof anchor.getAttribute === 'function') {
    const hrefAttr = anchor.getAttribute('href');
    if (hrefAttr) {
      try {
        const anchorUrl = new URL(hrefAttr, baseUri);
        const looksLikeHtmlFilePage = /\/wiki\/[^/]*:/i.test(anchorUrl.pathname) || /\/w\/index\.php$/i.test(anchorUrl.pathname);
        if (
          !looksLikeHtmlFilePage &&
          (anchorUrl.protocol === 'http:' || anchorUrl.protocol === 'https:') &&
          anchorUrl.href !== baseSrc &&
          /\.(?:jpe?g|png|webp)(?:$|[?#])/i.test(anchorUrl.pathname)
        ) {
          const anchorDomain = getRegistrableDomain(anchorUrl.hostname);
          const pageDomain = getRegistrableDomain(typeof window !== 'undefined' && window.location && window.location.hostname);
          let srcDomain = '';
          if (baseSrc.startsWith('http://') || baseSrc.startsWith('https://')) {
            try { srcDomain = getRegistrableDomain(new URL(baseSrc).hostname); } catch (_) {}
          }
          if (anchorDomain && (anchorDomain === pageDomain || (srcDomain && anchorDomain === srcDomain))) {
            return anchorUrl.href;
          }
        }
      } catch (_) {}
    }
  }

  // 4. Safe same-CDN URL upgrade
  if (baseSrc.startsWith('http://') || baseSrc.startsWith('https://')) {
    const cdnUpgraded = upgradeSameCdnUrl(baseSrc, el);
    if (cdnUpgraded && cdnUpgraded !== baseSrc) {
      return cdnUpgraded;
    }
  }

  return null;
}

function processImageElement(el, isLazy = false) {
  if (!settingsLoaded) return null;
  if (isBellowsCollectorActive()) return null;
  if (!el || el.tagName.toLowerCase() !== 'img') return null;
  if (typeof el.closest === 'function' && el.closest('#fvf-community-report-backdrop')) return null;

  const src = el.currentSrc || el.src || el.getAttribute('src') || '';
  const prevTrackedId = elementToItemId.get(el);
  if (processedImages.has(el)) {
    if (src && prevTrackedId && !isSameResponsiveImageAsset(prevTrackedId, src, el)) {
      releaseElementTracking(el);
    } else {
      if (prevTrackedId && itemResults.has(prevTrackedId) && !el.classList.contains('ai-detected')) {
        applyStylesForItem(prevTrackedId);
      }
      return null;
    }
  }

  if (!src) {
    observeImage(el);
    return null;
  }

  const nw = el.naturalWidth || 0;
  const nh = el.naturalHeight || 0;
  if (nw === 0 || nh === 0) {
    if (el.dataset.aiLoadListenerSrc !== src) {
      el.dataset.aiLoadListenerSrc = src;
      el.dataset.aiLoadListenerAttached = 'true';

      const onImageReady = () => {
        if (!el.isConnected) return;
        const curNw = el.naturalWidth || 0;
        const curNh = el.naturalHeight || 0;
        if (curNw === 0 || curNh === 0) return;
        processedImages.delete(el);
        unobserveImage(el);
        const img = processImageElement(el);
        if (img) scheduleClassification([img]);
      };

      el.addEventListener('load', onImageReady, { once: true });
      el.addEventListener('error', () => { unobserveImage(el); }, { once: true });

      if (typeof el.decode === 'function') {
        el.decode().then(onImageReady).catch(() => {});
      }

      // Fast and staggered checks for lazy/virtualized images (e.g. YouTube sidebar & scroll feed)
      requestAnimationFrame(() => {
        if (el.naturalWidth > 0 && el.naturalHeight > 0) {
          onImageReady();
        } else {
          setTimeout(onImageReady, 200);
          setTimeout(onImageReady, 650);
          setTimeout(onImageReady, 1500);
          setTimeout(onImageReady, 2800);
        }
      });
    }
    observeImage(el);
    return null;
  }

  // If element has natural dimensions >= minDim but its DOM box is still 0x0 or mid-transition (<24px), retry shortly
  const renderedNow = getRenderedImageDimensions(el);
  if (
    nw >= getMinImageDimension() &&
    nh >= 54 &&
    ((renderedNow.width > 0 && renderedNow.width < 24) || (renderedNow.height > 0 && renderedNow.height < 24)) &&
    !el.dataset.aiLayoutRetryScheduled
  ) {
    el.dataset.aiLayoutRetryScheduled = 'true';
    setTimeout(() => {
      if (el.dataset) delete el.dataset.aiLayoutRetryScheduled;
      if (!el.isConnected || processedImages.has(el)) return;
      const img = processImageElement(el);
      if (img) scheduleClassification([img]);
    }, 300);
    return null;
  }

  // Filter out physically small images (profile avatars, UI/navigation icons, tiny sprites)
  if (isPhysicallySmallImage(el, nw, nh)) {
    return null;
  }

  if (settings.altTextOnly) {
    const alt = el.alt || el.getAttribute('alt') || '';
    const host = (typeof window !== 'undefined' && window.location && window.location.hostname)
      ? window.location.hostname.toLowerCase()
      : '';
    const hasSrcset = Boolean(
      (el.getAttribute && (el.getAttribute('srcset') || el.getAttribute('data-srcset'))) ||
      (el.parentElement && el.parentElement.tagName && el.parentElement.tagName.toLowerCase() === 'picture')
    );
    const inContentContainer = Boolean(
      typeof el.closest === 'function' &&
      el.closest('article, figure, picture, main, [data-testid*="card"], [data-testid*="media"], [data-testid*="image"], [class*="promo"], [class*="media"], [class*="card"]')
    );
    const isExemptAltTextRule =
      hasSrcset ||
      inContentContainer ||
      host.includes('youtube.com') ||
      host.includes('bbc.co.uk') ||
      host.includes('bbc.com') ||
      (el.classList && el.classList.contains('yt-core-image')) ||
      isYoutubeVideoThumbnailElement(el) ||
      src.includes('ichef.bbci.co.uk/') ||
      src.includes('ytimg.com/') ||
      src.includes('ggpht.com/') ||
      src.includes('googleusercontent.com/') ||
      src.includes('encrypted-tbn') ||
      src.includes('gstatic.com') ||
      src.includes('pbs.twimg.com/') ||
      src.includes('redd.it/') ||
      src.includes('redditmedia.com/') ||
      src.includes('pinimg.com/') ||
      src.includes('imgur.com/') ||
      src.includes('wikimedia.org/');

    if (!isExemptAltTextRule && (!alt || !alt.trim().length)) {
      return null;
    }
  }

  processedImages.add(el);
  unobserveImage(el);
  trackElement(src, el);
  const rawAttrSrc = (el.getAttribute && el.getAttribute('src')) || '';
  if (rawAttrSrc && rawAttrSrc !== src) {
    trackElement(rawAttrSrc, el);
  }
  const canonicalKey = extractCanonicalResponsiveImageKey(src);
  if (canonicalKey) {
    trackElement(canonicalKey, el);
  }

  if (itemResults.has(src)) {
    seenImageUrls.add(src);
    applyStylesForItem(src);
    return null;
  }
  if (canonicalKey && itemResults.has(canonicalKey)) {
    seenImageUrls.add(src);
    itemResults.set(src, itemResults.get(canonicalKey));
    applyStylesForItem(src);
    return null;
  }

  if (seenImageUrls.has(src)) {
    // If already pending in Native Host, trackElement(src, el) above has attached this hydrated DOM node
    // so when CLASSIFY_ITEM_RESULT arrives for `src`, it will style `el`.
    // However, if it is NEITHER in itemResults NOR in pendingItemIds, re-queue it!
    if (pendingItemIds.has(src)) {
      return null;
    }
  }

  seenImageUrls.add(src);

  const effDims = getEffectiveImageDimensions(el, src);
  const trueWidth = effDims.width || nw;
  const trueHeight = effDims.height || nh;

  const investigatorSrc = (
    settings.investigator !== false &&
    Math.max(trueWidth, trueHeight) < INVESTIGATOR_MAX_SIDE_THRESHOLD
  ) ? resolveInvestigatorImageUrl(el, src, nw, nh) : null;

  return {
    src,
    investigatorSrc,
    alt: el.alt || '',
    width: trueWidth || el.width || null,
    height: trueHeight || el.height || null,
    element: el,
    lazy: isLazy,
    inViewport: isElementInViewport(el)
  };
}

function scanAllImages() {
  const images = Array.from(document.querySelectorAll('img'));
  const newImages = [];

  images.forEach((el) => {
    const img = processImageElement(el);
    if (img) newImages.push(img);
  });

  if (settings.verboseLogs && newImages.length > 0) {
    console.log(`[AI Detector] scanAllImages found ${newImages.length} new images (total on page: ${images.length})`);
  }

  return newImages;
}

function getEnsembleDisplayLabel(ensembleCfg) {
  if (!ensembleCfg) return 'AI generated';

  const name = typeof ensembleCfg.name === 'string' ? ensembleCfg.name.trim() : '';
  if (name) return name;

  const ensembles = getEnsembleConfigs();
  const index = ensembles.findIndex((ens) => ens && ens.id === ensembleCfg.id);
  if (index >= 0) {
    return `Category ${index + 1}`;
  }

  return 'AI generated';
}

function recomputeWeightedEnsembleResult(ensembleCfg, result) {
  if (!ensembleCfg || !result || !result.classifiers || typeof result.classifiers !== 'object') {
    return result;
  }

  const classifierIds = Array.isArray(ensembleCfg.classifiers) ? ensembleCfg.classifiers : [];
  if (classifierIds.length <= 1) {
    return result;
  }

  const detailsByClassifier = result.classifiers;
  const weights = Array.isArray(ensembleCfg.weights) ? ensembleCfg.weights : [];
  const scoredClassifiers = [];

  classifierIds.forEach((classifierId, index) => {
    const details = detailsByClassifier[classifierId];
    const score = details && typeof details.score === 'number' ? details.score : Number(details && details.score);
    if (!Number.isFinite(score) || score < 0) return;
    scoredClassifiers.push({
      score,
      weight: ENS.normalizeEnsembleWeight(weights[index]),
      label: details && details.label ? details.label : null
    });
  });

  if (!scoredClassifiers.length) {
    return result;
  }

  const weightSum = scoredClassifiers.reduce((sum, item) => sum + item.weight, 0);
  if (weightSum <= 0) {
    return result;
  }

  const combinedScore = scoredClassifiers.reduce((sum, item) => sum + (item.score * item.weight), 0) / weightSum;
  const displayLabel = getEnsembleDisplayLabel(ensembleCfg)
    || scoredClassifiers.find((item) => item.label && item.label !== 'AI generated')?.label
    || result.display_label
    || result.displayLabel
    || 'AI generated';

  return {
    ...result,
    score: Math.round(combinedScore * 10000) / 10000,
    label: combinedScore >= 0.5 ? 'ai' : 'real',
    display_label: displayLabel,
    displayLabel
  };
}

async function updateClassifierTimings(results) {
  if (!Array.isArray(results) || results.length === 0) return;
  try {
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    if (!storageArea || !storageArea.get || !storageArea.set) return;

    const res = await storageArea.get(['classifierTimings']);
    const timings = (res && res.classifierTimings) ? res.classifierTimings : {};

    const samples = new Map();
    results.forEach((result) => {
      if (!result) return;
      const duration = typeof result.durationMs === 'number' ? result.durationMs : (result.duration || null);
      if (!Number.isFinite(duration) || duration <= 0) return;

      const clsMap = result.classifiers && typeof result.classifiers === 'object' ? Object.keys(result.classifiers) : [];
      if (clsMap.length === 1) {
        const id = clsMap[0];
        if (!samples.has(id)) samples.set(id, []);
        samples.get(id).push(duration);
      } else if (clsMap.length > 1) {
        const per = duration / clsMap.length;
        clsMap.forEach((id) => {
          if (!samples.has(id)) samples.set(id, []);
          samples.get(id).push(per);
        });
      } else {
        const ensembleId = result.ensembleId || (result.ensemble || null);
        if (ensembleId) {
          const cfg = getEnsembleById(ensembleId);
          if (cfg && Array.isArray(cfg.classifiers) && cfg.classifiers.length === 1) {
            const id = cfg.classifiers[0];
            if (!samples.has(id)) samples.set(id, []);
            samples.get(id).push(duration);
          }
        }
      }
    });

    samples.forEach((vals, classifierId) => {
      if (!vals || !vals.length) return;
      for (const sample of vals) {
        const prev = timings[classifierId] || { avgMs: 0, count: 0, lastMs: 0, updatedAt: 0 };
        if (prev.count > 0 && sample > (prev.avgMs * 5 + 1e-6)) {
          continue;
        }
        if (prev.count === 0 && sample > 10000) {
          continue;
        }
        const newCount = prev.count + 1;
        const newAvg = prev.count === 0 ? sample : ((prev.avgMs * prev.count + sample) / newCount);
        timings[classifierId] = {
          avgMs: Math.round(newAvg),
          count: newCount,
          lastMs: Math.round(sample),
          updatedAt: Date.now()
        };
      }
    });

    await storageArea.set({ classifierTimings: timings });
  } catch (e) {
    if (settings.verboseLogs) console.warn('[AI Detector] Failed to update classifier timings:', e);
  }
}

let pendingTimingResults = [];
let timingFlushTimer = null;

function queueClassifierTimings(results) {
  if (!Array.isArray(results) || results.length === 0) return;
  pendingTimingResults.push(...results);
  if (timingFlushTimer) return;
  timingFlushTimer = setTimeout(async () => {
    timingFlushTimer = null;
    const batch = pendingTimingResults;
    pendingTimingResults = [];
    try {
      await updateClassifierTimings(batch);
    } catch (e) {
      if (settings.verboseLogs) console.warn('[AI Detector] updateClassifierTimings failed', e);
    }
  }, 15000);
}

function applyStylesForItem(itemId) {
  if (isBellowsCollectorActive()) return;
  const entry = itemResults.get(itemId);
  if (!entry) return;
  const elements = getElementsForItem(itemId).filter((el) => el && el.isConnected && !(typeof el.closest === 'function' && el.closest('#fvf-community-report-backdrop')));
  if (!elements.length) {
    if (entry.modality === 'image') {
      const liveImgs = Array.from(document.querySelectorAll('img')).filter((img) => {
        if (typeof img.closest === 'function' && img.closest('#fvf-community-report-backdrop')) return false;
        const cSrc = img.currentSrc || '';
        const sSrc = img.src || (img.getAttribute && img.getAttribute('src')) || '';
        if (cSrc === itemId || sSrc === itemId) return true;
        if (isSameResponsiveImageAsset(itemId, cSrc || sSrc, img)) return true;
        const srcset = (img.getAttribute && (img.getAttribute('srcset') || img.getAttribute('data-srcset'))) || '';
        return Boolean(srcset && srcset.includes(itemId));
      });
      if (liveImgs.length) {
        liveImgs.forEach((img) => {
          trackElement(itemId, img);
          processedImages.add(img);
        });
        elements.push(...liveImgs);
      }
    }
    if (!elements.length) {
      return;
    }
  }

  const modality = entry.modality;
  const matched = [];

  Object.entries(entry.ensembles || {}).forEach(([ensembleId, data]) => {
    const cfg = getEnsembleById(ensembleId);
    if (!cfg || cfg.enabled === false) return;
    if (cfg.modality !== modality) return;
    const threshold = typeof cfg.threshold === 'number' ? cfg.threshold : 0.5;
    if (data.label !== 'uncertain' && data.score >= 0 && data.score >= threshold) {
      let displayLabel = 'AI generated';
      const isSingleClassifier = Array.isArray(cfg.classifiers) && cfg.classifiers.length === 1;
      if (isSingleClassifier) {
        const classifierId = cfg.classifiers[0];
        const classifierDetail = data.classifiers && data.classifiers[classifierId];
        displayLabel = (classifierDetail && classifierDetail.label) || getEnsembleDisplayLabel(cfg) || 'AI generated';
      } else {
        displayLabel = getEnsembleDisplayLabel(cfg) || data.displayLabel || 'AI generated';
      }
      matched.push({
        cfg,
        score: data.score,
        displayLabel,
        classifiers: data.classifiers || {}
      });
    }
  });

  if (modality === 'image') {
    elements.forEach((element) => {
      const wasDetected = element.classList.contains('ai-detected');
      if (!matched.length) {
        if (wasDetected) aiDetectedImageCount = Math.max(0, aiDetectedImageCount - 1);
        VE.clearImageStyles(element);
        updateHoverHandlerAttachment();
        return;
      }

      VE.applyImageMatchedStyles(element, matched, settings);
      if (!wasDetected) aiDetectedImageCount += 1;
      updateHoverHandlerAttachment();
    });
    return;
  }

  if (modality === 'text') {
    elements.forEach((element) => {
      if (!matched.length) {
        VE.clearTextStyles(element);
        return;
      }

      VE.applyTextMatchedStyles(element, matched, settings, {
        onMouseLeave: () => applyStylesForItem(itemId)
      });
    });
  }
}

function applyResultChunk(response) {
  if (isBellowsCollectorActive() || !response || !Array.isArray(response.results)) return;

  if (settings.verboseLogs && Array.isArray(response.errors) && response.errors.length > 0) {
    response.errors.filter((e) => e.type === 'info').forEach((log) => console.log(log.message));
  }

  response.results.forEach((result) => {
    if (!result || !result.id) return;
    if (result.notes === 'paused_bellows_training') {
      pendingItemIds.delete(result.id);
      seenImageUrls.delete(result.id);
      getElementsForItem(result.id).forEach((el) => {
        processedImages.delete(el);
      });
      return;
    }
    if (result.notes === 'classifier_unavailable') {
      return;
    }
    const entry = itemResults.get(result.id) || { modality: result.modality, ensembles: {} };
    entry.modality = result.modality || entry.modality;
    const ensembleId = result.ensembleId || response.ensembleId || 'default';
    const ensembleCfg = getEnsembleById(ensembleId);
    const finalResult = recomputeWeightedEnsembleResult(ensembleCfg, result);
    entry.ensembles[ensembleId] = {
      score: typeof finalResult.score === 'number' ? finalResult.score : 0.5,
      label: finalResult.label || 'uncertain',
      displayLabel: finalResult.display_label || finalResult.displayLabel || 'AI generated',
      classifiers: finalResult.classifiers || {}
    };
    itemResults.set(result.id, entry);
    const respCanonKey = extractCanonicalResponsiveImageKey(result.id);
    if (respCanonKey) {
      itemResults.set(respCanonKey, entry);
    }
    pendingItemIds.delete(result.id);
    if (hoverPriorityItemId === result.id) {
      if (hoverPriorityTimer) {
        clearTimeout(hoverPriorityTimer);
        hoverPriorityTimer = null;
      }
      hoverPriorityItemId = null;
    }
    if (hoverPrioritySentId === result.id) {
      hoverPrioritySentId = null;
    }

    // Unobserve once classified
    getElementsForItem(result.id).forEach((el) => unobserveImage(el));

    const elements = getElementsForItem(result.id);
    if (elements.some((el) => el && el.isConnected) || result.modality === 'image') {
      applyStylesForItem(result.id);
    }
  });

  updateHoverHandlerAttachment();

  try {
    queueClassifierTimings(response.results);
  } catch (e) {
    if (settings.verboseLogs) console.warn('[AI Detector] queueClassifierTimings failed to start', e);
  }
}

function markElementsClassified(elems, textPreview) {
  elems.forEach((el) => {
    clearStreamingTextElement(el);
    processedTextElements.add(el);
    const elText = TU.normalizeElementText(el);
    elementClassifiedMeta.set(el, {
      preview: elText.substring(0, 200),
      length: elText.length
    });
    trackElement(textPreview, el);
  });
}

function acceptCohesiveTextSection(sec) {
  if (!sec || !sec.text) return null;
  const elems = Array.isArray(sec.elements) && sec.elements.length ? sec.elements : [sec.element];
  if (elems.some((el) => isTextElementBlocked(el))) {
    return null;
  }

  const textPreview = sec.text.substring(0, 500);
  if (itemResults.has(textPreview)) {
    markElementsClassified(elems, textPreview);
    applyStylesForItem(textPreview);
    return null;
  }

  if (pendingItemIds.has(textPreview)) {
    markElementsClassified(elems, textPreview);
    if (elems.some((el) => isElementInViewport(el))) {
      ext.runtime.sendMessage({
        type: 'BOOST_ITEMS',
        pageId: currentPageId,
        itemIds: [textPreview]
      }).catch(() => {});
    }
    return null;
  }

  seenTextContent.add(sec.text);
  markElementsClassified(elems, textPreview);
  return sec;
}

function extractLongText() {
  if (!settingsLoaded) return [];
  if (!hasActiveModality('text')) return [];
  try {
    if (typeof TU.extractCohesiveTextSections === 'function') {
      return extractSectionsFromScope(document, []);
    }

    const nodes = Array.from(document.querySelectorAll(TU.TEXT_SELECTORS));
    const sections = [];
    nodes.forEach((el) => {
      if (isTextElementBlocked(el)) return;
      const text = getEligibleTextForElement(el);
      if (!text) return;
      const sec = TU.buildTextSection(el, sections.length, text, [el]);
      const accepted = acceptCohesiveTextSection(sec);
      if (accepted) sections.push(accepted);
    });
    return sections;
  } catch (e) {
    console.error('[Extension] Error extracting long text:', e);
    return [];
  }
}

async function loadSettings() {
  try {
    const keys = [
      'enabled',
      'altTextOnly',
      'ignoreSmallImages',
      'investigator',
      'blurAmount',
      'borderMultiplier',
      'borderColor',
      'miniBatchSize',
      'classificationDelay',
      'imageCaptureQuality',
      'textBlurAmount',
      'textStrikethroughEnabled',
      'imageAiThreshold',
      'textAiThreshold',
      'verboseLogs',
      'lazyLoad',
      'minTextLength',
      'minImageDimension',
      'ensembleConfigsV2'
    ];

    const safeStorageGet = (area, k) => {
      if (!area || typeof area.get !== 'function') return Promise.resolve({});
      return new Promise((resolve) => {
        let settled = false;
        const done = (val) => {
          if (!settled) {
            settled = true;
            resolve(val && typeof val === 'object' ? val : {});
          }
        };
        try {
          const ret = area.get(k, (data) => done(data));
          if (ret && typeof ret.then === 'function') {
            ret.then(done).catch(() => done({}));
          }
        } catch (_) {
          done({});
        }
        setTimeout(() => done({}), 1500);
      });
    };

    const syncArea = ext.storage && ext.storage.sync;
    const localArea = ext.storage && ext.storage.local;
    if (!syncArea && !localArea) {
      console.warn('[Extension] Storage API not available');
      settingsLoaded = true;
      return;
    }

    const [syncRes, localRes] = await Promise.all([
      safeStorageGet(syncArea, keys),
      safeStorageGet(localArea, keys)
    ]);

    let res = { ...localRes, ...syncRes };
    if (!Array.isArray(res.ensembleConfigsV2) || !res.ensembleConfigsV2.length) {
      if (Array.isArray(localRes.ensembleConfigsV2) && localRes.ensembleConfigsV2.length) {
        res.ensembleConfigsV2 = localRes.ensembleConfigsV2;
      } else if (Array.isArray(syncRes.ensembleConfigsV2) && syncRes.ensembleConfigsV2.length) {
        res.ensembleConfigsV2 = syncRes.ensembleConfigsV2;
      }
    }

    if (res && typeof res.enabled === 'boolean') settings.enabled = res.enabled;
    if (res && typeof res.altTextOnly === 'boolean') settings.altTextOnly = res.altTextOnly;
    if (res && typeof res.ignoreSmallImages === 'boolean') settings.ignoreSmallImages = res.ignoreSmallImages;
    if (res && typeof res.investigator === 'boolean') settings.investigator = res.investigator;
    if (res && typeof res.blurAmount === 'number') settings.blurAmount = res.blurAmount;
    if (res && typeof res.borderMultiplier === 'number') settings.borderMultiplier = res.borderMultiplier;
    if (res && typeof res.borderColor === 'string') settings.borderColor = res.borderColor;
    if (res && typeof res.miniBatchSize === 'number') settings.miniBatchSize = res.miniBatchSize;
    if (res && typeof res.classificationDelay === 'number') settings.classificationDelay = res.classificationDelay;
    if (res && typeof res.imageCaptureQuality === 'number') settings.imageCaptureQuality = res.imageCaptureQuality;
    if (res && typeof res.textBlurAmount === 'number') settings.textBlurAmount = res.textBlurAmount;
    if (res && typeof res.textStrikethroughEnabled === 'boolean') settings.textStrikethroughEnabled = res.textStrikethroughEnabled;
    if (res && typeof res.imageAiThreshold === 'number') settings.imageAiThreshold = res.imageAiThreshold;
    if (res && typeof res.textAiThreshold === 'number') settings.textAiThreshold = res.textAiThreshold;
    if (res && typeof res.verboseLogs === 'boolean') settings.verboseLogs = res.verboseLogs;
    if (res && typeof res.lazyLoad === 'boolean') settings.lazyLoad = res.lazyLoad;
    if (res && typeof res.minTextLength === 'number' && Number.isFinite(res.minTextLength)) {
      settings.minTextLength = Math.max(50, Math.min(2000, res.minTextLength));
    }
    if (res && typeof res.minImageDimension === 'number' && Number.isFinite(res.minImageDimension)) {
      settings.minImageDimension = Math.max(16, Math.min(512, res.minImageDimension));
    }
    if (res && Array.isArray(res.ensembleConfigsV2)) settings.ensembles = res.ensembleConfigsV2;
    settingsLoaded = true;
    updateEnsembleSignatures();
    updateHoverHandlerAttachment();
  } catch (e) {
    console.warn('[Extension] Error loading settings:', e);
    settingsLoaded = true;
  }
}

// Drop pending items for this page context when the user navigates away or closes the tab
function bindPageLifecycleCleanup() {
  if (window.__aiDetectorPageHideBound) return;
  window.__aiDetectorPageHideBound = true;
  window.addEventListener('pagehide', () => {
    try {
      ext.runtime.sendMessage({
        type: 'DROP_PAGE_CONTEXT',
        pageId: currentPageId
      }).catch(() => {});
    } catch (_) {}
  });
}

let bootstrapStarted = false;

async function _bootstrapNormalScanningIfNotCollector() {
  if (isBellowsCollectorActive()) return;
  if (bootstrapStarted) return;
  bootstrapStarted = true;

  const isCandidateCollector = Boolean(
    typeof window !== 'undefined' &&
    window.location &&
    typeof window.location.hash === 'string' &&
    window.location.hash.includes('fvf_bellows_collector=1')
  );
  if (isCandidateCollector && ext && ext.runtime && ext.runtime.sendMessage) {
    try {
      const resp = await ext.runtime.sendMessage({ type: 'BELLOWS_QUERY_COLLECTOR_TAB' });
      if (resp && resp.active) {
        activateBellowsCollector(resp);
        return;
      }
    } catch (_) {}
  }

  // Start observers immediately so no early DOM mutations are missed while storage loads
  initMutationObserver();
  initIntersectionObserver();

  await loadSettings();
  if (isBellowsCollectorActive()) return;

  scanPaused = false;
  startDynamicDetection();
  performInitialExtraction();

  // Schedule staggered follow-up sweeps AFTER settings have loaded so lazy/SPA feeds (e.g. YouTube) are caught reliably
  [250, 750, 1600, 3000, 5000].forEach((delayMs) => {
    setTimeout(() => {
      if (!document.hidden && settings.enabled && !isBellowsCollectorActive()) {
        scanPaused = false;
        performInitialExtraction();
      }
    }, delayMs);
  });
}

function initAutoExtract() {
  bindPageLifecycleCleanup();
  const runBootstrap = () => {
    _bootstrapNormalScanningIfNotCollector();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', runBootstrap, { once: true });
    window.addEventListener('load', runBootstrap, { once: true });
  } else {
    runBootstrap();
  }
  window.addEventListener('load', () => {
    if (!document.hidden && settings.enabled && !isBellowsCollectorActive()) {
      scanPaused = false;
      performInitialExtraction();
    }
  }, { once: true });
}

let scrollScanThrottleTimer = null;
let scrollScanTrailingTimer = null;

function triggerLightweightDomScan() {
  if (!settingsLoaded) return;
  if (scanPaused && !document.hidden) scanPaused = false;
  if (scanPaused || !settings.enabled || isBellowsCollectorActive()) return;
  if (hasActiveModality('image')) {
    const newImages = scanAllImages();
    if (newImages.length > 0) {
      scheduleClassification(newImages);
    }
  }
  if (hasActiveModality('text')) {
    const newText = extractLongText();
    if (newText.length > 0) {
      scheduleTextClassification(newText);
    }
  }
}

function startDynamicDetection() {
  if (isBellowsCollectorActive()) return;
  if (document.documentElement && document.documentElement.dataset.fvfOnboarding === 'true') {
    return;
  }
  if (!settings.enabled) return;
  initMutationObserver();
  initIntersectionObserver();

  if (!document.__aiDetectorVisibilityBound) {
    document.__aiDetectorVisibilityBound = true;
    scanPaused = Boolean(document.hidden);

    const onVisibilityOrFocusChange = () => {
      scanPaused = Boolean(document.hidden);
      if (scanPaused && mutationFlushTimer) {
        clearTimeout(mutationFlushTimer);
        mutationFlushTimer = null;
        pendingMutations = [];
      } else if (!scanPaused && settings.enabled && !isBellowsCollectorActive()) {
        triggerLightweightDomScan();
      }
      updateHoverHandlerAttachment();
    };

    const onScrollActivity = () => {
      if (scanPaused || !settings.enabled || isBellowsCollectorActive()) return;
      if (!scrollScanThrottleTimer) {
        scrollScanThrottleTimer = setTimeout(() => {
          scrollScanThrottleTimer = null;
          triggerLightweightDomScan();
        }, 300);
      }
      if (scrollScanTrailingTimer) clearTimeout(scrollScanTrailingTimer);
      scrollScanTrailingTimer = setTimeout(() => {
        scrollScanTrailingTimer = null;
        triggerLightweightDomScan();
      }, 260);
    };

    const onSpaNavigation = () => {
      if (!settings.enabled || isBellowsCollectorActive()) return;
      triggerLightweightDomScan();
      [200, 700, 1600, 3000].forEach((ms) => {
        setTimeout(() => {
          if (!scanPaused && settings.enabled && !isBellowsCollectorActive()) {
            triggerLightweightDomScan();
          }
        }, ms);
      });
    };

    document.addEventListener('visibilitychange', onVisibilityOrFocusChange);
    window.addEventListener('focus', onVisibilityOrFocusChange);
    window.addEventListener('pageshow', onVisibilityOrFocusChange);
    window.addEventListener('scroll', onScrollActivity, { capture: true, passive: true });
    window.addEventListener('popstate', onSpaNavigation);
    window.addEventListener('yt-navigate-finish', onSpaNavigation);
    document.addEventListener('yt-navigate-finish', onSpaNavigation);
    document.addEventListener('yt-page-data-updated', onSpaNavigation);
  }

  if (scanIntervalId) clearInterval(scanIntervalId);
  scanIntervalId = setInterval(() => {
    if (scanPaused && !document.hidden) scanPaused = false;
    if (scanPaused || isBellowsCollectorActive()) return;
    cleanupStaleEntries();
    if (hasActiveModality('image')) {
      const newImages = scanAllImages();
      if (newImages.length > 0) {
        scheduleClassification(newImages);
      }
    }
    if (hasActiveModality('text')) {
      const newText = extractLongText();
      if (newText.length > 0) {
        scheduleTextClassification(newText);
      }
    }
  }, PERIODIC_SCAN_MS);
}

function performInitialExtraction() {
  if (!settingsLoaded) return;
  if (isBellowsCollectorActive()) {
    return;
  }
  if (document.documentElement && document.documentElement.dataset.fvfOnboarding === 'true') {
    return;
  }
  if (!settings.enabled) {
    console.log('[AI Detector] Disabled in options; skipping scan');
    return;
  }
  const hasImages = hasActiveModality('image');
  const hasText = hasActiveModality('text');

  if (!hasImages && !hasText) {
    if (settings.verboseLogs) {
      console.log('[AI Detector] No active image or text classifiers configured; skipping scan');
    }
    return;
  }

  const initialImages = hasImages ? scanAllImages() : [];
  const initialTextSections = hasText ? extractLongText() : [];

  if (initialImages.length > 0 || initialTextSections.length > 0) {
    console.log(`[AI Detector] Streaming ${initialImages.length} images, ${initialTextSections.length} text sections individually`);
  }

  if (initialImages.length > 0) {
    scheduleClassification(initialImages);
  }
  if (initialTextSections.length > 0) {
    scheduleTextClassification(initialTextSections);
  }
}

initAutoExtract();

// Onboarding Web-to-Extension Bridge
const ALLOWED_ONBOARDING_ACTIONS = new Set([
  'PING',
  'CHECK_NATIVE_HOST',
  'CHECK_NATIVE_UPDATE',
  'APPLY_NATIVE_UPDATE',
  'OPEN_OPTIONS_FULLPAGE',
  'OPEN_MODELS_HUB',
  'OPEN_GUIDE_PAGE',
  'OPEN_BELLOWS',
  'GET_OFFICIAL_CATALOG',
  'DOWNLOAD_OFFICIAL_MODEL',
  'UNINSTALL_OFFICIAL_MODEL',
  'AUTO_SETUP_CATEGORIES',
  'SAVE_ONBOARDING_ENSEMBLES',
  'BELLOWS_PREFLIGHT',
  'BELLOWS_CHECK_ETHICS',
  'BELLOWS_PREPARE_SESSION',
  'BELLOWS_ADD_SAMPLES',
  'BELLOWS_REMOVE_SAMPLE',
  'BELLOWS_START_COLLECTOR',
  'BELLOWS_STOP_COLLECTOR',
  'BELLOWS_START_TRAINING',
  'BELLOWS_GET_STATUS',
  'BELLOWS_STOP_TRAINING',
  'BELLOWS_CLEANUP_STORAGE',
  'BELLOWS_FINALIZE_CATEGORY'
]);

window.addEventListener('message', (event) => {
  if (event.source !== window) return;
  const data = event.data;
  if (!data || data.source !== 'fvf-onboarding-page') return;

  const action = data.action;
  const requestId = data.requestId;
  if (!ALLOWED_ONBOARDING_ACTIONS.has(action)) {
    window.postMessage({
      source: 'fvf-extension-bridge',
      requestId,
      response: { ok: false, error: `Unsupported action: ${action}` }
    }, '*');
    return;
  }

  const payload = Object.assign({}, data.payload || {}, { type: action });
  try {
    ext.runtime.sendMessage(payload)
      .then((response) => {
        window.postMessage({
          source: 'fvf-extension-bridge',
          requestId,
          response: response || { ok: true }
        }, '*');
      })
      .catch((err) => {
        window.postMessage({
          source: 'fvf-extension-bridge',
          requestId,
          response: { ok: false, error: err?.message || String(err) }
        }, '*');
      });
  } catch (err) {
    window.postMessage({
      source: 'fvf-extension-bridge',
      requestId,
      response: { ok: false, error: err?.message || String(err) }
    }, '*');
  }
});

let lastContextMenuTarget = null;
let lastContextMenuPoint = null;
let lastContextMenuResolvedImage = null;
let lastContextMenuResolvedText = null;

function scoreCandidateImage(img, x, y, zRank = 0) {
  if (!img || !img.isConnected || img.tagName?.toLowerCase() !== 'img') return -Infinity;
  const rect = typeof img.getBoundingClientRect === 'function' ? img.getBoundingClientRect() : null;
  if (!rect) return -Infinity;

  const natW = img.naturalWidth || 0;
  const natH = img.naturalHeight || 0;
  const w = rect.width > 0 ? rect.width : natW;
  const h = rect.height > 0 ? rect.height : natH;

  // 1. Invisible or zero-size images / 1x1 tracking pixels
  if (w <= 0 || h <= 0) return -Infinity;
  if (natW === 1 && natH === 1) return -Infinity;
  if (typeof window !== 'undefined' && typeof window.getComputedStyle === 'function') {
    try {
      const cs = window.getComputedStyle(img);
      if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity || '1') === 0) {
        return -Infinity;
      }
    } catch (_) {}
  }

  let score = 0;

  // 2. Point containment
  const containsX = rect.left <= x && x <= rect.right;
  const containsY = rect.top <= y && y <= rect.bottom;
  if (containsX && containsY) {
    score += 500;
  } else {
    // If click was near bounds (e.g. edge or border margin)
    const distX = Math.max(0, rect.left - x, x - rect.right);
    const distY = Math.max(0, rect.top - y, y - rect.bottom);
    if (distX <= 12 && distY <= 12) {
      score += 250 - (distX + distY) * 10;
    } else {
      return -Infinity;
    }
  }

  // 3. Size / Content Prominence (filter out tiny icons/emojis/avatars)
  const minDim = Math.min(w, h);
  if (minDim < 20) {
    score -= 800; // Tiny icon / favicon
  } else if (minDim < 44) {
    score -= 150; // Small badge
  } else {
    score += Math.min(400, Math.round(Math.sqrt(w * h)));
  }

  // 4. Was this image tracked or scanned by FvF?
  const src = img.currentSrc || img.src || '';
  if (elementToItemId.has(img) || (src && seenImageUrls.has(src)) || processedImages.has(img)) {
    score += 250;
  }

  // 5. Cursor proximity to image center (normalized by image size)
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  const distToCenter = Math.hypot(x - centerX, y - centerY);
  const maxDim = Math.max(w, h, 1);
  score += Math.max(0, 100 - (distToCenter / maxDim) * 100);

  // 6. Z-order rank (elements earlier in elementsFromPoint are closer to the user)
  score += Math.max(0, 50 - zRank * 5);

  return score;
}

function findTargetImageFromPoint(x, y, directTarget) {
  // If coordinates are invalid (e.g. keyboard Menu key invocation), fallback to directTarget inspection
  if (typeof x !== 'number' || typeof y !== 'number' || (x === 0 && y === 0 && (!directTarget || directTarget === document.body))) {
    if (directTarget) {
      if (directTarget.tagName?.toLowerCase() === 'img') return directTarget;
      const sub = directTarget.querySelector?.('img');
      if (sub) return sub;
    }
    return currentHoverEl && currentHoverEl.tagName?.toLowerCase() === 'img' ? currentHoverEl : null;
  }

  // Priority 1: Check if an element at/near this point has been classified and flagged (.ai-detected)
  // Check currentHoverEl first (actively maintained during hover with fvf-hover-active)
  if (currentHoverEl && currentHoverEl.isConnected && currentHoverEl.tagName?.toLowerCase() === 'img') {
    const r = currentHoverEl.getBoundingClientRect();
    if (x >= r.left - 8 && x <= r.right + 8 && y >= r.top - 8 && y <= r.bottom + 8) {
      return currentHoverEl;
    }
  }

  // Next, use document.elementsFromPoint(x, y) to penetrate all overlay elements
  let elements = [];
  if (typeof document.elementsFromPoint === 'function') {
    try {
      elements = document.elementsFromPoint(x, y) || [];
    } catch (_) {}
  }

  // First pass: Flagged elements (.ai-detected)
  for (let i = 0; i < elements.length; i++) {
    const el = elements[i];
    if (!el || !el.isConnected) continue;
    if (el.tagName && el.tagName.toLowerCase() === 'img') {
      if (el.classList.contains('ai-detected')) {
        return el;
      }
    } else {
      if (el.classList && el.classList.contains('ai-detected')) {
        const nestedImg = el.querySelector('img');
        if (nestedImg) return nestedImg;
      }
      if (typeof el.closest === 'function') {
        const flaggedAncestor = el.closest('.ai-detected');
        if (flaggedAncestor) {
          if (flaggedAncestor.tagName?.toLowerCase() === 'img') return flaggedAncestor;
          const nestedImg = flaggedAncestor.querySelector('img');
          if (nestedImg) return nestedImg;
        }
      }
      if (el.querySelectorAll) {
        const flaggedImgs = el.querySelectorAll('img.ai-detected');
        for (let j = 0; j < flaggedImgs.length; j++) {
          const cImg = flaggedImgs[j];
          const cr = cImg.getBoundingClientRect();
          if (x >= cr.left - 6 && x <= cr.right + 6 && y >= cr.top - 6 && y <= cr.bottom + 6) {
            return cImg;
          }
        }
      }
    }
  }

  // Check if any element in the stack has an itemId in itemResults with a positive image score
  for (let i = 0; i < elements.length; i++) {
    const el = elements[i];
    if (!el) continue;
    const itemId = elementToItemId.get(el);
    if (itemId && itemResults.has(itemId)) {
      const res = itemResults.get(itemId);
      if (res && res.modality === 'image') {
        if (el.tagName?.toLowerCase() === 'img') return el;
        const sub = el.querySelector?.('img');
        if (sub) return sub;
      }
    }
  }

  // Priority 2: Intelligent candidate selection for unflagged elements (False negative case)
  const candidateMap = new Map();

  const addCandidate = (img, zRank = 0) => {
    if (!img || !img.isConnected || img.tagName?.toLowerCase() !== 'img') return;
    if (candidateMap.has(img)) return;
    const score = scoreCandidateImage(img, x, y, zRank);
    if (Number.isFinite(score) && score > -500) {
      candidateMap.set(img, score);
    }
  };

  if (directTarget && directTarget.tagName?.toLowerCase() === 'img') {
    addCandidate(directTarget, 0);
  }

  for (let i = 0; i < elements.length; i++) {
    const el = elements[i];
    if (!el) continue;
    if (el.tagName?.toLowerCase() === 'img') {
      addCandidate(el, i);
    } else {
      if (el.querySelectorAll) {
        const imgs = el.querySelectorAll('img');
        for (let j = 0; j < imgs.length; j++) {
          const img = imgs[j];
          const r = img.getBoundingClientRect();
          if (x >= r.left - 6 && x <= r.right + 6 && y >= r.top - 6 && y <= r.bottom + 6) {
            addCandidate(img, i + 1);
          }
        }
      }
    }
  }

  if (candidateMap.size > 0) {
    let bestImg = null;
    let bestScore = -Infinity;
    for (const [img, score] of candidateMap.entries()) {
      if (score > bestScore) {
        bestScore = score;
        bestImg = img;
      }
    }
    if (bestImg) return bestImg;
  }

  // Fallback to directTarget if it's an img
  if (directTarget && directTarget.tagName?.toLowerCase() === 'img') {
    return directTarget;
  }

  return null;
}

function findTargetTextFromPoint(x, y, directTarget) {
  let elements = [];
  if (typeof document.elementsFromPoint === 'function' && typeof x === 'number' && typeof y === 'number') {
    try {
      elements = document.elementsFromPoint(x, y) || [];
    } catch (_) {}
  }
  if (directTarget && !elements.includes(directTarget)) {
    elements.unshift(directTarget);
  }

  // Pass 1: Flagged text (.ai-detected-text)
  for (let i = 0; i < elements.length; i++) {
    const el = elements[i];
    if (!el || !el.isConnected) continue;
    if (el.classList && el.classList.contains('ai-detected-text')) {
      return el;
    }
    if (typeof el.closest === 'function') {
      const parent = el.closest('.ai-detected-text');
      if (parent) return parent;
    }
  }

  // Pass 2: Unflagged text element
  for (let i = 0; i < elements.length; i++) {
    const el = elements[i];
    if (!el || !el.isConnected) continue;
    const tag = el.tagName ? el.tagName.toLowerCase() : '';
    if (TU.isTextTagName(tag)) {
      return el;
    }
  }

  return null;
}

function getTextLabelFromElement(element, fallback = 'AI text') {
  if (!element?.title) return fallback;
  const match = element.title.match(/^(.+?)\s+\(/);
  return match ? match[1] : fallback;
}

function applyManualBlurStyles(itemId, element, manualEnsemble) {
  trackElement(itemId, element);
  pendingItemIds.delete(itemId);
  itemResults.set(itemId, {
    modality: manualEnsemble.modality,
    ensembles: {
      [manualEnsemble.id]: {
        score: 1.0,
        label: 'manual',
        displayLabel: 'Blur'
      }
    }
  });

  const currentEnsembles = getEnsembleConfigs();
  const hasManualEnsemble = currentEnsembles.some((e) => e.id === manualEnsemble.id);
  if (!hasManualEnsemble) {
    settings.ensembles = [...currentEnsembles, manualEnsemble];
  }

  applyStylesForItem(itemId);
  updateHoverHandlerAttachment();

  if (!hasManualEnsemble) {
    settings.ensembles = currentEnsembles;
  }
}

function sendContextMenuUpdate(payload) {
  const key = JSON.stringify(payload);
  if (key === lastContextMenuState) return;
  lastContextMenuState = key;
  ext.runtime.sendMessage({
    type: 'UPDATE_CONTEXT_MENU',
    ...payload
  }).catch(() => {});
}

document.addEventListener('contextmenu', (evt) => {
  lastContextMenuPoint = { x: evt.clientX, y: evt.clientY };
  lastContextMenuTarget = evt.target;
  const resolvedImg = findTargetImageFromPoint(evt.clientX, evt.clientY, evt.target);
  lastContextMenuResolvedImage = resolvedImg;

  if (resolvedImg) {
    const hasVisibleAiStyling = resolvedImg.classList.contains('ai-detected');
    let label = 'tag';

    if (hasVisibleAiStyling) {
      label = VE.getImageBadgeLabel(resolvedImg, 'tag');
    }

    sendContextMenuUpdate({
      hasAiTags: hasVisibleAiStyling,
      label: label,
      modality: 'image'
    });
  } else {
    const resolvedText = findTargetTextFromPoint(evt.clientX, evt.clientY, evt.target);
    lastContextMenuResolvedText = resolvedText;

    if (resolvedText) {
      const hasVisibleAiStyling = resolvedText.classList.contains('ai-detected-text');
      sendContextMenuUpdate({
        hasAiTags: hasVisibleAiStyling,
        label: hasVisibleAiStyling ? getTextLabelFromElement(resolvedText) : 'tag',
        modality: 'text'
      });
    } else if (evt.target && evt.target.tagName) {
      const tag = evt.target.tagName.toLowerCase();
      if (TU.isTextTagName(tag)) {
        sendContextMenuUpdate({
          hasAiTags: false,
          label: 'tag',
          modality: 'text'
        });
      }
    }
  }
}, true);

// Listen for streamed classification results
if (ext && ext.runtime && ext.runtime.onMessage) {
  ext.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message || !message.type) return false;

    if (message.type === 'ONBOARDING_DOWNLOAD_PROGRESS') {
      window.postMessage({
        source: 'fvf-extension-bridge',
        type: 'ONBOARDING_DOWNLOAD_PROGRESS',
        payload: message.payload
      }, '*');
      return false;
    }

    // Immediate single-item classification result from Native Host
    if (message.type === 'CLASSIFY_ITEM_RESULT') {
      const payload = message.payload;
      if (payload && payload.pageId && payload.pageId !== currentPageId) {
        return false;
      }
      applyResultChunk(payload);
      return false;
    }

    if (message.type === 'CLASSIFY_RESULT_CHUNK') {
      const payload = message.payload;
      if (payload && payload.jobId && activeJobIds.size > 0 && !activeJobIds.has(payload.jobId)) {
        return false;
      }
      applyResultChunk(payload);
      return false;
    }

    if (message.type === 'CLASSIFY_JOB_COMPLETE') {
      const payload = message.payload;
      if (payload && payload.jobId) {
        activeJobIds.delete(payload.jobId);
        if (activeJobTimeouts.has(payload.jobId)) {
          clearTimeout(activeJobTimeouts.get(payload.jobId));
          activeJobTimeouts.delete(payload.jobId);
        }
      }
      return false;
    }

    if (message.type === 'RECLASSIFY_ALL') {
      reloadSettingsAndRescan();
      if (typeof sendResponse === 'function') sendResponse({ success: true });
      return false;
    }

    if (message.type === 'TRIGGER_DOM_SCAN') {
      if (!isBellowsCollectorActive()) {
        loadSettings().then(() => {
          if (settings.enabled && !isBellowsCollectorActive()) {
            scanPaused = false;
            startDynamicDetection();
            performInitialExtraction();
          }
        }).catch(() => {});
      }
      if (typeof sendResponse === 'function') sendResponse({ ok: true });
      return false;
    }

    if (message.type === 'CANCEL_ALL_CLASSIFICATIONS') {
      activeJobIds.clear();
      pendingClassification = [];
      pendingTextSections = [];
      if (classificationTimeout) {
        clearTimeout(classificationTimeout);
        classificationTimeout = null;
      }
      pendingItemIds.clear();
      clearHoverPriority();
      updateHoverHandlerAttachment();

      ext.runtime.sendMessage({
        type: 'CANCEL_ALL'
      }).catch(() => {});

      console.log('[AI Detector] All classifications cancelled');
      if (typeof sendResponse === 'function') sendResponse({ success: true });
      return false;
    }

    if (message.type === 'TOGGLE_BLUR') {
      const srcUrl = message.srcUrl;
      const targetImage = (srcUrl ? Array.from(document.querySelectorAll('img')).find((img) => img.currentSrc === srcUrl || img.src === srcUrl) : null)
        || (lastContextMenuResolvedImage && lastContextMenuResolvedImage.isConnected ? lastContextMenuResolvedImage : null)
        || (lastContextMenuPoint ? findTargetImageFromPoint(lastContextMenuPoint.x, lastContextMenuPoint.y, lastContextMenuTarget) : null)
        || (lastContextMenuTarget && lastContextMenuTarget.tagName?.toLowerCase() === 'img' ? lastContextMenuTarget : null);

      if (targetImage) {
        const itemId = elementToItemId.get(targetImage) || targetImage.currentSrc || targetImage.src;
        const hasVisibleAiStyling = targetImage.classList.contains('ai-detected');

        if (hasVisibleAiStyling) {
          const badgeLabel = VE.getImageBadgeLabel(targetImage, 'AI');
          if (itemId) {
            itemResults.delete(itemId);
          }
          if (targetImage.classList.contains('ai-detected')) {
            aiDetectedImageCount = Math.max(0, aiDetectedImageCount - 1);
          }
          VE.clearImageStyles(targetImage);
          updateHoverHandlerAttachment();
          if (currentHoverEl === targetImage) {
            currentHoverEl = null;
          }
          if (typeof sendResponse === 'function') sendResponse({ hasAiTags: false, label: badgeLabel, modality: 'image' });
          return false;
        } else {
          applyManualBlurStyles(itemId, targetImage, {
            id: 'manual_blur',
            name: 'Manual Blur',
            modality: 'image',
            enabled: true,
            threshold: 0.5,
            classifiers: ['manual'],
            weights: null,
            styles: {
              image: {
                applyBlur: true,
                blurAmount: (settings.blurAmount ?? 4) * 2,
                blurMode: 'hover',
                applyBorder: false,
                borderMultiplier: (settings.borderMultiplier ?? 1) * 2,
                borderColor: '#808080',
                borderMode: 'off',
                applyBadge: true,
                badgeMode: 'always'
              }
            }
          });
          if (typeof sendResponse === 'function') sendResponse({ hasAiTags: true, label: 'Blur', modality: 'image' });
          return false;
        }
      }

      const targetText = (lastContextMenuResolvedText && lastContextMenuResolvedText.isConnected ? lastContextMenuResolvedText : null)
        || lastContextMenuTarget;
      if (targetText) {
        const tag = targetText.tagName ? targetText.tagName.toLowerCase() : '';
        if (TU.isTextTagName(tag) || targetText.classList.contains('ai-detected-text')) {
          const text = TU.normalizeElementText(targetText);
          const itemId = text.substring(0, 500);
          const hasVisibleAiStyling = targetText.classList.contains('ai-detected-text');

          if (hasVisibleAiStyling) {
            const label = getTextLabelFromElement(targetText);
            if (itemId) {
              itemResults.delete(itemId);
            }
            VE.clearTextStyles(targetText);
            if (typeof sendResponse === 'function') sendResponse({ hasAiTags: false, label: label, modality: 'text' });
            return false;
          } else {
            applyManualBlurStyles(itemId, targetText, {
              id: 'manual_blur_text',
              name: 'Manual Blur Text',
              modality: 'text',
              enabled: true,
              threshold: 0.5,
              classifiers: ['manual'],
              weights: null,
              styles: {
                text: {
                  applyBlur: true,
                  blurAmount: (settings.textBlurAmount ?? 2) * 2,
                  blurMode: 'hover',
                  applyStrikethrough: false,
                  applyUnderline: false,
                  applyHighlight: false
                }
              }
            });
            if (typeof sendResponse === 'function') sendResponse({ hasAiTags: true, label: 'Blur', modality: 'text' });
            return false;
          }
        }
      }
      if (typeof sendResponse === 'function') sendResponse({ hasAiTags: false, modality: 'image' });
      return false;
    }

    if (message.type === 'OPEN_COMMUNITY_REPORT_MODAL') {
      openCommunityReportModal(message.srcUrl || null);
      if (typeof sendResponse === 'function') sendResponse({ ok: true });
      return false;
    }

    if (message && message.type === 'BELLOWS_ACTIVATE_COLLECTOR') {
      activateBellowsCollector(message);
      if (typeof sendResponse === 'function') sendResponse({ ok: true });
      return false;
    }

    if (message && (message.type === 'BELLOWS_PROGRESS' || message.type === 'BELLOWS_COLLECTOR_COUNTS' || message.type === 'ONBOARDING_DOWNLOAD_PROGRESS')) {
      window.postMessage({
        source: 'fvf-extension-bridge',
        type: message.type,
        payload: message.payload || message
      }, '*');
      return false;
    }

    return false;
  });
}

// ============================================================================
// FvF BELLOWS — INTERACTIVE DATASET GATHERING MODE HELPERS
// ============================================================================

function _extractBellowsImageDataUrl(img) {
  return new Promise((resolve) => {
    const src = img.currentSrc || img.src || '';
    if (src.startsWith('data:image/')) {
      resolve({ dataUrl: src, url: null });
      return;
    }
    try {
      const w = Math.min(320, img.naturalWidth || img.clientWidth || 256);
      const h = Math.min(320, img.naturalHeight || img.clientHeight || 256);
      if (w >= 48 && h >= 48) {
        const c = document.createElement('canvas');
        c.width = w;
        c.height = h;
        const ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
        const du = c.toDataURL('image/jpeg', 0.88);
        resolve({ dataUrl: du, url: null });
        return;
      }
    } catch (_) {
      // Cross-origin tainted canvas: hand URL to background script for CORS-free fetch
    }
    resolve({ dataUrl: null, url: src });
  });
}

function _repositionBellowsOverlays() {
  for (const [el, ov] of bellowsCollector.overlayRefs.entries()) {
    if (!document.contains(el)) {
      ov.remove();
      bellowsCollector.overlayRefs.delete(el);
      continue;
    }
    const rect = el.getBoundingClientRect();
    if (rect.width < 12 || rect.height < 12 || rect.bottom < 0 || rect.top > window.innerHeight) {
      ov.style.display = 'none';
      continue;
    }
    ov.style.display = 'flex';
    ov.style.left = `${rect.left}px`;
    ov.style.top = `${rect.top}px`;
    ov.style.width = `${rect.width}px`;
    ov.style.height = `${rect.height}px`;
  }
}

function _applyBellowsItemVisual(el, kind, sampleId) {
  bellowsCollector.elToRecord.set(el, { kind, sampleId });
  bellowsCollector.trackedEls.add(el);
  el.setAttribute('data-fvf-bellows-state', kind);

  let ov = bellowsCollector.overlayRefs.get(el);
  if (!ov) {
    ov = document.createElement('div');
    document.body.appendChild(ov);
    bellowsCollector.overlayRefs.set(el, ov);
  }
  ov.className = `fvf-bellows-item-overlay ${kind === 'positive' ? 'is-positive' : 'is-hard-neg'}`;
  ov.innerHTML = `<span class="fvf-bellows-overlay-pill">${kind === 'positive' ? '✓ Target (Saved)' : '✕ Hard Negative'}</span>`;
  _repositionBellowsOverlays();
}

function _clearBellowsItemVisual(el) {
  bellowsCollector.elToRecord.delete(el);
  bellowsCollector.trackedEls.delete(el);
  el.removeAttribute('data-fvf-bellows-state');
  const ov = bellowsCollector.overlayRefs.get(el);
  if (ov) {
    ov.remove();
    bellowsCollector.overlayRefs.delete(el);
  }
}

function _updateBellowsHud() {
  if (!bellowsCollector.hudEl) return;
  const pos = bellowsCollector.counts.positives || 0;
  const hn = bellowsCollector.counts.hard_negatives || 0;
  const goalMax = bellowsCollector.modality === 'image' ? 300 : 150;
  const minNeed = bellowsCollector.modality === 'image' ? 40 : 30;
  const solidNeed = bellowsCollector.modality === 'image' ? 120 : 80;
  const pct = Math.min(100, Math.round((pos / goalMax) * 100));

  let tierLabel = `Need ${Math.max(0, minNeed - pos)} more for minimum`;
  if (pos >= goalMax) tierLabel = 'High-Precision Dataset Ready!';
  else if (pos >= solidNeed) tierLabel = 'Strong Dataset (keep going or Done)';
  else if (pos >= minNeed) tierLabel = 'Usable Minimum Reached (more = better)';

  const posPill = bellowsCollector.hudEl.querySelector('[data-bellows-pos]');
  const hnPill = bellowsCollector.hudEl.querySelector('[data-bellows-hn]');
  const fill = bellowsCollector.hudEl.querySelector('.fvf-bellows-hud-fill');
  const statusLbl = bellowsCollector.hudEl.querySelector('[data-bellows-tier]');
  if (posPill) posPill.textContent = `✓ ${pos} Targets`;
  if (hnPill) hnPill.textContent = `✕ ${hn} Hard Negatives`;
  if (fill) fill.style.width = `${pct}%`;
  if (statusLbl) statusLbl.textContent = tierLabel;
}

async function _captureBellowsElements(elements, kind = 'positive') {
  if (!elements.length || !bellowsCollector.active) return;

  if (bellowsCollector.modality === 'image') {
    const dataPayloads = [];
    const urlPayloads = [];
    for (const el of elements) {
      const res = await _extractBellowsImageDataUrl(el);
      if (res.dataUrl) {
        dataPayloads.push({ el, dataUrl: res.dataUrl });
      } else if (res.url && /^https?:\/\//i.test(res.url)) {
        urlPayloads.push({ el, url: res.url });
      }
    }
    if (dataPayloads.length > 0) {
      const resp = await ext.runtime.sendMessage({
        type: 'BELLOWS_ADD_SAMPLES',
        jobId: bellowsCollector.jobId,
        modality: 'image',
        kind,
        samples: dataPayloads.map((p) => p.dataUrl)
      }).catch(() => null);
      if (resp && resp.ok) {
        const ids = resp.addedIds || [];
        dataPayloads.forEach((p, idx) => _applyBellowsItemVisual(p.el, kind, ids[idx] || ''));
        if (resp.counts) bellowsCollector.counts = resp.counts;
      }
    }
    for (const p of urlPayloads) {
      const resp = await ext.runtime.sendMessage({
        type: 'BELLOWS_CAPTURE_IMAGE_URL',
        jobId: bellowsCollector.jobId,
        kind,
        url: p.url
      }).catch(() => null);
      if (resp && resp.ok) {
        const sid = (resp.addedIds && resp.addedIds[0]) || '';
        _applyBellowsItemVisual(p.el, kind, sid);
        if (resp.counts) bellowsCollector.counts = resp.counts;
      }
    }
  } else {
    const pairs = elements
      .map((el) => ({ el, text: (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim() }))
      .filter((x) => x.text.length >= 50 && x.text.length <= 2400);
    if (pairs.length > 0) {
      const resp = await ext.runtime.sendMessage({
        type: 'BELLOWS_ADD_SAMPLES',
        jobId: bellowsCollector.jobId,
        modality: 'text',
        kind,
        samples: pairs.map((x) => ({ text: x.text }))
      }).catch(() => null);
      if (resp && resp.ok) {
        const ids = resp.addedIds || [];
        pairs.forEach((p, idx) => _applyBellowsItemVisual(p.el, kind, ids[idx] || ''));
        if (resp.counts) bellowsCollector.counts = resp.counts;
      }
    }
  }
  _updateBellowsHud();
}

function _getVisibleEligibleElements() {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  if (bellowsCollector.modality === 'image') {
    return Array.from(document.querySelectorAll('img')).filter((img) => {
      if (bellowsCollector.hudEl && bellowsCollector.hudEl.contains(img)) return false;
      const r = img.getBoundingClientRect();
      return r.width >= 64 && r.height >= 64 && r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw;
    });
  }
  return Array.from(document.querySelectorAll('p, [data-testid="tweetText"], shreddit-post p, ytd-comment-renderer #content-text')).filter((el) => {
    if (bellowsCollector.hudEl && bellowsCollector.hudEl.contains(el)) return false;
    const r = el.getBoundingClientRect();
    const txt = (el.innerText || el.textContent || '').trim();
    return txt.length >= 55 && txt.length <= 2400 && r.height >= 16 && r.bottom > 0 && r.top < vh;
  });
}

function _findCollectorTarget(target) {
  if (!target || (bellowsCollector.hudEl && bellowsCollector.hudEl.contains(target))) return null;
  if (bellowsCollector.modality === 'image') {
    const img = target.tagName && target.tagName.toLowerCase() === 'img'
      ? target
      : target.closest?.('a, div, figure')?.querySelector?.('img');
    if (img) {
      const r = img.getBoundingClientRect();
      if (r.width >= 56 && r.height >= 56) return img;
    }
    return null;
  }
  const el = target.closest?.('p, [data-testid="tweetText"], #content-text, blockquote');
  if (el) {
    const txt = (el.innerText || el.textContent || '').trim();
    if (txt.length >= 50 && txt.length <= 2400) return el;
  }
  return null;
}

function _startGreenEmbers() {
  const canvas = document.createElement('canvas');
  canvas.id = 'fvf-bellows-collector-embers';
  document.documentElement.appendChild(canvas);
  bellowsCollector.emberCanvas = canvas;
  const ctx = canvas.getContext('2d');
  const resize = () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  resize();
  window.addEventListener('resize', resize, { passive: true });
  const embers = Array.from({ length: 42 }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    vx: (Math.random() - 0.3) * 0.65,
    vy: -0.4 - Math.random() * 0.75,
    r: 1.4 + Math.random() * 2.6,
    a: 0.25 + Math.random() * 0.55
  }));
  const tick = () => {
    if (!bellowsCollector.active) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of embers) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.y < -12) { p.y = canvas.height + 10; p.x = Math.random() * canvas.width; }
      if (p.x > canvas.width + 12) p.x = -10;
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4);
      g.addColorStop(0, `rgba(110, 231, 183, ${p.a})`);
      g.addColorStop(0.5, `rgba(16, 185, 129, ${p.a * 0.45})`);
      g.addColorStop(1, 'rgba(16, 185, 129, 0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 4, 0, Math.PI * 2);
      ctx.fill();
    }
    bellowsCollector.emberRaf = requestAnimationFrame(tick);
  };
  bellowsCollector.emberRaf = requestAnimationFrame(tick);
}

function activateBellowsCollector(opts = {}) {
  if (bellowsCollector.active) {
    if (opts.counts) {
      bellowsCollector.counts = opts.counts;
      _updateBellowsHud();
    }
    return;
  }
  bellowsCollector.active = true;
  bellowsCollector.jobId = opts.jobId;
  bellowsCollector.modality = opts.modality === 'text' ? 'text' : 'image';
  bellowsCollector.prompt = opts.prompt || '';
  bellowsCollector.counts = opts.counts || { positives: 0, hard_negatives: 0, base_negatives: 0 };
  window.__FVF_BELLOWS_COLLECTOR_ACTIVE = true;
  resetObservers();
  resetState();
  try {
    document.querySelectorAll('.ai-detected').forEach((el) => VE.clearImageStyles(el));
    document.querySelectorAll('.ai-detected-text').forEach((el) => VE.clearTextStyles(el));
  } catch (_) {}

  _startGreenEmbers();
  const frame = document.createElement('div');
  frame.id = 'fvf-bellows-collector-frame';
  document.documentElement.appendChild(frame);
  bellowsCollector.frameEl = frame;

  const marquee = document.createElement('div');
  marquee.id = 'fvf-bellows-marquee';
  document.documentElement.appendChild(marquee);
  bellowsCollector.marqueeEl = marquee;

  const hud = document.createElement('div');
  hud.id = 'fvf-bellows-collector-hud';
  const minT = bellowsCollector.modality === 'image' ? 40 : 30;
  const solidT = bellowsCollector.modality === 'image' ? 120 : 80;
  const strongT = bellowsCollector.modality === 'image' ? 300 : 150;
  hud.innerHTML = `
    <div class="fvf-bellows-hud-top">
      <div class="fvf-bellows-hud-title">
        <span>🟢 FvF Bellows Dataset Gathering Mode</span>
        <span style="font-weight:500;color:#a7f3d0;">— Left-click: Target • Right-click: Hard Negative • Drag/Shift+Drag: Box Select</span>
      </div>
      <div class="fvf-bellows-hud-counts">
        <span class="fvf-bellows-count-pill" data-bellows-pos>✓ 0 Targets</span>
        <span class="fvf-bellows-count-pill is-hn" data-bellows-hn>✕ 0 Hard Negatives</span>
      </div>
    </div>
    <div class="fvf-bellows-hud-progress-wrap">
      <div class="fvf-bellows-hud-bar"><div class="fvf-bellows-hud-fill" style="width:0%"></div></div>
      <div class="fvf-bellows-hud-ticks">
        <span data-bellows-tier>Loading...</span>
        <span>${minT} Min • ${solidT} Strong • ${strongT}+ High-Precision</span>
      </div>
    </div>
    <div class="fvf-bellows-hud-actions">
      <div class="fvf-bellows-hud-btns">
        <button type="button" class="fvf-bellows-btn" id="fvf-bellows-box-btn">⬚ Box Select</button>
        <button type="button" class="fvf-bellows-btn" id="fvf-bellows-all-btn">Select All Visible</button>
        <button type="button" class="fvf-bellows-btn" id="fvf-bellows-pause-btn">Pause Capture (Browse)</button>
      </div>
      <button type="button" class="fvf-bellows-btn is-done" id="fvf-bellows-done-btn">Done — Return to FvF Bellows →</button>
    </div>
  `;
  document.documentElement.appendChild(hud);
  bellowsCollector.hudEl = hud;
  _updateBellowsHud();

  const boxBtn = hud.querySelector('#fvf-bellows-box-btn');
  const allBtn = hud.querySelector('#fvf-bellows-all-btn');
  const pauseBtn = hud.querySelector('#fvf-bellows-pause-btn');
  const doneBtn = hud.querySelector('#fvf-bellows-done-btn');

  boxBtn.addEventListener('click', () => {
    bellowsCollector.boxMode = !bellowsCollector.boxMode;
    boxBtn.classList.toggle('is-active', bellowsCollector.boxMode);
  });

  pauseBtn.addEventListener('click', () => {
    bellowsCollector.paused = !bellowsCollector.paused;
    pauseBtn.classList.toggle('is-active', bellowsCollector.paused);
    pauseBtn.textContent = bellowsCollector.paused ? 'Resume Capture' : 'Pause Capture (Browse)';
  });

  allBtn.addEventListener('click', async () => {
    const visible = _getVisibleEligibleElements().filter((el) => !bellowsCollector.elToRecord.has(el));
    await _captureBellowsElements(visible, 'positive');
  });

  doneBtn.addEventListener('click', () => {
    ext.runtime.sendMessage({ type: 'BELLOWS_STOP_COLLECTOR', jobId: bellowsCollector.jobId }).catch(() => {});
  });

  document.addEventListener('click', async (e) => {
    if (!bellowsCollector.active || bellowsCollector.paused || e.button !== 0) return;
    if (bellowsCollector.hudEl && bellowsCollector.hudEl.contains(e.target)) return;
    const hit = _findCollectorTarget(e.target);
    if (!hit) return;
    e.preventDefault();
    e.stopPropagation();

    const existing = bellowsCollector.elToRecord.get(hit);
    if (existing) {
      if (existing.sampleId) {
        const resp = await ext.runtime.sendMessage({
          type: 'BELLOWS_REMOVE_SAMPLE',
          jobId: bellowsCollector.jobId,
          modality: bellowsCollector.modality,
          sampleId: existing.sampleId
        }).catch(() => null);
        if (resp && resp.counts) bellowsCollector.counts = resp.counts;
      }
      _clearBellowsItemVisual(hit);
      _updateBellowsHud();
      return;
    }
    await _captureBellowsElements([hit], 'positive');
  }, true);

  document.addEventListener('contextmenu', async (e) => {
    if (!bellowsCollector.active || bellowsCollector.paused) return;
    if (bellowsCollector.hudEl && bellowsCollector.hudEl.contains(e.target)) return;
    const hit = _findCollectorTarget(e.target);
    if (!hit) return;
    e.preventDefault();
    e.stopPropagation();
    const existing = bellowsCollector.elToRecord.get(hit);
    if (existing && existing.kind === 'hard_negative') {
      if (existing.sampleId) {
        const resp = await ext.runtime.sendMessage({
          type: 'BELLOWS_REMOVE_SAMPLE',
          jobId: bellowsCollector.jobId,
          modality: bellowsCollector.modality,
          sampleId: existing.sampleId
        }).catch(() => null);
        if (resp && resp.counts) bellowsCollector.counts = resp.counts;
      }
      _clearBellowsItemVisual(hit);
      _updateBellowsHud();
      return;
    }
    await _captureBellowsElements([hit], 'hard_negative');
  }, true);

  document.addEventListener('pointerdown', (e) => {
    if (!bellowsCollector.active || bellowsCollector.paused || e.button !== 0) return;
    if (!(bellowsCollector.boxMode || e.shiftKey)) return;
    if (bellowsCollector.hudEl && bellowsCollector.hudEl.contains(e.target)) return;
    bellowsCollector.dragStart = { x: e.clientX, y: e.clientY };
    marquee.style.display = 'block';
    marquee.style.left = `${e.clientX}px`;
    marquee.style.top = `${e.clientY}px`;
    marquee.style.width = '0px';
    marquee.style.height = '0px';
    e.preventDefault();
  }, true);

  document.addEventListener('pointermove', (e) => {
    if (!bellowsCollector.dragStart) return;
    const x1 = Math.min(bellowsCollector.dragStart.x, e.clientX);
    const y1 = Math.min(bellowsCollector.dragStart.y, e.clientY);
    const w = Math.abs(e.clientX - bellowsCollector.dragStart.x);
    const h = Math.abs(e.clientY - bellowsCollector.dragStart.y);
    marquee.style.left = `${x1}px`;
    marquee.style.top = `${y1}px`;
    marquee.style.width = `${w}px`;
    marquee.style.height = `${h}px`;
  }, true);

  document.addEventListener('pointerup', async (e) => {
    if (!bellowsCollector.dragStart) return;
    const x1 = Math.min(bellowsCollector.dragStart.x, e.clientX);
    const y1 = Math.min(bellowsCollector.dragStart.y, e.clientY);
    const x2 = Math.max(bellowsCollector.dragStart.x, e.clientX);
    const y2 = Math.max(bellowsCollector.dragStart.y, e.clientY);
    bellowsCollector.dragStart = null;
    marquee.style.display = 'none';
    if ((x2 - x1) < 16 || (y2 - y1) < 16) return;

    const hits = _getVisibleEligibleElements().filter((el) => {
      if (bellowsCollector.elToRecord.has(el)) return false;
      const r = el.getBoundingClientRect();
      return !(r.right < x1 || r.left > x2 || r.bottom < y1 || r.top > y2);
    });
    if (hits.length > 0) {
      await _captureBellowsElements(hits, 'positive');
    }
  }, true);

  window.addEventListener('scroll', _repositionBellowsOverlays, { passive: true });
  window.addEventListener('resize', _repositionBellowsOverlays, { passive: true });
}

if (ext && ext.runtime && ext.runtime.sendMessage) {
  ext.runtime.sendMessage({ type: 'BELLOWS_QUERY_COLLECTOR_TAB' }).then((resp) => {
    if (resp && resp.active) {
      activateBellowsCollector(resp);
    }
  }).catch(() => {});
}

// ---------------------------------------------------------------------------
// Live reload when options change (no page reload needed)
// ---------------------------------------------------------------------------

function reapplyVisualEffects() {
  itemResults.forEach((_value, itemId) => {
    applyStylesForItem(itemId);
  });
}

let settingsUpdateGeneration = 0;
let rescanGeneration = 0;

function recalculateEnsembleScoresForAllItems() {
  itemResults.forEach((entry) => {
    Object.entries(entry.ensembles || {}).forEach(([ensembleId, storedResult]) => {
      const cfg = getEnsembleById(ensembleId);
      if (cfg && storedResult && storedResult.classifiers) {
        entry.ensembles[ensembleId] = recomputeWeightedEnsembleResult(cfg, storedResult);
      }
    });
  });
}

async function applyLatestVisualSettings({ recalculateScores = false } = {}) {
  const generation = ++settingsUpdateGeneration;
  await loadSettings();
  if (generation !== settingsUpdateGeneration) return;

  if (recalculateScores) {
    recalculateEnsembleScoresForAllItems();
    if (generation !== settingsUpdateGeneration) return;
  }

  reapplyVisualEffects();
}

function resetObservers() {
  if (mutationObserver) {
    mutationObserver.disconnect();
    mutationObserver = null;
  }
  if (intersectionObserver) {
    intersectionObserver.disconnect();
    intersectionObserver = null;
  }
  if (scanIntervalId) {
    clearInterval(scanIntervalId);
    scanIntervalId = null;
  }
  if (mutationFlushTimer) {
    clearTimeout(mutationFlushTimer);
    mutationFlushTimer = null;
  }
  pendingMutations = [];
  clearAllStreamingTextElements();
  if (timingFlushTimer) {
    clearTimeout(timingFlushTimer);
    timingFlushTimer = null;
  }
  pendingTimingResults = [];
  detachGlobalHoverHandler();
}

function resetState() {
  clearHoverPriority();
  // Drop any queued items on the Native Host for the previous page state before rescanning
  try {
    ext.runtime.sendMessage({
      type: 'DROP_PAGE_CONTEXT',
      pageId: currentPageId
    }).catch(() => {});
  } catch (_) {}

  currentPageId = 'page-' + Math.random().toString(16).slice(2, 10) + '-' + Date.now();
  cachedImages = [];
  cachedTextSections = [];
  elementMap.clear();
  elementToItemId = new WeakMap();
  processedImages = new WeakSet();
  processedTextElements = new WeakSet();
  clearAllStreamingTextElements();
  elementClassifiedMeta = new WeakMap();
  seenTextContent.clear();
  seenImageUrls.clear();
  pendingClassification = [];
  pendingTextSections = [];
  for (const timeoutId of activeJobTimeouts.values()) {
    clearTimeout(timeoutId);
  }
  activeJobTimeouts.clear();
  activeJobIds.clear();
  itemResults.clear();
  pendingItemIds.clear();
  aiDetectedImageCount = 0;
  currentHoverEl = null;
  lastContextMenuState = null;
  lastContextMenuTarget = null;
  lastContextMenuPoint = null;
  lastContextMenuResolvedImage = null;
  lastContextMenuResolvedText = null;
  if (classificationTimeout) {
    clearTimeout(classificationTimeout);
    classificationTimeout = null;
  }
  updateHoverHandlerAttachment();
}

async function reloadSettingsAndRescan() {
  if (isBellowsCollectorActive()) return;
  const generation = ++rescanGeneration;
  ++settingsUpdateGeneration;
  await loadSettings();
  if (generation !== rescanGeneration) return;
  resetObservers();
  resetState();
  startDynamicDetection();
  performInitialExtraction();
}

if (ext && ext.storage && ext.storage.onChanged) {
  ext.storage.onChanged.addListener((changes) => {
    const classificationKeys = ['enabled', 'altTextOnly', 'ignoreSmallImages', 'investigator', 'minImageDimension', 'minTextLength'];
    const visualKeys = ['blurAmount', 'borderMultiplier', 'borderColor', 'textBlurAmount', 'textStrikethroughEnabled', 'imageAiThreshold', 'textAiThreshold'];
    const performanceKeys = ['classificationDelay', 'verboseLogs', 'lazyLoad', 'imageCaptureQuality', 'miniBatchSize'];

    const ensembleChanged = Object.prototype.hasOwnProperty.call(changes, 'ensembleConfigsV2');
    let handledByEnsemble = false;

    if (ensembleChanged) {
      const prevSig = lastEnsembleSignature;
      const prevStyleSig = lastEnsembleStyleSignature;
      const newEnsembles = Array.isArray(changes.ensembleConfigsV2?.newValue)
        ? changes.ensembleConfigsV2.newValue
        : [];
      const newSig = computeEnsembleSignature(newEnsembles, false);
      const newStyleSig = computeEnsembleSignature(newEnsembles, true);
      const classifierChanged = prevSig !== newSig;
      const styleChanged = prevStyleSig !== newStyleSig;

      if (classifierChanged) {
        reloadSettingsAndRescan();
        return;
      }

      if (styleChanged) {
        applyLatestVisualSettings({ recalculateScores: true });
        return;
      }

      handledByEnsemble = true;
    }

    const needsReclassify = classificationKeys.some((k) => Object.prototype.hasOwnProperty.call(changes, k));
    const needsVisualUpdate = visualKeys.some((k) => Object.prototype.hasOwnProperty.call(changes, k));
    const needsSimpleReload = performanceKeys.some((k) => Object.prototype.hasOwnProperty.call(changes, k));

    if (needsReclassify) {
      reloadSettingsAndRescan();
    } else if (needsVisualUpdate) {
      applyLatestVisualSettings();
    } else if (needsSimpleReload && !handledByEnsemble) {
      ++settingsUpdateGeneration;
      loadSettings();
    }
  });
}

// ============================================================================
// OPT-IN NON-ANONYMOUS COMMUNITY MISCLASSIFICATION REPORT MODAL
// ============================================================================

function findReportTargetImageElement(srcUrl) {
  // 1. Check lastContextMenuResolvedImage (from positional detection on contextmenu)
  if (lastContextMenuResolvedImage && lastContextMenuResolvedImage.isConnected) {
    if (!srcUrl || lastContextMenuResolvedImage.currentSrc === srcUrl || lastContextMenuResolvedImage.src === srcUrl) {
      return lastContextMenuResolvedImage;
    }
    if (lastContextMenuPoint) {
      const r = lastContextMenuResolvedImage.getBoundingClientRect();
      if (lastContextMenuPoint.x >= r.left - 4 && lastContextMenuPoint.x <= r.right + 4 &&
          lastContextMenuPoint.y >= r.top - 4 && lastContextMenuPoint.y <= r.bottom + 4) {
        return lastContextMenuResolvedImage;
      }
    }
  }

  // 2. Re-run positional detection from lastContextMenuPoint if available
  if (lastContextMenuPoint) {
    const hitImg = findTargetImageFromPoint(lastContextMenuPoint.x, lastContextMenuPoint.y, lastContextMenuTarget);
    if (hitImg) {
      lastContextMenuResolvedImage = hitImg;
      return hitImg;
    }
  }

  // 3. Exact URL match in document if srcUrl is provided
  if (srcUrl) {
    const imgs = Array.from(document.querySelectorAll('img'));
    const match = imgs.find((im) => im.currentSrc === srcUrl || im.src === srcUrl);
    if (match) return match;
  }

  // 4. Current hovered element from pointer tracking
  if (currentHoverEl && currentHoverEl.isConnected && currentHoverEl.tagName?.toLowerCase() === 'img') {
    return currentHoverEl;
  }

  // 5. Fallback to lastContextMenuTarget
  if (lastContextMenuTarget && lastContextMenuTarget.isConnected) {
    if (lastContextMenuTarget.tagName?.toLowerCase() === 'img') return lastContextMenuTarget;
    const sub = lastContextMenuTarget.querySelector?.('img');
    if (sub) return sub;
  }

  return null;
}

function extractImageClassificationSummary(imgEl, srcUrl) {
  const itemId = (imgEl && elementToItemId.get(imgEl)) || (imgEl && (imgEl.currentSrc || imgEl.src)) || srcUrl || '';
  const entry = itemId ? itemResults.get(itemId) : null;
  let modelId = 'fvf_heimdall04_b';
  let modelName = 'FvF Heimdall';
  let modelScore = null;
  let wasFlagged = Boolean(imgEl && imgEl.classList && imgEl.classList.contains('ai-detected'));

  if (entry && entry.ensembles && typeof entry.ensembles === 'object') {
    const entries = Object.entries(entry.ensembles);
    for (const [ensId, ensVal] of entries) {
      if (!ensVal || typeof ensVal.score !== 'number') continue;
      if (modelScore === null || ensVal.score > modelScore) {
        modelScore = ensVal.score;
        const clfs = ensVal.classifiers && typeof ensVal.classifiers === 'object'
          ? Object.entries(ensVal.classifiers)
          : [];
        if (clfs.length > 0) {
          const [bestCid, bestCInfo] = clfs[0];
          modelId = bestCid || ensId;
          modelName = bestCInfo?.model_name || bestCid || 'FvF Heimdall';
          if (typeof bestCInfo?.score === 'number') modelScore = bestCInfo.score;
        } else {
          modelId = ensId;
          modelName = ensVal.displayLabel || ensId;
        }
      }
    }
  }

  return { modelId, modelName, modelScore, wasFlagged };
}

function tryExtractCleanCanvasJpegDataUrl(imgEl, maxDim = 1024, quality = 0.92) {
  if (!imgEl || typeof document === 'undefined') return null;
  try {
    const natW = imgEl.naturalWidth || imgEl.width || 0;
    const natH = imgEl.naturalHeight || imgEl.height || 0;
    if (natW < 32 || natH < 32) return null;
    let w = natW;
    let h = natH;
    if (w > maxDim || h > maxDim) {
      if (w >= h) {
        h = Math.max(1, Math.round((h * maxDim) / w));
        w = maxDim;
      } else {
        w = Math.max(1, Math.round((w * maxDim) / h));
        h = maxDim;
      }
    }
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(imgEl, 0, 0, w, h);
    const dataUrl = canvas.toDataURL('image/jpeg', quality);
    canvas.width = 0;
    canvas.height = 0;
    return dataUrl && dataUrl.startsWith('data:image/jpeg;base64,') ? dataUrl : null;
  } catch (_) {
    return null;
  }
}

async function openCommunityReportModal(srcUrl) {
  const existing = document.getElementById('fvf-community-report-backdrop');
  if (existing) existing.remove();

  const imgEl = findReportTargetImageElement(srcUrl);
  const resolvedSrc = srcUrl || (imgEl && (imgEl.currentSrc || imgEl.src)) || '';
  const summary = extractImageClassificationSummary(imgEl, resolvedSrc);
  const pageDomain = getRegistrableDomain(typeof window !== 'undefined' && window.location ? window.location.hostname : '') || 'unknown';

  const backdrop = document.createElement('div');
  backdrop.id = 'fvf-community-report-backdrop';
  backdrop.className = 'fvf-report-modal-backdrop';

  const modal = document.createElement('div');
  modal.className = 'fvf-report-modal';
  modal.innerHTML = `
    <div class="fvf-report-header">
      <h3 class="fvf-report-title">Report Incorrect Classification</h3>
      <button type="button" class="fvf-report-close" id="fvf-report-close-btn" aria-label="Close">✕</button>
    </div>
    <div id="fvf-report-body" class="fvf-report-status">Loading image…</div>
  `;
  backdrop.appendChild(modal);
  document.documentElement.appendChild(backdrop);

  let activeProgressListener = null;
  let activeProgressTimer = null;
  let autoCloseTimer = null;

  const closeModal = () => {
    if (autoCloseTimer) {
      clearTimeout(autoCloseTimer);
      autoCloseTimer = null;
    }
    if (activeProgressTimer) {
      clearInterval(activeProgressTimer);
      activeProgressTimer = null;
    }
    if (activeProgressListener && ext.runtime && ext.runtime.onMessage && ext.runtime.onMessage.removeListener) {
      try { ext.runtime.onMessage.removeListener(activeProgressListener); } catch (_) {}
      activeProgressListener = null;
    }
    document.removeEventListener('keydown', onKeyDown, true);
    backdrop.remove();
  };
  const onKeyDown = (e) => {
    if (e.key === 'Escape' && !backdrop.classList.contains('is-minimized')) {
      e.preventDefault();
      e.stopPropagation();
      closeModal();
    }
  };
  document.addEventListener('keydown', onKeyDown, true);
  backdrop.addEventListener('click', (e) => {
    if (e.target === backdrop && !backdrop.classList.contains('is-minimized')) closeModal();
  });
  modal.querySelector('#fvf-report-close-btn')?.addEventListener('click', closeModal);

  const bodyEl = modal.querySelector('#fvf-report-body');
  if (!resolvedSrc) {
    bodyEl.textContent = 'Could not locate an image under the cursor to report.';
    return;
  }

  let prep = null;
  let candidateSrc = resolvedSrc;
  if (candidateSrc.startsWith('blob:') && imgEl) {
    const fallbackUrl = tryExtractCleanCanvasJpegDataUrl(imgEl, 1024, 0.92);
    if (fallbackUrl) candidateSrc = fallbackUrl;
  }
  try {
    prep = await ext.runtime.sendMessage({
      type: 'PREPARE_COMMUNITY_REPORT_IMAGE',
      srcUrl: candidateSrc
    });
  } catch (err) {
    bodyEl.textContent = `Error preparing image: ${err?.message || err}`;
    return;
  }

  if ((!prep || !prep.ok) && imgEl && !candidateSrc.startsWith('data:')) {
    const fallbackUrl = tryExtractCleanCanvasJpegDataUrl(imgEl, 1024, 0.92);
    if (fallbackUrl) {
      try {
        prep = await ext.runtime.sendMessage({
          type: 'PREPARE_COMMUNITY_REPORT_IMAGE',
          srcUrl: fallbackUrl
        });
      } catch (_) {}
    }
  }

  if (!prep || !prep.ok) {
    bodyEl.textContent = `Failed to sanitize image: ${prep?.error || 'Unknown error'}`;
    return;
  }

  if (!prep.hasToken || !prep.username) {
    bodyEl.innerHTML = '';
    const msg = document.createElement('p');
    msg.style.margin = '0 0 12px 0';
    msg.textContent = 'Link and verify your Hugging Face account in Fire vs Fire Options before submitting non-anonymous community reports.';
    const btnRow = document.createElement('div');
    btnRow.className = 'fvf-report-actions';
    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.className = 'fvf-report-btn-secondary';
    cancelBtn.textContent = 'Close';
    cancelBtn.addEventListener('click', closeModal);
    const optsBtn = document.createElement('button');
    optsBtn.type = 'button';
    optsBtn.className = 'fvf-report-btn-primary';
    optsBtn.textContent = 'Open Fire vs Fire Options';
    optsBtn.addEventListener('click', () => {
      ext.runtime.sendMessage({ type: 'OPEN_OPTIONS_FULLPAGE' }).catch(() => {});
      closeModal();
    });
    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(optsBtn);
    bodyEl.appendChild(msg);
    bodyEl.appendChild(btnRow);
    return;
  }

  const hfRepoUrl = prep.hfRepoUrl || 'https://huggingface.co/datasets/LuckyCow/fvf-community-reports';

  bodyEl.className = '';
  bodyEl.innerHTML = '';

  const mainRow = document.createElement('div');
  mainRow.className = 'fvf-report-main-row';

  const previewWrap = document.createElement('div');
  previewWrap.className = 'fvf-report-preview-wrap';
  const previewImg = document.createElement('img');
  previewImg.className = 'fvf-report-preview-img';
  previewImg.src = prep.sanitizedDataUrl;
  previewImg.alt = 'Image preview';
  previewWrap.appendChild(previewImg);

  const choiceCol = document.createElement('div');
  choiceCol.className = 'fvf-report-choice-col';

  const aiBtn = document.createElement('button');
  aiBtn.type = 'button';
  aiBtn.className = 'fvf-report-choice-btn is-ai';
  aiBtn.textContent = 'I think this is AI generated';

  const authBtn = document.createElement('button');
  authBtn.type = 'button';
  authBtn.className = 'fvf-report-choice-btn is-authentic';
  authBtn.textContent = 'I think this is authentic';

  choiceCol.appendChild(aiBtn);
  choiceCol.appendChild(authBtn);

  const minThankYou = document.createElement('div');
  minThankYou.className = 'fvf-report-min-thankyou';
  minThankYou.hidden = true;
  const minThankTitle = document.createElement('div');
  minThankTitle.className = 'fvf-report-thankyou-title';
  minThankTitle.textContent = 'Thank you!';
  const minThankSub = document.createElement('div');
  minThankSub.className = 'fvf-report-thankyou-sub';
  minThankSub.textContent = 'Uploading report…';
  minThankYou.appendChild(minThankTitle);
  minThankYou.appendChild(minThankSub);
  choiceCol.appendChild(minThankYou);

  mainRow.appendChild(previewWrap);
  mainRow.appendChild(choiceCol);

  const progressWrap = document.createElement('div');
  progressWrap.className = 'fvf-report-progress-wrap';
  progressWrap.hidden = true;

  const progressBar = document.createElement('div');
  progressBar.className = 'fvf-report-progress-bar';
  const progressFill = document.createElement('div');
  progressFill.className = 'fvf-report-progress-fill';
  progressFill.style.width = '0%';
  progressBar.appendChild(progressFill);

  const progressText = document.createElement('div');
  progressText.className = 'fvf-report-progress-text';

  progressWrap.appendChild(progressBar);
  progressWrap.appendChild(progressText);

  const footer = document.createElement('div');
  footer.className = 'fvf-report-footer';
  footer.appendChild(document.createTextNode(`Submitting as @${prep.username} to `));
  const repoLink = document.createElement('a');
  repoLink.href = hfRepoUrl;
  repoLink.target = '_blank';
  repoLink.rel = 'noopener noreferrer';
  repoLink.textContent = hfRepoUrl;
  footer.appendChild(repoLink);

  const applyReportedFlagOnPage = (reportedLabel) => {
    if (!settings.enabled || !imgEl || !imgEl.isConnected) return;
    const activeImageEnsembles = getEnsemblesByModality().image || [];
    const activeFvfEnsembles = activeImageEnsembles.filter(
      (ens) => Array.isArray(ens.classifiers) && ens.classifiers.some((cid) => /^fvf_/i.test(String(cid)))
    );
    if (!activeFvfEnsembles.length) return;

    const targetItemId = (elementToItemId && elementToItemId.get(imgEl)) || imgEl.currentSrc || imgEl.src || resolvedSrc;
    if (!targetItemId) return;

    trackElement(targetItemId, imgEl);
    pendingItemIds.delete(targetItemId);
    const entry = itemResults.get(targetItemId) || { modality: 'image', ensembles: {} };
    entry.modality = 'image';

    if (reportedLabel === 1) {
      const targetEns =
        activeFvfEnsembles.find((ens) => Array.isArray(ens.classifiers) && ens.classifiers.includes(summary.modelId)) ||
        activeFvfEnsembles[0];
      const primaryCid =
        (targetEns.classifiers || []).find((cid) => /^fvf_/i.test(String(cid))) ||
        (targetEns.classifiers || [])[0] ||
        'fvf_community_report';
      const rawCategoryName = typeof targetEns.name === 'string' ? targetEns.name.trim() : '';
      const looksLikeRawModelName = !rawCategoryName
        || rawCategoryName === primaryCid
        || rawCategoryName === summary.modelName
        || /^(fvf[\s_-]+)?(heimdall|m[ií]mir|huginn|bragi)\b/i.test(rawCategoryName)
        || /^category\s+\d+$/i.test(rawCategoryName);
      const fallbackSuggestedLabel = /^res_net50_spider$/i.test(primaryCid)
        ? 'Spider'
        : ((targetEns.modality || 'image') === 'text' ? 'AI text' : 'AI generated');
      const displayLabel = looksLikeRawModelName ? fallbackSuggestedLabel : rawCategoryName;
      entry.ensembles[targetEns.id] = {
        score: 1.0,
        label: 'ai',
        displayLabel,
        classifiers: {
          [primaryCid]: {
            score: 1.0,
            label: displayLabel,
            model_name: summary.modelName || displayLabel
          }
        }
      };
    } else {
      activeFvfEnsembles.forEach((ens) => {
        if (entry.ensembles[ens.id]) {
          entry.ensembles[ens.id] = {
            score: 0.0,
            label: 'human',
            displayLabel: 'Human',
            classifiers: {}
          };
        }
      });
    }

    itemResults.set(targetItemId, entry);
    applyStylesForItem(targetItemId);
    updateHoverHandlerAttachment();
  };

  let currentPct = 0;
  const setProgress = (pct) => {
    const clamped = Math.max(0, Math.min(100, Number(pct) || 0));
    if (clamped > currentPct) currentPct = clamped;
    progressFill.style.width = `${currentPct}%`;
  };

  const startSubmit = async (reportedLabel) => {
    aiBtn.disabled = true;
    authBtn.disabled = true;
    aiBtn.hidden = true;
    authBtn.hidden = true;
    minThankYou.hidden = false;
    minThankTitle.textContent = 'Thank you!';
    minThankSub.textContent = 'Uploading report…';

    backdrop.classList.add('is-minimized');
    modal.classList.add('is-minimized');
    footer.hidden = true;

    currentPct = 0;
    progressFill.style.width = '0%';
    progressWrap.hidden = false;
    progressText.hidden = true;
    progressText.textContent = '';

    const reportId = `report_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    setProgress(8);

    if (activeProgressListener && ext.runtime?.onMessage?.removeListener) {
      try { ext.runtime.onMessage.removeListener(activeProgressListener); } catch (_) {}
    }
    if (ext.runtime?.onMessage?.addListener) {
      activeProgressListener = (msg) => {
        if (msg && msg.type === 'COMMUNITY_REPORT_PROGRESS' && msg.reportId === reportId) {
          setProgress(msg.percent);
        }
      };
      ext.runtime.onMessage.addListener(activeProgressListener);
    }

    if (activeProgressTimer) clearInterval(activeProgressTimer);
    activeProgressTimer = setInterval(() => {
      if (currentPct < 88) {
        setProgress(Math.min(88, currentPct + 2));
      }
    }, 250);

    const restoreOnError = (errMsg) => {
      backdrop.classList.remove('is-minimized');
      modal.classList.remove('is-minimized');
      footer.hidden = false;
      minThankYou.hidden = true;
      aiBtn.hidden = false;
      authBtn.hidden = false;
      aiBtn.disabled = false;
      authBtn.disabled = false;
      progressText.hidden = false;
      progressText.style.color = '#f87171';
      progressText.textContent = errMsg;
    };

    try {
      const res = await ext.runtime.sendMessage({
        type: 'SUBMIT_COMMUNITY_REPORT',
        reportId,
        report: {
          sanitizedDataUrl: prep.sanitizedDataUrl,
          width: prep.width,
          height: prep.height,
          reportedLabel,
          modelId: summary.modelId,
          modelName: summary.modelName,
          modelScore: summary.modelScore,
          domain: pageDomain,
          note: ''
        }
      });
      if (activeProgressTimer) {
        clearInterval(activeProgressTimer);
        activeProgressTimer = null;
      }
      if (res && res.ok) {
        try { applyReportedFlagOnPage(reportedLabel); } catch (_) {}
      }
      if (!backdrop.isConnected) return;
      if (res && res.ok) {
        setProgress(100);
        minThankSub.textContent = 'Upload complete';

        modal.classList.remove('fvf-report-thankyou-flash');
        void modal.offsetWidth;
        modal.classList.add('fvf-report-thankyou-flash');

        autoCloseTimer = setTimeout(() => {
          autoCloseTimer = null;
          if (backdrop.isConnected) closeModal();
        }, 500);
      } else {
        restoreOnError(res?.error || 'Upload failed.');
      }
    } catch (err) {
      if (activeProgressTimer) {
        clearInterval(activeProgressTimer);
        activeProgressTimer = null;
      }
      if (!backdrop.isConnected) return;
      restoreOnError(err?.message || 'Upload error.');
    }
  };

  aiBtn.addEventListener('click', () => startSubmit(1));
  authBtn.addEventListener('click', () => startSubmit(0));

  bodyEl.appendChild(mainRow);
  bodyEl.appendChild(progressWrap);
  bodyEl.appendChild(footer);
}

}
