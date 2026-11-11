// Cross-browser extension API abstraction (Chrome/Firefox compatibility)
if (window.__AI_DETECTOR_BOOTSTRAPPED__) {
  console.warn('[AI Detector] Content script already initialized; skipping duplicate init');
} else {
  window.__AI_DETECTOR_BOOTSTRAPPED__ = true;

const ext = typeof browser !== 'undefined' ? browser : chrome;
const VE = globalThis.FVFVisualEffects;
const ENS = globalThis.FVFEnsembles;
const TU = globalThis.FVFTextUtils;

// Unique page context identifier for this page instance (used by Native Host to drop stale items on navigation)
let currentPageId = 'page-' + Math.random().toString(16).slice(2, 10) + '-' + Date.now();

/**
 * USER SETTINGS - Loaded from extension storage
 * These values are synced from the options page
 */
let settings = {
  enabled: true,                  // Master toggle for the extension
  altTextOnly: true,              // If true, only process images with alt text (default: true)
  blurAmount: 4,                  // Blur radius (px) for flagged images
  borderMultiplier: 1,            // Multiplier for border width calculation
  borderColor: '#ff0064',       // Color for image borders and text strikethrough (red)
  miniBatchSize: 16,              // Legacy setting
  classificationDelay: 10,        // Minimal micro-task yield (ms)
  imageCaptureQuality: 1,         // JPEG quality (0.2-1) for image capture
  textBlurAmount: 2,              // Blur radius (px) for flagged text
  textStrikethroughEnabled: true, // Whether to apply strikethrough on AI text
  imageAiThreshold: 0.5,          // Threshold (0-1) for flagging images as AI (0.5 = 50%)
  minImageDimension: 64,          // Skip tiny images/icons below this size (px)
  textAiThreshold: 0.5,           // Threshold (0-1) for flagging text as AI
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
let hoverRafPending = false;     // Throttle pointermove hover updates to one per frame
let lastPointerEvent = null;
let scanPaused = false;            // Pause periodic scans while tab is hidden
let processedImages = new Set(); // Set of img elements already processed
let activeJobIds = new Set();    // Legacy job ID tracking
let itemResults = new Map();     // itemId -> { modality, ensembles: { ensembleId: { score, label, displayLabel } } }
let lastEnsembleSignature = null;
let lastEnsembleStyleSignature = null;
let ensembleByIdCache = null;
let seenTextContent = new Set();
let activeJobTimeouts = new Map();
let hoverPointerHandler = null;
let aiDetectedImageCount = 0;
let lastContextMenuState = null;
const JOB_TRACKING_MS = 5 * 60 * 1000;

/**
 * OBSERVERS - Monitor page for dynamic content & viewport visibility
 */
let mutationObserver = null;     // Watches for new page updates
let intersectionObserver = null; // Watches for images entering viewport (lazy load + P0 priority boost)
let pendingClassification = [];   // Legacy queue
let pendingTextSections = [];    // Legacy queue
let classificationTimeout = null;
let scanIntervalId = null;       // Periodic scan interval
let seenImageUrls = new Set();   // Image URLs already queued or classified
let pendingMutations = [];       // Batched MutationObserver records
let mutationFlushTimer = null;   // Debounce timer for mutation processing
const MUTATION_FLUSH_MS = 120;
const MAX_ITEM_RESULTS = 3000;
const PERIODIC_SCAN_MS = 30000;

let processedTextElements = new Set();

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
      base.threshold = typeof ens.threshold === 'number' ? ens.threshold : 0.5;
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
  if (entry instanceof Set) return Array.from(entry);
  return [entry];
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

function releaseElementTracking(el) {
  if (!el || !(el instanceof Element)) return;

  const itemId = elementToItemId.get(el);
  if (itemId) {
    const tracked = elementMap.get(itemId);
    if (tracked instanceof Set) {
      tracked.delete(el);
      if (tracked.size === 0) elementMap.delete(itemId);
    } else if (tracked === el) {
      elementMap.delete(itemId);
    }
  }

  if (el.tagName?.toLowerCase() === 'img') {
    if (el.classList.contains('ai-detected')) {
      aiDetectedImageCount = Math.max(0, aiDetectedImageCount - 1);
      VE.clearImageStyles(el);
      updateHoverHandlerAttachment();
    }
    unobserveImage(el);
    processedImages.delete(el);
    if (currentHoverEl === el) currentHoverEl = null;
  } else {
    processedTextElements.delete(el);
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

  for (const img of processedImages) {
    if (!img?.isConnected) {
      processedImages.delete(img);
      unobserveImage(img);
    }
  }

  for (const text of processedTextElements) {
    if (!text?.isConnected) {
      processedTextElements.delete(text);
    }
  }

  if (currentHoverEl && !currentHoverEl.isConnected) {
    currentHoverEl = null;
  }

  pruneItemResults();
}

function findAiDetectedUnderPointer(evt) {
  const x = evt.clientX;
  const y = evt.clientY;
  if (typeof x !== 'number' || typeof y !== 'number') return null;

  // 1. Check direct target or closest ancestor
  if (evt.target instanceof Element) {
    const direct = evt.target.closest('.ai-detected');
    if (direct) return direct;
  }

  // 2. Penetrate transparent overlay divs/elements using document.elementsFromPoint(x, y)
  // This penetrates ANY overlay elements, click traps, or buttons sitting in front of the image
  if (typeof document.elementsFromPoint === 'function') {
    try {
      const elements = document.elementsFromPoint(x, y);
      if (elements && elements.length) {
        for (let i = 0; i < elements.length; i++) {
          const el = elements[i];
          if (el && el.classList && el.classList.contains('ai-detected')) {
            return el;
          }
          const parent = el && el.closest && el.closest('.ai-detected');
          if (parent) {
            return parent;
          }
        }
      }
    } catch (_) {}
  }

  // 3. Fallback: Check bounding client rects of tracked processed images
  // (In case an overlay has pointer-events: none, opaque backdrop, or complex stacking context)
  for (const img of processedImages) {
    if (img && img.isConnected && img.classList && img.classList.contains('ai-detected')) {
      try {
        const rect = img.getBoundingClientRect();
        if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
          return img;
        }
      } catch (_) {}
    }
  }

  return null;
}

function handlePointerMoveHover(evt) {
  const candidate = findAiDetectedUnderPointer(evt);

  if (candidate === currentHoverEl) return;
  if (currentHoverEl && currentHoverEl !== candidate) {
    currentHoverEl.classList.remove('fvf-hover-active');
    if (hasResultsForElement(currentHoverEl)) {
      if (currentHoverEl.dataset.aiImageStyle) {
        VE.applyImageInteractionState(currentHoverEl, 'rest', settings);
      } else {
        const blurAmount = parseFloat(currentHoverEl.dataset.aiBlur || settings.blurAmount || 0);
        VE.hideImageAi(currentHoverEl, blurAmount);
      }
    }
  }
  currentHoverEl = candidate;
  if (currentHoverEl && hasResultsForElement(currentHoverEl)) {
    currentHoverEl.classList.add('fvf-hover-active');
    if (currentHoverEl.dataset.aiImageStyle) {
      VE.applyImageInteractionState(currentHoverEl, 'hover', settings);
    } else {
      VE.showImageAi(currentHoverEl);
    }
  }
}

function detachGlobalHoverHandler() {
  if (!hoverHandlerAttached || !hoverPointerHandler) return;
  document.removeEventListener('pointermove', hoverPointerHandler, true);
  hoverHandlerAttached = false;
  hoverPointerHandler = null;
  currentHoverEl = null;
}

function updateHoverHandlerAttachment() {
  if (aiDetectedImageCount > 0 && !hoverHandlerAttached) {
    attachGlobalHoverHandler();
  } else if (aiDetectedImageCount <= 0 && hoverHandlerAttached) {
    detachGlobalHoverHandler();
  }
}

function attachGlobalHoverHandler() {
  if (hoverHandlerAttached) return;
  hoverHandlerAttached = true;
  hoverPointerHandler = (evt) => {
    lastPointerEvent = evt;
    if (hoverRafPending) return;
    hoverRafPending = true;
    requestAnimationFrame(() => {
      hoverRafPending = false;
      if (lastPointerEvent) handlePointerMoveHover(lastPointerEvent);
    });
  };
  document.addEventListener('pointermove', hoverPointerHandler, true);
}

// ============================================================================
// Single-Item Real-Time Streaming Dispatch
// ============================================================================

// Extract image as compact base64 from DOM; returns null if tainted by cross-origin CORS
function imageToBase64(imgElement) {
  try {
    let w = imgElement.naturalWidth || imgElement.width || 0;
    let h = imgElement.naturalHeight || imgElement.height || 0;
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

    const quality = typeof settings.imageCaptureQuality === 'number'
      ? Math.max(0.6, Math.min(0.95, settings.imageCaptureQuality))
      : 0.90;
    return canvas.toDataURL('image/jpeg', quality);
  } catch (_) {
    // Tainted cross-origin canvas: background.js will capture & downscale via extension `<all_urls>` privileges
    return null;
  }
}

function dispatchSingleImageItem(imgObj, modelConfig) {
  if (!imgObj || !imgObj.element) return;
  const itemId = imgObj.src || ('img-unknown-' + Math.random());
  trackElement(itemId, imgObj.element);

  const inViewport = Boolean(imgObj.inViewport ?? isElementInViewport(imgObj.element));

  // If offscreen, keep observing so IntersectionObserver can boost its priority to P0 if user scrolls to it!
  if (!inViewport) {
    observeImage(imgObj.element);
  }

  const base64Data = imageToBase64(imgObj.element);
  const quality = typeof settings.imageCaptureQuality === 'number'
    ? Math.max(0.6, Math.min(0.95, settings.imageCaptureQuality))
    : 0.90;

  if (settings.verboseLogs) {
    console.log(
      `[AI Detector] Streaming image (${inViewport ? 'P0 viewport' : 'P1 offscreen'}, ${base64Data ? 'canvas-b64' : 'bg-cors-fetch'}):`,
      itemId.substring(0, 80)
    );
  }

  ext.runtime.sendMessage({
    type: 'CLASSIFY_SINGLE_ITEM',
    pageId: currentPageId,
    itemId,
    modality: 'image',
    inViewport,
    data: base64Data,
    src: imgObj.src,
    captureQuality: quality,
    width: imgObj.width || null,
    height: imgObj.height || null,
    model: modelConfig
  }).catch((err) => {
    if (settings.verboseLogs) {
      console.warn('[AI Detector] Failed to send single image item:', err);
    }
  });
}

function dispatchSingleTextItem(section, modelConfig) {
  if (!section || !section.text) return;
  const itemId = section.text.substring(0, 500);
  trackElement(itemId, section.element);

  const inViewport = isElementInViewport(section.element);

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
   * Immediately streams discovered images individually to the Native Host,
   * sending in-viewport (P0) images first followed by offscreen (P1) images.
   */
  if (!settings.enabled || !images || !images.length) return;
  const modelConfig = buildModelConfig();

  // Annotate viewport status and sort so in-viewport items are dispatched first
  images.forEach((img) => {
    img.inViewport = isElementInViewport(img.element);
  });
  const sorted = images.slice().sort((a, b) => (b.inViewport ? 1 : 0) - (a.inViewport ? 1 : 0));

  sorted.forEach((img) => {
    dispatchSingleImageItem(img, modelConfig);
  });
}

function scheduleTextClassification(textSections) {
  /**
   * Immediately streams discovered text sections individually to the Native Host.
   */
  if (!settings.enabled || !textSections || !textSections.length) return;
  const modelConfig = buildModelConfig();

  textSections.forEach((section) => {
    dispatchSingleTextItem(section, modelConfig);
  });
}

function processMutationBatch(mutations) {
  if (!settings.enabled || scanPaused) return;

  const newImages = [];
  const newText = [];

  mutations.forEach((mutation) => {
    mutation.removedNodes.forEach((node) => cleanupRemovedSubtree(node));

    if (mutation.type === 'attributes' && mutation.attributeName === 'src') {
      const el = mutation.target;
      if (el.tagName && el.tagName.toLowerCase() === 'img') {
        const oldItemId = elementToItemId.get(el);
        if (oldItemId) {
          elementMap.delete(oldItemId);
          itemResults.delete(oldItemId);
          seenImageUrls.delete(oldItemId);
        }
        processedImages.delete(el);
        const img = processImageElement(el);
        if (img) {
          newImages.push(img);
          if (settings.verboseLogs) {
            console.log(`[AI Detector] Image src changed, re-processing:`, el.src.substring(0, 80));
          }
        }
      }
    }

    mutation.addedNodes.forEach((node) => {
      if (node.nodeType !== 1) return;

      if (node.tagName && node.tagName.toLowerCase() === 'img') {
        const img = processImageElement(node);
        if (img) newImages.push(img);
        return;
      }

      if (!node.querySelectorAll) return;

      try {
        if (node.tagName && TU.isTextTagName(node.tagName.toLowerCase())) {
          const text = processTextElement(node);
          if (text) newText.push(text);
        }

        if (node.querySelector('img')) {
          node.querySelectorAll('img').forEach((el) => {
            const img = processImageElement(el);
            if (img) newImages.push(img);
          });
        }

        if (node.querySelector(TU.TEXT_SELECTORS)) {
          node.querySelectorAll(TU.TEXT_SELECTORS).forEach((el) => {
            const text = processTextElement(el);
            if (text) newText.push(text);
          });
        }
      } catch (_) {}
    });
  });

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
    attributeFilter: ['src'],
    characterData: false
  });
}

// Detect lazy-loaded images entering viewport AND boost already-queued offscreen images to P0
function initIntersectionObserver() {
  if (intersectionObserver) return;

  intersectionObserver = new IntersectionObserver((entries) => {
    const newImages = [];
    const boostItemIds = [];

    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      if (el.tagName && el.tagName.toLowerCase() === 'img') {
        // Case 1: Image was already queued offscreen (P1) and just scrolled into view -> boost to P0!
        if (processedImages.has(el)) {
          const trackedId = elementToItemId.get(el) || el.src;
          if (trackedId && !itemResults.has(trackedId)) {
            boostItemIds.push(trackedId);
          }
          unobserveImage(el);
          return;
        }

        // Case 2: Lazy-loaded image now has dimensions in viewport -> process & dispatch as P0
        if (el.naturalWidth > 0 && el.naturalHeight > 0) {
          const img = processImageElement(el);
          if (img) {
            img.inViewport = true;
            newImages.push(img);
            if (settings.verboseLogs) {
              console.log(`[AI Detector] IntersectionObserver: Lazy image entered viewport:`, el.src.substring(0, 80));
            }
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

function processTextElement(el) {
  if (!el || !(el instanceof Element)) return null;
  if (processedTextElements.has(el)) return null;

  const text = TU.normalizeElementText(el);
  if (!text || text.length < settings.minTextLength) return null;

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

function processImageElement(el, isLazy = false) {
  if (!el || el.tagName.toLowerCase() !== 'img') return null;
  if (processedImages.has(el)) return null;

  const src = el.src || el.getAttribute('src');
  if (!src) return null;

  const nw = el.naturalWidth || 0;
  const nh = el.naturalHeight || 0;
  if (nw === 0 || nh === 0) {
    if (!el.dataset.aiLoadListenerAttached) {
      el.dataset.aiLoadListenerAttached = 'true';
      el.addEventListener('load', () => {
        if (settings.verboseLogs) {
          console.log(`[AI Detector] Image load event:`, el.src.substring(0, 80));
        }
        processedImages.delete(el);
        unobserveImage(el);
        const img = processImageElement(el);
        if (img) scheduleClassification([img]);
      }, { once: true });
    }
    observeImage(el);
    return null;
  }

  const displayW = el.clientWidth || el.width || 0;
  const displayH = el.clientHeight || el.height || 0;
  const effectiveW = Math.max(nw, displayW);
  const effectiveH = Math.max(nh, displayH);
  const minDim = Number(settings.minImageDimension || 64);
  if (effectiveW < minDim || effectiveH < minDim) {
    return null;
  }

  if (settings.altTextOnly) {
    const alt = el.alt || el.getAttribute('alt') || '';
    const isExemptAltTextRule =
      src.includes('i.ytimg.com/vi/') ||
      src.includes('i.ytimg.com/vi_webp/') ||
      src.includes('encrypted-tbn') ||
      src.includes('gstatic.com');

    if (!isExemptAltTextRule && (!alt || !alt.trim().length)) {
      return null;
    }
  }

  processedImages.add(el);
  unobserveImage(el);

  if (seenImageUrls.has(src)) {
    trackElement(src, el);
    if (itemResults.has(src)) {
      applyStylesForItem(src);
    }
    return null;
  }

  if (itemResults.has(src)) {
    seenImageUrls.add(src);
    trackElement(src, el);
    applyStylesForItem(src);
    return null;
  }

  seenImageUrls.add(src);

  return {
    src,
    alt: el.alt || '',
    width: el.naturalWidth || el.width || null,
    height: el.naturalHeight || el.height || null,
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
  }, 3000);
}

function applyStylesForItem(itemId) {
  const entry = itemResults.get(itemId);
  if (!entry) return;
  const elements = getElementsForItem(itemId).filter((el) => el && el.isConnected);
  if (!elements.length) {
    if (entry.modality === 'image') {
      const liveImgs = Array.from(document.querySelectorAll('img')).filter((img) => img.src === itemId);
      if (liveImgs.length) {
        liveImgs.forEach((img) => trackElement(itemId, img));
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
  if (!response || !Array.isArray(response.results)) return;

  if (settings.verboseLogs && Array.isArray(response.errors) && response.errors.length > 0) {
    response.errors.filter((e) => e.type === 'info').forEach((log) => console.log(log.message));
  }

  response.results.forEach((result) => {
    if (!result || !result.id) return;
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

    // Unobserve once classified
    getElementsForItem(result.id).forEach((el) => unobserveImage(el));

    const elements = getElementsForItem(result.id);
    if (elements.some((el) => el && el.isConnected) || result.modality === 'image') {
      applyStylesForItem(result.id);
    }
  });

  try {
    queueClassifierTimings(response.results);
  } catch (e) {
    if (settings.verboseLogs) console.warn('[AI Detector] queueClassifierTimings failed to start', e);
  }
}

function extractLongText() {
  try {
    const nodes = Array.from(document.querySelectorAll(TU.TEXT_SELECTORS));
    const seen = new Set();
    const sections = [];

    nodes.forEach((el) => {
      if (TU.hasNestedTextCandidate(el)) return;
      const text = TU.normalizeElementText(el);
      if (!text || text.length < settings.minTextLength) return;
      if (seen.has(text)) return;
      seen.add(text);
      sections.push(TU.buildTextSection(el, sections.length, text));
    });

    return sections;
  } catch (e) {
    console.error('[Extension] Error extracting long text:', e);
    return [];
  }
}

async function loadSettings() {
  try {
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    if (!storageArea || !storageArea.get) {
      console.warn('[Extension] Storage API not available');
      return;
    }

    const res = await storageArea.get([
      'enabled',
      'altTextOnly',
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
    ]);

    if (res && typeof res.enabled === 'boolean') settings.enabled = res.enabled;
    if (res && typeof res.altTextOnly === 'boolean') settings.altTextOnly = res.altTextOnly;
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
    if (res && typeof res.minTextLength === 'number') settings.minTextLength = res.minTextLength;
    if (res && typeof res.minImageDimension === 'number') settings.minImageDimension = res.minImageDimension;
    if (res && Array.isArray(res.ensembleConfigsV2)) settings.ensembles = res.ensembleConfigsV2;
    updateEnsembleSignatures();
  } catch (e) {
    console.warn('[Extension] Error loading settings:', e);
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

function initAutoExtract() {
  bindPageLifecycleCleanup();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', async () => {
      await loadSettings();
      startDynamicDetection();
      performInitialExtraction();
    });
  } else {
    loadSettings().then(() => {
      startDynamicDetection();
      performInitialExtraction();
    });
  }
}

function startDynamicDetection() {
  if (document.documentElement && document.documentElement.dataset.fvfOnboarding === 'true') {
    return;
  }
  if (!settings.enabled) return;
  initMutationObserver();
  initIntersectionObserver();

  if (!document.__aiDetectorVisibilityBound) {
    document.__aiDetectorVisibilityBound = true;
    document.addEventListener('visibilitychange', () => {
      scanPaused = document.hidden;
      if (scanPaused && mutationFlushTimer) {
        clearTimeout(mutationFlushTimer);
        mutationFlushTimer = null;
        pendingMutations = [];
      }
    });
  }

  if (scanIntervalId) clearInterval(scanIntervalId);
  scanIntervalId = setInterval(() => {
    if (scanPaused) return;
    cleanupStaleEntries();
    const newImages = scanAllImages();
    if (newImages.length > 0) {
      scheduleClassification(newImages);
    }
  }, PERIODIC_SCAN_MS);
}

function performInitialExtraction() {
  if (document.documentElement && document.documentElement.dataset.fvfOnboarding === 'true') {
    return;
  }
  if (!settings.enabled) {
    console.log('[AI Detector] Disabled in options; skipping scan');
    return;
  }
  const initialImages = scanAllImages();
  const initialTextSections = extractLongText();

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
  'GET_OFFICIAL_CATALOG',
  'DOWNLOAD_OFFICIAL_MODEL',
  'AUTO_SETUP_CATEGORIES',
  'SAVE_ONBOARDING_ENSEMBLES'
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

function getTextLabelFromElement(element, fallback = 'AI text') {
  if (!element?.title) return fallback;
  const match = element.title.match(/^(.+?)\s+\(/);
  return match ? match[1] : fallback;
}

function applyManualBlurStyles(itemId, element, manualEnsemble) {
  trackElement(itemId, element);
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
  lastContextMenuTarget = evt.target;

  if (evt.target && evt.target.tagName && evt.target.tagName.toLowerCase() === 'img') {
    const hasVisibleAiStyling = evt.target.classList.contains('ai-detected');
    let label = 'tag';

    if (hasVisibleAiStyling) {
      label = VE.getImageBadgeLabel(evt.target, 'tag');
    }

    sendContextMenuUpdate({
      hasAiTags: hasVisibleAiStyling,
      label: label,
      modality: 'image'
    });
  } else if (evt.target && evt.target.classList && evt.target.classList.contains('ai-detected-text')) {
    sendContextMenuUpdate({
      hasAiTags: true,
      label: getTextLabelFromElement(evt.target),
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
}, true);

// Listen for streamed classification results
if (ext && ext.runtime && ext.runtime.onMessage) {
  ext.runtime.onMessage.addListener((message) => {
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
      return Promise.resolve({ success: true });
    }

    if (message.type === 'CANCEL_ALL_CLASSIFICATIONS') {
      activeJobIds.clear();
      pendingClassification = [];
      pendingTextSections = [];
      if (classificationTimeout) {
        clearTimeout(classificationTimeout);
        classificationTimeout = null;
      }

      ext.runtime.sendMessage({
        type: 'CANCEL_ALL'
      }).catch(() => {});

      console.log('[AI Detector] All classifications cancelled');
      return Promise.resolve({ success: true });
    }

    if (message.type === 'TOGGLE_BLUR') {
      const srcUrl = message.srcUrl;

      if (srcUrl) {
        const images = Array.from(document.querySelectorAll('img'));
        const targetImage = images.find((img) => img.src === srcUrl);
        if (targetImage) {
          const itemId = targetImage.src;
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
            return Promise.resolve({ hasAiTags: false, label: badgeLabel, modality: 'image' });
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
                  applyBorder: true,
                  borderMultiplier: (settings.borderMultiplier ?? 1) * 2,
                  borderColor: '#808080',
                  borderMode: 'hover',
                  applyBadge: true,
                  badgeMode: 'hover'
                }
              }
            });
            return Promise.resolve({ hasAiTags: true, label: 'Blur', modality: 'image' });
          }
        }
      }

      if (!srcUrl && lastContextMenuTarget) {
        const targetText = lastContextMenuTarget;
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
            return Promise.resolve({ hasAiTags: false, label: label, modality: 'text' });
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
            return Promise.resolve({ hasAiTags: true, label: 'Blur', modality: 'text' });
          }
        }
      }
      return Promise.resolve({ hasAiTags: false, modality: 'image' });
    }

    return false;
  });
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
  if (timingFlushTimer) {
    clearTimeout(timingFlushTimer);
    timingFlushTimer = null;
  }
  pendingTimingResults = [];
  detachGlobalHoverHandler();
}

function resetState() {
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
  processedImages.clear();
  processedTextElements.clear();
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
  aiDetectedImageCount = 0;
  currentHoverEl = null;
  lastContextMenuState = null;
  if (classificationTimeout) {
    clearTimeout(classificationTimeout);
    classificationTimeout = null;
  }
}

async function reloadSettingsAndRescan() {
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
    const classificationKeys = ['enabled', 'altTextOnly', 'minTextLength'];
    const visualKeys = ['blurAmount', 'borderMultiplier', 'borderColor', 'textBlurAmount', 'textStrikethroughEnabled', 'imageAiThreshold', 'textAiThreshold'];
    const performanceKeys = ['classificationDelay', 'verboseLogs', 'lazyLoad', 'imageCaptureQuality', 'miniBatchSize', 'minImageDimension'];

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
      const classifierChanged = prevSig && newSig && prevSig !== newSig;
      const styleChanged = !prevStyleSig || !newStyleSig || prevStyleSig !== newStyleSig;

      if (classifierChanged) {
        reloadSettingsAndRescan();
        return;
      }

      if (styleChanged) {
        applyLatestVisualSettings({ recalculateScores: true });
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

}
