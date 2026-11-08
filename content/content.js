// Cross-browser extension API abstraction (Chrome/Firefox compatibility)
if (window.__AI_DETECTOR_BOOTSTRAPPED__) {
  console.warn('[AI Detector] Content script already initialized; skipping duplicate init');
} else {
  window.__AI_DETECTOR_BOOTSTRAPPED__ = true;

const ext = typeof browser !== 'undefined' ? browser : chrome;
const VE = globalThis.FVFVisualEffects;
const ENS = globalThis.FVFEnsembles;
const TU = globalThis.FVFTextUtils;


/**
 * USER SETTINGS - Loaded from extension storage
 * These values are synced from the options page
 */
let settings = {
  enabled: true,                  // Master toggle for the extension
  altTextOnly: true,              // If true, only process images with alt text
  blurAmount: 4,                  // Blur radius (px) for flagged images
  borderMultiplier: 1,            // Multiplier for border width calculation
  borderColor: '#ff0064',       // Color for image borders and text strikethrough (red)
  miniBatchSize: 16,              // Batch size for native host progressive streaming mini-batches
  classificationDelay: 100,       // Debounce delay (ms) before batching and sending for classification
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
let cachedImages = [];           // Array of {src, alt, width, height, element, lazy}
let cachedTextSections = [];     // Array of {index, selector, tag, length, text, element}
let elementMap = new Map();      // Maps itemId to DOM elements
let elementToItemId = new WeakMap(); // Reverse lookup: DOM element -> itemId
let hoverHandlerAttached = false; // Tracks weather a global hover handler is actuive already
let currentHoverEl = null;       // tracks the current hovered AI-detected element
let hoverRafPending = false;     // Throttle pointermove hover updates to one per frame
let lastPointerEvent = null;
let scanPaused = false;            // Pause periodic scans while tab is hidden
let processedImages = new Set(); // Set of img elements already added to cachedImages
let activeJobIds = new Set(); // Track in-flight job IDs for chunked results
let itemResults = new Map();  // itemId -> { modality, ensembles: { ensembleId: { score, label, displayLabel } } }
let lastEnsembleSignature = null;
let lastEnsembleStyleSignature = null;
let ensembleByIdCache = null;
let seenTextContent = new Set();
let activeJobTimeouts = new Map();
let hoverPointerHandler = null;
let aiDetectedImageCount = 0;
let lastContextMenuState = null;
const JOB_TRACKING_MS = 5 * 60 * 1000;
const CHUNK_SEND_CONCURRENCY = 2;

/**
 * OBSERVERS - Monitor page for dynamic content
 */
let mutationObserver = null;     // Watches for new page updates
let intersectionObserver = null; // Watches for images entering viewport
let pendingClassification = [];   // Queue of images waiting to be classified
let pendingTextSections = [];    // Queue of text sections waiting to be classified
let classificationTimeout = null; // Debounce timer for batching classifications
let scanIntervalId = null;       // Periodic scan interval
let classificationInFlight = false; // Prevent overlapping classification requests
let seenImageUrls = new Set();   // Image URLs already queued or classified (by URL, not element)
let pendingMutations = [];       // Batched MutationObserver records
let mutationFlushTimer = null;   // Debounce timer for mutation processing
const MUTATION_FLUSH_MS = 300;
const MAX_ITEM_RESULTS = 3000;
const PERIODIC_SCAN_MS = 30000;

/**
 * ID SYSTEM - Uses content-based IDs that persist across page reloads
 * Images: Use their URL as the ID
 * Text: Use first 500 characters as the ID
 * The native host will remember classifications, browser just re-discovers content
 */
let processedTextElements = new Set(); // Set of text elements already added to cachedTextSections

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

function computeEnsembleSignature(ensembles, includeStyles) {
  if (!Array.isArray(ensembles)) return '';
  const normalized = ensembles.map((ens) => {
    const base = {
      id: ens.id || '',
      modality: ens.modality || '',
      enabled: ens.enabled !== false,
      classifiers: Array.isArray(ens.classifiers) ? [...ens.classifiers].sort() : [],
      // weights are considered a visual-level setting and are included
      // only when `includeStyles` is true below
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
    // Keep itemResults so styles can be re-applied if the element re-enters the DOM
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
  } catch (_) {
    // querySelectorAll may fail on some elements
  }
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

function handlePointerMoveHover(evt) {
  const candidate = evt.target instanceof Element
    ? evt.target.closest('.ai-detected')
    : null;

  if (candidate === currentHoverEl) return;
  if (currentHoverEl && currentHoverEl !== candidate) {
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

function scheduleClassification(images) {
  /**
   * Queue new images for debounced classification
   */
  if (!settings.enabled) return;
  if (!images || !images.length) return;
  images.forEach(img => pendingClassification.push(img));
  triggerDebouncedClassification();
}

function scheduleTextClassification(textSections) {
  /**
   * Queue new text sections for debounced classification
   */
  if (!settings.enabled) return;
  if (!textSections || !textSections.length) return;
  textSections.forEach(text => pendingTextSections.push(text));
  triggerDebouncedClassification();
}

function triggerDebouncedClassification() {
  /**
   * Batches and detected content and sends them for classification
   */
  if (!settings.enabled) return;
  
  // Clear existing timeout
  if (classificationTimeout) clearTimeout(classificationTimeout);
  
  // Debounce: wait for configured delay to batch mutations/loads
  classificationTimeout = setTimeout(() => {
    if (pendingClassification.length === 0 && pendingTextSections.length === 0) return;
    
    if (pendingClassification.length > 0) {
      console.log(`[AI Detector] Scanning ${pendingClassification.length} new image(s)`);
    }
    if (pendingTextSections.length > 0) {
      console.log(`[AI Detector] Scanning ${pendingTextSections.length} new text section(s)`);
    }
    
    // Add pending images to cache
    pendingClassification.forEach(img => {
      cachedImages.push(img);
    });
    
    // Add pending text sections to cache
    pendingTextSections.forEach(text => {
      cachedTextSections.push(text);
    });
    
    pendingClassification = [];
    pendingTextSections = [];
    requestClassification();
  }, settings.classificationDelay);
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
      } catch (_) {
        // querySelectorAll may fail on some elements
      }
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

// Detect lazy-loaded images entering viewport using IntersectionObserver
function initIntersectionObserver() {
  if (intersectionObserver) return;

  intersectionObserver = new IntersectionObserver((entries) => {
    const newImages = [];

    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      if (el.tagName && el.tagName.toLowerCase() === 'img') {
        if (el.naturalWidth > 0 && el.naturalHeight > 0) {
          const img = processImageElement(el);
          if (img) {
            newImages.push(img);
            if (settings.verboseLogs) {
              console.log(`[AI Detector] IntersectionObserver: Image entered viewport and is loaded:`, el.src.substring(0, 80));
            }
          }
        }
      }
    });

    if (newImages.length > 0) {
      scheduleClassification(newImages);
    }
  }, {
    rootMargin: '200px'
  });
}

// Process and extract text from a text element if not already seen
function processTextElement(el) {
  if (!el || !(el instanceof Element)) return null;
  
  // Skip if already processed
  if (processedTextElements.has(el)) return null;
  
  const text = TU.normalizeElementText(el);
  
  // Skip if too short
  if (!text || text.length < settings.minTextLength) return null;
  
  // Skip if we've already seen this exact text
  if (seenTextContent.has(text)) return null;
  for (let section of cachedTextSections) {
    if (section.text === text) return null;
  }
  for (let section of pendingTextSections) {
    if (section.text === text) return null;
  }

  seenTextContent.add(text);
  
  // Mark as processed
  processedTextElements.add(el);
  
  const textPreview = text.substring(0, 500);
  if (itemResults.has(textPreview)) {
    trackElement(textPreview, el);
    applyStylesForItem(textPreview);
    return null;
  }
  
  return TU.buildTextSection(el, cachedTextSections.length + pendingTextSections.length, text);
}

// Return the desired properties of an image element if it hasn't been seen before
function processImageElement(el, isLazy = false) {
  if (!el || el.tagName.toLowerCase() !== 'img') return null;
  
  // Skip if already processed
  if (processedImages.has(el)) return null;
  
  // Skip if no src or src is invalid
  const src = el.src || el.getAttribute('src');
  if (!src) return null;

  // Check if image has loaded
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

  // Skip tiny/placeholder images and UI icons.
  const displayW = el.clientWidth || el.width || 0;
  const displayH = el.clientHeight || el.height || 0;
  const effectiveW = Math.max(nw, displayW);
  const effectiveH = Math.max(nh, displayH);
  const minDim = Number(settings.minImageDimension || 64);
  // Skip if either dimension is below minimum (catches 1x1 tracking pixels)
  if (effectiveW < minDim || effectiveH < minDim) {
    return null;
  }
  
  // Check alt-text filter with exceptions
  if (settings.altTextOnly) {
    const alt = el.alt || el.getAttribute('alt') || '';
    
    // Bypass alt-text requirement for:
    // - YouTube thumbnails: i.ytimg.com/vi/VIDEO_ID/...
    // - Google Images: encrypted-tbn*.gstatic.com (Google's thumbnail CDN)
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
    if (itemResults.has(src)) {
      trackElement(src, el);
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
  
  if (settings.verboseLogs) {
    console.log(`[AI Detector] Processing image (${nw}x${nh}):`, src.substring(0, 80));
  }

  seenImageUrls.add(src);
  
  const img = {
    src: src,
    alt: el.alt || '',
    width: el.naturalWidth || el.width || null,
    height: el.naturalHeight || el.height || null,
    element: el,
    lazy: isLazy
  };

  return img;
}

// Scan all current images for new ones (periodic scan + initial)
function scanAllImages() {
  const images = Array.from(document.querySelectorAll('img'));
  const newImages = [];
  
  images.forEach((el) => {
    const img = processImageElement(el);
    if (img) newImages.push(img);
  });
  
  if (settings.verboseLogs && newImages.length > 0) {
    console.log(`[AI Detector] scanAllImages found ${newImages.length} new images (total on page: ${images.length}, already processed: ${processedImages.size})`);
  }
  
  return newImages;
}

// Try to extract image as base64 from DOM to avoid re-downloading
function imageToBase64(imgElement) {
  try {
    // Use natural dimensions so the classifier receives the full-resolution image.
    // Pre-resizing on canvas before JPEG encoding introduces block artifacts that
    // the artifact branch misidentifies as AI-generation artifacts.
    const w = imgElement.naturalWidth || imgElement.width || 0;
    const h = imgElement.naturalHeight || imgElement.height || 0;
    if (!w || !h) return null;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0, w, h);
    const base64 = canvas.toDataURL('image/png');  // Lossless: no JPEG block artifacts
    
    // Skip if base64 is suspiciously small (likely a low-quality thumbnail)
    // Typical full images are 80KB+, thumbnails are 20-40KB
    // Return null to skip this image for now - it will be re-processed when full version loads
    if (base64 && base64.length < 60000) {  // ~45KB threshold
      if (settings.verboseLogs) {
        console.log(`[AI Detector] Image too small (${Math.round(base64.length/1024)}KB), waiting for full resolution:`, imgElement.src.substring(0, 80));
      }
      return 'TOO_SMALL';  // Special marker to indicate we should skip this image
    }
    
    return base64;
  } catch (e) {
    // CORS or other canvas errors - will fall back to URL fetch
    return null;
  }
}

// Build classification batches - splits into multiple requests if needed to stay under message size limit
function buildClassificationBatches(images = cachedImages, textSections = cachedTextSections) {
  const batches = [];
  const MAX_MESSAGE_SIZE = 6000000;  // 6MB per chunk (prevents browser Native Messaging pipe disconnect on image-heavy pages)
  const MAX_ITEMS_PER_CHUNK = 16;    // Keep chunks compact so streaming starts immediately
  let currentBatch = [];
  let currentSize = 1000;  // Start with overhead estimate
  
  // DO NOT clear elementMap - preserve mappings from previous batches
  // Only add new entries for images we haven't seen yet

  images.forEach((img) => {
    // Use URL as the ID (persistent across page reloads)
    const itemId = img.src || 'img-unknown-' + Math.random();
    
    // Update elementMap to track this element
    trackElement(itemId, img.element);
    
    // Try to get base64 from DOM first (avoids network fetch)
    let imageData = null;
    let itemSize = 500;  // Rough estimate for JSON overhead per item
    
    try {
      imageData = imageToBase64(img.element);
      if (imageData === 'TOO_SMALL') {
        processedImages.delete(img.element);
        seenImageUrls.delete(itemId);
        if (settings.verboseLogs) {
          console.log(`[AI Detector] Skipping thumbnail ${itemId}, will retry when full image loads`);
        }
        return;  // Skip this image
      }
      if (imageData) {
        itemSize += imageData.length;
      }
    } catch (e) {
      imageData = null;
    }
    
    // If adding this item would exceed limit, save current batch and start new one
    if ((currentSize + itemSize > MAX_MESSAGE_SIZE || currentBatch.length >= MAX_ITEMS_PER_CHUNK) && currentBatch.length > 0) {
      batches.push(currentBatch);
      currentBatch = [];
      currentSize = 1000;
    }
    
    const item = {
      id: itemId,
      modality: 'image',
      source: 'img',
      url: imageData || img.src || null,  // Base64 if available, else URL
      data: null,
      mime: imageData ? 'image/png' : null,
      width: img.width || null,
      height: img.height || null
    };
    
    if (settings.verboseLogs) {
      const isBase64 = imageData != null;
      const urlPreview = isBase64 ? 'base64' : (img.src || 'null').substring(0, 80);
      console.log(`[AI Detector] Batching ${itemId}: ${img.width}x${img.height}, ${isBase64 ? 'base64 (' + Math.round(imageData.length/1024) + 'KB)' : 'URL: ' + urlPreview}`);
    }
    
    currentBatch.push(item);
    
    currentSize += itemSize;
  });

  // Add text sections to FIRST batch only (they're small)
  textSections.forEach((section) => {
    // Use first 500 characters of text as the ID (persistent across page reloads)
    const textPreview = section.text.substring(0, 500);
    const itemId = textPreview;
    
    // Update elementMap to track this element
    trackElement(itemId, section.element);
    
    currentBatch.push({
      id: itemId,
      modality: 'text',
      text: section.text,
      length: section.text.length,
      context: section.selector || section.tag
    });
  });

  if (currentBatch.length > 0) {
    batches.push(currentBatch);
  }
  
  if (settings.verboseLogs) {
    if (batches.length > 1) {
      console.log(`[AI Detector] Split into ${batches.length} batches (${batches.map(b => Math.round(JSON.stringify(b).length / 1024) + 'KB').join(', ')})`);
    } else {
      console.log(`[AI Detector] Message size: ~${Math.round(currentSize / 1024)}KB`);
    }
  }
  
  return batches;
}

// Send items to the background script for native classification
async function sendClassificationChunks(jobId, batches, modelConfig) {
  let nextIndex = 0;

  async function sendOne() {
    while (nextIndex < batches.length) {
      const idx = nextIndex++;
      await ext.runtime.sendMessage({
        type: 'CLASSIFY_JOB_CHUNK',
        jobId,
        chunkIndex: idx,
        totalChunks: batches.length,
        items: batches[idx],
        model: modelConfig
      });
    }
  }

  const workers = Math.min(CHUNK_SEND_CONCURRENCY, batches.length);
  await Promise.all(Array.from({ length: workers }, () => sendOne()));
}

function trackActiveJob(jobId) {
  activeJobIds.add(jobId);
  if (activeJobTimeouts.has(jobId)) {
    clearTimeout(activeJobTimeouts.get(jobId));
  }
  activeJobTimeouts.set(jobId, setTimeout(() => {
    activeJobIds.delete(jobId);
    activeJobTimeouts.delete(jobId);
    if (settings.verboseLogs) {
      console.warn(`[AI Detector] Job ${jobId} tracking expired before completion`);
    }
  }, JOB_TRACKING_MS));
}

function completeActiveJob(jobId) {
  activeJobIds.delete(jobId);
  if (activeJobTimeouts.has(jobId)) {
    clearTimeout(activeJobTimeouts.get(jobId));
    activeJobTimeouts.delete(jobId);
  }
}

async function requestClassification() {
  if (classificationInFlight) return;

  const imagesToSend = cachedImages.slice();
  const textToSend = cachedTextSections.slice();

  // Build batches that respect message size limits from cached images/text
  const batches = buildClassificationBatches(imagesToSend, textToSend);
  if (!batches || !batches.length) return;

  // Clear caches now; new items will accumulate separately while request is in-flight
  cachedImages = [];
  cachedTextSections = [];
  classificationInFlight = true;

  try {
    const jobId = `job-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    trackActiveJob(jobId);

    const t0 = performance.now();
    if (settings.verboseLogs) console.log(`[AI Detector] Sending classification job ${jobId} with lazyLoad=${settings.lazyLoad}`);

    const ensemblesByModality = getEnsemblesByModality();
    const modelConfig = {
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
      miniBatchSize: settings.miniBatchSize || 1000,
      streamResults: true
    };

    await sendClassificationChunks(jobId, batches, modelConfig);

    const totalMs = Math.round(performance.now() - t0);
    const totalSecs = (totalMs / 1000).toFixed(2);
    console.log(`[AI Detector] Sent ${batches.length} chunk(s) for job ${jobId} in %c${totalSecs}s%c`, 'font-weight: bold', '');
  } catch (err) {
    console.warn('[AI Detector] Classification request error:', err);
    // Re-queue items on failure so they aren't lost
    cachedImages = imagesToSend.concat(cachedImages);
    cachedTextSections = textToSend.concat(cachedTextSections);
  } finally {
    classificationInFlight = false;
    // If new items arrived while we were in-flight, send them now
    if (cachedImages.length > 0 || cachedTextSections.length > 0) {
      requestClassification();
    }
  }
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

    // Collect samples per classifier
    const samples = new Map();
    results.forEach((result) => {
      if (!result) return;
      const duration = typeof result.durationMs === 'number' ? result.durationMs : (result.duration || null);
      if (!Number.isFinite(duration) || duration <= 0) return;

      // If classifiers map exists, use its keys; else, attempt to attribute from ensembleId
      const clsMap = result.classifiers && typeof result.classifiers === 'object' ? Object.keys(result.classifiers) : [];
      if (clsMap.length === 1) {
        const id = clsMap[0];
        if (!samples.has(id)) samples.set(id, []);
        samples.get(id).push(duration);
      } else if (clsMap.length > 1) {
        // split evenly across participating classifiers
        const per = duration / clsMap.length;
        clsMap.forEach((id) => {
          if (!samples.has(id)) samples.set(id, []);
          samples.get(id).push(per);
        });
      } else {
        // No classifiers map (legacy/ensemble-level only). Try to use ensembleId when single-classifier in config
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

    // Merge samples into timings with simple running average and outlier filtering
    samples.forEach((vals, classifierId) => {
      if (!vals || !vals.length) return;
      for (const sample of vals) {
        const prev = timings[classifierId] || { avgMs: 0, count: 0, lastMs: 0, updatedAt: 0 };

        // If we have history, treat very large samples as load-time outliers and skip them
        if (prev.count > 0 && sample > (prev.avgMs * 5 + 1e-6)) {
          // skip this sample assuming it includes model load
          continue;
        }

        // If no history and sample is extremely large (>10s), assume it's a load and skip
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
    // Fallback: if DOM element was recreated with same src on a dynamic page, re-link it
    if (entry.modality === 'image') {
      const liveImgs = Array.from(document.querySelectorAll('img')).filter((img) => img.src === itemId);
      if (liveImgs.length) {
        liveImgs.forEach((img) => trackElement(itemId, img));
        elements.push(...liveImgs);
      }
    }
    if (!elements.length) {
      if (settings.verboseLogs) {
        console.log(`[AI Detector] Element ${itemId} no longer in DOM, skipping style application`);
      }
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
    if (data.score >= threshold) {
      // For single-classifier ensembles, prefer the classifier's label; otherwise use ensemble name
      let displayLabel = 'AI generated';
      const isSingleClassifier = Array.isArray(cfg.classifiers) && cfg.classifiers.length === 1;
      if (isSingleClassifier) {
        // Get label from the classifier's detail object
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

  let appliedCount = 0;
  let skippedCount = 0;
  
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
    
    // Check if element still exists before applying styles
    const elements = getElementsForItem(result.id);
    if (elements.some((el) => el && el.isConnected) || result.modality === 'image') {
      applyStylesForItem(result.id);
      appliedCount++;
    } else {
      skippedCount++;
    }
  });
  // Update classifier timing stats (async - don't block rendering)
  try {
    queueClassifierTimings(response.results);
  } catch (e) {
    if (settings.verboseLogs) console.warn('[AI Detector] queueClassifierTimings failed to start', e);
  }
  
  if (settings.verboseLogs && skippedCount > 0) {
    console.log(`[AI Detector] Applied ${appliedCount} results, skipped ${skippedCount} (elements no longer in DOM)`);
  }
}

// Extract long text sections from the page (>=250 chars)
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

// Load settings from browser storage
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

// Auto-extract data when page loads
function initAutoExtract() {
  // Wait for DOM to be fully loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', async () => {
      await loadSettings();
      performInitialExtraction();
      startDynamicDetection();
    });
  } else {
    loadSettings().then(() => {
      performInitialExtraction();
      startDynamicDetection();
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
  // Scan all images on page load
  const initialImages = scanAllImages();
  const initialTextSections = extractLongText();

  // Only log if content was found
  if (initialImages.length > 0 || initialTextSections.length > 0) {
    console.log(`[AI Detector] Scanning ${initialImages.length} images, ${initialTextSections.length} text sections`);
  }

  // Debounce initial classification to allow late-arriving items to join the first batch
  if (initialImages.length > 0) {
    scheduleClassification(initialImages);
  }
  if (initialTextSections.length > 0) {
    scheduleTextClassification(initialTextSections);
  }
}

// Initialize
initAutoExtract();

// Onboarding Web-to-Extension Bridge (allows the GitHub Pages /docs onboarding app to communicate with the extension)
const ALLOWED_ONBOARDING_ACTIONS = new Set([
  'PING',
  'CHECK_NATIVE_HOST',
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

// Track last right-clicked element for context menu
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

// Update context menu title when user right-clicks on an image or text
document.addEventListener('contextmenu', (evt) => {
  lastContextMenuTarget = evt.target;
  
  // Check if it's an image
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
  }
  // Check if it's text with ai-detected-text class
  else if (evt.target && evt.target.classList && evt.target.classList.contains('ai-detected-text')) {
    sendContextMenuUpdate({
      hasAiTags: true,
      label: getTextLabelFromElement(evt.target),
      modality: 'text'
    });
  }
  // Check if right-clicking on normal text (for adding blur)
  else if (evt.target && evt.target.tagName) {
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

    if (message.type === 'CLASSIFY_RESULT_CHUNK') {
      const payload = message.payload;
      if (payload && payload.jobId && !activeJobIds.has(payload.jobId)) {
        return false;
      }
      applyResultChunk(payload);
      return false;
    }

    if (message.type === 'CLASSIFY_JOB_COMPLETE') {
      const payload = message.payload;
      if (payload && payload.jobId) completeActiveJob(payload.jobId);
      return false;
    }

    if (message.type === 'RECLASSIFY_ALL') {
      reloadSettingsAndRescan();
      return Promise.resolve({ success: true });
    }

    if (message.type === 'CANCEL_ALL_CLASSIFICATIONS') {
      // Cancel all in-flight classifications (both browser-side and native host-side)
      const jobCount = activeJobIds.size;
      activeJobIds.clear();
      classificationInFlight = false;
      pendingClassification = [];
      pendingTextSections = [];
      if (classificationTimeout) {
        clearTimeout(classificationTimeout);
        classificationTimeout = null;
      }
      
      // Send cancel request to background script to cancel native host jobs
      ext.runtime.sendMessage({
        type: 'CANCEL_ALL'
      }).catch(() => {});
      
      console.log('[AI Detector] All classifications cancelled ('+jobCount+' jobs)');
      return Promise.resolve({ success: true });
    }

    if (message.type === 'TOGGLE_BLUR') {
      const srcUrl = message.srcUrl;
      const selectionText = message.selectionText;
      
      // Handle image toggle
      if (srcUrl) {
        const images = Array.from(document.querySelectorAll('img'));
        const targetImage = images.find(img => img.src === srcUrl);
        if (targetImage) {
          // Use URL as the itemId
          const itemId = targetImage.src;
          
          // Check if image has visible AI styling (ai-detected class)
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
            // Return new state: no AI tags, with the label that was removed
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
      
      // Handle text toggle
      if (!srcUrl && lastContextMenuTarget) {
        const targetText = lastContextMenuTarget;
        const tag = targetText.tagName ? targetText.tagName.toLowerCase() : '';
        
        // Check if it's a valid text element
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
  /**
   * Re-apply visual effects to already-classified elements without re-classifying.
   * Called when visual settings change (blur, border, color, etc.)
   */
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
  performInitialExtraction();
  startDynamicDetection();
}

if (ext && ext.storage && ext.storage.onChanged) {
  ext.storage.onChanged.addListener((changes) => {
    // Settings that require re-classification
    const classificationKeys = ['enabled', 'altTextOnly', 'minTextLength'];
    
    // Settings that only affect visual appearance (no re-classification needed)
    const visualKeys = ['blurAmount', 'borderMultiplier', 'borderColor', 'textBlurAmount', 'textStrikethroughEnabled', 'imageAiThreshold', 'textAiThreshold'];
    
    // Settings that don't affect existing results (logging, debounce, VRAM management)
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
