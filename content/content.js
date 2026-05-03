// Cross-browser extension API abstraction (Chrome/Firefox compatibility)content
if (window.__AI_DETECTOR_BOOTSTRAPPED__) {
  console.warn('[AI Detector] Content script already initialized; skipping duplicate init');
} else {
  window.__AI_DETECTOR_BOOTSTRAPPED__ = true;

const ext = typeof browser !== 'undefined' ? browser : chrome;


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
  miniBatchSize: 10,              // Batch size for sending classifications (unused currently)
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
let hoverHandlerAttached = false; // Tracks weather a global hover handler is actuive already
let currentHoverEl = null;       // tracks the current hovered AI-detected element
let processedImages = new Set(); // Set of img elements already added to cachedImages
let activeJobIds = new Set(); // Track in-flight job IDs for chunked results
let itemResults = new Map();  // itemId -> { modality, ensembles: { ensembleId: { score, label, displayLabel } } }
let lastEnsembleSignature = null;
let lastEnsembleStyleSignature = null;

/**
 * OBSERVERS - Monitor page for dynamic content
 */
let mutationObserver = null;     // Watches for new page updates
let resizeObserver = null;       // Watches for lazy-loaded images
let intersectionObserver = null; // Watches for images entering viewport
let pendingClassification = [];   // Queue of images waiting to be classified
let pendingTextSections = [];    // Queue of text sections waiting to be classified
let classificationTimeout = null; // Debounce timer for batching classifications
let scanIntervalId = null;       // Periodic scan interval
let classificationInFlight = false; // Prevent overlapping classification requests

/**
 * STABLE ID COUNTERS - Ensure each image/text gets a unique, persistent ID
 * These increment globally and never reset, preventing ID collisions on scroll
 * Used by: buildClassificationBatches()
 */
let nextImageId = 0;  // Counter for image IDs (img-0, img-1, img-2...)
let nextTextId = 0;   // Counter for text IDs (text-0, text-1, text-2...)
let processedTextElements = new Set(); // Set of text elements already added to cachedTextSections

function computeBorderWidth(el, multiplier = 1) {
  /**
   * Calculate border width for flagged images based on image size as 2% of smaller side (width/height), clamped 2-10px, then multiplied by setting
   */
  if (!el) return 0;
  if (multiplier === 0) return 0;
  const w = el.naturalWidth || el.clientWidth || 0;
  const h = el.naturalHeight || el.clientHeight || 0;
  const minSide = Math.max(1, Math.min(w, h));
  const base = Math.max(2, Math.min(10, Math.round(minSide * 0.02)));
  return Math.max(0, Math.round(base * multiplier));
}

function hexToRgb(hex) {
  if (!hex || typeof hex !== 'string') return null;
  const clean = hex.startsWith('#') ? hex.slice(1) : hex;
  if (clean.length !== 6) return null;
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return null;
  return { r, g, b };
}

function rgbToHex({ r, g, b }) {
  const clamp = (n) => Math.max(0, Math.min(255, Math.round(n)));
  return `#${clamp(r).toString(16).padStart(2, '0')}${clamp(g).toString(16).padStart(2, '0')}${clamp(b).toString(16).padStart(2, '0')}`.toUpperCase();
}

function averageColors(colors) {
  if (!colors || !colors.length) return null;
  let r = 0, g = 0, b = 0, count = 0;
  colors.forEach((c) => {
    const rgb = hexToRgb(c);
    if (rgb) {
      r += rgb.r;
      g += rgb.g;
      b += rgb.b;
      count += 1;
    }
  });
  if (!count) return null;
  return rgbToHex({ r: r / count, g: g / count, b: b / count });
}

function buildDefaultEnsembles() {
  return [
    {
      id: 'ai_images',
      name: 'AI Images',
      modality: 'image',
      enabled: true,
      threshold: settings.imageAiThreshold ?? 0.5,
      classifiers: ['res_net50_fft'],
      weights: null,
      styles: {
        image: {
          applyBlur: true,
          blurAmount: settings.blurAmount ?? 4,
          applyBorder: true,
          borderMultiplier: settings.borderMultiplier ?? 1,
          borderColor: settings.borderColor ?? '#ff0064',
          applyGlow: true,
          glowColor: settings.borderColor ?? '#ff0064',
          applyBadge: true
        }
      }
    },
    {
      id: 'ai_text',
      name: 'AI Text',
      modality: 'text',
      enabled: true,
      threshold: settings.textAiThreshold ?? 0.5,
      classifiers: ['text'],
      weights: null,
      styles: {
        text: {
          applyBlur: true,
          blurAmount: settings.textBlurAmount ?? 2,
          applyStrikethrough: settings.textStrikethroughEnabled ?? true,
          strikethroughColor: settings.borderColor ?? '#ff0064',
          applyUnderline: false,
          underlineColor: settings.borderColor ?? '#ff0064',
          applyHighlight: false,
          highlightColor: '#fff3a1'
        }
      }
    }
  ];
}

function getEnsembleConfigs() {
  if (!Array.isArray(settings.ensembles) || settings.ensembles.length === 0) {
    return buildDefaultEnsembles();
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
}

function getOverlayContainer(imgEl) {
  /**
   * Find the best parent container for placing the AI badge overlay
   * Avoids <picture> elements which don't support absolute positioning well
   * 
   * Priority: figure/link/div/article/section > parent > null
   * Used by: applyVisualEffects() to place confidence badges near images
   */
  if (!imgEl || !(imgEl instanceof Element)) return null;
  const preferred = imgEl.closest('figure, a, div, article, section');
  if (preferred && preferred.tagName.toLowerCase() !== 'picture') {
    return preferred;
  }
  const parent = imgEl.parentElement;
  if (parent && parent.tagName.toLowerCase() !== 'picture') return parent;
  return null;
}

function showAi(el) {
  /**
   * reveal a flagged image and hide the badge
   */
  if (!el) return;
  el.style.setProperty('filter', 'blur(0px) saturate(1) brightness(1)', 'important');
  el.style.setProperty('animation', 'none', 'important');

  // hide the badge
  setImageBadgeOpacity(el, '0');
}

function hideAi(el, blurAmount = null) {
  /**
   * blur a flagged image and show the badge
   */
  if (!el) return;
  const blurPx = blurAmount != null ? blurAmount : settings.blurAmount;
  el.style.setProperty('filter', `blur(${blurPx}px) saturate(1) brightness(0.85)`, 'important');
  el.style.setProperty('animation', 'ai-pulse 2s ease-in-out infinite', 'important');

  // show the badge
  setImageBadgeOpacity(el, '1');
}

function attachGlobalHoverHandler() {
  /**
   * Use a global pointermove listener to show/hide AI images on hover through overlapping elements on the page
   * 
   * LOGIC:
   * - Tracks currentHoverEl (last element the user hovered over)
   * - When user moves to a new .ai-detected element, calls showAi() to reveal it
   * - When user moves away, calls hideAi() to blur it again
   * 
   * Called by: applyVisualEffects() after adding AI-detected images
   */
  if (hoverHandlerAttached) return;
  hoverHandlerAttached = true;
  document.addEventListener('pointermove', (evt) => {
    const hits = document.elementsFromPoint(evt.clientX, evt.clientY);
    const candidate = hits.find((el) => el.classList && el.classList.contains('ai-detected')) // stops at first element found
      || (evt.target instanceof Element ? evt.target.closest('.ai-detected') : null);

    if (candidate === currentHoverEl) return;
    if (currentHoverEl && currentHoverEl !== candidate) {
      // Check if element still has results before applying hover-off styles
      let hasResults = false;
      for (let [itemId, element] of elementMap) {
        if (element === currentHoverEl && itemResults.has(itemId)) {
          hasResults = true;
          break;
        }
      }
      
      if (hasResults) {
        const raw = currentHoverEl.dataset.aiImageStyle;
        if (raw) {
          let state;
          try {
            state = JSON.parse(raw);
          } catch {
            state = null;
          }
          if (state) {
            const showBlur = state.blurMode !== 'off';
            if (showBlur && state.blurAmount > 0) {
              hideAi(currentHoverEl, state.blurAmount);
            } else {
              showAi(currentHoverEl);
            }
            const showBorder = state.borderMode !== 'off';
            const bw = showBorder && state.borderMultiplier > 0 ? computeBorderWidth(currentHoverEl, state.borderMultiplier) : 0;
            if (bw > 0) {
              currentHoverEl.style.setProperty('outline', `${bw}px solid ${state.borderColor}`, 'important');
              currentHoverEl.style.setProperty('outline-offset', `-${bw}px`, 'important');
            } else {
              currentHoverEl.style.setProperty('outline', 'none', 'important');
            }
            currentHoverEl.style.setProperty('box-shadow', 'none', 'important');
            const showBadge = state.badgeMode !== 'off';
            setImageBadgeOpacity(currentHoverEl, showBadge ? '1' : '0');
          }
        } else {
          const blurAmount = parseFloat(currentHoverEl.dataset.aiBlur || settings.blurAmount || 0);
          hideAi(currentHoverEl, blurAmount);
        }
      }
    }
    currentHoverEl = candidate;
    if (currentHoverEl) {
      // Check if element still has results before applying hover-on styles
      let hasResults = false;
      for (let [itemId, element] of elementMap) {
        if (element === currentHoverEl && itemResults.has(itemId)) {
          hasResults = true;
          break;
        }
      }
      
      if (hasResults) {
        const raw = currentHoverEl.dataset.aiImageStyle;
        if (raw) {
          let state;
          try {
            state = JSON.parse(raw);
          } catch {
            state = null;
          }
          if (state) {
            const showBlur = state.blurMode === 'always';
            if (showBlur && state.blurAmount > 0) {
              hideAi(currentHoverEl, state.blurAmount);
            } else {
              showAi(currentHoverEl);
            }
            const showBorder = state.borderMode === 'always';
            const bw = showBorder && state.borderMultiplier > 0 ? computeBorderWidth(currentHoverEl, state.borderMultiplier) : 0;
            if (bw > 0) {
              currentHoverEl.style.setProperty('outline', `${bw}px solid ${state.borderColor}`, 'important');
              currentHoverEl.style.setProperty('outline-offset', `-${bw}px`, 'important');
            } else {
              currentHoverEl.style.setProperty('outline', 'none', 'important');
            }
            currentHoverEl.style.setProperty('box-shadow', 'none', 'important');
            const showBadge = state.badgeMode === 'always';
            setImageBadgeOpacity(currentHoverEl, showBadge ? '1' : '0');
          }
        } else {
          showAi(currentHoverEl);
        }
      }
    }
  }, true);
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

function initMutationObserver() {
    /**
   * watches for new page updates, triggers classification on new content
   */
  if (mutationObserver) return;
  
  mutationObserver = new MutationObserver((mutations) => {
    const newImages = [];
    const newText = [];
    
    mutations.forEach((mutation) => {
      // Check for src attribute changes (Google Images swapping thumbnail for full image)
      if (mutation.type === 'attributes' && mutation.attributeName === 'src') {
        const el = mutation.target;
        if (el.tagName && el.tagName.toLowerCase() === 'img') {
          // Image src changed, remove from processed set and re-process
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
      
      // Check added nodes
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType !== 1) return; // Only elements
        
        // Direct img tags
        if (node.tagName && node.tagName.toLowerCase() === 'img') {
          const img = processImageElement(node);
          if (img) newImages.push(img);
          // Also observe new images with IntersectionObserver
          if (intersectionObserver) intersectionObserver.observe(node);
          if (resizeObserver) resizeObserver.observe(node);
        }
        
        // Search for img tags within added subtrees
        if (node.querySelectorAll) {
          try {
            const images = Array.from(node.querySelectorAll('img'));
            images.forEach(el => {
              const img = processImageElement(el);
              if (img) newImages.push(img);
              // Also observe new images
              if (intersectionObserver) intersectionObserver.observe(el);
              if (resizeObserver) resizeObserver.observe(el);
            });
          } catch (e) {
            // querySelectorAll may fail on some elements
          }
        }
        
        // Extract text from this node and its children
        const textSelectors = ['p', 'li', 'pre', 'blockquote', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].join(',');
        if (node.querySelectorAll) {
          try {
            // Check if node itself is a text element
            if (node.tagName && node.querySelectorAll === 'function') {
              const nodeTagLower = node.tagName.toLowerCase();
              if (['p', 'li', 'pre', 'blockquote', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(nodeTagLower)) {
                const text = processTextElement(node);
                if (text) newText.push(text);
              }
            }
            // Search for text elements within subtree
            const textElements = Array.from(node.querySelectorAll(textSelectors));
            textElements.forEach(el => {
              const text = processTextElement(el);
              if (text) newText.push(text);
            });
          } catch (e) {
            // querySelectorAll may fail on some elements
          }
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
  });
  
  // Observe entire document for changes
  mutationObserver.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,  // Watch for attribute changes
    attributeFilter: ['src'],  // Only watch src attribute
    characterData: false
  });
}

// Detect lazy-loaded images entering viewport using IntersectionObserver
function initIntersectionObserver() {
  if (intersectionObserver) return;
  
  intersectionObserver = new IntersectionObserver((entries) => {
    const newImages = [];
    
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const el = entry.target;
        if (el.tagName && el.tagName.toLowerCase() === 'img') {
          // Image entered viewport, check if it's loaded now
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
      }
    });
    
    if (newImages.length > 0) {
      scheduleClassification(newImages);
    }
  }, {
    rootMargin: '200px' // Start loading 200px before entering viewport
  });
  
  // Observe all current images
  const allImages = Array.from(document.querySelectorAll('img'));
  allImages.forEach(img => intersectionObserver.observe(img));
  
  if (settings.verboseLogs) {
    console.log(`[AI Detector] IntersectionObserver initialized, observing ${allImages.length} images`);
  }
}

// Detect lazy-loaded images entering viewport
function initResizeObserver() {
  if (resizeObserver) return;
  
  resizeObserver = new ResizeObserver((entries) => {
    const newImages = [];
    
    entries.forEach((entry) => {
      const el = entry.target;
      if (el.tagName && el.tagName.toLowerCase() === 'img') {
        // Image may have just loaded (width/height changed)
        if (el.naturalWidth > 0 && el.naturalHeight > 0) {
          const img = processImageElement(el);
          if (img) {
            newImages.push(img);
            if (settings.verboseLogs) {
              console.log(`[AI Detector] ResizeObserver: Image loaded:`, el.src.substring(0, 80));
            }
          }
        }
      }
    });
    if (newImages.length > 0) {
      scheduleClassification(newImages);
    }
  });
  
  // Start observing all current images
  const allImages = Array.from(document.querySelectorAll('img'));
  allImages.forEach(img => resizeObserver.observe(img));
  
  if (settings.verboseLogs) {
    console.log(`[AI Detector] ResizeObserver initialized, observing ${allImages.length} images`);
  }
}

// Process and extract text from a text element if not already seen
function processTextElement(el) {
  if (!el || !(el instanceof Element)) return null;
  
  // Skip if already processed
  if (processedTextElements.has(el)) return null;
  
  // Extract and clean text
  const text = (el.innerText || el.textContent || '')
    .replace(/\s+/g, ' ')
    .trim();
  
  // Skip if too short
  if (!text || text.length < settings.minTextLength) return null;
  
  // Skip if we've already seen this exact text
  for (let section of cachedTextSections) {
    if (section.text === text) return null;
  }
  for (let section of pendingTextSections) {
    if (section.text === text) return null;
  }
  
  // Mark as processed
  processedTextElements.add(el);
  
  const tag = el.tagName.toLowerCase();
  const id = el.id ? '#' + el.id : '';
  const classes = el.classList && el.classList.length ? '.' + Array.from(el.classList).join('.') : '';
  const selector = tag + id + classes;
  
  return {
    index: cachedTextSections.length + pendingTextSections.length,
    selector,
    tag,
    length: text.length,
    text,
    element: el
  };
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
    // Image hasn't loaded yet, attach load listener
    if (!el.dataset.aiLoadListenerAttached) {
      el.dataset.aiLoadListenerAttached = 'true';
      el.addEventListener('load', () => {
        if (settings.verboseLogs) {
          console.log(`[AI Detector] Image load event:`, el.src.substring(0, 80));
        }
        processedImages.delete(el);
        const img = processImageElement(el);
        if (img) scheduleClassification([img]);
      }, { once: true });
    }
    if (resizeObserver) resizeObserver.observe(el);
    if (intersectionObserver) intersectionObserver.observe(el);
    if (settings.verboseLogs) {
      console.log(`[AI Detector] Image not loaded yet (${nw}x${nh}):`, src.substring(0, 80));
    }
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
  
  // Mark as processed
  processedImages.add(el);
  
  if (settings.verboseLogs) {
    console.log(`[AI Detector] Processing image (${nw}x${nh}):`, src.substring(0, 80));
  }
  
  const img = {
    src: src,
    alt: el.alt || '',
    width: el.naturalWidth || el.width || null,
    height: el.naturalHeight || el.height || null,
    element: el,
    lazy: isLazy
  };
  
  // Register observers on this specific image
  if (resizeObserver) {
    resizeObserver.observe(el);
  }
  if (intersectionObserver) {
    intersectionObserver.observe(el);
  }
  
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
  const MAX_MESSAGE_SIZE = 50000000;  // 50MB (well under 4GB limit, but reasonable for performance)
  let currentBatch = [];
  let currentSize = 1000;  // Start with overhead estimate
  
  // DO NOT clear elementMap - preserve mappings from previous batches
  // Only add new entries for images we haven't seen yet
  
  // Track which images we've already assigned IDs to
  const assignedIds = new Map();  // img element -> itemId
  for (let [itemId, element] of elementMap) {
    if (itemId.startsWith('img-')) {
      assignedIds.set(element, itemId);
    }
  }

  images.forEach((img) => {
    let itemId = assignedIds.get(img.element);
    if (!itemId) {
      // New image, assign a fresh stable ID
      itemId = 'img-' + (nextImageId++);
      elementMap.set(itemId, img.element);
      assignedIds.set(img.element, itemId);
    }
    
    // Try to get base64 from DOM first (avoids network fetch)
    let imageData = null;
    let itemSize = 500;  // Rough estimate for JSON overhead per item
    
    try {
      imageData = imageToBase64(img.element);
      if (imageData === 'TOO_SMALL') {
        // Image is a low-quality thumbnail, skip it for now
        // Remove from processedImages so it can be re-processed when full version loads
        processedImages.delete(img.element);
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
    if (currentSize + itemSize > MAX_MESSAGE_SIZE && currentBatch.length > 0) {
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
    let itemId = null;
    // Check if this text element already has an ID
    for (let [id, element] of elementMap) {
      if (id.startsWith('text-') && element === section.element) {
        itemId = id;
        break;
      }
    }
    
    if (!itemId) {
      // New text section, assign a fresh stable ID
      itemId = 'text-' + (nextTextId++);
      elementMap.set(itemId, section.element);
    }
    
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
      console.log(`[AI Detector] Message size: ~${Math.round(currentSize / 1024)}KB of 900KB`);
    }
  }
  
  return batches;
}

// Send items to the background script for native classification (handles multiple batches in parallel)
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
    activeJobIds.add(jobId);
    setTimeout(() => activeJobIds.delete(jobId), 5 * 60 * 1000);

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

    await Promise.all(
      batches.map((batch, idx) =>
        ext.runtime.sendMessage({
          type: 'CLASSIFY_JOB_CHUNK',
          jobId,
          chunkIndex: idx,
          totalChunks: batches.length,
          items: batch,
          model: modelConfig
        })
      )
    );

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

function clearImageStyles(element) {
  if (!element) return;
  element.classList.remove('ai-detected');
  element.style.setProperty('filter', 'blur(0px) saturate(1) brightness(1)', 'important');
  element.style.setProperty('animation', 'none', 'important');
  element.style.setProperty('outline', 'none', 'important');
  element.style.setProperty('box-shadow', 'none', 'important');
  if (element.dataset && element.dataset.aiBlur) {
    delete element.dataset.aiBlur;
  }
  getImageBadgeIds(element).forEach((badgeId) => {
    const badge = badgeId ? document.getElementById(badgeId) : null;
    if (badge) badge.remove();
  });
  if (element.dataset) {
    delete element.dataset.badgeIds;
    delete element.dataset.badgeId;
    delete element.dataset.aiImageStyle;
    delete element.dataset.aiToggleAttached;
  }
}

function getImageBadgeIds(element) {
  if (!element || !element.dataset) return [];
  if (element.dataset.badgeIds) {
    try {
      const ids = JSON.parse(element.dataset.badgeIds);
      return Array.isArray(ids) ? ids : [];
    } catch {
      return [];
    }
  }
  return element.dataset.badgeId ? [element.dataset.badgeId] : [];
}

function setImageBadgeOpacity(element, opacity) {
  getImageBadgeIds(element).forEach((badgeId) => {
    const badge = badgeId ? document.getElementById(badgeId) : null;
    if (badge) {
      badge.style.opacity = opacity;
    }
  });
}

function clearTextStyles(element) {
  if (!element) return;
  element.classList.remove('ai-detected-text');
  element.style.removeProperty('filter');
  element.style.removeProperty('text-decoration');
  element.style.removeProperty('text-decoration-color');
  element.style.removeProperty('text-decoration-thickness');
  element.style.removeProperty('background-color');
  element.style.removeProperty('padding');
  element.style.removeProperty('cursor');
  element.style.removeProperty('transition');
}

function getEnsembleById(ensembleId) {
  return getEnsembleConfigs().find((ens) => ens.id === ensembleId);
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

function normalizeEnsembleWeight(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) {
    return 1;
  }
  return numeric;
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
      weight: normalizeEnsembleWeight(weights[index]),
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

function applyStylesForItem(itemId) {
  const entry = itemResults.get(itemId);
  if (!entry) return;
  const element = elementMap.get(itemId);
  if (!element) return;
  
  // Check if element is still in the document
  if (!element.isConnected) {
    if (settings.verboseLogs) {
      console.log(`[AI Detector] Element ${itemId} no longer in DOM, skipping style application`);
    }
    return;
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
        displayLabel
      });
    }
  });

  if (modality === 'image') {
    if (!matched.length) {
      clearImageStyles(element);
      return;
    }

    const blurCandidates = matched.map(({ cfg }) => {
      const styles = cfg.styles?.image || {};
      const mode = styles.blurMode || (styles.applyBlur === false ? 'off' : 'hover');
      const amount = styles.blurAmount ?? settings.blurAmount ?? 4;
      return { mode, amount };
    });

    const borderCandidates = matched.map(({ cfg }) => {
      const styles = cfg.styles?.image || {};
      const mode = styles.borderMode || (styles.applyBorder === false ? 'off' : 'hover');
      const multiplier = styles.borderMultiplier ?? settings.borderMultiplier ?? 1;
      const color = styles.borderColor ?? settings.borderColor;
      return { mode, multiplier, color };
    });

    const badgeCandidates = matched.map(({ cfg }) => {
      const styles = cfg.styles?.image || {};
      const mode = styles.badgeMode || (styles.applyBadge === false ? 'off' : 'hover');
      return { mode };
    });

    const resolveCombinedMode = (candidates) => {
      if (candidates.some((c) => c.mode === 'always')) return 'always';
      if (candidates.some((c) => c.mode === 'hover')) return 'hover';
      return 'off';
    };

    const blurMode = resolveCombinedMode(blurCandidates);
    const borderMode = resolveCombinedMode(borderCandidates);
    const badgeMode = resolveCombinedMode(badgeCandidates);

    const blurAmounts = blurCandidates.filter((c) => c.mode !== 'off').map((c) => c.amount);
    const blurAmount = blurAmounts.length ? (blurAmounts.reduce((a, b) => a + b, 0) / blurAmounts.length) : 0;
    const borderMultipliers = borderCandidates.filter((c) => c.mode !== 'off').map((c) => c.multiplier);
    const borderMultiplier = borderMultipliers.length ? (borderMultipliers.reduce((a, b) => a + b, 0) / borderMultipliers.length) : 0;
    const borderColor = averageColors(borderCandidates.filter((c) => c.mode !== 'off').map((c) => c.color)) || settings.borderColor || '#ff0064';

    element.classList.add('ai-detected');
    const maxScore = Math.max(...matched.map((m) => m.score));
    const badgeLabel = matched.find((m) => m.displayLabel && m.displayLabel !== 'AI generated')?.displayLabel
      || matched[0]?.displayLabel
      || 'AI generated';
    element.setAttribute('data-ai-score', maxScore);
    element.title = `${badgeLabel} (${(maxScore * 100).toFixed(1)}% confidence)`;
    element.style.setProperty('pointer-events', 'auto', 'important');
    const showBlur = blurMode !== 'off';
    if (showBlur && blurAmount > 0) {
      hideAi(element, blurAmount);
    } else {
      showAi(element);
    }
    element.dataset.aiBlur = String(blurAmount);

    const showBorder = borderMode !== 'off';
    const bw = showBorder && borderMultiplier > 0 ? computeBorderWidth(element, borderMultiplier) : 0;
    if (bw > 0) {
      element.style.setProperty('outline', `${bw}px solid ${borderColor}`, 'important');
      element.style.setProperty('outline-offset', `-${bw}px`, 'important');
    } else {
      element.style.setProperty('outline', 'none', 'important');
    }

    element.style.setProperty('box-shadow', 'none', 'important');

    const showBadge = badgeMode !== 'off';
    const existingBadgeIds = getImageBadgeIds(element);
    existingBadgeIds.forEach((badgeId) => {
      const existingBadge = badgeId ? document.getElementById(badgeId) : null;
      if (existingBadge) existingBadge.remove();
    });
    if (element.dataset) {
      delete element.dataset.badgeIds;
      delete element.dataset.badgeId;
    }

    if (showBadge) {
      const container = getOverlayContainer(element);
      if (container) {
        if (getComputedStyle(container).position === 'static') {
          container.style.position = 'relative';
        }

        const badgeEntries = [];
        matched.forEach(({ cfg, score, displayLabel, classifiers }) => {
          const classifierEntries = classifiers && typeof classifiers === 'object'
            ? Object.entries(classifiers)
            : [];

          if (classifierEntries.length) {
            classifierEntries.forEach(([, classifierData]) => {
              badgeEntries.push({
                label: classifierData?.label || classifierData?.displayLabel || displayLabel || badgeLabel,
                score: typeof classifierData?.score === 'number' ? classifierData.score : score,
                color: cfg.styles?.image?.borderColor || borderColor
              });
            });
          } else {
            badgeEntries.push({
              label: displayLabel || badgeLabel,
              score,
              color: cfg.styles?.image?.borderColor || borderColor
            });
          }
        });

        const badgeIds = [];
        badgeEntries.forEach((entry, index) => {
          const badge = document.createElement('span');
          badge.style.position = 'absolute';
          badge.style.top = `${6 + (index * 28)}px`;
          badge.style.right = '6px';
          badge.style.padding = '4px 6px';
          badge.style.color = '#fff';
          badge.style.fontSize = '12px';
          badge.style.fontWeight = '700';
          badge.style.borderRadius = '4px';
          badge.style.zIndex = String(9999 - index);
          badge.style.cursor = 'pointer';
          badge.style.userSelect = 'none';
          badge.style.opacity = '1';
          badge.style.transition = 'opacity 0.2s ease';
          badge.className = 'ai-badge';
          badge.textContent = `${entry.label} ${Math.round(entry.score * 100)}%`;
          const badgeRgb = hexToRgb(entry.color) || { r: 255, g: 0, b: 100 };
          badge.style.background = `rgba(${badgeRgb.r}, ${badgeRgb.g}, ${badgeRgb.b}, 0.9)`;

          const bid = `ai-badge-${Math.random().toString(36).slice(2, 8)}`;
          badge.id = bid;
          badgeIds.push(bid);
          container.appendChild(badge);

          if (!badge.dataset.aiToggleAttached) {
            badge.dataset.aiToggleAttached = 'true';
            // Toggle functionality now handled via context menu
          }
        });

        if (badgeIds.length) {
          element.dataset.badgeIds = JSON.stringify(badgeIds);
          element.dataset.badgeId = badgeIds[0];
        }
      }
    }

    element.dataset.aiImageStyle = JSON.stringify({
      blurMode,
      blurAmount,
      borderMode,
      borderMultiplier,
      borderColor,
      badgeMode
    });

    if (!element.dataset.aiToggleAttached) {
      element.dataset.aiToggleAttached = 'true';
      // Toggle functionality now handled via context menu
    }

    attachGlobalHoverHandler();
    return;
  }

  if (modality === 'text') {
    if (!matched.length) {
      clearTextStyles(element);
      return;
    }

    const resolveMode = (styles, key, legacyKey, legacyEnabledValue) => {
      if (styles && styles[key]) return styles[key];
      if (styles && Object.prototype.hasOwnProperty.call(styles, legacyKey)) {
        return styles[legacyKey] === legacyEnabledValue ? 'hover' : 'off';
      }
      return 'hover';
    };

    const blurCandidates = matched.map(({ cfg }) => {
      const styles = cfg.styles?.text || {};
      const mode = styles.blurMode || (styles.applyBlur === false ? 'off' : 'hover');
      const amount = styles.blurAmount ?? settings.textBlurAmount ?? 2;
      return { mode, amount };
    });

    const strikeCandidates = matched.map(({ cfg }) => {
      const styles = cfg.styles?.text || {};
      const mode = styles.strikethroughMode || (styles.applyStrikethrough === false ? 'off' : 'hover');
      const color = styles.strikethroughColor ?? settings.borderColor;
      return { mode, color };
    });

    const underlineCandidates = matched.map(({ cfg }) => {
      const styles = cfg.styles?.text || {};
      const mode = styles.underlineMode || (styles.applyUnderline === true ? 'hover' : 'off');
      const color = styles.underlineColor ?? settings.borderColor;
      return { mode, color };
    });

    const highlightCandidates = matched.map(({ cfg }) => {
      const styles = cfg.styles?.text || {};
      const mode = styles.highlightMode || (styles.applyHighlight === true ? 'hover' : 'off');
      const color = styles.highlightColor ?? '#fff3a1';
      return { mode, color };
    });

    const resolveCombinedMode = (candidates) => {
      if (candidates.some((c) => c.mode === 'always')) return 'always';
      if (candidates.some((c) => c.mode === 'hover')) return 'hover';
      return 'off';
    };

    const blurMode = resolveCombinedMode(blurCandidates);
    const strikeMode = resolveCombinedMode(strikeCandidates);
    const underlineMode = resolveCombinedMode(underlineCandidates);
    const highlightMode = resolveCombinedMode(highlightCandidates);

    const blurAmounts = blurCandidates.filter((c) => c.mode !== 'off').map((c) => c.amount);
    const blurAmount = blurAmounts.length ? (blurAmounts.reduce((a, b) => a + b, 0) / blurAmounts.length) : 0;

    const strikeColor = averageColors(strikeCandidates.filter((c) => c.mode !== 'off').map((c) => c.color))
      || settings.borderColor || '#ff0064';
    const underlineColor = averageColors(underlineCandidates.filter((c) => c.mode !== 'off').map((c) => c.color))
      || settings.borderColor || '#ff0064';
    const highlightColor = averageColors(highlightCandidates.filter((c) => c.mode !== 'off').map((c) => c.color))
      || null;

    element.classList.add('ai-detected-text');
    const maxScore = Math.max(...matched.map((m) => m.score));
    element.setAttribute('data-ai-score', maxScore);
    const textLabel = matched.find((m) => m.displayLabel && m.displayLabel !== 'AI generated')?.displayLabel
      || matched[0]?.displayLabel
      || 'AI-generated text';
    element.title = `${textLabel} (${(maxScore * 100).toFixed(1)}% confidence)`;

    const showBlur = blurMode !== 'off';
    const showStrike = strikeMode !== 'off';
    const showUnderline = underlineMode !== 'off';
    const showHighlight = highlightMode !== 'off';

    if (showBlur && blurAmount > 0) {
      element.style.setProperty('filter', `blur(${blurAmount}px)`, 'important');
    } else {
      element.style.removeProperty('filter');
    }

    const decorations = [];
    if (showStrike) decorations.push('line-through');
    if (showUnderline) decorations.push('underline');
    if (decorations.length) {
      element.style.setProperty('text-decoration', decorations.join(' '), 'important');
      element.style.setProperty('text-decoration-color', showUnderline ? underlineColor : strikeColor, 'important');
      element.style.setProperty('text-decoration-thickness', '2px', 'important');
    } else {
      element.style.removeProperty('text-decoration');
      element.style.removeProperty('text-decoration-color');
      element.style.removeProperty('text-decoration-thickness');
    }

    if (showHighlight && highlightColor) {
      element.style.setProperty('background-color', highlightColor, 'important');
      element.style.setProperty('padding', '0 2px', 'important');
    } else {
      element.style.removeProperty('background-color');
      element.style.removeProperty('padding');
    }

    element.dataset.aiTextStyle = JSON.stringify({
      blurMode,
      blurAmount,
      strikeMode,
      strikeColor,
      underlineMode,
      underlineColor,
      highlightMode,
      highlightColor
    });

    element.style.setProperty('cursor', 'pointer', 'important');
    element.style.setProperty('transition', 'all 0.2s ease', 'important');

    if (!element.dataset.aiTextHoverAttached) {
      element.dataset.aiTextHoverAttached = 'true';
      element.addEventListener('mouseenter', () => {
        const raw = element.dataset.aiTextStyle;
        if (!raw) return;
        let state;
        try {
          state = JSON.parse(raw);
        } catch {
          return;
        }
        const showBlur = state.blurMode === 'always';
        const showStrike = state.strikeMode === 'always';
        const showUnderline = state.underlineMode === 'always';
        const showHighlight = state.highlightMode === 'always';

        if (showBlur && state.blurAmount > 0) {
          element.style.setProperty('filter', `blur(${state.blurAmount}px)`, 'important');
        } else {
          element.style.removeProperty('filter');
        }

        const decorations = [];
        if (showStrike) decorations.push('line-through');
        if (showUnderline) decorations.push('underline');
        if (decorations.length) {
          element.style.setProperty('text-decoration', decorations.join(' '), 'important');
          element.style.setProperty('text-decoration-color', showUnderline ? state.underlineColor : state.strikeColor, 'important');
          element.style.setProperty('text-decoration-thickness', '2px', 'important');
        } else {
          element.style.removeProperty('text-decoration');
          element.style.removeProperty('text-decoration-color');
          element.style.removeProperty('text-decoration-thickness');
        }

        if (showHighlight && state.highlightColor) {
          element.style.setProperty('background-color', state.highlightColor, 'important');
          element.style.setProperty('padding', '0 2px', 'important');
        } else {
          element.style.removeProperty('background-color');
          element.style.removeProperty('padding');
        }
      });

      element.addEventListener('mouseleave', () => {
        applyStylesForItem(itemId);
      });
    }
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
    const element = elementMap.get(result.id);
    if (element && element.isConnected) {
      applyStylesForItem(result.id);
      appliedCount++;
    } else {
      skippedCount++;
    }
  });
  
  if (settings.verboseLogs && skippedCount > 0) {
    console.log(`[AI Detector] Applied ${appliedCount} results, skipped ${skippedCount} (elements no longer in DOM)`);
  }
}

// Extract long text sections from the page (>=250 chars)
function extractLongText() {
  try {
    const selectors = [
      'p', 'li', 'pre', 'blockquote', 'td', 'th',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
    ].join(',');

    const nodes = Array.from(document.querySelectorAll(selectors));

    // Extract cleaned text and metadata
    const raw = nodes.map((el) => {
      const text = (el.innerText || el.textContent || '')
        .replace(/\s+/g, ' ')
        .trim();
      return { el, text };
    });

    // Filter sections >= 250 chars and deduplicate by text
    const seen = new Set();
    const sections = [];
    raw.forEach(({ el, text }) => {
      // Skip container elements that contain other candidate text elements
      try {
        if (el.querySelector(`:scope ${selectors}`)) return;
      } catch (e) {
        if (el.querySelector(selectors)) return;
      }
      if (!text || text.length < settings.minTextLength) return;
      if (seen.has(text)) return;
      seen.add(text);

      // Build a minimal selector for context (tag#id.class1.class2)
      const tag = el.tagName.toLowerCase();
      const id = el.id ? '#' + el.id : '';
      const classes = el.classList && el.classList.length ? '.' + Array.from(el.classList).join('.') : '';
      const selector = tag + id + classes;

      sections.push({
        index: sections.length,
        selector,
        tag,
        length: text.length,
        text,
        element: el
      });
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
  if (!settings.enabled) return;
  // Initialize observers to catch dynamically loaded images
  initMutationObserver();
  initResizeObserver();
  initIntersectionObserver();
  
  // Periodic scan every 3 seconds for edge cases (catches lazy-loaded images that observers miss)
  if (scanIntervalId) clearInterval(scanIntervalId);
  scanIntervalId = setInterval(() => {
    // Clean up disconnected elements from tracking sets
    for (const img of processedImages) {
      if (!img.isConnected) {
        processedImages.delete(img);
      }
    }
    for (const text of processedTextElements) {
      if (!text.isConnected) {
        processedTextElements.delete(text);
      }
    }
    
    const newImages = scanAllImages();
    const newText = extractLongText().filter(section => !processedTextElements.has(section.element));
    
    if (newImages.length > 0) {
      scheduleClassification(newImages);
    }
    if (newText.length > 0) {
      newText.forEach(text => processedTextElements.add(text.element));
      scheduleTextClassification(newText);
    }
  }, 3000);
}

function performInitialExtraction() {
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

// Track last right-clicked element for context menu
let lastContextMenuTarget = null;

// Update context menu title when user right-clicks on an image or text
document.addEventListener('contextmenu', (evt) => {
  lastContextMenuTarget = evt.target;
  
  // Check if it's an image
  if (evt.target && evt.target.tagName && evt.target.tagName.toLowerCase() === 'img') {
    const hasVisibleAiStyling = evt.target.classList.contains('ai-detected');
    let label = 'tag';
    
    if (hasVisibleAiStyling) {
      // Get the badge label
      const badgeIds = getImageBadgeIds(evt.target);
      if (badgeIds.length > 0) {
        const badge = document.getElementById(badgeIds[0]);
        if (badge && badge.textContent) {
          const match = badge.textContent.match(/^(.+?)\s+\d+%$/);
          label = match ? match[1] : badge.textContent.split(' ')[0];
        }
      }
    }
    
    ext.runtime.sendMessage({ 
      type: 'UPDATE_CONTEXT_MENU', 
      hasAiTags: hasVisibleAiStyling,
      label: label,
      modality: 'image'
    }).catch(() => {});
  }
  // Check if it's text with ai-detected-text class
  else if (evt.target && evt.target.classList && evt.target.classList.contains('ai-detected-text')) {
    const hasVisibleAiStyling = true;
    let label = 'AI text';
    
    // Try to extract label from title attribute if available
    if (evt.target.title) {
      const match = evt.target.title.match(/^(.+?)\s+\(/); 
      label = match ? match[1] : 'AI text';
    }
    
    ext.runtime.sendMessage({ 
      type: 'UPDATE_CONTEXT_MENU', 
      hasAiTags: hasVisibleAiStyling,
      label: label,
      modality: 'text'
    }).catch(() => {});
  }
  // Check if right-clicking on normal text (for adding blur)
  else if (evt.target && evt.target.tagName) {
    const tag = evt.target.tagName.toLowerCase();
    if (['p', 'li', 'pre', 'blockquote', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tag)) {
      ext.runtime.sendMessage({ 
        type: 'UPDATE_CONTEXT_MENU', 
        hasAiTags: false,
        label: 'tag',
        modality: 'text'
      }).catch(() => {});
    }
  }
}, true);

// Listen for streamed classification results
if (ext && ext.runtime && ext.runtime.onMessage) {
  ext.runtime.onMessage.addListener((message) => {
    if (!message || !message.type) return false;

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
      if (payload && payload.jobId) activeJobIds.delete(payload.jobId);
      return false;
    }

    if (message.type === 'RECLASSIFY_ALL') {
      reloadSettingsAndRescan();
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
          // Find the itemId for this image
          let itemId = null;
          for (let [id, element] of elementMap) {
            if (element === targetImage && id.startsWith('img-')) {
              itemId = id;
              break;
            }
          }
          
          // Check if image has visible AI styling (ai-detected class)
          const hasVisibleAiStyling = targetImage.classList.contains('ai-detected');
          
          if (hasVisibleAiStyling) {
            // Has visible tags - remove them
            // Get the badge label to return
            let badgeLabel = 'AI';
            const badgeIds = getImageBadgeIds(targetImage);
            if (badgeIds.length > 0) {
              const badge = document.getElementById(badgeIds[0]);
              if (badge && badge.textContent) {
                // Extract label from badge text (e.g., "Blur 100%" -> "Blur")
                const match = badge.textContent.match(/^(.+?)\s+\d+%$/);
                badgeLabel = match ? match[1] : badge.textContent.split(' ')[0];
              }
            }
            
            if (itemId) {
              itemResults.delete(itemId);
            }
            clearImageStyles(targetImage);
            if (currentHoverEl === targetImage) {
              currentHoverEl = null;
            }
            // Return new state: no AI tags, with the label that was removed
            return Promise.resolve({ hasAiTags: false, label: badgeLabel, modality: 'image' });
          } else {
            // No tags - add manual blur
            if (!itemId) {
              itemId = 'img-' + (nextImageId++);
              elementMap.set(itemId, targetImage);
            }
            
            itemResults.set(itemId, {
              modality: 'image',
              ensembles: {
                'manual_blur': {
                  score: 1.0,
                  label: 'manual',
                  displayLabel: 'Blur'
                }
              }
            });
            
            const manualEnsemble = {
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
            };
            
            const currentEnsembles = getEnsembleConfigs();
            const hasManualEnsemble = currentEnsembles.some(e => e.id === 'manual_blur');
            if (!hasManualEnsemble) {
              settings.ensembles = [...currentEnsembles, manualEnsemble];
            }
            
            applyStylesForItem(itemId);
            
            if (!hasManualEnsemble) {
              settings.ensembles = currentEnsembles;
            }
            // Return new state: has AI tags, with the label that was added
            return Promise.resolve({ hasAiTags: true, label: 'Blur', modality: 'image' });
          }
        }
      }
      
      // Handle text toggle
      if (!srcUrl && lastContextMenuTarget) {
        const targetText = lastContextMenuTarget;
        const tag = targetText.tagName ? targetText.tagName.toLowerCase() : '';
        
        // Check if it's a valid text element
        if (['p', 'li', 'pre', 'blockquote', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tag) || 
            targetText.classList.contains('ai-detected-text')) {
          // Find the itemId for this text
          let itemId = null;
          for (let [id, element] of elementMap) {
            if (element === targetText && id.startsWith('text-')) {
              itemId = id;
              break;
            }
          }
          
          const hasVisibleAiStyling = targetText.classList.contains('ai-detected-text');
          
          if (hasVisibleAiStyling) {
            // Has visible tags - remove them
            let label = 'AI text';
            if (targetText.title) {
              const match = targetText.title.match(/^(.+?)\s+\(/);
              label = match ? match[1] : 'AI text';
            }
            
            if (itemId) {
              itemResults.delete(itemId);
            }
            clearTextStyles(targetText);
            return Promise.resolve({ hasAiTags: false, label: label, modality: 'text' });
          } else {
            // No tags - add manual blur
            if (!itemId) {
              itemId = 'text-' + (nextTextId++);
              elementMap.set(itemId, targetText);
            }
            
            itemResults.set(itemId, {
              modality: 'text',
              ensembles: {
                'manual_blur_text': {
                  score: 1.0,
                  label: 'manual',
                  displayLabel: 'Blur'
                }
              }
            });
            
            const manualEnsemble = {
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
            };
            
            const currentEnsembles = getEnsembleConfigs();
            const hasManualEnsemble = currentEnsembles.some(e => e.id === 'manual_blur_text');
            if (!hasManualEnsemble) {
              settings.ensembles = [...currentEnsembles, manualEnsemble];
            }
            
            applyStylesForItem(itemId);
            
            if (!hasManualEnsemble) {
              settings.ensembles = currentEnsembles;
            }
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

function resetObservers() {
  if (mutationObserver) {
    mutationObserver.disconnect();
    mutationObserver = null;
  }
  if (resizeObserver) {
    resizeObserver.disconnect();
    resizeObserver = null;
  }
  if (intersectionObserver) {
    intersectionObserver.disconnect();
    intersectionObserver = null;
  }
  if (scanIntervalId) {
    clearInterval(scanIntervalId);
    scanIntervalId = null;
  }
}

function resetState() {
  cachedImages = [];
  cachedTextSections = [];
  elementMap.clear();
  processedImages.clear();
  processedTextElements.clear();
  pendingClassification = [];
  pendingTextSections = [];
  activeJobIds.clear();
  itemResults.clear();
  if (classificationTimeout) {
    clearTimeout(classificationTimeout);
    classificationTimeout = null;
  }
}

async function reloadSettingsAndRescan() {
  await loadSettings();
  resetObservers();
  resetState();
  performInitialExtraction();
  startDynamicDetection();
}

if (ext && ext.storage && ext.storage.onChanged) {
  ext.storage.onChanged.addListener(async (changes) => {
    // Settings that require re-classification
    const classificationKeys = ['enabled', 'altTextOnly', 'imageCaptureQuality', 'minTextLength'];
    
    // Settings that only affect visual appearance (no re-classification needed)
    const visualKeys = ['blurAmount', 'borderMultiplier', 'borderColor', 'textBlurAmount', 'textStrikethroughEnabled', 'imageAiThreshold', 'textAiThreshold'];
    
    // Settings that don't affect existing results (logging, debounce, VRAM management)
    const performanceKeys = ['classificationDelay', 'verboseLogs', 'lazyLoad'];
    
    const ensembleChanged = Object.prototype.hasOwnProperty.call(changes, 'ensembleConfigsV2');
    let handledByEnsemble = false;

    if (ensembleChanged) {
      const prevSig = lastEnsembleSignature;
      const prevStyleSig = lastEnsembleStyleSignature;
      await loadSettings();
      const newSig = lastEnsembleSignature;
      const newStyleSig = lastEnsembleStyleSignature;
      const classifierChanged = prevSig && newSig && prevSig !== newSig;
      const styleChanged = prevStyleSig && newStyleSig && prevStyleSig !== newStyleSig;

      if (classifierChanged) {
        reloadSettingsAndRescan();
        return;
      }

      if (styleChanged) {
        // Recalculate ensemble scores for all items using stored per-classifier results
        itemResults.forEach((entry, itemId) => {
          Object.entries(entry.ensembles || {}).forEach(([ensembleId, storedResult]) => {
            const cfg = getEnsembleById(ensembleId);
            if (cfg && storedResult && storedResult.classifiers) {
              const recalculated = recomputeWeightedEnsembleResult(cfg, storedResult);
              entry.ensembles[ensembleId] = recalculated;
            }
          });
        });
        reapplyVisualEffects();
      }

      handledByEnsemble = true;
    }

    const needsReclassify = classificationKeys.some((k) => Object.prototype.hasOwnProperty.call(changes, k));
    const needsVisualUpdate = visualKeys.some((k) => Object.prototype.hasOwnProperty.call(changes, k));
    const needsSimpleReload = performanceKeys.some((k) => Object.prototype.hasOwnProperty.call(changes, k));

    if (needsReclassify) {
      reloadSettingsAndRescan();
    } else if (needsVisualUpdate) {
      await loadSettings();
      reapplyVisualEffects();
    } else if (needsSimpleReload && !handledByEnsemble) {
      await loadSettings();
    }
  });
}

}
