# AI Detection Extension - Code Flow Guide

## Overview

The extension detects AI-generated images and text on web pages, flags them with visual effects (blur for images, strikethrough for text), and allows users to hover to reveal content.

**Architecture**:

- **content.js** (Content Script) - Runs on every page, detects & classifies content
- **background.js** (Service Worker/Background Page) - Manages native messaging bridge
- **native/echo_host_V2.py** (Native Host) - Python orchestrator for ML classifiers

---

## content.js - Main Detection & UI Logic

### Global State

```javascript
// Settings - synced from options page
settings = {
  enabled,
  altTextOnly,
  blurAmount,
  borderMultiplier,
  borderColor,
  imageAiThreshold,
  textAiThreshold,
};

// Page data
cachedImages = []; // Detected images on page
cachedTextSections = []; // Detected text sections on page
elementMap = Map(); // itemId -> DOM element (maps IDs to elements for later retrieval)

// Observers
mutationObserver; // Watches for new DOM nodes
resizeObserver; // Watches for lazy-loaded images

// ID Counters (STABLE - never reset to prevent collisions)
nextImageId = 0; // Increments: img-0, img-1, img-2...
nextTextId = 0; // Increments: text-0, text-1, text-2...
```

### Main Execution Flow

```
PAGE LOADS
    ↓
initAutoExtract()
    ↓
performInitialExtraction()
    ├→ scanAllImages()          → finds all <img> on page → cachedImages
    ├→ extractLongText()        → finds text >=250 chars → cachedTextSections
    └→ requestClassification()  → sends to native host for ML classification
         ↓
    buildClassificationBatches()    → assigns stable IDs (img-0, text-0, etc)
         ↓
    ext.runtime.sendMessage() → sends to background.js
         ↓
    background.js → sendToNative() → echo_host_V2.py
         ↓
    CLASSIFICATION RESULTS (scores 0-1)
         ↓
    applyVisualEffects()
        ├→ Blur images where score >= imageAiThreshold
        ├→ Strikethrough text where score >= textAiThreshold
        └→ attachGlobalHoverHandler() → monitor mouse position for reveals

+ startDynamicDetection()
    ├→ initMutationObserver()    → watches for new DOM nodes
    ├→ initResizeObserver()      → watches for lazy-loaded images
    └→ scanAllImages() every 5s  → periodic fallback
```

### Function Reference

#### **Initialization Functions** (called once on page load)

| Function                     | Called By            | Purpose                                                   |
| ---------------------------- | -------------------- | --------------------------------------------------------- |
| `initAutoExtract()`          | Auto (end of script) | Entry point - waits for DOM ready, loads settings         |
| `performInitialExtraction()` | initAutoExtract      | Scans page for images and text, starts classification     |
| `startDynamicDetection()`    | initAutoExtract      | Sets up observers for dynamic content (scroll, lazy-load) |
| `loadSettings()`             | initAutoExtract      | Loads user settings from chrome.storage.sync              |

#### **Image Detection Functions**

| Function                          | Called By                                                  | Purpose                                       | Returns                    |
| --------------------------------- | ---------------------------------------------------------- | --------------------------------------------- | -------------------------- |
| `scanAllImages()`                 | performInitialExtraction, startDynamicDetection (every 5s) | Scans all `<img>` on page                     | Array of new image objects |
| `processImageElement(el, isLazy)` | scanAllImages, initMutationObserver, initResizeObserver    | Validates single image, prevents duplicates   | Image object or null       |
| `imageToBase64(imgElement)`       | buildClassificationBatches                                 | Converts img to data URL to avoid re-download | Base64 string or null      |

#### **Text Detection Functions**

| Function            | Called By                | Purpose                          | Returns                       |
| ------------------- | ------------------------ | -------------------------------- | ----------------------------- |
| `extractLongText()` | performInitialExtraction | Finds text sections >= 250 chars | Array of text section objects |

#### **Classification & Batching**

| Function                         | Called By                                               | Purpose                                                            |
| -------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------ |
| `scheduleClassification(images)` | initMutationObserver, initResizeObserver, scanAllImages | Debounces image detection (500ms) before sending to native         |
| `requestClassification()`        | performInitialExtraction, scheduleClassification        | Sends cachedImages + cachedTextSections for classification         |
| `buildClassificationBatches()`   | requestClassification                                   | Splits items into batches, assigns STABLE IDs (img-0, text-0, etc) |

**ID Assignment Logic in `buildClassificationBatches()`:**

```javascript
// Check if image already has an ID in elementMap
let itemId = elementMap.get(img.element);
if (!itemId) {
  // New image - assign fresh ID using global counter
  itemId = "img-" + nextImageId++;
  elementMap.set(itemId, img.element);
}
// Result: Same image always gets same ID across multiple batches → no ID collisions on scroll
```

#### **Visual Effects Functions**

| Function                       | Called By                                    | Purpose                                                    |
| ------------------------------ | -------------------------------------------- | ---------------------------------------------------------- |
| `applyVisualEffects(response)` | requestClassification                        | Processes native host response, applies blur/strikethrough |
| `hideAi(el)`                   | applyVisualEffects, attachGlobalHoverHandler | Blur image + show badge                                    |
| `showAi(el)`                   | attachGlobalHoverHandler                     | Unblur image + hide badge                                  |
| `computeBorderWidth(el)`       | applyVisualEffects                           | Calculate responsive border width                          |
| `getOverlayContainer(imgEl)`   | applyVisualEffects                           | Find best parent for badge placement                       |

#### **Interaction Functions**

| Function                     | Called By             | Purpose                                                    |
| ---------------------------- | --------------------- | ---------------------------------------------------------- |
| `attachGlobalHoverHandler()` | applyVisualEffects    | Attach global mousemove listener for image reveal/hide     |
| `initMutationObserver()`     | startDynamicDetection | Watch for new DOM nodes → call scheduleClassification      |
| `initResizeObserver()`       | startDynamicDetection | Watch for lazy-loaded images → call scheduleClassification |

#### **Settings Management**

| Function                    | Called By                  | Purpose                                  |
| --------------------------- | -------------------------- | ---------------------------------------- |
| `reloadSettingsAndRescan()` | storage.onChanged listener | Re-run detection if user changes options |
| `resetObservers()`          | reloadSettingsAndRescan    | Clean up old observers                   |
| `resetState()`              | reloadSettingsAndRescan    | Clear cached images/text/maps            |

---

## background.js - Native Messaging Bridge

### Purpose

Acts as intermediary between content script (web context) and native Python host (system context).
Handles request/response wrapping, error handling, model selection.

### Key Functions

| Function                                      | Called By                      | Purpose                                                      |
| --------------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| `buildClassifyRequest(items, modelOverrides)` | handleClassifyItems            | Wraps items in official request envelope (version 1 format)  |
| `sendToNative(message)`                       | handleClassifyItems            | Sends to native host via ext.runtime.sendNativeMessage()     |
| `handleClassifyItems(message)`                | ext.runtime.onMessage listener | Receives from content.js, routes to native, returns response |
| `makeRequestId()`                             | buildClassifyRequest           | Generates UUID for request tracking                          |

### Message Flow

```
Content Script (content.js)
    │
    └→ ext.runtime.sendMessage({
           type: 'CLASSIFY_ITEMS',
           items: [...batch...]
       })

    ↓ (crosses boundary to background context)

Background Script (background.js)
    │
    ├→ handleClassifyItems()
    │  └→ buildClassifyRequest()
    │     └→ sendToNative()
    │        └→ ext.runtime.sendNativeMessage('com.aidetector.classifier', envelope)
    │
    ↓ (crosses boundary to native context)

Native Host (echo_host_V2.py)
    │
    └→ classify_items()
       ├→ fetch_images_from_urls()
       ├→ fetch_text_from_items()
       └→ ensemble.classify_batch()

    ↓ (returns to native message handler)

Background Script
    │
    └→ (response sent back through promise)

Content Script
    │
    └→ applyVisualEffects(response)
```

### Native Host Setup

Native messaging uses manifest files:

- **Chrome**: `manifest.json` → `"host": "native/echo_host_V2.py"`
- **Firefox**: `manifest.firefox.json` → same reference

The host is launched by the browser when first message is sent, stays alive for session.

---

## echo_host_V2.py - Classification Pipeline

### Architecture

```
NATIVE MESSAGING LOOP
    │
    └→ main()
       └→ read_message() → parse JSON

          if type == 'classify':
              └→ classify_items(envelope)
                 ├─ Phase 1a: fetch_images_from_urls()  → download images from URLs
                 ├─ Phase 1b: fetch_text_from_items()   → extract text (already in browser)
                 ├─ Phase 2:  EnsembleClassifier.__init__()
                 ├─ Phase 3:  classify_batch(images, modality='image')
                 │            └→ For each classifier:
                 │               ├─ classifier.load_model()
                 │               ├─ classifier.process_batch(images)
                 │               └─ return scores [0-1]
                 │            └→ Weighted ensemble average
                 └─ Phase 4:  classify_batch(texts, modality='text')
                              └→ Same as Phase 3

              └→ send_message(response)  → send JSON back to browser
```

### Key Functions

| Function                                              | Purpose             | Input                       | Output                                   |
| ----------------------------------------------------- | ------------------- | --------------------------- | ---------------------------------------- |
| `classify_items(envelope)`                            | Main orchestrator   | Request envelope with items | Response with results                    |
| `fetch_images_from_urls(image_items)`                 | Fetch remote images | List of {id, url}           | List of (id, PIL.Image, error, duration) |
| `fetch_text_from_items(text_items)`                   | Extract text items  | List of {id, text}          | List of (id, text, None, 0)              |
| `EnsembleClassifier.classify_batch(inputs, modality)` | Run all classifiers | Images/texts, modality      | (results, device_info)                   |

### Classifier Integration

```python
# Model registry auto-discovers classifiers in classifiers/ folder
classifier = registry.get_classifier('res_net50_fft')  # or 'text'

# Each classifier supports specific modalities
classifier.process_batch(inputs, modality='image')  # → scores [0-1]

# Ensemble combines scores from multiple classifiers
# If image classifier fails: score = -1.0 (filtered out)
# Weighted average: score = sum(scores[i] * weights[i]) / sum(weights)
```

### Current Classifiers

1. **ResNet50FFT** (`res_net50_fft`)
   - Modalities: `{'image'}`
   - Input: PIL Images
   - Output: Scores in [0, 1] where 1.0 = AI-generated

2. **TextClassifier** (`text`)
   - Modalities: `{'text'}`
   - Input: Text strings (≥10 chars)
   - Output: Scores in [0, 1] where 1.0 = AI-generated
   - Uses: TF-IDF + Logistic Regression (scikit-learn)

### Result Format

```python
{
  'version': 1,
  'type': 'classifyResult',
  'requestId': 'req-xxx',
  'timestamp': 1234567890,
  'results': [
    {
      'id': 'img-0',
      'modality': 'image',
      'label': 'ai' | 'real' | 'uncertain',    # label based on score >= 0.5
      'score': 0.92,                            # 0-1 confidence
      'model': 'res_net50_fft,text',            # classifiers used
      'durationMs': 45
    },
    ...
  ],
  'errors': [
    {'type': 'info', 'message': '[Native Host V2] Processing 42 image(s)...'},
    {'type': 'fetch_timing', 'successful': 41, 'durationMs': 1230}
  ]
}
```

### Threshold Logic

**Current (Fixed):**

```python
# In ensemble: if avg_score >= 0.5 → label = 'ai'
label = 'ai' if avg_score >= 0.5 else 'real'
```

**Content Script (User Threshold):**

```javascript
// Only flag if BOTH native score AND user threshold met
if (result.score >= settings.imageAiThreshold) {
  // e.g., 0.75 = 75%
  // Apply blur
}
```

---

## Data Flow Example: Scroll Detection

```
USER SCROLLS ON GOOGLE IMAGES
    │
    ├→ New images enter viewport
    ├→ Browser renders them
    └→ Dimensions change (naturalWidth from 0 → actual size)

    ResizeObserver detects size change
    │
    ├→ processImageElement(img) → validates
    ├→ scheduleClassification([img])
    │
    └→ 500ms debounce timeout
       │
       ├→ Add img to cachedImages
       ├→ requestClassification()
       │
       ├→ buildClassificationBatches()
       │  └→ Assign ID: itemId = 'img-' + nextImageId++
       │  └→ elementMap.set('img-42', imgElement)  ← STABLE MAPPING
       │
       ├→ ext.runtime.sendMessage({type: 'CLASSIFY_ITEMS', items: [{id: 'img-42', ...}]})
       │
       ├→ background.js → native host
       │
       └→ RESPONSE: [{id: 'img-42', score: 0.87, ...}]

       applyVisualEffects()
       │
       └→ element = elementMap.get('img-42')  ← RETRIEVES CORRECT ELEMENT
          └→ hideAi(element)  ← BLURS CORRECT IMAGE
```

**Why this works:**

- Each element gets a unique, persistent ID using `nextImageId++` global counter
- `elementMap` is never cleared, only appended to
- Results always map to correct elements, no ID collisions

---

## Settings & Storage

### Storage Keys (Chrome Storage API)

```javascript
// Synced across devices (if user signed in to Chrome)
chrome.storage.sync.set({
  enabled: true,
  altTextOnly: false,
  blurAmount: 4,
  borderMultiplier: 1,
  borderColor: "#ff0064",
  miniBatchSize: 10,
  imageAiThreshold: 0.5, // 0-1 range (0.5 = 50% threshold)
  textAiThreshold: 0.5,
});
```

### Settings Change Detection

```javascript
// Listener attached to storage.onChanged
ext.storage.onChanged.addListener((changes) => {
  if (changes has 'enabled' or 'blurAmount' or similar) {
    reloadSettingsAndRescan()  // Reset observers, re-extract
  }
})
```

---

## Error Handling

### Content Script

- Try/catch on canvas extraction (CORS failures)
- Settings load failures silently default to hardcoded values
- Storage API availability check

### Background Script

- Feature-detect native messaging availability
- Parse native host errors
- Return `{ok: false, error: message}`

### Native Host

- HTTP/URL errors caught (timeout, 404, etc)
- Image load errors (corrupt file, wrong format)
- Model load errors (file missing, incompatible)
- Falls back to uncertain score (0.5) on errors

---

## Performance Optimizations

1. **Image Base64 Encoding**
   - Try to encode image from DOM canvas (avoids re-download)
   - Fall back to original URL if CORS fails

2. **Debounced Detection**
   - 500ms delay on mutation → batches multiple images
   - Prevents flooding native host with small requests

3. **Lazy Loading Classifiers**
   - Models only loaded when needed
   - Unloaded after batch → frees VRAM

4. **Mini-batching**
   - Large requests split if > 900KB to avoid message size limit

5. **Stable ID Counters**
   - No ID collisions on scroll
   - Prevents incorrect blur mapping

---

## Debugging

### Console Logs

Content script prefixes: `[AI Detector]`, `[Extension]`
Native host log file: `native_host_v2.log` in project root

### Check Current State

```javascript
// In browser console on any page with extension active:
elementMap; // See all item ID → element mappings
cachedImages; // See detected images
settings; // See user settings
nextImageId; // See current counter value
```

### Check Native Host

```bash
# View logs
cat native_host_v2.log | tail -100

# Test native host directly
python native/echo_host_V2.py  # Reads from stdin, expect JSON message
```

---

## Future Improvements

1. **Per-classifier weighting** - Let user control ResNet50 vs Text classifier weight
2. **Batch size tuning** - Auto-detect optimal batch size for GPU
3. **Cache model predictions** - Skip re-classifying same image/text
4. **Confidence scoring UI** - Show breakdown (ResNet: 0.92, Text: 0.65 → avg: 0.785)
5. **Whitelist/blacklist** - Let user manually mark content as real or AI
