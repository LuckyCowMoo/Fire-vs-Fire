# Implementation: Browser Extension

## 4.1 Content Discovery and Extraction

The browser extension employs a multi-layered detection strategy to identify and extract content from webpages in real-time, handling both static and dynamically-loaded content.

### 4.1.1 Initial Page Scan

Upon page load, the content script ([content.js](content/content.js)) performs an initial extraction of all classifiable content. The system targets two primary modalities:

**Image Discovery**: The extension queries all `<img>` elements using `document.querySelectorAll('img')` and applies filtering logic to eliminate false positives. Images are validated against multiple criteria:
- **Dimension filtering**: Images below 64×64 pixels (configurable via `minImageDimension`) are rejected to exclude UI icons and tracking pixels
- **Load state verification**: Only images with `naturalWidth > 0` and `naturalHeight > 0` are processed, ensuring the browser has fully decoded the image data
- **Alt-text filtering**: When enabled, images without alt attributes are skipped, with exceptions for known content platforms (YouTube thumbnails via `i.ytimg.com/vi/`, Google Images CDN via `encrypted-tbn*.gstatic.com`)
- **Thumbnail detection**: Images that encode to less than 45KB in base64 format are identified as thumbnails and temporarily skipped, with the element removed from `processedImages` to allow re-processing when the full-resolution image loads

**Text Discovery**: Long-form text is extracted from semantic HTML elements (`<p>`, `<li>`, `<pre>`, `<blockquote>`, `<td>`, `<th>`, `<h1>`-`<h6>`). The extraction process:
1. Retrieves `innerText` or `textContent` from each element
2. Normalizes whitespace using `.replace(/\s+/g, ' ').trim()`
3. Filters sections below the minimum character threshold (default 250 characters)
4. Deduplicates identical text blocks to prevent redundant classification
5. Excludes container elements that contain other text elements to avoid processing parent nodes

### 4.1.2 Dynamic Content Detection

Modern websites employ infinite scrolling, lazy loading, and JavaScript-driven content injection. To handle these patterns, the extension implements three complementary observers:

**MutationObserver**: Monitors the DOM for structural changes using `{ childList: true, subtree: true, attributes: true, attributeFilter: ['src'] }` configuration. When new nodes are added, the observer:
- Directly processes `<img>` tags added to the tree
- Recursively searches added subtrees using `querySelectorAll('img')` and text selectors
- Watches for `src` attribute changes on existing images to detect thumbnail-to-full-image swaps (common on Google Images and other lazy-loading platforms)
- Automatically re-processes images when their `src` changes, allowing previously-skipped thumbnails to be classified once the full-resolution version loads
- Queues discovered content for debounced classification

**IntersectionObserver**: Detects when images enter the viewport with a 200px margin (`rootMargin: '200px'`). This preemptively loads and classifies images before they become visible, reducing perceived latency. The observer triggers when `entry.isIntersecting` is true and the image has loaded (`naturalWidth > 0`).

**ResizeObserver**: Captures lazy-loaded images that change dimensions after initial render. Many websites load placeholder images (1×1 transparent pixels) that are later replaced with actual content. The ResizeObserver detects these dimension changes and triggers classification.

**Periodic Fallback Scan**: A 3-second interval scan (`setInterval`) catches edge cases missed by observers, such as images loaded via non-standard JavaScript frameworks. This scan also performs garbage collection, removing disconnected elements from tracking sets.

### 4.1.3 Data Encoding and Transmission

To minimize network overhead and avoid re-downloading images, the extension attempts to extract image data directly from the DOM using HTML5 Canvas:

```javascript
function imageToBase64(imgElement) {
  const canvas = document.createElement('canvas');
  canvas.width = imgElement.naturalWidth;
  canvas.height = imgElement.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imgElement, 0, 0);
  return canvas.toDataURL('image/png');  // Lossless PNG to preserve artifacts
}
```

This approach provides two critical advantages:
1. **Performance**: Eliminates redundant HTTP requests for images already in browser memory
2. **Artifact preservation**: Uses lossless PNG encoding to prevent JPEG compression artifacts that could interfere with frequency-domain analysis (DFT-based detection)

If canvas extraction fails due to CORS restrictions, the extension falls back to sending the image URL, allowing the backend to fetch it directly.

Text content is transmitted as UTF-8 strings with minimal metadata (selector, tag name, character length) to provide context for classification.

### 4.1.4 Batch Construction and Message Size Management

Classification requests are batched to reduce IPC overhead between the extension and native host. The `buildClassificationBatches()` function implements intelligent splitting:

- **Size limit**: Enforces a 900KB maximum per batch to stay within native messaging constraints (1MB hard limit)
- **Stable IDs**: Assigns persistent identifiers (`img-0`, `img-1`, `text-0`) that survive across multiple batches and page mutations
- **Element mapping**: Maintains a `Map<itemId, DOMElement>` to correlate classification results with their corresponding page elements

Batches exceeding the size limit are automatically split into multiple chunks, sent in parallel via `Promise.all()`, and reassembled by the backend using `jobId` and `chunkIndex` metadata.

---

## 4.2 Communication with Backend Server

The extension communicates with the local Python classification server via Firefox's Native Messaging API, establishing a persistent bidirectional channel.

### 4.2.1 Native Messaging Architecture

The background script ([background.js](background/background.js)) manages a persistent port connection to the native host:

```javascript
const HOST_NAME = 'com.aidetector.classifier';
nativePort = ext.runtime.connectNative(HOST_NAME);
```

This connection is registered via a JSON manifest file (`register-firefox-host.reg`) that maps the host name to the Python executable path. The manifest specifies:
- **Allowed extensions**: Restricts communication to the Fire vs Fire extension ID
- **Host path**: Points to the Python script wrapper (`run_host.bat`)
- **Communication type**: `stdio` for JSON message exchange

### 4.2.2 Message Protocol

The extension implements a versioned JSON protocol for classification requests:

**Request Schema (Version 2)**:
```json
{
  "version": 2,
  "type": "classifyJobChunk",
  "jobId": "job-1234567890-abc123",
  "chunkIndex": 0,
  "totalChunks": 3,
  "timestamp": 1234567890000,
  "payload": {
    "items": [
      {
        "id": "img-0",
        "modality": "image",
        "url": "data:image/png;base64,...",
        "width": 1920,
        "height": 1080
      },
      {
        "id": "text-0",
        "modality": "text",
        "text": "Lorem ipsum...",
        "length": 500
      }
    ],
    "model": {
      "ensemblesByModality": {
        "image": [{"id": "ai_images", "classifiers": ["res_net50_fft"], "weights": null}],
        "text": [{"id": "ai_text", "classifiers": ["text"], "weights": null}]
      },
      "lazyLoad": false,
      "streamResults": true
    }
  }
}
```

**Response Schema (Streamed Chunks)**:
```json
{
  "type": "classifyResultChunk",
  "jobId": "job-1234567890-abc123",
  "results": [
    {
      "id": "img-0",
      "modality": "image",
      "ensembleId": "ai_images",
      "score": 0.87,
      "label": "ai",
      "displayLabel": "AI generated"
    }
  ]
}
```

### 4.2.3 Streaming Results and Progressive Rendering

To provide immediate user feedback, the backend streams classification results as they complete rather than waiting for the entire batch. The background script forwards these chunks to the content script via `ext.tabs.sendMessage()`, allowing the UI to update incrementally.

The content script maintains a `Map<itemId, ResultData>` to accumulate results from multiple ensembles. When a new chunk arrives, `applyResultChunk()` updates the map and immediately applies visual effects to the corresponding DOM elements.

### 4.2.4 Error Handling and Reconnection

The native messaging port can disconnect due to backend crashes, extension reloads, or system resource constraints. The background script handles disconnection gracefully:

```javascript
nativePort.onDisconnect.addListener(() => {
  nativePort = null;
  // Reject all pending requests
  for (const [reqId, { reject, timeoutId }] of pendingNativeRequests) {
    clearTimeout(timeoutId);
    reject(new Error('Native host disconnected'));
  }
  pendingJobs.clear();
  activeClassifyCount = 0;
});
```

Subsequent classification requests automatically trigger reconnection via `ensureNativePort()`, which calls `ext.runtime.connectNative()` if no active port exists.

### 4.2.5 Activity Indicator

The background script tracks in-flight classification jobs using `activeClassifyCount`. When jobs are active, the extension icon changes from red (idle) to green (loading) via `ext.browserAction.setIcon()`, and a badge displays the number of concurrent jobs. This provides real-time feedback on classification activity without requiring the user to open the popup.

---

## 4.3 Options GUI and Configuration Management

The options page ([options.html](options/options.html), [options.js](options/options.js)) provides a comprehensive interface for configuring detection behavior, visual styling, and performance parameters.

### 4.3.1 Category-Based Configuration (Ensembles)

The extension introduces a "Category" abstraction (internally called "ensembles") that allows users to define multiple independent detection filters. Each category specifies:

- **Modality**: Image or text classification
- **Enabled state**: Toggle detection on/off without deleting configuration
- **Threshold**: Confidence score (0-100%) required to flag content
- **Classifiers**: One or more ML models to combine (e.g., `res_net50_fft`, `convnext_large_artifact`)
- **Weights**: Optional per-classifier weights for ensemble averaging (null = equal weighting)
- **Visual styles**: Modality-specific appearance settings

**Default Categories**:
1. **AI Images**: Detects synthetic images using ResNet50 FFT artifact analysis (threshold: 50%)
2. **AI Text**: Detects machine-generated text using TF-IDF + Logistic Regression (threshold: 50%)

### 4.3.2 Visual Style Configuration

Each category defines granular visual effects with three-state modes (Off, Hover, Always):

**Image Styles**:
- **Blur**: Gaussian blur radius (0-20px), applied via CSS `filter: blur()`
- **Border**: Colored outline with configurable multiplier (0-5×), calculated as 2% of image's smaller dimension
- **Badge**: Confidence percentage overlay positioned in top-right corner
- **Glow**: Box-shadow effect (currently unused in production)

**Text Styles**:
- **Blur**: Gaussian blur radius (0-10px)
- **Strikethrough**: Line-through decoration with configurable color
- **Underline**: Underline decoration with configurable color
- **Highlight**: Background color overlay

Colors are configured via RGB sliders (0-255 per channel) with live preview, stored as hex strings (`#FF0064`).

### 4.3.3 Performance and Debugging Settings

**Detection Parameters**:
- **Alt-text filtering**: Skip images without alt attributes (reduces false positives on UI elements)
- **Minimum text length**: Character threshold for text classification (50-2000 chars, default 250)

**Performance Tuning**:
- **Classification delay**: Debounce timer (0-500ms) to batch rapid DOM mutations
- **Image capture quality**: JPEG quality for canvas encoding (0.2-1.0, default 1.0 for lossless PNG)
- **Lazy load models**: Unload models from VRAM after each batch (reduces memory usage, increases latency)

**Developer Options**:
- **Verbose logs**: Enable detailed console output for batch splitting, GPU info, and timing metrics

### 4.3.4 Storage and Synchronization

Settings are persisted to `browser.storage.sync` (or `browser.storage.local` as fallback), allowing configuration to sync across devices. The options page implements auto-save: every input change immediately triggers `storage.set()` without requiring a "Save" button.

The content script listens for `storage.onChanged` events and implements intelligent reload logic:
- **Classification settings** (thresholds, enabled state): Trigger full page rescan
- **Visual settings** (blur, colors, category names, threshold display state): Reapply styles to existing results without re-classification
- **Performance settings** (debounce, logging): Update in-memory variables only

This approach eliminates the need for page reloads while ensuring settings changes take effect immediately.

### 4.3.5 Classifier Discovery

The options page queries available classifiers via the `LIST_MODELS` message, which the backend responds to with:
```json
{
  "classifiers": [
    {"id": "res_net50_fft", "name": "ResNet50 FFT", "modalities": ["image"]},
    {"id": "text", "name": "Text Classifier", "modalities": ["text"]},
    {"id": "convnext_large_artifact", "name": "ConvNeXt Artifact", "modalities": ["image"]}
  ]
}
```

The GUI dynamically generates checkboxes for each classifier, disabling those incompatible with the selected modality (e.g., text classifiers are disabled for image categories).

---

## 4.4 Visual Flagging System

The content script applies CSS-based visual effects to flagged content, with hover interactions and click-to-reveal functionality.

### 4.4.1 Image Flagging

Flagged images receive the `.ai-detected` class and multiple CSS properties:

**Blur Effect**:
```javascript
element.style.setProperty('filter', `blur(${blurPx}px) saturate(1) brightness(0.85)`, 'important');
element.style.setProperty('animation', 'ai-pulse 2s ease-in-out infinite', 'important');
```
The `ai-pulse` animation creates a subtle pulsing glow effect (oscillating box-shadow from 15px to 25px with rgba(255, 0, 100) color) to draw attention without being distracting.

**Border Outline**:
```javascript
const borderWidth = Math.max(2, Math.min(10, Math.round(minSide * 0.02))) * multiplier;
element.style.setProperty('outline', `${borderWidth}px solid ${borderColor}`, 'important');
element.style.setProperty('outline-offset', `-${borderWidth}px`, 'important');
```
Border width scales with image size (2% of smaller dimension, clamped 2-10px) to maintain visibility across resolutions.

**Confidence Badge**:
A dynamically-created `<span>` element is injected into the nearest positioned ancestor (figure, link, div):
```javascript
badge.textContent = `AI generated 87%`;
badge.style.background = `rgba(255, 0, 100, 0.9)`;
badge.style.position = 'absolute';
badge.style.top = '6px';
badge.style.right = '6px';
```

Badge labels are determined by:
- **Single classifier** (no ensemble or 1-classifier ensemble): Uses the label returned by the classifier (e.g., `get_model_name()`)
- **Multi-classifier ensemble**: Uses the ensemble name (from options UI) or defaults to `Category N` if name is blank

Multiple ensembles can flag the same image, resulting in stacked badges with decreasing z-index.

### 4.4.2 Text Flagging

Flagged text receives the `.ai-detected-text` class with configurable decorations:

```javascript
element.style.setProperty('text-decoration', 'line-through underline', 'important');
element.style.setProperty('text-decoration-color', '#ff0064', 'important');
element.style.setProperty('background-color', '#fff3a1', 'important');
```

Text elements support multiple simultaneous effects (blur + strikethrough + highlight), allowing users to create highly visible warnings.

### 4.4.3 Hover Interactions

The extension implements a global `pointermove` listener to handle hover states:

```javascript
document.addEventListener('pointermove', (evt) => {
  const hits = document.elementsFromPoint(evt.clientX, evt.clientY);
  const candidate = hits.find(el => el.classList.contains('ai-detected'));
  
  if (candidate !== currentHoverEl) {
    if (currentHoverEl) hideAi(currentHoverEl);  // Restore blur
    currentHoverEl = candidate;
    if (currentHoverEl) showAi(currentHoverEl);  // Remove blur
  }
}, true);
```

This approach uses `elementsFromPoint()` to detect hover through overlapping elements (e.g., transparent overlays), which standard `:hover` CSS cannot handle.

### 4.4.4 Context Menu Toggle System

The extension implements a sophisticated right-click context menu system for toggling AI detection tags, replacing the previous click-based approach to avoid interfering with normal webpage interactions.

**Context Menu Registration**: When the user right-clicks on an image or text element, the content script captures the target element and sends metadata to the background script:

```javascript
document.addEventListener('contextmenu', (evt) => {
  lastContextMenuTarget = evt.target;
  
  if (evt.target.tagName.toLowerCase() === 'img') {
    const hasVisibleAiStyling = evt.target.classList.contains('ai-detected');
    let label = 'tag';
    
    if (hasVisibleAiStyling) {
      // Extract label from badge (e.g., "AI generated 87%" -> "AI generated")
      const badgeIds = getImageBadgeIds(evt.target);
      const badge = document.getElementById(badgeIds[0]);
      const match = badge.textContent.match(/^(.+?)\s+\d+%$/);
      label = match ? match[1] : badge.textContent.split(' ')[0];
    }
    
    ext.runtime.sendMessage({ 
      type: 'UPDATE_CONTEXT_MENU', 
      hasAiTags: hasVisibleAiStyling,
      label: label,
      modality: 'image'
    });
  }
}, true);
```

The background script dynamically updates the context menu title based on the element's current state:
- **Flagged content**: "Remove [Label] tag" (e.g., "Remove AI generated tag")
- **Unflagged content**: "Add blur tag"

**Toggle Logic**: When the user selects the context menu item, the background script sends a `TOGGLE_BLUR` message back to the content script with the target element's `src` URL (for images) or selection text (for text elements). The content script then:

1. **Locates the target element** using the stored `lastContextMenuTarget` reference
2. **Checks current state** by testing for the `.ai-detected` or `.ai-detected-text` class
3. **Removes existing tags** if present:
   - Deletes the entry from `itemResults` Map
   - Calls `clearImageStyles()` or `clearTextStyles()` to remove all CSS properties
   - Removes badge overlays from the DOM
   - Clears the hover handler reference if applicable
4. **Adds manual blur tag** if not present:
   - Creates a temporary "manual_blur" ensemble configuration with gray border color (#808080)
   - Assigns a new `itemId` if the element wasn't previously tracked
   - Adds an entry to `itemResults` with score 1.0 and label "Blur"
   - Calls `applyStylesForItem()` to render the blur effect

**Manual Blur Ensemble**: Manual tags use a temporary ensemble configuration that mimics AI-detected styling but with distinct visual markers:

```javascript
const manualEnsemble = {
  id: 'manual_blur',
  modality: 'image',
  enabled: true,
  threshold: 0.5,
  styles: {
    image: {
      blurAmount: settings.blurAmount ?? 4,
      blurMode: 'hover',
      borderColor: '#808080',  // Gray to distinguish from AI-detected (red)
      borderMode: 'hover',
      badgeMode: 'hover'
    }
  }
};
```

This ensemble is temporarily injected into the settings during style application but not persisted to storage, ensuring manual tags don't interfere with saved configurations.

**State Synchronization**: The toggle operation returns the new state to the background script, which updates the context menu title for subsequent right-clicks without requiring a page refresh. This creates a seamless toggle experience where the menu text accurately reflects the current state ("Add blur tag" ↔ "Remove Blur tag").

**Text Element Support**: The system extends to text elements by checking for valid semantic tags (`<p>`, `<li>`, `<h1>`-`<h6>`, etc.) and applying text-specific blur styling without strikethrough or highlight effects, making manual text blurring less visually intrusive than AI-detected text.

### 4.4.5 Mode-Based Rendering

Visual effects respect the configured mode (Off, Hover, Always):

- **Off**: No effects applied, element appears normal
- **Hover**: Effects applied by default, removed on hover (allows inspection)
- **Always**: Effects permanently applied, hover has no effect

The mode is stored in `element.dataset.aiImageStyle` as JSON, allowing the hover handler to restore the correct state when the cursor leaves.

### 4.4.6 Performance Optimizations

**Debounced Style Application**: Style changes are batched and applied once per classification chunk rather than per-item, reducing layout thrashing.

**Element Connectivity Checks**: Before applying styles, the script verifies `element.isConnected` to skip elements removed from the DOM (e.g., infinite scroll removing off-screen content).

**Badge Reuse Prevention**: Existing badges are removed before creating new ones to prevent duplicate overlays when ensembles are reconfigured.

---

## 4.5 Technical Challenges and Solutions

### 4.5.1 CORS and Canvas Tainting

Cross-origin images cannot be drawn to canvas without CORS headers, causing `toDataURL()` to throw a security exception. The extension handles this gracefully by catching the exception and falling back to URL-based fetching in the backend.

### 4.5.2 Infinite Scroll and Memory Management

Websites like Twitter and Reddit continuously inject new content while removing off-screen elements. The extension implements garbage collection in the periodic scan, removing disconnected elements from `processedImages` and `processedTextElements` sets to prevent memory leaks.

### 4.5.3 YouTube and Dynamic Platforms

YouTube's heavy use of JavaScript and shadow DOM requires special handling. The extension:
- Exempts YouTube thumbnails (`i.ytimg.com`) from alt-text filtering
- Uses `MutationObserver` with `subtree: true` to catch content injected into shadow roots
- Implements a 3-second fallback scan to catch content loaded via non-standard frameworks

### 4.5.4 Message Size Limits

Native messaging enforces a 1MB message size limit. The extension implements automatic batch splitting at 900KB (leaving 100KB margin for JSON overhead), sending chunks in parallel and reassembling them in the backend using `jobId` and `chunkIndex` metadata.

---

## 4.6 Summary

The browser extension implements a sophisticated content discovery and flagging system that handles both static and dynamic web content. Key innovations include:

1. **Multi-observer architecture** combining MutationObserver, IntersectionObserver, and ResizeObserver for comprehensive content detection
2. **Canvas-based image extraction** to eliminate redundant network requests and preserve artifact data
3. **Streaming classification results** for immediate user feedback
4. **Category-based configuration** allowing users to define multiple independent detection filters
5. **Mode-based visual effects** (Off/Hover/Always) providing flexible content flagging
6. **Intelligent batch splitting** to respect native messaging constraints while maximizing throughput

The extension achieves its design goal of fully automatic, real-time detection without requiring user interaction, while providing extensive customization options for power users.
