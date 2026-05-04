/* global browser, chrome */
const ext = typeof browser !== 'undefined' ? browser : chrome;

console.log('[Background] Script loaded, ext:', ext ? 'Available' : 'Unavailable');

// Name of the native messaging host (update to match your installed host manifest)
const HOST_NAME = 'com.aidetector.classifier';

// Default model hints sent with requests; the native app can ignore or override
const DEFAULT_MODELS = { image: 'local-img-v1', text: 'local-text-v1' };

let selectedImageModel = null;
const STORAGE_KEY = 'selectedImageModel';
let activeClassifyCount = 0;

function setBadgeLoading(activeCount) {
  const badgeApi = ext.browserAction || ext.action;
  if (!badgeApi) return;

  try {
    if (typeof badgeApi.setBadgeText === 'function') {
      const text = activeCount > 99 ? '99+' : (activeCount > 0 ? String(activeCount) : '');
      badgeApi.setBadgeText({ text });
    }
    if (typeof badgeApi.setBadgeBackgroundColor === 'function') {
      badgeApi.setBadgeBackgroundColor({ color: activeCount > 0 ? '#18A34A' : '#C02B2B' });
    }
  } catch (e) {
    console.warn('[Badge] Failed to update badge state:', e);
  }
}

function updateToolbarState() {
  const isLoading = activeClassifyCount > 0;
  setIconLoading(isLoading);
  setBadgeLoading(activeClassifyCount);
}

function beginClassificationActivity() {
  activeClassifyCount += 1;
  updateToolbarState();
}

function endClassificationActivity() {
  if (activeClassifyCount > 0) {
    activeClassifyCount -= 1;
  }
  updateToolbarState();
}

// Helper function to change the extension icon
function setIconLoading(isLoading) {
  const filename = isLoading ? 'icon-loading.svg' : 'icon.svg';
  const fullPath = ext.runtime.getURL('icons/' + filename);
  console.log(`[Icon] Changing to ${filename}`);
  console.log(`[Icon] Full path: ${fullPath}`);
  
  try {
    if (ext.browserAction && typeof ext.browserAction.setIcon === 'function') {
      console.log('[Icon] Using browserAction.setIcon()');
      ext.browserAction.setIcon({ path: fullPath }, () => {
        if (ext.runtime.lastError) {
          console.error('[Icon] setIcon error:', ext.runtime.lastError);
        } else {
          console.log('[Icon] Icon set successfully');
        }
      });
    } else if (ext.action && typeof ext.action.setIcon === 'function') {
      console.log('[Icon] Using action.setIcon()');
      ext.action.setIcon({ path: fullPath }, () => {
        if (ext.runtime.lastError) {
          console.error('[Icon] setIcon error:', ext.runtime.lastError);
        } else {
          console.log('[Icon] Icon set successfully');
        }
      });
    } else {
      console.warn('[Icon] No icon API found. Available:', Object.keys(ext));
    }
  } catch (e) {
    console.error('[Icon] Exception:', e.message, e.stack);
  }
}

// Development aid: set to true to simulate native responses when the
// native messaging API isn't available yet.
const ENABLE_FAKE_NATIVE = false;

// Create a UUID without pulling in extra deps
function makeRequestId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'req-' + Math.random().toString(16).slice(2) + Date.now();
}

// Build a classify request envelope following the agreed schema
function buildClassifyRequest(items, modelOverrides) {
  const model = { ...DEFAULT_MODELS, ...(modelOverrides || {}) };
  return {
    version: 1,
    type: 'classify',
    requestId: makeRequestId(),
    timestamp: Date.now(),
    payload: {
      items,
      model
    }
  };
}

function persistSelectedModel(value) {
  if (!ext.storage || !ext.storage.local) return;
  try {
    ext.storage.local.set({ [STORAGE_KEY]: value });
  } catch (err) {
    console.warn('Failed to persist selected model', err);
  }
}

function loadStoredModel() {
  if (!ext.storage || !ext.storage.local) return;
  try {
    ext.storage.local.get(STORAGE_KEY, (result) => {
      if (ext.runtime.lastError) {
        console.warn('Storage get failed', ext.runtime.lastError);
        return;
      }
      if (result && result[STORAGE_KEY]) {
        selectedImageModel = result[STORAGE_KEY];
        console.log('Restored selected image model:', selectedImageModel);
      }
    });
  } catch (err) {
    console.warn('Failed to load stored model', err);
  }
}

loadStoredModel();
updateToolbarState();

// Native messaging persistent port + request tracking
let nativePort = null;
const pendingNativeRequests = new Map();
const pendingJobs = new Map();

function ensureNativePort() {
  if (nativePort) return nativePort;

  const canConnect = ext && ext.runtime && typeof ext.runtime.connectNative === 'function';
  if (!canConnect) {
    return null;
  }

  nativePort = ext.runtime.connectNative(HOST_NAME);

  nativePort.onMessage.addListener((response) => {
    const resType = response && response.type;
    const reqId = response && response.requestId;

    // Handle job cancellation confirmation
    if (resType === 'jobCancelled') {
      const jobId = response.jobId;
      console.log(`[Native] Job ${jobId} cancellation confirmed`);
      pendingJobs.delete(jobId);
      return;
    }

    if (resType === 'classifyResultChunk' || resType === 'classifyJobComplete') {
      const jobId = response.jobId;
      const jobInfo = jobId ? pendingJobs.get(jobId) : null;
      if (!jobInfo) {
        console.warn('[Native] Unmatched job response:', response);
        return;
      }
      const { tabId } = jobInfo;
      if (tabId != null && ext.tabs && ext.tabs.sendMessage) {
        ext.tabs.sendMessage(tabId, {
          type: resType === 'classifyResultChunk' ? 'CLASSIFY_RESULT_CHUNK' : 'CLASSIFY_JOB_COMPLETE',
          payload: response
        }).catch(() => {});
      }
      if (resType === 'classifyJobComplete') {
        pendingJobs.delete(jobId);
        endClassificationActivity();
      }
      return;
    }

    if (!reqId || !pendingNativeRequests.has(reqId)) {
      console.warn('[Native] Unmatched response:', response);
      return;
    }
    const { resolve, timeoutId } = pendingNativeRequests.get(reqId);
    if (timeoutId) clearTimeout(timeoutId);
    pendingNativeRequests.delete(reqId);
    console.log('[Native] Response from host:', response);
    resolve(response);
  });

  nativePort.onDisconnect.addListener(() => {
    const err = ext.runtime.lastError;
    console.warn('[Native] Port disconnected', err);
    nativePort = null;

    // Reject all pending requests
    for (const [reqId, { reject, timeoutId }] of pendingNativeRequests) {
      if (timeoutId) clearTimeout(timeoutId);
      reject(new Error(err?.message || 'Native host disconnected'));
      pendingNativeRequests.delete(reqId);
    }

    if (pendingJobs.size > 0) {
      pendingJobs.clear();
      activeClassifyCount = 0;
      updateToolbarState();
    }
  });

  return nativePort;
}

// Send a message to the native host and return a promise for its response
function sendToNative(message) {
  // If dev mode is enabled, always use fake responses
  if (ENABLE_FAKE_NATIVE) {
    return new Promise((resolve) => {
      setTimeout(() => {
        const results = (message.payload?.items || []).map((it) => ({
          id: it.id,
          modality: it.modality,
          label: 'uncertain',
          score: 0.5,
          model: (message.payload?.model || {})[it.modality] || 'dev-fake',
          durationMs: 5,
          notes: 'Fake native response (dev)'
        }));
        resolve({
          version: 1,
          type: 'classifyResult',
          requestId: message.requestId,
          timestamp: Date.now(),
          results,
          errors: []
        });
      }, 25);
    });
  }

  const port = ensureNativePort();
  if (port) {
    return new Promise((resolve, reject) => {
      const requestId = message.requestId || makeRequestId();
      message.requestId = requestId;

      const timeoutId = setTimeout(() => {
        if (pendingNativeRequests.has(requestId)) {
          pendingNativeRequests.delete(requestId);
          reject(new Error('Native host timeout'));
        }
      }, 60000);

      pendingNativeRequests.set(requestId, { resolve, reject, timeoutId });
      console.log('[Native] Sending to host (port):', HOST_NAME, message);
      port.postMessage(message);
    });
  }

  // Fallback to one-off sendNativeMessage if connectNative isn't available
  const canSend = ext && ext.runtime && typeof ext.runtime.sendNativeMessage === 'function';
  if (!canSend) {
    return Promise.reject(new Error('Native messaging API unavailable: ext.runtime.connectNative/sendNativeMessage not available. Ensure nativeMessaging permission, MV2 background page, and a supported browser.'));
  }

  return new Promise((resolve, reject) => {
    console.log('[Native] Sending to host:', HOST_NAME, message);
    ext.runtime.sendNativeMessage(HOST_NAME, message, (response) => {
      const err = ext.runtime.lastError;
      if (err) {
        console.error('[Native] Error from host:', err);
        reject(new Error(`Native host error: ${err.message || JSON.stringify(err)}`));
        return;
      }
      console.log('[Native] Response from host:', response);
      resolve(response);
    });
  });
}

function sendToNativeFireAndForget(message) {
  const port = ensureNativePort();
  if (port) {
    console.log('[Native] Sending to host (port):', HOST_NAME, message);
    port.postMessage(message);
    return true;
  }

  const canSend = ext && ext.runtime && typeof ext.runtime.sendNativeMessage === 'function';
  if (!canSend) return false;

  ext.runtime.sendNativeMessage(HOST_NAME, message, () => {
    const err = ext.runtime.lastError;
    if (err) {
      console.warn('[Native] sendNativeMessage error:', err);
    }
  });
  return true;
}

// Handle CLASSIFY_ITEMS messages from content/popup
async function handleClassifyItems(message) {
  console.log('[Classify] handleClassifyItems called', message);
  const items = Array.isArray(message.items) ? message.items : [];
  if (!items.length) {
    return { ok: false, error: 'No items to classify' };
  }

  const modelOverrides = { ...message.model };

  if (selectedImageModel) {
    // V2 expects classifier IDs
    modelOverrides.classifiers = [selectedImageModel];
  }
  
  // Pass lazyLoad setting from content script
  if (typeof message.lazyLoad === 'boolean') {
    modelOverrides.lazyLoad = message.lazyLoad;
  }

  const envelope = buildClassifyRequest(items, modelOverrides);

  try {
    console.log('[Classify] Setting icon to loading');
    beginClassificationActivity();
    const nativeResponse = await sendToNative(envelope);
    console.log('[Classify] Setting icon back to idle');
    endClassificationActivity();
    return { ok: true, requestId: envelope.requestId, response: nativeResponse };
  } catch (err) {
    console.error('[Classify] Error, setting icon back to idle:', err);
    endClassificationActivity();
    return { ok: false, requestId: envelope.requestId, error: err.message || String(err) };
  }
}

function handleClassifyJobChunk(message, sender) {
  const jobId = message.jobId || makeRequestId();
  const tabId = sender && sender.tab ? sender.tab.id : null;
  if (tabId != null) {
    if (!pendingJobs.has(jobId)) {
      pendingJobs.set(jobId, { tabId, createdAt: Date.now() });
      beginClassificationActivity();
    }
  }

  const payload = {
    items: Array.isArray(message.items) ? message.items : [],
    model: message.model || {}
  };

  const ok = sendToNativeFireAndForget({
    version: 2,
    type: 'classifyJobChunk',
    jobId,
    chunkIndex: message.chunkIndex || 0,
    totalChunks: message.totalChunks || 1,
    timestamp: Date.now(),
    payload
  });

  if (!ok && pendingJobs.has(jobId)) {
    pendingJobs.delete(jobId);
    endClassificationActivity();
  }

  return { ok, jobId };
}

ext.runtime.onInstalled.addListener(() => {
  console.log('Extension installed');
  updateToolbarState();
  if (ext.contextMenus && ext.contextMenus.create) {
    try {
      // Context menu for images and all page content
      ext.contextMenus.create({
        id: 'toggle-blur',
        title: 'Toggle AI blur',
        contexts: ['image', 'page']
      });
    } catch (e) {
      console.warn('Context menu create failed:', e);
    }
  }
});

if (ext.contextMenus && ext.contextMenus.onClicked) {
  ext.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'toggle-blur' && tab && tab.id != null) {
      // Send message to content script and wait for response to update menu
      const message = { type: 'TOGGLE_BLUR' };
      
      if (info.srcUrl) {
        // Image context
        message.srcUrl = info.srcUrl;
      } else if (info.selectionText) {
        // Text context
        message.selectionText = info.selectionText;
      }
      
      ext.tabs.sendMessage(tab.id, message)
        .then((response) => {
          // Update context menu based on new state
          if (response && ext.contextMenus && ext.contextMenus.update) {
            let title;
            if (response.hasAiTags) {
              // Just added tags, so next time show remove option
              const label = response.label || 'tag';
              title = `Remove ${label} tag`;
            } else {
              // Just removed tags, so next time show blur option
              const modality = response.modality || 'this';
              title = modality === 'text' ? 'Blur this text' : 'Blur this image';
            }
            ext.contextMenus.update('toggle-blur', { title }).catch(() => {});
          }
        })
        .catch(() => {});
    }
  });
}

// Test function - call from console: testIconChange()
window.testIconChange = function() {
  console.log('[Test] Changing icon to loading (green)');
  setIconLoading(true);
  setTimeout(() => {
    console.log('[Test] Changing icon back to idle (red)');
    setIconLoading(false);
  }, 3000);
};

ext.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === 'PING') {
    sendResponse({ ok: true, time: Date.now() });
    return false;
  }

  if (message && message.type === 'UPDATE_CONTEXT_MENU') {
    if (ext.contextMenus && ext.contextMenus.update) {
      let title;
      if (message.hasAiTags) {
        const label = message.label || 'tag';
        title = `Remove ${label} tag`;
      } else {
        const modality = message.modality || 'image';
        title = modality === 'text' ? 'Blur this text' : 'Blur this image';
      }
      ext.contextMenus.update('toggle-blur', { title }).catch(() => {});
    }
    sendResponse({ ok: true });
    return false;
  }

  if (message && message.type === 'GET_ACTIVITY_STATUS') {
    sendResponse({ ok: true, isLoading: activeClassifyCount > 0, activeCount: activeClassifyCount });
    return false;
  }

  if (message && message.type === 'CLASSIFY_ITEMS') {
    handleClassifyItems(message)
      .then((result) => sendResponse(result))
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true; // async response
  }

  if (message && message.type === 'CLASSIFY_JOB_CHUNK') {
    const result = handleClassifyJobChunk(message, sender);
    sendResponse(result);
    return false;
  }

  if (message && message.type === 'LIST_MODELS') {
    sendToNative({ type: 'listModels', version: 2 })
      .then((response) => {
        const rawClassifiers = Array.isArray(response.classifiers) ? response.classifiers : [];
        const models = rawClassifiers.length
          ? rawClassifiers.map((m) => (typeof m === 'string' ? m : m.id)).filter(Boolean)
          : (response.models || response.model_files || []);
        sendResponse({ ok: true, models, raw: response });
      })
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }

  if (message && message.type === 'SET_MODEL') {
    selectedImageModel = message.model || null;
    persistSelectedModel(selectedImageModel);
    sendResponse({ ok: true, selected: selectedImageModel });
    return false;
  }

  if (message && message.type === 'GET_MODEL') {
    sendResponse({ ok: true, selected: selectedImageModel });
    return false;
  }

  if (message && message.type === 'CANCEL_ALL') {
    // Cancel all active native host jobs
    for (let jobId of pendingJobs.keys()) {
      try {
        const port = ensureNativePort();
        if (port) {
          port.postMessage({
            type: 'cancelJob',
            jobId: jobId,
            timestamp: Date.now()
          });
        }
      } catch (e) {
        console.warn(`[Background] Failed to cancel job ${jobId}:`, e);
      }
    }
    pendingJobs.clear();
    activeClassifyCount = 0;
    updateToolbarState();
    sendResponse({ ok: true });
    return false;
  }

  return false;
});
