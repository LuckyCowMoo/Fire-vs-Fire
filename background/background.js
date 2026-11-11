/* global browser, chrome */
const ext = typeof browser !== 'undefined' ? browser : chrome;

console.log('[Background] Script loaded, ext:', ext ? 'Available' : 'Unavailable');

// Name of the native messaging host (update to match your installed host manifest)
const HOST_NAME = 'com.aidetector.classifier';
const GITHUB_PAGES_ONBOARDING_URL = 'https://luckycowmoo.github.io/Fire-vs-Fire/?onboarding=1';
const RAW_GITHUB_CATALOG_URL = 'https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/FVF-clean/native/official_models.json';

// Unique source identifier for this browser profile/session (supports multi-window & future LAN routing)
const SOURCE_ID = 'browser-' + Math.random().toString(16).slice(2, 10) + '-' + Date.now();

// Safe packet chunk size (450 KB) so individual messages stay well under Firefox/Chrome's 1 MB Native Messaging limit
const MAX_PACKET_CHUNK_CHARS = 450 * 1024;

// Exact count of items remaining to be classified (queued + in-flight) reported by Native Host
let activeClassifyCount = 0;

function setBadgeLoading(activeCount) {
  const badgeApi = ext.browserAction || ext.action;
  if (!badgeApi) return;

  try {
    if (typeof badgeApi.setBadgeText === 'function') {
      const text = activeCount > 999 ? '999+' : (activeCount > 0 ? String(activeCount) : '');
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

function setExactRemainingItemCount(remainingCount) {
  const count = Math.max(0, Number(remainingCount) || 0);
  if (count !== activeClassifyCount) {
    activeClassifyCount = count;
    updateToolbarState();
  }
}

function beginClassificationActivity(itemCount = 1) {
  activeClassifyCount += Math.max(1, Number(itemCount) || 1);
  updateToolbarState();
}

function endClassificationActivity(itemCount = 1) {
  activeClassifyCount = Math.max(0, activeClassifyCount - Math.max(1, Number(itemCount) || 1));
  updateToolbarState();
}

function setIconLoading(isLoading) {
  const filename = isLoading ? 'icon-loading.svg' : 'icon.svg';
  const fullPath = ext.runtime.getURL('icons/' + filename);

  try {
    if (ext.browserAction && typeof ext.browserAction.setIcon === 'function') {
      ext.browserAction.setIcon({ path: fullPath }, () => {
        if (ext.runtime.lastError) {
          console.error('[Icon] setIcon error:', ext.runtime.lastError);
        }
      });
    } else if (ext.action && typeof ext.action.setIcon === 'function') {
      ext.action.setIcon({ path: fullPath }, () => {
        if (ext.runtime.lastError) {
          console.error('[Icon] setIcon error:', ext.runtime.lastError);
        }
      });
    }
  } catch (e) {
    console.error('[Icon] Exception:', e.message, e.stack);
  }
}

const ENABLE_FAKE_NATIVE = false;

function makeRequestId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'req-' + Math.random().toString(16).slice(2) + Date.now();
}

updateToolbarState();

let nativePort = null;
const pendingNativeRequests = new Map();
const pendingJobs = new Map();
const JOB_STALE_MS = 10 * 60 * 1000;

// Track current URL per tabId so we can detect top-level page navigations and drop stale queued items
const tabLastUrl = new Map();

function expireStaleJobs() {
  const now = Date.now();
  for (const [jobId, info] of pendingJobs) {
    if (now - (info.createdAt || 0) > JOB_STALE_MS) {
      pendingJobs.delete(jobId);
      console.warn(`[Background] Expired stale job ${jobId}`);
    }
  }
}

setInterval(expireStaleJobs, 60000);

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

    // Real-time queue status updates (exact item count remaining)
    if (resType === 'queueStatus') {
      if (typeof response.remainingCount === 'number') {
        setExactRemainingItemCount(response.remainingCount);
      }
      return;
    }

    // Real-time single-item classification result
    if (resType === 'classifyItemResult') {
      if (typeof response.remainingCount === 'number') {
        setExactRemainingItemCount(response.remainingCount);
      } else {
        endClassificationActivity(1);
      }

      const targetTabId = response.tabId;
      if (targetTabId != null && ext.tabs && ext.tabs.sendMessage) {
        ext.tabs.sendMessage(targetTabId, {
          type: 'CLASSIFY_ITEM_RESULT',
          payload: response
        }).catch(() => {});
      }
      return;
    }

    if (resType === 'jobCancelled') {
      const jobId = response.jobId;
      console.log(`[Native] Job ${jobId} cancellation confirmed`);
      if (jobId === 'ALL') {
        pendingJobs.clear();
        setExactRemainingItemCount(0);
      } else {
        pendingJobs.delete(jobId);
      }
      return;
    }

    if (resType === 'downloadProgress') {
      const reqEntry = reqId ? pendingNativeRequests.get(reqId) : null;
      if (reqEntry && typeof reqEntry.onProgress === 'function') {
        reqEntry.onProgress(response);
      }
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
        endClassificationActivity(1);
      }
      return;
    }

    let matchedReqId = reqId;
    if (!matchedReqId || !pendingNativeRequests.has(matchedReqId)) {
      if (response && (response.type === 'pong' || response.ok)) {
        for (const [id] of pendingNativeRequests.entries()) {
          if (id.startsWith('req-ping-') || pendingNativeRequests.size === 1) {
            matchedReqId = id;
            break;
          }
        }
      }
    }
    if (!matchedReqId || !pendingNativeRequests.has(matchedReqId)) {
      console.warn('[Native] Unmatched response:', response);
      return;
    }
    const { resolve, timeoutId } = pendingNativeRequests.get(matchedReqId);
    if (timeoutId) clearTimeout(timeoutId);
    pendingNativeRequests.delete(matchedReqId);
    resolve(response);
  });

  nativePort.onDisconnect.addListener(() => {
    const err = ext.runtime.lastError;
    console.warn('[Native] Port disconnected', err);
    nativePort = null;

    for (const [reqId, { reject, timeoutId }] of pendingNativeRequests) {
      if (timeoutId) clearTimeout(timeoutId);
      reject(new Error(err?.message || 'Native host disconnected'));
      pendingNativeRequests.delete(reqId);
    }

    pendingJobs.clear();
    setExactRemainingItemCount(0);
  });

  // Sync current active tab context as soon as port connects
  syncActiveTabContext();

  return nativePort;
}

function sendToNative(message, options = {}) {
  if (ENABLE_FAKE_NATIVE) {
    return Promise.resolve({
      version: 2,
      type: 'listModels',
      classifiers: []
    });
  }

  const timeoutMs = options.timeoutMs || 60000;
  const onProgress = options.onProgress || null;

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
      }, timeoutMs);

      pendingNativeRequests.set(requestId, { resolve, reject, timeoutId, onProgress });
      port.postMessage(message);
    });
  }

  const canSend = ext && ext.runtime && typeof ext.runtime.sendNativeMessage === 'function';
  if (!canSend) {
    return Promise.reject(new Error('Native messaging API unavailable: ext.runtime.connectNative/sendNativeMessage not available.'));
  }

  return new Promise((resolve, reject) => {
    ext.runtime.sendNativeMessage(HOST_NAME, message, (response) => {
      const err = ext.runtime.lastError;
      if (err) {
        reject(new Error(`Native host error: ${err.message || JSON.stringify(err)}`));
        return;
      }
      resolve(response);
    });
  });
}

function sendToNativeFireAndForget(message) {
  const port = ensureNativePort();
  if (port) {
    try {
      port.postMessage(message);
      return true;
    } catch (err) {
      console.warn('[Native] port.postMessage failed; resetting port:', err);
      nativePort = null;
    }
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

// ============================================================================
// CORS-Safe Background Image Capture (Zero Native Host Internet Fetching)
// ============================================================================

async function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(typeof reader.result === 'string' ? reader.result : null);
    reader.onerror = () => reject(reader.error || new Error('FileReader failed'));
    reader.readAsDataURL(blob);
  });
}

/**
 * Fetches an image URL inside the privileged background context (where `<all_urls>`
 * host permissions bypass page CORS restrictions) and downscales/compresses it to
 * a compact JPEG data URL so the Native Host never needs internet access.
 */
async function fetchCrossOriginImageAsBase64(imageUrl, maxDim = 768, quality = 0.88) {
  if (!imageUrl || typeof imageUrl !== 'string') return null;
  if (imageUrl.startsWith('data:')) return imageUrl;
  if (!imageUrl.startsWith('http://') && !imageUrl.startsWith('https://')) return null;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);
    const response = await fetch(imageUrl, {
      signal: controller.signal,
      credentials: 'omit',
      headers: { Accept: 'image/*,*/*;q=0.8' }
    });
    clearTimeout(timeoutId);

    if (!response || !response.ok) return null;
    const blob = await response.blob();
    if (!blob || blob.size === 0) return null;

    // Downscale using createImageBitmap + OffscreenCanvas / HTMLCanvasElement if available
    if (typeof createImageBitmap === 'function') {
      try {
        const bitmap = await createImageBitmap(blob);
        let w = bitmap.width || 0;
        let h = bitmap.height || 0;
        if (w >= 32 && h >= 32) {
          if (w > maxDim || h > maxDim) {
            if (w >= h) {
              h = Math.round((h * maxDim) / w);
              w = maxDim;
            } else {
              w = Math.round((w * maxDim) / h);
              h = maxDim;
            }
          }
          if (typeof OffscreenCanvas !== 'undefined') {
            const offCanvas = new OffscreenCanvas(Math.max(1, w), Math.max(1, h));
            const ctx = offCanvas.getContext('2d');
            ctx.drawImage(bitmap, 0, 0, w, h);
            bitmap.close?.();
            const outBlob = await offCanvas.convertToBlob({
              type: 'image/jpeg',
              quality: Math.max(0.5, Math.min(0.95, quality))
            });
            return await blobToDataUrl(outBlob);
          } else if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, w);
            canvas.height = Math.max(1, h);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(bitmap, 0, 0, w, h);
            bitmap.close?.();
            return canvas.toDataURL('image/jpeg', Math.max(0.5, Math.min(0.95, quality)));
          }
        }
        bitmap.close?.();
      } catch (_) {
        // Fallback to raw blob data URL below
      }
    }

    return await blobToDataUrl(blob);
  } catch (err) {
    console.warn('[Background] CORS image fetch fallback failed for:', imageUrl.substring(0, 80), err?.message || err);
    return null;
  }
}

// ============================================================================
// Single-Item Packet Streaming & Browser Context Priority Sync
// ============================================================================

async function handleClassifySingleItem(message, sender) {
  const tabId = sender && sender.tab ? sender.tab.id : (message.tabId ?? null);
  const windowId = sender && sender.tab ? sender.tab.windowId : (message.windowId ?? null);
  const pageId = message.pageId || null;
  const itemId = message.itemId || message.id;
  const modality = message.modality || 'image';
  const inViewport = Boolean(message.inViewport);
  const modelConfig = message.model || {};

  if (!itemId) {
    return { ok: false, error: 'Missing itemId' };
  }

  if (await isPowerPolicyPaused()) {
    return { ok: false, error: 'paused_by_power_policy' };
  }

  let dataPayload = message.data || message.text || '';

  // If content script could not extract base64 due to CORS canvas taint, capture it here in background.js
  if (modality === 'image' && (!dataPayload || !dataPayload.startsWith('data:'))) {
    const candidateUrl = message.src || message.url || itemId;
    const quality = typeof message.captureQuality === 'number' ? message.captureQuality : 0.88;
    const fetchedDataUrl = await fetchCrossOriginImageAsBase64(candidateUrl, 768, quality);
    if (!fetchedDataUrl) {
      return { ok: false, error: 'cors_image_capture_failed' };
    }
    dataPayload = fetchedDataUrl;
  }

  if (!dataPayload) {
    return { ok: false, error: 'Empty item payload' };
  }

  // Split large single items into safe packet chunks under the 1 MB Native Messaging limit
  const totalLength = dataPayload.length;
  const totalPackets = Math.max(1, Math.ceil(totalLength / MAX_PACKET_CHUNK_CHARS));
  const packetGroupId = `${SOURCE_ID}:${pageId || tabId || '0'}:${itemId}:${Date.now()}`;

  // Optimistically increment badge counter until Native Host replies with authoritative `queueStatus`
  beginClassificationActivity(1);

  for (let pktIdx = 0; pktIdx < totalPackets; pktIdx++) {
    const start = pktIdx * MAX_PACKET_CHUNK_CHARS;
    const chunk = dataPayload.slice(start, start + MAX_PACKET_CHUNK_CHARS);
    const packetMsg = {
      version: 2,
      type: 'submitItemPacket',
      packetGroupId,
      packetIndex: pktIdx,
      totalPackets,
      sourceId: SOURCE_ID,
      windowId,
      tabId,
      pageId,
      itemId,
      modality,
      inViewport,
      dataChunk: chunk,
      timestamp: Date.now()
    };
    // Attach full model/ensemble configuration on the first packet
    if (pktIdx === 0) {
      packetMsg.model = modelConfig;
    }
    const sent = sendToNativeFireAndForget(packetMsg);
    if (!sent) {
      endClassificationActivity(1);
      return { ok: false, error: 'Native port unavailable' };
    }
  }

  return { ok: true, itemId, totalPackets };
}

function handleClassifyJobChunk(message, sender) {
  const jobId = message.jobId || makeRequestId();
  const tabId = sender && sender.tab ? sender.tab.id : null;
  if (tabId != null) {
    if (!pendingJobs.has(jobId)) {
      pendingJobs.set(jobId, { tabId, createdAt: Date.now() });
      beginClassificationActivity(1);
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
    tabId,
    chunkIndex: message.chunkIndex || 0,
    totalChunks: message.totalChunks || 1,
    timestamp: Date.now(),
    payload
  });

  if (!ok && pendingJobs.has(jobId)) {
    pendingJobs.delete(jobId);
    endClassificationActivity(1);
  }

  return { ok, jobId };
}

async function syncActiveTabContext() {
  if (!ext.tabs || !ext.tabs.query) return;
  try {
    const activeTabs = await ext.tabs.query({ active: true, currentWindow: true });
    if (activeTabs && activeTabs.length > 0 && activeTabs[0].id != null) {
      sendToNativeFireAndForget({
        version: 2,
        type: 'updateTabContext',
        sourceId: SOURCE_ID,
        activeTabId: activeTabs[0].id,
        windowId: activeTabs[0].windowId,
        timestamp: Date.now()
      });
    }
  } catch (_) {}
}

// Listen for tab switching, window focus changes, tab closures, and top-level URL navigations
if (ext.tabs) {
  if (ext.tabs.onActivated) {
    ext.tabs.onActivated.addListener((activeInfo) => {
      if (!activeInfo || activeInfo.tabId == null) return;
      sendToNativeFireAndForget({
        version: 2,
        type: 'updateTabContext',
        sourceId: SOURCE_ID,
        activeTabId: activeInfo.tabId,
        windowId: activeInfo.windowId,
        timestamp: Date.now()
      });
    });
  }

  if (ext.tabs.onRemoved) {
    ext.tabs.onRemoved.addListener((tabId) => {
      tabLastUrl.delete(tabId);
      sendToNativeFireAndForget({
        version: 2,
        type: 'dropContext',
        sourceId: SOURCE_ID,
        tabId,
        timestamp: Date.now()
      });
    });
  }

  if (ext.tabs.onUpdated) {
    ext.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
      if (changeInfo.status === 'loading' && changeInfo.url) {
        const prevUrl = tabLastUrl.get(tabId);
        // Strip hash to only purge on real page navigations, not in-page anchor jumps
        const cleanPrev = prevUrl ? prevUrl.split('#')[0] : null;
        const cleanNew = changeInfo.url.split('#')[0];
        tabLastUrl.set(tabId, changeInfo.url);
        if (cleanPrev && cleanPrev !== cleanNew) {
          sendToNativeFireAndForget({
            version: 2,
            type: 'dropContext',
            sourceId: SOURCE_ID,
            tabId,
            timestamp: Date.now()
          });
        }
      } else if (tab && tab.url && !tabLastUrl.has(tabId)) {
        tabLastUrl.set(tabId, tab.url);
      }
    });
  }
}

if (ext.windows && ext.windows.onFocusChanged) {
  ext.windows.onFocusChanged.addListener((windowId) => {
    if (windowId === ext.windows.WINDOW_ID_NONE || windowId < 0) return;
    if (ext.tabs && ext.tabs.query) {
      ext.tabs.query({ active: true, windowId })
        .then((tabs) => {
          if (tabs && tabs.length > 0 && tabs[0].id != null) {
            sendToNativeFireAndForget({
              version: 2,
              type: 'updateTabContext',
              sourceId: SOURCE_ID,
              activeTabId: tabs[0].id,
              windowId,
              timestamp: Date.now()
            });
          }
        })
        .catch(() => {});
    }
  });
}

let powerSettings = {
  pauseOnBattery: false,
  pauseBelowThreshold: false,
  batteryThreshold: 20
};

ext.storage.local.get(['pauseOnBattery', 'pauseBelowThreshold', 'batteryThreshold']).then((res) => {
  if (res) {
    if (typeof res.pauseOnBattery === 'boolean') powerSettings.pauseOnBattery = res.pauseOnBattery;
    if (typeof res.pauseBelowThreshold === 'boolean') powerSettings.pauseBelowThreshold = res.pauseBelowThreshold;
    if (typeof res.batteryThreshold === 'number') powerSettings.batteryThreshold = res.batteryThreshold;
  }
}).catch(() => {});

ext.storage.onChanged.addListener((changes, area) => {
  if (area === 'local') {
    if (changes.pauseOnBattery) powerSettings.pauseOnBattery = Boolean(changes.pauseOnBattery.newValue);
    if (changes.pauseBelowThreshold) powerSettings.pauseBelowThreshold = Boolean(changes.pauseBelowThreshold.newValue);
    if (changes.batteryThreshold) powerSettings.batteryThreshold = Number(changes.batteryThreshold.newValue);
  }
});

let cachedPowerStatus = {
  ok: true,
  is_plugged_in: true,
  has_battery: false,
  battery_percent: 100,
  charging: false,
  timestamp: 0
};

async function getSystemPowerStatus() {
  const now = Date.now();
  if (now - cachedPowerStatus.timestamp < 3000) {
    return cachedPowerStatus;
  }

  // 1. Browser API check if available
  if (typeof navigator !== 'undefined' && typeof navigator.getBattery === 'function') {
    try {
      const b = await navigator.getBattery();
      cachedPowerStatus = {
        ok: true,
        is_plugged_in: Boolean(b.charging),
        has_battery: true,
        battery_percent: Math.round(b.level * 100),
        charging: Boolean(b.charging),
        source: 'browser_api',
        timestamp: now
      };
      return cachedPowerStatus;
    } catch (_) {}
  }

  // 2. Fallback to Native Host (Windows / Linux power detection)
  try {
    const res = await sendToNative({ type: 'getPowerStatus', requestId: makeRequestId() }, { timeoutMs: 3000 });
    if (res && res.ok) {
      cachedPowerStatus = {
        ok: true,
        is_plugged_in: res.is_plugged_in !== false,
        has_battery: Boolean(res.has_battery),
        battery_percent: typeof res.battery_percent === 'number' ? res.battery_percent : 100,
        charging: Boolean(res.charging),
        source: res.source || 'native_host',
        timestamp: now
      };
      return cachedPowerStatus;
    }
  } catch (_) {}

  return cachedPowerStatus;
}

async function isPowerPolicyPaused() {
  if (!powerSettings.pauseOnBattery && !powerSettings.pauseBelowThreshold) {
    return false;
  }
  const power = await getSystemPowerStatus();
  if (powerSettings.pauseOnBattery && !power.is_plugged_in) {
    return true;
  }
  if (powerSettings.pauseBelowThreshold && typeof power.battery_percent === 'number' && power.battery_percent <= powerSettings.batteryThreshold) {
    return true;
  }
  return false;
}

async function openOnboardingExperience(targetScreen = 1) {
  const localOnboardingUrl = ext.runtime.getURL(`docs/index.html?screen=${targetScreen}`);
  if (ext.tabs && ext.tabs.create) {
    ext.tabs.create({ url: localOnboardingUrl });
  }
}

async function fetchFallbackCatalog() {
  const urls = [
    'https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/FVF-clean/native/official_models.json',
    'https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/main/native/official_models.json'
  ];
  for (const url of urls) {
    try {
      const resp = await fetch(`${url}?_t=${Date.now()}`, { cache: 'no-store' });
      if (resp && resp.ok) {
        return await resp.json();
      }
    } catch (_) {}
  }

  try {
    const localUrl = ext.runtime.getURL('docs/official_models.json');
    const resp = await fetch(`${localUrl}?_t=${Date.now()}`);
    if (resp && resp.ok) {
      return await resp.json();
    }
  } catch (_) {}

  return { catalog_version: 2, repository: 'LuckyCowMoo/Fire-vs-Fire', models: [] };
}

async function buildAndSaveAutoCategories(downloadedModelIds = [], customThreshold = 0.75) {
  const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);

  // Rule 1: ONLY if there are no categories at all!
  const existingData = await new Promise((resolve) => {
    if (storageArea && storageArea.get) {
      storageArea.get(['ensembleConfigsV2'], (res) => resolve(res || {}));
    } else {
      resolve({});
    }
  });

  const existingCategories = existingData && existingData.ensembleConfigsV2;
  if (Array.isArray(existingCategories) && existingCategories.length > 0) {
    console.log('[AutoSetupCategories] Categories already configured (' + existingCategories.length + '). Skipping auto-setup.');
    return existingCategories;
  }

  // Fetch official catalog to inspect models and modalities
  let catalog = null;
  try {
    const catResp = await sendToNative({ type: 'getOfficialCatalog', version: 2, force: true }, { timeoutMs: 5000 });
    if (catResp && catResp.catalog && Array.isArray(catResp.catalog.models)) {
      catalog = catResp.catalog;
    }
  } catch (_) {}

  if (!catalog) {
    catalog = await fetchFallbackCatalog();
  }
  const models = Array.isArray(catalog.models) ? catalog.models : [];

  // Determine which models are ACTUALLY installed / downloaded
  const installedSet = new Set(Array.isArray(downloadedModelIds) ? downloadedModelIds : []);
  try {
    const catResp = await sendToNative({ type: 'getOfficialCatalog', version: 2, force: true }, { timeoutMs: 5000 });
    if (catResp && Array.isArray(catResp.installedModels)) {
      catResp.installedModels.forEach((id) => installedSet.add(id));
    }
  } catch (_) {}

  // If no models are actually installed, create NO categories
  if (installedSet.size === 0) {
    console.log('[AutoSetupCategories] No models installed on disk. Zero categories created.');
    return [];
  }

  // Base threshold is 75% (0.75) or higher
  const threshold = typeof customThreshold === 'number'
    ? Math.max(0.05, Math.min(0.99, customThreshold))
    : 0.75;

  const modalities = ['image', 'text', 'audio', 'video'];
  const newEnsembles = [];

  // Freshness comparator to select the NEWEST FvF model for a modality
  const compareModelFreshness = (a, b) => {
    const vA = String(a.version || '0').replace(/[^0-9.]/g, '').split('.').map(Number);
    const vB = String(b.version || '0').replace(/[^0-9.]/g, '').split('.').map(Number);
    for (let i = 0; i < Math.max(vA.length, vB.length); i++) {
      const numA = vA[i] || 0;
      const numB = vB[i] || 0;
      if (numA !== numB) return numB - numA;
    }
    const matchA = (a.id || '').match(/\d+/);
    const matchB = (b.id || '').match(/\d+/);
    const numA = matchA ? parseInt(matchA[0], 10) : 0;
    const numB = matchB ? parseInt(matchB[0], 10) : 0;
    return numB - numA;
  };

  for (const mod of modalities) {
    // Find all INSTALLED FvF models for this modality
    const installedFvFModels = models.filter((m) => {
      return (m.modality || 'image') === mod &&
             installedSet.has(m.id) &&
             m.is_ai_detector !== false;
    });

    // ONLY if a model of that modality is ACTUALLY downloaded!
    if (installedFvFModels.length === 0) {
      continue;
    }

    // Sort to pick the NEWEST FvF model
    installedFvFModels.sort(compareModelFreshness);
    const newestModel = installedFvFModels[0];

    const borderColor = newestModel.default_color || (mod === 'image' ? '#FF1744' : '#FF6218');
    const ensemble = {
      id: `auto_${mod}_${newestModel.id}`,
      name: newestModel.name || newestModel.category_name || `AI ${mod}`,
      modality: mod,
      enabled: true,
      threshold, // 75%
      classifiers: [newestModel.id],
      weights: [1],
      styles: mod === 'text'
        ? {
            text: {
              applyBlur: true,
              blurMode: 'hover',
              blurAmount: 2,
              applyStrikethrough: true,
              strikethroughMode: 'hover',
              strikethroughColor: borderColor,
              applyUnderline: false,
              underlineMode: 'off',
              underlineColor: borderColor,
              applyHighlight: false,
              highlightMode: 'off',
              highlightColor: '#fff3a1'
            }
          }
        : {
            image: {
              applyBlur: true,
              blurAmount: 3,
              blurMode: 'hover',
              applyBorder: false,
              borderMultiplier: 1,
              borderColor,
              borderMode: 'off',
              applyBadge: true,
              badgeMode: 'always'
            }
          }
    };
    newEnsembles.push(ensemble);
  }

  // Also include custom object detectors (like ResNet 50 Spider) IF downloaded
  const installedNonAiModels = models.filter((m) => {
    return installedSet.has(m.id) && m.is_ai_detector === false;
  });
  for (const m of installedNonAiModels) {
    const borderColor = m.default_color || '#FFB020';
    newEnsembles.push({
      id: `auto_image_${m.id}`,
      name: m.category_name || m.name || 'Object Detector',
      modality: 'image',
      enabled: true,
      threshold, // 75%
      classifiers: [m.id],
      weights: [1],
      styles: {
        image: {
          applyBlur: true,
          blurAmount: 3,
          blurMode: 'hover',
          applyBorder: false,
          borderMultiplier: 1,
          borderColor,
          borderMode: 'off',
          applyBadge: true,
          badgeMode: 'always'
        }
      }
    });
  }

  if (newEnsembles.length === 0) {
    return [];
  }

  const storagePayload = {
    enabled: true,
    imageAiThreshold: threshold,
    textAiThreshold: threshold,
    ensembleConfigsV2: newEnsembles,
    onboardingCompleted: true,
    onboardingCompletedAt: Date.now()
  };

  if (storageArea && storageArea.set) {
    await storageArea.set(storagePayload);
  }
  if (ext.storage && ext.storage.local && ext.storage.local !== storageArea) {
    try {
      await ext.storage.local.set(storagePayload);
    } catch (_) {}
  }

  if (ext.tabs && ext.tabs.query) {
    try {
      const tabs = await ext.tabs.query({});
      tabs.forEach((tab) => {
        if (tab && tab.id != null) {
          ext.tabs.sendMessage(tab.id, { type: 'RECLASSIFY_ALL' }).catch(() => {});
        }
      });
    } catch (_) {}
  }

  return newEnsembles;
}

ext.runtime.onInstalled.addListener((details) => {
  console.log('Extension installed:', details && details.reason);
  updateToolbarState();
  if (ext.contextMenus && ext.contextMenus.create) {
    try {
      ext.contextMenus.create({
        id: 'toggle-blur',
        title: 'Toggle AI blur',
        contexts: ['image', 'page']
      });
    } catch (e) {
      console.warn('Context menu create failed:', e);
    }
  }

  if (!details || details.reason === 'install') {
    ext.storage.local.get(['altTextOnly']).then((existing) => {
      if (typeof existing.altTextOnly !== 'boolean') {
        ext.storage.local.set({ altTextOnly: true }).catch(() => {});
      }
    }).catch(() => {});
    openOnboardingExperience();
  }
});

if (ext.contextMenus && ext.contextMenus.onClicked) {
  ext.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'toggle-blur' && tab && tab.id != null) {
      const message = { type: 'TOGGLE_BLUR' };

      if (info.srcUrl) {
        message.srcUrl = info.srcUrl;
      } else if (info.selectionText) {
        message.selectionText = info.selectionText;
      }

      ext.tabs.sendMessage(tab.id, message)
        .then((response) => {
          if (response && ext.contextMenus && ext.contextMenus.update) {
            let title;
            if (response.hasAiTags) {
              const label = response.label || 'tag';
              title = `Remove ${label} tag`;
            } else {
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

ext.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === 'PING') {
    sendResponse({ ok: true, time: Date.now(), extensionVersion: ext.runtime.getManifest?.()?.version || '0.1.0' });
    return false;
  }

  if (message && message.type === 'OPEN_ONBOARDING') {
    openOnboardingExperience();
    sendResponse({ ok: true });
    return false;
  }

  if (message && message.type === 'OPEN_OPTIONS_FULLPAGE') {
    const optionsUrl = ext.runtime.getURL('options/options.html');
    if (ext.tabs && ext.tabs.create) {
      ext.tabs.create({ url: optionsUrl });
    }
    sendResponse({ ok: true });
    return false;
  }

  if (message && message.type === 'OPEN_MODELS_HUB') {
    openOnboardingExperience(2);
    sendResponse({ ok: true });
    return false;
  }

  if (message && message.type === 'OPEN_GUIDE_PAGE') {
    openOnboardingExperience(3);
    sendResponse({ ok: true });
    return false;
  }

  // Real-time single-item classification dispatch
  if (message && message.type === 'CLASSIFY_SINGLE_ITEM') {
    handleClassifySingleItem(message, sender)
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }

  // Boost priority of items that scrolled into the user's viewport
  if (message && message.type === 'BOOST_ITEMS') {
    const tabId = sender && sender.tab ? sender.tab.id : (message.tabId ?? null);
    const windowId = sender && sender.tab ? sender.tab.windowId : (message.windowId ?? null);
    sendToNativeFireAndForget({
      version: 2,
      type: 'boostItems',
      sourceId: SOURCE_ID,
      windowId,
      tabId,
      pageId: message.pageId || null,
      itemIds: Array.isArray(message.itemIds) ? message.itemIds : [],
      timestamp: Date.now()
    });
    sendResponse({ ok: true });
    return false;
  }

  // Drop pending items from a page that unloaded or navigated away
  if (message && message.type === 'DROP_PAGE_CONTEXT') {
    const tabId = sender && sender.tab ? sender.tab.id : (message.tabId ?? null);
    sendToNativeFireAndForget({
      version: 2,
      type: 'dropContext',
      sourceId: SOURCE_ID,
      tabId,
      pageId: message.pageId || null,
      timestamp: Date.now()
    });
    sendResponse({ ok: true });
    return false;
  }

  if (message && message.type === 'CHECK_NATIVE_UPDATE') {
    sendToNative({ type: 'checkUpdate', version: 2 }, { timeoutMs: 5000 })
      .then((res) => {
        sendResponse({ ok: true, connected: true, updateInfo: res });
      })
      .catch(async (err) => {
        try {
          const resp = await fetch('https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/FVF-clean/native/official_models.json', { cache: 'no-store' });
          if (resp && resp.ok) {
            const cat = await resp.json();
            sendResponse({
              ok: true,
              connected: false,
              updateInfo: {
                currentVersion: 'Not connected',
                latestVersion: cat.host_version || '2.2.0',
                updateAvailable: true,
                releaseNotes: cat.release_notes || 'Updated Native Host installer',
                downloadUrl: 'https://github.com/LuckyCowMoo/Fire-vs-Fire/releases/download/v2.0/FireVsFire-NativeHost.zip'
              }
            });
            return;
          }
        } catch (_) {}
        sendResponse({ ok: false, connected: false, error: err.message || String(err) });
      });
    return true;
  }

  if (message && message.type === 'APPLY_NATIVE_UPDATE') {
    sendToNative({ type: 'applyUpdate', version: 2 }, { timeoutMs: 30000 })
      .then((res) => {
        sendResponse({ ok: true, result: res });
      })
      .catch((err) => {
        sendResponse({ ok: false, error: err.message || String(err) });
      });
    return true;
  }

  if (message && message.type === 'CHECK_NATIVE_HOST') {
    const pingReqId = 'req-ping-' + Date.now();
    sendToNative({ type: 'ping', requestId: pingReqId, version: 2 }, { timeoutMs: 4000 })
      .then((res) => {
        sendResponse({ ok: true, connected: Boolean(res && res.ok), hostResponse: res });
      })
      .catch((err) => {
        sendResponse({ ok: true, connected: false, error: err.message || String(err) });
      });
    return true;
  }

  if (message && message.type === 'GET_OFFICIAL_CATALOG') {
    sendToNative({ type: 'getOfficialCatalog', version: 2, force: Boolean(message.force) }, { timeoutMs: 8000 })
      .then((res) => {
        sendResponse({
          ok: true,
          nativeConnected: true,
          catalog: res.catalog,
          installedModels: res.installedModels || [],
          classifiers: res.classifiers || []
        });
      })
      .catch(async (err) => {
        const fallbackCatalog = await fetchFallbackCatalog();
        sendResponse({
          ok: true,
          nativeConnected: false,
          catalog: fallbackCatalog,
          installedModels: [],
          classifiers: [],
          nativeError: err.message || String(err)
        });
      });
    return true;
  }

  if (message && message.type === 'DOWNLOAD_OFFICIAL_MODEL') {
    const modelId = message.modelId;
    const forceDownload = Boolean(message.forceDownload);
    const tabId = sender && sender.tab ? sender.tab.id : null;

    sendToNative(
      {
        type: 'downloadOfficialModel',
        version: 2,
        modelId,
        forceDownload
      },
      {
        timeoutMs: 10 * 60 * 1000,
        onProgress: (progressMsg) => {
          if (tabId != null && ext.tabs && ext.tabs.sendMessage) {
            ext.tabs.sendMessage(tabId, {
              type: 'ONBOARDING_DOWNLOAD_PROGRESS',
              payload: progressMsg
            }).catch(() => {});
          }
          try {
            ext.runtime.sendMessage({
              type: 'ONBOARDING_DOWNLOAD_PROGRESS',
              payload: progressMsg
            }).catch(() => {});
          } catch (_) {}
        }
      }
    )
      .then((res) => {
        sendResponse(res);
      })
      .catch((err) => {
        sendResponse({ ok: false, modelId, error: err.message || String(err) });
      });
    return true;
  }

  if (message && message.type === 'AUTO_SETUP_CATEGORIES') {
    const downloadedModels = Array.isArray(message.downloadedModels) ? message.downloadedModels : [];
    const threshold = typeof message.threshold === 'number' ? message.threshold : 0.75;
    buildAndSaveAutoCategories(downloadedModels, threshold)
      .then((ensembles) => {
        sendResponse({ ok: true, ensembles });
      })
      .catch((err) => {
        sendResponse({ ok: false, error: err.message || String(err) });
      });
    return true;
  }

  if (message && message.type === 'SAVE_ONBOARDING_ENSEMBLES') {
    const ensembles = Array.isArray(message.ensembles) ? message.ensembles : [];
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    const payload = {
      enabled: true,
      ensembleConfigsV2: ensembles,
      onboardingCompleted: true,
      onboardingCompletedAt: Date.now()
    };
    Promise.resolve(storageArea && storageArea.set ? storageArea.set(payload) : null)
      .then(() => {
        sendResponse({ ok: true, ensembles });
      })
      .catch((err) => {
        sendResponse({ ok: false, error: err.message || String(err) });
      });
    return true;
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

  if (message && message.type === 'GET_POWER_STATUS') {
    getSystemPowerStatus()
      .then((status) => sendResponse(status))
      .catch((err) => sendResponse({ ok: false, error: err && err.message ? err.message : String(err) }));
    return true;
  }

  if (message && message.type === 'GET_ACTIVITY_STATUS') {
    getSystemPowerStatus().then((power) => {
      const isPaused = (powerSettings.pauseOnBattery && !power.is_plugged_in) ||
                       (powerSettings.pauseBelowThreshold && typeof power.battery_percent === 'number' && power.battery_percent <= powerSettings.batteryThreshold);
      sendResponse({
        ok: true,
        isLoading: activeClassifyCount > 0,
        activeCount: activeClassifyCount,
        isPaused,
        power
      });
    }).catch(() => {
      sendResponse({ ok: true, isLoading: activeClassifyCount > 0, activeCount: activeClassifyCount, isPaused: false });
    });
    return true;
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

  if (message && message.type === 'CANCEL_ALL') {
    sendToNativeFireAndForget({
      version: 2,
      type: 'cancelJob',
      jobId: 'ALL',
      timestamp: Date.now()
    });
    pendingJobs.clear();
    setExactRemainingItemCount(0);
    sendResponse({ ok: true });
    return false;
  }

  return false;
});
