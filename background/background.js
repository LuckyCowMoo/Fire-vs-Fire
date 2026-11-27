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
    if (resType === 'bellowsProgress') {
      try {
        ext.runtime.sendMessage({ type: 'BELLOWS_PROGRESS', payload: response }).catch(() => {});
        if (ext.tabs && ext.tabs.query) {
          ext.tabs.query({}, (tabs) => {
            (tabs || []).forEach((t) => {
              if (t && t.id != null) ext.tabs.sendMessage(t.id, { type: 'BELLOWS_PROGRESS', payload: response }).catch(() => {});
            });
          });
        }
      } catch (_) {}
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

async function restartNativeHost() {
  for (const [reqId, { reject, timeoutId }] of pendingNativeRequests) {
    if (timeoutId) clearTimeout(timeoutId);
    reject(new Error('Native host restarting'));
    pendingNativeRequests.delete(reqId);
  }
  pendingJobs.clear();
  setExactRemainingItemCount(0);

  if (nativePort) {
    try {
      nativePort.disconnect();
    } catch (_) {}
    nativePort = null;
  }

  await new Promise((resolve) => setTimeout(resolve, 250));
  return sendToNative({ type: 'ping', version: 2 }, { timeoutMs: 4000 });
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
async function fetchCrossOriginImageAsBase64(imageUrl, maxDim = 768, quality = 1.0, timeoutMs = 8000) {
  if (!imageUrl || typeof imageUrl !== 'string') return null;
  if (imageUrl.startsWith('data:')) return imageUrl;
  if (!imageUrl.startsWith('http://') && !imageUrl.startsWith('https://')) return null;

  const clampedQuality = Math.max(0.2, Math.min(1.0, typeof quality === 'number' ? quality : 1.0));

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch(imageUrl, {
      signal: controller.signal,
      credentials: 'omit',
      headers: { Accept: 'image/*,*/*;q=0.8' }
    });
    clearTimeout(timeoutId);

    if (!response || !response.ok) return null;
    const blob = await response.blob();
    if (!blob || blob.size === 0) return null;

    const contentType = (blob.type || response.headers?.get?.('content-type') || '').toLowerCase();
    if (
      contentType.startsWith('text/') ||
      contentType.includes('html') ||
      contentType.includes('json') ||
      contentType.includes('xml') ||
      contentType.includes('svg') ||
      (contentType && !contentType.startsWith('image/') && !contentType.startsWith('application/octet-stream'))
    ) {
      return null;
    }

    // Preserve untouched original bytes when within maxDim; otherwise downscale cleanly
    if (typeof createImageBitmap === 'function') {
      try {
        const bitmap = await createImageBitmap(blob);
        let w = bitmap.width || 0;
        let h = bitmap.height || 0;
        if (w >= 32 && h >= 32) {
          const mime = (blob.type || '').toLowerCase();
          const isDirectFormat = mime === 'image/jpeg' || mime === 'image/jpg' || mime === 'image/png' || mime === 'image/webp';
          if (w <= maxDim && h <= maxDim && clampedQuality >= 0.99 && isDirectFormat && blob.size <= 1572864) {
            bitmap.close?.();
            return await blobToDataUrl(blob);
          }
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
              quality: clampedQuality
            });
            offCanvas.width = 0;
            offCanvas.height = 0;
            return await blobToDataUrl(outBlob);
          } else if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, w);
            canvas.height = Math.max(1, h);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(bitmap, 0, 0, w, h);
            bitmap.close?.();
            const dataUrl = canvas.toDataURL('image/jpeg', clampedQuality);
            canvas.width = 0;
            canvas.height = 0;
            return dataUrl;
          }
        }
        bitmap.close?.();
      } catch (_) {
        // Non-decodable payload (e.g. corrupt/non-image response): allow caller fallback
        return null;
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

  // Capture image via background privileges (CORS bypass or Investigator high-res upgrade)
  if (modality === 'image' && (!dataPayload || !dataPayload.startsWith('data:'))) {
    const candidateUrl = message.src || message.url || itemId;
    const quality = typeof message.captureQuality === 'number' ? message.captureQuality : 1.0;
    const primaryTimeout = message.isInvestigatorUpgrade ? 4500 : 8000;
    let fetchedDataUrl = await fetchCrossOriginImageAsBase64(candidateUrl, 768, quality, primaryTimeout);

    // If an Investigator high-res URL 404s or times out, fall back to original thumbnail seamlessly
    if (!fetchedDataUrl && message.isInvestigatorUpgrade) {
      if (message.fallbackData && typeof message.fallbackData === 'string' && message.fallbackData.startsWith('data:')) {
        fetchedDataUrl = message.fallbackData;
      } else if (message.fallbackSrc && message.fallbackSrc !== candidateUrl) {
        fetchedDataUrl = await fetchCrossOriginImageAsBase64(message.fallbackSrc, 768, quality, 8000);
      }
    }

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
      bellowsCollectorTabs.delete(tabId);
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
    if (changes.communityReportingEnabled || changes.communityHfUsername) {
      syncCommunityReportContextMenu().catch(() => {});
    }
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

const bellowsCollectorTabs = new Map();
async function openBellowsWorkspace(opts = {}) {
  const qs = [];
  if (opts.modality) qs.push(`modality=${encodeURIComponent(opts.modality)}`);
  if (opts.resume) qs.push('resume=1');
  if (opts.jobId) qs.push(`jobId=${encodeURIComponent(opts.jobId)}`);
  const url = ext.runtime.getURL(`docs/bellows.html${qs.length ? '?' + qs.join('&') : ''}`);
  if (ext.tabs && ext.tabs.query && ext.tabs.update && ext.tabs.create) {
    const basePrefix = ext.runtime.getURL('docs/bellows.html');
    const existing = await ext.tabs.query({ url: `${basePrefix}*` }).catch(() => []);
    if (existing && existing.length > 0 && existing[0].id != null) {
      await ext.tabs.update(existing[0].id, { active: true, ...(opts.modality || opts.resume ? { url } : {}) });
      return;
    }
    await ext.tabs.create({ url });
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
  const resolveModelThreshold = (m) => {
    const raw = m && (typeof m.suggested_threshold === 'number' ? m.suggested_threshold : m.default_threshold);
    if (typeof raw === 'number' && Number.isFinite(raw) && raw >= 0 && raw <= 1) {
      return Math.round(raw * 100) / 100;
    }
    return threshold;
  };

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
      threshold: resolveModelThreshold(newestModel),
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
      threshold: resolveModelThreshold(m),
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
  syncCommunityReportContextMenu().catch(() => {});
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
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    if (storageArea && storageArea.get && storageArea.set) {
      storageArea.get(['altTextOnly', 'ignoreSmallImages', 'investigator', 'imageCaptureQuality', 'minImageDimension', 'minTextLength']).then((existing) => {
        const defaults = {};
        if (typeof existing?.altTextOnly !== 'boolean') defaults.altTextOnly = true;
        if (typeof existing?.ignoreSmallImages !== 'boolean') defaults.ignoreSmallImages = true;
        if (typeof existing?.investigator !== 'boolean') defaults.investigator = true;
        if (typeof existing?.imageCaptureQuality !== 'number') defaults.imageCaptureQuality = 1;
        if (typeof existing?.minImageDimension !== 'number') defaults.minImageDimension = 80;
        if (typeof existing?.minTextLength !== 'number') defaults.minTextLength = 250;
        if (Object.keys(defaults).length > 0) {
          storageArea.set(defaults).catch(() => {});
        }
      }).catch(() => {});
    }
    openOnboardingExperience();
  }
});

(async function ensureMaxCaptureQualityAndInvestigatorDefaults() {
  try {
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    if (!storageArea || !storageArea.get || !storageArea.set) return;
    const res = await storageArea.get(['imageCaptureQualityAutoMaxV1', 'investigator']);
    if (!res || !res.imageCaptureQualityAutoMaxV1) {
      const updates = {
        imageCaptureQuality: 1,
        imageCaptureQualityAutoMaxV1: true
      };
      if (typeof res?.investigator !== 'boolean') {
        updates.investigator = true;
      }
      await storageArea.set(updates);
    }
  } catch (_) {}
})();

if (ext.contextMenus && ext.contextMenus.onClicked) {
  ext.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'report-misclassification' && tab && tab.id != null) {
      ext.tabs.sendMessage(tab.id, {
        type: 'OPEN_COMMUNITY_REPORT_MODAL',
        srcUrl: info.srcUrl || null
      }).catch(() => {});
      return;
    }

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

  // Hover priority (P0): content script reports the pending item under the mouse
  if (message && message.type === 'HOVER_ITEM') {
    const tabId = sender && sender.tab ? sender.tab.id : (message.tabId ?? null);
    const windowId = sender && sender.tab ? sender.tab.windowId : (message.windowId ?? null);
    sendToNativeFireAndForget({
      version: 2,
      type: 'hoverItem',
      sourceId: SOURCE_ID,
      windowId,
      tabId,
      pageId: message.pageId || null,
      itemId: message.itemId || null,
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
    const force = message.force !== false;
    sendToNative({ type: 'checkUpdate', version: 2, force }, { timeoutMs: 6000 })
      .then((res) => {
        sendResponse({ ok: true, connected: true, updateInfo: res });
      })
      .catch(async (err) => {
        try {
          const relResp = await fetch('https://api.github.com/repos/LuckyCowMoo/Fire-vs-Fire/releases/latest', {
            cache: 'no-store',
            headers: { Accept: 'application/vnd.github+json' }
          });
          if (relResp && relResp.ok) {
            const rel = await relResp.json();
            const zipAsset = (rel.assets || []).find((a) => /nativehost.*\.zip$/i.test(a.name || '')) || (rel.assets || []).find((a) => a.name === 'FireVsFire-NativeHost.zip');
            sendResponse({
              ok: true,
              connected: false,
              updateInfo: {
                checked: true,
                currentVersion: 'Not connected',
                latestVersion: rel.tag_name || '',
                releaseTag: rel.tag_name || '',
                releaseName: rel.name || rel.tag_name || 'Latest Release',
                updateAvailable: true,
                releaseNotes: rel.body || 'Latest Native Host release on GitHub',
                downloadUrl: (zipAsset && zipAsset.browser_download_url) || rel.html_url || 'https://github.com/LuckyCowMoo/Fire-vs-Fire/releases/latest'
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
      .then(async (res) => {
        let restarted = false;
        if (res && res.ok) {
          try {
            await restartNativeHost();
            restarted = true;
          } catch (_) {}
        }
        sendResponse({ ok: true, result: res, restarted });
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

  if (message && message.type === 'UNINSTALL_OFFICIAL_MODEL') {
    const modelId = message.modelId;
    sendToNative(
      {
        type: 'uninstallOfficialModel',
        version: 2,
        modelId
      },
      { timeoutMs: 30000 }
    )
      .then((res) => {
        if (res && res.ok && modelId && modelId.startsWith('bellows_')) {
          const area = ext.storage && (ext.storage.sync || ext.storage.local);
          if (area && area.get && area.set) {
            area.get(['ensembleConfigsV2'], (st) => {
              const arr = Array.isArray(st?.ensembleConfigsV2) ? st.ensembleConfigsV2 : [];
              const filtered = arr.filter((e) => !(Array.isArray(e.classifiers) && e.classifiers.length === 1 && e.classifiers[0] === modelId));
              if (filtered.length !== arr.length) area.set({ ensembleConfigsV2: filtered });
            });
          }
        }
        if (res && res.ok && Array.isArray(res.installedModels)) {
          buildAndSaveAutoCategories(res.installedModels).catch(() => {});
        }
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
    sendToNative({
      type: 'listModels',
      version: 2,
      checkUpdates: Boolean(message.checkUpdates),
      forceUpdateCheck: Boolean(message.forceUpdateCheck)
    })
      .then((response) => {
        const rawClassifiers = Array.isArray(response.classifiers) ? response.classifiers : [];
        const models = rawClassifiers.length
          ? rawClassifiers.map((m) => (typeof m === 'string' ? m : m.id)).filter(Boolean)
          : (response.models || response.model_files || []);
        sendResponse({
          ok: true,
          models,
          raw: response,
          updates: response.updates || null
        });
      })
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }

  if (message && message.type === 'OPEN_BELLOWS') {
    openBellowsWorkspace(message).catch(() => {});
    sendResponse({ ok: true });
    return false;
  }
  if (message && message.type === 'BELLOWS_PREFLIGHT') {
    sendToNative({ type: 'bellowsPreflight', version: 2 }, { timeoutMs: 12000 })
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_CHECK_ETHICS') {
    sendToNative({ type: 'bellowsCheckEthics', version: 2, modality: message.modality, prompt: message.prompt }, { timeoutMs: 240000 })
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_PREPARE_SESSION') {
    sendToNative({ type: 'bellowsPrepareSession', version: 2, modality: message.modality, prompt: message.prompt, jobId: message.jobId }, { timeoutMs: 300000 })
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_ADD_SAMPLES') {
    sendToNative({ type: 'bellowsAddSamples', version: 2, jobId: message.jobId, modality: message.modality, kind: message.kind, samples: message.samples }, { timeoutMs: 30000 })
      .then((res) => {
        if (res && res.counts && sender?.tab?.id != null && bellowsCollectorTabs.has(sender.tab.id)) {
          const info = bellowsCollectorTabs.get(sender.tab.id);
          info.counts = res.counts;
          if (info.sourceTabId != null) ext.tabs.sendMessage(info.sourceTabId, { type: 'BELLOWS_COLLECTOR_COUNTS', payload: { jobId: message.jobId, counts: res.counts } }).catch(() => {});
        }
        sendResponse(res);
      })
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_REMOVE_SAMPLE') {
    sendToNative({ type: 'bellowsRemoveSample', version: 2, jobId: message.jobId, modality: message.modality, sampleId: message.sampleId }, { timeoutMs: 10000 })
      .then((res) => {
        if (res && res.counts && sender?.tab?.id != null && bellowsCollectorTabs.has(sender.tab.id)) {
          const info = bellowsCollectorTabs.get(sender.tab.id);
          info.counts = res.counts;
          if (info.sourceTabId != null) ext.tabs.sendMessage(info.sourceTabId, { type: 'BELLOWS_COLLECTOR_COUNTS', payload: { jobId: message.jobId, counts: res.counts } }).catch(() => {});
        }
        sendResponse(res);
      })
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_CAPTURE_IMAGE_URL') {
    (async () => {
      const resp = await fetch(message.url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      const bytes = new Uint8Array(buf);
      let bin = '';
      for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode(...bytes.subarray(i, i + 8192));
      const b64 = `data:image/jpeg;base64,${btoa(bin)}`;
      const nativeRes = await sendToNative({ type: 'bellowsAddSamples', version: 2, jobId: message.jobId, modality: 'image', kind: message.kind, samples: [b64] }, { timeoutMs: 20000 });
      if (nativeRes && nativeRes.counts && sender?.tab?.id != null && bellowsCollectorTabs.has(sender.tab.id)) {
        const info = bellowsCollectorTabs.get(sender.tab.id);
        info.counts = nativeRes.counts;
        if (info.sourceTabId != null) ext.tabs.sendMessage(info.sourceTabId, { type: 'BELLOWS_COLLECTOR_COUNTS', payload: { jobId: message.jobId, counts: nativeRes.counts } }).catch(() => {});
      }
      sendResponse(nativeRes);
    })().catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_START_COLLECTOR') {
    (async () => {
      const modality = message.modality === 'text' ? 'text' : 'image';
      const q = encodeURIComponent(String(message.prompt || '').trim());
      const sourceEngine = String(message.engine || (modality === 'image' ? 'google_images' : 'reddit'));
      let targetUrl = `https://www.google.com/search?tbm=isch&q=${q}`;
      if (modality === 'text') {
        if (sourceEngine === 'wikipedia') targetUrl = `https://en.wikipedia.org/w/index.php?search=${q}`;
        else if (sourceEngine === 'google') targetUrl = `https://www.google.com/search?q=${q}`;
        else targetUrl = `https://www.reddit.com/search/?q=${q}&type=comment`;
      }
      const tab = await ext.tabs.create({ url: targetUrl, active: true });
      const state = { jobId: message.jobId, modality, prompt: message.prompt || '', counts: message.counts || { positives: 0, hard_negatives: 0, base_negatives: 0 }, sourceTabId: sender?.tab?.id ?? null };
      bellowsCollectorTabs.set(tab.id, state);
      setTimeout(() => {
        ext.tabs.sendMessage(tab.id, { type: 'BELLOWS_ACTIVATE_COLLECTOR', ...state }).catch(() => {});
      }, 1500);
      sendResponse({ ok: true, tabId: tab.id });
    })().catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_QUERY_COLLECTOR_TAB') {
    const tid = sender?.tab?.id;
    const state = (tid != null && bellowsCollectorTabs.has(tid)) ? bellowsCollectorTabs.get(tid) : null;
    sendResponse(state ? { active: true, ...state } : { active: false });
    return false;
  }
  if (message && message.type === 'BELLOWS_STOP_COLLECTOR') {
    const tid = sender?.tab?.id;
    const state = (tid != null && bellowsCollectorTabs.has(tid)) ? bellowsCollectorTabs.get(tid) : null;
    if (tid != null) {
      bellowsCollectorTabs.delete(tid);
      ext.tabs.remove(tid).catch(() => {});
    }
    if (state && state.sourceTabId != null) {
      ext.tabs.update(state.sourceTabId, { active: true }).catch(() => {});
      ext.tabs.sendMessage(state.sourceTabId, { type: 'BELLOWS_COLLECTOR_COUNTS', payload: { jobId: message.jobId, counts: state.counts, closed: true } }).catch(() => {});
    }
    sendResponse({ ok: true });
    return false;
  }
  if (message && message.type === 'BELLOWS_START_TRAINING') {
    sendToNative({ ...message, type: 'bellowsStartTraining', version: 2 }, { timeoutMs: 20000 })
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_GET_STATUS') {
    sendToNative({ type: 'bellowsGetStatus', version: 2, jobId: message.jobId }, { timeoutMs: 8000 })
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_STOP_TRAINING') {
    sendToNative({ type: 'bellowsStopTraining', version: 2, jobId: message.jobId }, { timeoutMs: 8000 })
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_CLEANUP_STORAGE') {
    sendToNative({ type: 'bellowsCleanupStorage', version: 2, deleteCache: message.deleteCache !== false, deleteDatasets: message.deleteDatasets !== false }, { timeoutMs: 15000 })
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'BELLOWS_FINALIZE_CATEGORY') {
    (async () => {
      const area = ext.storage && (ext.storage.sync || ext.storage.local);
      const existing = area && area.get ? await new Promise((r) => area.get(['ensembleConfigsV2'], (d) => r(d || {}))) : {};
      const list = Array.isArray(existing.ensembleConfigsV2) ? existing.ensembleConfigsV2 : [];
      const cid = String(message.classifierId || '');
      if (!cid) throw new Error('Missing classifierId');
      const already = list.find((e) => Array.isArray(e.classifiers) && e.classifiers.length === 1 && e.classifiers[0] === cid);
      if (!already) {
        const mod = message.modality === 'text' ? 'text' : 'image';
        const thr = typeof message.threshold === 'number' ? Math.round(message.threshold * 100) / 100 : 0.75;
        const color = mod === 'text' ? '#10b981' : '#ff6218';
        list.push({
          id: `bellows_${cid}`,
          name: message.modelName || cid,
          modality: mod,
          enabled: true,
          threshold: thr,
          classifiers: [cid],
          weights: [1],
          styles: mod === 'text' ? {
            text: { applyBlur: true, blurMode: 'hover', blurAmount: 2, applyStrikethrough: true, strikethroughMode: 'hover', strikethroughColor: color, applyUnderline: false, underlineMode: 'off', underlineColor: color, applyHighlight: false, highlightMode: 'off', highlightColor: '#bbf7d0' }
          } : {
            image: { applyBlur: true, blurAmount: 3, blurMode: 'hover', applyBorder: false, borderMultiplier: 1, borderColor: color, borderMode: 'off', applyBadge: true, badgeMode: 'always' }
          }
        });
        if (area && area.set) await area.set({ enabled: true, ensembleConfigsV2: list });
      }
      sendResponse({ ok: true, ensembles: list });
    })().catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }
  if (message && message.type === 'VERIFY_COMMUNITY_HF_TOKEN') {
    verifyCommunityHfToken(message.token)
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }

  if (message && message.type === 'PREPARE_COMMUNITY_REPORT_IMAGE') {
    (async () => {
      const stored = await ext.storage.local.get(['communityReportingEnabled', 'communityHfToken', 'communityHfUsername']);
      const sanitized = await sanitizeImageForCommunityReport(message.srcUrl, 1024, 0.92);
      sendResponse({
        ok: true,
        ...sanitized,
        hasToken: Boolean(stored?.communityHfToken && stored?.communityHfUsername),
        username: stored?.communityHfUsername || ''
      });
    })().catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
    return true;
  }

  if (message && message.type === 'SUBMIT_COMMUNITY_REPORT') {
    submitCommunityReportToHf(message.report)
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err?.message || String(err) }));
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

// ============================================================================
// OPT-IN NON-ANONYMOUS COMMUNITY MISCLASSIFICATION REPORTING (HUGGING FACE PRs)
// ============================================================================

const COMMUNITY_REPORTS_HF_REPO = 'LuckyCow/fvf-community-reports';

async function syncCommunityReportContextMenu() {
  if (!ext.contextMenus || !ext.contextMenus.create) return;
  let enabled = false;
  try {
    const res = await ext.storage.local.get(['communityReportingEnabled']);
    enabled = Boolean(res?.communityReportingEnabled);
  } catch (_) {}

  try {
    await new Promise((resolve) => {
      try {
        const p = ext.contextMenus.remove('report-misclassification', () => {
          void ext.runtime.lastError;
          resolve();
        });
        if (p && typeof p.then === 'function') p.then(resolve).catch(resolve);
      } catch (_) {
        resolve();
      }
    });
  } catch (_) {}

  if (enabled) {
    try {
      ext.contextMenus.create({
        id: 'report-misclassification',
        title: 'Report incorrect FvF classification…',
        contexts: ['image']
      });
    } catch (e) {
      console.warn('[CommunityReport] Context menu create failed:', e);
    }
  }
}

syncCommunityReportContextMenu().catch(() => {});

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function dataUrlToUint8Array(dataUrl) {
  const commaIdx = dataUrl.indexOf(',');
  const b64 = commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : dataUrl;
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

async function sha256Hex(uint8Array) {
  const digest = await crypto.subtle.digest('SHA-256', uint8Array);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

async function sanitizeImageForCommunityReport(srcUrl, maxDim = 1024, quality = 0.92) {
  if (!srcUrl || typeof srcUrl !== 'string') {
    throw new Error('Missing image source URL');
  }
  let blob = null;
  if (srcUrl.startsWith('data:')) {
    const resp = await fetch(srcUrl);
    blob = await resp.blob();
  } else if (srcUrl.startsWith('http://') || srcUrl.startsWith('https://')) {
    const controller = new AbortController();
    const tid = setTimeout(() => controller.abort(), 10000);
    const resp = await fetch(srcUrl, {
      signal: controller.signal,
      credentials: 'omit',
      headers: { Accept: 'image/*,*/*;q=0.8' }
    });
    clearTimeout(tid);
    if (!resp || !resp.ok) {
      throw new Error(`Could not fetch image (HTTP ${resp ? resp.status : 'error'})`);
    }
    blob = await resp.blob();
  } else {
    throw new Error('Unsupported image URL scheme');
  }

  if (!blob || blob.size === 0) {
    throw new Error('Empty image payload');
  }

  let width = 0;
  let height = 0;
  let outBlob = null;

  if (typeof createImageBitmap === 'function') {
    const bitmap = await createImageBitmap(blob);
    width = bitmap.width || 0;
    height = bitmap.height || 0;
    if (width < 32 || height < 32) {
      bitmap.close?.();
      throw new Error('Image is too small (< 32px) to report');
    }
    if (width > maxDim || height > maxDim) {
      if (width >= height) {
        height = Math.max(1, Math.round((height * maxDim) / width));
        width = maxDim;
      } else {
        width = Math.max(1, Math.round((width * maxDim) / height));
        height = maxDim;
      }
    }
    if (typeof OffscreenCanvas !== 'undefined') {
      const off = new OffscreenCanvas(width, height);
      const ctx = off.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(bitmap, 0, 0, width, height);
      bitmap.close?.();
      outBlob = await off.convertToBlob({ type: 'image/jpeg', quality });
      off.width = 0;
      off.height = 0;
    } else if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(bitmap, 0, 0, width, height);
      bitmap.close?.();
      const dataUrl = canvas.toDataURL('image/jpeg', quality);
      canvas.width = 0;
      canvas.height = 0;
      const bytes = dataUrlToUint8Array(dataUrl);
      const hash = await sha256Hex(bytes);
      return {
        sanitizedDataUrl: dataUrl,
        sha256: hash,
        width,
        height,
        byteLength: bytes.byteLength
      };
    } else {
      bitmap.close?.();
      throw new Error('Canvas re-encoding unavailable in this browser context');
    }
  } else {
    throw new Error('createImageBitmap unavailable for EXIF stripping');
  }

  const ab = await outBlob.arrayBuffer();
  const bytes = new Uint8Array(ab);
  const hash = await sha256Hex(bytes);
  const sanitizedDataUrl = `data:image/jpeg;base64,${arrayBufferToBase64(ab)}`;
  return {
    sanitizedDataUrl,
    sha256: hash,
    width,
    height,
    byteLength: bytes.byteLength
  };
}

async function verifyCommunityHfToken(token) {
  const clean = String(token || '').trim();
  if (!clean || !clean.startsWith('hf_')) {
    return { ok: false, error: 'Token must start with hf_' };
  }
  const resp = await fetch('https://huggingface.co/api/whoami-v2', {
    method: 'GET',
    headers: { Authorization: `Bearer ${clean}` }
  });
  if (!resp.ok) {
    return { ok: false, error: `Hugging Face rejected token (HTTP ${resp.status})` };
  }
  const data = await resp.json();
  const username = data && typeof data.name === 'string' ? data.name.trim() : '';
  if (!username) {
    return { ok: false, error: 'Could not read Hugging Face username from token' };
  }
  const accessToken = data?.auth?.accessToken;
  const role = String(accessToken?.role || '').toLowerCase();
  if (role === 'read') {
    return {
      ok: false,
      error: 'Read-only tokens cannot submit community PRs. Use a Write token or enable Discussions/PRs write access.'
    };
  }
  if (role === 'finegrained' && accessToken?.fineGrained) {
    const fg = accessToken.fineGrained;
    const globalPerms = Array.isArray(fg.global) ? fg.global : [];
    const hasGlobalWrite = globalPerms.includes('discussion.write') || globalPerms.includes('repo.write');
    const scopedList = Array.isArray(fg.scoped) ? fg.scoped : [];
    const hasScopedTarget = scopedList.some((entry) => {
      const entityName = String(entry?.entity?.name || '');
      const perms = Array.isArray(entry?.permissions) ? entry.permissions : [];
      const canWrite = perms.includes('discussion.write') || perms.includes('repo.write');
      return canWrite && (entityName === COMMUNITY_REPORTS_HF_REPO || entityName === 'LuckyCow');
    });
    if (!hasGlobalWrite && !hasScopedTarget) {
      return {
        ok: false,
        error: 'Fine-grained token lacks "Write access to discussions and pull requests" permission.'
      };
    }
  }
  await ext.storage.local.set({
    communityHfToken: clean,
    communityHfUsername: username
  });
  return { ok: true, username };
}

async function submitCommunityReportToHf(reportInput) {
  const stored = await ext.storage.local.get(['communityReportingEnabled', 'communityHfToken', 'communityHfUsername']);
  if (!stored?.communityReportingEnabled) {
    return { ok: false, error: 'Opt-in community reporting is disabled in Options.' };
  }
  const token = String(stored?.communityHfToken || '').trim();
  const username = String(stored?.communityHfUsername || '').trim();
  if (!token || !username) {
    return { ok: false, error: 'Verify your Hugging Face token in Options → Advanced first.' };
  }

  const sanitizedDataUrl = String(reportInput?.sanitizedDataUrl || '');
  if (!sanitizedDataUrl.startsWith('data:image/jpeg;base64,')) {
    return { ok: false, error: 'Missing sanitized JPEG preview bytes.' };
  }

  const imgBytes = dataUrlToUint8Array(sanitizedDataUrl);
  const sha256 = await sha256Hex(imgBytes);
  const shortId = sha256.slice(0, 16);
  const safeUser = username.replace(/[^a-zA-Z0-9._-]/g, '_');
  const basePath = `reports/${safeUser}/${shortId}`;
  const imgPath = `${basePath}/image.jpg`;
  const jsonPath = `${basePath}/report.json`;

  const reportedLabel = Number(reportInput?.reportedLabel) === 1 ? 1 : 0;
  const verdict = reportedLabel === 1 ? 'false_negative' : 'false_positive';
  const reportObj = {
    sha256,
    reporter_hf_user: username,
    reported_label: reportedLabel,
    verdict,
    model_id: String(reportInput?.modelId || 'unknown'),
    model_name: String(reportInput?.modelName || 'FvF Heimdall'),
    model_score: typeof reportInput?.modelScore === 'number' ? Number(reportInput.modelScore.toFixed(4)) : null,
    domain: String(reportInput?.domain || ''),
    note: String(reportInput?.note || '').trim().slice(0, 140),
    width: Number(reportInput?.width || 0),
    height: Number(reportInput?.height || 0),
    byte_length: imgBytes.byteLength,
    timestamp_utc: new Date().toISOString()
  };

  const jsonStr = JSON.stringify(reportObj, null, 2);
  const jsonBytes = new TextEncoder().encode(jsonStr);
  const imgSampleB64 = arrayBufferToBase64(imgBytes.slice(0, 512).buffer);

  const preResp = await fetch(
    `https://huggingface.co/api/datasets/${COMMUNITY_REPORTS_HF_REPO}/preupload/main?create_pr=1`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        files: [
          { path: imgPath, size: imgBytes.byteLength, sample: imgSampleB64 }
        ]
      })
    }
  );
  if (!preResp.ok) {
    const errBody = await preResp.text().catch(() => '');
    return {
      ok: false,
      error: `Hugging Face preupload failed (HTTP ${preResp.status}). Ensure ${COMMUNITY_REPORTS_HF_REPO} exists and token has write access. ${errBody.slice(0, 140)}`
    };
  }
  const preJson = await preResp.json();
  const imgMode = preJson?.files?.[0]?.uploadMode || 'regular';

  let imgOpLine = null;
  if (imgMode === 'lfs') {
    const lfsResp = await fetch(
      `https://huggingface.co/datasets/${COMMUNITY_REPORTS_HF_REPO}.git/info/lfs/objects/batch`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: 'application/vnd.git-lfs+json',
          'Content-Type': 'application/vnd.git-lfs+json'
        },
        body: JSON.stringify({
          operation: 'upload',
          transfers: ['basic'],
          objects: [{ oid: sha256, size: imgBytes.byteLength }],
          hash_algo: 'sha256'
        })
      }
    );
    if (!lfsResp.ok) {
      return { ok: false, error: `Hugging Face LFS batch failed (HTTP ${lfsResp.status})` };
    }
    const lfsData = await lfsResp.json();
    const obj = lfsData?.objects?.[0];
    if (obj?.error) {
      const code = obj.error.code || 'error';
      const msg = obj.error.message || 'Check Hugging Face token permissions.';
      return { ok: false, error: `Hugging Face LFS rejected upload (${code}): ${msg}` };
    }
    if (obj?.actions?.upload?.href) {
      const putResp = await fetch(obj.actions.upload.href, {
        method: 'PUT',
        headers: obj.actions.upload.header || {},
        body: imgBytes
      });
      if (!putResp.ok) {
        return { ok: false, error: `Hugging Face LFS upload failed (HTTP ${putResp.status})` };
      }
      if (obj.actions.verify?.href) {
        const verifyHeaders = Object.assign({ 'Content-Type': 'application/vnd.git-lfs+json' }, obj.actions.verify.header || {});
        if (obj.actions.verify.href.startsWith('https://huggingface.co/') && !verifyHeaders.Authorization) {
          verifyHeaders.Authorization = `Bearer ${token}`;
        }
        const verifyResp = await fetch(obj.actions.verify.href, {
          method: 'POST',
          headers: verifyHeaders,
          body: JSON.stringify({ oid: sha256, size: imgBytes.byteLength })
        });
        if (!verifyResp.ok) {
          return { ok: false, error: `Hugging Face LFS verify failed (HTTP ${verifyResp.status})` };
        }
      }
    }
    imgOpLine = JSON.stringify({
      key: 'lfsFile',
      value: { path: imgPath, algo: 'sha256', oid: sha256, size: imgBytes.byteLength }
    });
  } else {
    imgOpLine = JSON.stringify({
      key: 'file',
      value: { path: imgPath, content: arrayBufferToBase64(imgBytes.buffer), encoding: 'base64' }
    });
  }

  const headerLine = JSON.stringify({
    key: 'header',
    value: {
      summary: `Community report (${verdict}) by @${username} [${shortId}]`,
      description: `Verdict: ${verdict} | Model: ${reportObj.model_name} (${reportObj.model_score ?? 'n/a'}) | Domain: ${reportObj.domain || 'n/a'}${reportObj.note ? ` | Note: ${reportObj.note}` : ''}`
    }
  });
  const jsonOpLine = JSON.stringify({
    key: 'file',
    value: { path: jsonPath, content: arrayBufferToBase64(jsonBytes.buffer), encoding: 'base64' }
  });

  const commitResp = await fetch(
    `https://huggingface.co/api/datasets/${COMMUNITY_REPORTS_HF_REPO}/commit/main?create_pr=1`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/x-ndjson'
      },
      body: `${headerLine}
${imgOpLine}
${jsonOpLine}
`
    }
  );
  if (!commitResp.ok) {
    const errText = await commitResp.text().catch(() => '');
    return { ok: false, error: `Hugging Face commit PR failed (HTTP ${commitResp.status}): ${errText.slice(0, 180)}` };
  }
  const commitJson = await commitResp.json().catch(() => ({}));
  return {
    ok: true,
    prUrl: commitJson?.pullRequestUrl || commitJson?.commitUrl || `https://huggingface.co/datasets/${COMMUNITY_REPORTS_HF_REPO}/discussions`,
    sha256
  };
}
