/* global browser, chrome */
const ext = typeof browser !== 'undefined' ? browser : chrome;

console.log('[Background] Script loaded, ext:', ext ? 'Available' : 'Unavailable');

// Name of the native messaging host (update to match your installed host manifest)
const HOST_NAME = 'com.aidetector.classifier';
const GITHUB_PAGES_ONBOARDING_URL = 'https://luckycowmoo.github.io/Fire-vs-Fire/?onboarding=1';
const RAW_GITHUB_CATALOG_URL = 'https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/main/native/official_models.json';

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

// Development aid: set to true to simulate native responses when the
// native messaging API isn't available yet.
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

function expireStaleJobs() {
  const now = Date.now();
  for (const [jobId, info] of pendingJobs) {
    if (now - (info.createdAt || 0) > JOB_STALE_MS) {
      pendingJobs.delete(jobId);
      endClassificationActivity();
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

    if (resType === 'jobCancelled') {
      const jobId = response.jobId;
      console.log(`[Native] Job ${jobId} cancellation confirmed`);
      pendingJobs.delete(jobId);
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
        endClassificationActivity();
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

    if (pendingJobs.size > 0) {
      pendingJobs.clear();
      activeClassifyCount = 0;
      updateToolbarState();
    }
  });

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

async function openOnboardingExperience(targetScreen = 1) {
  // Always use local extension docs page for zero-latency direct native messaging
  const localOnboardingUrl = ext.runtime.getURL(`docs/index.html?screen=${targetScreen}`);
  if (ext.tabs && ext.tabs.create) {
    ext.tabs.create({ url: localOnboardingUrl });
  }
}

async function fetchFallbackCatalog() {
  try {
    const resp = await fetch(RAW_GITHUB_CATALOG_URL, { cache: 'no-store' });
    if (resp && resp.ok) {
      return await resp.json();
    }
  } catch (_) {}

  try {
    const localUrl = ext.runtime.getURL('docs/official_models.json');
    const resp = await fetch(localUrl);
    if (resp && resp.ok) {
      return await resp.json();
    }
  } catch (_) {}

  return { catalog_version: 2, repository: 'LuckyCowMoo/Fire-vs-Fire', models: [] };
}

async function buildAndSaveAutoCategories(downloadedModelIds = [], customThreshold = 0.75) {
  const catalog = await fetchFallbackCatalog();
  const models = Array.isArray(catalog.models) ? catalog.models : [];
  const selectedSet = new Set(Array.isArray(downloadedModelIds) ? downloadedModelIds : []);

  // If no explicit list was passed, check native host for installed models
  if (selectedSet.size === 0) {
    try {
      const catResp = await sendToNative({ type: 'getOfficialCatalog', version: 2 }, { timeoutMs: 8000 });
      if (catResp && Array.isArray(catResp.installedModels)) {
        catResp.installedModels.forEach((id) => selectedSet.add(id));
      }
    } catch (_) {}
  }

  // Always fallback to at least fvf_heimdall03 (and fvf_heimdall02) if nothing was reported
  if (selectedSet.size === 0) {
    selectedSet.add('fvf_heimdall03');
    selectedSet.add('fvf_heimdall02');
  }

  const threshold = typeof customThreshold === 'number'
    ? Math.max(0.05, Math.min(0.99, customThreshold))
    : 0.75;

  const newEnsembles = [];

  // 1. Create a category for each selected/downloaded AI detector model
  for (const m of models) {
    if (!selectedSet.has(m.id) || m.is_ai_detector === false) continue;
    const mod = m.modality || 'image';
    const borderColor = m.default_color || '#FF4D1C';
    const ensemble = {
      id: `auto_${mod}_${m.id}`,
      name: m.category_name || m.name || `AI ${mod}`,
      modality: mod === 'text' ? 'text' : 'image',
      enabled: true,
      threshold,
      classifiers: [m.id],
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
              blurAmount: 6,
              blurMode: 'hover',
              applyBorder: true,
              borderMultiplier: 1.2,
              borderColor,
              borderMode: 'hover',
              applyBadge: true,
              badgeMode: 'hover'
            }
          }
    };
    newEnsembles.push(ensemble);
  }

  // 2. Also create dedicated categories for object classifiers (e.g. ResNet 50 - Spider)
  for (const m of models) {
    if (!selectedSet.has(m.id) || m.is_ai_detector !== false) continue;
    const borderColor = m.default_color || '#FFB020';
    newEnsembles.push({
      id: `auto_image_${m.id}`,
      name: m.category_name || m.name || 'Object Detector',
      modality: 'image',
      enabled: true,
      threshold,
      classifiers: [m.id],
      weights: [1],
      styles: {
        image: {
          applyBlur: true,
          blurAmount: 8,
          blurMode: 'hover',
          applyBorder: true,
          borderMultiplier: 1.2,
          borderColor,
          borderMode: 'hover',
          applyBadge: true,
          badgeMode: 'hover'
        }
      }
    });
  }

  const storagePayload = {
    enabled: true,
    imageAiThreshold: threshold,
    textAiThreshold: threshold,
    ensembleConfigsV2: newEnsembles,
    onboardingCompleted: true,
    onboardingCompletedAt: Date.now()
  };

  const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
  if (storageArea && storageArea.set) {
    await storageArea.set(storagePayload);
  }
  if (ext.storage && ext.storage.local && ext.storage.local !== storageArea) {
    try {
      await ext.storage.local.set(storagePayload);
    } catch (_) {}
  }

  // Trigger reclassify on open tabs
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
                latestVersion: cat.host_version || '2.1.0',
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
    sendToNative({ type: 'getOfficialCatalog', version: 2 }, { timeoutMs: 8000 })
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

  if (message && message.type === 'GET_ACTIVITY_STATUS') {
    sendResponse({ ok: true, isLoading: activeClassifyCount > 0, activeCount: activeClassifyCount });
    return false;
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
