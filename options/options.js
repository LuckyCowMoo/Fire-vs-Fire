/* global browser, chrome, FVFVisualEffects, FVFEnsembles */
const ext = typeof browser !== 'undefined' ? browser : chrome;
const VE = globalThis.FVFVisualEffects;
const ENS = globalThis.FVFEnsembles;

const PREVIEW_IMAGE_PATH = 'options/preview-fire.jpg';
const LOREM =
  "The crowd was quiet, rather pleased than otherwise; whew'd and whistled when the breeze came as if to encourage it: 'there's a flare-up (what we call shine) for the House o' Lords.'—'A judgment for the Poor-Law Bill!'—'There go their hacts' (acts)! Such exclamations seemed to be the prevailing ones. A man sorry I did not anywhere see.";

const MODE_ICONS = {
  off:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 6c3.79 0 7.17 2.13 8.82 5.5-.64 1.35-1.58 2.53-2.74 3.47l1.47 1.47c1.44-1.24 2.58-2.85 3.25-4.71C21.27 7.11 17 4 12 4c-1.27 0-2.49.2-3.64.57l1.65 1.65C10.66 6.09 11.32 6 12 6zm-1.07 1.14L13 9.21c.57.25 1.03.71 1.28 1.28l2.07 2.07c.08-.32.13-.66.13-1.01 0-2.21-1.79-4-4-4-.35 0-.69.05-1.01.13zM3.27 2.27 2 3.54l2.52 2.52C3.08 8.2 1.78 10.02 1 12c1.73 4.39 6 7.5 11 7.5 1.55 0 3.03-.3 4.38-.84l2.89 2.89 1.27-1.27L3.27 2.27zM7.53 9.8l1.55 1.55c-.05.21-.08.43-.08.65 0 1.66 1.34 3 3 3 .22 0 .44-.03.65-.08l1.55 1.55c-.67.33-1.41.53-2.2.53-2.76 0-5-2.24-5-5 0-.79.2-1.53.53-2.2z"/></svg>',
  hover:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13.64 21.97a.99.99 0 0 1-1.33-.47l-2.18-4.74-2.51 2.02c-.17.14-.38.22-.61.22a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1c.23 0 .44.08.62.22l2.51 2.02 2.18-4.74a1 1 0 0 1 1.33-.47c.41.2.67.62.67 1.08v16.86c0 .46-.26.88-.67 1.08z"/></svg>',
  always:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 12V4h1c.55 0 1-.45 1-1V2c0-.55-.45-1-1-1H7c-.55 0-1 .45-1 1v1c0 .55.45 1 1 1h1v8l-2 2v2h5.2v6h1.6v-6H18v-2l-2-2z"/></svg>'
};

const enabledInput = document.getElementById('enabled');
const altTextOnlyInput = document.getElementById('altTextOnly');
const ignoreSmallImagesInput = document.getElementById('ignoreSmallImages');
const investigatorInput = document.getElementById('investigator');
const minImageDimensionInput = document.getElementById('minImageDimension');
const classificationDelayInput = document.getElementById('classificationDelay');
const classificationDelayValue = document.getElementById('classificationDelayValue');
const imageCaptureQualityInput = document.getElementById('imageCaptureQuality');
const imageCaptureQualityValue = document.getElementById('imageCaptureQualityValue');
const verboseLogsInput = document.getElementById('verboseLogs');
const lazyLoadInput = document.getElementById('lazyLoad');
const disableUpdateNotificationsInput = document.getElementById('disableUpdateNotifications');
const enableManualBlurContextMenuInput = document.getElementById('enableManualBlurContextMenu');
const minTextLengthInput = document.getElementById('minTextLength');

const pauseOnBatteryInput = document.getElementById('pauseOnBattery');
const pauseBelowThresholdInput = document.getElementById('pauseBelowThreshold');
const batteryThresholdSelect = document.getElementById('batteryThreshold');

let isHostConnected = false;
let isClassifying = false;
let classifyCount = 0;
let isPowerPaused = false;
let currentPowerStatus = null;
let disableUpdateNotifications = false;
let settingsLoaded = false;
let latestUpdateStatus = null;

const refreshBtn = document.getElementById('refresh-models');
const modelStatusEl = document.getElementById('model-status');
const updateBannerWrapEl = document.getElementById('update-banner-wrap');
const hostUpdateBannerBtn = document.getElementById('host-update-banner-btn');
const classifierUpdateBannerBtn = document.getElementById('classifier-update-banner-btn');
const updateBannerStatusEl = document.getElementById('update-banner-status');

const categoryStripWrapEl = document.getElementById('category-strip-wrap');
const categoryStripEl = document.getElementById('category-strip');
const categoryPanelEl = document.getElementById('category-panel');
const addEnsembleBtn = document.getElementById('add-ensemble');
const runStateEl = document.getElementById('run-state');
const runStateTextEl = document.getElementById('run-state-text');
const statusLogoEl = document.getElementById('status-logo');
const particleLayerEl = document.getElementById('particle-layer');

const PARTICLE_COLORS = ['#ff4400', '#ff6622', '#ff8833', '#ffaa44', '#ff0064', '#ff5722'];
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const lastPointer = { x: 0, y: 0 };
let lastColorParticleAt = 0;

function trackPointerPosition() {
  document.addEventListener(
    'pointermove',
    (event) => {
      lastPointer.x = event.clientX;
      lastPointer.y = event.clientY;
    },
    { passive: true }
  );
}

function clampChannel(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function varyRgb({ r, g, b }, variance = 22) {
  const jitter = () => (Math.random() - 0.5) * variance * 2;
  return {
    r: clampChannel(r + jitter()),
    g: clampChannel(g + jitter()),
    b: clampChannel(b + jitter())
  };
}

function rgbToCss({ r, g, b }) {
  return `rgb(${r}, ${g}, ${b})`;
}

function pickParticleColor(options) {
  if (options.colorPicker && options.baseRgb) {
    return pickColorSliderParticleColor(options.baseRgb, options.variance ?? 46);
  }
  const colors = options.colors ?? PARTICLE_COLORS;
  return colors[Math.floor(Math.random() * colors.length)];
}

function pickColorSliderParticleColor(baseRgb, variance = 46) {
  const roll = Math.random();
  if (roll < 0.22) {
    return rgbToCss({
      r: clampChannel(baseRgb.r + 18 + Math.random() * 42),
      g: clampChannel(baseRgb.g + 10 + Math.random() * 38),
      b: clampChannel(baseRgb.b + Math.random() * 32)
    });
  }
  if (roll < 0.38) {
    return rgbToCss(
      varyRgb(
        {
          r: baseRgb.r * (0.62 + Math.random() * 0.18),
          g: baseRgb.g * (0.62 + Math.random() * 0.18),
          b: baseRgb.b * (0.62 + Math.random() * 0.18)
        },
        variance * 0.55
      )
    );
  }
  return rgbToCss(varyRgb(baseRgb, variance));
}

function sliderIntensity(value, min = 0, max = 255) {
  const span = Math.max(1, max - min);
  return Math.max(0, Math.min(1, (value - min) / span));
}

function rangeInputIntensity(input) {
  const min = parseFloat(input.min);
  const max = parseFloat(input.max);
  const value = parseFloat(input.value);
  const safeMin = Number.isFinite(min) ? min : 0;
  const safeMax = Number.isFinite(max) ? max : 255;
  const safeValue = Number.isFinite(value) ? value : safeMin;
  return sliderIntensity(safeValue, safeMin, safeMax);
}

function getRangeThumbPosition(input) {
  const rect = input.getBoundingClientRect();
  const min = parseFloat(input.min);
  const max = parseFloat(input.max);
  const value = parseFloat(input.value);
  const safeMin = Number.isFinite(min) ? min : 0;
  const safeMax = Number.isFinite(max) ? max : 100;
  const safeValue = Number.isFinite(value) ? value : safeMin;
  const ratio = (safeValue - safeMin) / Math.max(safeMax - safeMin, 1);

  const thumbRadius = 8;
  const trackWidth = Math.max(rect.width - thumbRadius * 2, 1);

  return {
    x: rect.left + thumbRadius + trackWidth * ratio,
    y: rect.top + rect.height / 2
  };
}

function particleSizeScale(intensity) {
  if (typeof intensity !== 'number') return 1;
  return 0.5 + intensity;
}

function pickParticleEase() {
  const startY = 0.02 + Math.random() * 0.14;
  const endY = 0.38 + Math.random() * 0.28;
  const endX = 0.84 + Math.random() * 0.1;
  return `cubic-bezier(0.52, ${startY.toFixed(3)}, ${endX.toFixed(3)}, ${endY.toFixed(3)})`;
}

function fireMotionProfile() {
  const rise = 38 + Math.random() * 78;
  const horizontal = (Math.random() - 0.5) * (36 + Math.random() * 52);
  const kick = 0.05 + Math.random() * 0.14;

  return {
    tx: horizontal,
    ty: -rise,
    tx0: horizontal * kick,
    ty0: -rise * kick * (0.75 + Math.random() * 0.5),
    duration: 0.48 + Math.random() * 0.72,
    ease: pickParticleEase()
  };
}

function spawnParticles(clientX, clientY, options = {}) {
  if (prefersReducedMotion || !particleLayerEl) return;
  if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return;

  const count = options.count ?? 8;
  const sizeScale = particleSizeScale(options.intensity);
  const spread = options.spread ?? 14;

  for (let i = 0; i < count; i += 1) {
    const particle = document.createElement('span');
    particle.className = 'particle';

    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * spread;
    const offsetX = Math.cos(angle) * radius;
    const offsetY = Math.sin(angle) * radius;
    const width = (2 + Math.random() * 4) * sizeScale;
    const height = width * (1.45 + Math.random() * 1.35);
    const motion = fireMotionProfile();
    const color = pickParticleColor(options);

    particle.style.left = `${clientX + offsetX}px`;
    particle.style.top = `${clientY + offsetY}px`;
    particle.style.width = `${width}px`;
    particle.style.height = `${height}px`;
    particle.style.background = color;
    particle.style.boxShadow = `0 0 ${(4 + Math.random() * 6) * sizeScale}px ${color}`;
    particle.style.setProperty('--tx0', `${motion.tx0}px`);
    particle.style.setProperty('--ty0', `${motion.ty0}px`);
    particle.style.setProperty('--tx', `${motion.tx}px`);
    particle.style.setProperty('--ty', `${motion.ty}px`);
    particle.style.animationDuration = `${motion.duration}s`;
    particle.style.animationTimingFunction = motion.ease;

    particleLayerEl.appendChild(particle);
    particle.addEventListener('animationend', () => particle.remove(), { once: true });
  }
}

function spawnColorParticles(clientX, clientY, rgb, sliderValue = 0) {
  const intensity = sliderIntensity(sliderValue, 0, 255);
  const now = performance.now();
  if (now - lastColorParticleAt < 32) return;
  lastColorParticleAt = now;

  spawnParticles(clientX, clientY, {
    count: 2,
    intensity,
    baseRgb: rgb,
    colorPicker: true,
    variance: 48
  });
}

function pointerCoords(event) {
  if (event && Number.isFinite(event.clientX) && Number.isFinite(event.clientY)) {
    return { x: event.clientX, y: event.clientY };
  }
  return { x: lastPointer.x, y: lastPointer.y };
}

function initCollapsibleAnimations() {
  document.querySelectorAll('.collapsible').forEach((details) => {
    details.addEventListener('toggle', () => {
      if (details.open && !prefersReducedMotion) {
        spawnParticles(lastPointer.x, lastPointer.y, { count: 8 });
      }
    });
  });
}

function initInteractionParticles() {
  if (prefersReducedMotion) return;

  trackPointerPosition();

  const root = document.querySelector('.popup-root');
  if (!root) return;

  root.addEventListener(
    'click',
    (event) => {
      const target = event.target.closest(
        'button, .switch, .category-tile, .mode-btn'
      );
      if (!target) return;

      let colors = PARTICLE_COLORS;
      const tileColor = target.style.getPropertyValue('--tile-color');
      if (tileColor) {
        colors = [tileColor, '#ff6622', '#ffaa44'];
      }

      const { x, y } = pointerCoords(event);
      const count = target.classList.contains('category-tile') ? 14 : 8;
      spawnParticles(x, y, { count, colors });
    },
    true
  );

  root.addEventListener(
    'input',
    (event) => {
      const input = event.target;
      if (!(input instanceof HTMLInputElement)) return;
      if (input.type !== 'range') return;
      if (input.classList.contains('color-channel-slider')) return;

      const thumb = getRangeThumbPosition(input);
      spawnParticles(thumb.x, thumb.y, {
        count: 4,
        intensity: rangeInputIntensity(input)
      });
    },
    true
  );

  root.addEventListener('change', (event) => {
    const input = event.target;
    if (!(input instanceof HTMLInputElement)) return;
    if (input.type !== 'checkbox') return;

    spawnParticles(lastPointer.x, lastPointer.y, { count: 6 });
  });
}

function animateCategoryPanelEntrance() {
  if (!categoryPanelEl || prefersReducedMotion) return;
  categoryPanelEl.classList.remove('category-panel--drop-in');
  void categoryPanelEl.offsetWidth;
  categoryPanelEl.classList.add('category-panel--drop-in');
}

const CATEGORY_PANEL_MODE = {
  DETAIL: 'detail',
  PICK_MODALITY: 'pick_modality',
  PICK_CLASSIFIER: 'pick_classifier'
};

let categoryPanelState = {
  mode: CATEGORY_PANEL_MODE.DETAIL,
  context: 'edit',
  draftModality: null,
  targetEnsembleIndex: null,
  previousIndex: 0
};

let availableClassifiers = [];
let ensembleConfigs = [];
let runStateTimer = null;
let classifierTimings = {};
let selectedCategoryIndex = 0;
let lockedPopupWidth = 0;

function getDocumentScrollY() {
  return window.scrollY || document.documentElement.scrollTop || 0;
}

function restoreDocumentScrollY(scrollY) {
  window.scrollTo(0, scrollY);
}

function scrollCategoryStripToSelected() {
  if (!categoryStripEl) return;
  const selectedTile = categoryStripEl.children[selectedCategoryIndex];
  if (!selectedTile) return;

  const tileLeft = selectedTile.offsetLeft;
  const tileWidth = selectedTile.offsetWidth;
  const stripWidth = categoryStripEl.clientWidth;
  categoryStripEl.scrollLeft = Math.max(0, tileLeft - (stripWidth - tileWidth) / 2);
}

function updateClassifierRateLabels() {
  if (!categoryPanelEl) return;
  categoryPanelEl.querySelectorAll('[data-classifier-id]').forEach((row) => {
    const classifierId = row.dataset.classifierId;
    const rateEl = row.querySelector('.classifier-rate');
    const model = availableClassifiers.find((m) => m.id === classifierId);
    if (!rateEl || !model) return;
    rateEl.textContent = formatClassifierRate(classifierId, model.modalities || []);
  });
}

function stabilizePopupWidth() {
  const html = document.documentElement;
  const winW = window.innerWidth || html.clientWidth || 0;
  if (winW >= 680) {
    html.style.width = '';
    html.style.minWidth = '';
    lockedPopupWidth = 0;
    return;
  }
  const measured = Math.max(html.clientWidth || 0, 420);
  if (measured > lockedPopupWidth) {
    lockedPopupWidth = measured;
    html.style.width = `${lockedPopupWidth}px`;
    html.style.minWidth = `${lockedPopupWidth}px`;
  }
}

function getPreviewImageUrl() {
  if (ext.runtime && typeof ext.runtime.getURL === 'function') {
    return ext.runtime.getURL(PREVIEW_IMAGE_PATH);
  }
  return './preview-fire.jpg';
}

function updateHeaderStatus() {
  if (!runStateEl || !runStateTextEl) return;
  runStateEl.className = 'run-state';

  if (!isHostConnected) {
    runStateEl.classList.add('disconnected');
    runStateTextEl.textContent = 'Native Host Disconnected';
    if (statusLogoEl && ext.runtime?.getURL) {
      statusLogoEl.src = ext.runtime.getURL('icons/icon.svg');
    }
  } else if (isPowerPaused) {
    runStateEl.classList.add('paused');
    const reason = (currentPowerStatus && currentPowerStatus.is_plugged_in === false) ? 'On Battery' : 'Low Battery';
    runStateTextEl.textContent = `Connected — Paused (${reason})`;
    if (statusLogoEl && ext.runtime?.getURL) {
      statusLogoEl.src = ext.runtime.getURL('icons/icon.svg');
    }
  } else if (isClassifying) {
    runStateEl.classList.add('loading');
    runStateTextEl.textContent = classifyCount > 0
      ? `Connected — Classifying (${classifyCount})`
      : 'Connected — Classifying';
    if (statusLogoEl && ext.runtime?.getURL) {
      statusLogoEl.src = ext.runtime.getURL('icons/icon-loading.svg');
    }
  } else {
    runStateEl.classList.add('idle');
    runStateTextEl.textContent = 'Connected — Idle';
    if (statusLogoEl && ext.runtime?.getURL) {
      statusLogoEl.src = ext.runtime.getURL('icons/icon.svg');
    }
  }
}

function setRunState(isLoading, activeCount = 0) {
  isClassifying = !!isLoading;
  classifyCount = Math.max(0, Number(activeCount) || 0);
  updateHeaderStatus();
}

async function refreshRunState() {
  if (document.hidden) return;
  try {
    const resp = await ext.runtime.sendMessage({ type: 'GET_ACTIVITY_STATUS' });
    if (!resp || resp.ok !== true) return;
    isClassifying = !!resp.isLoading;
    classifyCount = Math.max(0, Number(resp.activeCount) || 0);
    if (typeof resp.isPaused === 'boolean') {
      isPowerPaused = resp.isPaused;
    }
    if (resp.power) {
      currentPowerStatus = resp.power;
      updatePowerUI(resp.power);
    }
    updateHeaderStatus();
  } catch {
    // Ignore transient messaging failures
  }
}

async function refreshPowerStatus() {
  try {
    if (typeof navigator !== 'undefined' && typeof navigator.getBattery === 'function') {
      try {
        const b = await navigator.getBattery();
        currentPowerStatus = {
          ok: true,
          is_plugged_in: Boolean(b.charging),
          battery_percent: Math.round(b.level * 100),
          charging: Boolean(b.charging),
          has_battery: true
        };
      } catch (_) {}
    }

    if (!currentPowerStatus) {
      const resp = await ext.runtime.sendMessage({ type: 'GET_POWER_STATUS' }).catch(() => null);
      if (resp && resp.ok) {
        currentPowerStatus = resp;
      }
    }

    if (currentPowerStatus) {
      updatePowerUI(currentPowerStatus);
    }
  } catch (_) {}
}

function updatePowerUI(power) {
  if (!power) return;
  const isPlugged = power.is_plugged_in !== false;
  const pct = typeof power.battery_percent === 'number' ? power.battery_percent : null;

  const pauseOnBattery = pauseOnBatteryInput ? pauseOnBatteryInput.checked : false;
  const pauseBelow = pauseBelowThresholdInput ? pauseBelowThresholdInput.checked : false;
  const threshold = parseInt(batteryThresholdSelect ? batteryThresholdSelect.value : '20', 10);

  isPowerPaused = (pauseOnBattery && !isPlugged) ||
                  (pauseBelow && pct != null && pct <= threshold);

  updateHeaderStatus();
}

function updateDisplays() {
  classificationDelayValue.textContent = classificationDelayInput.value;
  imageCaptureQualityValue.textContent = parseFloat(imageCaptureQualityInput.value).toFixed(2);
  document.querySelectorAll('input[type="range"].range-accent, input[type="range"]').forEach((slider) => {
    if (slider.id !== 'classificationDelay' && slider.id !== 'imageCaptureQuality') return;
    updateSliderBackground(slider);
  });
}

function updateSliderBackground(slider) {
  const min = parseFloat(slider.min) || 0;
  const max = parseFloat(slider.max) || 100;
  const value = parseFloat(slider.value) || 0;
  const percent = ((value - min) / (max - min)) * 100;
  const root = getComputedStyle(document.documentElement);
  const fill = root.getPropertyValue('--slider-fill').trim() || '#ff4400';
  const track = root.getPropertyValue('--slider-track').trim() || '#260a00';
  slider.style.background = `linear-gradient(to right, ${fill} 0%, ${fill} ${percent}%, ${track} ${percent}%, ${track} 100%)`;
}

function getEnsembleAccentColor(ensemble) {
  const mod = ensemble.modality || 'image';
  if (mod === 'image' || mod === 'video') {
    return ensemble.styles?.image?.borderColor || ensemble.styles?.video?.borderColor || '#FF0064';
  }
  if (mod === 'text') {
    const text = ensemble.styles?.text || {};
    return text.strikethroughColor || text.underlineColor || text.highlightColor || '#FF0064';
  }
  if (mod === 'audio') {
    return ensemble.styles?.audio?.badgeColor || '#FF0064';
  }
  return '#FF0064';
}

function createIconModeControl(labelText, value, onChange) {
  const row = document.createElement('div');
  row.className = 'style-row';

  const header = document.createElement('div');
  header.className = 'style-row-header';

  const label = document.createElement('span');
  label.className = 'style-row-label';
  label.textContent = labelText;

  const trio = document.createElement('div');
  trio.className = 'mode-trio';
  trio.setAttribute('role', 'group');
  trio.setAttribute('aria-label', `${labelText} mode`);

  const initialMode = value || 'hover';
  trio.dataset.mode = initialMode;

  // Smooth sliding frosted glass glider
  const glider = document.createElement('span');
  glider.className = 'mode-glider';
  glider.setAttribute('aria-hidden', 'true');
  trio.appendChild(glider);

  const modes = [
    { value: 'off', label: 'Off' },
    { value: 'hover', label: 'Hover' },
    { value: 'always', label: 'Always' }
  ];

  modes.forEach((mode) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'mode-btn';
    btn.title = mode.label;
    btn.setAttribute('aria-label', mode.label);
    btn.innerHTML = MODE_ICONS[mode.value];
    if (initialMode === mode.value) btn.classList.add('active');
    btn.addEventListener('click', () => {
      trio.dataset.mode = mode.value;
      trio.querySelectorAll('.mode-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      onChange(mode.value);
    });
    trio.appendChild(btn);
  });

  header.appendChild(label);
  header.appendChild(trio);
  row.appendChild(header);
  return row;
}

function appendSliderToRow(row, slider) {
  slider.className = 'style-row-slider';
  updateSliderBackground(slider);
  row.appendChild(slider);
}

function createColorSliders(labelText, initialColor, onChange) {
  const wrapper = document.createElement('div');
  wrapper.className = 'color-sliders-wrapper';

  const label = document.createElement('span');
  label.textContent = labelText;
  wrapper.appendChild(label);

  const slidersContainer = document.createElement('div');
  slidersContainer.className = 'sliders-container';

  const rgb = VE.hexToRgb(initialColor) || { r: 255, g: 0, b: 100 };
  const preview = document.createElement('div');
  preview.className = 'color-preview';
  preview.style.backgroundColor = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;

  const readCurrentRgb = () => ({
    r: parseInt(rInput.value, 10) || 0,
    g: parseInt(gInput.value, 10) || 0,
    b: parseInt(bInput.value, 10) || 0
  });

  let rInput;
  let gInput;
  let bInput;

  const emitColorParticles = (_event, sliderInput) => {
    const thumb = getRangeThumbPosition(sliderInput);
    const sliderValue = parseInt(sliderInput.value, 10) || 0;
    spawnColorParticles(thumb.x, thumb.y, readCurrentRgb(), sliderValue);
  };

  const makeSlider = (name, value) => {
    const container = document.createElement('label');
    container.textContent = `${name}`;
    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = '255';
    input.value = String(value);
    input.className = 'color-channel-slider';

    if (name === 'R') {
      input.style.background = `linear-gradient(to right, rgb(0, ${rgb.g}, ${rgb.b}), rgb(255, ${rgb.g}, ${rgb.b}))`;
    } else if (name === 'G') {
      input.style.background = `linear-gradient(to right, rgb(${rgb.r}, 0, ${rgb.b}), rgb(${rgb.r}, 255, ${rgb.b}))`;
    } else if (name === 'B') {
      input.style.background = `linear-gradient(to right, rgb(${rgb.r}, ${rgb.g}, 0), rgb(${rgb.r}, ${rgb.g}, 255))`;
    }

    const valueEl = document.createElement('span');
    valueEl.textContent = String(value);

    const syncColor = (event) => {
      valueEl.textContent = input.value;
      const updated = readCurrentRgb();

      rInput.style.background = `linear-gradient(to right, rgb(0, ${updated.g}, ${updated.b}), rgb(255, ${updated.g}, ${updated.b}))`;
      gInput.style.background = `linear-gradient(to right, rgb(${updated.r}, 0, ${updated.b}), rgb(${updated.r}, 255, ${updated.b}))`;
      bInput.style.background = `linear-gradient(to right, rgb(${updated.r}, ${updated.g}, 0), rgb(${updated.r}, ${updated.g}, 255))`;

      preview.style.backgroundColor = `rgb(${updated.r}, ${updated.g}, ${updated.b})`;
      onChange(VE.rgbToHex(updated));
      emitColorParticles(event, input);
    };

    input.addEventListener('input', syncColor);
    input.addEventListener('pointermove', (event) => {
      if (event.buttons !== 1) return;
      emitColorParticles(event, input);
    });

    container.appendChild(input);
    container.appendChild(valueEl);
    return { container, input };
  };

  const rSlider = makeSlider('R', rgb.r);
  rInput = rSlider.input;
  const gSlider = makeSlider('G', rgb.g);
  gInput = gSlider.input;
  const bSlider = makeSlider('B', rgb.b);
  bInput = bSlider.input;

  slidersContainer.appendChild(rSlider.container);
  slidersContainer.appendChild(gSlider.container);
  slidersContainer.appendChild(bSlider.container);

  wrapper.appendChild(slidersContainer);
  wrapper.appendChild(preview);

  return wrapper;
}

function createLivePreview(ensemble) {
  const wrap = document.createElement('div');
  wrap.className = 'live-preview';

  const stage = document.createElement('div');
  stage.className = 'preview-stage';

  let targetEl = null;
  let isHovered = false;
  const previewSettings = {
    blurAmount: 3,
    borderMultiplier: 1,
    borderColor: '#ff0064',
    textBlurAmount: 2
  };

  if ((ensemble.modality || 'image') === 'image') {
    stage.classList.add('preview-stage-image');
    const imageWrap = document.createElement('div');
    imageWrap.className = 'preview-image-wrap';

    targetEl = document.createElement('img');
    targetEl.src = getPreviewImageUrl();
    targetEl.alt = 'Preview';
    targetEl.addEventListener('load', () => {
      requestAnimationFrame(() => {
        stabilizePopupWidth();
        refresh();
      });
    });

    imageWrap.appendChild(targetEl);
    stage.appendChild(imageWrap);

    stage.addEventListener('mouseenter', () => {
      isHovered = true;
      if (targetEl) VE.applyImageInteractionState(targetEl, 'hover');
    });
    stage.addEventListener('mouseleave', () => {
      isHovered = false;
      if (targetEl) VE.applyImageInteractionState(targetEl, 'rest');
    });
  } else {
    targetEl = document.createElement('p');
    targetEl.className = 'preview-text';
    targetEl.textContent = LOREM;
    stage.appendChild(targetEl);
  }

  const refresh = () => {
    if (!targetEl || !VE) return;
    VE.applyPreviewFromEnsemble(targetEl, ensemble, previewSettings, availableClassifiers);
    if (isHovered && (ensemble.modality || 'image') === 'image') {
      VE.applyImageInteractionState(targetEl, 'hover');
    }
  };

  refresh();
  wrap.appendChild(stage);

  const caption = document.createElement('p');
  caption.className = 'preview-caption';
  if ((ensemble.modality || 'image') === 'image') {
    caption.textContent =
      '“The Burning of the Houses of Lords and Commons, October 16, 1834” by real human J. M. W. Turner';
  } else {
    caption.textContent =
      'Eyewitness account by Thomas Carlyle (1834) of the Houses of Parliament burning.';
  }
  wrap.appendChild(caption);

  return { element: wrap, refresh };
}

function createSwitch(checked, onChange) {
  const label = document.createElement('label');
  label.className = 'switch';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = checked;
  const track = document.createElement('span');
  track.className = 'capsule-track';
  track.setAttribute('aria-hidden', 'true');
  const thumb = document.createElement('span');
  thumb.className = 'capsule-thumb';
  const dots = document.createElement('span');
  dots.className = 'capsule-dots';
  dots.innerHTML = '<i></i><i></i><i></i>';
  thumb.appendChild(dots);
  track.appendChild(thumb);
  input.addEventListener('change', () => onChange(input.checked));
  label.appendChild(input);
  label.appendChild(track);
  return label;
}

function generateEnsembleId(modality) {
  const prefix = modality || 'cat';
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
}

function formatModalityLabel(modality) {
  if (!modality) return 'Images';
  switch (String(modality).toLowerCase()) {
    case 'image': return 'Images';
    case 'text': return 'Text';
    case 'video': return 'Video';
    case 'audio': return 'Audio';
    default: return String(modality).charAt(0).toUpperCase() + String(modality).slice(1);
  }
}

function getModalityIconSrc(modality) {
  const mod = String(modality || 'image').toLowerCase();
  if (typeof ext !== 'undefined' && ext.runtime && typeof ext.runtime.getURL === 'function') {
    try {
      return ext.runtime.getURL(`docs/assets/modality-${mod}.png`);
    } catch (_) {}
  }
  return `../docs/assets/modality-${mod}.png`;
}

function getCompatibleClassifiers(modality) {
  const supported = modality || 'image';
  return availableClassifiers.filter((model) => {
    const modalities = Array.isArray(model.modalities)
      ? model.modalities
      : (model.modality ? [model.modality] : []);
    return modalities.length === 0 || modalities.includes(supported);
  });
}

function formatClassifierRate(classifierId, modalities) {
  const t = classifierTimings && classifierTimings[classifierId];
  if (!t || !t.avgMs || t.avgMs <= 0) return '—';
  const perSec = 1000 / t.avgMs;
  const mods = Array.isArray(modalities) ? modalities : (modalities ? [modalities] : []);
  const unit = mods.includes('image')
    ? 'img/s'
    : mods.includes('text')
      ? 'txt/s'
      : mods.includes('video')
        ? 'vid/s'
        : mods.includes('audio')
          ? 'aud/s'
          : '/s';
  return perSec >= 1 ? `${Math.round(perSec)} ${unit}` : `${perSec.toFixed(2)} ${unit}`;
}

function getAlignedWeights(classifierIds, weights) {
  const sourceWeights = Array.isArray(weights) ? weights : [];
  return classifierIds.map((_, index) => ENS.normalizeEnsembleWeight(sourceWeights[index]));
}

function normalizeWeightsToSumOne(classifierIds, weights) {
  const alignedWeights = getAlignedWeights(classifierIds, weights);
  const total = alignedWeights.reduce((sum, weight) => sum + weight, 0);
  if (!alignedWeights.length) return [];
  if (alignedWeights.length === 1) return [1];
  if (total <= 0) return alignedWeights.map(() => 1 / alignedWeights.length);
  return alignedWeights.map((weight) => weight / total);
}

function redistributeLinkedWeights(classifierIds, currentWeights, activeIndex, nextActiveValue) {
  const count = Array.isArray(classifierIds) ? classifierIds.length : 0;
  if (count <= 0) return [];
  if (count === 1) return [1];

  const nextWeights = normalizeWeightsToSumOne(classifierIds, currentWeights);
  const activeValue = Math.max(0, Math.min(1, Number(nextActiveValue) || 0));
  const otherIndices = classifierIds.map((_, index) => index).filter((index) => index !== activeIndex);

  nextWeights[activeIndex] = activeValue;
  const remaining = Math.max(0, 1 - activeValue);
  const currentOtherTotal = otherIndices.reduce((sum, index) => sum + Math.max(0, nextWeights[index]), 0);

  if (otherIndices.length === 1) {
    nextWeights[otherIndices[0]] = remaining;
  } else if (currentOtherTotal > 0) {
    otherIndices.forEach((index) => {
      nextWeights[index] = (Math.max(0, nextWeights[index]) / currentOtherTotal) * remaining;
    });
  } else {
    const evenShare = remaining / otherIndices.length;
    otherIndices.forEach((index) => {
      nextWeights[index] = evenShare;
    });
  }

  const total = nextWeights.reduce((sum, weight) => sum + weight, 0);
  const correction = 1 - total;
  if (Math.abs(correction) > 1e-9) {
    const correctionIndex = otherIndices.length ? otherIndices[otherIndices.length - 1] : activeIndex;
    nextWeights[correctionIndex] = Math.max(0, nextWeights[correctionIndex] + correction);
  }

  return nextWeights;
}

function updateEnsembleClassifiers(ensemble, nextClassifiers) {
  const currentClassifiers = Array.isArray(ensemble.classifiers) ? ensemble.classifiers : [];
  const currentWeights = Array.isArray(ensemble.weights) ? ensemble.weights : [];
  const weightByClassifier = new Map(
    currentClassifiers.map((classifierId, index) => [classifierId, currentWeights[index]])
  );

  ensemble.classifiers = nextClassifiers;
  ensemble.weights = normalizeWeightsToSumOne(
    nextClassifiers,
    nextClassifiers.map((classifierId) => weightByClassifier.get(classifierId))
  );
}

function renderWeightsEditor(ensemble, classifierIds, classifierLookup) {
  const section = document.createElement('div');
  section.className = 'ensemble-weight-editor';



  const alignedWeights = normalizeWeightsToSumOne(classifierIds, ensemble.weights);
  if (!Array.isArray(ensemble.weights) || ensemble.weights.length !== classifierIds.length) {
    ensemble.weights = alignedWeights.slice();
  }

  const weightInputs = [];
  const weightValueEls = [];

  const syncWeights = (nextWeights) => {
    const normalizedWeights = normalizeWeightsToSumOne(classifierIds, nextWeights);
    ensemble.weights = normalizedWeights;
    weightInputs.forEach((slider, index) => {
      slider.value = String(normalizedWeights[index] ?? 0);
      updateSliderBackground(slider);
    });
    weightValueEls.forEach((valueEl, index) => {
      valueEl.textContent = `${Math.round((normalizedWeights[index] ?? 0) * 100)}%`;
    });
  };

  classifierIds.forEach((classifierId, index) => {
    const row = document.createElement('div');
    row.className = 'ensemble-weight-row';

    const rowHeader = document.createElement('div');
    rowHeader.className = 'ensemble-weight-row-header';

    const name = document.createElement('span');
    name.className = 'ensemble-weight-name';
    name.textContent = classifierLookup.get(classifierId) || classifierId;

    const value = document.createElement('span');
    value.className = 'ensemble-weight-value';
    value.textContent = `${Math.round((alignedWeights[index] ?? 0) * 100)}%`;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = '1';
    input.step = '0.01';
    input.value = String(alignedWeights[index] ?? 0);
    updateSliderBackground(input);
    input.addEventListener('input', () => {
      syncWeights(redistributeLinkedWeights(classifierIds, ensemble.weights, index, input.value));
      scheduleSaveEnsembles();
    });

    rowHeader.appendChild(name);
    rowHeader.appendChild(value);
    row.appendChild(rowHeader);
    row.appendChild(input);
    section.appendChild(row);

    weightInputs.push(input);
    weightValueEls.push(value);
  });

  syncWeights(alignedWeights);
  return section;
}

function getDefaultEnsembleName(index) {
  return `Category ${index + 1}`;
}

function getCategoryLabelForSelection(selectedClassifiers, classifierLookup, index, modality = 'image') {
  if (selectedClassifiers.length === 1) {
    const cid = selectedClassifiers[0];
    const fromLookup = classifierLookup && classifierLookup.get(cid);
    if (fromLookup) return fromLookup;
    return modality === 'text' ? 'AI text' : 'AI generated';
  }
  return getDefaultEnsembleName(index);
}

function getCategoryDisplayName(ensemble, index) {
  const selectedClassifiers = Array.isArray(ensemble.classifiers) ? ensemble.classifiers : [];
  if (typeof ensemble.name === 'string' && ensemble.name.trim()) {
    return ensemble.name.trim();
  }
  const labelsMap = new Map();
  availableClassifiers.forEach((m) => {
    if (m && m.id && typeof m.label === 'string' && m.label.trim()) {
      labelsMap.set(m.id, m.label.trim());
    }
  });
  return getCategoryLabelForSelection(selectedClassifiers, labelsMap, index, ensemble.modality || 'image');
}

async function saveEnsembles() {
  try {
    const payload = { ensembleConfigsV2: ensembleConfigs };
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    if (storageArea && storageArea.set) {
      await storageArea.set(payload).catch((err) => {
        console.warn('Failed to save ensemble configs to primary storage:', err);
      });
    }
    if (ext.storage && ext.storage.local && ext.storage.local !== storageArea) {
      await ext.storage.local.set(payload).catch(() => {});
    }
  } catch (e) {
    console.error('Failed to save ensemble configs:', e);
  }
}

let saveEnsemblesTimer = null;

function scheduleSaveEnsembles() {
  if (saveEnsemblesTimer) clearTimeout(saveEnsemblesTimer);
  saveEnsemblesTimer = setTimeout(() => {
    saveEnsemblesTimer = null;
    saveEnsembles();
  }, 100);
}

function flushSaveEnsembles() {
  if (!saveEnsemblesTimer) return saveEnsembles();
  clearTimeout(saveEnsemblesTimer);
  saveEnsemblesTimer = null;
  return saveEnsembles();
}

document.addEventListener('pointerup', () => {
  if (saveEnsemblesTimer) flushSaveEnsembles();
}, true);

function normalizeSelectedIndex() {
  if (!ensembleConfigs.length) {
    selectedCategoryIndex = 0;
    return;
  }
  if (selectedCategoryIndex < 0 || selectedCategoryIndex >= ensembleConfigs.length) {
    selectedCategoryIndex = 0;
  }
}

function renderCategoryStrip(classifierNamesById) {
  if (!categoryStripEl) return;
  categoryStripEl.innerHTML = '';

  ensembleConfigs.forEach((ensemble, index) => {
    const color = getEnsembleAccentColor(ensemble);
    const name = getCategoryDisplayName(ensemble, index, classifierNamesById);
    const isEnabled = ensemble.enabled !== false;
    const isSelected = categoryPanelState.mode === CATEGORY_PANEL_MODE.DETAIL && index === selectedCategoryIndex;

    // Card Container (Holds Indented Tile & Attached Delete Button Beneath)
    const card = document.createElement('div');
    card.className = `category-card ${isSelected ? 'selected' : ''} ${!isEnabled ? 'disabled-cat' : ''}`;
    card.style.setProperty('--tile-color', color);

    // Indented Tile (Hole in canvas with tint of category color)
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = `category-tile ${isSelected ? 'selected' : ''} ${!isEnabled ? 'disabled-cat' : ''}`;
    tile.setAttribute('role', 'tab');
    tile.setAttribute('aria-selected', isSelected ? 'true' : 'false');
    tile.style.setProperty('--tile-color', color);

    const nameEl = document.createElement('span');
    nameEl.className = 'tile-name';
    nameEl.textContent = name;

    const statusEl = document.createElement('span');
    statusEl.className = 'tile-status';
    statusEl.textContent = isEnabled ? 'Enabled' : 'Disabled';

    tile.appendChild(nameEl);
    tile.appendChild(statusEl);

    tile.addEventListener('click', () => {
      categoryPanelState.mode = CATEGORY_PANEL_MODE.DETAIL;
      categoryPanelState.context = 'edit';
      categoryPanelState.draftModality = null;
      categoryPanelState.targetEnsembleIndex = null;
      selectedCategoryIndex = index;
      renderEnsembles({ scrollStripToSelection: true });
    });

    // Attached Remove Button beneath (3s hold to delete with left-to-right fill)
    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'category-delete-btn';
    deleteBtn.setAttribute('aria-label', `Hold for 3 seconds to remove category ${name}`);

    const fillEl = document.createElement('div');
    fillEl.className = 'category-delete-fill';

    const labelWrap = document.createElement('span');
    labelWrap.className = 'category-delete-label';
    labelWrap.innerHTML = `
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
      </svg>
      <span class="delete-text">Hold to delete</span>
    `;

    deleteBtn.appendChild(fillEl);
    deleteBtn.appendChild(labelWrap);

    let holdTimeout = null;
    let isHolding = false;
    const HOLD_DURATION_MS = 3000;

    const startHold = (e) => {
      if (e.button !== undefined && e.button !== 0) return;
      e.stopPropagation();
      if (isHolding) return;
      isHolding = true;

      deleteBtn.classList.add('holding');
      fillEl.style.transition = `width ${HOLD_DURATION_MS}ms linear`;
      fillEl.style.width = '100%';

      const deleteText = labelWrap.querySelector('.delete-text');
      if (deleteText) deleteText.textContent = 'Deleting...';

      holdTimeout = setTimeout(() => {
        isHolding = false;
        deleteBtn.classList.remove('holding');
        deleteBtn.classList.add('deleted');

        if (typeof spawnClickParticles === 'function') {
          const rect = deleteBtn.getBoundingClientRect();
          spawnClickParticles(rect.left + rect.width / 2, rect.top + rect.height / 2, color);
        }

        ensembleConfigs = ensembleConfigs.filter((_, idx) => idx !== index);
        normalizeSelectedIndex();
        renderEnsembles({ scrollStripToSelection: true });
        saveEnsembles();
      }, HOLD_DURATION_MS);
    };

    const cancelHold = (e) => {
      if (!isHolding) return;
      isHolding = false;
      if (holdTimeout) {
        clearTimeout(holdTimeout);
        holdTimeout = null;
      }
      deleteBtn.classList.remove('holding');
      fillEl.style.transition = 'width 0.22s cubic-bezier(0.25, 1, 0.5, 1)';
      fillEl.style.width = '0%';

      const deleteText = labelWrap.querySelector('.delete-text');
      if (deleteText) deleteText.textContent = 'Hold to delete';
    };

    deleteBtn.addEventListener('pointerdown', startHold);
    deleteBtn.addEventListener('pointerup', cancelHold);
    deleteBtn.addEventListener('pointercancel', cancelHold);
    deleteBtn.addEventListener('pointerleave', cancelHold);
    deleteBtn.addEventListener('contextmenu', (e) => e.preventDefault());

    card.appendChild(tile);
    card.appendChild(deleteBtn);
    categoryStripEl.appendChild(card);
  });
}

function renderModalityPicker(classifierNamesById) {
  categoryPanelEl.innerHTML = '';

  const container = document.createElement('div');
  container.className = 'wizard-container';

  // Top header with Back/Cancel button and step indicator
  const header = document.createElement('div');
  header.className = 'wizard-header';

  const topBar = document.createElement('div');
  topBar.className = 'wizard-top-bar';

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'wizard-cancel-btn';
  cancelBtn.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <line x1="18" y1="6" x2="6" y2="18"></line>
      <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>
    <span>Cancel</span>
  `;
  cancelBtn.addEventListener('click', () => {
    categoryPanelState.mode = CATEGORY_PANEL_MODE.DETAIL;
    categoryPanelState.context = 'edit';
    categoryPanelState.draftModality = null;
    categoryPanelState.targetEnsembleIndex = null;
    if (categoryPanelState.previousIndex >= 0 && categoryPanelState.previousIndex < ensembleConfigs.length) {
      selectedCategoryIndex = categoryPanelState.previousIndex;
    } else {
      normalizeSelectedIndex();
    }
    renderEnsembles();
  });

  topBar.appendChild(cancelBtn);
  header.appendChild(topBar);

  const title = document.createElement('h3');
  title.className = 'wizard-title';
  title.textContent = 'Choose Category Modality';

  const subtitle = document.createElement('p');
  subtitle.className = 'wizard-subtitle';
  subtitle.textContent = 'Select what type of content this category will inspect and classify.';

  header.appendChild(title);
  header.appendChild(subtitle);
  container.appendChild(header);

  // Modality Grid (2x2)
  const grid = document.createElement('div');
  grid.className = 'modality-picker-grid';

  const modalities = [
    {
      key: 'image',
      name: 'Images',
      desc: 'Detect synthetic photos, digital artwork, and imagery'
    },
    {
      key: 'text',
      name: 'Text',
      desc: 'Detect AI-generated articles, reviews, and written content'
    },
    {
      key: 'video',
      name: 'Video',
      desc: 'Detect synthetic video footage, animations, and deepfakes'
    },
    {
      key: 'audio',
      name: 'Audio',
      desc: 'Detect synthetic voice, speech clones, and generated music'
    }
  ];

  modalities.forEach((mod) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'modality-pick-card';

    const count = getCompatibleClassifiers(mod.key).length;
    const countLabel = count === 1 ? '1 model available' : `${count} models available`;
    const iconUrl = getModalityIconSrc(mod.key);

    const iconWrap = document.createElement('div');
    iconWrap.className = 'modality-card-icon';
    const img = document.createElement('img');
    img.src = iconUrl;
    img.alt = mod.name;
    img.className = 'modality-card-img';
    iconWrap.appendChild(img);

    const info = document.createElement('div');
    info.className = 'modality-card-info';
    const titleEl = document.createElement('span');
    titleEl.className = 'modality-card-title';
    titleEl.textContent = mod.name;
    const descEl = document.createElement('span');
    descEl.className = 'modality-card-desc';
    descEl.textContent = mod.desc;
    info.appendChild(titleEl);
    info.appendChild(descEl);

    const countSpan = document.createElement('span');
    countSpan.className = 'modality-card-count';
    countSpan.textContent = countLabel;

    card.appendChild(iconWrap);
    card.appendChild(info);
    card.appendChild(countSpan);

    card.addEventListener('click', () => {
      categoryPanelState.draftModality = mod.key;
      categoryPanelState.mode = CATEGORY_PANEL_MODE.PICK_CLASSIFIER;
      animateCategoryPanelEntrance();
      renderCategoryPanel(selectedCategoryIndex, classifierNamesById);
    });

    grid.appendChild(card);
  });

  container.appendChild(grid);
  categoryPanelEl.appendChild(container);
}

function renderClassifierPicker(classifierNamesById) {
  categoryPanelEl.innerHTML = '';

  const isCreate = categoryPanelState.context === 'create';
  const modality = categoryPanelState.draftModality || 'image';
  const modalityLabel = formatModalityLabel(modality);

  const container = document.createElement('div');
  container.className = 'wizard-container';

  // Header
  const header = document.createElement('div');
  header.className = 'wizard-header';

  const topBar = document.createElement('div');
  topBar.className = 'wizard-top-bar';

  const backBtn = document.createElement('button');
  backBtn.type = 'button';
  backBtn.className = 'wizard-back-btn';
  backBtn.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="15 18 9 12 15 6"></polyline>
    </svg>
    <span>Back</span>
  `;

  if (isCreate) {
    backBtn.addEventListener('click', () => {
      categoryPanelState.mode = CATEGORY_PANEL_MODE.PICK_MODALITY;
      animateCategoryPanelEntrance();
      renderCategoryPanel(selectedCategoryIndex, classifierNamesById);
    });
  } else {
    backBtn.addEventListener('click', () => {
      categoryPanelState.mode = CATEGORY_PANEL_MODE.DETAIL;
      categoryPanelState.context = 'edit';
      categoryPanelState.draftModality = null;
      categoryPanelState.targetEnsembleIndex = null;
      animateCategoryPanelEntrance();
      renderCategoryPanel(selectedCategoryIndex, classifierNamesById);
    });
  }

  topBar.appendChild(backBtn);
  header.appendChild(topBar);

  const titleRow = document.createElement('div');
  titleRow.className = 'wizard-title-row';

  const title = document.createElement('h3');
  title.className = 'wizard-title';

  const targetEnsemble = !isCreate && categoryPanelState.targetEnsembleIndex != null
    ? ensembleConfigs[categoryPanelState.targetEnsembleIndex]
    : null;
  const targetName = targetEnsemble
    ? getCategoryDisplayName(targetEnsemble, categoryPanelState.targetEnsembleIndex, classifierNamesById)
    : '';

  title.textContent = isCreate
    ? 'Select Initial Classifier'
    : `Add Classifier to ${targetName}`;

  titleRow.appendChild(title);
  header.appendChild(titleRow);

  const subtitle = document.createElement('p');
  subtitle.className = 'wizard-subtitle';
  subtitle.textContent = isCreate
    ? `Pick the primary classifier model for your ${modalityLabel.toLowerCase()} category.`
    : `Choose a compatible ${modalityLabel.toLowerCase()} model to add to this ensemble category.`;
  header.appendChild(subtitle);

  container.appendChild(header);

  // List of classifiers
  const list = document.createElement('div');
  list.className = 'classifier-picker-list';

  const compatible = getCompatibleClassifiers(modality);
  const activeClassifierIds = targetEnsemble && Array.isArray(targetEnsemble.classifiers)
    ? targetEnsemble.classifiers
    : [];

  if (compatible.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'classifier-picker-empty';
    empty.innerHTML = `
      <div class="picker-empty-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
      </div>
      <h4>No ${modalityLabel} Classifiers Installed</h4>
      <p>Download ${modalityLabel.toLowerCase()} classifiers from the Classifiers page to create and run categories for this media type.</p>
    `;

    const openHubBtn = document.createElement('button');
    openHubBtn.type = 'button';
    openHubBtn.className = 'btn btn-outline';
    openHubBtn.textContent = 'Download Classifiers';
    openHubBtn.addEventListener('click', () => {
      const msgType = isHostConnected ? 'OPEN_MODELS_HUB' : 'OPEN_ONBOARDING';
      ext.runtime.sendMessage({ type: msgType }).catch(() => {});
    });

    empty.appendChild(openHubBtn);
    if (modality === 'image' || modality === 'text') {
      const forgeBtn = document.createElement('button');
      forgeBtn.type = 'button';
      forgeBtn.className = 'btn btn-outline';
      forgeBtn.style.marginTop = '8px';
      forgeBtn.style.borderColor = 'rgba(52, 211, 153, 0.55)';
      forgeBtn.style.color = '#6ee7b7';
      forgeBtn.textContent = '🔥 Make Your Own Classifier (FvF Bellows)';
      forgeBtn.addEventListener('click', () => {
        ext.runtime.sendMessage({ type: 'OPEN_BELLOWS', modality }).catch(() => {});
      });
      empty.appendChild(forgeBtn);
    }
    list.appendChild(empty);
  } else {
    compatible.forEach((model) => {
      const isAlreadyActive = !isCreate && activeClassifierIds.includes(model.id);

      const card = document.createElement('div');
      card.className = `classifier-pick-card ${isAlreadyActive ? 'is-active' : ''}`;

      const main = document.createElement('div');
      main.className = 'classifier-pick-card-main';

      const nameEl = document.createElement('span');
      nameEl.className = 'classifier-pick-name';
      nameEl.textContent = model.name || model.id;

      const meta = document.createElement('div');
      meta.className = 'classifier-pick-meta';

      const rateEl = document.createElement('span');
      rateEl.className = 'classifier-rate';
      rateEl.textContent = formatClassifierRate(model.id, model.modalities || (model.modality ? [model.modality] : []));

      meta.appendChild(rateEl);
      if (model.description) {
        const descEl = document.createElement('span');
        descEl.className = 'classifier-rate';
        descEl.textContent = `• ${model.description}`;
        meta.appendChild(descEl);
      }

      main.appendChild(nameEl);
      main.appendChild(meta);
      card.appendChild(main);

      const selectBtn = document.createElement('button');
      selectBtn.type = 'button';
      selectBtn.className = `btn-pick-classifier ${isAlreadyActive ? 'is-active' : ''}`;
      selectBtn.disabled = isAlreadyActive;
      selectBtn.textContent = isAlreadyActive
        ? 'Active ✓'
        : (isCreate ? 'Select Classifier' : '+ Add Classifier');

      const handleSelect = async () => {
        if (isAlreadyActive) return;

        if (isCreate) {
          const newEnsemble = createNewEnsemble(modality);
          newEnsemble.classifiers = [model.id];
          const suggestedLabel = typeof model.label === 'string' && model.label.trim()
            ? model.label.trim()
            : (modality === 'text' ? 'AI text' : 'AI generated');
          newEnsemble.name = suggestedLabel;
          newEnsemble.threshold = resolveSuggestedThreshold(model);
          ensembleConfigs.push(newEnsemble);
          selectedCategoryIndex = ensembleConfigs.length - 1;
          await saveEnsembles();
          categoryPanelState.mode = CATEGORY_PANEL_MODE.DETAIL;
          categoryPanelState.context = 'edit';
          categoryPanelState.draftModality = null;
          categoryPanelState.targetEnsembleIndex = null;
          renderEnsembles({ scrollStripToSelection: true });
        } else if (targetEnsemble) {
          const nextClassifiers = [...activeClassifierIds, model.id];
          updateEnsembleClassifiers(targetEnsemble, nextClassifiers);
          await saveEnsembles();
          categoryPanelState.mode = CATEGORY_PANEL_MODE.DETAIL;
          categoryPanelState.context = 'edit';
          categoryPanelState.draftModality = null;
          categoryPanelState.targetEnsembleIndex = null;
          renderEnsembles();
        }
      };

      selectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        handleSelect();
      });

      if (!isAlreadyActive) {
        card.style.cursor = 'pointer';
        card.addEventListener('click', handleSelect);
      }

      card.appendChild(selectBtn);
      list.appendChild(card);
    });
  }

  if (modality === 'image' || modality === 'text') {
    const bellowsCard = document.createElement('div');
    bellowsCard.className = 'classifier-pick-card';
    bellowsCard.style.borderColor = 'rgba(52, 211, 153, 0.45)';
    bellowsCard.style.background = 'rgba(16, 185, 129, 0.08)';
    bellowsCard.style.cursor = 'pointer';
    bellowsCard.innerHTML = `
      <div class="classifier-pick-card-main">
        <span class="classifier-pick-name" style="color: #6ee7b7;">🔥 Make Your Own Classifier (FvF Bellows)</span>
        <div class="classifier-pick-meta">
          <span class="classifier-rate">Train a custom ${modalityLabel.toLowerCase()} classifier locally on your own hardware</span>
        </div>
      </div>
      <button type="button" class="btn-pick-classifier" style="border-color: rgba(52, 211, 153, 0.55); color: #6ee7b7;">Launch Bellows →</button>
    `;
    bellowsCard.addEventListener('click', () => {
      ext.runtime.sendMessage({ type: 'OPEN_BELLOWS', modality }).catch(() => {});
    });
    list.appendChild(bellowsCard);
  }

  container.appendChild(list);
  categoryPanelEl.appendChild(container);
}

function renderCategoryPanel(index, classifierNamesById) {
  if (!categoryPanelEl) return;

  if (categoryPanelState.mode === CATEGORY_PANEL_MODE.PICK_MODALITY) {
    renderModalityPicker(classifierNamesById);
    return;
  }

  if (categoryPanelState.mode === CATEGORY_PANEL_MODE.PICK_CLASSIFIER) {
    renderClassifierPicker(classifierNamesById);
    return;
  }

  const ensemble = ensembleConfigs[index];
  if (!ensemble) {
    categoryPanelEl.innerHTML = '';
    return;
  }

  categoryPanelEl.innerHTML = '';

  const preview = createLivePreview(ensemble);
  categoryPanelEl.appendChild(preview.element);

  const header = document.createElement('div');
  header.className = 'category-detail-header';

  const selectedClassifiers = Array.isArray(ensemble.classifiers) ? ensemble.classifiers : [];
  const suggestedModel = selectedClassifiers.length === 1
    ? availableClassifiers.find((m) => m && m.id === selectedClassifiers[0])
    : null;
  const suggestedLabel = (suggestedModel && typeof suggestedModel.label === 'string' && suggestedModel.label.trim())
    ? suggestedModel.label.trim()
    : getCategoryLabelForSelection(selectedClassifiers, null, index, ensemble.modality || 'image');

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.className = 'input-text';
  nameInput.value = typeof ensemble.name === 'string' && ensemble.name.trim() ? ensemble.name : suggestedLabel;
  nameInput.placeholder = suggestedLabel;
  nameInput.addEventListener('input', () => {
    ensemble.name = nameInput.value.trim();
    scheduleSaveEnsembles();
    renderCategoryStrip();
  });
  header.appendChild(nameInput);

  // Modality display badge (display only, not dropdown)
  const modalityBadge = document.createElement('span');
  modalityBadge.className = 'category-modality-badge';
  modalityBadge.textContent = formatModalityLabel(ensemble.modality);
  header.appendChild(modalityBadge);

  const enabledWrap = document.createElement('div');
  enabledWrap.className = 'toggle-row';
  enabledWrap.style.padding = '0';
  const enabledLabel = document.createElement('span');
  enabledLabel.className = 'toggle-row-text';
  enabledLabel.textContent = 'Enabled';
  const enabledSwitch = createSwitch(ensemble.enabled !== false, (checked) => {
    ensemble.enabled = checked;
    saveEnsembles();
    renderCategoryStrip(classifierNamesById);
  });
  enabledWrap.appendChild(enabledLabel);
  enabledWrap.appendChild(enabledSwitch);
  header.appendChild(enabledWrap);

  categoryPanelEl.appendChild(header);

  const thresholdBlock = document.createElement('div');
  thresholdBlock.className = 'threshold-row';
  const thresholdLabel = document.createElement('div');
  thresholdLabel.className = 'field-label';
  const thresholdValue = document.createElement('span');
  thresholdValue.className = 'field-value-inline';
  thresholdValue.textContent = `${Math.round((ensemble.threshold ?? 0.5) * 100)}%`;
  thresholdLabel.appendChild(document.createTextNode('Detection threshold'));
  thresholdLabel.appendChild(thresholdValue);

  const thresholdInput = document.createElement('input');
  thresholdInput.type = 'range';
  thresholdInput.min = '0';
  thresholdInput.max = '1';
  thresholdInput.step = '0.01';
  thresholdInput.value = typeof ensemble.threshold === 'number' ? ensemble.threshold : 0.5;
  updateSliderBackground(thresholdInput);
  thresholdInput.addEventListener('input', () => {
    ensemble.threshold = parseFloat(thresholdInput.value);
    thresholdValue.textContent = `${Math.round(ensemble.threshold * 100)}%`;
    updateSliderBackground(thresholdInput);
    scheduleSaveEnsembles();
  });

  thresholdBlock.appendChild(thresholdLabel);
  thresholdBlock.appendChild(thresholdInput);
  categoryPanelEl.appendChild(thresholdBlock);

  // Active Classifiers section
  const classifiersLabelRow = document.createElement('div');
  classifiersLabelRow.style.display = 'flex';
  classifiersLabelRow.style.alignItems = 'center';
  classifiersLabelRow.style.justifyContent = 'space-between';
  classifiersLabelRow.style.marginBottom = '6px';

  const classifiersLabel = document.createElement('div');
  classifiersLabel.className = 'section-label';
  classifiersLabel.style.margin = '0';
  classifiersLabel.textContent = 'Classifiers';

  const currentSelected = Array.isArray(ensemble.classifiers) ? ensemble.classifiers : [];
  const countTag = document.createElement('span');
  countTag.className = 'classifier-rate';
  countTag.textContent = `${currentSelected.length} active`;

  classifiersLabelRow.appendChild(classifiersLabel);
  classifiersLabelRow.appendChild(countTag);
  categoryPanelEl.appendChild(classifiersLabelRow);

  const activeList = document.createElement('div');
  activeList.className = 'active-classifier-list';

  if (!currentSelected.length) {
    const empty = document.createElement('div');
    empty.className = 'active-classifier-empty';
    empty.textContent = 'No active classifiers in this category. Add at least one classifier to enable scanning.';
    activeList.appendChild(empty);
  } else {
    currentSelected.forEach((classifierId) => {
      const model = availableClassifiers.find((m) => m.id === classifierId) || { id: classifierId, name: classifierNamesById?.get(classifierId) || classifierId };

      const row = document.createElement('div');
      row.className = 'active-classifier-row';
      row.dataset.classifierId = classifierId;

      const info = document.createElement('div');
      info.className = 'classifier-toggle-info';

      const nameEl = document.createElement('span');
      nameEl.className = 'classifier-toggle-name';
      nameEl.textContent = model.name || model.id || classifierId;

      const rateEl = document.createElement('span');
      rateEl.className = 'classifier-rate';
      rateEl.textContent = formatClassifierRate(classifierId, model.modalities || (model.modality ? [model.modality] : []));

      info.appendChild(nameEl);
      info.appendChild(rateEl);

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'classifier-remove-btn';
      removeBtn.setAttribute('aria-label', `Remove classifier ${nameEl.textContent}`);
      removeBtn.innerHTML = `
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
        <span>Remove</span>
      `;

      removeBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const nextClassifiers = (ensemble.classifiers || []).filter((id) => id !== classifierId);
        updateEnsembleClassifiers(ensemble, nextClassifiers);
        await saveEnsembles();
        renderEnsembles();
      });

      row.appendChild(info);
      row.appendChild(removeBtn);
      activeList.appendChild(row);
    });
  }

  categoryPanelEl.appendChild(activeList);

  const addClassifierBtn = document.createElement('button');
  addClassifierBtn.type = 'button';
  addClassifierBtn.className = 'btn-add-classifier';
  addClassifierBtn.id = 'btn-add-ensemble-classifier';
  addClassifierBtn.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <line x1="12" y1="5" x2="12" y2="19"></line>
      <line x1="5" y1="12" x2="19" y2="12"></line>
    </svg>
    <span>+ Add Classifier</span>
  `;
  addClassifierBtn.addEventListener('click', () => {
    categoryPanelState = {
      mode: CATEGORY_PANEL_MODE.PICK_CLASSIFIER,
      context: 'add_classifier',
      draftModality: ensemble.modality || 'image',
      targetEnsembleIndex: index,
      previousIndex: index
    };
    animateCategoryPanelEntrance();
    renderCategoryPanel(index, classifierNamesById);
  });
  categoryPanelEl.appendChild(addClassifierBtn);

  if (currentSelected.length > 1) {
    categoryPanelEl.appendChild(renderWeightsEditor(ensemble, currentSelected, classifierNamesById));
  }

  const stylesLabel = document.createElement('div');
  stylesLabel.className = 'section-label';
  stylesLabel.textContent = 'Appearance';
  categoryPanelEl.appendChild(stylesLabel);

  const onPreviewUpdate = () => preview.refresh();
  const onColorChange = (updater) => (value) => {
    updater(value);
    scheduleSaveEnsembles();
    preview.refresh();
    renderCategoryStrip(classifierNamesById);
  };

  if ((ensemble.modality || 'image') === 'image') {
    const imageStyles = ensemble.styles?.image || {};

    // 1. Badge mode control is now FIRST
    const badgeMode = imageStyles.badgeMode || (imageStyles.applyBadge === false ? 'off' : 'hover');
    const badgeRow = createIconModeControl('Badge', badgeMode, (value) => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.image = ensemble.styles.image || {};
      ensemble.styles.image.badgeMode = value;
      saveEnsembles();
      onPreviewUpdate();
    });
    categoryPanelEl.appendChild(badgeRow);

    // 2. Blur mode control
    const blurMode = imageStyles.blurMode || (imageStyles.applyBlur === false ? 'off' : 'hover');
    const blurRow = createIconModeControl('Blur', blurMode, (value) => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.image = ensemble.styles.image || {};
      ensemble.styles.image.blurMode = value;
      saveEnsembles();
      onPreviewUpdate();
    });
    const blurRange = document.createElement('input');
    blurRange.type = 'range';
    blurRange.min = '0';
    blurRange.max = '20';
    blurRange.value = imageStyles.blurAmount ?? 3;
    blurRange.addEventListener('input', () => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.image = ensemble.styles.image || {};
      ensemble.styles.image.blurAmount = parseFloat(blurRange.value) || 0;
      updateSliderBackground(blurRange);
      scheduleSaveEnsembles();
      onPreviewUpdate();
    });
    appendSliderToRow(blurRow, blurRange);
    categoryPanelEl.appendChild(blurRow);

    // 3. Border mode control
    const borderMode = imageStyles.borderMode || (imageStyles.applyBorder === false ? 'off' : 'hover');
    const borderRow = createIconModeControl('Border', borderMode, (value) => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.image = ensemble.styles.image || {};
      ensemble.styles.image.borderMode = value;
      saveEnsembles();
      onPreviewUpdate();
    });
    const borderMultiplier = document.createElement('input');
    borderMultiplier.type = 'range';
    borderMultiplier.min = '0';
    borderMultiplier.max = '5';
    borderMultiplier.step = '0.1';
    borderMultiplier.value = imageStyles.borderMultiplier ?? 1;
    borderMultiplier.addEventListener('input', () => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.image = ensemble.styles.image || {};
      ensemble.styles.image.borderMultiplier = parseFloat(borderMultiplier.value) || 1;
      updateSliderBackground(borderMultiplier);
      scheduleSaveEnsembles();
      onPreviewUpdate();
    });
    appendSliderToRow(borderRow, borderMultiplier);
    categoryPanelEl.appendChild(borderRow);

    // 4. Renamed 'Border color' simply to 'Color'
    categoryPanelEl.appendChild(
      createColorSliders(
        'Color',
        imageStyles.borderColor || '#ff0064',
        onColorChange((value) => {
          ensemble.styles = ensemble.styles || {};
          ensemble.styles.image = ensemble.styles.image || {};
          ensemble.styles.image.borderColor = value;
        })
      )
    );
  } else {
    const textStyles = ensemble.styles?.text || {};

    const blurMode = textStyles.blurMode || (textStyles.applyBlur === false ? 'off' : 'hover');
    const blurRow = createIconModeControl('Blur', blurMode, (value) => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.text = ensemble.styles.text || {};
      ensemble.styles.text.blurMode = value;
      saveEnsembles();
      onPreviewUpdate();
    });
    const blurRange = document.createElement('input');
    blurRange.type = 'range';
    blurRange.min = '0';
    blurRange.max = '10';
    blurRange.step = '0.5';
    blurRange.value = textStyles.blurAmount ?? 2;
    blurRange.addEventListener('input', () => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.text = ensemble.styles.text || {};
      ensemble.styles.text.blurAmount = parseFloat(blurRange.value) || 0;
      updateSliderBackground(blurRange);
      scheduleSaveEnsembles();
      onPreviewUpdate();
    });
    appendSliderToRow(blurRow, blurRange);
    categoryPanelEl.appendChild(blurRow);

    const underlineMode = textStyles.underlineMode || (textStyles.applyUnderline === true ? 'hover' : 'off');
    const underlineRow = createIconModeControl('Underline', underlineMode, (value) => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.text = ensemble.styles.text || {};
      ensemble.styles.text.underlineMode = value;
      saveEnsembles();
      onPreviewUpdate();
    });
    categoryPanelEl.appendChild(underlineRow);

    const strikeMode =
      textStyles.strikethroughMode || (textStyles.applyStrikethrough === false ? 'off' : 'hover');
    const strikeRow = createIconModeControl('Strikethrough', strikeMode, (value) => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.text = ensemble.styles.text || {};
      ensemble.styles.text.strikethroughMode = value;
      saveEnsembles();
      onPreviewUpdate();
    });
    categoryPanelEl.appendChild(strikeRow);

    const lineColor = textStyles.strikethroughColor || textStyles.underlineColor || '#ff0064';
    categoryPanelEl.appendChild(
      createColorSliders(
        'Line color',
        lineColor,
        onColorChange((value) => {
          ensemble.styles = ensemble.styles || {};
          ensemble.styles.text = ensemble.styles.text || {};
          ensemble.styles.text.strikethroughColor = value;
          ensemble.styles.text.underlineColor = value;
        })
      )
    );

    const highlightMode = textStyles.highlightMode || (textStyles.applyHighlight === true ? 'hover' : 'off');
    const highlightRow = createIconModeControl('Highlight', highlightMode, (value) => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.text = ensemble.styles.text || {};
      ensemble.styles.text.highlightMode = value;
      saveEnsembles();
      onPreviewUpdate();
    });
    categoryPanelEl.appendChild(highlightRow);

    categoryPanelEl.appendChild(
      createColorSliders(
        'Highlight color',
        textStyles.highlightColor || '#fff3a1',
        onColorChange((value) => {
          ensemble.styles = ensemble.styles || {};
          ensemble.styles.text = ensemble.styles.text || {};
          ensemble.styles.text.highlightColor = value;
        })
      )
    );
  }


}

function renderEnsembles(options = {}) {
  const { scrollStripToSelection = false } = options;
  const scrollY = getDocumentScrollY();

  const classifierNamesById = new Map();
  const classifierLabelsById = new Map();
  availableClassifiers.forEach((m) => {
    classifierNamesById.set(m.id, m.name || m.id);
    if (typeof m.label === 'string' && m.label.trim()) {
      classifierLabelsById.set(m.id, m.label.trim());
    }
  });

  let ensemblesDirty = false;
  ensembleConfigs.forEach((ensemble, index) => {
    if (!ensemble.id) {
      ensemble.id = generateEnsembleId(ensemble.modality || 'image');
      ensemblesDirty = true;
    }
    const selected = Array.isArray(ensemble.classifiers) ? ensemble.classifiers : [];
    const curName = typeof ensemble.name === 'string' ? ensemble.name.trim() : '';
    if (selected.length === 1) {
      const cid = selected[0];
      const suggestedLabel = classifierLabelsById.get(cid);
      const rawModelName = classifierNamesById.get(cid);
      const isLegacyModelName = !curName
        || curName === cid
        || (rawModelName && curName === rawModelName)
        || /^(fvf[\s_-]+)?(heimdall|m[ií]mir|huginn|bragi)\b/i.test(curName)
        || /^category\s+\d+$/i.test(curName);
      if (isLegacyModelName) {
        const nextLabel = suggestedLabel || (/^res_net50_spider$/i.test(cid) ? 'Spider' : ((ensemble.modality || 'image') === 'text' ? 'AI text' : 'AI generated'));
        if (curName !== nextLabel) {
          ensemble.name = nextLabel;
          ensemblesDirty = true;
        }
      }
    } else if (!curName) {
      ensemble.name = getDefaultEnsembleName(index);
      ensemblesDirty = true;
    }
  });

  if (ensemblesDirty) {
    scheduleSaveEnsembles();
  }

  normalizeSelectedIndex();

  const hasCategories = ensembleConfigs.length > 0;
  const isWizardMode = categoryPanelState.mode !== CATEGORY_PANEL_MODE.DETAIL;

  if (categoryStripWrapEl) {
    categoryStripWrapEl.hidden = !hasCategories;
  }
  if (categoryPanelEl) {
    categoryPanelEl.hidden = !hasCategories && !isWizardMode;
  }

  if (hasCategories) {
    renderCategoryStrip(classifierNamesById);
  } else if (categoryStripEl) {
    categoryStripEl.innerHTML = '';
  }

  if (hasCategories || isWizardMode) {
    renderCategoryPanel(selectedCategoryIndex, classifierNamesById);

    if (scrollStripToSelection && hasCategories) {
      scrollCategoryStripToSelected();
      animateCategoryPanelEntrance();
    }
  } else if (categoryPanelEl) {
    categoryPanelEl.innerHTML = '';
  }

  requestAnimationFrame(() => {
    stabilizePopupWidth();
    restoreDocumentScrollY(scrollY);
  });
}

function resolveSuggestedThreshold(model) {
  const raw = model && (
    typeof model.suggested_threshold === 'number'
      ? model.suggested_threshold
      : (typeof model.default_threshold === 'number' ? model.default_threshold : null)
  );
  if (typeof raw === 'number' && Number.isFinite(raw) && raw >= 0 && raw <= 1) {
    return Math.round(raw * 100) / 100;
  }
  return 0.75;
}

function createNewEnsemble(modality = 'image') {
  const id = generateEnsembleId(modality);
  if (modality === 'image') {
    return {
      id,
      modality,
      enabled: true,
      threshold: 0.75,
      classifiers: [],
      weights: null,
      styles: {
        image: {
          applyBlur: true,
          blurAmount: 3,
          applyBorder: false,
          borderMultiplier: 1,
          borderColor: '#7C4DFF',
          applyBadge: true,
          blurMode: 'hover',
          borderMode: 'off',
          badgeMode: 'always'
        }
      }
    };
  }

  if (modality === 'video') {
    return {
      id,
      modality,
      enabled: true,
      threshold: 0.75,
      classifiers: [],
      weights: null,
      styles: {
        video: {
          applyBlur: true,
          blurAmount: 3,
          applyBorder: false,
          borderMultiplier: 1,
          borderColor: '#7C4DFF',
          applyBadge: true,
          blurMode: 'hover',
          borderMode: 'off',
          badgeMode: 'always'
        }
      }
    };
  }

  if (modality === 'audio') {
    return {
      id,
      modality,
      enabled: true,
      threshold: 0.75,
      classifiers: [],
      weights: null,
      styles: {
        audio: {
          applyBadge: true,
          badgeMode: 'always',
          badgeColor: '#7C4DFF'
        }
      }
    };
  }

  return {
    id,
    modality,
    enabled: true,
    threshold: 0.75,
    classifiers: [],
    weights: null,
    styles: {
      text: {
        applyBlur: true,
        blurAmount: 2,
        applyStrikethrough: true,
        strikethroughColor: '#7C4DFF',
        applyUnderline: false,
        underlineColor: '#7C4DFF',
        applyHighlight: false,
        highlightColor: '#C9B6FF',
        blurMode: 'hover',
        strikethroughMode: 'hover',
        underlineMode: 'off',
        highlightMode: 'off'
      }
    }
  };
}

const UI_SNAPSHOT_CACHE_KEY = 'fvf_options_snapshot_v1';

function readUiSnapshotCache() {
  try {
    const raw = window.localStorage && window.localStorage.getItem(UI_SNAPSHOT_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch (_) {
    return null;
  }
}

function writeUiSnapshotCache() {
  try {
    if (!window.localStorage) return;
    const snapshot = {
      enabled: Boolean(enabledInput?.checked),
      altTextOnly: Boolean(altTextOnlyInput?.checked),
      ignoreSmallImages: Boolean(ignoreSmallImagesInput?.checked),
      investigator: Boolean(investigatorInput?.checked),
      minImageDimension: parseInt(minImageDimensionInput?.value || '80', 10) || 80,
      pauseOnBattery: Boolean(pauseOnBatteryInput?.checked),
      pauseBelowThreshold: Boolean(pauseBelowThresholdInput?.checked),
      batteryThreshold: String(batteryThresholdSelect?.value || '20'),
      classificationDelay: parseInt(classificationDelayInput?.value || '100', 10) || 100,
      imageCaptureQuality: parseFloat(imageCaptureQualityInput?.value || '1') || 1,
      verboseLogs: Boolean(verboseLogsInput?.checked),
      lazyLoad: Boolean(lazyLoadInput?.checked),
      disableUpdateNotifications: Boolean(disableUpdateNotifications),
      enableManualBlurContextMenu: Boolean(enableManualBlurContextMenuInput?.checked),
      minTextLength: parseInt(minTextLengthInput?.value || '250', 10) || 250,
      ensembleConfigsV2: ensembleConfigs,
      classifierTimings,
      availableClassifiers,
      latestUpdateStatus
    };
    window.localStorage.setItem(UI_SNAPSHOT_CACHE_KEY, JSON.stringify(snapshot));
  } catch (_) {}
}

function applySettingsToUI(res, { render = true } = {}) {
  if (!res || typeof res !== 'object') return false;

  if (typeof res.enabled === 'boolean') enabledInput.checked = res.enabled;
  if (typeof res.altTextOnly === 'boolean') {
    altTextOnlyInput.checked = res.altTextOnly;
  } else if (!settingsLoaded) {
    altTextOnlyInput.checked = true;
  }
  if (ignoreSmallImagesInput) {
    if (typeof res.ignoreSmallImages === 'boolean') {
      ignoreSmallImagesInput.checked = res.ignoreSmallImages;
    } else if (!settingsLoaded) {
      ignoreSmallImagesInput.checked = true;
    }
  }
  if (investigatorInput) {
    if (typeof res.investigator === 'boolean') {
      investigatorInput.checked = res.investigator;
    } else if (!settingsLoaded) {
      investigatorInput.checked = true;
    }
  }
  if (minImageDimensionInput) {
    if (typeof res.minImageDimension === 'number') {
      minImageDimensionInput.value = res.minImageDimension;
    }
    if (ignoreSmallImagesInput) {
      minImageDimensionInput.disabled = !ignoreSmallImagesInput.checked;
    }
  }
  if (pauseOnBatteryInput && typeof res.pauseOnBattery === 'boolean') {
    pauseOnBatteryInput.checked = res.pauseOnBattery;
  }
  if (pauseBelowThresholdInput && typeof res.pauseBelowThreshold === 'boolean') {
    pauseBelowThresholdInput.checked = res.pauseBelowThreshold;
  }
  if (batteryThresholdSelect && (typeof res.batteryThreshold === 'number' || typeof res.batteryThreshold === 'string')) {
    batteryThresholdSelect.value = String(res.batteryThreshold);
  }
  if (typeof res.classificationDelay === 'number') {
    classificationDelayInput.value = res.classificationDelay;
  }
  if (typeof res.imageCaptureQuality === 'number') {
    imageCaptureQualityInput.value = res.imageCaptureQuality;
  } else if (imageCaptureQualityInput && !settingsLoaded) {
    imageCaptureQualityInput.value = '1';
  }
  if (typeof res.verboseLogs === 'boolean') verboseLogsInput.checked = res.verboseLogs;
  if (typeof res.lazyLoad === 'boolean') lazyLoadInput.checked = res.lazyLoad;
  if (typeof res.disableUpdateNotifications === 'boolean') {
    disableUpdateNotifications = res.disableUpdateNotifications;
  }
  if (disableUpdateNotificationsInput) {
    disableUpdateNotificationsInput.checked = disableUpdateNotifications;
  }
  if (enableManualBlurContextMenuInput) {
    enableManualBlurContextMenuInput.checked = Boolean(res.enableManualBlurContextMenu);
  }
  if (typeof res.minTextLength === 'number') minTextLengthInput.value = res.minTextLength;

  if (Array.isArray(res.availableClassifiers) && res.availableClassifiers.length) {
    availableClassifiers = res.availableClassifiers;
  }
  if (res.classifierTimings && typeof res.classifierTimings === 'object') {
    classifierTimings = res.classifierTimings;
  }
  if (res.latestUpdateStatus && !disableUpdateNotifications) {
    latestUpdateStatus = res.latestUpdateStatus;
    renderUpdateBanner(latestUpdateStatus);
  } else if (disableUpdateNotifications) {
    renderUpdateBanner(null);
  }

  let ensemblesChanged = false;
  if (Array.isArray(res.ensembleConfigsV2)) {
    const prevStr = JSON.stringify(ensembleConfigs);
    const nextStr = JSON.stringify(res.ensembleConfigsV2);
    if (prevStr !== nextStr) {
      ensembleConfigs = res.ensembleConfigsV2;
      ensemblesChanged = true;
    }
  }

  updateDisplays();
  if (render) {
    renderEnsembles();
  }
  return ensemblesChanged;
}

let hasHydratedFromSnapshot = false;
const initialSnapshot = readUiSnapshotCache();
if (initialSnapshot) {
  applySettingsToUI(initialSnapshot, { render: true });
  hasHydratedFromSnapshot = true;
}

async function load() {
  try {
    const prevUiSignature = JSON.stringify({ ensembleConfigs, availableClassifiers });
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    const keys = [
      'enabled',
      'altTextOnly',
      'ignoreSmallImages',
      'investigator',
      'minImageDimension',
      'pauseOnBattery',
      'pauseBelowThreshold',
      'batteryThreshold',
      'classificationDelay',
      'imageCaptureQuality',
      'verboseLogs',
      'lazyLoad',
      'disableUpdateNotifications',
      'enableManualBlurContextMenu',
      'minTextLength',
      'classifierTimings',
      'ensembleConfigsV2'
    ];
    let res = storageArea && storageArea.get ? await storageArea.get(keys) : {};
    if (!res) res = {};

    if (!Array.isArray(res.ensembleConfigsV2) || !res.ensembleConfigsV2.length) {
      if (ext.storage && ext.storage.local && ext.storage.local !== storageArea) {
        try {
          const localRes = await ext.storage.local.get(keys);
          if (Array.isArray(localRes?.ensembleConfigsV2) && localRes.ensembleConfigsV2.length) {
            res = { ...localRes, ...res, ensembleConfigsV2: localRes.ensembleConfigsV2 };
          }
        } catch (_) {}
      }
    }

    applySettingsToUI(res || {}, { render: false });
    settingsLoaded = true;

    if (!res || !Array.isArray(res.ensembleConfigsV2) || res.ensembleConfigsV2.length === 0) {
      if (initialSnapshot && Array.isArray(initialSnapshot.ensembleConfigsV2) && initialSnapshot.ensembleConfigsV2.length) {
        ensembleConfigs = initialSnapshot.ensembleConfigsV2;
        await saveEnsembles();
      } else if (!res || !Array.isArray(res.ensembleConfigsV2)) {
        ensembleConfigs = ENS.buildDefaultEnsembles();
      }
    }

    const nextUiSignature = JSON.stringify({ ensembleConfigs, availableClassifiers });
    if (!hasHydratedFromSnapshot || prevUiSignature !== nextUiSignature) {
      renderEnsembles();
    }
    writeUiSnapshotCache();
  } catch {
    // Ignore load errors for missing storage values.
  }
}

async function saveSetting(key, value) {
  try {
    const payload = { [key]: value };
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    if (storageArea && storageArea.set) {
      await storageArea.set(payload);
    }
    if (ext.storage && ext.storage.local && ext.storage.local !== storageArea && ext.storage.local.set) {
      await ext.storage.local.set(payload).catch(() => {});
    }
  } catch (e) {
    console.error(`Failed to save ${key}:`, e);
  }
}

const pendingSettingSaves = new Map();

function scheduleSaveSetting(key, value) {
  pendingSettingSaves.set(key, value);
  if (scheduleSaveSetting.timer) clearTimeout(scheduleSaveSetting.timer);
  scheduleSaveSetting.timer = setTimeout(async () => {
    scheduleSaveSetting.timer = null;
    const entries = Array.from(pendingSettingSaves.entries());
    pendingSettingSaves.clear();
    try {
      const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
      if (storageArea && storageArea.set && entries.length) {
        await storageArea.set(Object.fromEntries(entries));
      }
    } catch (e) {
      console.error('Failed to save settings batch:', e);
    }
  }, 100);
}

function flushSaveSetting() {
  if (!scheduleSaveSetting.timer && pendingSettingSaves.size === 0) return Promise.resolve();
  if (scheduleSaveSetting.timer) {
    clearTimeout(scheduleSaveSetting.timer);
    scheduleSaveSetting.timer = null;
  }
  const entries = Array.from(pendingSettingSaves.entries());
  pendingSettingSaves.clear();
  if (!entries.length) return Promise.resolve();
  const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
  if (!storageArea || !storageArea.set) return Promise.resolve();
  return storageArea.set(Object.fromEntries(entries)).catch((e) => {
    console.error('Failed to flush settings batch:', e);
  });
}

document.addEventListener('pointerup', () => {
  flushSaveSetting();
}, true);

classificationDelayInput.addEventListener('input', () => {
  updateDisplays();
  const value = Math.max(0, Math.min(500, parseInt(classificationDelayInput.value, 10) || 100));
  updateSliderBackground(classificationDelayInput);
  scheduleSaveSetting('classificationDelay', value);
});

imageCaptureQualityInput.addEventListener('input', () => {
  updateDisplays();
  const value = Math.max(0.2, Math.min(1, parseFloat(imageCaptureQualityInput.value) || 1));
  updateSliderBackground(imageCaptureQualityInput);
  scheduleSaveSetting('imageCaptureQuality', value);
});

enabledInput.addEventListener('change', async () => {
  await saveSetting('enabled', enabledInput.checked);
});

altTextOnlyInput.addEventListener('change', async () => {
  await saveSetting('altTextOnly', altTextOnlyInput.checked);
});

if (ignoreSmallImagesInput) {
  ignoreSmallImagesInput.addEventListener('change', async () => {
    if (minImageDimensionInput) {
      minImageDimensionInput.disabled = !ignoreSmallImagesInput.checked;
    }
    await saveSetting('ignoreSmallImages', ignoreSmallImagesInput.checked);
  });
}

if (investigatorInput) {
  investigatorInput.addEventListener('change', async () => {
    await saveSetting('investigator', investigatorInput.checked);
  });
}

if (minImageDimensionInput) {
  minImageDimensionInput.addEventListener('change', async () => {
    const value = Math.max(16, Math.min(512, parseInt(minImageDimensionInput.value, 10) || 80));
    minImageDimensionInput.value = value;
    await saveSetting('minImageDimension', value);
  });
}

if (pauseOnBatteryInput) {
  pauseOnBatteryInput.addEventListener('change', async () => {
    await saveSetting('pauseOnBattery', pauseOnBatteryInput.checked);
    await refreshPowerStatus();
  });
}

if (pauseBelowThresholdInput) {
  pauseBelowThresholdInput.addEventListener('change', async () => {
    await saveSetting('pauseBelowThreshold', pauseBelowThresholdInput.checked);
    await refreshPowerStatus();
  });
}

if (batteryThresholdSelect) {
  batteryThresholdSelect.addEventListener('change', async () => {
    await saveSetting('batteryThreshold', parseInt(batteryThresholdSelect.value, 10));
    await refreshPowerStatus();
  });
}

verboseLogsInput.addEventListener('change', async () => {
  await saveSetting('verboseLogs', verboseLogsInput.checked);
});

lazyLoadInput.addEventListener('change', async () => {
  await saveSetting('lazyLoad', lazyLoadInput.checked);
});

if (disableUpdateNotificationsInput) {
  disableUpdateNotificationsInput.addEventListener('change', async () => {
    disableUpdateNotifications = disableUpdateNotificationsInput.checked;
    await saveSetting('disableUpdateNotifications', disableUpdateNotifications);
    if (disableUpdateNotifications) {
      latestUpdateStatus = null;
      renderUpdateBanner(null);
    } else {
      fetchModels({ forceUpdateCheck: false }).catch(() => {});
    }
  });
}

if (enableManualBlurContextMenuInput) {
  enableManualBlurContextMenuInput.addEventListener('change', async () => {
    await saveSetting('enableManualBlurContextMenu', enableManualBlurContextMenuInput.checked);
    writeUiSnapshotCache();
  });
}

minTextLengthInput.addEventListener('change', async () => {
  const value = Math.max(50, Math.min(2000, parseInt(minTextLengthInput.value, 10) || 250));
  minTextLengthInput.value = value;
  await saveSetting('minTextLength', value);
});

initCollapsibleAnimations();
initInteractionParticles();
load().then(() => {
  requestAnimationFrame(stabilizePopupWidth);
  fetchModels().catch(() => {});
});
refreshRunState();
runStateTimer = setInterval(refreshRunState, 800);

if (ext && ext.storage && ext.storage.onChanged) {
  ext.storage.onChanged.addListener((changes) => {
    if (Object.prototype.hasOwnProperty.call(changes, 'classifierTimings')) {
      classifierTimings = changes.classifierTimings.newValue || {};
      updateClassifierRateLabels();
    }
    if (Object.prototype.hasOwnProperty.call(changes, 'ensembleConfigsV2')) {
      const nextEnsembles = changes.ensembleConfigsV2.newValue;
      if (Array.isArray(nextEnsembles)) {
        const prevStr = JSON.stringify(ensembleConfigs);
        const nextStr = JSON.stringify(nextEnsembles);
        if (prevStr !== nextStr) {
          ensembleConfigs = nextEnsembles;
          normalizeSelectedIndex();
          renderEnsembles();
          writeUiSnapshotCache();
        }
      }
    }
  });
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    refreshRunState();
  }
});

window.addEventListener('beforeunload', () => {
  if (runStateTimer) {
    clearInterval(runStateTimer);
    runStateTimer = null;
  }
});

if (addEnsembleBtn) {
  addEnsembleBtn.addEventListener('click', () => {
    categoryPanelState = {
      mode: CATEGORY_PANEL_MODE.PICK_MODALITY,
      context: 'create',
      draftModality: null,
      targetEnsembleIndex: null,
      previousIndex: selectedCategoryIndex
    };
    if (categoryPanelEl) {
      categoryPanelEl.hidden = false;
    }
    renderEnsembles();
    if (categoryPanelEl) {
      categoryPanelEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  });
}

const reclassifyBtn = document.getElementById('reclassify-all');
if (reclassifyBtn) {
  reclassifyBtn.addEventListener('click', async () => {
    try {
      reclassifyBtn.disabled = true;
      reclassifyBtn.textContent = 'Reclassifying…';

      const tabs = await ext.tabs.query({});
      await Promise.all(
        tabs.map((tab) =>
          ext.tabs.sendMessage(tab.id, { type: 'RECLASSIFY_ALL' }).catch(() => {})
        )
      );

      reclassifyBtn.textContent = 'Reclassified!';
      setTimeout(() => {
        reclassifyBtn.disabled = false;
        reclassifyBtn.textContent = 'Reclassify all pages';
      }, 2000);
    } catch (err) {
      console.error('Reclassify error:', err);
      reclassifyBtn.disabled = false;
      reclassifyBtn.textContent = 'Reclassify all pages';
    }
  });
}

const cancelAllBtn = document.getElementById('cancel-all');
if (cancelAllBtn) {
  cancelAllBtn.addEventListener('click', async () => {
    try {
      cancelAllBtn.disabled = true;
      cancelAllBtn.textContent = 'Cancelling…';

      await ext.runtime.sendMessage({ type: 'CANCEL_ALL' });

      const tabs = await ext.tabs.query({});
      await Promise.all(
        tabs.map((tab) =>
          ext.tabs.sendMessage(tab.id, { type: 'CANCEL_ALL_CLASSIFICATIONS' }).catch(() => {})
        )
      );

      cancelAllBtn.textContent = 'Cancelled!';
      setTimeout(() => {
        cancelAllBtn.disabled = false;
        cancelAllBtn.textContent = 'Cancel all classifications';
      }, 2000);
    } catch (err) {
      console.error('Cancel error:', err);
      cancelAllBtn.disabled = false;
      cancelAllBtn.textContent = 'Cancel all classifications';
    }
  });
}

function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function setModelStatus(text, type = 'info') {
  if (!modelStatusEl) return;
  modelStatusEl.textContent = text;
  modelStatusEl.hidden = !text;
  modelStatusEl.style.color = type === 'error' ? '#c04a4a' : '';
}

let updateStatusHideTimer = null;

function renderUpdateBanner(updates) {
  if (!updateBannerWrapEl || !hostUpdateBannerBtn || !classifierUpdateBannerBtn) return;

  const hasTransientStatus = Boolean(updateBannerStatusEl && !updateBannerStatusEl.hidden);

  if (disableUpdateNotifications || !updates || !updates.anyUpdateAvailable) {
    hostUpdateBannerBtn.hidden = true;
    classifierUpdateBannerBtn.hidden = true;
    if (!hasTransientStatus) {
      updateBannerWrapEl.hidden = true;
    }
    return;
  }

  let anyBtnVisible = false;
  const hostInfo = updates.host || {};
  if (hostInfo.updateAvailable) {
    const relTag = hostInfo.releaseTag || '';
    const diffCount = Number(hostInfo.diffCount) || 0;
    hostUpdateBannerBtn.textContent = relTag
      ? `↑ Update Native Host (${relTag})`
      : '↑ Update Native Host';
    const diffSummary = diffCount > 0 ? `${diffCount} file(s) differ from GitHub release` : '';
    hostUpdateBannerBtn.title = [diffSummary, hostInfo.releaseNotes].filter(Boolean).join(' — ') || 'Update local Native Host components now';
    hostUpdateBannerBtn.disabled = false;
    hostUpdateBannerBtn.hidden = false;
    anyBtnVisible = true;
  } else {
    hostUpdateBannerBtn.hidden = true;
  }

  const clfInfo = updates.classifiers || {};
  const outdated = Array.isArray(clfInfo.outdated) ? clfInfo.outdated : [];
  if (clfInfo.updateAvailable && outdated.length > 0) {
    if (outdated.length === 1) {
      const item = outdated[0];
      const prefix = item.reason === 'newer_generation' ? '↑ Newer Classifier' : '↑ Update Classifier';
      classifierUpdateBannerBtn.textContent = `${prefix}: ${item.latestName || item.latestId}`;
    } else {
      classifierUpdateBannerBtn.textContent = `↑ ${outdated.length} Classifier Updates Available`;
    }
    classifierUpdateBannerBtn.title = outdated
      .map((item) => (
        item.reason === 'release_updated'
          ? `${item.installedName || item.installedId}: updated release weights available`
          : `${item.installedName || item.installedId} → ${item.latestName || item.latestId}`
      ))
      .join('\n');
    classifierUpdateBannerBtn.hidden = false;
    anyBtnVisible = true;
  } else {
    classifierUpdateBannerBtn.hidden = true;
  }

  updateBannerWrapEl.hidden = !(anyBtnVisible || hasTransientStatus);
}

async function fetchModels({ forceUpdateCheck = false } = {}) {
  setModelStatus('Loading models…');
  try {
    const resp = await ext.runtime.sendMessage({
      type: 'LIST_MODELS',
      checkUpdates: !disableUpdateNotifications,
      forceUpdateCheck
    });
    if (!resp || resp.ok !== true) throw new Error(resp?.error || 'Failed to list models');
    const models = Array.isArray(resp.models) ? resp.models : [];
    const rawClassifiers = resp.raw && Array.isArray(resp.raw.classifiers) ? resp.raw.classifiers : [];
    const nextClassifiers = rawClassifiers.length
      ? rawClassifiers
      : models.map((id) => ({ id, name: id, label: 'AI generated', modalities: [] }));
    const classifiersChanged = JSON.stringify(availableClassifiers) !== JSON.stringify(nextClassifiers);
    availableClassifiers = nextClassifiers;
    if (classifiersChanged || (categoryPanelEl && !categoryPanelEl.hidden && categoryPanelEl.childElementCount === 0)) {
      renderEnsembles();
    }
    setModelStatus(models.length ? `${models.length} model(s) loaded` : 'No models found');

    if (!disableUpdateNotifications && resp.updates) {
      latestUpdateStatus = resp.updates;
      renderUpdateBanner(latestUpdateStatus);
    } else if (disableUpdateNotifications) {
      renderUpdateBanner(null);
    }
    writeUiSnapshotCache();
  } catch (err) {
    console.warn('LIST_MODELS failed', err);
    if (!availableClassifiers.length) {
      renderEnsembles();
    }
    setModelStatus(err.message || 'Failed to list models', 'error');
  }
}

if (refreshBtn) {
  refreshBtn.addEventListener('click', () => fetchModels({ forceUpdateCheck: !disableUpdateNotifications }));
  const openOnboardingBtn = document.getElementById('open-onboarding');
  if (openOnboardingBtn) {
    openOnboardingBtn.addEventListener('click', () => {
      const msgType = isHostConnected ? 'OPEN_MODELS_HUB' : 'OPEN_ONBOARDING';
      ext.runtime.sendMessage({ type: msgType }).catch(() => {});
    });
  }
}

window.addEventListener('focus', () => {
  if (settingsLoaded) {
    fetchModels().catch(() => {});
    refreshBellowsBanner().catch(() => {});
  }
});

const bellowsBannerEl = document.getElementById('bellows-active-banner');
const bellowsReopenBtn = document.getElementById('bellows-reopen-btn');

async function refreshBellowsBanner() {
  if (!bellowsBannerEl || !bellowsReopenBtn) return;
  try {
    const res = await ext.runtime.sendMessage({ type: 'BELLOWS_GET_STATUS' });
    const st = res && res.status;
    if (st && (res.active || st.state === 'running' || st.state === 'exporting' || st.state === 'initializing' || st.state === 'downloading_backbone')) {
      const pct = Math.round(Number(st.progress_pct || 0));
      bellowsReopenBtn.textContent = `🔥 FvF Bellows Training: ${st.model_name || 'Custom'} (${pct}%) — Click to Reopen`;
      bellowsReopenBtn.onclick = () => ext.runtime.sendMessage({ type: 'OPEN_BELLOWS', resume: true, jobId: st.job_id }).catch(() => {});
      bellowsBannerEl.hidden = false;
      return;
    }
    bellowsBannerEl.hidden = true;
  } catch (_) {
    bellowsBannerEl.hidden = true;
  }
}

setInterval(() => {
  refreshBellowsBanner().catch(() => {});
}, 6000);
refreshBellowsBanner().catch(() => {});

// Initialize accent sliders on load
[classificationDelayInput, imageCaptureQualityInput].forEach((el) => {
  if (el) updateSliderBackground(el);
});


// ============================================================================
// FLOATING EMBERS ENGINE (BACKGROUND EFFECT)
// ============================================================================

function initFloatingEmbers() {
  if (prefersReducedMotion) return;
  const canvas = document.getElementById('ember-webgl-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const isPopup = (window.innerWidth || 0) < 680;
  let width = (canvas.width = Math.max(window.innerWidth || 0, document.documentElement.clientWidth || 0, isPopup ? 420 : 680));
  let height = (canvas.height = Math.max(window.innerHeight || 0, document.documentElement.clientHeight || 0, 580));

  function spawnEmberParticle(p, initial = false) {
    const target = p || {};
    target.maxLife = 220 + Math.floor(Math.random() * 240);
    target.vx = 0.2 + Math.random() * 0.6;
    target.vy = -0.3 - Math.random() * 0.6;
    target.r = 1.2 + Math.random() * 2.5;
    target.baseAlpha = 0.18 + Math.random() * 0.55;
    target.hue = 14 + Math.random() * 32;

    if (initial) {
      target.x = Math.random() * width;
      target.y = Math.random() * height;
      target.age = Math.floor((0.12 + Math.random() * 0.76) * target.maxLife);
    } else {
      if (Math.random() < 0.65) {
        target.x = -10 + Math.random() * width;
        target.y = height + 4 + Math.random() * 14;
      } else {
        target.x = -4 - Math.random() * 14;
        target.y = Math.random() * (height + 10);
      }
      target.age = 0;
    }
    return target;
  }

  const embers = Array.from({ length: 48 }, () => spawnEmberParticle({}, true));

  window.addEventListener('resize', () => {
    const newWidth = Math.max(window.innerWidth || 0, document.documentElement.clientWidth || 0, 1);
    const newHeight = Math.max(window.innerHeight || 0, document.documentElement.clientHeight || 0, 1);
    if (width > 0 && height > 0 && (newWidth !== width || newHeight !== height)) {
      const scaleX = newWidth / width;
      const scaleY = newHeight / height;
      for (const p of embers) {
        p.x *= scaleX;
        p.y *= scaleY;
      }
    }
    width = canvas.width = newWidth;
    height = canvas.height = newHeight;
  });

  function drawEmbersFrame(advance = true) {
    ctx.clearRect(0, 0, width, height);
    for (const p of embers) {
      if (advance) {
        p.x += p.vx;
        p.y += p.vy;
        p.age += 1;
        if (p.age >= p.maxLife || p.x > width + 20 || p.y < -20) {
          spawnEmberParticle(p, false);
        }
      }

      const lifeRatio = p.age / Math.max(1, p.maxLife);
      let lifeFade = 1;
      if (lifeRatio < 0.12) {
        lifeFade = lifeRatio / 0.12;
      } else if (lifeRatio > 0.82) {
        lifeFade = Math.max(0, (1 - lifeRatio) / 0.18);
      }
      const alpha = p.baseAlpha * lifeFade;
      if (alpha <= 0.01) continue;

      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4.2);
      grad.addColorStop(0, `hsla(${p.hue}, 100%, 72%, ${alpha})`);
      grad.addColorStop(0.45, `hsla(${p.hue - 6}, 100%, 54%, ${alpha * 0.45})`);
      grad.addColorStop(1, 'hsla(10, 100%, 40%, 0)');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 4.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  let rafId = null;
  function renderLoop() {
    rafId = null;
    if (document.hidden) return;
    drawEmbersFrame(true);
    rafId = requestAnimationFrame(renderLoop);
  }

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && rafId === null) {
      rafId = requestAnimationFrame(renderLoop);
    }
  });

  drawEmbersFrame(false);
  rafId = requestAnimationFrame(renderLoop);
}

// ============================================================================
// NAVIGATION BUTTONS & NATIVE HOST UPDATE SYSTEM
// ============================================================================

function initNavButtonsAndHostUpdater() {
  // 1. Navigation Buttons
  const fullPageBtn = document.getElementById('open-fullpage-btn');
  if (fullPageBtn) {
    fullPageBtn.addEventListener('click', () => {
      ext.runtime.sendMessage({ type: 'OPEN_OPTIONS_FULLPAGE' }).catch(() => {
        if (ext.tabs && ext.tabs.create) {
          ext.tabs.create({ url: ext.runtime.getURL('options/options.html') });
        }
      });
    });
  }

  const modelsBtn = document.getElementById('open-models-btn');
  if (modelsBtn) {
    modelsBtn.addEventListener('click', () => {
      ext.runtime.sendMessage({ type: 'OPEN_MODELS_HUB' }).catch(() => {
        if (ext.tabs && ext.tabs.create) {
          ext.tabs.create({ url: ext.runtime.getURL('docs/index.html?screen=2&manage=1') });
        }
      });
    });
  }

  const guideBtn = document.getElementById('open-guide-btn');
  if (guideBtn) {
    guideBtn.addEventListener('click', () => {
      ext.runtime.sendMessage({ type: 'OPEN_GUIDE_PAGE' }).catch(() => {
        if (ext.tabs && ext.tabs.create) {
          ext.tabs.create({ url: ext.runtime.getURL('docs/index.html?screen=3') });
        }
      });
    });
  }

  // 2. Native Host Status & Updates
  const badgeEl = document.getElementById('host-status-badge');
  const detailEl = document.getElementById('host-detail-text');
  const checkUpdateBtn = document.getElementById('check-host-update-btn');
  const updateStatusEl = document.getElementById('host-update-status');

  async function refreshHostStatus() {
    try {
      const resp = await ext.runtime.sendMessage({ type: 'CHECK_NATIVE_HOST' });
      isHostConnected = Boolean(resp && resp.connected);
      if (badgeEl) {
        if (isHostConnected) {
          badgeEl.className = 'host-status-badge connected';
          badgeEl.textContent = '● Connected';
          if (detailEl) detailEl.textContent = 'Local hardware-accelerated AI host is active';
        } else {
          badgeEl.className = 'host-status-badge disconnected';
          badgeEl.textContent = '○ Not Connected';
          if (detailEl) detailEl.textContent = 'Run Install-FireVsFire-NativeHost.bat to connect';
        }
      }
    } catch (_) {
      isHostConnected = false;
      if (badgeEl) {
        badgeEl.className = 'host-status-badge disconnected';
        badgeEl.textContent = '○ Disconnected';
      }
    }
    updateHeaderStatus();
    await refreshPowerStatus();
  }

  const reconnectBtn = document.getElementById('reconnect-host-btn');
  if (reconnectBtn) {
    reconnectBtn.addEventListener('click', async () => {
      reconnectBtn.disabled = true;
      if (badgeEl) {
        badgeEl.className = 'host-status-badge checking';
        badgeEl.textContent = 'Connecting…';
      }
      if (detailEl) detailEl.textContent = 'Testing native messaging connection…';
      await refreshHostStatus();
      reconnectBtn.disabled = false;
    });
  }

  if (hostUpdateBannerBtn) {
    hostUpdateBannerBtn.addEventListener('click', async () => {
      hostUpdateBannerBtn.disabled = true;
      hostUpdateBannerBtn.textContent = 'Updating Native Host…';
      if (updateBannerStatusEl) {
        updateBannerStatusEl.hidden = true;
      }
      try {
        const applyResp = await ext.runtime.sendMessage({ type: 'APPLY_NATIVE_UPDATE' });
        if (applyResp && applyResp.ok && applyResp.result && applyResp.result.ok) {
          const relTag = applyResp.result.releaseTag || '';
          hostUpdateBannerBtn.hidden = true;
          if (updateBannerStatusEl) {
            updateBannerStatusEl.style.color = '#2ecc71';
            updateBannerStatusEl.textContent = applyResp.result.alreadyUpToDate
              ? `✓ Native Host already matches GitHub${relTag ? ` (${relTag})` : ''}`
              : `✓ Native Host updated${relTag ? ` (${relTag})` : ''}${applyResp.restarted ? ' — restarted' : ''}`;
            updateBannerStatusEl.hidden = false;
            updateBannerWrapEl.hidden = false;
          }
          await refreshHostStatus();
          await fetchModels({ forceUpdateCheck: true });
          if (updateStatusHideTimer) clearTimeout(updateStatusHideTimer);
          updateStatusHideTimer = setTimeout(() => {
            if (updateBannerStatusEl) updateBannerStatusEl.hidden = true;
            renderUpdateBanner(latestUpdateStatus);
          }, 3500);
        } else {
          const errText = applyResp?.result?.errors?.join('; ') || applyResp?.error || 'Update failed';
          if (updateBannerStatusEl) {
            updateBannerStatusEl.style.color = '#e74c3c';
            updateBannerStatusEl.textContent = `✕ Native Host update failed: ${errText}`;
            updateBannerStatusEl.hidden = false;
            updateBannerWrapEl.hidden = false;
          }
          hostUpdateBannerBtn.disabled = false;
          renderUpdateBanner(latestUpdateStatus);
        }
      } catch (err) {
        if (updateBannerStatusEl) {
          updateBannerStatusEl.style.color = '#e74c3c';
          updateBannerStatusEl.textContent = `✕ Error: ${err.message || String(err)}`;
          updateBannerStatusEl.hidden = false;
          updateBannerWrapEl.hidden = false;
        }
        hostUpdateBannerBtn.disabled = false;
        renderUpdateBanner(latestUpdateStatus);
      }
    });
  }

  if (classifierUpdateBannerBtn) {
    classifierUpdateBannerBtn.addEventListener('click', () => {
      ext.runtime.sendMessage({ type: 'OPEN_MODELS_HUB' }).catch(() => {
        if (ext.tabs && ext.tabs.create) {
          ext.tabs.create({ url: ext.runtime.getURL('docs/index.html?screen=2&manage=1') });
        }
      });
    });
  }

  if (checkUpdateBtn && updateStatusEl) {
    checkUpdateBtn.addEventListener('click', async () => {
      checkUpdateBtn.disabled = true;
      checkUpdateBtn.innerHTML = '<span>Checking for updates…</span>';
      updateStatusEl.style.display = 'block';
      updateStatusEl.innerHTML = '<span style="color: #f1c40f;">Connecting to GitHub / Native Host…</span>';

      try {
        const resp = await ext.runtime.sendMessage({ type: 'CHECK_NATIVE_UPDATE', force: true });
        checkUpdateBtn.disabled = false;
        checkUpdateBtn.innerHTML = '<span>↻ Check for Updates</span>';

        if (resp && resp.ok && resp.updateInfo) {
          const info = resp.updateInfo;
          const clfUpdates = info.classifierUpdates || info.updates?.classifiers || {};
          const outdated = Array.isArray(clfUpdates.outdated) ? clfUpdates.outdated : [];

          if (info.updates && !disableUpdateNotifications) {
            latestUpdateStatus = info.updates;
            renderUpdateBanner(latestUpdateStatus);
          }

          let html = '';
          const relTag = info.releaseTag || info.latestVersion || 'latest';
          const diffCount = Number(info.diffCount) || 0;
          if (info.updateAvailable) {
            const dlUrl = escapeHtml(info.downloadUrl || 'https://github.com/LuckyCowMoo/Fire-vs-Fire/releases/latest');
            const diffDetail = diffCount > 0 ? ` — ${diffCount} file(s) differ` : '';
            html += `
              <div style="color: #ff9d66; font-weight: 600; margin-bottom: 4px;">⚡ Host Update Available (${escapeHtml(relTag)}${diffDetail})</div>
              <div style="color: var(--text-muted); font-size: 11px; margin-bottom: 8px;">${escapeHtml(info.releaseNotes || 'Differs from current GitHub release.')}</div>
              <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: ${outdated.length ? '8px' : '0'};">
                ${resp.connected ? '<button type="button" id="apply-native-update-btn" class="btn btn-outline" style="font-size: 11px; padding: 4px 10px; border-color: #2ecc71; color: #2ecc71;">⚡ Update In-Place Now</button>' : ''}
                <a href="${dlUrl}" target="_blank" rel="noopener noreferrer" class="btn btn-outline" style="font-size: 11px; padding: 4px 10px; text-decoration: none;">Download ZIP</a>
              </div>
            `;
          } else if (info.checked === false) {
            html += `<div style="color: #f1c40f; margin-bottom: ${outdated.length ? '6px' : '0'};">⚠️ Could not verify GitHub release (network unavailable).</div>`;
          } else {
            html += `<div style="color: #2ecc71; margin-bottom: ${outdated.length ? '6px' : '0'};">✅ Native Host matches GitHub release (${escapeHtml(relTag)}).</div>`;
          }

          if (outdated.length > 0) {
            const listItems = outdated
              .map((item) => (
                item.reason === 'release_updated'
                  ? `<li><strong>${escapeHtml(item.installedName || item.installedId)}</strong> — updated weights on GitHub release</li>`
                  : `<li>${escapeHtml(item.installedName || item.installedId)} → <strong>${escapeHtml(item.latestName || item.latestId)}</strong></li>`
              ))
              .join('');
            html += `
              <div style="color: #ffd099; font-weight: 600; margin-bottom: 4px;">⚡ Classifier Update${outdated.length > 1 ? 's' : ''} Available:</div>
              <ul style="margin: 0 0 8px 16px; padding: 0; font-size: 11px; color: var(--text-secondary);">${listItems}</ul>
              <button type="button" id="open-classifier-updates-btn" class="btn btn-outline" style="font-size: 11px; padding: 4px 10px; border-color: #ff9838; color: #ffd099;">Open FvF Classifiers</button>
            `;
          }

          updateStatusEl.innerHTML = html;

          const applyBtn = document.getElementById('apply-native-update-btn');
          if (applyBtn) {
            applyBtn.addEventListener('click', async () => {
              applyBtn.disabled = true;
              applyBtn.textContent = 'Updating components…';
              try {
                const applyResp = await ext.runtime.sendMessage({ type: 'APPLY_NATIVE_UPDATE' });
                if (applyResp && applyResp.ok && applyResp.result && applyResp.result.ok) {
                  updateStatusEl.innerHTML = `<span style="color: #2ecc71; font-weight: 600;">✅ ${escapeHtml(applyResp.result?.message || 'Updated successfully!')}</span>`;
                  await refreshHostStatus();
                  await fetchModels({ forceUpdateCheck: true });
                } else {
                  updateStatusEl.innerHTML = `<span style="color: #e74c3c;">❌ Update failed: ${escapeHtml(applyResp?.error || 'Unknown error')}</span>`;
                }
              } catch (e) {
                updateStatusEl.innerHTML = `<span style="color: #e74c3c;">❌ Error: ${escapeHtml(e.message)}</span>`;
              }
            });
          }

          const openClfBtn = document.getElementById('open-classifier-updates-btn');
          if (openClfBtn) {
            openClfBtn.addEventListener('click', () => {
              ext.runtime.sendMessage({ type: 'OPEN_MODELS_HUB' }).catch(() => {
                if (ext.tabs && ext.tabs.create) {
                  ext.tabs.create({ url: ext.runtime.getURL('docs/index.html?screen=2&manage=1') });
                }
              });
            });
          }
        } else {
          updateStatusEl.innerHTML = `<span style="color: #e74c3c;">Could not check updates: ${escapeHtml(resp?.error || 'Native Host not responding')}</span>`;
        }
      } catch (err) {
        checkUpdateBtn.disabled = false;
        checkUpdateBtn.innerHTML = '<span>↻ Check for Updates</span>';
        updateStatusEl.innerHTML = `<span style="color: #e74c3c;">Error: ${escapeHtml(err.message || String(err))}</span>`;
      }
    });
  }

  refreshHostStatus();
}

function initCommunityReportingSettings() {
  const toggleEl = document.getElementById('communityReportingEnabled');
  const configEl = document.getElementById('community-reporting-config');
  const instructionsEl = document.getElementById('community-hf-token-instructions');
  const tokenEl = document.getElementById('communityHfToken');
  const toggleTokenVisBtn = document.getElementById('toggle-community-hf-token-btn');
  const verifyBtn = document.getElementById('verify-community-hf-btn');
  const statusEl = document.getElementById('community-hf-status');
  if (!toggleEl || !configEl || !tokenEl || !verifyBtn || !statusEl) return;

  let verifySeq = 0;
  let autoVerifyTimer = null;
  let lastVerifiedToken = '';

  function setInstructionsVisible(visible) {
    if (instructionsEl) {
      instructionsEl.hidden = !visible;
    }
  }

  function renderStatus(username, errText = '', infoText = '') {
    if (errText) {
      statusEl.textContent = errText;
      statusEl.style.color = '#f87171';
      setInstructionsVisible(true);
    } else if (infoText) {
      statusEl.textContent = infoText;
      statusEl.style.color = '#fbbf24';
    } else if (username) {
      statusEl.textContent = `✓ Write permission verified — connected as @${username}`;
      statusEl.style.color = '#34d399';
      setInstructionsVisible(false);
    } else {
      statusEl.textContent = 'Not connected';
      statusEl.style.color = 'var(--text-muted)';
      setInstructionsVisible(true);
    }
  }

  async function runTokenVerification(tokenValue) {
    const seq = ++verifySeq;
    const token = String(tokenValue || '').trim();
    if (!token) {
      lastVerifiedToken = '';
      await ext.storage.local.remove(['communityHfToken', 'communityHfUsername']).catch(() => {});
      if (seq === verifySeq) {
        renderStatus('');
      }
      return;
    }
    if (!token.startsWith('hf_')) {
      lastVerifiedToken = '';
      await ext.storage.local.set({ communityHfToken: token }).catch(() => {});
      await ext.storage.local.remove(['communityHfUsername']).catch(() => {});
      if (seq === verifySeq) {
        renderStatus('', 'Token must start with hf_ (make sure you use a Write token)');
      }
      return;
    }
    if (token.length < 10) {
      lastVerifiedToken = '';
      if (seq === verifySeq) {
        renderStatus('', '', 'Finish pasting your Hugging Face token…');
      }
      return;
    }

    verifyBtn.disabled = true;
    verifyBtn.textContent = 'Checking…';
    if (seq === verifySeq) {
      renderStatus('', '', 'Checking Hugging Face token write permissions…');
    }

    try {
      const resp = await ext.runtime.sendMessage({
        type: 'VERIFY_COMMUNITY_HF_TOKEN',
        token
      });
      if (seq !== verifySeq) return;
      if (resp && resp.ok && resp.username) {
        lastVerifiedToken = token;
        renderStatus(resp.username);
      } else {
        lastVerifiedToken = '';
        renderStatus('', resp?.error || 'Invalid Hugging Face token or missing write permission');
      }
    } catch (err) {
      if (seq !== verifySeq) return;
      lastVerifiedToken = '';
      renderStatus('', err?.message || 'Verification failed');
    } finally {
      if (seq === verifySeq) {
        verifyBtn.disabled = false;
        verifyBtn.textContent = 'Check Token';
      }
    }
  }

  if (toggleTokenVisBtn) {
    toggleTokenVisBtn.addEventListener('click', () => {
      const revealing = tokenEl.type === 'password';
      tokenEl.type = revealing ? 'text' : 'password';
      toggleTokenVisBtn.textContent = revealing ? 'Hide' : 'Show';
      toggleTokenVisBtn.setAttribute('aria-label', revealing ? 'Hide token' : 'Show token');
      toggleTokenVisBtn.setAttribute('aria-pressed', revealing ? 'true' : 'false');
    });
  }

  ext.storage.local
    .get(['communityReportingEnabled', 'communityHfToken', 'communityHfUsername'])
    .then((res) => {
      const enabled = Boolean(res?.communityReportingEnabled);
      toggleEl.checked = enabled;
      configEl.hidden = !enabled;
      const savedToken = typeof res?.communityHfToken === 'string' ? res.communityHfToken.trim() : '';
      const savedUser = typeof res?.communityHfUsername === 'string' ? res.communityHfUsername.trim() : '';
      if (savedToken) {
        tokenEl.value = savedToken;
      }
      if (savedToken && savedUser) {
        lastVerifiedToken = savedToken;
        renderStatus(savedUser);
      } else if (savedToken) {
        runTokenVerification(savedToken);
      } else {
        renderStatus('');
      }
    })
    .catch(() => {});

  toggleEl.addEventListener('change', async () => {
    const enabled = Boolean(toggleEl.checked);
    configEl.hidden = !enabled;
    try {
      await ext.storage.local.set({ communityReportingEnabled: enabled });
    } catch (_) {}
  });

  tokenEl.addEventListener('input', () => {
    if (autoVerifyTimer) {
      clearTimeout(autoVerifyTimer);
      autoVerifyTimer = null;
    }
    const current = (tokenEl.value || '').trim();
    if (!current) {
      verifySeq += 1;
      lastVerifiedToken = '';
      ext.storage.local.remove(['communityHfToken', 'communityHfUsername']).catch(() => {});
      renderStatus('');
      return;
    }
    if (current !== lastVerifiedToken) {
      setInstructionsVisible(true);
      if (current.startsWith('hf_') && current.length >= 10) {
        renderStatus('', '', 'Checking Hugging Face token write permissions…');
      }
    }
    autoVerifyTimer = setTimeout(() => {
      autoVerifyTimer = null;
      runTokenVerification(tokenEl.value);
    }, 400);
  });

  verifyBtn.addEventListener('click', () => {
    if (autoVerifyTimer) {
      clearTimeout(autoVerifyTimer);
      autoVerifyTimer = null;
    }
    runTokenVerification(tokenEl.value);
  });
}

initFloatingEmbers();
initNavButtonsAndHostUpdater();
initCommunityReportingSettings();
