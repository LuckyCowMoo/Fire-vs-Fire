/* global browser, chrome, FVFVisualEffects, FVFEnsembles */
const ext = typeof browser !== 'undefined' ? browser : chrome;
const VE = globalThis.FVFVisualEffects;
const ENS = globalThis.FVFEnsembles;

const PREVIEW_IMAGE_PATH = 'options/preview-fire.jpg';
const LOREM =
  'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore.';

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
const classificationDelayInput = document.getElementById('classificationDelay');
const classificationDelayValue = document.getElementById('classificationDelayValue');
const imageCaptureQualityInput = document.getElementById('imageCaptureQuality');
const imageCaptureQualityValue = document.getElementById('imageCaptureQualityValue');
const verboseLogsInput = document.getElementById('verboseLogs');
const lazyLoadInput = document.getElementById('lazyLoad');
const minTextLengthInput = document.getElementById('minTextLength');

const pauseOnBatteryInput = document.getElementById('pauseOnBattery');
const pauseBelowThresholdInput = document.getElementById('pauseBelowThreshold');
const batteryThresholdSelect = document.getElementById('batteryThreshold');
const powerLiveBadge = document.getElementById('power-live-badge');
const powerLiveDot = document.getElementById('power-live-dot');
const powerLiveText = document.getElementById('power-live-text');

let isHostConnected = false;
let isClassifying = false;
let classifyCount = 0;
let isPowerPaused = false;
let currentPowerStatus = null;

const refreshBtn = document.getElementById('refresh-models');
const modelStatusEl = document.getElementById('model-status');

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
  const measured = html.clientWidth;
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
  if (!powerLiveBadge || !powerLiveText) return;
  const isPlugged = power.is_plugged_in !== false;
  const pct = typeof power.battery_percent === 'number' ? power.battery_percent : null;
  const hasBattery = power.has_battery !== false;

  powerLiveBadge.className = 'power-live-badge';
  if (isPlugged) {
    powerLiveBadge.classList.add('plugged');
    powerLiveText.textContent = pct != null ? `Plugged In (${pct}%)` : 'Plugged In';
  } else if (hasBattery) {
    if (pct != null && pct <= 20) {
      powerLiveBadge.classList.add('low');
      powerLiveText.textContent = `Low Battery: ${pct}%`;
    } else {
      powerLiveBadge.classList.add('battery');
      powerLiveText.textContent = pct != null ? `Battery: ${pct}%` : 'On Battery';
    }
  } else {
    powerLiveBadge.classList.add('plugged');
    powerLiveText.textContent = 'AC Power';
  }

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
  if ((ensemble.modality || 'image') === 'image') {
    return ensemble.styles?.image?.borderColor || '#FF0064';
  }
  const text = ensemble.styles?.text || {};
  return text.strikethroughColor || text.underlineColor || text.highlightColor || '#FF0064';
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
    if ((value || 'hover') === mode.value) btn.classList.add('active');
    btn.addEventListener('click', () => {
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
    blurAmount: 4,
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

  return { element: wrap, refresh };
}

function createSwitch(checked, onChange) {
  const label = document.createElement('label');
  label.className = 'switch';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = checked;
  const slider = document.createElement('span');
  slider.className = 'switch-slider';
  slider.setAttribute('aria-hidden', 'true');
  input.addEventListener('change', () => onChange(input.checked));
  label.appendChild(input);
  label.appendChild(slider);
  return label;
}

function generateEnsembleId(modality) {
  const prefix = modality === 'text' ? 'text' : 'image';
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
}

function formatClassifierRate(classifierId, modalities) {
  const t = classifierTimings && classifierTimings[classifierId];
  if (!t || !t.avgMs || t.avgMs <= 0) return '—';
  const perSec = 1000 / t.avgMs;
  const unit = modalities && modalities.includes && modalities.includes('image')
    ? 'img/s'
    : modalities && modalities.includes && modalities.includes('text')
      ? 'txt/s'
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

  const title = document.createElement('div');
  title.className = 'hint';
  title.textContent = 'Classifier weights';
  section.appendChild(title);

  const hint = document.createElement('div');
  hint.className = 'hint';
  hint.textContent = 'Weights are normalized when used.';
  section.appendChild(hint);

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

function getCategoryLabelForSelection(selectedClassifiers, classifierNamesById, index) {
  if (selectedClassifiers.length === 1) {
    return classifierNamesById.get(selectedClassifiers[0]) || selectedClassifiers[0];
  }
  return getDefaultEnsembleName(index);
}

function getCategoryDisplayName(ensemble, index, classifierNamesById) {
  const selectedClassifiers = Array.isArray(ensemble.classifiers) ? ensemble.classifiers : [];
  const autoLabel = getCategoryLabelForSelection(selectedClassifiers, classifierNamesById, index);
  if (selectedClassifiers.length >= 2 && typeof ensemble.name === 'string' && ensemble.name.trim()) {
    return ensemble.name.trim();
  }
  if (typeof ensemble.name === 'string' && ensemble.name.trim()) {
    return ensemble.name.trim();
  }
  return autoLabel;
}

async function saveEnsembles() {
  try {
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    if (storageArea && storageArea.set) {
      await storageArea.set({ ensembleConfigsV2: ensembleConfigs });
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
    const isSelected = index === selectedCategoryIndex;

    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = 'category-tile';
    tile.setAttribute('role', 'tab');
    tile.setAttribute('aria-selected', isSelected ? 'true' : 'false');
    if (isSelected) tile.classList.add('selected');
    if (!isEnabled) tile.classList.add('disabled-cat');
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
      selectedCategoryIndex = index;
      renderEnsembles({ scrollStripToSelection: true });
    });

    categoryStripEl.appendChild(tile);
  });
}

function renderCategoryPanel(index, classifierNamesById) {
  if (!categoryPanelEl) return;

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
  const categoryLabel = getCategoryLabelForSelection(selectedClassifiers, classifierNamesById, index);

  if (selectedClassifiers.length >= 2) {
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.className = 'input-text';
    nameInput.value = typeof ensemble.name === 'string' ? ensemble.name : '';
    nameInput.placeholder = categoryLabel;
    nameInput.addEventListener('input', () => {
      ensemble.name = nameInput.value.trim();
      scheduleSaveEnsembles();
      renderCategoryStrip(classifierNamesById);
    });
    header.appendChild(nameInput);
  } else {
    if (ensemble.name !== categoryLabel) {
      ensemble.name = categoryLabel;
      saveEnsembles();
    }
    const nameSpan = document.createElement('span');
    nameSpan.className = 'style-row-label';
    nameSpan.textContent = categoryLabel;
    header.appendChild(nameSpan);
  }

  const modalitySelect = document.createElement('select');
  modalitySelect.className = 'select-styled';
  ['image', 'text'].forEach((mod) => {
    const opt = document.createElement('option');
    opt.value = mod;
    opt.textContent = mod === 'image' ? 'Images' : 'Text';
    modalitySelect.appendChild(opt);
  });
  modalitySelect.value = ensemble.modality || 'image';
  modalitySelect.addEventListener('change', () => {
    ensemble.modality = modalitySelect.value;
    renderEnsembles();
    saveEnsembles();
  });
  header.appendChild(modalitySelect);

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

  const classifiersLabel = document.createElement('div');
  classifiersLabel.className = 'section-label';
  classifiersLabel.textContent = 'Classifiers';
  categoryPanelEl.appendChild(classifiersLabel);

  const supported = ensemble.modality || 'image';
  const compatible = availableClassifiers.filter((model) => {
    const modalities = model.modalities || [];
    return modalities.length === 0 || modalities.includes(supported);
  });

  const toggleList = document.createElement('div');
  toggleList.className = 'classifier-toggles';

  if (!compatible.length) {
    const empty = document.createElement('div');
    empty.className = 'classifier-empty';
    empty.textContent = 'No compatible classifiers — try Refresh classifiers';
    toggleList.appendChild(empty);
  } else {
    compatible.forEach((model) => {
      const row = document.createElement('div');
      row.className = 'classifier-toggle-row';
      row.dataset.classifierId = model.id;

      const info = document.createElement('div');
      info.className = 'classifier-toggle-info';

      const nameEl = document.createElement('span');
      nameEl.className = 'classifier-toggle-name';
      nameEl.textContent = model.name || model.id;

      const rateEl = document.createElement('span');
      rateEl.className = 'classifier-rate';
      rateEl.textContent = formatClassifierRate(model.id, model.modalities || []);

      info.appendChild(nameEl);
      info.appendChild(rateEl);

      const isSelected =
        Array.isArray(ensemble.classifiers) && ensemble.classifiers.includes(model.id);

      const toggle = createSwitch(isSelected, (checked) => {
        const currentClassifiers = Array.isArray(ensemble.classifiers) ? ensemble.classifiers : [];
        const nextClassifiers = currentClassifiers.filter((id) => id !== model.id);
        if (checked) nextClassifiers.push(model.id);
        updateEnsembleClassifiers(ensemble, nextClassifiers);
        renderEnsembles();
        saveEnsembles();
      });

      row.appendChild(info);
      row.appendChild(toggle);
      toggleList.appendChild(row);
    });
  }

  categoryPanelEl.appendChild(toggleList);

  const currentSelected = Array.isArray(ensemble.classifiers) ? ensemble.classifiers : [];
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
    blurRange.value = imageStyles.blurAmount ?? 4;
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

    categoryPanelEl.appendChild(
      createColorSliders(
        'Strike color',
        textStyles.strikethroughColor || '#ff0064',
        onColorChange((value) => {
          ensemble.styles = ensemble.styles || {};
          ensemble.styles.text = ensemble.styles.text || {};
          ensemble.styles.text.strikethroughColor = value;
        })
      )
    );

    const underlineMode = textStyles.underlineMode || (textStyles.applyUnderline === true ? 'hover' : 'off');
    const underlineRow = createIconModeControl('Underline', underlineMode, (value) => {
      ensemble.styles = ensemble.styles || {};
      ensemble.styles.text = ensemble.styles.text || {};
      ensemble.styles.text.underlineMode = value;
      saveEnsembles();
      onPreviewUpdate();
    });
    categoryPanelEl.appendChild(underlineRow);

    categoryPanelEl.appendChild(
      createColorSliders(
        'Underline color',
        textStyles.underlineColor || '#ff0064',
        onColorChange((value) => {
          ensemble.styles = ensemble.styles || {};
          ensemble.styles.text = ensemble.styles.text || {};
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

  const actions = document.createElement('div');
  actions.className = 'category-detail-actions';

  const removeBtn = document.createElement('button');
  removeBtn.type = 'button';
  removeBtn.className = 'btn btn-outline btn-danger';
  removeBtn.textContent = 'Remove category';

  removeBtn.addEventListener('click', () => {
    const confirm = document.createElement('div');
    confirm.className = 'confirm-inline';
    confirm.innerHTML = '<span>Remove this category?</span>';

    const yesBtn = document.createElement('button');
    yesBtn.type = 'button';
    yesBtn.className = 'btn btn-danger';
    yesBtn.textContent = 'Yes';

    const noBtn = document.createElement('button');
    noBtn.type = 'button';
    noBtn.className = 'btn btn-outline';
    noBtn.textContent = 'No';

    yesBtn.addEventListener('click', () => {
      ensembleConfigs = ensembleConfigs.filter((_, idx) => idx !== index);
      normalizeSelectedIndex();
      renderEnsembles();
      saveEnsembles();
    });

    noBtn.addEventListener('click', () => {
      confirm.remove();
      removeBtn.disabled = false;
    });

    confirm.appendChild(yesBtn);
    confirm.appendChild(noBtn);
    actions.appendChild(confirm);
    removeBtn.disabled = true;
  });

  actions.appendChild(removeBtn);
  categoryPanelEl.appendChild(actions);
}

function renderEnsembles(options = {}) {
  const { scrollStripToSelection = false } = options;
  const scrollY = getDocumentScrollY();

  const classifierNamesById = new Map();
  availableClassifiers.forEach((m) => {
    classifierNamesById.set(m.id, m.name || m.id);
  });

  let ensemblesDirty = false;
  ensembleConfigs.forEach((ensemble, index) => {
    if (!ensemble.id) {
      ensemble.id = generateEnsembleId(ensemble.modality || 'image');
      ensemblesDirty = true;
    }
    if (typeof ensemble.name !== 'string' || !ensemble.name.trim()) {
      ensemble.name = getDefaultEnsembleName(index);
      ensemblesDirty = true;
    }
  });

  if (ensemblesDirty) {
    scheduleSaveEnsembles();
  }

  normalizeSelectedIndex();

  const hasCategories = ensembleConfigs.length > 0;

  if (categoryStripWrapEl) {
    categoryStripWrapEl.hidden = !hasCategories;
  }
  if (categoryPanelEl) {
    categoryPanelEl.hidden = !hasCategories;
  }

  if (hasCategories) {
    renderCategoryStrip(classifierNamesById);
    renderCategoryPanel(selectedCategoryIndex, classifierNamesById);

    if (scrollStripToSelection) {
      scrollCategoryStripToSelected();
      animateCategoryPanelEntrance();
    }
  } else if (categoryStripEl) {
    categoryStripEl.innerHTML = '';
    categoryPanelEl.innerHTML = '';
  }

  requestAnimationFrame(() => {
    stabilizePopupWidth();
    restoreDocumentScrollY(scrollY);
  });
}

function createNewEnsemble(modality = 'image') {
  const id = generateEnsembleId(modality);
  if (modality === 'image') {
    return {
      id,
      modality,
      enabled: true,
      threshold: 0.5,
      classifiers: [],
      weights: null,
      styles: {
        image: {
          applyBlur: true,
          blurAmount: 4,
          applyBorder: true,
          borderMultiplier: 1,
          borderColor: '#7C4DFF',
          applyBadge: true,
          blurMode: 'hover',
          borderMode: 'hover',
          badgeMode: 'hover'
        }
      }
    };
  }

  return {
    id,
    modality,
    enabled: true,
    threshold: 0.5,
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

async function load() {
  try {
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    const res = storageArea && storageArea.get
      ? await storageArea.get([
          'enabled',
          'altTextOnly',
          'pauseOnBattery',
          'pauseBelowThreshold',
          'batteryThreshold',
          'classificationDelay',
          'imageCaptureQuality',
          'verboseLogs',
          'lazyLoad',
          'minTextLength',
          'ensembleConfigsV2'
        ])
      : {};

    if (res && typeof res.enabled === 'boolean') enabledInput.checked = res.enabled;
    if (res && typeof res.altTextOnly === 'boolean') {
      altTextOnlyInput.checked = res.altTextOnly;
    } else {
      altTextOnlyInput.checked = true; // Default true on first install
    }
    if (pauseOnBatteryInput && res && typeof res.pauseOnBattery === 'boolean') {
      pauseOnBatteryInput.checked = res.pauseOnBattery;
    }
    if (pauseBelowThresholdInput && res && typeof res.pauseBelowThreshold === 'boolean') {
      pauseBelowThresholdInput.checked = res.pauseBelowThreshold;
    }
    if (batteryThresholdSelect && res && (typeof res.batteryThreshold === 'number' || typeof res.batteryThreshold === 'string')) {
      batteryThresholdSelect.value = String(res.batteryThreshold);
    }
    if (res && typeof res.classificationDelay === 'number') {
      classificationDelayInput.value = res.classificationDelay;
    }
    if (res && typeof res.imageCaptureQuality === 'number') {
      imageCaptureQualityInput.value = res.imageCaptureQuality;
    }
    if (res && typeof res.verboseLogs === 'boolean') verboseLogsInput.checked = res.verboseLogs;
    if (res && typeof res.lazyLoad === 'boolean') lazyLoadInput.checked = res.lazyLoad;
    if (res && typeof res.minTextLength === 'number') minTextLengthInput.value = res.minTextLength;

    if (res && Array.isArray(res.ensembleConfigsV2) && res.ensembleConfigsV2.length) {
      ensembleConfigs = res.ensembleConfigsV2;
    } else {
      ensembleConfigs = ENS.buildDefaultEnsembles();
      await saveEnsembles();
    }

    if (res && res.classifierTimings && typeof res.classifierTimings === 'object') {
      classifierTimings = res.classifierTimings;
    }

    updateDisplays();
    renderEnsembles();
  } catch {
    // Ignore load errors for missing storage values.
  }
}

async function saveSetting(key, value) {
  try {
    const storageArea = ext.storage && (ext.storage.sync || ext.storage.local);
    if (storageArea && storageArea.set) {
      await storageArea.set({ [key]: value });
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

minTextLengthInput.addEventListener('change', async () => {
  const value = Math.max(50, Math.min(2000, parseInt(minTextLengthInput.value, 10) || 250));
  minTextLengthInput.value = value;
  await saveSetting('minTextLength', value);
});

initCollapsibleAnimations();
initInteractionParticles();
load().then(() => requestAnimationFrame(stabilizePopupWidth));
refreshRunState();
runStateTimer = setInterval(refreshRunState, 800);

if (ext && ext.storage && ext.storage.onChanged) {
  ext.storage.onChanged.addListener((changes) => {
    if (Object.prototype.hasOwnProperty.call(changes, 'classifierTimings')) {
      classifierTimings = changes.classifierTimings.newValue || {};
      updateClassifierRateLabels();
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
  addEnsembleBtn.addEventListener('click', async () => {
    ensembleConfigs.push(createNewEnsemble('image'));
    selectedCategoryIndex = ensembleConfigs.length - 1;
    renderEnsembles({ scrollStripToSelection: true });
    await saveEnsembles();
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

function setModelStatus(text, type = 'info') {
  if (!modelStatusEl) return;
  modelStatusEl.textContent = text;
  modelStatusEl.hidden = !text;
  modelStatusEl.style.color = type === 'error' ? '#c04a4a' : '';
}

async function fetchModels() {
  setModelStatus('Loading models…');
  try {
    const resp = await ext.runtime.sendMessage({ type: 'LIST_MODELS' });
    if (!resp || resp.ok !== true) throw new Error(resp?.error || 'Failed to list models');
    const models = Array.isArray(resp.models) ? resp.models : [];
    const rawClassifiers = resp.raw && Array.isArray(resp.raw.classifiers) ? resp.raw.classifiers : [];
    availableClassifiers = rawClassifiers.length
      ? rawClassifiers
      : models.map((id) => ({ id, name: id, label: 'AI generated', modalities: [] }));
    renderEnsembles();
    setModelStatus(models.length ? `${models.length} model(s) loaded` : 'No models found');
  } catch (err) {
    console.warn('LIST_MODELS failed', err);
    availableClassifiers = [];
    renderEnsembles();
    setModelStatus(err.message || 'Failed to list models', 'error');
  }
}

if (refreshBtn) {
  refreshBtn.addEventListener('click', () => fetchModels());
  const openOnboardingBtn = document.getElementById('open-onboarding');
  if (openOnboardingBtn) {
    openOnboardingBtn.addEventListener('click', () => {
      ext.runtime.sendMessage({ type: 'OPEN_ONBOARDING' }).catch(() => {});
    });
  }
}

(async () => {
  await fetchModels();
})();

window.addEventListener('focus', () => {
  fetchModels().catch(() => {});
});

// Initialize accent sliders on load
[classificationDelayInput, imageCaptureQualityInput].forEach((el) => {
  if (el) updateSliderBackground(el);
});


// ============================================================================
// FLOATING EMBERS ENGINE (BACKGROUND EFFECT)
// ============================================================================

function initFloatingEmbers() {
  const canvas = document.getElementById('ember-webgl-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const embers = Array.from({ length: 48 }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    vx: 0.2 + Math.random() * 0.6,
    vy: -0.3 - Math.random() * 0.6,
    r: 1.2 + Math.random() * 2.5,
    alpha: 0.18 + Math.random() * 0.55,
    hue: 14 + Math.random() * 32
  }));

  function renderEmbers() {
    ctx.clearRect(0, 0, width, height);
    for (const p of embers) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x > width + 20) p.x = -10;
      if (p.y < -20) p.y = height + 10;

      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4.2);
      grad.addColorStop(0, `hsla(${p.hue}, 100%, 72%, ${p.alpha})`);
      grad.addColorStop(0.45, `hsla(${p.hue - 6}, 100%, 54%, ${p.alpha * 0.45})`);
      grad.addColorStop(1, 'hsla(10, 100%, 40%, 0)');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 4.2, 0, Math.PI * 2);
      ctx.fill();
    }
    requestAnimationFrame(renderEmbers);
  }
  requestAnimationFrame(renderEmbers);
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

  if (checkUpdateBtn && updateStatusEl) {
    checkUpdateBtn.addEventListener('click', async () => {
      checkUpdateBtn.disabled = true;
      checkUpdateBtn.innerHTML = '<span>Checking for updates…</span>';
      updateStatusEl.style.display = 'block';
      updateStatusEl.innerHTML = '<span style="color: #f1c40f;">Connecting to GitHub / Native Host…</span>';

      try {
        const resp = await ext.runtime.sendMessage({ type: 'CHECK_NATIVE_UPDATE' });
        checkUpdateBtn.disabled = false;
        checkUpdateBtn.innerHTML = '<span>↻ Check for Updates</span>';

        if (resp && resp.ok && resp.updateInfo) {
          const info = resp.updateInfo;
          if (info.updateAvailable) {
            updateStatusEl.innerHTML = `
              <div style="color: #ff9d66; font-weight: 600; margin-bottom: 4px;">⚡ Update Available: v${info.latestVersion} (Current: ${info.currentVersion})</div>
              <div style="color: var(--text-muted); font-size: 11px; margin-bottom: 8px;">${info.releaseNotes || 'Includes new features & optimizations.'}</div>
              <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                ${resp.connected ? '<button type="button" id="apply-native-update-btn" class="btn btn-outline" style="font-size: 11px; padding: 4px 10px; border-color: #2ecc71; color: #2ecc71;">⚡ Update In-Place Now</button>' : ''}
                <a href="${info.downloadUrl || 'https://github.com/LuckyCowMoo/Fire-vs-Fire/releases/tag/v2.0'}" target="_blank" class="btn btn-outline" style="font-size: 11px; padding: 4px 10px; text-decoration: none;">Download ZIP</a>
              </div>
            `;

            const applyBtn = document.getElementById('apply-native-update-btn');
            if (applyBtn) {
              applyBtn.addEventListener('click', async () => {
                applyBtn.disabled = true;
                applyBtn.textContent = 'Updating components…';
                try {
                  const applyResp = await ext.runtime.sendMessage({ type: 'APPLY_NATIVE_UPDATE' });
                  if (applyResp && applyResp.ok) {
                    updateStatusEl.innerHTML = `<span style="color: #2ecc71; font-weight: 600;">✅ ${applyResp.result?.message || 'Updated successfully!'}</span>`;
                    refreshHostStatus();
                  } else {
                    updateStatusEl.innerHTML = `<span style="color: #e74c3c;">❌ Update failed: ${applyResp?.error || 'Unknown error'}</span>`;
                  }
                } catch (e) {
                  updateStatusEl.innerHTML = `<span style="color: #e74c3c;">❌ Error: ${e.message}</span>`;
                }
              });
            }
          } else {
            updateStatusEl.innerHTML = `<span style="color: #2ecc71;">✅ Native Host is up to date (v${info.currentVersion || '2.1.0'}).</span>`;
          }
        } else {
          updateStatusEl.innerHTML = `<span style="color: #e74c3c;">Could not check updates: ${resp?.error || 'Native Host not responding'}</span>`;
        }
      } catch (err) {
        checkUpdateBtn.disabled = false;
        checkUpdateBtn.innerHTML = '<span>↻ Check for Updates</span>';
        updateStatusEl.innerHTML = `<span style="color: #e74c3c;">Error: ${err.message || String(err)}</span>`;
      }
    });
  }

  refreshHostStatus();
}

initFloatingEmbers();
initNavButtonsAndHostUpdater();
