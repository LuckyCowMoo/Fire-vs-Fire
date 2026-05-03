/* global browser, chrome */
const ext = typeof browser !== 'undefined' ? browser : chrome;

const enabledInput = document.getElementById('enabled');
const altTextOnlyInput = document.getElementById('altTextOnly');
const classificationDelayInput = document.getElementById('classificationDelay');
const classificationDelayValue = document.getElementById('classificationDelayValue');
const imageCaptureQualityInput = document.getElementById('imageCaptureQuality');
const imageCaptureQualityValue = document.getElementById('imageCaptureQualityValue');
const verboseLogsInput = document.getElementById('verboseLogs');
const lazyLoadInput = document.getElementById('lazyLoad');
const minTextLengthInput = document.getElementById('minTextLength');

// Model list elements
const refreshBtn = document.getElementById('refresh-models');
const modelStatusEl = document.getElementById('model-status');

// Ensemble editor elements
const ensembleListEl = document.getElementById('ensemble-list');
const addEnsembleBtn = document.getElementById('add-ensemble');
const runStateEl = document.getElementById('run-state');
const runStateTextEl = document.getElementById('run-state-text');
const statusLogoEl = document.getElementById('status-logo');

let availableClassifiers = [];
let ensembleConfigs = [];
let runStateTimer = null;

function setRunState(isLoading, activeCount = 0) {
  if (runStateEl) {
    runStateEl.classList.toggle('loading', !!isLoading);
    runStateEl.classList.toggle('idle', !isLoading);
  }

  if (runStateTextEl) {
    runStateTextEl.textContent = isLoading
      ? `Classifying (${activeCount})`
      : 'Idle';
  }

  if (statusLogoEl && ext.runtime && typeof ext.runtime.getURL === 'function') {
    const nextIcon = isLoading ? 'icons/icon-loading.svg' : 'icons/icon.svg';
    statusLogoEl.src = ext.runtime.getURL(nextIcon);
  }
}

async function refreshRunState() {
  try {
    const resp = await ext.runtime.sendMessage({ type: 'GET_ACTIVITY_STATUS' });
    if (!resp || resp.ok !== true) return;
    setRunState(!!resp.isLoading, Number(resp.activeCount) || 0);
  } catch {
    // Ignore transient messaging failures while reloading extension pages.
  }
}

function updateDisplays() {
  classificationDelayValue.textContent = classificationDelayInput.value;
  imageCaptureQualityValue.textContent = parseFloat(imageCaptureQualityInput.value).toFixed(2);
}

function updateSliderBackground(slider) {
  const min = parseFloat(slider.min) || 0;
  const max = parseFloat(slider.max) || 100;
  const value = parseFloat(slider.value) || 0;
  const percent = ((value - min) / (max - min)) * 100;
  slider.style.background = `linear-gradient(to right, #ff4400 0%, #ff4400 ${percent}%, #260a00 ${percent}%, #260a00 100%)`;
}

function hexToRgb(hex) {
  if (!hex || typeof hex !== 'string') return { r: 255, g: 0, b: 100 };
  const clean = hex.startsWith('#') ? hex.slice(1) : hex;
  if (clean.length !== 6) return { r: 255, g: 0, b: 100 };
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return { r: 255, g: 0, b: 100 };
  return { r, g, b };
}

function rgbToHex({ r, g, b }) {
  const clamp = (n) => Math.max(0, Math.min(255, Math.round(n)));
  return `#${clamp(r).toString(16).padStart(2, '0')}${clamp(g).toString(16).padStart(2, '0')}${clamp(b).toString(16).padStart(2, '0')}`.toUpperCase();
}

function createModeSelect(labelText, value, onChange) {
  const wrapper = document.createElement('label');
  wrapper.textContent = `${labelText}: `;

  const select = document.createElement('select');
  const options = [
    { value: 'off', label: 'Off' },
    { value: 'hover', label: 'Hover' },
    { value: 'always', label: 'Always' }
  ];
  options.forEach((opt) => {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.label;
    select.appendChild(option);
  });
  select.value = value || 'hover';
  select.addEventListener('change', () => onChange(select.value));

  wrapper.appendChild(select);
  return wrapper;
}

function createColorSliders(labelText, initialColor, onChange) {
  const wrapper = document.createElement('div');
  wrapper.className = 'color-sliders-wrapper';

  const label = document.createElement('span');
  label.textContent = labelText;
  wrapper.appendChild(label);

  const slidersContainer = document.createElement('div');
  slidersContainer.className = 'sliders-container';

  const rgb = hexToRgb(initialColor);
  const preview = document.createElement('div');
  preview.className = 'color-preview';
  preview.style.backgroundColor = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
  const makeSlider = (name, value) => {
    const container = document.createElement('label');
    container.textContent = `${name}: `;
    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = '255';
    input.value = String(value);
    
    // Set gradient background based on color channel
    if (name === 'R') {
      input.style.background = `linear-gradient(to right, rgb(0, ${rgb.g}, ${rgb.b}), rgb(255, ${rgb.g}, ${rgb.b}))`;
    } else if (name === 'G') {
      input.style.background = `linear-gradient(to right, rgb(${rgb.r}, 0, ${rgb.b}), rgb(${rgb.r}, 255, ${rgb.b}))`;
    } else if (name === 'B') {
      input.style.background = `linear-gradient(to right, rgb(${rgb.r}, ${rgb.g}, 0), rgb(${rgb.r}, ${rgb.g}, 255))`;
    }
    
    const valueEl = document.createElement('span');
    valueEl.textContent = String(value);
    input.addEventListener('input', () => {
      valueEl.textContent = input.value;
      const updated = {
        r: parseInt(rInput.value, 10) || 0,
        g: parseInt(gInput.value, 10) || 0,
        b: parseInt(bInput.value, 10) || 0
      };
      
      // Update slider backgrounds to reflect current RGB values
      rInput.style.background = `linear-gradient(to right, rgb(0, ${updated.g}, ${updated.b}), rgb(255, ${updated.g}, ${updated.b}))`;
      gInput.style.background = `linear-gradient(to right, rgb(${updated.r}, 0, ${updated.b}), rgb(${updated.r}, 255, ${updated.b}))`;
      bInput.style.background = `linear-gradient(to right, rgb(${updated.r}, ${updated.g}, 0), rgb(${updated.r}, ${updated.g}, 255))`;
      
      preview.style.backgroundColor = `rgb(${updated.r}, ${updated.g}, ${updated.b})`;
      onChange(rgbToHex(updated));
    });
    container.appendChild(input);
    container.appendChild(valueEl);
    return { container, input };
  };

  const { container: rContainer, input: rInput } = makeSlider('R', rgb.r);
  const { container: gContainer, input: gInput } = makeSlider('G', rgb.g);
  const { container: bContainer, input: bInput } = makeSlider('B', rgb.b);

  slidersContainer.appendChild(rContainer);
  slidersContainer.appendChild(gContainer);
  slidersContainer.appendChild(bContainer);
  
  wrapper.appendChild(slidersContainer);
  wrapper.appendChild(preview);

  return wrapper;
}

function generateEnsembleId(modality) {
  const prefix = modality === 'text' ? 'text' : 'image';
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
}

function buildDefaultEnsemblesFromSettings() {
  const defaultColor = '#FF0064';
  return [
    {
      id: 'ai_images',
      name: 'AI Images',
      modality: 'image',
      enabled: true,
      threshold: 0.5,
      classifiers: ['res_net50_fft'],
      weights: null,
      styles: {
        image: {
          applyBlur: true,
          blurAmount: 4,
          applyBorder: true,
          borderMultiplier: 1,
          borderColor: defaultColor,
          applyBadge: true,
          blurMode: 'hover',
          borderMode: 'hover',
          badgeMode: 'hover'
        }
      }
    },
    {
      id: 'ai_text',
      name: 'AI Text',
      modality: 'text',
      enabled: true,
      threshold: 0.5,
      classifiers: ['text'],
      weights: null,
      styles: {
        text: {
          applyBlur: true,
          blurAmount: 2,
          applyStrikethrough: true,
          strikethroughColor: defaultColor,
          applyUnderline: false,
          underlineColor: defaultColor,
          applyHighlight: false,
          highlightColor: '#fff3a1'
        }
      }
    }
  ];
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

function renderEnsembles() {
  if (!ensembleListEl) return;
  ensembleListEl.innerHTML = '';

  const modalitiesById = new Map();
  availableClassifiers.forEach((m) => {
    modalitiesById.set(m.id, m.modalities || []);
  });

  ensembleConfigs.forEach((ensemble, index) => {
    if (!ensemble.id) {
      ensemble.id = generateEnsembleId(ensemble.modality || 'image');
      saveEnsembles();
    }
    const card = document.createElement('div');
    card.className = 'ensemble-card';

    const header = document.createElement('div');
    header.className = 'ensemble-row';

    const title = document.createElement('strong');
    title.textContent = `Category ${index + 1}`;

    const modalitySelect = document.createElement('select');
    ['image', 'text'].forEach((mod) => {
      const opt = document.createElement('option');
      opt.value = mod;
      opt.textContent = mod;
      modalitySelect.appendChild(opt);
    });
    modalitySelect.value = ensemble.modality || 'image';
    modalitySelect.addEventListener('change', () => {
      ensemble.modality = modalitySelect.value;
      renderEnsembles();
      saveEnsembles();
    });

    const enabledCheckbox = document.createElement('input');
    enabledCheckbox.type = 'checkbox';
    enabledCheckbox.checked = ensemble.enabled !== false;
    enabledCheckbox.addEventListener('change', () => {
      ensemble.enabled = enabledCheckbox.checked;
      saveEnsembles();
    });

    const enabledLabel = document.createElement('label');
    enabledLabel.textContent = 'Enabled';
    enabledLabel.prepend(enabledCheckbox);

    header.appendChild(title);
    header.appendChild(modalitySelect);
    header.appendChild(enabledLabel);
    card.appendChild(header);

    const thresholdRow = document.createElement('div');
    thresholdRow.className = 'ensemble-row';
    const thresholdInput = document.createElement('input');
    thresholdInput.type = 'range';
    thresholdInput.min = '0';
    thresholdInput.max = '1';
    thresholdInput.step = '0.01';
    thresholdInput.value = typeof ensemble.threshold === 'number' ? ensemble.threshold : 0.5;
    updateSliderBackground(thresholdInput);
    const thresholdValue = document.createElement('span');
    thresholdValue.textContent = Math.round(parseFloat(thresholdInput.value) * 100);
    thresholdInput.addEventListener('input', () => {
      ensemble.threshold = parseFloat(thresholdInput.value);
      thresholdValue.textContent = Math.round(ensemble.threshold * 100);
      updateSliderBackground(thresholdInput);
      saveEnsembles();
    });
    thresholdRow.appendChild(document.createTextNode('Threshold: '));
    thresholdRow.appendChild(thresholdInput);
    thresholdRow.appendChild(thresholdValue);
    thresholdRow.appendChild(document.createTextNode('%'));
    card.appendChild(thresholdRow);

    const classifiersTitle = document.createElement('p');
    classifiersTitle.className = 'hint';
    classifiersTitle.textContent = 'Select classifiers:';
    card.appendChild(classifiersTitle);

    const classifierList = document.createElement('div');
    classifierList.className = 'classifier-list';

    const supported = ensemble.modality || 'image';
    availableClassifiers.forEach((model) => {
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      const isSelected = Array.isArray(ensemble.classifiers) && ensemble.classifiers.includes(model.id);
      checkbox.checked = isSelected;
      const modalities = model.modalities || [];
      checkbox.disabled = modalities.length > 0 && !modalities.includes(supported);
      checkbox.addEventListener('change', () => {
        const list = new Set(Array.isArray(ensemble.classifiers) ? ensemble.classifiers : []);
        if (checkbox.checked) list.add(model.id);
        else list.delete(model.id);
        ensemble.classifiers = Array.from(list);
        saveEnsembles();
      });
      label.appendChild(checkbox);
      label.appendChild(document.createTextNode(` ${model.name || model.id}`));
      classifierList.appendChild(label);
    });
    card.appendChild(classifierList);

    const styleGrid = document.createElement('div');
    styleGrid.className = 'ensemble-grid';

    if ((ensemble.modality || 'image') === 'image') {
      const imageStyles = ensemble.styles?.image || {};

      const blurMode = imageStyles.blurMode || (imageStyles.applyBlur === false ? 'off' : 'hover');
      const blurModeSelect = createModeSelect('Blur', blurMode, (value) => {
        ensemble.styles = ensemble.styles || {};
        ensemble.styles.image = ensemble.styles.image || {};
        ensemble.styles.image.blurMode = value;
        saveEnsembles();
      });
      const blurRange = document.createElement('input');
      blurRange.type = 'range';
      blurRange.min = '0';
      blurRange.max = '20';
      blurRange.value = imageStyles.blurAmount ?? 4;
      updateSliderBackground(blurRange);
      blurRange.addEventListener('input', () => {
        ensemble.styles = ensemble.styles || {};
        ensemble.styles.image = ensemble.styles.image || {};
        ensemble.styles.image.blurAmount = parseFloat(blurRange.value) || 0;
        updateSliderBackground(blurRange);
        saveEnsembles();
      });
      styleGrid.appendChild(blurModeSelect);
      styleGrid.appendChild(blurRange);

      const borderMode = imageStyles.borderMode || (imageStyles.applyBorder === false ? 'off' : 'hover');
      const borderModeSelect = createModeSelect('Border', borderMode, (value) => {
        ensemble.styles = ensemble.styles || {};
        ensemble.styles.image = ensemble.styles.image || {};
        ensemble.styles.image.borderMode = value;
        saveEnsembles();
      });
      const borderMultiplier = document.createElement('input');
      borderMultiplier.type = 'range';
      borderMultiplier.min = '0';
      borderMultiplier.max = '5';
      borderMultiplier.step = '0.1';
      borderMultiplier.value = imageStyles.borderMultiplier ?? 1;
      updateSliderBackground(borderMultiplier);
      borderMultiplier.addEventListener('input', () => {
        ensemble.styles = ensemble.styles || {};
        ensemble.styles.image = ensemble.styles.image || {};
        ensemble.styles.image.borderMultiplier = parseFloat(borderMultiplier.value) || 1;
        updateSliderBackground(borderMultiplier);
        saveEnsembles();
      });
      styleGrid.appendChild(borderModeSelect);
      styleGrid.appendChild(borderMultiplier);

      const borderColorSliders = createColorSliders(
        'Border color',
        imageStyles.borderColor || '#ff0064',
        (value) => {
          ensemble.styles = ensemble.styles || {};
          ensemble.styles.image = ensemble.styles.image || {};
          ensemble.styles.image.borderColor = value;
          saveEnsembles();
        }
      );
      styleGrid.appendChild(borderColorSliders);

      const badgeMode = imageStyles.badgeMode || (imageStyles.applyBadge === false ? 'off' : 'hover');
      const badgeModeSelect = createModeSelect('Badge', badgeMode, (value) => {
        ensemble.styles = ensemble.styles || {};
        ensemble.styles.image = ensemble.styles.image || {};
        ensemble.styles.image.badgeMode = value;
        saveEnsembles();
      });
      styleGrid.appendChild(badgeModeSelect);
    } else {
      const textStyles = ensemble.styles?.text || {};

      const blurMode = textStyles.blurMode || (textStyles.applyBlur === false ? 'off' : 'hover');
      const blurModeSelect = createModeSelect('Blur', blurMode, (value) => {
        ensemble.styles = ensemble.styles || {};
        ensemble.styles.text = ensemble.styles.text || {};
        ensemble.styles.text.blurMode = value;
        saveEnsembles();
      });
      const blurRange = document.createElement('input');
      blurRange.type = 'range';
      blurRange.min = '0';
      blurRange.max = '10';
      blurRange.step = '0.5';
      blurRange.value = textStyles.blurAmount ?? 2;
      updateSliderBackground(blurRange);
      blurRange.addEventListener('input', () => {
        ensemble.styles = ensemble.styles || {};
        ensemble.styles.text = ensemble.styles.text || {};
        ensemble.styles.text.blurAmount = parseFloat(blurRange.value) || 0;
        updateSliderBackground(blurRange);
        saveEnsembles();
      });
      styleGrid.appendChild(blurModeSelect);
      styleGrid.appendChild(blurRange);

      const strikeMode = textStyles.strikethroughMode || (textStyles.applyStrikethrough === false ? 'off' : 'hover');
      const strikeModeSelect = createModeSelect('Strikethrough', strikeMode, (value) => {
        ensemble.styles = ensemble.styles || {};
        ensemble.styles.text = ensemble.styles.text || {};
        ensemble.styles.text.strikethroughMode = value;
        saveEnsembles();
      });
      styleGrid.appendChild(strikeModeSelect);

      const strikeColorSliders = createColorSliders(
        'Strike color',
        textStyles.strikethroughColor || '#ff0064',
        (value) => {
          ensemble.styles = ensemble.styles || {};
          ensemble.styles.text = ensemble.styles.text || {};
          ensemble.styles.text.strikethroughColor = value;
          saveEnsembles();
        }
      );
      styleGrid.appendChild(strikeColorSliders);

      const underlineMode = textStyles.underlineMode || (textStyles.applyUnderline === true ? 'hover' : 'off');
      const underlineModeSelect = createModeSelect('Underline', underlineMode, (value) => {
        ensemble.styles = ensemble.styles || {};
        ensemble.styles.text = ensemble.styles.text || {};
        ensemble.styles.text.underlineMode = value;
        saveEnsembles();
      });
      styleGrid.appendChild(underlineModeSelect);

      const underlineColorSliders = createColorSliders(
        'Underline color',
        textStyles.underlineColor || '#ff0064',
        (value) => {
          ensemble.styles = ensemble.styles || {};
          ensemble.styles.text = ensemble.styles.text || {};
          ensemble.styles.text.underlineColor = value;
          saveEnsembles();
        }
      );
      styleGrid.appendChild(underlineColorSliders);

      const highlightMode = textStyles.highlightMode || (textStyles.applyHighlight === true ? 'hover' : 'off');
      const highlightModeSelect = createModeSelect('Highlight', highlightMode, (value) => {
        ensemble.styles = ensemble.styles || {};
        ensemble.styles.text = ensemble.styles.text || {};
        ensemble.styles.text.highlightMode = value;
        saveEnsembles();
      });
      styleGrid.appendChild(highlightModeSelect);

      const highlightColorSliders = createColorSliders(
        'Highlight color',
        textStyles.highlightColor || '#fff3a1',
        (value) => {
          ensemble.styles = ensemble.styles || {};
          ensemble.styles.text = ensemble.styles.text || {};
          ensemble.styles.text.highlightColor = value;
          saveEnsembles();
        }
      );
      styleGrid.appendChild(highlightColorSliders);
    }

    card.appendChild(styleGrid);

    const actions = document.createElement('div');
    actions.className = 'ensemble-actions';
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.textContent = 'Remove';
    removeBtn.addEventListener('click', () => {
      ensembleConfigs = ensembleConfigs.filter((_, idx) => idx !== index);
      renderEnsembles();
      saveEnsembles();
    });
    actions.appendChild(removeBtn);
    card.appendChild(actions);

    ensembleListEl.appendChild(card);
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
    const res = storageArea && storageArea.get ? await storageArea.get([
      'enabled',
      'altTextOnly',
      'classificationDelay',
      'imageCaptureQuality',
      'verboseLogs',
      'lazyLoad',
      'minTextLength',
      'ensembleConfigsV2'
    ]) : {};
    if (res && typeof res.enabled === 'boolean') enabledInput.checked = res.enabled;
    if (res && typeof res.altTextOnly === 'boolean') altTextOnlyInput.checked = res.altTextOnly;
    if (res && typeof res.classificationDelay === 'number') classificationDelayInput.value = res.classificationDelay;
    if (res && typeof res.imageCaptureQuality === 'number') imageCaptureQualityInput.value = res.imageCaptureQuality;
    if (res && typeof res.verboseLogs === 'boolean') verboseLogsInput.checked = res.verboseLogs;
    if (res && typeof res.lazyLoad === 'boolean') lazyLoadInput.checked = res.lazyLoad;
    if (res && typeof res.minTextLength === 'number') minTextLengthInput.value = res.minTextLength;
    if (res && Array.isArray(res.ensembleConfigsV2) && res.ensembleConfigsV2.length) {
      ensembleConfigs = res.ensembleConfigsV2;
    } else {
      ensembleConfigs = buildDefaultEnsemblesFromSettings();
      await saveEnsembles();
    }
    updateDisplays();
    renderEnsembles();
  } catch {}
}

// Auto-save individual settings immediately when they change
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

classificationDelayInput.addEventListener('input', async () => {
  updateDisplays();
  const value = Math.max(0, Math.min(500, parseInt(classificationDelayInput.value, 10) || 100));
  await saveSetting('classificationDelay', value);
});

imageCaptureQualityInput.addEventListener('input', async () => {
  updateDisplays();
  const value = Math.max(0.2, Math.min(1, parseFloat(imageCaptureQualityInput.value) || 1));
  await saveSetting('imageCaptureQuality', value);
});

// Auto-save checkboxes
enabledInput.addEventListener('change', async () => {
  await saveSetting('enabled', enabledInput.checked);
});

altTextOnlyInput.addEventListener('change', async () => {
  await saveSetting('altTextOnly', altTextOnlyInput.checked);
});

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

load();
refreshRunState();
runStateTimer = setInterval(refreshRunState, 800);

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
    renderEnsembles();
    await saveEnsembles();
  });
}

const reclassifyBtn = document.getElementById('reclassify-all');
if (reclassifyBtn) {
  reclassifyBtn.addEventListener('click', async () => {
    try {
      reclassifyBtn.disabled = true;
      reclassifyBtn.textContent = 'Reclassifying...';
      
      // Send message to all tabs to reclassify
      const tabs = await ext.tabs.query({});
      await Promise.all(
        tabs.map(tab => 
          ext.tabs.sendMessage(tab.id, { type: 'RECLASSIFY_ALL' })
            .catch(() => {}) // Ignore tabs that don't have content script
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

// ============ Model List Logic ============

function setModelStatus(text, type = 'info') {
  modelStatusEl.textContent = text;
  modelStatusEl.hidden = !text;
  modelStatusEl.style.color = type === 'error' ? '#d32f2f' : '#666';
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
      : models.map((id) => ({ id, name: id, modalities: [] }));
    renderEnsembles();
    setModelStatus(models.length ? `${models.length} model(s) loaded` : 'No models found');
  } catch (err) {
    console.warn('LIST_MODELS failed', err);
    availableClassifiers = [];
    renderEnsembles();
    setModelStatus(err.message || 'Failed to list models', 'error');
  }
}
refreshBtn.addEventListener('click', () => fetchModels());

// Load models and selection on page open
(async () => {
  await fetchModels();
})();
