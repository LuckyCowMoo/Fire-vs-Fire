/**
 * Shared AI visual effect helpers used by the content script and options preview.
 */
(function initVisualEffects(global) {
  'use strict';

  const textHoverHandlers = new WeakMap();

  function computeBorderWidth(el, multiplier = 1) {
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
    let r = 0;
    let g = 0;
    let b = 0;
    let count = 0;
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

  function resolveCombinedMode(candidates) {
    if (candidates.some((c) => c.mode === 'always')) return 'always';
    if (candidates.some((c) => c.mode === 'hover')) return 'hover';
    return 'off';
  }

  function getOverlayContainer(imgEl) {
    if (!imgEl || !(imgEl instanceof Element)) return null;
    const preferred = imgEl.closest('figure, a, div, article, section');
    if (preferred && preferred.tagName.toLowerCase() !== 'picture') {
      return preferred;
    }
    const parent = imgEl.parentElement;
    if (parent && parent.tagName.toLowerCase() !== 'picture') return parent;
    return null;
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

  function showImageAi(el) {
    if (!el) return;
    el.style.setProperty('filter', 'blur(0px) saturate(1) brightness(1)', 'important');
    el.style.setProperty('animation', 'none', 'important');
    setImageBadgeOpacity(el, '0');
  }

  function hideImageAi(el, blurAmount = 4) {
    if (!el) return;
    const blurPx = blurAmount != null ? blurAmount : 4;
    el.style.setProperty('filter', `blur(${blurPx}px) saturate(1) brightness(0.85)`, 'important');
    el.style.setProperty('animation', 'ai-pulse 2s ease-in-out infinite', 'important');
    setImageBadgeOpacity(el, '1');
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
      delete element.dataset.aiPreview;
    }
  }

  function clearTextStyles(element) {
    if (!element) return;
    const handlers = textHoverHandlers.get(element);
    if (handlers) {
      element.removeEventListener('mouseenter', handlers.enterHandler);
      element.removeEventListener('mouseleave', handlers.leaveHandler);
      textHoverHandlers.delete(element);
    }
    element.classList.remove('ai-detected-text');
    element.style.removeProperty('filter');
    element.style.removeProperty('text-decoration');
    element.style.removeProperty('text-decoration-color');
    element.style.removeProperty('text-decoration-thickness');
    element.style.removeProperty('background-color');
    element.style.removeProperty('padding');
    element.style.removeProperty('cursor');
    element.style.removeProperty('transition');
    if (element.dataset) {
      delete element.dataset.aiTextStyle;
      delete element.dataset.aiTextHoverAttached;
      delete element.dataset.aiPreview;
    }
  }

  function applyImageInteractionState(element, phase, settings = {}) {
    const raw = element?.dataset?.aiImageStyle;
    if (!raw) return;
    let state;
    try {
      state = JSON.parse(raw);
    } catch {
      return;
    }

    const useAlwaysModes = phase === 'hover';
    const showBlur = useAlwaysModes ? state.blurMode === 'always' : state.blurMode !== 'off';
    if (showBlur && state.blurAmount > 0) {
      hideImageAi(element, state.blurAmount);
    } else {
      showImageAi(element);
    }

    const showBorder = useAlwaysModes ? state.borderMode === 'always' : state.borderMode !== 'off';
    const bw = showBorder && state.borderMultiplier > 0
      ? computeBorderWidth(element, state.borderMultiplier)
      : 0;
    if (bw > 0) {
      element.style.setProperty('outline', `${bw}px solid ${state.borderColor}`, 'important');
      element.style.setProperty('outline-offset', `-${bw}px`, 'important');
    } else {
      element.style.setProperty('outline', 'none', 'important');
    }
    element.style.setProperty('box-shadow', 'none', 'important');

    const showBadge = useAlwaysModes ? state.badgeMode === 'always' : state.badgeMode !== 'off';
    setImageBadgeOpacity(element, showBadge ? '1' : '0');
  }

  function applyTextInteractionState(element, phase) {
    const raw = element?.dataset?.aiTextStyle;
    if (!raw) return;
    let state;
    try {
      state = JSON.parse(raw);
    } catch {
      return;
    }

    const useAlwaysModes = phase === 'hover';
    const showBlur = useAlwaysModes ? state.blurMode === 'always' : state.blurMode !== 'off';
    const showStrike = useAlwaysModes ? state.strikeMode === 'always' : state.strikeMode !== 'off';
    const showUnderline = useAlwaysModes ? state.underlineMode === 'always' : state.underlineMode !== 'off';
    const showHighlight = useAlwaysModes ? state.highlightMode === 'always' : state.highlightMode !== 'off';

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
      element.style.setProperty(
        'text-decoration-color',
        showUnderline ? state.underlineColor : state.strikeColor,
        'important'
      );
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
  }

  function applyImageMatchedStyles(element, matched, settings = {}, options = {}) {
    if (!element || !matched.length) {
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

    const blurMode = resolveCombinedMode(blurCandidates);
    const borderMode = resolveCombinedMode(borderCandidates);
    const badgeMode = resolveCombinedMode(badgeCandidates);

    const blurAmounts = blurCandidates.filter((c) => c.mode !== 'off').map((c) => c.amount);
    const blurAmount = blurAmounts.length
      ? blurAmounts.reduce((a, b) => a + b, 0) / blurAmounts.length
      : 0;
    const borderMultipliers = borderCandidates.filter((c) => c.mode !== 'off').map((c) => c.multiplier);
    const borderMultiplier = borderMultipliers.length
      ? borderMultipliers.reduce((a, b) => a + b, 0) / borderMultipliers.length
      : 0;
    const borderColor = averageColors(borderCandidates.filter((c) => c.mode !== 'off').map((c) => c.color))
      || settings.borderColor
      || '#ff0064';

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
      hideImageAi(element, blurAmount);
    } else {
      showImageAi(element);
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
    getImageBadgeIds(element).forEach((badgeId) => {
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
          const badge = createImageBadgeElement(entry, index, borderColor);

          const bid = `ai-badge-${Math.random().toString(36).slice(2, 8)}`;
          badge.id = bid;
          badgeIds.push(bid);
          container.appendChild(badge);

          if (!badge.dataset.aiToggleAttached) {
            badge.dataset.aiToggleAttached = 'true';
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
    }

    if (options.preview) {
      element.dataset.aiPreview = 'true';
    }

    if (typeof options.onApplied === 'function') {
      options.onApplied();
    }
  }

  function applyTextMatchedStyles(element, matched, settings = {}, options = {}) {
    if (!element || !matched.length) {
      clearTextStyles(element);
      return;
    }

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

    const blurMode = resolveCombinedMode(blurCandidates);
    const strikeMode = resolveCombinedMode(strikeCandidates);
    const underlineMode = resolveCombinedMode(underlineCandidates);
    const highlightMode = resolveCombinedMode(highlightCandidates);

    const blurAmounts = blurCandidates.filter((c) => c.mode !== 'off').map((c) => c.amount);
    const blurAmount = blurAmounts.length
      ? blurAmounts.reduce((a, b) => a + b, 0) / blurAmounts.length
      : 0;

    const strikeColor = averageColors(strikeCandidates.filter((c) => c.mode !== 'off').map((c) => c.color))
      || settings.borderColor
      || '#ff0064';
    const underlineColor = averageColors(underlineCandidates.filter((c) => c.mode !== 'off').map((c) => c.color))
      || settings.borderColor
      || '#ff0064';
    const highlightColor = averageColors(highlightCandidates.filter((c) => c.mode !== 'off').map((c) => c.color))
      || null;

    element.classList.add('ai-detected-text');
    const maxScore = Math.max(...matched.map((m) => m.score));
    const textLabel = matched.find((m) => m.displayLabel && m.displayLabel !== 'AI generated')?.displayLabel
      || matched[0]?.displayLabel
      || 'AI-generated text';
    element.setAttribute('data-ai-score', maxScore);
    element.title = `${textLabel} (${(maxScore * 100).toFixed(1)}% confidence)`;

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

    applyTextInteractionState(element, 'rest');

    element.style.setProperty('cursor', 'pointer', 'important');
    element.style.setProperty('transition', 'all 0.2s ease', 'important');

    if (options.preview) {
      element.dataset.aiPreview = 'true';
    }

    if (!element.dataset.aiTextHoverAttached) {
      element.dataset.aiTextHoverAttached = 'true';
      const enterHandler = () => {
        applyTextInteractionState(element, 'hover');
      };
      const leaveHandler = () => {
        if (typeof options.onMouseLeave === 'function') {
          options.onMouseLeave();
        }
      };
      element.addEventListener('mouseenter', enterHandler);
      element.addEventListener('mouseleave', leaveHandler);
      textHoverHandlers.set(element, { enterHandler, leaveHandler });
    }
  }

  function getClassifierBadgeLabel(model) {
    if (!model) return 'AI generated';
    const label = typeof model.label === 'string' ? model.label.trim() : '';
    return label || 'AI generated';
  }

  function findClassifierModel(availableClassifiers, classifierId) {
    if (!Array.isArray(availableClassifiers) || !classifierId) return null;
    return availableClassifiers.find((model) => model && model.id === classifierId) || null;
  }

  function createImageBadgeElement(entry, index, fallbackColor) {
    const badge = document.createElement('span');
    badge.style.position = 'absolute';
    badge.style.top = `${6 + (index * 28)}px`;
    badge.style.right = '6px';
    badge.style.padding = '4px 6px';
    badge.style.color = '#fff';
    badge.style.fontSize = '12px';
    badge.style.fontWeight = '700';
    badge.style.lineHeight = '1.25';
    badge.style.borderRadius = '4px';
    badge.style.zIndex = String(9999 - index);
    badge.style.cursor = 'pointer';
    badge.style.userSelect = 'none';
    badge.style.opacity = '1';
    badge.style.transition = 'opacity 0.2s ease';
    badge.style.display = 'inline-block';
    badge.style.whiteSpace = 'nowrap';
    badge.style.boxSizing = 'border-box';
    badge.className = 'ai-badge';
    badge.textContent = `${entry.label} ${Math.round(entry.score * 100)}%`;
    const badgeRgb = hexToRgb(entry.color || fallbackColor) || { r: 255, g: 0, b: 100 };
    badge.style.background = `rgba(${badgeRgb.r}, ${badgeRgb.g}, ${badgeRgb.b}, 0.9)`;
    return badge;
  }

  function getImageBadgeLabel(element, fallback = 'tag') {
    const badgeIds = getImageBadgeIds(element);
    if (!badgeIds.length) return fallback;
    const badge = document.getElementById(badgeIds[0]);
    if (!badge || !badge.textContent) return fallback;
    const match = badge.textContent.match(/^(.+?)\s+\d+%$/);
    return match ? match[1] : badge.textContent.split(' ')[0];
  }

  function buildPreviewMatchFromEnsemble(ensemble, availableClassifiers = [], score = 0.85) {
    const classifierIds = Array.isArray(ensemble.classifiers) ? ensemble.classifiers : [];
    const ensembleLabel = typeof ensemble.name === 'string' && ensemble.name.trim()
      ? ensemble.name.trim()
      : 'AI generated';

    if (classifierIds.length === 1) {
      const classifierId = classifierIds[0];
      const model = findClassifierModel(availableClassifiers, classifierId);
      const tagLabel = getClassifierBadgeLabel(model);
      return [{
        cfg: ensemble,
        score,
        displayLabel: tagLabel,
        classifiers: {
          [classifierId]: {
            score,
            label: tagLabel
          }
        }
      }];
    }

    const classifierDetails = {};
    classifierIds.forEach((classifierId) => {
      const model = findClassifierModel(availableClassifiers, classifierId);
      classifierDetails[classifierId] = {
        score,
        label: getClassifierBadgeLabel(model)
      };
    });

    return [{
      cfg: ensemble,
      score,
      displayLabel: ensembleLabel,
      classifiers: classifierDetails
    }];
  }

  function applyPreviewFromEnsemble(element, ensemble, settings = {}, availableClassifiers = []) {
    if (!element || !ensemble) return;
    const matched = buildPreviewMatchFromEnsemble(ensemble, availableClassifiers);
    const modality = ensemble.modality || 'image';

    if (modality === 'image') {
      applyImageMatchedStyles(element, matched, settings, { preview: true });
      return;
    }

    applyTextMatchedStyles(element, matched, settings, {
      preview: true,
      onMouseLeave: () => applyPreviewFromEnsemble(element, ensemble, settings, availableClassifiers)
    });
  }

  global.FVFVisualEffects = {
    computeBorderWidth,
    hexToRgb,
    rgbToHex,
    averageColors,
    resolveCombinedMode,
    getOverlayContainer,
    getImageBadgeIds,
    getImageBadgeLabel,
    setImageBadgeOpacity,
    showImageAi,
    hideImageAi,
    clearImageStyles,
    clearTextStyles,
    applyImageInteractionState,
    applyTextInteractionState,
    applyImageMatchedStyles,
    applyTextMatchedStyles,
    buildPreviewMatchFromEnsemble,
    applyPreviewFromEnsemble
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
