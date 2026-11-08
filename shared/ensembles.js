/**
 * Shared ensemble defaults and weight helpers.
 */
(function initEnsembles(global) {
  'use strict';

  function normalizeEnsembleWeight(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0) return 1;
    return numeric;
  }

  function buildDefaultEnsembles(settings = {}) {
    const borderColor = settings.borderColor || '#FF0064';
    return [
      {
        id: 'ai_images',
        name: 'FvF Heimdall 02',
        modality: 'image',
        enabled: true,
        threshold: settings.imageAiThreshold ?? 0.5,
        classifiers: ['fvf_heimdall02'],
        weights: null,
        styles: {
          image: {
            applyBlur: true,
            blurAmount: settings.blurAmount ?? 4,
            applyBorder: true,
            borderMultiplier: settings.borderMultiplier ?? 1,
            borderColor,
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
        threshold: settings.textAiThreshold ?? 0.5,
        classifiers: ['text'],
        weights: null,
        styles: {
          text: {
            applyBlur: true,
            blurAmount: settings.textBlurAmount ?? 2,
            applyStrikethrough: settings.textStrikethroughEnabled ?? true,
            strikethroughColor: borderColor,
            applyUnderline: false,
            underlineColor: borderColor,
            applyHighlight: false,
            highlightColor: '#fff3a1'
          }
        }
      }
    ];
  }

  global.FVFEnsembles = {
    buildDefaultEnsembles,
    normalizeEnsembleWeight
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
