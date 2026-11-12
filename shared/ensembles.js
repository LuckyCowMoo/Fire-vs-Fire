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
        id: 'auto_image_fvf_heimdall04_b',
        name: 'FvF Heimdall 04 B',
        modality: 'image',
        enabled: true,
        threshold: settings.imageAiThreshold ?? 0.75,
        classifiers: ['fvf_heimdall04_b'],
        weights: null,
        styles: {
          image: {
            applyBlur: true,
            blurAmount: settings.blurAmount ?? 3,
            applyBorder: false,
            borderMultiplier: settings.borderMultiplier ?? 1,
            borderColor: '#FF0055',
            applyBadge: true,
            blurMode: 'hover',
            borderMode: 'off',
            badgeMode: 'always'
          }
        }
      },
      {
        id: 'auto_image_fvf_heimdall04',
        name: 'FvF Heimdall 04',
        modality: 'image',
        enabled: true,
        threshold: settings.imageAiThreshold ?? 0.75,
        classifiers: ['fvf_heimdall04'],
        weights: null,
        styles: {
          image: {
            applyBlur: true,
            blurAmount: settings.blurAmount ?? 3,
            applyBorder: false,
            borderMultiplier: settings.borderMultiplier ?? 1,
            borderColor: '#FF1744',
            applyBadge: true,
            blurMode: 'hover',
            borderMode: 'off',
            badgeMode: 'always'
          }
        }
      },
      {
        id: 'ai_images_03',
        name: 'FvF Heimdall 03',
        modality: 'image',
        enabled: false,
        threshold: settings.imageAiThreshold ?? 0.75,
        classifiers: ['fvf_heimdall03'],
        weights: null,
        styles: {
          image: {
            applyBlur: true,
            blurAmount: settings.blurAmount ?? 3,
            applyBorder: false,
            borderMultiplier: settings.borderMultiplier ?? 1,
            borderColor,
            applyBadge: true,
            blurMode: 'hover',
            borderMode: 'off',
            badgeMode: 'always'
          }
        }
      },
      {
        id: 'ai_images_02',
        name: 'FvF Heimdall 02',
        modality: 'image',
        enabled: false,
        threshold: settings.imageAiThreshold ?? 0.75,
        classifiers: ['fvf_heimdall02'],
        weights: null,
        styles: {
          image: {
            applyBlur: true,
            blurAmount: settings.blurAmount ?? 3,
            applyBorder: false,
            borderMultiplier: settings.borderMultiplier ?? 1,
            borderColor: '#FF4D1C',
            applyBadge: true,
            blurMode: 'hover',
            borderMode: 'off',
            badgeMode: 'always'
          }
        }
      },
      {
        id: 'ai_text',
        name: 'Mimir 02 B',
        modality: 'text',
        enabled: true,
        threshold: settings.textAiThreshold ?? 0.75,
        classifiers: ['fvf_mimir_02'],
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
