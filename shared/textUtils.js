/**
 * Shared text element discovery helpers for the content script.
 */
(function initTextUtils(global) {
  'use strict';

  const TEXT_TAG_NAMES = ['p', 'li', 'pre', 'blockquote', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
  const TEXT_SELECTORS = TEXT_TAG_NAMES.join(',');

  function isTextTagName(tag) {
    return TEXT_TAG_NAMES.includes(tag);
  }

  function normalizeElementText(el) {
    return (el.textContent || '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function buildElementSelector(el) {
    const tag = el.tagName.toLowerCase();
    const id = el.id ? `#${el.id}` : '';
    const classes = el.classList && el.classList.length ? `.${Array.from(el.classList).join('.')}` : '';
    return tag + id + classes;
  }

  function hasNestedTextCandidate(el) {
    try {
      return Boolean(el.querySelector(`:scope ${TEXT_SELECTORS}`));
    } catch (_) {
      return Boolean(el.querySelector(TEXT_SELECTORS));
    }
  }

  function buildTextSection(el, index, text) {
    const normalized = text || normalizeElementText(el);
    return {
      index,
      selector: buildElementSelector(el),
      tag: el.tagName.toLowerCase(),
      length: normalized.length,
      text: normalized,
      element: el
    };
  }

  global.FVFTextUtils = {
    TEXT_TAG_NAMES,
    TEXT_SELECTORS,
    isTextTagName,
    normalizeElementText,
    buildElementSelector,
    hasNestedTextCandidate,
    buildTextSection
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
