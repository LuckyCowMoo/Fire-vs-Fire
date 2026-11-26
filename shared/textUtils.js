/**
 * Shared text element discovery helpers for the content script.
 */
(function initTextUtils(global) {
  'use strict';

  const TEXT_TAG_NAMES = ['p', 'li', 'pre', 'blockquote', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
  const TEXT_SELECTORS = TEXT_TAG_NAMES.join(',');
  const FEED_BOUNDARY_SELECTOR = [
    '[data-testid="tweet"]',
    '[data-testid="tweetText"]',
    '[data-testid*="comment"]',
    'shreddit-comment',
    'ytd-comment-renderer',
    '[role="listitem"]',
    '.comment',
    '.tweet'
  ].join(',');
  const ALWAYS_ISOLATED_TAGS = new Set(['li', 'td', 'th', 'blockquote', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']);

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

  function getFeedCardRoot(el) {
    if (!el || typeof el.closest !== 'function') return null;
    const explicit = el.closest(FEED_BOUNDARY_SELECTOR);
    if (explicit) return explicit;
    const articleAncestor = el.closest('article, [role="article"]');
    if (articleAncestor) {
      const doc = el.ownerDocument || (typeof document !== 'undefined' ? document : null);
      if (doc && doc.querySelectorAll('article, [role="article"]').length > 1) {
        return articleAncestor;
      }
    }
    return null;
  }

  function isIsolatedFeedElement(el) {
    if (!el || !el.tagName) return true;
    const tag = el.tagName.toLowerCase();
    if (ALWAYS_ISOLATED_TAGS.has(tag)) return true;
    return Boolean(getFeedCardRoot(el));
  }

  function buildTextSection(el, index, text, elements) {
    const normalized = text || normalizeElementText(el);
    const group = Array.isArray(elements) && elements.length ? elements : [el];
    return {
      index,
      selector: buildElementSelector(el),
      tag: el.tagName.toLowerCase(),
      length: normalized.length,
      text: normalized,
      element: group[0],
      elements: group,
      isGrouped: group.length > 1
    };
  }

  function extractCohesiveTextSections(root, minTextLength, isProcessedFn, seenSet, hasResultFn) {
    const scope = root || (typeof document !== 'undefined' ? document : null);
    if (!scope || !scope.querySelectorAll) return [];

    const nodes = [];
    if (typeof Element !== 'undefined' && scope instanceof Element && isTextTagName(scope.tagName.toLowerCase())) {
      nodes.push(scope);
    }
    scope.querySelectorAll(TEXT_SELECTORS).forEach((n) => nodes.push(n));

    const minLen = Number.isFinite(Number(minTextLength)) ? Number(minTextLength) : 250;
    const visited = new Set();
    const sections = [];

    nodes.forEach((el) => {
      if (!el || visited.has(el)) return;
      if (isProcessedFn && isProcessedFn(el)) return;
      if (hasNestedTextCandidate(el)) return;

      const tag = el.tagName ? el.tagName.toLowerCase() : '';
      if (tag === 'p' && !isIsolatedFeedElement(el)) {
        const firstText = normalizeElementText(el);
        const group = [];
        const parts = [];
        let accLen = 0;

        if (firstText) {
          group.push(el);
          parts.push(firstText);
          accLen = firstText.length;
        } else {
          visited.add(el);
          return;
        }

        const firstCached = Boolean(hasResultFn && hasResultFn(firstText.substring(0, 500)));
        let sib = firstCached ? null : el.nextElementSibling;
        while (sib && group.length < 4 && accLen < 1800) {
          if (visited.has(sib) || (isProcessedFn && isProcessedFn(sib))) break;
          const sibTag = sib.tagName ? sib.tagName.toLowerCase() : '';
          if (sibTag !== 'p') break;
          if (hasNestedTextCandidate(sib) || isIsolatedFeedElement(sib)) break;
          const sibText = normalizeElementText(sib);
          if (!sibText) break;
          if (hasResultFn && hasResultFn(sibText.substring(0, 500))) break;
          if (accLen + 2 + sibText.length > 1800 && accLen >= minLen) break;
          group.push(sib);
          parts.push(sibText);
          accLen += 2 + sibText.length;
          sib = sib.nextElementSibling;
        }

        group.forEach((gEl) => visited.add(gEl));
        const joined = parts.join('\n\n');
        if (!joined || joined.length < minLen) return;
        if (seenSet && seenSet.has(joined)) return;
        if (seenSet) seenSet.add(joined);
        sections.push(buildTextSection(group[0], sections.length, joined, group));
        return;
      }

      visited.add(el);
      const text = normalizeElementText(el);
      if (!text || text.length < minLen) return;
      if (seenSet && seenSet.has(text)) return;
      if (seenSet) seenSet.add(text);
      sections.push(buildTextSection(el, sections.length, text, [el]));
    });

    return sections;
  }

  global.FVFTextUtils = {
    TEXT_TAG_NAMES,
    TEXT_SELECTORS,
    isTextTagName,
    normalizeElementText,
    buildElementSelector,
    hasNestedTextCandidate,
    getFeedCardRoot,
    isIsolatedFeedElement,
    buildTextSection,
    extractCohesiveTextSections
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
