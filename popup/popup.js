/* global browser, chrome */
const ext = typeof browser !== 'undefined' ? browser : chrome;

const toggleBtn = document.getElementById('toggle');
const cancelBtn = document.getElementById('cancelBtn');
const statusEl = document.getElementById('status');

// Use PointerEvent instead of MouseEvent to avoid deprecated properties like mozInputSource
toggleBtn.addEventListener('pointerdown', async (ev) => {
  // prevent default to avoid also triggering a click handler elsewhere
  if (ev && typeof ev.preventDefault === 'function') ev.preventDefault();
    console.log('Popup button clicked');
  try {
    const [tab] = await ext.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.id != null) {
      // Ask the content script to list images and display the result
      const res = await ext.tabs.sendMessage(tab.id, { type: 'LIST_IMAGES' });
      console.log('LIST_IMAGES response', res);
      if (res && res.ok) {
        statusEl.textContent = 'Images found: ' + res.count;
        statusEl.style.display = 'block';
        // Also log a concise list of sources for easier export later
        const sources = (res.images || []).map((img) => img.src).filter(Boolean);
        console.groupCollapsed('Image sources (' + sources.length + ')');
        console.log(sources);
        console.groupEnd();
      } else {
        statusEl.textContent = 'Listing failed';
        statusEl.style.display = 'block';
      }
    }
  } catch (e) {
    statusEl.textContent = 'Could not talk to this page. Is this URL permitted?';
    statusEl.style.display = 'block';
    console.warn(e);
  }
});

// Cancel all in-flight classifications
cancelBtn.addEventListener('pointerdown', async (ev) => {
  if (ev && typeof ev.preventDefault === 'function') ev.preventDefault();
  console.log('Cancel button clicked');
  try {
    const [tab] = await ext.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.id != null) {
      await ext.tabs.sendMessage(tab.id, { type: 'CANCEL_ALL_CLASSIFICATIONS' });
      statusEl.textContent = 'All classifications cancelled';
      statusEl.style.display = 'block';
      setTimeout(() => {
        statusEl.style.display = 'none';
      }, 2000);
    }
  } catch (e) {
    statusEl.textContent = 'Could not cancel classifications';
    statusEl.style.display = 'block';
    console.warn(e);
  }
});
