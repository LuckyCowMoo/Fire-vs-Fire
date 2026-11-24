(function initShowcase() {
  'use strict';

  const TOTAL_SLIDES = 10;
  const SLIDE_W = 1920;
  const SLIDE_H = 1080;

  const params = new URLSearchParams(window.location.search);
  const isExport = params.get('export') === '1' || params.get('export') === 'true';
  const exportSlide = Math.max(1, Math.min(TOTAL_SLIDES, parseInt(params.get('slide') || '1', 10) || 1));
  const exportRatio = params.get('ratio') || '16x9';

  const scroller = document.getElementById('viewport-scroller');
  const track = document.getElementById('panorama-track');

  let currentScale = 1;

  function applyScale() {
    if (isExport) {
      document.body.classList.add('export-mode');
      const scale = exportRatio === '16x10' ? (1280 / SLIDE_W) : 1;
      currentScale = scale;
      document.documentElement.style.setProperty('--slide-scale', String(scale));
      if (track) {
        const offsetPx = -(exportSlide - 1) * SLIDE_W;
        track.style.transform = `scale(${scale}) translateX(${offsetPx}px)`;
      }
      return;
    }

    const prevScale = currentScale || 1;
    const prevSlideOffset = scroller ? (scroller.scrollLeft / (SLIDE_W * prevScale)) : 0;

    const scaleY = window.innerHeight / SLIDE_H;
    const scale = scaleY > 0 ? scaleY : 1;
    currentScale = scale;
    document.documentElement.style.setProperty('--slide-scale', String(scale));
    if (track) {
      track.style.transform = `scale(${scale})`;
    }
    if (scroller && prevSlideOffset > 0) {
      scroller.scrollLeft = prevSlideOffset * SLIDE_W * scale;
    }
  }

  function getActiveSlideIndex() {
    if (!scroller) return 1;
    const scale = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--slide-scale')) || 1;
    const slideSpan = SLIDE_W * scale;
    const idx = Math.round(scroller.scrollLeft / slideSpan) + 1;
    return Math.max(1, Math.min(TOTAL_SLIDES, idx));
  }

  function scrollToSlide(slideNum) {
    if (!scroller) return;
    const clamped = Math.max(1, Math.min(TOTAL_SLIDES, slideNum));
    const scale = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--slide-scale')) || 1;
    scroller.scrollTo({
      left: (clamped - 1) * SLIDE_W * scale,
      behavior: 'smooth'
    });
  }

  function bindNavigation() {
    if (isExport || !scroller) return;

    // Translate vertical mouse wheel into smooth horizontal panning unless hovering an inner scrollable list
    scroller.addEventListener(
      'wheel',
      (e) => {
        const innerScroll = e.target.closest('.grab-scroll-list');
        if (innerScroll && Math.abs(e.deltaY) >= Math.abs(e.deltaX)) {
          const atTop = innerScroll.scrollTop <= 0 && e.deltaY < 0;
          const atBottom =
            innerScroll.scrollTop + innerScroll.clientHeight >= innerScroll.scrollHeight - 1 &&
            e.deltaY > 0;
          if (!atTop && !atBottom) {
            return; // Let the native vertical scrollbar inside the extracted list handle it
          }
        }

        if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
          e.preventDefault();
          scroller.scrollLeft += e.deltaY * 1.15;
        }
      },
      { passive: false }
    );

    window.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        scrollToSlide(getActiveSlideIndex() + 1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        scrollToSlide(getActiveSlideIndex() - 1);
      }
    });

    // Optional ?slide=N start jump in live mode
    const initialSlide = parseInt(params.get('slide') || '0', 10);
    if (initialSlide >= 1 && initialSlide <= TOTAL_SLIDES) {
      const scale = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--slide-scale')) || 1;
      scroller.scrollLeft = (initialSlide - 1) * SLIDE_W * scale;
    }
  }

  function bindOptionsDemo() {
    const toggle = document.getElementById('demo-master-toggle');
    const stateText = document.getElementById('demo-run-text');
    const dot = document.querySelector('#demo-run-state .dot');
    if (!toggle || !stateText || !dot) return;

    toggle.addEventListener('change', () => {
      if (toggle.checked) {
        stateText.textContent = 'Connected — Idle';
        dot.style.background = '#2ecc71';
        dot.style.boxShadow = '0 0 8px rgba(46, 204, 113, 0.7)';
      } else {
        stateText.textContent = 'Detection Paused';
        dot.style.background = '#f39c12';
        dot.style.boxShadow = '0 0 8px rgba(243, 156, 18, 0.7)';
      }
    });
  }

  function initEmberCanvas() {
    const canvas = document.getElementById('ember-webgl-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    // Deterministic pseudo-random generator for clean, non-flickering export renders
    let seed = 4219;
    function rand() {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    }

    const embers = Array.from({ length: 34 }, () => ({
      x: rand() * width,
      y: rand() * height,
      vx: 0.12 + rand() * 0.35,
      vy: -0.18 - rand() * 0.42,
      r: 0.9 + rand() * 1.8,
      alpha: 0.14 + rand() * 0.35,
      hue: 14 + rand() * 24
    }));

    function drawEmbers() {
      ctx.clearRect(0, 0, width, height);
      for (const p of embers) {
        if (!isExport) {
          p.x += p.vx;
          p.y += p.vy;
          if (p.x > width + 20) p.x = -10;
          if (p.y < -20) p.y = height + 10;
        }
        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4);
        grad.addColorStop(0, `hsla(${p.hue}, 100%, 68%, ${p.alpha})`);
        grad.addColorStop(0.5, `hsla(${p.hue - 5}, 100%, 52%, ${p.alpha * 0.35})`);
        grad.addColorStop(1, 'hsla(10, 100%, 40%, 0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * 4, 0, Math.PI * 2);
        ctx.fill();
      }
      if (!isExport) {
        requestAnimationFrame(drawEmbers);
      }
    }

    window.addEventListener('resize', () => {
      if (isExport) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      applyScale();
    });

    drawEmbers();
  }

  applyScale();
  bindNavigation();
  bindOptionsDemo();
  initEmberCanvas();
})();
