// Hero: air through a scramjet. Cold free stream (blue) is compressed by the inlet, burns in the
// combustor (orange) and leaves the nozzle hot (red). The stage labels map the engine onto the
// compiler: Python goes in, types and lowering happen in the middle, machine code comes out.
(function () {
  'use strict';
  const canvas = document.querySelector('.flow canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const RM = matchMedia('(prefers-reduced-motion:reduce)').matches;
  const BG = [8, 10, 18];

  // Engine geometry in local units: x from 0 (inlet lip) to 100 (nozzle exit), y from the centreline.
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
  const smooth = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
  function half(u) {               // channel half-height
    if (u < 0) return 14;
    if (u < 35) return lerp(14, 5, smooth(u / 35));
    if (u < 65) return lerp(5, 6, (u - 35) / 30);
    if (u < 100) return lerp(6, 16, smooth((u - 65) / 35));
    return 16 + (u - 100) * 0.22;  // the plume spreads
  }
  function outer(u) {              // hull outer surface
    if (u < 0) return 14;
    if (u < 26) return lerp(14, 21, smooth(u / 26));
    return 21;
  }
  function disp(u) {               // how far the outside flow is pushed away from the hull
    if (u < 0) return 1.5 * Math.max(0, 1 + u / 18);
    if (u <= 100) return outer(u) - 14 + 1.5;
    return 8.5 * Math.exp(-(u - 100) / 30);
  }
  function heat(u) {               // temperature of captured air, 0 cold .. 1 burning
    if (u < 0) return 0;
    if (u < 38) return 0.3 * u / 38;
    if (u < 54) return lerp(0.3, 1, smooth((u - 38) / 16));
    if (u < 66) return 1;
    if (u < 100) return lerp(1, 0.82, (u - 66) / 34);
    return 0.82 * Math.exp(-(u - 100) / 90);
  }
  function speed(u, captured) {
    if (!captured || u < 0) return 62;
    if (u < 38) return lerp(62, 48, u / 38);
    if (u < 66) return 46;
    if (u < 100) return lerp(46, 120, smooth((u - 66) / 34));
    return 120;
  }
  // colour ramp: blue → pale blue → amber → orange → red
  const STOPS = [[0, 56, 115, 255], [0.3, 140, 190, 255], [0.55, 255, 196, 120], [0.75, 249, 115, 22], [1, 239, 50, 50]];
  function color(t) {
    for (let i = 1; i < STOPS.length; i++) {
      if (t <= STOPS[i][0]) {
        const a = STOPS[i - 1], b = STOPS[i], k = (t - a[0]) / (b[0] - a[0]);
        return [lerp(a[1], b[1], k) | 0, lerp(a[2], b[2], k) | 0, lerp(a[3], b[3], k) | 0];
      }
    }
    return STOPS[STOPS.length - 1].slice(1);
  }

  let W = 0, H = 0, dpr = 1, ox = 0, oy = 0, sc = 1, parts = [], narrow = false;
  function layout() {
    const r = canvas.getBoundingClientRect();
    dpr = Math.min(2, devicePixelRatio || 1);
    W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    narrow = W < 900;
    if (narrow) { ox = W * 0.06; sc = W * 0.8 / 100; oy = H * 0.5; }
    else { ox = Math.max(W * 0.47, W / 2 - 590 + 600); sc = Math.min(W - ox - W * 0.1, H * 1.4) / 100; oy = H * 0.56; }
    const n = Math.round(clamp(W * H / 520, 500, 2000));
    parts = Array.from({ length: n }, () => spawn(true));
    ctx.fillStyle = `rgb(${BG})`; ctx.fillRect(0, 0, W, H);
  }
  function spawn(anywhere) {
    const vmax = (H / 2) / sc + 4;
    // bias spawns toward the intake so the engine stays busy
    let v0 = Math.random() < 0.45 ? (Math.random() * 2 - 1) * 13 : (Math.random() * 2 - 1) * vmax;
    const u = anywhere ? -ox / sc + Math.random() * (W / sc + 20) : -ox / sc - Math.random() * 20;
    return { u, v0, cap: Math.abs(v0) < 13, px: null, py: null, j: Math.random() };
  }
  function place(p) {
    let v;
    if (p.cap) v = p.u < 0 ? p.v0 : p.v0 * half(p.u) / 14;
    else v = p.v0 + Math.sign(p.v0) * disp(p.u) * Math.exp(-(Math.abs(p.v0) - 14) / 10);
    return [ox + p.u * sc, oy + v * sc];
  }

  function hullPath(sign) {
    ctx.beginPath();
    for (let u = 0; u <= 100; u += 2) { const x = ox + u * sc, y = oy + sign * half(u) * sc; u ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    for (let u = 100; u >= 0; u -= 2) ctx.lineTo(ox + u * sc, oy + sign * outer(u) * sc);
    ctx.closePath();
  }
  let edge = null;
  function drawHull(t) {
    if (!edge || edge.w !== W) {
      edge = ctx.createLinearGradient(ox, 0, ox + 100 * sc, 0);
      edge.addColorStop(0, 'rgba(96,150,255,.9)'); edge.addColorStop(0.45, 'rgba(160,180,230,.5)');
      edge.addColorStop(0.6, 'rgba(249,115,22,.95)'); edge.addColorStop(1, 'rgba(239,68,68,.95)');
      edge.w = W;
    }
    for (const s of [-1, 1]) {
      hullPath(s);
      ctx.fillStyle = '#0d111d'; ctx.fill();
      ctx.lineWidth = 1.2; ctx.strokeStyle = edge; ctx.stroke();
    }
    // fuel injectors: a row of sparks on both walls of the combustor
    ctx.globalCompositeOperation = 'lighter';
    for (let k = 0; k < 4; k++) {
      const u = 40 + k * 3.2, f = 0.55 + 0.45 * Math.sin(t * 9 + k * 1.7);
      for (const s of [-1, 1]) {
        ctx.fillStyle = `rgba(255,170,90,${0.5 * f})`;
        ctx.beginPath(); ctx.arc(ox + u * sc, oy + s * (half(u) - 0.4) * sc, 1.4 + f, 0, 7); ctx.fill();
      }
    }
    ctx.globalCompositeOperation = 'source-over';
  }
  function drawLabels() {
    ctx.font = '500 11px "DM Mono", ui-monospace, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const y = oy + 25 * sc;
    const L = [[6, 'Python in', '#6f8fd8'], [50, 'types · lowering · C', '#c98a5e'], [104, 'machine code out', '#d06a6a']];
    for (const [u, text, c] of L) {
      const w = ctx.measureText(text).width / 2;
      ctx.fillStyle = c; ctx.fillText(text, clamp(ox + u * sc, w + 8, W - w - 8), y);
    }
  }

  let last = 0, t = 0, running = false;
  function step(dt) {
    t += dt;
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = `rgba(${BG},.24)`; ctx.fillRect(0, 0, W, H);
    // combustor glow
    const gx = ox + 55 * sc, glow = ctx.createRadialGradient(gx, oy, 0, gx, oy, 26 * sc);
    const f = 0.1 + 0.03 * Math.sin(t * 7);
    glow.addColorStop(0, `rgba(249,115,22,${f})`); glow.addColorStop(1, 'rgba(249,115,22,0)');
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = glow; ctx.fillRect(gx - 26 * sc, oy - 26 * sc, 52 * sc, 52 * sc);
    ctx.lineCap = 'round';
    for (let i = 0; i < parts.length; i++) {
      let p = parts[i];
      p.u += speed(p.u, p.cap) * dt * (0.85 + 0.3 * p.j);
      const [x, y] = place(p);
      if (x > W + 10) { parts[i] = spawn(false); continue; }
      if (p.px !== null) {
        const h = p.cap ? heat(p.u) : 0, c = color(h);
        const a = p.cap ? (h > 0.4 ? 0.9 : 0.6) : 0.32;
        ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${a})`;
        ctx.lineWidth = p.cap ? (h > 0.4 ? 1.7 : 1.2) : 1;
        ctx.beginPath(); ctx.moveTo(p.px, p.py); ctx.lineTo(x, y); ctx.stroke();
      }
      p.px = x; p.py = y;
    }
    ctx.globalCompositeOperation = 'source-over';
    drawHull(t);
    drawLabels();
  }
  function frame(now) {
    if (!running) return;
    const dt = Math.min(0.05, (now - last) / 1000 || 0.016); last = now;
    step(dt);
    requestAnimationFrame(frame);
  }
  function start() { if (running || RM) return; running = true; last = performance.now(); requestAnimationFrame(frame); }
  function stop() { running = false; }

  layout();
  if (RM) { for (let i = 0; i < 140; i++) step(0.016); }  // one settled, still frame
  else start();
  let rt = null;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { edge = null; layout(); if (RM) for (let i = 0; i < 140; i++) step(0.016); }, 120); });
  if ('IntersectionObserver' in window)
    new IntersectionObserver(es => { es[0].isIntersecting && !document.hidden ? start() : stop(); }).observe(canvas);
  document.addEventListener('visibilitychange', () => { document.hidden ? stop() : start(); });
})();
