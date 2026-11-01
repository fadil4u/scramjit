// The live demo: four simulations in plain Python (in Pyodide, in a worker); turn scramjit on and
// the same function is compiled to WebAssembly (syntax tree -> types -> C -> cimo) and runs much
// faster. three.js draws the terrain; a 2D canvas draws the fractal, the sand and the flock.
(function () {
  'use strict';
  const DEMOS = {
    terrain: { presets: { erosion: 'Hydraulic erosion', beacon: 'Beacon (viewshed)', landslides: 'Landslides' }, draw: 'terrain', reset: 'New terrain' },
    fractal: { presets: { fractal: 'Mandelbrot dive' }, draw: 'image', fit: 'cover', smooth: true, reset: 'New dive' },
    sand: { presets: { sand: 'Falling sand' }, draw: 'image', fit: 'contain', smooth: false, reset: 'Clear' },
    flock: { presets: { flock: 'Boids' }, draw: 'flock', fit: 'cover', reset: 'Scatter' },
  };
  const PRESETS = Object.assign({}, ...Object.values(DEMOS).map(d => d.presets));
  const DEMO_OF = {};
  for (const [d, v] of Object.entries(DEMOS)) for (const p in v.presets) DEMO_OF[p] = d;
  const ENTRY = { erosion: 'erode', beacon: 'viewshed', landslides: 'slide', fractal: 'mandelbrot', sand: 'sand', flock: 'flock' };
  const UNITS = { erosion: 'droplets/s', beacon: 'sightlines/s', landslides: 'cells/s', fractal: 'iterations/s', sand: 'cells/s', flock: 'boid pairs/s' };
  const RESET_ICON = {
    terrain: '<path d="M2 20 9 8l4 6.5"/><path d="m11 17 5-8 6 11H2"/><path d="m7.4 10.7 1.6 1.3 1.4-1.2"/>',
    fractal: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1.2"/>',
    sand: '<path d="M6 3h12M6 21h12M7 3c0 4.5 4 6 5 9 1-3 5-4.5 5-9M7 21c0-4.5 4-6 5-9 1 3 5 4.5 5 9"/>',
    flock: '<path d="M2 11c2.5-2.5 5-2.5 7 0 2-2.5 4.5-2.5 7 0M10 17c1.6-1.6 3.2-1.6 4.5 0 1.3-1.6 2.9-1.6 4.5 0"/>',
  };
  const root = document.getElementById('lab-demo');
  if (!root) return;
  const $ = s => root.querySelector(s);
  const RM = matchMedia('(prefers-reduced-motion:reduce)').matches;

  let lab = null, editor = null, viewer = null, demo = 'terrain', preset = 'erosion', compiled = false, running = false;
  let sources = {}, info = {}, stage = 'python', pyRate = {}, rate = 0, editTimer = null, presetsLoaded = null;
  let nudged = false, startedAt = 0;

  // ---- the Python side runs in a worker (lab/worker.js), so compiling never freezes the page
  let worker = null, codeId = 0, pending = null, frameIn = null, watWaiters = [];
  function status(m) { $('.lab-status').textContent = m; }
  async function text(path) {
    const r = await fetch(path, { cache: 'no-cache' });
    if (!r.ok) throw new Error(path + ': ' + r.status);
    return r.text();
  }
  function boot() {
    root.classList.add('loading');
    status('Starting…');
    worker = new Worker('lab/worker.js?v=5');
    worker.onmessage = e => {
      const m = e.data;
      if (m.type === 'status') status(m.msg);
      else if (m.type === 'fatal') { status('Could not start: ' + m.msg); root.classList.remove('loading'); }
      else if (m.type === 'ready') presetsLoaded.then(() => {
        lab = true;
        setCode();                        // what's in the editor, edits made before loading included
        root.classList.remove('loading');
        root.classList.add('running');
        running = true;
        startedAt = performance.now();
      });
      else if (m.type === 'frame') { if (m.preset === preset) frameIn = m; else worker.postMessage({ type: 'ack' }); }
      else if (m.type === 'info' && pending && m.id === pending.id) { const k = pending.done; pending = null; k(m.info); }
      else if (m.type === 'wat') watWaiters.splice(0).forEach(k => k(m.text));
      else if (m.type === 'loop-error') { console.error('demo worker:', m.msg); info = { state: 'error', error: m.msg }; finish(); }
    };
    worker.onerror = e => { status('Could not start: ' + (e.message || 'worker error')); root.classList.remove('loading'); };
    worker.postMessage({ type: 'boot', base: new URL('.', location.href).href });
  }
  // send code to the worker; done(info) runs with the reply for the latest code only
  function sendCode(src, on, done) {
    pending = { id: ++codeId, done };
    worker.postMessage({ type: 'code', id: codeId, preset, src, compiled: on });
  }
  // a progress bar while scramjit compiles, paced by the last compile's time
  let lastCompile = 1600, barTimer = null;
  function progress(on) {
    const bar = $('.lab-progress'), fill = bar.firstElementChild;
    clearTimeout(barTimer);
    if (on) {
      bar.classList.add('on');
      fill.style.transition = 'none'; fill.style.width = '0%'; fill.offsetWidth;
      fill.style.transition = `width ${lastCompile * 1.1}ms cubic-bezier(.15,.6,.3,1)`;
      fill.style.width = '92%';
    } else {
      fill.style.transition = 'width .18s ease-out'; fill.style.width = '100%';
      barTimer = setTimeout(() => bar.classList.remove('on'), 260);
    }
  }

  // ---- code and the pipeline. One pane: the Python editor; with scramjit on, tabs for the syntax
  // tree with its inferred types, the generated C and the WebAssembly.
  function showStage(s) {
    stage = s;
    root.querySelectorAll('.lab-tab').forEach(c => c.classList.toggle('open', c.dataset.s === s));
    $('.lab-code').hidden = s !== 'python';
    $('.lab-panel').hidden = s !== 'c' && s !== 'wasm';
    $('.lab-tree').hidden = s !== 'tree';
    if (s === 'python') { if (editor) editor.refresh(); return; }
    if (s === 'tree') { renderTree(); return; }
    if (info.state === 'error') return view(info.error, null);
    if (info.state === 'fallback') return view('scramjit runs this as Python:\n\n' + info.reason, null);
    if (s === 'c') return view(info.c || '', 'text/x-csrc');
    if (!info.wat) {                                    // disassemble when asked, then keep it
      view(';; disassembling…', 'wast');
      setTimeout(() => {
        const forInfo = info;
        watWaiters.push(t => { forInfo.wat = t; if (stage === 'wasm' && info === forInfo) view(t, 'wast'); });
        worker.postMessage({ type: 'wasm' });
      }, 0);
    } else view(info.wat, 'wast');
  }
  // the read-only viewer for the C and the WebAssembly text; kept as is (and scrolled) if unchanged
  function view(text, mode) {
    if (viewer.getValue() !== text || viewer.getOption('mode') !== mode) {
      viewer.setOption('mode', mode);
      viewer.setValue(text);
      viewer.scrollTo(0, 0);
    }
    viewer.refresh();
  }

  const esc = t => String(t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const KIND = { FunctionDef: 'def', For: 'loop', While: 'loop', If: 'cond', Else: 'cond' };
  function tyClass(t) { return t.includes('[') ? 'arr' : /float/.test(t) ? 'flt' : /int|bool/.test(t) ? 'int' : ''; }
  function badge(name, t) { return `<span class="lab-ty ${tyClass(t)}">${name ? `<b>${esc(name)}</b>: ` : ''}${esc(t)}</span>`; }
  function exprHTML(e) {
    const kids = e.c ? `<ul>${e.c.map(exprHTML).join('')}</ul>` : '';
    return `<li><div class="lab-row"><span class="lab-k ex">${esc(e.k)}</span><span class="lab-t">${esc(e.t)}</span>${e.ty ? badge('', e.ty) : ''}</div>${kids}</li>`;
  }
  function stmtHTML(n) {
    const x = n.e ? ' x' : '';
    const kids = n.c ? `<ul>${n.c.map(stmtHTML).join('')}</ul>` : '';
    const ex = n.e ? `<div class="lab-ex" hidden><ul>${n.e.map(exprHTML).join('')}</ul></div>` : '';
    return `<li><div class="lab-row${x}"><span class="lab-caret">${n.e ? '▸' : ''}</span><span class="lab-k ${KIND[n.k] || ''}">${esc(n.k)}</span>` +
      `<span class="lab-t">${esc(n.t)}</span>${(n.b || []).map(b => badge(b[0], b[1])).join('')}</div>${ex}${kids}</li>`;
  }
  function renderTree() {
    const el = $('.lab-tree');
    if (info.state !== 'compiled') { el.textContent = info.state === 'error' ? info.error : 'scramjit runs this as Python:\n\n' + (info.reason || ''); return; }
    if (el.dataset.for === info.c) return;              // already drawn for this code
    el.dataset.for = info.c;
    el.innerHTML = `<p class="lab-tree-head">The function's syntax tree, statement by statement, with the type scramjit inferred for each name it binds. Click a statement for its expressions.</p><ul>${stmtHTML(info.tree)}</ul>`;
  }

  function note(t, busy, title) { const n = $('.lab-note'); n.textContent = t; n.title = title || ''; n.classList.toggle('busy', !!busy); }
  // the raster line sweeps down the editor; the C replaces the Python line by line, then the C tab opens
  function sweep() {
    const pane = $('.lab-pane');
    editor.scrollTo(0, 0);
    $('.lab-code').hidden = false; $('.lab-panel').hidden = false; $('.lab-tree').hidden = true;
    view(info.c, 'text/x-csrc');
    viewer.scrollTo(0, 0);
    pane.style.setProperty('--p', 0);
    pane.classList.add('sweep');
    const t0 = performance.now(), d = RM ? 0 : 1500;
    (function step(now) {
      const u = d ? Math.min(1, (now - t0) / d) : 1, e = u < .5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u);
      pane.style.setProperty('--p', e.toFixed(4));
      if (u < 1 && compiled) requestAnimationFrame(step);
      else { pane.classList.remove('sweep'); showStage(compiled ? 'c' : 'python'); }
    })(t0);
  }
  // load the editor's code. Turning scramjit on (intro) compiles, then sweeps the Python into C;
  // edits while it's on recompile in place.
  function setCode(intro) {
    if (!lab) return;
    const src = editor.getValue();
    $('.lab-err').hidden = true;
    if (!compiled) {
      progress(false);
      sendCode(src, false, r => {
        info = r;
        note('');
        finish();
        if (stage !== 'python') showStage('python');
      });
      return;
    }
    note('compiling…', true);                          // the worker compiles; the page keeps moving
    progress(true);
    sendCode(src, true, r => {
      progress(false);
      if (!compiled) return;
      info = r;
      if (info.compile_ms) lastCompile = info.compile_ms;
      const sec = (info.compile_ms / 1000).toFixed(1);
      note(info.state === 'compiled' ? sec + ' s' : '', false, info.state === 'compiled' ? `scramjit and cimo compiled this in ${sec} s, in this page` : '');
      finish();
      if (info.state === 'compiled' && intro) sweep();
      else if (info.state === 'fallback' && intro) showStage('c');
      else showStage(stage);
    });
  }
  function finish() {
    const e = $('.lab-err');
    e.hidden = info.state !== 'error';
    if (info.state === 'error') e.textContent = info.error;
    root.classList.toggle('fast', info.state === 'compiled');
  }

  // ---- 3D terrain
  let renderer, scene, camera, mesh, geom, colors, N = 0, az = 0.6, el = 0.62, drag = null;
  function initGL() {
    const canvas = $('.lab-gl');
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, devicePixelRatio));
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(38, 1, 0.1, 50);
    scene.add(new THREE.HemisphereLight(0xb9d4ff, 0x2a1408, 0.75));
    const sun = new THREE.DirectionalLight(0xffe2c4, 1.1);
    sun.position.set(-1.5, 2.2, 1.2);
    scene.add(sun);
    resize();
    addEventListener('resize', resize);
    canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, az, el }; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      az = drag.az - (e.clientX - drag.x) * 0.01;
      el = Math.max(0.2, Math.min(1.35, drag.el + (e.clientY - drag.y) * 0.006));
    });
    canvas.addEventListener('pointerup', () => { drag = null; });
  }
  function resize() {
    const c = $('.lab-right'), w = c.clientWidth, h = c.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const k = $('.lab-2d'), dpr = Math.min(2, devicePixelRatio);
    k.width = Math.round(w * dpr); k.height = Math.round(h * dpr);
    if (last2d) draw2d(last2d, true);
  }
  function buildMesh(n) {
    if (mesh) {                           // replace the old mesh (the placeholder before Python loads)
      scene.remove(mesh);
      geom.dispose();
      mesh.material.dispose();
    }
    N = n;
    geom = new THREE.PlaneGeometry(2, 2, n - 1, n - 1);
    colors = new Float32Array(n * n * 3);
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    mesh = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ vertexColors: true }));
    mesh.rotation.x = -Math.PI / 2;
    scene.add(mesh);
  }
  const WATER = [0.12, 0.33, 0.72], SHORE = [0.36, 0.5, 0.32], GRASS = [0.3, 0.44, 0.24],
    ROCK = [0.45, 0.4, 0.36], SNOW = [0.92, 0.92, 0.95], HOT = [0.98, 0.5, 0.18], SHADE = [0.04, 0.05, 0.1];
  function mix(a, b, t) { t = Math.max(0, Math.min(1, t)); return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  const LEVEL = 0.1, SCALE = 0.75;
  function updateMesh(h, vis) {
    const n = Math.round(Math.sqrt(h.length));
    if (N !== n) buildMesh(n);
    const pos = geom.attributes.position.array;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const i = y * n + x, v = h[i];
      pos[i * 3 + 2] = Math.max(v, LEVEL) * SCALE;           // water is flat at its level
      const gx = h[y * n + Math.min(n - 1, x + 1)] - h[y * n + Math.max(0, x - 1)];
      const gy = h[Math.min(n - 1, y + 1) * n + x] - h[Math.max(0, y - 1) * n + x];
      const slope = Math.sqrt(gx * gx + gy * gy) * n * 0.6;
      let c;
      if (v < LEVEL) c = mix(WATER, [0.2, 0.45, 0.85], v / LEVEL);
      else {
        c = v < LEVEL + 0.04 ? SHORE : mix(GRASS, ROCK, (v - 0.25) / 0.3);
        c = mix(c, ROCK, slope - 0.4);                        // steep: bare rock
        c = mix(c, SNOW, (v - 0.72) / 0.12 - slope * 0.3);    // high and gentle: snow
      }
      if (vis) c = vis[i] ? mix(c, HOT, 0.7) : mix(c, SHADE, 0.5);
      colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
    }
    geom.attributes.position.needsUpdate = true;
    geom.attributes.color.needsUpdate = true;
    geom.computeVertexNormals();
  }
  // ---- the 2D demos: an image (the fractal, the sand) or the flock, fitted to the view
  let last2d = null;
  const off = document.createElement('canvas'), offc = off.getContext('2d');
  function fitRect(w, h, mode) {                      // where a w x h picture lands on the canvas
    const k = $('.lab-2d'), s = (mode === 'cover' ? Math.max : Math.min)(k.width / w, k.height / h);
    return { s, x: (k.width - w * s) / 2, y: (k.height - h * s) / 2 };
  }
  function draw2d(r, redraw) {
    const k = $('.lab-2d'), ctx = k.getContext('2d'), d = DEMOS[demo];
    last2d = r;
    if (d.draw === 'image') {
      const a = r.arrays.rgba, [h, w] = a.shape;
      if (off.width !== w || off.height !== h) { off.width = w; off.height = h; }
      offc.putImageData(new ImageData(new Uint8ClampedArray(a.data.buffer, a.data.byteOffset, a.data.length), w, h), 0, 0);
      const f = fitRect(w, h, d.fit);
      ctx.fillStyle = '#080a12'; ctx.fillRect(0, 0, k.width, k.height);
      ctx.imageSmoothingEnabled = d.smooth;
      ctx.drawImage(off, f.x, f.y, w * f.s, h * f.s);
    } else if (d.draw === 'flock') {
      const pos = r.arrays.pos.data, vel = r.arrays.vel.data, n = pos.length / 2;
      const f = fitRect(1200, 760, 'cover'), z = Math.max(1, f.s * 1.2);
      ctx.fillStyle = redraw ? '#080a12' : 'rgba(8,10,18,.28)';   // trails: fade the last frame
      ctx.fillRect(0, 0, k.width, k.height);
      const B = 6, paths = Array.from({ length: B }, () => new Path2D());
      for (let i = 0; i < n; i++) {
        const x = f.x + pos[2 * i] * f.s, y = f.y + pos[2 * i + 1] * f.s;
        const vx = vel[2 * i], vy = vel[2 * i + 1], v = Math.hypot(vx, vy) || 1, ux = vx / v, uy = vy / v;
        const b = Math.min(B - 1, Math.floor((Math.atan2(uy, ux) / Math.PI + 1) / 2 * B));
        const p = paths[b];
        p.moveTo(x + ux * 4 * z, y + uy * 4 * z);
        p.lineTo(x - ux * 3 * z - uy * 2 * z, y - uy * 3 * z + ux * 2 * z);
        p.lineTo(x - ux * 3 * z + uy * 2 * z, y - uy * 3 * z - ux * 2 * z);
        p.closePath();
      }
      const COL = ['#3b82f6', '#60a5fa', '#c4b5fd', '#fdba74', '#f97316', '#ef4444'];
      paths.forEach((p, b) => { ctx.fillStyle = COL[b]; ctx.fill(p); });
    }
  }
  // the page's pointer on a 2D demo, in the demo's own coordinates
  function pointerAt(e) {
    const k = $('.lab-2d'), rc = k.getBoundingClientRect(), dpr = k.width / rc.width;
    const px = (e.clientX - rc.left) * dpr, py = (e.clientY - rc.top) * dpr;
    if (demo === 'sand') { const f = fitRect(180, 210, 'contain'); return { x: (px - f.x) / f.s, y: (py - f.y) / f.s }; }
    if (demo === 'flock') { const f = fitRect(1200, 760, 'cover'); return { x: (px - f.x) / f.s / 1200, y: (py - f.y) / f.s / 760 }; }
    const f = fitRect(240, 240, 'cover');
    return { x: (px - f.x) / f.s / 240, y: (py - f.y) / f.s / 240 };
  }
  let tool = 2, painting = false;
  function init2d() {
    const k = $('.lab-2d');
    k.addEventListener('pointerdown', e => {
      if (!running) return;
      const p = pointerAt(e);
      if (demo === 'fractal') worker.postMessage({ type: 'target', x: p.x, y: p.y });
      if (demo === 'sand') { painting = true; k.setPointerCapture(e.pointerId); worker.postMessage({ type: 'paint', x: p.x, y: p.y, r: tool === 1 ? 2 : 5, kind: tool }); }
    });
    k.addEventListener('pointermove', e => {
      if (!running) return;
      const p = pointerAt(e);
      if (demo === 'sand' && painting) worker.postMessage({ type: 'paint', x: p.x, y: p.y, r: tool === 1 ? 2 : 5, kind: tool });
      if (demo === 'flock') worker.postMessage({ type: 'pointer', x: p.x, y: p.y });
    });
    const up = () => { painting = false; if (running && demo === 'flock') worker.postMessage({ type: 'pointer', x: null, y: null }); };
    k.addEventListener('pointerup', up); k.addEventListener('pointercancel', up); k.addEventListener('pointerleave', up);
    root.querySelectorAll('.lab-tool').forEach(b => b.addEventListener('click', () => {
      tool = +b.dataset.kind;
      root.querySelectorAll('.lab-tool').forEach(c => c.setAttribute('aria-pressed', c === b));
    }));
  }

  function draw() {
    if (!drag && !RM) az += 0.0015;
    const r = 3.9;
    camera.position.set(Math.sin(az) * Math.cos(el) * r, Math.sin(el) * r, Math.cos(az) * Math.cos(el) * r);
    camera.lookAt(0, 0.05, 0);
    renderer.render(scene, camera);
  }

  // ---- the frame loop: one time-boxed step of Python or compiled work, then draw
  function frame() {
    requestAnimationFrame(frame);
    const visible = inView && !document.hidden;           // out of sight: no work, no drawing
    if (frameIn && frameIn.preset !== preset) { frameIn = null; worker.postMessage({ type: 'ack' }); }   // from the last demo
    if (running && frameIn && visible) {
      const r = frameIn;                              // the worker's latest step; ask for the next
      frameIn = null;
      worker.postMessage({ type: 'ack' });
      if (r.error) { info = { state: 'error', error: r.error }; finish(); }
      if (r.compiled === compiled) {
        if (r.rate) rate = rate ? rate * 0.85 + r.rate * 0.15 : r.rate;
        if (!compiled && info.state !== 'error') pyRate[preset] = rate;
      }
      if (DEMOS[demo].draw === 'terrain') updateMesh(r.arrays.h.data, r.arrays.vis ? r.arrays.vis.data : null);
      else draw2d(r, false);
      const n = `${Math.round(rate).toLocaleString()} ${UNITS[preset]}`;
      const x = pyRate[preset] ? ` · ${Math.round(rate / pyRate[preset])}× Python` : '';
      $('.lab-rate').innerHTML = info.state === 'compiled' ? `Compiled by scramjit · ${n}${x}`
        : info.state === 'fallback' ? `scramjit fell back to Python (see the C stage) · ${n}`
        : `Plain Python · ${n}`;
      if (!compiled && !nudged && performance.now() - startedAt > 4000) { root.classList.add('nudge'); nudged = true; }
    }
    if (renderer && visible && DEMOS[demo].draw === 'terrain') draw();
  }
  let inView = true;
  const tellActive = () => { if (worker) worker.postMessage({ type: 'active', on: inView && !document.hidden }); };
  new IntersectionObserver(es => { inView = es[es.length - 1].isIntersecting; tellActive(); }).observe(root);   // the newest entry: a batch can hold several
  document.addEventListener('visibilitychange', tellActive);

  // ---- controls
  function setup() {
    const ta = $('textarea');
    editor = CodeMirror.fromTextArea(ta, { mode: 'python', theme: 'sj', lineNumbers: true, indentUnit: 4, viewportMargin: Infinity });
    editor.setValue('# loading the code…');
    viewer = CodeMirror($('.lab-panel'), { theme: 'sj', lineNumbers: true, readOnly: 'nocursor', mode: 'text/x-csrc' });   // no focus, so no phone keyboard
    const names = () => {                               // the tabs are named after the files
      const t = root.querySelectorAll('.lab-tname');
      t[0].textContent = preset + '.py'; t[1].textContent = ENTRY[preset] + '.c'; t[2].textContent = ENTRY[preset] + '.wat';
    };
    const sel = $('.lab-preset');
    // a demo tab: its presets, its canvas, its controls; the running demo switches over
    const setDemo = d => {
      demo = d;
      const v = DEMOS[d];
      root.querySelectorAll('.lab-demo').forEach(b => { b.classList.toggle('open', b.dataset.d === d); b.setAttribute('aria-selected', b.dataset.d === d); });
      sel.innerHTML = '';
      for (const [k, t] of Object.entries(v.presets)) sel.add(new Option(t, k));
      sel.hidden = Object.keys(v.presets).length < 2;
      $('.lab-demo-name').textContent = Object.values(v.presets)[0];
      $('.lab-demo-name').hidden = !sel.hidden;
      $('.lab-reset').innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${RESET_ICON[d]}</svg>${v.reset}`;
      $('.lab-gl').hidden = v.draw !== 'terrain';
      $('.lab-2d').hidden = v.draw === 'terrain';
      $('.lab-2d').dataset.d = d;
      root.dataset.demo = d;
      last2d = null;
      const k = $('.lab-2d'); k.getContext('2d').clearRect(0, 0, k.width, k.height);
      setPreset(Object.keys(v.presets)[0]);
    };
    // every demo starts as plain Python, so its slow speed comes first; the switch and the nudge return
    const startSlow = () => {
      compiled = false;
      $('.lab-sj').setAttribute('aria-checked', false);
      $('.lab-state').textContent = 'off';
      root.classList.remove('sj', 'fast', 'nudge');
      nudged = false;
      startedAt = performance.now();
      const m = $('.lab-mode');
      m.style.animation = 'none'; m.offsetWidth; m.style.animation = '';
      showStage('python');
    };
    const setPreset = p => {
      preset = p; rate = 0; sel.value = p; names();
      startSlow();
      if (sources[p]) editor.setValue(sources[p]);
      if (running) { worker.postMessage({ type: 'reset', preset: p }); clearTimeout(editTimer); setCode(compiled); }
    };
    root.querySelectorAll('.lab-demo').forEach(b => b.addEventListener('click', () => { if (b.dataset.d !== demo) setDemo(b.dataset.d); }));
    presetsLoaded = Promise.all(Object.keys(PRESETS).map(async p => { sources[p] = await text('lab/' + p + '.py'); }))
      .then(() => { if (!running) editor.setValue(sources[preset]); })
      .catch(e => editor.setValue('# could not load the presets: ' + e.message));
    // phones: while typing, the terrain shrinks to a strip and the demo fits above the keyboard
    const narrow = matchMedia('(max-width:960px)');
    const fit = () => {
      if (!root.classList.contains('typing')) return;
      const vv = window.visualViewport, top = document.querySelector('.top'), head = top ? top.offsetHeight : 0;
      root.style.setProperty('--vvh', ((vv ? vv.height : innerHeight) - head - 8) + 'px');
      requestAnimationFrame(() => {
        scrollTo(0, root.getBoundingClientRect().top + scrollY - head - 4);
        resize(); editor.refresh(); editor.scrollIntoView(null, 40);
      });
    };
    editor.on('focus', () => { if (narrow.matches && !root.classList.contains('is-full')) { root.classList.add('typing'); fit(); } });
    editor.on('blur', () => { if (root.classList.contains('typing')) { root.classList.remove('typing'); requestAnimationFrame(() => { resize(); editor.refresh(); }); } });
    if (window.visualViewport) visualViewport.addEventListener('resize', fit);
    editor.on('change', () => { clearTimeout(editTimer); if (running) editTimer = setTimeout(setCode, 700); });
    sel.addEventListener('change', () => setPreset(sel.value));
    const setMode = on => {
      if (on === compiled) return;
      compiled = on; rate = 0;
      $('.lab-sj').setAttribute('aria-checked', on);
      $('.lab-state').textContent = on ? 'on' : 'off';
      root.classList.toggle('sj', on);
      root.classList.remove('nudge'); nudged = true;
      if (!on) showStage('python');
      if (running) setCode(on);
    };
    $('.lab-sj').addEventListener('click', () => setMode(!compiled));
    $('.lab-reset').addEventListener('click', () => { if (running) { worker.postMessage({ type: 'reset', seed: Math.floor(Math.random() * 1e9) }); rate = 0; } });
    root.querySelectorAll('.lab-tab').forEach(c => c.addEventListener('click', () => showStage(c.dataset.s)));
    $('.lab-tree').addEventListener('click', e => {        // a statement opens its expressions
      const row = e.target.closest('.lab-row.x');
      if (!row) return;
      row.classList.toggle('on');
      row.nextElementSibling.hidden = !row.classList.contains('on');
    });
    // full screen: the Fullscreen API where it works on elements, a fixed full-window layout elsewhere
    const relayout = () => { resize(); if (editor) editor.refresh(); if (viewer) viewer.refresh(); };
    const setFull = on => { root.classList.toggle('is-full', on); $('.lab-full').setAttribute('aria-label', on ? 'Exit full screen' : 'Full screen'); requestAnimationFrame(relayout); };
    $('.lab-full').addEventListener('click', () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (root.classList.contains('is-full')) setFull(false);
      else if (root.requestFullscreen) root.requestFullscreen().catch(() => setFull(true));
      else setFull(true);
    });
    document.addEventListener('fullscreenchange', () => setFull(document.fullscreenElement === root));
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && root.classList.contains('is-full') && !document.fullscreenElement) setFull(false); });
    $('.lab-start').addEventListener('click', boot);
    initGL();
    init2d();
    setDemo('terrain');
    // a still terrain before Python loads
    const n = 64, h = new Float32Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const v = 0.5 + 0.25 * Math.sin(x * 0.19) * Math.cos(y * 0.15) + 0.15 * Math.sin((x + y) * 0.07);
      h[y * n + x] = v * v;
    }
    updateMesh(h, null);
    requestAnimationFrame(frame);
  }
  setup();
})();
