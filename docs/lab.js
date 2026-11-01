// The live demo: Python (in Pyodide) erodes a terrain; turn scramjit on and the same function is
// compiled to WebAssembly (syntax tree -> types -> C -> cimo) and runs much faster. three.js draws.
(function () {
  'use strict';
  const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
  const PRESETS = { erosion: 'Hydraulic erosion', beacon: 'Beacon (viewshed)', landslides: 'Landslides' };
  const UNITS = { erosion: 'droplets/s', beacon: 'sightlines/s', landslides: 'cells/s' };
  const SJ_FILES = ['__init__.py', '_compiler.py', '_dispatch.py', '_errors.py', '_runtime.py', '_types.py'];
  const root = document.getElementById('lab-demo');
  if (!root) return;
  const $ = s => root.querySelector(s);
  const RM = matchMedia('(prefers-reduced-motion:reduce)').matches;

  let py = null, lab = null, editor = null, preset = 'erosion', compiled = false, running = false;
  let sources = {}, info = {}, stage = 'python', pyRate = {}, rate = 0, editTimer = null, presetsLoaded = null;
  let nudged = false, startedAt = 0;

  // ---- loading
  function script(src) {
    return new Promise((ok, fail) => {
      const s = document.createElement('script');
      s.src = src; s.onload = ok; s.onerror = () => fail(new Error('could not load ' + src));
      document.head.appendChild(s);
    });
  }
  function status(m) { $('.lab-status').textContent = m; }
  async function text(path) {
    const r = await fetch(path, { cache: 'no-cache' });
    if (!r.ok) throw new Error(path + ': ' + r.status);
    return r.text();
  }
  async function boot() {
    root.classList.add('loading');
    status('Loading Pyodide (CPython in WebAssembly, 7 MB)…');
    if (!window.loadPyodide) await script(PYODIDE + 'pyodide.js');
    py = await loadPyodide({ indexURL: PYODIDE });
    status('Loading numpy…');
    await py.loadPackage('numpy');
    status('Loading cimo and scramjit (pure Python)…');
    py.FS.mkdirTree('/home/pyodide/scramjit');
    py.FS.writeFile('/home/pyodide/cimo.py', await text('py/cimo.py'));
    for (const f of SJ_FILES) py.FS.writeFile('/home/pyodide/scramjit/' + f, await text('py/scramjit/' + f));
    py.FS.writeFile('/home/pyodide/lab.py', await text('lab/lab.py'));
    py.FS.writeFile('/home/pyodide/wat.py', await text('lab/wat.py'));
    await presetsLoaded;
    py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
    status('Importing…');
    await py.runPythonAsync('import lab');
    lab = py.pyimport('lab');
    setCode();                            // what's in the editor, edits made before loading included
    root.classList.remove('loading');
    root.classList.add('running');
    running = true;
    startedAt = performance.now();
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
    const pre = $('.lab-panel pre');
    if (info.state === 'error') { pre.textContent = info.error; return; }
    if (info.state === 'fallback') { pre.textContent = 'scramjit runs this as Python:\n\n' + info.reason; return; }
    if (s === 'c') { pre.textContent = info.c || ''; return; }
    if (!info.wat) {                                    // disassemble when asked, then keep it
      pre.textContent = ';; disassembling…';
      setTimeout(() => {
        try { info.wat = lab.wasm_text(); } catch (e) { info.wat = ';; could not disassemble: ' + e.message; }
        if (stage === 'wasm') pre.textContent = info.wat;
      }, 30);
    } else pre.textContent = info.wat;
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

  function note(t, busy) { const n = $('.lab-note'); n.textContent = t; n.classList.toggle('busy', !!busy); }
  // the raster line sweeps down the editor; the C replaces the Python line by line, then the C tab opens
  function sweep() {
    const pane = $('.lab-pane');
    $('.lab-panel pre').textContent = info.c;
    $('.lab-panel pre').scrollTop = 0;
    editor.scrollTo(0, 0);
    $('.lab-code').hidden = false; $('.lab-panel').hidden = false; $('.lab-tree').hidden = true;
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
      info = JSON.parse(lab.set_code(preset, src, false));
      note('');
      finish();
      if (stage !== 'python') showStage('python');
      return;
    }
    note('scramjit is compiling…', true);
    requestAnimationFrame(() => setTimeout(() => {     // compiling blocks the page, so paint first
      if (!compiled) return;
      info = JSON.parse(lab.set_code(preset, src, true));
      note(info.state === 'compiled' ? `compiled in ${(info.compile_ms / 1000).toFixed(1)} s` : info.state === 'fallback' ? 'runs as Python' : '');
      finish();
      if (info.state === 'compiled' && intro) sweep();
      else if (info.state === 'fallback' && intro) showStage('c');
      else showStage(stage);
    }, 30));
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
    const canvas = $('canvas');
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
    const c = $('canvas'), w = c.clientWidth, h = c.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
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
  function draw() {
    if (!drag && !RM) az += 0.0015;
    const r = 3.9;
    camera.position.set(Math.sin(az) * Math.cos(el) * r, Math.sin(el) * r, Math.cos(az) * Math.cos(el) * r);
    camera.lookAt(0, 0.05, 0);
    renderer.render(scene, camera);
  }

  // ---- the frame loop: one time-boxed step of Python or compiled work, then draw
  function frame() {
    if (running && lab && !document.hidden && inView) {
      const r = JSON.parse(lab.step(compiled ? 22 : 14));
      if (r.error) { info = { state: 'error', error: r.error }; finish(); }
      rate = rate ? rate * 0.85 + r.rate * 0.15 : r.rate;
      if (!compiled && info.state !== 'error') pyRate[preset] = rate;
      const hp = lab.heights(), hb = hp.getBuffer('f32');     // the array's memory, no copy
      const vo = lab.overlay(), vb = vo ? vo.getBuffer('u8') : null;
      updateMesh(hb.data, vb ? vb.data : null);
      hb.release(); hp.destroy();
      if (vo) { vb.release(); vo.destroy(); }
      const n = `${Math.round(rate).toLocaleString()} ${UNITS[preset]}`;
      const x = pyRate[preset] ? ` · ${Math.round(rate / pyRate[preset])}× Python` : '';
      $('.lab-rate').innerHTML = info.state === 'compiled' ? `Compiled by scramjit · ${n}${x}`
        : info.state === 'fallback' ? `scramjit fell back to Python (see the C stage) · ${n}`
        : `Plain Python · ${n}`;
      if (!compiled && !nudged && performance.now() - startedAt > 4000) { root.classList.add('nudge'); nudged = true; }
    }
    if (renderer) draw();
    requestAnimationFrame(frame);
  }
  let inView = true;
  new IntersectionObserver(es => { inView = es[0].isIntersecting; }).observe(root);

  // ---- controls
  function setup() {
    const ta = $('textarea');
    editor = CodeMirror.fromTextArea(ta, { mode: 'python', theme: 'sj', lineNumbers: true, indentUnit: 4, viewportMargin: Infinity });
    editor.setValue('# loading the code…');
    presetsLoaded = Promise.all(Object.keys(PRESETS).map(async p => { sources[p] = await text('lab/' + p + '.py'); }))
      .then(() => { if (!running) editor.setValue(sources[preset]); })
      .catch(e => editor.setValue('# could not load the presets: ' + e.message));
    editor.on('change', () => { clearTimeout(editTimer); if (running) editTimer = setTimeout(setCode, 700); });
    const sel = $('.lab-preset');
    for (const [k, v] of Object.entries(PRESETS)) sel.add(new Option(v, k));
    sel.addEventListener('change', () => {
      preset = sel.value; rate = 0;
      if (sources[preset]) editor.setValue(sources[preset]);
      if (running) { lab.reset(); clearTimeout(editTimer); setCode(compiled); }
    });
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
    $('.lab-reset').addEventListener('click', () => { if (running) { lab.reset(Math.floor(Math.random() * 1e9)); rate = 0; } });
    root.querySelectorAll('.lab-tab').forEach(c => c.addEventListener('click', () => showStage(c.dataset.s)));
    $('.lab-tree').addEventListener('click', e => {        // a statement opens its expressions
      const row = e.target.closest('.lab-row.x');
      if (!row) return;
      row.classList.toggle('on');
      row.nextElementSibling.hidden = !row.classList.contains('on');
    });
    // full screen: the Fullscreen API where it works on elements, a fixed full-window layout elsewhere
    const relayout = () => { resize(); if (editor) editor.refresh(); };
    const setFull = on => { root.classList.toggle('is-full', on); $('.lab-full').setAttribute('aria-label', on ? 'Exit full screen' : 'Full screen'); requestAnimationFrame(relayout); };
    $('.lab-full').addEventListener('click', () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (root.classList.contains('is-full')) setFull(false);
      else if (root.requestFullscreen) root.requestFullscreen().catch(() => setFull(true));
      else setFull(true);
    });
    document.addEventListener('fullscreenchange', () => setFull(document.fullscreenElement === root));
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && root.classList.contains('is-full') && !document.fullscreenElement) setFull(false); });
    $('.lab-start').addEventListener('click', () => boot().catch(e => { status('Could not start: ' + e.message); root.classList.remove('loading'); }));
    initGL();
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
