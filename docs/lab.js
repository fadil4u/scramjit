// The live demo: Python (in Pyodide) erodes a terrain; flip scramjit on and the same function is
// compiled to WebAssembly (syntax tree -> types -> C -> cimo) and runs much faster. three.js draws.
(function () {
  'use strict';
  const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
  const PRESETS = { erosion: 'Hydraulic erosion', beacon: 'Beacon (viewshed)', landslides: 'Landslides' };
  const UNITS = { erosion: 'droplets/s', beacon: 'sightlines/s', landslides: 'cells/s' };
  const SJ_FILES = ['__init__.py', '_compiler.py', '_dispatch.py', '_errors.py', '_runtime.py', '_types.py'];
  const STAGES = ['python', 'ast', 'types', 'c', 'wasm'];
  const root = document.getElementById('lab-demo');
  if (!root) return;
  const $ = s => root.querySelector(s);
  const RM = matchMedia('(prefers-reduced-motion:reduce)').matches;

  let py = null, lab = null, editor = null, preset = 'erosion', compiled = false, running = false;
  let sources = {}, info = {}, stage = 'python', pyRate = {}, rate = 0, editTimer = null, presetsLoaded = null;

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
  }

  // ---- code and the pipeline
  function showStage(s) {
    stage = s;
    root.querySelectorAll('.lab-chip').forEach(c => c.classList.toggle('open', c.dataset.s === s));
    const pre = $('.lab-panel pre');
    const t = {
      python: 'Plain Python. Pyodide runs it in the browser, one bytecode at a time.',
      ast: info.ast || '',
      types: info.types ? 'Inferred for this call:\n\n' + info.types : (compiled ? '' : 'Turn scramjit on to infer types.'),
      c: info.c || (info.state === 'fallback' ? 'Runs as Python: ' + info.reason : 'Turn scramjit on to generate C.'),
      wasm: info.state === 'compiled' ? `cimo compiled the C to WebAssembly in ${info.compile_ms} ms, in this page, and linked it into Pyodide's memory. The function now runs as native WebAssembly on the same numpy array.` : (info.state === 'fallback' ? 'Not compiled: ' + info.reason : 'Turn scramjit on to compile.'),
    }[s];
    pre.textContent = info.state === 'error' ? info.error : t;
    if (s === 'wasm' && info.state === 'compiled') {        // disassemble when asked, then keep it
      if (!info.wat) {
        pre.textContent = ';; disassembling…';
        setTimeout(() => {
          try { info.wat = lab.wasm_text(); } catch (e) { info.wat = ';; could not disassemble: ' + e.message; }
          if (stage === 'wasm') pre.textContent = info.wat;
        }, 30);
      } else pre.textContent = info.wat;
    }
  }
  function lightStages(n) {
    root.querySelectorAll('.lab-chip').forEach((c, i) => c.classList.toggle('lit', i < n));
  }
  function setCode() {
    if (!lab) return;
    const src = editor.getValue();
    $('.lab-err').hidden = true;
    if (!compiled) {
      info = JSON.parse(lab.set_code(preset, src, false));
      lightStages(1);
      finish();
      return;
    }
    // scramjit on: the stages light up as the pipeline runs (compiling blocks, so paint first)
    lightStages(1);
    let k = 1;
    const tick = () => { if (k < 3) { lightStages(++k); setTimeout(tick, RM ? 0 : 220); } else compile(); };
    setTimeout(tick, RM ? 0 : 220);
    function compile() {
      requestAnimationFrame(() => setTimeout(() => {
        info = JSON.parse(lab.set_code(preset, src, true));
        lightStages(info.state === 'compiled' ? 5 : 3);
        showStage(info.state === 'compiled' ? 'c' : 'c');
        finish();
      }, 0));
    }
  }
  function finish() {
    if (info.state === 'error') {
      const e = $('.lab-err');
      e.hidden = false; e.textContent = info.error;
    }
    root.classList.toggle('fast', info.state === 'compiled');
    if (!compiled) showStage(stage === 'c' || stage === 'types' || stage === 'wasm' ? 'python' : stage);
    else if (stage === 'python') showStage('c');
    else showStage(stage);
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
      const sp = compiled && pyRate[preset] ? `  ·  ${Math.round(rate / pyRate[preset])}× Python` : '';
      const where = info.state === 'compiled' ? ' <small>· compiled in this page</small>' : '';
      $('.lab-rate').innerHTML = `${Math.round(rate).toLocaleString()} ${UNITS[preset]}${sp}${where}`;
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
      if (running) { lab.reset(); clearTimeout(editTimer); setCode(); }
    });
    $('.lab-switch').addEventListener('click', () => {
      compiled = !compiled; rate = 0;
      $('.lab-switch').setAttribute('aria-pressed', compiled);
      if (running) setCode();
    });
    $('.lab-reset').addEventListener('click', () => { if (running) { lab.reset(); rate = 0; } });
    root.querySelectorAll('.lab-chip').forEach(c => c.addEventListener('click', () => showStage(c.dataset.s)));
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
