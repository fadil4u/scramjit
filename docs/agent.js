// Prompt for Agents. On the landing page, the banner's top right corner peels back to show the
// prompt behind it. Elsewhere, a corner links there.
(function () {
  var here = document.currentScript.src;
  var ROBOT = '<svg class="i-robot" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v4"/><circle cx="12" cy="3" r=".6"/><rect x="4" y="7" width="16" height="13" rx="3"/><path d="M1.5 12.5v3M22.5 12.5v3M9 16h6"/><circle class="eye" cx="9" cy="11.5" r="1.2"/><circle class="eye eye-r" cx="15" cy="11.5" r="1.2"/></svg>';
  var LABEL = '<span class="peel-label">Prompt for Agents</span>';
  var banner = document.querySelector('.banner');

  if (!banner) {
    var link = document.createElement('a');
    link.className = 'peel peel-link';
    link.href = new URL('./#agents', here).href;
    link.setAttribute('aria-label', 'Prompt for Agents');
    link.innerHTML = '<span class="peel-under">' + ROBOT + '</span><span class="peel-flap"><i></i></span>' + LABEL;
    var header = document.querySelector('header.top');
    header.parentNode.insertBefore(link, header.nextSibling);
    return;
  }

  var phone = matchMedia('(max-width:600px)');
  var REST = phone.matches ? 44 : 60, HOVER = REST + 12, TIP = 96;
  var SHARE = 0.4, ROLL = phone.matches ? 44 : 70;  // how much of the banner peels, and how wide the fold rolls up
  var COPY = '<svg class="i-copy" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg><svg class="i-done" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
  var front = banner.querySelector('.banner-front');

  var back = document.createElement('div');
  back.className = 'rip-back';
  back.id = 'agents';
  back.inert = true;
  back.innerHTML = '<div class="rip-in"><div class="rip-bar">' + ROBOT +
    '<h2>Prompt for Agents</h2>' +
    '<button type="button" class="rip-copy" disabled>' + COPY + '<span>Copy</span></button>' +
    '<button type="button" class="rip-close">Put it back</button></div>' +
    '<pre tabindex="0">Loading…</pre></div>';
  banner.insertBefore(back, front);

  var flap = document.createElement('div');
  flap.className = 'rip-flap';
  flap.innerHTML = '<i></i>';
  banner.appendChild(flap);
  var paper = flap.firstChild;

  var peel = document.createElement('button');
  peel.type = 'button';
  peel.className = 'peel';
  peel.setAttribute('aria-label', 'Prompt for Agents');
  peel.setAttribute('aria-expanded', 'false');
  peel.setAttribute('aria-controls', 'agents');
  peel.innerHTML = '<span class="peel-under">' + ROBOT + '</span><i class="peel-hit"></i>' + LABEL;
  banner.appendChild(peel);

  var text = back.querySelector('pre'), copy = back.querySelector('.rip-copy'), label = copy.querySelector('span');
  var url = new URL('agent-prompt.txt', here).href;
  fetch(url, { cache: 'no-cache' }).then(function (r) { if (!r.ok) throw r; return r.text(); }).then(function (t) {
    text.textContent = t;
    copy.disabled = false;
  }).catch(function () { text.textContent = 'The prompt did not load. It is at ' + url; });

  // Folding the corner to (W - q, q) creases the banner along y = x - W + q. The part above the
  // crease is gone from the front and shows, reflected across the crease, as the flap.
  var q = REST, isOpen = false, raf = 0, drag = null;
  var points = function (ps) { return ps.length ? 'polygon(' + ps.map(function (p) { return p[0] + 'px ' + p[1] + 'px'; }).join(',') + ')' : 'polygon(0 0)'; };
  function cut(poly, d) {
    var out = [];
    poly.forEach(function (a, i) {
      var b = poly[(i + 1) % poly.length], da = d(a), db = d(b);
      if (da >= 0) out.push(a);
      if (da * db < 0) { var t = da / (da - db); out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]); }
    });
    return out;
  }
  function draw(v) {
    q = v;
    var W = banner.clientWidth, H = banner.clientHeight, rect = [[0, 0], [W, 0], [W, H], [0, H]];
    var below = function (p) { return p[1] - p[0] + W - q; };
    front.style.clipPath = points(cut(rect, below));
    var flipped = cut(rect, function (p) { return -below(p); }).map(function (p) { return [p[1] + W - q, p[0] - W + q]; });
    paper.style.clipPath = points(cut(flipped, function (p) { return ROLL * Math.SQRT2 - below(p); }));
    paper.style.setProperty('--crease', q / Math.SQRT2 + 'px');
    var size = Math.min(q, TIP);
    peel.style.width = peel.style.height = size + 'px';
    peel.style.opacity = Math.max(0, Math.min(1, (170 - q) / 60));
  }
  // Open, the crease stops where SHARE of the banner has peeled or just short of the headline and
  // lede, whichever comes first, unless the prompt's panel needs more room, as on a phone. The
  // panel fits exactly above the crease.
  var panel = back.querySelector('.rip-in'), TOP = 16, EDGE = 14;
  function peeled(v, W, H) {
    var d = function (p) { return p[0] - p[1] - W + v; };
    var ps = cut([[0, 0], [W, 0], [W, H], [0, H]], d), area = 0;
    ps.forEach(function (a, i) { var b = ps[(i + 1) % ps.length]; area += a[0] * b[1] - b[0] * a[1]; });
    return Math.abs(area) / 2 / (W * H);
  }
  function clearOfCopy(W) {
    var box = banner.getBoundingClientRect(), range = document.createRange(), limit = Infinity;
    banner.querySelectorAll('.banner-copy h1, .banner-copy .lede').forEach(function (el) {
      range.selectNodeContents(el);
      [].forEach.call(range.getClientRects(), function (r) {
        limit = Math.min(limit, r.top - box.top + W - (r.right - box.left) - ROLL * Math.SQRT2 - 8);
      });
    });
    return limit;
  }
  function full() {
    var W = banner.clientWidth, H = banner.clientHeight, lo = 0, hi = W + H, side = phone.matches ? 16 : 24;
    for (var i = 0; i < 24; i++) { var m = (lo + hi) / 2; if (peeled(m, W, H) < SHARE) lo = m; else hi = m; }
    var gaps = side + TOP + EDGE, maxW = W - 2 * side, maxH = H - 2 * TOP;
    var v = Math.max(Math.min(lo, clearOfCopy(W)), Math.min(maxW, 320) + Math.min(maxH, 280) + gaps);
    var pw = Math.min(maxW, Math.round((v - gaps) * 0.56)), ph = Math.min(maxH, v - gaps - pw);
    panel.style.cssText = 'top:' + TOP + 'px;right:' + side + 'px;width:' + pw + 'px;height:' + ph + 'px';
    return v;
  }
  var still = matchMedia('(prefers-reduced-motion:reduce)');
  function move(to, ms, done) {
    cancelAnimationFrame(raf);
    if (still.matches) { draw(to); if (done) done(); return; }
    var from = q, t0 = performance.now();
    (function step(now) {
      var t = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - t, 3);
      draw(from + (to - from) * e);
      if (t < 1) raf = requestAnimationFrame(step); else if (done) done();
    })(t0);
  }

  function open(instant) {
    isOpen = true;
    back.inert = false;
    peel.setAttribute('aria-expanded', 'true');
    var end = function () { peel.hidden = true; if (!instant) text.focus({ preventScroll: true }); };
    if (instant) { draw(full()); end(); } else move(full(), 650, end);
  }
  function close() {
    isOpen = false;
    back.inert = true;
    peel.hidden = false;
    peel.setAttribute('aria-expanded', 'false');
    move(REST, 550, function () { peel.focus({ preventScroll: true }); });
  }

  // Opens on a click, or when pulled far enough; a short pull falls back.
  peel.addEventListener('pointerdown', function (e) {
    if (e.button || isOpen) return;
    cancelAnimationFrame(raf);
    drag = { x: e.clientX, y: e.clientY, q: q, moved: 0 };
    peel.setPointerCapture(e.pointerId);
    peel.classList.add('pulling');
  });
  peel.addEventListener('pointermove', function (e) {
    if (!drag) return;
    drag.moved = Math.max(0, (drag.x - e.clientX + e.clientY - drag.y) / 2);
    draw(Math.min(drag.q + drag.moved, full()));
  });
  peel.addEventListener('pointerup', function () {
    if (!drag) return;
    var moved = drag.moved;
    drag = null;
    peel.classList.remove('pulling');
    if (moved < 4 || moved >= 40) open(); else move(REST, 300);
  });
  peel.addEventListener('pointercancel', function () { drag = null; peel.classList.remove('pulling'); move(REST, 300); });
  peel.addEventListener('click', function (e) { if (e.detail === 0 && !isOpen) open(); });  // keyboard
  peel.addEventListener('pointerenter', function (e) { if (e.pointerType === 'mouse' && !drag && !isOpen) move(HOVER, 150); });
  peel.addEventListener('pointerleave', function (e) { if (e.pointerType === 'mouse' && !drag && !isOpen) move(REST, 150); });
  peel.addEventListener('focus', function () { if (peel.matches(':focus-visible') && !isOpen) move(HOVER, 150); });
  peel.addEventListener('blur', function () { if (!drag && !isOpen) move(REST, 150); });

  back.querySelector('.rip-close').addEventListener('click', close);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && isOpen) close(); });
  copy.addEventListener('click', function () {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(text.textContent).then(function () {
      copy.classList.add('done');
      label.textContent = 'Copied';
      setTimeout(function () { copy.classList.remove('done'); label.textContent = 'Copy'; }, 1400);
    }, function () {});
  });

  if ('ResizeObserver' in window) new ResizeObserver(function () { if (!drag) draw(isOpen ? full() : q); }).observe(banner);
  full();
  draw(REST);
  if (location.hash === '#agents') open(true);
})();
