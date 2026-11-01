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
  var ABOUT = 'Everything a coding agent needs to use scramjit: install, API, CLI, what compiles and how to fix what doesn\u2019t. Paste it into your agent\u2019s chat or rules file.';
  var LOGOS = [
    ['Claude', 'm4.7144 15.9555 4.7174-2.6471.079-.2307-.079-.1275h-.2307l-.7893-.0486-2.6956-.0729-2.3375-.0971-2.2646-.1214-.5707-.1215-.5343-.7042.0546-.3522.4797-.3218.686.0608 1.5179.1032 2.2767.1578 1.6514.0972 2.4468.255h.3886l.0546-.1579-.1336-.0971-.1032-.0972L6.973 9.8356l-2.55-1.6879-1.3356-.9714-.7225-.4918-.3643-.4614-.1578-1.0078.6557-.7225.8803.0607.2246.0607.8925.686 1.9064 1.4754 2.4893 1.8336.3643.3035.1457-.1032.0182-.0728-.164-.2733-1.3539-2.4467-1.445-2.4893-.6435-1.032-.17-.6194c-.0607-.255-.1032-.4674-.1032-.7285L6.287.1335 6.6997 0l.9957.1336.419.3642.6192 1.4147 1.0018 2.2282 1.5543 3.0296.4553.8985.2429.8318.091.255h.1579v-.1457l.1275-1.706.2368-2.0947.2307-2.6957.0789-.7589.3764-.9107.7468-.4918.5828.2793.4797.686-.0668.4433-.2853 1.8517-.5586 2.9021-.3643 1.9429h.2125l.2429-.2429.9835-1.3053 1.6514-2.0643.7286-.8196.85-.9046.5464-.4311h1.0321l.759 1.1293-.34 1.1657-1.0625 1.3478-.8804 1.1414-1.2628 1.7-.7893 1.36.0729.1093.1882-.0183 2.8535-.607 1.5421-.2794 1.8396-.3157.8318.3886.091.3946-.3278.8075-1.967.4857-2.3072.4614-3.4364.8136-.0425.0304.0486.0607 1.5482.1457.6618.0364h1.621l3.0175.2247.7892.522.4736.6376-.079.4857-1.2142.6193-1.6393-.3886-3.825-.9107-1.3113-.3279h-.1822v.1093l1.0929 1.0686 2.0035 1.8092 2.5075 2.3314.1275.5768-.3218.4554-.34-.0486-2.2039-1.6575-.85-.7468-1.9246-1.621h-.1275v.17l.4432.6496 2.3436 3.5214.1214 1.0807-.17.3521-.6071.2125-.6679-.1214-1.3721-1.9246L14.38 17.959l-1.1414-1.9428-.1397.079-.674 7.2552-.3156.3703-.7286.2793-.6071-.4614-.3218-.7468.3218-1.4753.3886-1.9246.3157-1.53.2853-1.9004.17-.6314-.0121-.0425-.1397.0182-1.4328 1.9672-2.1796 2.9446-1.7243 1.8456-.4128.164-.7164-.3704.0667-.6618.4008-.5889 2.386-3.0357 1.4389-1.882.929-1.0868-.0062-.1579h-.0546l-6.3385 4.1164-1.1293.1457-.4857-.4554.0608-.7467.2307-.2429 1.9064-1.3114Z'],
    ['OpenAI', 'M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z'],
    ['Gemini', 'M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 3.81 2.55t2.55 3.81'],
    ['DeepSeek', 'M23.748 4.651c-.254-.124-.364.113-.512.233-.051.04-.094.09-.137.137-.372.397-.806.657-1.373.626-.829-.046-1.537.214-2.163.848-.133-.782-.575-1.248-1.247-1.548-.352-.155-.708-.311-.955-.65-.172-.24-.219-.509-.305-.774-.055-.16-.11-.323-.293-.35-.2-.031-.278.136-.356.276-.313.572-.434 1.202-.422 1.84.027 1.436.633 2.58 1.838 3.393.137.094.172.187.129.323-.082.28-.18.553-.266.833-.055.179-.137.218-.328.14a5.5 5.5 0 0 1-1.737-1.179c-.857-.828-1.631-1.743-2.597-2.46a12 12 0 0 0-.689-.47c-.985-.957.13-1.743.387-1.836.27-.098.094-.433-.778-.428-.872.003-1.67.295-2.687.685a3 3 0 0 1-.465.136 9.6 9.6 0 0 0-2.883-.101c-1.885.21-3.39 1.1-4.497 2.622C.082 8.776-.231 10.854.152 13.02c.403 2.284 1.568 4.175 3.36 5.653 1.857 1.533 3.997 2.284 6.438 2.14 1.482-.085 3.132-.284 4.994-1.86.47.234.962.328 1.78.398.629.058 1.235-.031 1.705-.129.735-.155.684-.836.418-.961-2.155-1.004-1.682-.595-2.112-.926 1.095-1.295 2.768-3.598 3.284-6.733.05-.346.115-.834.108-1.114-.004-.171.035-.238.23-.257a4.2 4.2 0 0 0 1.545-.475c1.397-.763 1.96-2.016 2.093-3.517.02-.23-.004-.467-.247-.588M11.58 18.168c-2.088-1.642-3.101-2.183-3.52-2.16-.39.024-.32.472-.234.763.09.288.207.487.371.74.114.167.192.416-.113.603-.673.416-1.842-.14-1.897-.168-1.361-.801-2.5-1.86-3.301-3.306-.775-1.393-1.225-2.888-1.299-4.482-.02-.385.094-.522.477-.592a4.7 4.7 0 0 1 1.53-.038c2.131.311 3.946 1.264 5.467 2.774.868.86 1.525 1.887 2.202 2.89.72 1.066 1.494 2.082 2.48 2.915.348.291.626.513.892.677-.802.09-2.14.109-3.055-.615zm1.001-6.44a.306.306 0 0 1 .415-.287.3.3 0 0 1 .113.074.3.3 0 0 1 .086.214c0 .17-.136.307-.308.307a.303.303 0 0 1-.306-.307m3.11 1.596c-.2.081-.4.151-.591.16a1.25 1.25 0 0 1-.798-.254c-.274-.23-.47-.358-.551-.758a1.7 1.7 0 0 1 .015-.588c.07-.327-.007-.537-.238-.727-.188-.156-.426-.199-.689-.199a.6.6 0 0 1-.254-.078.253.253 0 0 1-.114-.358 1 1 0 0 1 .192-.21c.356-.202.767-.136 1.146.016.352.144.618.408 1.001.782.392.451.462.576.685.915.176.264.336.536.446.848.066.194-.02.353-.25.45'],
    ['GitHub Copilot', 'M23.922 16.997C23.061 18.492 18.063 22.02 12 22.02 5.937 22.02.939 18.492.078 16.997A.641.641 0 0 1 0 16.741v-2.869a.883.883 0 0 1 .053-.22c.372-.935 1.347-2.292 2.605-2.656.167-.429.414-1.055.644-1.517a10.098 10.098 0 0 1-.052-1.086c0-1.331.282-2.499 1.132-3.368.397-.406.89-.717 1.474-.952C7.255 2.937 9.248 1.98 11.978 1.98c2.731 0 4.767.957 6.166 2.093.584.235 1.077.546 1.474.952.85.869 1.132 2.037 1.132 3.368 0 .368-.014.733-.052 1.086.23.462.477 1.088.644 1.517 1.258.364 2.233 1.721 2.605 2.656a.841.841 0 0 1 .053.22v2.869a.641.641 0 0 1-.078.256Zm-11.75-5.992h-.344a4.359 4.359 0 0 1-.355.508c-.77.947-1.918 1.492-3.508 1.492-1.725 0-2.989-.359-3.782-1.259a2.137 2.137 0 0 1-.085-.104L4 11.746v6.585c1.435.779 4.514 2.179 8 2.179 3.486 0 6.565-1.4 8-2.179v-6.585l-.098-.104s-.033.045-.085.104c-.793.9-2.057 1.259-3.782 1.259-1.59 0-2.738-.545-3.508-1.492a4.359 4.359 0 0 1-.355-.508Zm2.328 3.25c.549 0 1 .451 1 1v2c0 .549-.451 1-1 1-.549 0-1-.451-1-1v-2c0-.549.451-1 1-1Zm-5 0c.549 0 1 .451 1 1v2c0 .549-.451 1-1 1-.549 0-1-.451-1-1v-2c0-.549.451-1 1-1Zm3.313-6.185c.136 1.057.403 1.913.878 2.497.442.544 1.134.938 2.344.938 1.573 0 2.292-.337 2.657-.751.384-.435.558-1.15.558-2.361 0-1.14-.243-1.847-.705-2.319-.477-.488-1.319-.862-2.824-1.025-1.487-.161-2.192.138-2.533.529-.269.307-.437.808-.438 1.578v.021c0 .265.021.562.063.893Zm-1.626 0c.042-.331.063-.628.063-.894v-.02c-.001-.77-.169-1.271-.438-1.578-.341-.391-1.046-.69-2.533-.529-1.505.163-2.347.537-2.824 1.025-.462.472-.705 1.179-.705 2.319 0 1.211.175 1.926.558 2.361.365.414 1.084.751 2.657.751 1.21 0 1.902-.394 2.344-.938.475-.584.742-1.44.878-2.497Z'],
    ['Cursor', 'M11.503.131 1.891 5.678a.84.84 0 0 0-.42.726v11.188c0 .3.162.575.42.724l9.609 5.55a1 1 0 0 0 .998 0l9.61-5.55a.84.84 0 0 0 .42-.724V6.404a.84.84 0 0 0-.42-.726L12.497.131a1.01 1.01 0 0 0-.996 0M2.657 6.338h18.55c.263 0 .43.287.297.515L12.23 22.918c-.062.107-.229.064-.229-.06V12.335a.59.59 0 0 0-.295-.51l-9.11-5.257c-.109-.063-.064-.23.061-.23']
  ].map(function (l) { return '<svg viewBox="0 0 24 24" role="img" aria-label="' + l[0] + '"><title>' + l[0] + '</title><path d="' + l[1] + '"/></svg>'; }).join('');
  var front = banner.querySelector('.banner-front');

  var back = document.createElement('div');
  back.className = 'rip-back';
  back.id = 'agents';
  back.inert = true;
  back.innerHTML = '<div class="rip-in"><div class="rip-bar">' + ROBOT +
    '<h2>Prompt for Agents</h2>' +
    '<button type="button" class="rip-copy" disabled>' + COPY + '<span>Copy</span></button>' +
    '<button type="button" class="rip-close">Close the prompt</button></div>' +
    '<pre tabindex="0">Loading…</pre>' +
    '<p class="rip-about">' + ABOUT + '</p>' +
    '<div class="rip-works"><span>Works with</span>' + LOGOS + '</div></div>';
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
    var v = Math.max(Math.min(lo, clearOfCopy(W)), Math.min(maxW, 320) + Math.min(maxH, 350) + gaps);
    var pw = Math.min(maxW, Math.round((v - gaps) * 0.5)), ph = Math.min(maxH, v - gaps - pw);
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
    var end = function () { peel.hidden = true; banner.classList.add('is-open'); if (!instant) text.focus({ preventScroll: true }); };
    if (instant) { draw(full()); end(); } else move(full(), 650, end);
  }
  function close() {
    isOpen = false;
    back.inert = true;
    banner.classList.remove('is-open');
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
  // A click on what's left of the banner puts it back, unless it's on a link or button, or selects text.
  front.addEventListener('click', function (e) {
    if (isOpen && !e.target.closest('a, button') && !String(getSelection())) close();
  });
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
