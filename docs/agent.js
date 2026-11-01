// Prompt for Agents: a page corner you can click or pull, opening the prompt in agent-prompt.txt.
(function () {
  var here = document.currentScript.src;
  var ROBOT = '<svg class="i-robot" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v4"/><circle cx="12" cy="3" r=".6"/><rect x="4" y="7" width="16" height="13" rx="3"/><path d="M1.5 12.5v3M22.5 12.5v3M9 16h6"/><circle class="eye" cx="9" cy="11.5" r="1.2"/><circle class="eye" cx="15" cy="11.5" r="1.2"/></svg>';
  var COPY = '<svg class="i-copy" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg><svg class="i-done" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
  var CLOSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
  var url = new URL('agent-prompt.txt', here).href;

  var peel = document.createElement('button');
  peel.type = 'button';
  peel.className = 'peel';
  peel.setAttribute('aria-label', 'Prompt for Agents');
  peel.innerHTML = '<span class="peel-under">' + ROBOT + '</span><span class="peel-flap"><i></i></span><span class="peel-label">Prompt for Agents</span>';

  var dlg = document.createElement('dialog');
  dlg.className = 'agent';
  dlg.id = 'agent-prompt';
  dlg.setAttribute('aria-labelledby', 'agent-title');
  dlg.innerHTML = '<div class="agent-bar">' + ROBOT + '<h2 id="agent-title">Prompt for Agents</h2>' +
    '<button type="button" class="agent-copy" aria-label="Copy prompt" disabled>' + COPY + '<span>Copy</span></button>' +
    '<button type="button" class="agent-close" aria-label="Close">' + CLOSE + '</button></div>' +
    '<p class="agent-note">Paste this into your coding agent. It covers everything needed to use scramjit.</p>' +
    '<pre id="agent-text" tabindex="0">Loading…</pre>';

  var top = document.querySelector('header.top');
  top.parentNode.insertBefore(peel, top.nextSibling);
  document.body.appendChild(dlg);

  var text = dlg.querySelector('pre'), copy = dlg.querySelector('.agent-copy'), label = copy.querySelector('span'), loaded = false;
  function open() {
    dlg.showModal();
    if (loaded) return;
    loaded = true;
    fetch(url, { cache: 'no-cache' }).then(function (r) { if (!r.ok) throw r; return r.text(); }).then(function (t) {
      text.textContent = t;
      copy.disabled = false;
    }).catch(function () {
      loaded = false;
      text.textContent = 'The prompt did not load. It is at ' + url;
    });
  }

  // Opens on a click, or when pulled far enough; a short pull springs back.
  var start = null, pulled = 0;
  function drop() { start = null; peel.classList.remove('pulling'); peel.style.removeProperty('--p'); }
  peel.addEventListener('pointerdown', function (e) {
    if (e.button) return;
    start = { x: e.clientX, y: e.clientY, p: peel.offsetWidth };
    pulled = 0;
    peel.setPointerCapture(e.pointerId);
    peel.classList.add('pulling');
  });
  peel.addEventListener('pointermove', function (e) {
    if (!start) return;
    pulled = Math.max(0, start.x - e.clientX, e.clientY - start.y);
    peel.style.setProperty('--p', Math.min(start.p + pulled, Math.min(innerWidth, innerHeight) * .6) + 'px');
  });
  peel.addEventListener('pointerup', function () {
    if (!start) return;
    var d = pulled;
    drop();
    if (d < 6 || d >= 40) open();
  });
  peel.addEventListener('pointercancel', drop);
  peel.addEventListener('click', function (e) { if (e.detail === 0) open(); });  // keyboard

  dlg.querySelector('.agent-close').addEventListener('click', function () { dlg.close(); });
  dlg.addEventListener('click', function (e) { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener('keydown', function (e) { e.stopPropagation(); });  // keeps the page's shortcuts out of the dialog
  copy.addEventListener('click', function () {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(text.textContent).then(function () {
      copy.classList.add('done');
      label.textContent = 'Copied';
      setTimeout(function () { copy.classList.remove('done'); label.textContent = 'Copy'; }, 1400);
    }, function () {});
  });
})();
