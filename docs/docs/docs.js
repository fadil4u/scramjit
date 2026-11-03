// scramjit docs: Markdown pages (content/*.md, ordered by nav.json) rendered into the site's skin.
// Routes are ?p=<page>#<heading>; navigation stays in the page. Search indexes every section.
(function () {
  'use strict';
  const $ = s => document.querySelector(s);
  const side = $('.side'), doc = $('.doc'), toc = $('.toc'), pager = $('.pager'), edit = $('.edit');
  const walk = $('.walk'), searchEl = $('.search'), input = $('.search-in input'), results = $('.results');
  const REPO = 'https://github.com/farizrahman4u/scramjit';

  let pages = [], bySlug = {}, cache = {}, current = null;

  // ---- helpers
  const esc = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const slugify = t => t.toLowerCase().replace(/<[^>]+>/g, '').replace(/[`*_]/g, '').replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-');
  function readSet() { try { return new Set(JSON.parse(localStorage.getItem('scramjit-docs-read') || '[]')); } catch (e) { return new Set(); } }
  function markRead(slug) {
    const s = readSet(); s.add(slug);
    try { localStorage.setItem('scramjit-docs-read', JSON.stringify([...s])); } catch (e) {}
    const a = side.querySelector(`a[data-p="${slug}"]`); if (a && a.querySelector('.n')) a.classList.add('read');
  }
  function fetchPage(slug) {
    // no-cache: revalidate with the server on every load (GitHub Pages lets browsers keep
    // files for 10 minutes), so an edited page shows on the next reload
    if (!cache[slug]) cache[slug] = fetch(`content/${slug}.md`, { cache: 'no-cache' }).then(r => { if (!r.ok) throw new Error(r.status); return r.text(); });
    return cache[slug];
  }

  // ---- syntax highlighting
  function tokens(text, re, cls) {
    let out = '', last = 0;
    text.replace(re, (m, ...g) => {
      const at = g[g.length - 2];
      out += esc(text.slice(last, at));
      const k = g.findIndex((x, i) => i < cls.length && x !== undefined);
      out += k < 0 || !cls[k] ? esc(m) : `<span class="t-${cls[k]}">${esc(m)}</span>`;
      last = at + m.length;
      return m;
    });
    return out + esc(text.slice(last));
  }
  const HL = {
    c: t => tokens(t, /(\/\*[\s\S]*?\*\/|\/\/[^\n]*)|(^[ \t]*#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)')|\b(for|if|else|while|do|return|static|const|restrict|struct|union|sizeof|switch|case|break|continue|goto|typedef|extern|inline)\b|\b(void|int|long|short|char|float|double|unsigned|signed|size_t|int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t|FILE|_Bool|bool)\b|(\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?[fFuUlL]*\b)|\b([A-Za-z_]\w*)(?=\s*\()/gm, ['cm', 'pp', 'str', 'kw', 'ty', 'num', 'fn']),
    python: t => tokens(t, /(#[^\n]*)|(^[ \t]*@[\w.]+)|([rbf]?"""[\s\S]*?"""|[rbf]?'''[\s\S]*?'''|[rbf]?"(?:[^"\\\n]|\\.)*"|[rbf]?'(?:[^'\\\n]|\\.)*')|\b(import|from|as|def|return|with|for|in|if|else|elif|while|break|continue|pass|try|except|raise|lambda|await|async|class|None|True|False|not|and|or|is|yield|del|assert|global|nonlocal)\b|(\b\d+(?:\.\d+)?(?:e[-+]?\d+)?\b)|\b([A-Za-z_]\w*)(?=\()/gm, ['cm', 'dec', 'str', 'kw', 'num', 'fn']),
    console: t => t.split('\n').map(l => {
      const m = /^(\$ )(.*?)(\s+#.*)?$/.exec(l);
      if (!m) return `<span class="t-out">${esc(l)}</span>`;
      return `<span class="t-pr">$ </span>${esc(m[2])}${m[3] ? `<span class="t-cm">${esc(m[3])}</span>` : ''}`;
    }).join('\n'),
  };
  // C inside r"""...""" strings (cimo.compile sources) is highlighted as C
  const pyOnly = HL.python;
  HL.python = t => t.split(/([rR]"""[\s\S]*?""")/).map((part, i) => i % 2
    ? `<span class="t-str">${part.slice(0, 4)}</span>${HL.c(part.slice(4, -3))}<span class="t-str">"""</span>`
    : pyOnly(part)).join('');
  HL.sh = HL.console; HL.py = HL.python;

  // ---- navigation
  function buildSide(nav) {
    let n = 0;
    const read = readSet();
    side.innerHTML = nav.parts.map((part, pi) => {
      const items = part.pages.map(([slug, title]) => {
        const p = { slug, title, part: part.title, partIndex: pi, walk: part.walk };
        if (part.walk) p.step = ++n;
        pages.push(p); bySlug[slug] = p;
        const mark = part.walk ? `<span class="n"><i>${p.step}</i></span>` : '<span class="dot"></span>';
        return `<li><a href="?p=${slug}" data-p="${slug}" class="${part.walk && read.has(slug) ? 'read' : ''}">${mark}${esc(title)}</a></li>`;
      }).join('');
      return `<section class="part"><h2>${esc(part.title)}${part.walk ? `<small>part ${pi + 1}</small>` : ''}</h2><ol>${items}</ol></section>`;
    }).join('');
    const steps = pages.filter(p => p.walk).length;
    pages.forEach(p => { p.steps = steps; });
  }

  function linkTo(href) {  // content links: "arrays.md#x" -> "?p=arrays#x"
    const m = /^([\w-]+)\.md(#.*)?$/.exec(href);
    return m ? `?p=${m[1]}${m[2] || ''}` : href;
  }

  function enhance() {
    const used = {};
    doc.querySelectorAll('h2, h3').forEach(h => {
      let id = slugify(h.textContent);
      if (used[id]) id += '-' + (++used[id]); else used[id] = 1;
      h.id = id;
      const a = document.createElement('a');
      a.className = 'anchor'; a.href = '#' + id; a.textContent = '#'; a.setAttribute('aria-label', 'link to this section');
      h.appendChild(a);
    });
    doc.querySelectorAll('a[href]').forEach(a => {
      const h = a.getAttribute('href');
      if (/^[\w-]+\.md(#.*)?$/.test(h)) a.setAttribute('href', linkTo(h));
      else if (/^https?:/.test(h)) { a.target = '_blank'; a.rel = 'noopener'; }
    });
    doc.querySelectorAll('table').forEach(t => {
      const w = document.createElement('div'); w.className = 'tbl';
      t.parentNode.insertBefore(w, t); w.appendChild(t);
    });
    doc.querySelectorAll('pre > code').forEach(code => {
      const lang = ((code.className.match(/language-(\w+)/) || [])[1] || 'text').toLowerCase();
      const text = code.textContent.replace(/\n$/, '');
      code.innerHTML = HL[lang] ? HL[lang](text) : esc(text);
      const pre = code.parentNode;
      if (lang !== 'text') pre.insertAdjacentHTML('afterbegin', `<span class="lang">${lang}</span>`);
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'copy'; b.setAttribute('aria-label', 'Copy');
      b.innerHTML = '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg>';
      b.addEventListener('click', () => {
        const t = lang === 'console' ? text.split('\n').filter(l => l.startsWith('$ ')).map(l => l.slice(2).replace(/\s+#.*$/, '')).join('\n') : text;
        navigator.clipboard && navigator.clipboard.writeText(t).then(() => { b.classList.add('done'); setTimeout(() => b.classList.remove('done'), 1200); });
      });
      pre.appendChild(b);
    });
    // a Python snippet followed by the C scramjit generates for it (tools/docs_c.py): two tabs, Python open
    doc.querySelectorAll('pre > code.language-c-generated').forEach(code => {
      const cpre = code.parentNode, ppre = cpre.previousElementSibling;
      if (!ppre || ppre.tagName !== 'PRE' || !ppre.querySelector('code.language-python')) return;
      const box = document.createElement('div'), bar = document.createElement('div');
      box.className = 'tabs'; bar.className = 'tab-bar'; bar.setAttribute('role', 'tablist');
      ppre.parentNode.insertBefore(box, ppre);
      box.append(bar, ppre, cpre);
      const panes = [ppre, cpre];
      ['Python', 'Generated C'].forEach((label, k) => {
        const b = document.createElement('button');
        b.type = 'button'; b.textContent = label; b.setAttribute('role', 'tab');
        b.addEventListener('click', () => {
          panes.forEach((p, j) => { p.hidden = j !== k; });
          bar.querySelectorAll('button').forEach((x, j) => { x.classList.toggle('on', j === k); x.setAttribute('aria-selected', j === k); });
        });
        bar.appendChild(b);
      });
      bar.firstChild.click();
    });
    doc.querySelectorAll('blockquote').forEach(q => {
      const p = q.querySelector('p');
      const m = p && /^\[!(NOTE|TIP|WARNING|PLANNED)\]\s*/i.exec(p.innerHTML);
      if (!m) return;
      const kind = m[1].toLowerCase();
      q.classList.add(kind);
      p.innerHTML = `<span class="callout-t">${{ note: 'Note', tip: 'Tip', warning: 'Warning', planned: 'Planned' }[kind]}</span>` + p.innerHTML.slice(m[0].length);
    });
  }

  function buildToc() {
    const hs = [...doc.querySelectorAll('h2, h3')];
    toc.innerHTML = hs.length < 2 ? '' : '<h2>On this page</h2>' + hs.map(h =>
      `<a href="#${h.id}" class="${h.tagName.toLowerCase()}" data-id="${h.id}">${esc(h.firstChild.textContent)}</a>`).join('');
  }
  function spy() {
    const hs = [...doc.querySelectorAll('h2, h3')];
    let on = hs.length ? hs[0].id : null;
    for (const h of hs) if (h.getBoundingClientRect().top < 120) on = h.id;
    toc.querySelectorAll('a').forEach(a => a.classList.toggle('on', a.dataset.id === on));
  }
  addEventListener('scroll', spy, { passive: true });

  let readObs = null;
  async function show(slug, hash, push) {
    const p = bySlug[slug] || pages[0];
    if (push) history.pushState({}, '', `?p=${p.slug}${hash || ''}`);
    current = p;
    document.body.classList.remove('nav-open');
    side.querySelectorAll('a').forEach(a => a.classList.toggle('on', a.dataset.p === p.slug));
    let md;
    try { md = await fetchPage(p.slug); } catch (e) { doc.innerHTML = '<h1>Page not found</h1><p>Try the search.</p>'; return; }
    if (current !== p) return;
    doc.innerHTML = marked.parse(md);
    enhance();
    buildToc();
    document.title = `${p.title} · scramjit docs`;
    walk.hidden = !p.walk;
    if (p.walk) {
      walk.querySelector('.walk-t').textContent = `Part ${p.partIndex + 1} · ${p.part} · step ${p.step} of ${p.steps}`;
      walk.querySelector('.walk-bar b').style.width = (100 * p.step / p.steps) + '%';
    }
    const i = pages.indexOf(p), prev = pages[i - 1], next = pages[i + 1];
    pager.innerHTML = (prev ? `<a class="prev" href="?p=${prev.slug}"><small>← Previous</small><b>${esc(prev.title)}</b></a>` : '') +
      (next ? `<a class="next" href="?p=${next.slug}"><small>${p.walk && !next.walk ? 'Reference →' : 'Next →'}</small><b>${esc(next.title)}</b></a>` : '');
    edit.innerHTML = `<a href="${REPO}/edit/main/docs/docs/content/${p.slug}.md" target="_blank" rel="noopener">Edit this page on GitHub</a>`;
    if (readObs) readObs.disconnect();
    readObs = new IntersectionObserver(es => { if (es[0].isIntersecting) markRead(p.slug); }, { threshold: 1 });
    readObs.observe(pager);
    const target = hash && document.getElementById(hash.slice(1));
    if (target) target.scrollIntoView(); else if (push) scrollTo(0, 0);
    spy();
  }

  document.addEventListener('click', e => {  // in-page navigation for ?p= links
    const a = e.target.closest('a[href]');
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey || a.target === '_blank') return;
    const url = new URL(a.getAttribute('href'), location.href);
    if (url.origin !== location.origin || url.pathname !== location.pathname) return;
    const p = url.searchParams.get('p');
    if (p) { e.preventDefault(); if (current && p === current.slug && url.hash) { history.pushState({}, '', url.search + url.hash); document.getElementById(url.hash.slice(1))?.scrollIntoView({ behavior: 'smooth' }); } else show(p, url.hash, true); }
    else if (url.hash && !url.search) { e.preventDefault(); history.pushState({}, '', location.search + url.hash); document.getElementById(url.hash.slice(1))?.scrollIntoView({ behavior: 'smooth' }); }
  });
  addEventListener('popstate', () => show(new URLSearchParams(location.search).get('p'), location.hash, false));
  $('.menu').addEventListener('click', () => {
    const open = document.body.classList.toggle('nav-open');
    $('.menu').setAttribute('aria-expanded', open);
  });

  // ---- search: every section of every page, ranked by where the words appear
  let index = null;
  async function buildIndex() {
    if (index) return index;
    const out = [];
    await Promise.all(pages.map(async p => {
      const md = await fetchPage(p.slug);
      const used = {};
      let sec = { heading: p.title, id: '', text: [] };
      const flush = () => { out.push({ page: p, heading: sec.heading, id: sec.id, text: plain(sec.text.join('\n')) }); };
      for (const line of md.split('\n')) {
        const h = /^(#{2,3})\s+(.*)$/.exec(line);
        if (h) {
          flush();
          let id = slugify(h[2]);
          if (used[id]) id += '-' + (++used[id]); else used[id] = 1;
          sec = { heading: h[2].replace(/[`*]/g, ''), id, text: [] };
        } else if (!/^#\s/.test(line)) sec.text.push(line);
      }
      flush();
    }));
    index = out;
    return index;
  }
  function plain(md) {
    return md.replace(/```\w*\n?/g, ' ').replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/[`*_>|#]/g, ' ')
      .replace(/<[^>]+>/g, ' ').replace(/\[!(NOTE|TIP|WARNING|PLANNED)\]/gi, ' ').replace(/\s+/g, ' ').trim();
  }
  function search(q) {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return [];
    const hits = [];
    for (const s of index) {
      const hd = s.heading.toLowerCase(), pt = s.page.title.toLowerCase(), tx = s.text.toLowerCase();
      let score = 0, ok = true;
      for (const t of terms) {
        const inH = hd.includes(t), inP = pt.includes(t);
        const count = tx.split(t).length - 1;
        if (!inH && !inP && !count) { ok = false; break; }
        score += (inH ? 10 : 0) + (inP ? 5 : 0) + Math.min(count, 5);
        if (hd.startsWith(t)) score += 4;
      }
      if (ok) { if (hd.includes(q.toLowerCase())) score += 8; if (!s.page.walk) score += 1; hits.push({ s, score }); }
    }
    return hits.sort((a, b) => b.score - a.score).slice(0, 20).map(h => h.s);
  }
  function mark(text, terms) {
    let h = esc(text);
    for (const t of terms) h = h.replace(new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>');
    return h;
  }
  function snippet(text, terms) {
    const lo = text.toLowerCase();
    let at = Math.min(...terms.map(t => { const i = lo.indexOf(t); return i < 0 ? Infinity : i; }));
    if (!isFinite(at)) at = 0;
    const start = Math.max(0, at - 40);
    return (start ? '…' : '') + text.slice(start, start + 160);
  }
  let sel = 0, shown = [];
  async function runSearch() {
    await buildIndex();
    const q = input.value.trim(), terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    shown = search(q);
    sel = 0;
    if (!q) { results.innerHTML = '<li class="empty">Search every page: functions, options, errors, topics.</li>'; return; }
    results.innerHTML = shown.length ? shown.map((s, i) =>
      `<li class="${i === 0 ? 'on' : ''}" role="option"><a href="?p=${s.page.slug}${s.id ? '#' + s.id : ''}"><span class="where">${esc(s.page.part)} · ${esc(s.page.title)}</span><b>${mark(s.heading, terms)}</b><span class="snip">${mark(snippet(s.text, terms), terms)}</span></a></li>`).join('')
      : `<li class="empty">Nothing for “${esc(q)}”.</li>`;
  }
  function openSearch() { searchEl.hidden = false; input.value = ''; runSearch(); setTimeout(() => input.focus(), 0); }
  function closeSearch() { searchEl.hidden = true; }
  $('.search-open').addEventListener('click', openSearch);
  searchEl.addEventListener('click', e => { if (e.target === searchEl) closeSearch(); if (e.target.closest('.results a')) closeSearch(); });
  input.addEventListener('input', runSearch);
  input.addEventListener('keydown', e => {
    const items = results.querySelectorAll('li[role=option]');
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!items.length) return;
      sel = (sel + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      items.forEach((li, i) => li.classList.toggle('on', i === sel));
      items[sel].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter' && items[sel]) { e.preventDefault(); items[sel].querySelector('a').click(); closeSearch(); }
    else if (e.key === 'Escape') closeSearch();
  });
  document.addEventListener('keydown', e => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if ((e.key === '/' && !typing) || (e.key.toLowerCase() === 'k' && (e.ctrlKey || e.metaKey))) { e.preventDefault(); openSearch(); }
    else if (e.key === 'Escape' && !searchEl.hidden) closeSearch();
  });

  // ---- start
  marked.use({ gfm: true });
  fetch('nav.json', { cache: 'no-cache' }).then(r => r.json()).then(nav => {
    buildSide(nav);
    const q = new URLSearchParams(location.search);
    show(q.get('p') || pages[0].slug, location.hash, false);
    setTimeout(buildIndex, 1500);  // warm the search index
  });
})();
