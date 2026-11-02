"""The site's agent-facing index stays in step with the docs."""
import json
import os
import re

DOCS = os.path.join(os.path.dirname(__file__), '..', 'docs')


def test_llms_txt_lists_every_docs_page():
    with open(os.path.join(DOCS, 'docs', 'nav.json')) as f:
        pages = [p[0] for part in json.load(f)['parts'] for p in part['pages']]
    with open(os.path.join(DOCS, 'llms.txt')) as f:
        linked = re.findall(r'https://www\.scramjit\.dev/docs/content/([\w-]+)\.md', f.read())
    assert linked == pages
    for page in linked:
        assert os.path.exists(os.path.join(DOCS, 'docs', 'content', page + '.md'))


def test_workers_load_every_module_scramjit_imports():
    """A page loading scramjit file by file breaks in the browser, and only there, if one is missing."""
    import glob
    docs = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')
    package = os.path.join(docs, '..', 'scramjit')

    def imports(name):
        with open(os.path.join(package, name), encoding='utf-8') as f:
            text = f.read()
        found = set(re.findall(r'^from \.(_\w+) import', text, re.M))
        for line in re.findall(r'^from \. import ([\w, ]+)', text, re.M):
            found |= {x.split(' as ')[0].strip() for x in line.split(',')}
        return {x + '.py' for x in found}

    workers = glob.glob(os.path.join(docs, '**', 'worker.js'), recursive=True)
    assert len(workers) >= 2
    for path in workers:
        with open(path, encoding='utf-8') as f:
            lists = [m for m in re.findall(r'\[([^\]]*_dispatch\.py[^\]]*)\]', f.read())]
        assert lists, path
        listed = set(re.findall(r"'([\w.]+\.py)'", lists[0]))
        need, seen = {'__init__.py'}, set()
        while need:
            name = need.pop()
            seen.add(name)
            need |= imports(name) - seen
        assert seen <= listed, (path, sorted(seen - listed))
