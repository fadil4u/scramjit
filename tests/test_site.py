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
