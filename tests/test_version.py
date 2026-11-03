"""The version is written in a few places, and they agree; so does the cimo it needs."""

import os
import re

import scramjit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def read(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def test_versions_agree():
    project = re.search(r'^version = "([^"]+)"', read('pyproject.toml'), re.M).group(1)
    site = re.search(r"^__version__ = '([^']+)'", read('docs', 'py', 'scramjit', '__init__.py'), re.M).group(1)
    assert project == scramjit.__version__ == site


def test_the_readme_names_the_released_version():
    assert f'{scramjit.__version__} is on PyPI' in read('README.md')


def test_cimo_is_new_enough():
    import cimo
    need = re.search(r'"cimo>=([^"]+)"', read('pyproject.toml')).group(1)
    assert tuple(map(int, cimo.__version__.split('.'))) >= tuple(map(int, need.split('.'))) or os.environ.get('CIMO_DEV')
