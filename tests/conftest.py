import os
import sys

os.environ.setdefault('CIMO_FMA', '0')   # before cimo is imported: Python never contracts

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
try:
    import cimo  # noqa: F401
except ImportError:  # a cimo checkout next to this repository
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'cimo', 'python'))
