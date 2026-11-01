import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
try:
    import cimo  # noqa: F401
except ImportError:  # a cimo checkout next to this repository
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'cimo', 'python'))
