import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Make python/ packages, the pure-Python engine, and the compiled C++ module importable
for path in (ROOT, os.path.join(ROOT, 'python'), os.path.join(ROOT, 'python', 'train'),
             os.path.join(ROOT, 'build')):
    if path not in sys.path:
        sys.path.insert(0, path)
