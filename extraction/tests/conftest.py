"""Make `psm_extraction` importable when running pytest from the repo root.

Adds `extraction/` to sys.path so tests can run without `pip install -e .`.
"""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
