"""Make direct test-file execution import the project package.

When running a test as ``python tests/test_stage0_parser.py``, Python puts the
tests directory on sys.path instead of the repository root. This keeps that
developer convenience working without changing parser package code.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

