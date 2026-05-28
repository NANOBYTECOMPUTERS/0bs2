#!/usr/bin/env python3
"""Generate a neural tracker training dataset."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.neural_tracker.dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
