#!/usr/bin/env python3
"""CLI wrapper for synthetic PID governor dataset generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.pid_governor.dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
