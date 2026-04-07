#!/usr/bin/env python3
"""Shim: use ``python examples/example_synthesis.py`` (same CLI).

Kept so older docs and muscle memory for ``run_single.py`` still work.
"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "examples" / "example_synthesis.py"
    runpy.run_path(str(target), run_name="__main__")
