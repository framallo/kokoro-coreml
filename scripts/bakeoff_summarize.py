#!/usr/bin/env python3
"""Bakeoff summarize mode -- read results and emit tables plus gate answers.

Separated from bakeoff_harness.py per the LOC guard (harness must stay <= 800).
This module has zero coupling to benchmark contexts or model loading.

Usage (standalone)::

    python scripts/bakeoff_summarize.py --results outputs/bakeoff/results_m2_ultra.json

Usage (via harness)::

    python scripts/bakeoff_harness.py summarize --results outputs/bakeoff/results_m2_ultra.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BAKEOFF_DIR = _REPO_ROOT / "outputs" / "bakeoff"


def cmd_summarize(args: argparse.Namespace) -> None:
    """Read results files and emit tables + gate answers."""
    raise NotImplementedError("summarize mode is implemented in Phase 5")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bakeoff Summarize")
    parser.add_argument("--results", required=True, nargs="+", help="Results JSON path(s)")
    args = parser.parse_args()
    cmd_summarize(args)


if __name__ == "__main__":
    main()
