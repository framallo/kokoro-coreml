#!/usr/bin/env python3
"""Shared tensor dump I/O for audio parity debugging.

The format is intentionally small and language-neutral:

``tensor_manifest.json`` records tensor names, dtypes, shapes, and raw file
paths. Tensor payloads are little-endian ``float32`` or ``int32`` binaries so
Swift can write them without a NumPy dependency and Python can load them
directly.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
MANIFEST_NAME = "tensor_manifest.json"


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "tensor"


def _summary(array: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(array).reshape(-1)
    if flat.size == 0:
        return {"count": 0}
    if np.issubdtype(flat.dtype, np.floating):
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return {
                "count": int(flat.size),
                "finite_count": 0,
                "nan_count": int(np.isnan(flat).sum()),
                "inf_count": int(np.isinf(flat).sum()),
            }
        return {
            "count": int(flat.size),
            "finite_count": int(finite.size),
            "nan_count": int(np.isnan(flat).sum()),
            "inf_count": int(np.isinf(flat).sum()),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "l2": float(np.sqrt(np.mean(finite.astype(np.float64) ** 2))),
        }
    flat64 = flat.astype(np.int64)
    return {
        "count": int(flat.size),
        "min": int(flat64.min()),
        "max": int(flat64.max()),
        "mean": float(flat64.mean()),
    }


class TensorDumpWriter:
    """Write tensors to a parity dump directory."""

    def __init__(self, directory: Path | str, metadata: dict[str, Any] | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata: dict[str, Any] = metadata or {}
        self.records: list[dict[str, Any]] = []

    def write(self, name: str, array: Any) -> None:
        arr = np.asarray(array)
        if np.issubdtype(arr.dtype, np.integer):
            dtype = "int32"
            payload = arr.astype("<i4", copy=False)
            suffix = "i32"
        else:
            dtype = "float32"
            payload = arr.astype("<f4", copy=False)
            suffix = "f32"

        filename = f"{_safe_name(name)}.{suffix}"
        payload.tofile(self.directory / filename)
        self.records.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": [int(v) for v in arr.shape],
                "path": filename,
                "summary": _summary(payload),
            }
        )

    def close(self, metadata: dict[str, Any] | None = None) -> Path:
        if metadata:
            self.metadata.update(metadata)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "metadata": self.metadata,
            "tensors": self.records,
        }
        path = self.directory / MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return path


def load_tensor_dump(directory: Path | str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load a tensor dump directory created by Python or Swift."""

    root = Path(directory)
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported tensor dump schema: {manifest.get('schema_version')!r}")

    tensors: dict[str, np.ndarray] = {}
    for record in manifest.get("tensors", []):
        dtype = record["dtype"]
        if dtype == "float32":
            np_dtype = np.dtype("<f4")
        elif dtype == "int32":
            np_dtype = np.dtype("<i4")
        else:
            raise ValueError(f"unsupported tensor dtype for {record['name']}: {dtype}")

        shape = tuple(int(v) for v in record["shape"])
        arr = np.fromfile(root / record["path"], dtype=np_dtype)
        expected = math.prod(shape) if shape else 1
        if int(arr.size) != int(expected):
            raise ValueError(
                f"tensor {record['name']} has {arr.size} values, expected {expected} for shape {shape}"
            )
        tensors[record["name"]] = arr.reshape(shape)

    return manifest, tensors
