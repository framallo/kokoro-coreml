#!/usr/bin/env python3
"""Inspect Hugging Face Kokoro Core ML artifacts without downloading them."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "mattmireles/kokoro-coreml"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the HF artifact inspector."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--revision", help="Optional HF revision/commit")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    return parser.parse_args()


def fetch_model_info(repo_id: str, revision: str | None) -> dict[str, Any]:
    """Fetch Hugging Face model metadata through the public API."""

    quoted = urllib.parse.quote(repo_id, safe="/")
    suffix = f"/revision/{urllib.parse.quote(revision, safe='')}" if revision else ""
    params = {"blobs": "true"}
    url = f"https://huggingface.co/api/models/{quoted}{suffix}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def package_name(path: str) -> str | None:
    """Return the `.mlpackage` path prefix for a repo file path."""

    marker = ".mlpackage/"
    if marker not in path:
        return None
    return path[: path.index(marker) + len(".mlpackage")]


def inspect_artifacts(info: dict[str, Any], repo_id: str, revision: str | None) -> dict[str, Any]:
    """Build a stable report from Hugging Face API metadata."""

    siblings = info.get("siblings") or []
    packages: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "file_count": 0,
        "bytes": 0,
        "files": [],
    })
    voices = []
    top_level_runtime_metadata = set()

    for sibling in siblings:
        path = sibling.get("rfilename")
        if not isinstance(path, str):
            continue
        size = int(sibling.get("size") or sibling.get("lfs", {}).get("size") or 0)
        pkg = package_name(path)
        if pkg:
            packages[pkg]["file_count"] += 1
            packages[pkg]["bytes"] += size
            packages[pkg]["files"].append({
                "path": path,
                "bytes": size,
                "sha256": sibling.get("lfs", {}).get("sha256"),
            })
        elif path.startswith("kokoro.js/voices/") and path.endswith(".bin"):
            voices.append({
                "path": path,
                "voice": Path(path).stem,
                "bytes": size,
                "sha256": sibling.get("lfs", {}).get("sha256"),
            })
        elif path in {"KokoroRuntimeManifest.json", "HostedManifest.json"}:
            top_level_runtime_metadata.add(path)

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": info.get("sha"),
        "last_modified": info.get("lastModified"),
        "private": info.get("private"),
        "gated": info.get("gated"),
        "sibling_count": len(siblings),
        "model_packages": [
            {
                "path": path,
                "file_count": value["file_count"],
                "bytes": value["bytes"],
                "files": sorted(value["files"], key=lambda item: item["path"]),
            }
            for path, value in sorted(packages.items())
        ],
        "voices": sorted(voices, key=lambda item: item["path"]),
        "missing_sdk_metadata": sorted(
            {"KokoroRuntimeManifest.json", "HostedManifest.json"} - top_level_runtime_metadata
        ),
    }
    return report


def main() -> None:
    """CLI entry point for the HF artifact inspector."""

    args = parse_args()
    info = fetch_model_info(args.repo_id, args.revision)
    report = inspect_artifacts(info, args.repo_id, args.revision)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
