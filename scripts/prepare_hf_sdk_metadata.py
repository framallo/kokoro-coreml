#!/usr/bin/env python3
"""Prepare the Hugging Face SDK metadata payload for KokoroTTS releases.

This script intentionally uploads only lightweight metadata and docs:

- README.md, sourced from README/hf-model-card.md.
- Top-level HostedManifest.json and KokoroRuntimeManifest.json for the starter
  profile, preserving the current public discovery contract.
- sdk/<profile>/HostedManifest.json and sdk/<profile>/KokoroRuntimeManifest.json
  for each checked bundle profile.
- sdk/SDKReleaseManifest.json, which records checksums and profile summaries.

Model packages and voice binaries remain the canonical large artifacts already
hosted in the Hugging Face model repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "mattmireles/kokoro-coreml"


@dataclass(frozen=True)
class ProfileInput:
    """Input paths and metadata for one SDK bundle profile."""

    name: str
    bundle: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for payload preparation and upload."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--output", required=True, type=Path, help="Directory for the prepared payload")
    parser.add_argument("--starter-bundle", required=True, type=Path, help="Validated starter SDK bundle")
    parser.add_argument("--full-bundle", required=True, type=Path, help="Validated full SDK bundle")
    parser.add_argument(
        "--model-card",
        default=REPO_ROOT / "README" / "hf-model-card.md",
        type=Path,
        help="Model card markdown to upload as README.md",
    )
    parser.add_argument(
        "--sdk-commit",
        help="Expected SDK commit. Defaults to git rev-parse HEAD in this checkout.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the prepared payload to the Hugging Face repo.",
    )
    return parser.parse_args()


def git_head() -> str:
    """Return the current Git HEAD commit for manifest compatibility checks."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(root: Path, path: Path) -> dict[str, Any]:
    """Return a checksum record for one file relative to a payload root."""

    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_hf_token() -> str | None:
    """Load an HF token using the same env/.env/cache order as the downloader."""

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("HF_TOKEN="):
                return stripped.split("=", 1)[1].strip()
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


def copy_manifest_pair(profile: ProfileInput, output: Path, sdk_commit: str) -> dict[str, Any]:
    """Copy profile manifests into the payload and return release metadata."""

    runtime_src = profile.bundle / "KokoroRuntimeManifest.json"
    hosted_src = profile.bundle / "HostedManifest.json"
    if not runtime_src.exists() or not hosted_src.exists():
        raise SystemExit(f"{profile.name} bundle is missing required manifests: {profile.bundle}")

    runtime = load_json(runtime_src)
    hosted = load_json(hosted_src)
    observed_commit = runtime.get("sdk_commit")
    if observed_commit != sdk_commit:
        raise SystemExit(
            f"{profile.name} manifest sdk_commit mismatch: expected {sdk_commit}, observed {observed_commit}"
        )

    profile_dir = output / "sdk" / profile.name
    profile_dir.mkdir(parents=True, exist_ok=True)
    runtime_dest = profile_dir / "KokoroRuntimeManifest.json"
    hosted_dest = profile_dir / "HostedManifest.json"
    shutil.copy2(runtime_src, runtime_dest)
    shutil.copy2(hosted_src, hosted_dest)

    return {
        "profile": profile.name,
        "sdk_commit": observed_commit,
        "hf_repo_id": runtime.get("hf_repo_id"),
        "hf_revision": runtime.get("hf_revision"),
        "hosted_version": hosted.get("version"),
        "minimum_platforms": runtime.get("minimum_platforms"),
        "buckets": runtime.get("buckets"),
        "duration_token_sizes": runtime.get("duration_token_sizes"),
        "model_package_count": len(runtime.get("model_packages") or []),
        "voice_count": len(runtime.get("voices") or []),
        "runtime_manifest": file_record(output, runtime_dest),
        "hosted_manifest": file_record(output, hosted_dest),
    }


def prepare_payload(args: argparse.Namespace) -> Path:
    """Create a deterministic Hugging Face metadata payload directory."""

    sdk_commit = args.sdk_commit or git_head()
    output = args.output.resolve()
    if output == REPO_ROOT or REPO_ROOT in output.parents:
        raise SystemExit("refusing to write HF payload inside the repo checkout")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    readme_dest = output / "README.md"
    shutil.copy2(args.model_card, readme_dest)

    profiles = [
        ProfileInput("starter", args.starter_bundle.resolve()),
        ProfileInput("full", args.full_bundle.resolve()),
    ]
    profile_records = [copy_manifest_pair(profile, output, sdk_commit) for profile in profiles]

    starter_profile_dir = output / "sdk" / "starter"
    shutil.copy2(starter_profile_dir / "HostedManifest.json", output / "HostedManifest.json")
    shutil.copy2(starter_profile_dir / "KokoroRuntimeManifest.json", output / "KokoroRuntimeManifest.json")

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "sdk_commit": sdk_commit,
        "model_card": file_record(output, readme_dest),
        "top_level_hosted_manifest": file_record(output, output / "HostedManifest.json"),
        "top_level_runtime_manifest": file_record(output, output / "KokoroRuntimeManifest.json"),
        "profiles": profile_records,
    }
    release_manifest = output / "sdk" / "SDKReleaseManifest.json"
    release_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"prepared HF SDK metadata payload at {output}")
    print(f"  sdk_commit={sdk_commit}")
    for profile in profile_records:
        print(
            f"  {profile['profile']}: models={profile['model_package_count']} "
            f"voices={profile['voice_count']} hosted={profile['hosted_version']}"
        )
    return output


def upload_payload(repo_id: str, payload: Path) -> None:
    """Upload the prepared metadata payload to Hugging Face Hub."""

    from huggingface_hub import HfApi

    token = load_hf_token()
    if not token:
        raise SystemExit("HF upload requires HF_TOKEN, .env HF_TOKEN, or a Hugging Face login cache")
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(payload),
        commit_message="Publish KokoroTTS SDK metadata",
    )
    print(f"uploaded HF SDK metadata payload to {repo_id}")


def main() -> None:
    """Prepare and optionally upload the HF SDK metadata payload."""

    args = parse_args()
    payload = prepare_payload(args)
    if args.upload:
        upload_payload(args.repo_id, payload)


if __name__ == "__main__":
    main()
