#!/usr/bin/env python3
"""Download CoreML models and voice files from Hugging Face Hub.

Models are stored on HF instead of Git LFS to avoid storage costs.
This script downloads them to the local paths the pipeline expects.

Usage::

    python scripts/download_models.py              # download everything
    python scripts/download_models.py --coreml     # only coreml/ packages
    python scripts/download_models.py --voices     # only kokoro.js/voices/
    python scripts/download_models.py --force      # re-download even if present

The HF token is read from (in order):
  1. HF_TOKEN environment variable
  2. .env file in the repo root
  3. huggingface-cli login cache (~/.huggingface/token)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Hugging Face repo that holds the model artifacts.
HF_REPO_ID = "mattmireles/kokoro-coreml"

# Patterns for each download group.  Matched against HF repo file paths.
COREML_PATTERNS = [
    "coreml/**",
    "coreml_fp32/**",
    "KokoroVocoder.mlpackage/**",
]
VOICE_PATTERNS = [
    "kokoro.js/voices/*.bin",
]
# Files to always skip (not model artifacts).
IGNORE_PATTERNS = [
    ".DS_Store",
    "*.py",
    "*.md",
    "*.txt",
    "*.json",       # repo-level config, not model weights
    "*.yml",
    "*.yaml",
    "*.sh",
    "*.toml",
    ".claude/**",
    ".github/**",
    ".gitignore",
    ".gitattributes",
    ".venv*/**",
    "kokoro/**",
    "kokoro.js/src/**",
    "kokoro.js/demo/**",
    "kokoro.js/tests/**",
    "kokoro.js/*.json",
    "kokoro.js/*.js",
    "kokoro.js/*.ts",
    "kokoro.js/*.md",
    "tests/**",
    "scripts/**",
    "examples/**",
    "export_synth/**",
    "export_duration.py",
    "export_synthesizers.py",
    "convert_checkpoint.py",
    "README/**",
    "outputs/**",
]


def _load_token() -> str | None:
    """Load HF token from env, .env file, or huggingface-cli cache."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = _REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None  # Falls back to huggingface-cli login cache.


def download(
    allow_patterns: list[str],
    token: str | None,
    force: bool = False,
) -> str:
    """Download matching files from the HF repo to the repo root."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(_REPO_ROOT),
        allow_patterns=allow_patterns,
        ignore_patterns=IGNORE_PATTERNS,
        force_download=force,
        token=token,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CoreML models and voices from Hugging Face Hub.",
    )
    parser.add_argument(
        "--coreml", action="store_true",
        help="Download only CoreML .mlpackage files",
    )
    parser.add_argument(
        "--voices", action="store_true",
        help="Download only kokoro.js voice .bin files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist locally",
    )
    args = parser.parse_args()

    token = _load_token()
    if token:
        print(f"Using HF token: {token[:8]}...")
    else:
        print("No HF_TOKEN found; using anonymous access (may fail for private repos).")

    # Determine which patterns to download.
    if args.coreml and not args.voices:
        patterns = COREML_PATTERNS
        label = "CoreML models"
    elif args.voices and not args.coreml:
        patterns = VOICE_PATTERNS
        label = "voice files"
    else:
        patterns = COREML_PATTERNS + VOICE_PATTERNS
        label = "all models and voices"

    print(f"\nDownloading {label} from {HF_REPO_ID}...")
    print(f"  Patterns: {patterns}")
    print(f"  Target:   {_REPO_ROOT}")
    print()

    try:
        result_dir = download(patterns, token, force=args.force)
        print(f"\nDownload complete: {result_dir}")
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Verify key artifacts exist.
    checks = [
        ("coreml/kokoro_duration.mlpackage", "Duration model"),
        ("coreml/kokoro_decoder_har_post_3s.mlpackage", "HAR-post 3s"),
        ("coreml/kokoro_decoder_har_post_10s.mlpackage", "HAR-post 10s"),
    ]
    all_ok = True
    for rel_path, label in checks:
        p = _REPO_ROOT / rel_path
        if p.exists():
            print(f"  {label}: {rel_path}")
        else:
            print(f"  MISSING: {label} ({rel_path})")
            all_ok = False

    if all_ok:
        print("\nAll key models present. Pipeline is ready.")
    else:
        print("\nSome models missing. Check the download output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
