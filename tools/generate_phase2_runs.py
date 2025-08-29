#!/usr/bin/env python3
"""
Generate Phase 2 runs (Swift HAR path) for a list of texts to create
(input, target) training pairs for the post-filter.

- Reads tools/postfilter_texts.json (array of sentences)
- For each text, runs kokoro-phase2-cli with HAR bucket and saves output under outputs/local
- Assumes CLI uses a fixed fixture; we override by supplying text via env KOKORO_INPUT_TEXT when supported.
  If not supported, this script currently reuses the existing fixture but still increases run diversity if
  other inputs vary (to be extended to dynamic fixtures).
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--texts', default='tools/postfilter_texts.json')
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--cli', default=None, help='Path to kokoro-phase2-cli (optional)')
    ap.add_argument('--out', default='outputs/local', help='Output base directory')
    args = ap.parse_args()

    texts = json.loads(Path(args.texts).read_text())
    texts = texts[: args.limit]

    repo_root = Path(__file__).resolve().parents[1]
    cli_path = Path(args.cli) if args.cli else repo_root / 'Swift' / 'KokoroPhase2' / '.build' / 'arm64-apple-macosx' / 'release' / 'kokoro-phase2-cli'
    fixture = repo_root / 'Swift' / 'KokoroPhase2' / 'Fixtures' / 'fixture_har_5s.json'
    model = repo_root / 'coreml' / 'KokoroDecoder_HAR.mlpackage'
    outdir = repo_root / args.out

    if not cli_path.exists():
        raise SystemExit(f"kokoro-phase2-cli not found at {cli_path}. Build with: swift build -c release (in Swift/KokoroPhase2)")
    if not fixture.exists():
        raise SystemExit(f"fixture not found: {fixture}")
    if not model.exists():
        raise SystemExit(f"HAR model not found: {model}")

    for i, text in enumerate(texts, 1):
        env = os.environ.copy()
        env['KOKORO_FORCE_HAR'] = '1'
        env['KOKORO_PHASE_SCALE'] = '0.3'
        env['KOKORO_DISABLE_POSTFILTER'] = '1'  # ensure inputs are pre-filter audio for training X
        # TODO(multi-text): Add dynamic fixture export per text and pass it in
        subprocess.run([
            str(cli_path),
            str(fixture),
            str(model),
            str(outdir)
        ], check=True, env=env)
        print(f'[{i}/{len(texts)}] generated run')


if __name__ == '__main__':
    main()
