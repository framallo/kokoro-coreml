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
    args = ap.parse_args()

    texts = json.loads(Path(args.texts).read_text())
    texts = texts[: args.limit]

    for i, text in enumerate(texts, 1):
        env = os.environ.copy()
        env['KOKORO_FORCE_HAR'] = '1'
        env['KOKORO_PHASE_SCALE'] = '0.3'
        # TODO(multi-text): Add dynamic fixture export per text and pass it in
        subprocess.run([
            './Swift/KokoroPhase2/.build/arm64-apple-macosx/release/kokoro-phase2-cli',
            './Swift/KokoroPhase2/Fixtures/fixture_har_5s.json',
            './coreml/KokoroDecoder_HAR.mlpackage',
            './outputs/local'
        ], check=True, env=env)
        print(f'[{i}/{len(texts)}] generated run')


if __name__ == '__main__':
    main()
