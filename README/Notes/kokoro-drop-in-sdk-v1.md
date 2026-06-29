# Kokoro Drop-In SDK v1 Notes

Institutional memory for the drop-in Swift SDK implementation. Keep local
execution evidence, backend decisions, drift reports, and rejected paths here;
do not put local-only analysis in `README/Guides/`.

**Quick filter:** `grep -n "— Active" README/Notes/kokoro-drop-in-sdk-v1.md`

---

## Issue: Phase 0 Prep Self-Hosting — Resolved

**First spotted:** 2026-06-28
**Resolved:** 2026-06-28
**Status:** Resolved

### Summary

The copied prep script still defaulted to Botnet's
`packages/kokoro-coreml-runtime` layout, so this repo could not prepare text
from a clean checkout path. Phase 0 changed the JS prep bridge to prefer this
repo's runtime assets, added explicit runtime-root handling, and added a Botnet
comparison harness with edge-case fixtures.

### Symptom

```log
Error: Kokoro phonemizer not found under /Users/mm/Documents/GitHub/kokoro-coreml/packages/kokoro-coreml-runtime
```

### Root Cause

`scripts/kokoro-prepare-input.mjs` was copied from Botnet and retained Botnet's
default runtime root. The local checkout already has `kokoro.js/src`,
`kokoro.js/voices`, and `_kokoro_vocab.json`, but the script did not search
those paths first.

### Related Guides

- [Runtime boundary](../Wiki/runtime-boundary.md) - Defines prepared-input
  synthesis as the low-level Swift boundary.
- [Runtime boundary note](kokoro-runtime-boundary.md) - Records tokenizer,
  voice, and manifest responsibilities for native TTS runtimes.

### Fix

**Files:**

- `scripts/kokoro-prepare-input.mjs`
- `scripts/kokoro-prepare-input.py`
- `scripts/compare_botnet_prepare_input.mjs`
- `tests/fixtures/kokoro-text-prep/*.json`

The JS bridge now accepts `--runtime-root`, honors `KOKORO_COREML_ROOT`, and
defaults to the current repo when `kokoro.js/src/phonemize.js` exists. It loads
vocab from `_kokoro_vocab.json` before falling back to generated or legacy
config paths, and it can load voice rows from either `kokoro.js/voices` or
`voices`.

The Python bridge now also accepts `--runtime-root` and inserts the runtime root
into `sys.path` for direct script execution. It remains a dev/proof bridge,
not an SDK dependency.

Install the JS phonemizer dependency where Node resolves it:

```bash
npm --prefix kokoro.js ci
```

### Verification

Regression test:

```bash
node scripts/compare_botnet_prepare_input.mjs --botnet-root /Users/mm/Documents/GitHub/botnet --fixtures tests/fixtures/kokoro-text-prep/*.json --compare full
```

Result on 2026-06-28: 12 fixtures passed against Botnet, including empty text,
whitespace, abbreviations, initials, numbers, currency, URLs/emails, quotes,
punctuation runs, emoji, long text near the active-token cap, `af_heart`, and
British voice `bf_lily`.

Observed behavior: unsupported vocab symbols are dropped after phonemization,
then BOS/EOS framing and padding still apply. Empty text or whitespace-only
text fails because the phonemizer returns no phonemes. The emoji fixture passed
against Botnet, proving emoji does not poison the prepared-input contract.

Regression test:

```bash
npm --prefix kokoro.js test
```

Result on 2026-06-28: 276 tests passed.

Regression test:

```bash
tmp=$(mktemp)
printf 'Hello world' > "$tmp"
uv run python scripts/kokoro-prepare-input.py --runtime-root /Users/mm/Documents/GitHub/kokoro-coreml --text-file "$tmp" --output /tmp/kokoro-prep-py-uv.json --key smoke --voice af_heart --speed 1
```

Result on 2026-06-28: generated a 32-token padded input, 32-entry attention
mask, and 256-float `ref_s`. Direct `python3` outside `uv` can still fail if the
active interpreter lacks `misaki[en]` / `spacy`; use `uv run` for the Python
bridge proof.

### If This Recurs

- [ ] Verify `kokoro.js/node_modules/phonemizer` exists or run
      `npm --prefix kokoro.js ci`.
- [ ] Run the Botnet comparison harness before changing tokenizer behavior.
- [ ] Check that `_kokoro_vocab.json` and `kokoro.js/voices/<voice>.bin` exist
      before debugging Core ML.

---
