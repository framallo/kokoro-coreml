# Kokoro Drop-In SDK v1 Notes

Institutional memory for the drop-in Swift SDK implementation. Keep local
execution evidence, backend decisions, drift reports, and rejected paths here;
do not put local-only analysis in `README/Guides/`.

**Quick filter:** `grep -n "— Active" README/Notes/kokoro-drop-in-sdk-v1.md`

---

## Issue: Phase 2 Runtime Asset Source of Truth - Resolved

**First spotted:** 2026-06-28
**Resolved:** 2026-06-28
**Status:** Resolved

### Summary

Phase 2 made the small SDK runtime inputs checked, hashed, and independently
verifiable. The SDK package now bundles `kokoro-vocab.json`,
`hnsf_weights.json`, and `KokoroRuntimeAssets.json` under
`swift-tts/Sources/KokoroTTS/Resources/KokoroRuntime/`. `KokoroPipeline` stays
unchanged; these resources belong to the higher-floor raw-text SDK package.

The old root `hnsf_weights.json` carried `"weights_sha256": "unverified"`.
Phase 2 promoted the verified copy from
`ios-bench/Resources/bench_inputs/hnsf_weights.json` to both the SDK resource
and root file so repo-local tools no longer default to unverified weights.

### Hashes

| Asset | SHA-256 | Notes |
| --- | --- | --- |
| `swift-tts/.../kokoro-vocab.json` | `353ca94410fde4575cb091a0ba32b8e99077fde4f38fded506f4f041d22571a3` | Byte copy from Gist iOS `Resources/Kokoro/kokoro-vocab.json`. |
| SDK vocab canonical JSON | `c888d4ef5abac125fcd45201e54aa6bf512722bcc4213f76bb053edb056923de` | Matches `_kokoro_vocab.json` and `ios-bench/Vendor/kokoro-ios/Resources/config.json:vocab` semantically. |
| `swift-tts/.../hnsf_weights.json` | `de73b717732da77b31736f67a108d35c478ab116b0a188e9787019b0408c0226` | Matches root `hnsf_weights.json` and `ios-bench/Resources/bench_inputs/hnsf_weights.json` byte-for-byte. |
| hn-NSF internal `weights_sha256` | `25a471a6fc81fc9c5ff7c46e4be9d9ec3710dbbfea6e121a99fac75e4a97ad99` | Replaces the old `"unverified"` marker. |

### Guardrails

`scripts/verify_runtime_assets.py` is the Phase 2 gate. It rejects missing
files, symlinked SDK resources or checked comparison files, malformed vocab
JSON, vocab canonical-hash drift, hn-NSF byte-hash drift, and
`weights_sha256: "unverified"`.

`checkpoints/config.json` remains a local symlink and is deliberately not used
as an SDK source. The verifier compares SDK vocab against `_kokoro_vocab.json`
and the checked iOS bench config, not the machine-local checkpoint symlink.

SwiftPM flattens processed resources in some test builds. `KokoroRuntimeAssets`
therefore checks `KokoroRuntime/<file>` first and then falls back to the
flattened bundle root while keeping the source tree organized under
`Resources/KokoroRuntime`.

### Verification

Regression test:

```bash
python3 scripts/verify_runtime_assets.py
```

Result on 2026-06-28: passed with the SDK vocab and hn-NSF hashes above.
Temporary negative checks also passed: replacing the SDK vocab with a symlink
failed as expected, replacing SDK `weights_sha256` with `"unverified"` failed
as expected, and the restored files passed the verifier again.

Regression test:

```bash
swift test --package-path swift-tts
```

Result on 2026-06-28: passed 7 tests with 2 Misaki runtime tests skipped by
default. The new resource tests prove the package bundle exposes the manifest,
vocab, and verified hn-NSF weights.

Regression test:

```bash
swift test --package-path swift
```

Result on 2026-06-28: passed 45 tests, confirming the low-level prepared-input
pipeline still builds after promoting the root hn-NSF weights file.

### If This Recurs

- [ ] Run `python3 scripts/verify_runtime_assets.py` before building SDK
      bundles.
- [ ] Do not use `checkpoints/config.json` as an SDK vocab source while it is a
      symlink.
- [ ] Keep `KokoroRuntimeAssets.json` hashes in lockstep with the checked SDK
      resource files.
- [ ] Preserve the `KokoroRuntimeAssets` flattened-resource fallback unless the
      package build is proven to preserve subdirectories across SwiftPM and
      Xcode.

---

## Issue: Phase 1 `swift-tts` Package Boundary - Resolved

**First spotted:** 2026-06-28
**Resolved:** 2026-06-28
**Status:** Resolved

### Summary

Phase 1 created a separate raw-text Swift package at `swift-tts/` so
`KokoroPipeline` can keep its lower iOS 16 / macOS 13 prepared-input boundary.
The new package depends on `../swift`, imports `KokoroPipeline`, pins the same
MisakiSwift fork/revision used by Gist, and exposes only the phonemizer spike,
diagnostics policy, and package facade placeholder. It does not expose
`KokoroTTS.synthesize(text:)` yet.

### Dependency Decision

**Package:** `https://github.com/mattmireles/MisakiSwift`
**Revision:** `3a27756a780fc138e328a96e533fb440a3419d5b`
**License:** Apache-2.0 from the resolved checkout's `LICENSE`
**SDK floor:** `swift-tts` is macOS 15.0 / iOS 18.0. The low-level `swift/`
package remains macOS 13 / iOS 16 and has no MisakiSwift dependency.

This follows Gist's working app-level package decision. The fork is required
because upstream MisakiSwift 1.0.0-1.0.6 copied resources as a shallow
top-level `Resources` directory, which Gist found breaks iOS code signing. The
fork copies resources under `MisakiData`, and Phase 1 observed that layout in
both SwiftPM and xcodebuild products:

```text
MisakiSwift_MisakiSwift.bundle/MisakiData/us_bart_config.json
MisakiSwift_MisakiSwift.bundle/MisakiData/us_bart.safetensors
MisakiSwift_MisakiSwift.bundle/MisakiData/gb_bart_config.json
MisakiSwift_MisakiSwift.bundle/MisakiData/gb_bart.safetensors
```

MisakiSwift declares a dynamic library product. App integrations must embed
`MisakiSwift.framework`; Gist's XcodeGen spec already documents this as a
launch-time requirement.

### Packaging Findings

MisakiSwift is not lexicon-only. `EnglishG2P` constructs
`EnglishFallbackNetwork`, which imports MLX and loads BART fallback weights.
That creates two practical SDK rules:

- Plain `swift run` / shell `swift test` can compile the package, but runtime
  Misaki calls fail unless the MLX `mlx-swift_Cmlx.bundle/default.metallib` is
  built and discoverable.
- xcodebuild is the correct proof path for app developers because it builds the
  MLX shader bundle. On this machine, `xcodebuild -downloadComponent
  MetalToolchain` was required before xcodebuild could compile the Metal
  shaders.

MLX's own docs state iOS Simulator cannot be used to run MLX applications.
Phase 1 therefore treats iOS Simulator as compile/resource validation only.
Runtime phonemization proof is macOS now; iPhone runtime proof remains a later
mandatory physical-device gate before iOS release readiness.

### Drift Table

Generated with:

```bash
node scripts/compare_misaki_botnet_phonemes.mjs \
  --probe-bin /tmp/kokoro-tts-dd/Build/Products/Debug/kokoro-misaki-probe \
  --dyld-framework-path /tmp/kokoro-tts-dd/Build/Products/Debug/PackageFrameworks
```

| Text | Misaki Swift | Botnet JS/eSpeak | Drift | Voice row consequence |
| --- | --- | --- | --- | --- |
| Empty string |  |  | empty output | Misaki throws `emptyOutput`; Botnet length 0 |
| Hello world. | həlˈO wˈɜɹld. | həlˈoʊ wˈɜːld. | phoneme drift | Misaki 13, Botnet 14 |
| Dr. Smith paid $12.50 for apples. | dˈɑktəɹ smˈɪθ pˈAd  fɔɹ ˈæpᵊlz. | dˈɑːktɚ smˈɪθ pˈeɪd twˈɛlv dˈɑːlɚz ænd fˈɪfti sˈɛnts fɔːɹ ˈæpəlz. | phoneme and normalization drift | Misaki 31, Botnet 65 |
| Visit https://example.com, then email me@example.com. | vˈɪzət t:ɪɡzˈæmpəlkˌɑm, ðˈɛn ˈimˌAl mˌiɪɡzˈæmpəlkˌɑm. | vˈɪzɪt ˌeɪtʃtˌiːtˈiːpˌiːˈɛs:slˈæʃslæʃ ɛɡzˈæmpəlkˈɑːm, ðˈɛn ˈiːmeɪl mˌiː æt ɛɡzˈæmpəlkˈɑːm. | phoneme and normalization drift | Misaki 53, Botnet 90 |
| I live in Reading. | ˌI lˈɪv ɪn ɹˈidɪŋ. | aɪ lˈɪv ɪn ɹˈiːdɪŋ. | phoneme drift | Misaki 18, Botnet 19 |

No fixture was an exact match. This means Swift prep must not pretend Misaki is
byte-identical to Botnet/eSpeak. Later phases must evaluate whether this drift
is acceptable perceptually, whether Gist-compatible Misaki behavior should be
the SDK default, and whether an eSpeak-compatible backend is needed for strict
fleet parity.

### Diagnostics Policy

`KokoroDiagnosticsPolicy.privacySafeDefault` allows counters, timings, stable
hashes, model identifiers, and typed error codes. Raw text and phoneme strings
require explicit caller opt-in through `interactiveDebugPayloads`. The SDK
policy refuses raw-payload persistence.

### Verification

Regression test:

```bash
swift test --package-path swift-tts
```

Result on 2026-06-28: passed 5 tests with 2 Misaki runtime tests skipped by
default. The skipped tests require `KOKORO_RUN_MISAKI_RUNTIME_TESTS=1` only
when MLX shader resources are available.

Regression test:

```bash
swift test --package-path swift
```

Result on 2026-06-28: passed 45 tests. This proves the lower-floor
`KokoroPipeline` package still builds without MisakiSwift.

Regression test:

```bash
xcodebuild -scheme kokoro-misaki-probe \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /tmp/kokoro-tts-dd build
```

Result on 2026-06-28: succeeded after installing Xcode's Metal Toolchain.

Runtime probe:

```bash
DYLD_FRAMEWORK_PATH=/tmp/kokoro-tts-dd/Build/Products/Debug/PackageFrameworks \
  /tmp/kokoro-tts-dd/Build/Products/Debug/kokoro-misaki-probe \
  'Hello world.'
```

Result on 2026-06-28: produced non-empty Misaki phonemes offline.

Regression test:

```bash
xcodebuild -scheme KokoroTTS \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /tmp/kokoro-tts-iossim-dd build
```

Result on 2026-06-28: succeeded for generic iOS Simulator compile/resource
validation. The machine also reported CoreSimulator
`1051.54.0 < 1051.55.0`, so named simulator execution is unavailable here.
Independent of that local mismatch, MLX documents that iOS Simulator cannot run
MLX apps, so physical-device runtime proof remains required.

### If This Recurs

- [ ] Check that Xcode's Metal Toolchain is installed:
      `xcodebuild -downloadComponent MetalToolchain`.
- [ ] Use xcodebuild, not plain `swift run`, for runtime Misaki/MLX probes.
- [ ] Embed `MisakiSwift.framework` in app targets.
- [ ] Treat iOS Simulator as compile/resource proof only; use a physical iPhone
      for runtime phonemization and synthesis proof.

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
