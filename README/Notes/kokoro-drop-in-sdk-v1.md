# Kokoro Drop-In SDK v1 Notes

Institutional memory for the drop-in Swift SDK implementation. Keep local
execution evidence, backend decisions, drift reports, and rejected paths here;
do not put local-only analysis in `README/Guides/`.

**Quick filter:** `grep -n "— Active" README/Notes/kokoro-drop-in-sdk-v1.md`

---

## Issue: Phase 6 Consumer Fixture and Smoke Foundation - In Progress

**First spotted:** 2026-06-28
**Status:** In progress

Phase 6 started with the smallest reliable proof that the SDK can be consumed
outside the implementation package:

- `swift-tts` `kokoro-sdk-smoke` now supports explicit `--bundle`, hosted
  `--manifest-url` plus `--cache-dir`, `--text`, `--voice`, and `--out`.
- The hosted mode works with `file://` manifests for local validation and uses
  the same `KokoroDownloadedModelStore` path as an app would use for HTTPS.
- WAV output is written as mono 16-bit little-endian PCM for quick listening or
  later `audio-judge` checks.
- `examples/KokoroConsumerFixture` is a standalone SwiftPM executable package
  with a local path dependency on `swift-tts` and imports only public
  `KokoroTTS` APIs.

Validation on 2026-06-28:

```bash
swift build --package-path examples/KokoroConsumerFixture

xcodebuild -quiet \
  -scheme KokoroConsumerFixture \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /tmp/kokoro-consumer-fixture-dd \
  build

DYLD_FRAMEWORK_PATH=/tmp/kokoro-consumer-fixture-dd/Build/Products/Debug/PackageFrameworks \
  /tmp/kokoro-consumer-fixture-dd/Build/Products/Debug/kokoro-consumer-fixture \
  --bundle /tmp/kokoro-sdk-starter-compiled \
  --text 'Hello world.' \
  --out /tmp/kokoro-consumer-fixture.wav

xcodebuild -quiet \
  -scheme KokoroConsumerFixture \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /tmp/kokoro-consumer-fixture-ios-dd \
  build
```

Results: macOS consumer fixture printed
`samples=37800 sampleRate=24000 duration=1.575` and wrote a 75,644-byte WAV.
Generic iOS Simulator compile succeeded. The machine emits a CoreSimulator
version warning (`1051.54.0` job vs `1051.55.0` framework), but the generic
simulator build still completed.

The paired devices visible to `xcrun devicectl list devices` are:

- `Commas?`, iPhone 15 Pro Max, available paired
- `Webcam`, iPhone 12 Pro, available paired

Do not mark Phase 6 or iOS readiness complete yet. The next slice must add the
remaining warm-call, long-text, cancellation, memory, and
background/foreground checks on a physical iPhone.

### Demo App Evidence

`examples/KokoroDemoApp` was added as an XcodeGen iOS app target. It uses the
public `KokoroTTS` API, a manifest URL field, text editor, voice picker,
synthesize button, status text, and `AVAudioEngine` playback. Launch arguments
support device automation:

```bash
xcodegen generate --spec examples/KokoroDemoApp/project.yml
xcodebuild -quiet \
  -scheme KokoroDemoApp \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /tmp/kokoro-demo-ios-sim-dd \
  build
xcodebuild -quiet \
  -scheme KokoroDemoApp \
  -destination 'id=8A12AEE8-0136-50BE-8EB3-91650E467F15' \
  -derivedDataPath /tmp/kokoro-demo-device-dd \
  build
xcrun devicectl device install app \
  --device 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  /tmp/kokoro-demo-device-dd/Build/Products/Debug-iphoneos/KokoroDemoApp.app
xcrun devicectl device process launch \
  --device 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  --terminate-existing \
  --console \
  --timeout 120 \
  com.mattmireles.KokoroDemoApp \
  --auto-run \
  --manifest-url http://192.168.4.47:8766/HostedManifest.json \
  --text 'Hello world.'
```

First device run failed with `NSURLErrorDomain Code=-1009` and
`Local network prohibited`. Adding `NSLocalNetworkUsageDescription` to the
generated Info.plist fixed the local-network permission path.

Second device run on `Commas?` (iPhone 15 Pro Max) succeeded:

```text
KOKORO_DEMO_DONE samples=37800 sampleRate=24000 duration=1.575
```

The local HTTP server observed the phone at `192.168.4.32` downloading
`HostedManifest.json`, `KokoroRuntimeManifest.json`, all starter duration
packages, `kokoro_f0ntrain_t600`, 15-second decoder-pre/HAR-post packages,
`runtime/hnsf_weights.json`, `runtime/kokoro-vocab.json`, and
`voices/af_heart.bin`.

---

## Issue: Phase 5 SDK Facade, Provider, and Runtime Smoke - Resolved

**First spotted:** 2026-06-28
**Resolved:** 2026-06-28
**Status:** Resolved

### Summary

Phase 5 added the first real drop-in API:

```swift
let tts = try await KokoroTTS.load(resources: .appBundle(.main))
let audio = try await tts.synthesize("Hello world.", voice: .afHeart)
```

The public facade is a loaded-only actor over `KokoroTextProcessor`,
`KokoroResourceProvider`, and `KokoroSDKModelProvider`. It supports explicit
generated-bundle directories, app bundles, package bundles, and downloaded cache
bundles. Generated bundle roots may include a `compiled/` directory, but V1 does
not support precompiled-only roots without the source `.mlpackage` manifests.
`KokoroSDKModelProvider` validates runtime/voice hashes at load, validates model
package tree hashes lazily before Core ML compiles or loads a selected model,
caches loaded `MLModel` instances, persists compiled `.mlmodelc` output when the
resource root is writable, and exposes `prewarm(...)`.

### Phase 5 Cross-Agent Audit

A read-only Codex cross-agent audit was run against the current uncommitted
Phase 5 diff on 2026-06-28. It graded the first pass:

```text
Architecture: C
Correctness risk: C
Complexity debt: B
Commit-readiness: No
```

The audit findings were valid:

- Hosted manifest paths in `KokoroDownloadedModelStore` needed stronger
  containment checks before local writes and remote URL construction.
- `public init()` made an unusable `KokoroTTS()` facade possible.
- The typed-error checklist overstated missing-voice and unsupported-language
  behavior.
- `.precompiledDirectory` implied precompiled-only support even though runtime
  validation still requires source `.mlpackage` manifests.
- The downloaded-store checklist claimed local version comparison, but the first
  implementation only validated file size/hash.

Fixes applied after the audit: reject absolute, `..`, empty, and backslash path
components in hosted manifests; clear the compiled cache when hosted bundle
version changes; remove the dead public initializer; map missing/malformed voice
assets to public `KokoroError` cases; remove the misleading precompiled-only
resource-provider case; and update the plan to describe the actual supported
bundle shape.

A second read-only cross-agent pass found additional correctness blockers:
compiled `.mlmodelc` cache reuse was trusted by sidecar alone, runtime/cache
parent symlinks were not rejected before all reads, hosted-download cancellation
could be swallowed by retry logic, `maxChunkSeconds` was ignored by the facade,
and model-set consistency was not fail-fast at load.

Fixes applied after the second pass: make `KokoroSDKModelProvider` internal so
the public app API cannot call synchronous model loading directly; require
compiled caches to live under the bundle root with non-symlink components and a
matching source-tree sidecar; reject symlinked parent components for runtime,
voice, downloaded-cache, and model-package paths before reads or writes;
validate duration and per-bucket stage packages at load; make hosted downloads
preserve cancellation; honor `KokoroSynthesisOptions.maxChunkSeconds`; and add
targeted tests for those edge cases.

A final focused cross-agent pass found release-safety issues in the bundle
builder and SDK surface: destructive `--output` deletion, local artifact
provenance being stamped as HF provenance without verification, hosted manifests
including platform-compiled caches, symlinked resource roots, unsupported
manifest schemas, stale starter voice constants, and British voices using the
U.S. Misaki path. Fixes applied: require a download manifest unless
`--allow-local-provenance 1` is explicit, add a bundle marker and dangerous-path
deletion guard, exclude `compiled/` from `HostedManifest.json`, reject symlinked
bundle/cache roots, enforce schema version 1, align starter constants to
`af_heart`, and choose Misaki British mode for `b*` voices.

### Starter Bundle Fix

The first macOS smoke attempt exposed a bad starter default: the starter bundle
included only `kokoro_duration_t512.mlpackage`, so even `"Hello world."` loaded
the largest duration model. Core ML spent minutes in E5 AOT specialization:

```text
KokoroSDKModelProvider.durationModel
MLModel(contentsOf:configuration:)
MLE5ProgramLibraryOnDeviceAOTCompilationImpl
e5rt_e5_compiler_compile_from_ir_program
```

Precompiling with `xcrun coremlcompiler` and forcing `.cpuOnly` did not fix the
t512 load cost. The simple fix was to make starter/custom bundles include the
full padded duration ladder `[32, 64, 128, 256, 320, 384, 512]`, while keeping
the starter synthesis bucket at 15 seconds and `af_heart`. After that, short
text selects `kokoro_duration_t32`.

### Verification

Rebuilt compiled starter bundle:

```bash
python3 scripts/download_models.py \
  --repo-id mattmireles/kokoro-coreml \
  --revision c02933e179932e51909ff3b29466a7debac7d0e6 \
  --sdk-profile starter \
  --manifest-out /tmp/kokoro-download-manifest.json

node scripts/build_sdk_bundle.mjs \
  --profile starter \
  --compile-models 1 \
  --output /tmp/kokoro-sdk-starter-compiled \
  --repo-id mattmireles/kokoro-coreml \
  --revision c02933e179932e51909ff3b29466a7debac7d0e6 \
  --download-manifest /tmp/kokoro-download-manifest.json
```

Result on 2026-06-28: built a starter bundle with 10 model packages, 1 voice,
and 34 hosted files. The local `compiled/` cache is present for the smoke bundle
but is intentionally excluded from `HostedManifest.json`.

Bundle validation:

```bash
node scripts/validate_sdk_bundle.mjs /tmp/kokoro-sdk-starter-compiled
```

Result on 2026-06-28: passed.

Swift tests:

```bash
swift test --package-path swift-tts
swift test --package-path swift
```

Result on 2026-06-28: `swift-tts` passed 40 tests with 2 MLX runtime tests
skipped by design; `swift` passed 46 tests after the SDK facade landed.

Xcode-built macOS smoke:

```bash
xcodebuild -quiet \
  -scheme kokoro-sdk-smoke \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /tmp/kokoro-tts-smoke-dd \
  build

DYLD_FRAMEWORK_PATH=/tmp/kokoro-tts-smoke-dd/Build/Products/Debug/PackageFrameworks \
  /tmp/kokoro-tts-smoke-dd/Build/Products/Debug/kokoro-sdk-smoke \
  /tmp/kokoro-sdk-starter-compiled \
  'Hello world.'
```

Result on 2026-06-28: passed with
`samples=37800 sampleRate=24000 duration=1.575`.

### Known Limitation

Direct `swift run --package-path swift-tts kokoro-sdk-smoke ...` currently fails
before synthesis because MLX Swift's `default.metallib` resource bundle is not
copied into `.build`:

```text
MLX error: Failed to load the default metallib.
```

The Xcode-built app path embeds `mlx-swift_Cmlx.bundle/default.metallib` and
runs successfully. Treat Xcode app builds as the reliable validation path for
Phase 5; Phase 6 owns fresh app fixture and physical-device proof.

---

## Issue: Phase 4 Reproducible HF Downloads and SDK Bundle Builder - Resolved

**First spotted:** 2026-06-28
**Resolved:** 2026-06-28
**Status:** Resolved

### Summary

Phase 4 added pinned Hugging Face provenance and deterministic SDK bundle
assembly. The current HF source of truth is:

- Repo: `mattmireles/kokoro-coreml`
- Revision: `c02933e179932e51909ff3b29466a7debac7d0e6`
- Last modified: `2026-06-10T05:43:09.000Z`
- Public/gated: public, not gated
- API inventory at that revision: 23 `.mlpackage` directories and 54 voice
  `.bin` files

HF does not currently publish `KokoroRuntimeManifest.json` or
`HostedManifest.json`; those are SDK-generated artifacts in this repo.

### Files Added

- `scripts/inspect_hf_artifacts.py`
- `scripts/hash_mlpackage_tree.py`
- `scripts/build_sdk_bundle.mjs`
- `scripts/validate_sdk_bundle.mjs`
- `schemas/KokoroRuntimeManifest.schema.json`

`scripts/download_models.py` now supports `--repo-id`, `--revision`,
`--manifest-out`, and `--sdk-profile starter|custom|full`. Starter downloads
all padded duration buckets, 15-second F0Ntrain/decoder-pre/HAR-post packages,
and `af_heart.bin`.

### Bundle Policy

`models.gist.is/coreml/v1` remains Gist app infrastructure. The public SDK does
not hard-code it. Instead, `scripts/build_sdk_bundle.mjs` emits the same kind of
hosted-manifest shape:

```json
{ "version": "...", "files": [{ "path": "...", "bytes": 0, "sha256": "..." }] }
```

A developer or release process can host that manifest anywhere.

### Verification

HF inspection:

```bash
python3 scripts/inspect_hf_artifacts.py \
  --repo-id mattmireles/kokoro-coreml \
  --revision c02933e179932e51909ff3b29466a7debac7d0e6 \
  --output /tmp/kokoro-hf-inspect.json
```

Result on 2026-06-28: resolved the pinned revision and reported 23 model
packages, 54 voices, and missing upstream SDK metadata
`HostedManifest.json`, `KokoroRuntimeManifest.json`.

Pinned starter download:

```bash
python3 scripts/download_models.py \
  --repo-id mattmireles/kokoro-coreml \
  --revision c02933e179932e51909ff3b29466a7debac7d0e6 \
  --sdk-profile starter \
  --manifest-out /tmp/kokoro-download-manifest.json
```

Result on 2026-06-28: passed, fetched/verified 31 files, and wrote a 31-file
download manifest. The first attempt exposed a symlink-order bug in manifest
writing around `checkpoints/config.json`; Phase 4 fixed it by skipping symlinks
before calling `is_file()`.

Package tree hash:

```bash
python3 scripts/hash_mlpackage_tree.py \
  coreml/kokoro_duration_t512.mlpackage \
  coreml/kokoro_decoder_pre_15s.mlpackage \
  --output /tmp/kokoro-package-hashes.json
```

Result on 2026-06-28: passed and produced stable per-package tree digests.

Bundle generation:

```bash
node scripts/build_sdk_bundle.mjs \
  --profile starter \
  --output /tmp/kokoro-sdk-starter \
  --repo-id mattmireles/kokoro-coreml \
  --revision c02933e179932e51909ff3b29466a7debac7d0e6
```

Result on 2026-06-28: initially built a starter bundle with 4 model packages,
1 voice, `KokoroRuntimeManifest.json`, and `HostedManifest.json`. Phase 5 later
changed starter/custom duration defaults to all padded duration buckets because
t512-only starter bundles caused multi-minute Core ML specialization for short
text. Current starter bundles contain 10 model packages, 1 voice,
`KokoroRuntimeManifest.json`, and `HostedManifest.json`.

Bundle validation:

```bash
node scripts/validate_sdk_bundle.mjs /tmp/kokoro-sdk-starter
```

Result on 2026-06-28: passed. A temporary newline appended to
`runtime/kokoro-vocab.json` made validation fail as expected; restoring the file
made validation pass again.

Custom profile smoke:

```bash
node scripts/build_sdk_bundle.mjs \
  --profile custom \
  --voices af_heart,af_bella \
  --buckets 3 \
  --output /tmp/kokoro-sdk-custom \
  --repo-id mattmireles/kokoro-coreml \
  --revision c02933e179932e51909ff3b29466a7debac7d0e6
```

Result on 2026-06-28: built a custom bundle with 4 model packages and 2 voices.

### If This Recurs

- [ ] Always pass `--revision` for release bundles.
- [ ] Treat missing upstream SDK manifests as expected until HF is updated.
- [ ] Validate generated bundles before handing them to app integration work.
- [ ] Keep Gist hosting outside SDK code; the SDK consumes a manifest shape, not
      a Gist-specific URL.

---

## Issue: Phase 3 Native Swift Text Prep - Resolved

**First spotted:** 2026-06-28
**Resolved:** 2026-06-28
**Status:** Resolved

### Summary

Phase 3 added the native prep contract needed before the SDK can expose
`synthesize(text:)`. The low-level `swift/` package now has
`KokoroPreparedInput`, which is only the prepared tensor contract and a bridge
back to `KokoroSynthesisRequest`. The higher-floor `swift-tts/` package owns
raw-text concerns: voice IDs, synthesis options, audio value type, checked vocab
tokenization, voice-row loading, and Botnet-compatible chunking.

The implementation follows Gist's app pattern: MisakiSwift remains behind
`KokoroPhonemizer`, tokenization drops unknown phoneme characters, token IDs
are framed with boundary token `0`, the unpadded token count is preserved as
metadata, and voice row selection uses phoneme UTF-16 count:
`rowIndex = clamp(phonemeCount - 1, 0, rowCount - 1)`.

### Public Prep Surface

Files added:

- `swift/Sources/KokoroPipeline/KokoroPreparedInput.swift`
- `swift-tts/Sources/KokoroTTS/KokoroVoiceID.swift`
- `swift-tts/Sources/KokoroTTS/KokoroSynthesisOptions.swift`
- `swift-tts/Sources/KokoroTTS/KokoroAudio.swift`
- `swift-tts/Sources/KokoroTTS/KokoroTextProcessor.swift`
- `swift-tts/Sources/KokoroTTS/VoiceTable.swift`
- `swift-tts/Sources/KokoroTTS/TextChunker.swift`

`TextChunker` is a direct port of Botnet's fleet chunker with one deliberate
SDK change: `maxChunkSeconds` is configurable and defaults to 15 seconds. The
Botnet 30-second value is still exposed as `TextChunker.botnetMaxChunkSeconds`
for parity tests and caller overrides.

`KokoroTextProcessor` is deterministic except for the injected phonemizer. Tests
use a stub phonemizer, so tokenization and validation are covered without
requiring MLX runtime resources. Misaki runtime proof remains xcodebuild/probe
based, as recorded in Phase 1.

### Verification

Regression test:

```bash
swift test --package-path swift-tts
```

Result on 2026-06-28: passed 20 tests with 2 Misaki runtime tests skipped by
default. New coverage includes checked vocab tokenization, unknown-token drop,
BOS/EOS framing, enum padding, metadata preservation, typed validation,
voice-row selection from real `.bin` files, and Botnet chunker fixtures.

Regression test:

```bash
swift test --package-path swift
```

Result on 2026-06-28: passed 46 tests. The added low-level prepared-input test
confirms `KokoroPreparedInput.synthesisRequest()` preserves input IDs,
attention mask, `refS`, and speed.

Regression test:

```bash
python3 scripts/verify_runtime_assets.py
```

Result on 2026-06-28: passed.

App-style compile checks:

```bash
xcodebuild -quiet -scheme KokoroTTS \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /tmp/kokoro-tts-iossim-dd build

xcodebuild -quiet -scheme kokoro-misaki-probe \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /tmp/kokoro-tts-dd build
```

Result on 2026-06-28: both returned success. The machine still reports
CoreSimulator `1051.54.0 < 1051.55.0`; this remains a local simulator-service
warning, not a compile failure.

Drift report:

```bash
node scripts/compare_misaki_botnet_phonemes.mjs \
  --probe-bin /tmp/kokoro-tts-dd/Build/Products/Debug/kokoro-misaki-probe \
  --dyld-framework-path /tmp/kokoro-tts-dd/Build/Products/Debug/PackageFrameworks
```

Result on 2026-06-28: completed and reproduced the approved Misaki-vs-Botnet
drift table. No fixture is exact; this remains an accepted V1 behavior choice
unless later audio judgment says otherwise.

Dependency scan:

```bash
rg -n "node|python|Botnet|KokoroWorkerCore|child_process|Process\(" \
  swift-tts/Sources swift/Sources/KokoroPipeline
```

Result on 2026-06-28: found only documentation comments mentioning Botnet; no
Node, Python, Botnet module, or process-spawning runtime imports were added.

### If This Recurs

- [ ] Keep `KokoroPhonemizer` injectable; do not make tests require MLX just to
      validate tokenization.
- [ ] Preserve UTF-16 phoneme length for voice-row selection.
- [ ] Do not change Botnet chunker edge behavior inside Phase 3; pin observed
      behavior first and only change it behind an explicit SDK policy later.
- [ ] Keep raw-text SDK prep in `swift-tts/`; do not raise `KokoroPipeline`'s
      platform floor.

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
