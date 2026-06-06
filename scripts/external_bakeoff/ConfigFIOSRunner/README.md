# Config F iOS Runner

Minimal physical-device runner for the repo's Config F Swift/Core ML path. It
uses the shared `KokoroPipeline` package, bundles the five runtime buckets
(`3s`, `7s`, `10s`, `15s`, `30s`), forces exact-duration model discovery, and
records one post-preflight cold call plus warmed inference calls per bucket.

The app intentionally reports timing JSON only. It does not include Whisper,
ASR, playback, Soniqo, or the echo demo.

## Generate

Refresh the small ignored input JSON resources first:

```bash
uv run --no-sync python scripts/prepare_swift_bench_inputs.py
```

Then generate the Xcode project:

```bash
cd scripts/external_bakeoff/ConfigFIOSRunner
xcodegen generate
```

## Build

```bash
xcodebuild -project ConfigFIOSRunner.xcodeproj \
  -scheme ConfigFIOSRunner \
  -destination 'platform=iOS,id=00008101-001134561A0A001E' \
  -derivedDataPath /tmp/kokoro-external-bakeoff/config-f-ios-derived \
  build
```

The first 30s generator compile can take a long time on older iPhones. Compare
only the warmed timings in the emitted JSON.
