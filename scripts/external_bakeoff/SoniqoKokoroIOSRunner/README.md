# Soniqo Kokoro iOS Runner

Minimal Kokoro-only iOS runner for the connected iPhone 12 Pro. This avoids
Soniqo's full `iOSEchoDemo`, which pulls in ASR, VAD, MLX, and speech-core
dependencies that are not part of the Core ML TTS timing boundary.

## Generate

Set `SPEECH_SWIFT_PATH` to a pinned `soniqo/speech-swift` clone and generate the
project:

```bash
cd scripts/external_bakeoff/SoniqoKokoroIOSRunner
SPEECH_SWIFT_PATH=/tmp/kokoro-external-bakeoff/speech-swift xcodegen generate
```

## Build For The Connected iPhone

Preflight requirements:

- The iPhone must appear in `xcrun devicectl list devices`.
- `DEVELOPMENT_TEAM` must be set to an Apple development team ID.
- `security find-identity -v -p codesigning` must show a valid Apple
  development signing identity.

```bash
xcodebuild -project SoniqoKokoroIOSRunner.xcodeproj \
  -scheme SoniqoKokoroIOSRunner \
  -destination 'id=00008101-001134561A0A001E' \
  -derivedDataPath /tmp/kokoro-external-bakeoff/ios-runner-derived \
  -allowProvisioningUpdates build
```

The app loads `KokoroTTSModel.fromPretrained(computeUnits: .all)`, synthesizes a
single `af_heart` input, and renders cold/warm wall time plus sample count on
screen. Phase 2 should replace the fixed text with a manifest-driven loop.
