# Soniqo Kokoro iOS Runner

Minimal Kokoro-only iOS runner for the connected iPhone 12 Pro. This avoids
Soniqo's full `iOSEchoDemo`, which pulls in ASR, VAD, MLX, and speech-core
dependencies that are not part of the Core ML TTS timing boundary.

## Generate

Set `SPEECH_SWIFT_PATH` to a pinned `soniqo/speech-swift` clone and generate the
runtime manifest source plus the project:

```bash
python scripts/external_bakeoff/generate_ios_runner_manifest.py
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

The app loads `KokoroTTSModel.fromPretrained(computeUnits: .all)`, synthesizes
the five runtime bucket inputs from
`outputs/external_bakeoff/runtime_input_manifest.json`, and renders JSON with
one cold call plus five warm calls per bucket. It remains Kokoro TTS only:
Whisper, ASR, VAD, playback, and the full Soniqo echo demo are outside the
measurement path.
