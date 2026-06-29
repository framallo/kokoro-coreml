import AVFoundation
import SwiftUI
import KokoroTTS

@main
struct KokoroDemoApp: App {
    var body: some Scene {
        WindowGroup {
            DemoView(model: DemoModel())
        }
    }
}

struct DemoView: View {
    @StateObject var model: DemoModel

    var body: some View {
        NavigationStack {
            Form {
                Section("Manifest") {
                    TextField("URL", text: $model.manifestURLString, axis: .vertical)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Section("Text") {
                    TextEditor(text: $model.text)
                        .frame(minHeight: 120)
                    Picker("Voice", selection: $model.voice) {
                        Text("af_heart").tag("af_heart")
                    }
                    Button(model.isRunning ? "Synthesizing..." : "Synthesize") {
                        Task { await model.synthesize() }
                    }
                    .disabled(model.isRunning)
                }
                Section("Status") {
                    Text(model.status)
                        .font(.footnote.monospaced())
                }
            }
            .navigationTitle("Kokoro")
        }
        .task {
            await model.runLaunchAutomationIfRequested()
        }
    }
}

@MainActor
final class DemoModel: ObservableObject {
    @Published var manifestURLString = ProcessInfo.processInfo.environment["KOKORO_MANIFEST_URL"] ?? ""
    @Published var text = "Hello world."
    @Published var voice = "af_heart"
    @Published var status = "Idle"
    @Published var isRunning = false

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var didAutoRun = false

    init() {
        let args = CommandLine.arguments
        if let manifest = Self.value(after: "--manifest-url", in: args) {
            manifestURLString = manifest
        }
        if let text = Self.value(after: "--text", in: args) {
            self.text = text
        }
    }

    func runLaunchAutomationIfRequested() async {
        guard !didAutoRun, CommandLine.arguments.contains("--auto-run") else {
            return
        }
        didAutoRun = true
        await synthesize()
    }

    func synthesize() async {
        guard !isRunning else {
            return
        }
        guard let manifestURL = URL(string: manifestURLString), !manifestURLString.isEmpty else {
            status = "Missing manifest URL"
            print("KOKORO_DEMO_ERROR missing-manifest-url")
            return
        }
        isRunning = true
        status = "Downloading resources..."
        do {
            let cache = try Self.cacheDirectory()
            let resources = try await KokoroDownloadedModelStore(
                manifestURL: manifestURL,
                cacheDirectory: cache
            ).hydrate()
            status = "Loading SDK..."
            let tts = try await KokoroTTS.load(resources: resources)
            status = "Synthesizing..."
            let audio = try await tts.synthesize(text, voice: KokoroVoiceID(voice))
            try play(audio)
            status = "Done: \(audio.samples.count) samples, \(audio.durationSeconds)s"
            print("KOKORO_DEMO_DONE samples=\(audio.samples.count) sampleRate=\(audio.sampleRate) duration=\(audio.durationSeconds)")
        } catch {
            status = "Error: \(error.localizedDescription)"
            print("KOKORO_DEMO_ERROR \(error)")
        }
        isRunning = false
    }

    private func play(_ audio: KokoroAudio) throws {
        let buffer = try audio.makePCMBuffer()
        if !engine.attachedNodes.contains(player) {
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: buffer.format)
        }
        if !engine.isRunning {
            try engine.start()
        }
        player.stop()
        player.scheduleBuffer(buffer, at: nil, options: [])
        player.play()
    }

    private static func cacheDirectory() throws -> URL {
        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return documents.appendingPathComponent("KokoroModelCache", isDirectory: true)
    }

    private static func value(after flag: String, in args: [String]) -> String? {
        guard let index = args.firstIndex(of: flag), args.indices.contains(index + 1) else {
            return nil
        }
        return args[index + 1]
    }
}
