import AVFoundation
import Darwin
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

    private let scenario: String
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
        scenario = Self.value(after: "--scenario", in: args) ?? "smoke"
    }

    func runLaunchAutomationIfRequested() async {
        guard !didAutoRun, CommandLine.arguments.contains("--auto-run") else {
            return
        }
        didAutoRun = true
        await runScenario(named: scenario)
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

    private func runScenario(named scenario: String) async {
        guard !isRunning else {
            return
        }
        guard let manifestURL = URL(string: manifestURLString), !manifestURLString.isEmpty else {
            status = "Missing manifest URL"
            print("KOKORO_DEMO_ERROR scenario=\(scenario) missing-manifest-url")
            return
        }
        isRunning = true
        do {
            status = "Hydrating resources..."
            let resources = try await KokoroDownloadedModelStore(
                manifestURL: manifestURL,
                cacheDirectory: try Self.cacheDirectory()
            ).hydrate()
            status = "Loading SDK..."
            let tts = try await KokoroTTS.load(resources: resources)
            switch scenario {
            case "smoke":
                let audio = try await runSynthesis(tts: tts, text: text, label: "smoke")
                try play(audio)
            case "warm":
                _ = try await runSynthesis(tts: tts, text: text, label: "warm-first")
                let audio = try await runSynthesis(tts: tts, text: text, label: "warm-second")
                try play(audio)
            case "long":
                let audio = try await runSynthesis(tts: tts, text: Self.longText, label: "long")
                try play(audio)
            case "cancel":
                try await runCancellation(tts: tts)
            case "all":
                _ = try await runSynthesis(tts: tts, text: text, label: "all-first")
                _ = try await runSynthesis(tts: tts, text: text, label: "all-warm")
                _ = try await runSynthesis(tts: tts, text: Self.longText, label: "all-long")
                try await runCancellation(tts: tts)
                print("KOKORO_DEMO_MEMORY physicalFootprintBytes=\(Self.physicalFootprintBytes())")
            default:
                print("KOKORO_DEMO_ERROR scenario=\(scenario) unsupported-scenario")
            }
            status = "Scenario \(scenario) complete"
            print("KOKORO_DEMO_SCENARIO_DONE scenario=\(scenario)")
        } catch {
            status = "Error: \(error.localizedDescription)"
            print("KOKORO_DEMO_ERROR scenario=\(scenario) \(error)")
        }
        isRunning = false
    }

    private func runSynthesis(tts: KokoroTTS, text: String, label: String) async throws -> KokoroAudio {
        let start = Date()
        let audio = try await tts.synthesize(text, voice: KokoroVoiceID(voice))
        let elapsed = Date().timeIntervalSince(start)
        print("KOKORO_DEMO_DONE label=\(label) samples=\(audio.samples.count) sampleRate=\(audio.sampleRate) duration=\(audio.durationSeconds) elapsedSeconds=\(elapsed)")
        return audio
    }

    private func runCancellation(tts: KokoroTTS) async throws {
        let task = Task {
            try await tts.synthesize(Self.longText, voice: KokoroVoiceID(voice))
        }
        task.cancel()
        do {
            _ = try await task.value
            print("KOKORO_DEMO_ERROR scenario=cancel cancellation-did-not-throw")
        } catch {
            print("KOKORO_DEMO_CANCELLED error=\(error)")
        }
    }

    private func play(_ audio: KokoroAudio) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio, options: [])
        try session.setActive(true)
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

    private static var longText: String {
        Array(repeating: "Local text to speech should keep working on device, even when the caller sends several sentences at once.", count: 12)
            .joined(separator: " ")
    }

    private static func physicalFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return 0
        }
        return info.phys_footprint
    }

    private static func value(after flag: String, in args: [String]) -> String? {
        guard let index = args.firstIndex(of: flag), args.indices.contains(index + 1) else {
            return nil
        }
        return args[index + 1]
    }
}
