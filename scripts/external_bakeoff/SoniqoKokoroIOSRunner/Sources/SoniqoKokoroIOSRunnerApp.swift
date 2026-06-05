import SwiftUI
import KokoroTTS
import CoreML

@main
struct SoniqoKokoroIOSRunnerApp: App {
    var body: some Scene {
        WindowGroup {
            BenchmarkView()
        }
    }
}

@MainActor
final class BenchmarkViewModel: ObservableObject {
    @Published var status = "Ready"
    @Published var result = ""

    private let text = "At the edge of the harbor, the engineer watched the status display while the model warmed, measured, and returned clean audio before the next request arrived."

    func run() {
        status = "Running"
        result = ""
        let benchmarkText = text
        Task {
            do {
                let metrics = try await Task.detached { () -> [String: Any] in
                    let model = try await KokoroTTSModel.fromPretrained(computeUnits: .all)
                    let coldStart = CFAbsoluteTimeGetCurrent()
                    _ = try model.synthesize(text: benchmarkText, voice: "af_heart", language: "en", speed: 1.0)
                    let cold = CFAbsoluteTimeGetCurrent() - coldStart

                    let warmStart = CFAbsoluteTimeGetCurrent()
                    let audio = try model.synthesize(text: benchmarkText, voice: "af_heart", language: "en", speed: 1.0)
                    let warm = CFAbsoluteTimeGetCurrent() - warmStart

                    return [
                        "cold_wall_time_s": cold,
                        "warm_wall_time_s": warm,
                        "sample_count": audio.count,
                        "sample_rate": KokoroTTSModel.outputSampleRate,
                        "duration_s": Double(audio.count) / Double(KokoroTTSModel.outputSampleRate),
                        "voice": "af_heart",
                    ]
                }.value
                let data = try JSONSerialization.data(withJSONObject: metrics, options: [.prettyPrinted, .sortedKeys])
                result = String(data: data, encoding: .utf8) ?? "\(metrics)"
                status = "Done"
            } catch {
                status = "Failed"
                result = String(describing: error)
            }
        }
    }
}

struct BenchmarkView: View {
    @StateObject private var model = BenchmarkViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Soniqo Kokoro")
                .font(.title2)
                .bold()
            Text(model.status)
                .font(.headline)
            Button("Run") {
                model.run()
            }
            .buttonStyle(.borderedProminent)
            ScrollView {
                Text(model.result)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
    }
}
