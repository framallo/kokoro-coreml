import SwiftUI

@main
struct Kokoro_iOS_TesterApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: InferenceViewModel())
        }
    }
}
