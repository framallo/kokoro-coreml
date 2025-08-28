import SwiftUI

struct ContentView: View {
    @StateObject var viewModel: InferenceViewModel

    var body: some View {
        VStack(spacing: 16) {
            Text(viewModel.statusText)
                .font(.headline)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            if let ms = viewModel.lastInferenceMs {
                Text(String(format: "Inference time: %.1f ms", ms))
                    .font(.title3)
            } else {
                Text("Inference time: —")
                    .font(.title3)
            }

            Button(action: { viewModel.runTest() }) {
                Text("Run Inference Test")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.isRunning)
        }
        .padding()
    }
}

#Preview {
    ContentView(viewModel: InferenceViewModel())
}
