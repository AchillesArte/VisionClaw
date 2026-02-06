import SwiftUI

struct GeminiStatusBar: View {
  @ObservedObject var geminiVM: GeminiSessionViewModel

  var body: some View {
    HStack {
      // Connection status pill
      HStack(spacing: 6) {
        Circle()
          .fill(statusColor)
          .frame(width: 8, height: 8)
        Text(statusText)
          .font(.system(size: 12, weight: .medium))
          .foregroundColor(.white)
      }
      .padding(.horizontal, 12)
      .padding(.vertical, 6)
      .background(Color.black.opacity(0.6))
      .cornerRadius(16)

    }
  }

  private var statusColor: Color {
    switch geminiVM.connectionState {
    case .ready: return .green
    case .connecting, .settingUp: return .yellow
    case .error: return .red
    case .disconnected: return .gray
    }
  }

  private var statusText: String {
    switch geminiVM.connectionState {
    case .ready: return "AI Active"
    case .connecting, .settingUp: return "Connecting..."
    case .error: return "Error"
    case .disconnected: return "Disconnected"
    }
  }
}

struct SpeakingIndicator: View {
  @State private var animating = false

  var body: some View {
    HStack(spacing: 3) {
      ForEach(0..<4, id: \.self) { index in
        RoundedRectangle(cornerRadius: 1.5)
          .fill(Color.white)
          .frame(width: 3, height: animating ? CGFloat.random(in: 8...20) : 6)
          .animation(
            .easeInOut(duration: 0.3)
              .repeatForever(autoreverses: true)
              .delay(Double(index) * 0.1),
            value: animating
          )
      }
    }
    .onAppear { animating = true }
    .onDisappear { animating = false }
  }
}

struct ApiKeyInputView: View {
  @Binding var isPresented: Bool
  let onSave: (String) -> Void
  @State private var apiKey: String = GeminiConfig.apiKey

  var body: some View {
    VStack(spacing: 16) {
      Text("Gemini API Key")
        .font(.system(size: 18, weight: .semibold))
        .foregroundColor(.primary)

      Text("Enter your Google AI Studio API key to enable the AI assistant.")
        .font(.system(size: 14))
        .foregroundColor(.secondary)
        .multilineTextAlignment(.center)

      TextField("API Key", text: $apiKey)
        .textFieldStyle(.roundedBorder)
        .autocorrectionDisabled()
        .textInputAutocapitalization(.never)

      HStack(spacing: 12) {
        Button("Cancel") {
          isPresented = false
        }
        .foregroundColor(.secondary)

        Button("Save") {
          onSave(apiKey)
        }
        .fontWeight(.semibold)
        .disabled(apiKey.isEmpty)
      }
    }
    .padding(24)
  }
}
