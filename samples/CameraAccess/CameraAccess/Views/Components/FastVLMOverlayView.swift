import SwiftUI

struct FastVLMStatusBar: View {
    let service: FastVLMService

    var body: some View {
        HStack {
            HStack(spacing: 6) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusText)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white)

                if service.sceneChangeCount > 0 {
                    Text("Scene #\(service.sceneChangeCount)")
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundColor(.white.opacity(0.7))
                }

                if !service.ttft.isEmpty {
                    Text("TTFT \(service.ttft)")
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundColor(.white.opacity(0.7))
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.black.opacity(0.6))
            .cornerRadius(16)
        }
    }

    private var statusColor: Color {
        if service.isSceneStable {
            return .blue
        }
        switch service.evaluationState {
        case .idle: return .gray
        case .loading: return .yellow
        case .processingPrompt: return .yellow
        case .generatingResponse: return .green
        }
    }

    private var statusText: String {
        if service.isSceneStable && service.evaluationState == .idle {
            return "Stable"
        }
        switch service.evaluationState {
        case .idle: return "FastVLM"
        case .loading: return service.modelInfo
        case .processingPrompt: return "Analyzing..."
        case .generatingResponse: return "Generating"
        }
    }
}

struct FastVLMOutputView: View {
    let text: String

    var body: some View {
        if !text.isEmpty {
            ScrollView {
                Text(text)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 120)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Color.black.opacity(0.6))
            .cornerRadius(12)
        }
    }
}

struct FastVLMOverlayView: View {
    let service: FastVLMService

    var body: some View {
        VStack {
            FastVLMStatusBar(service: service)
            Spacer()

            VStack(spacing: 8) {
                FastVLMOutputView(text: service.sceneLabel)

                if service.evaluationState == .processingPrompt {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.7)
                            .tint(.white)
                        Text("Analyzing frame...")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(.white)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(16)
                }
            }
            .padding(.bottom, 80)
        }
        .padding(.all, 24)
    }
}
