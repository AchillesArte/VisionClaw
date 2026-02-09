import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import UIKit

/// On-device vision-language model service using FastVLM (0.5B).
/// Accepts UIImage frames and produces text descriptions.
@Observable
@MainActor
class FastVLMService {

    // MARK: - Public state

    var isActive = false
    var isRunning = false
    var output = ""
    var ttft = ""
    var modelInfo = ""

    enum EvaluationState: String {
        case idle = "Idle"
        case loading = "Loading Model"
        case processingPrompt = "Processing"
        case generatingResponse = "Generating"
    }

    var evaluationState = EvaluationState.idle

    // MARK: - Configuration

    var prompt = "Describe what you see briefly, about 15 words or less."

    // MARK: - Private

    private enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    private let generateParameters = GenerateParameters(temperature: 0.0)
    private let maxTokens = 240
    private let displayEveryNTokens = 4

    private var loadState = LoadState.idle
    private var currentTask: Task<Void, Never>?

    private var lastFrameTime = Date.distantPast
    private let frameInterval: TimeInterval = 1.0

    // MARK: - Init

    init() {
        FastVLM.register(modelFactory: VLMModelFactory.shared)
    }

    // MARK: - Model loading

    private func _load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            evaluationState = .loading

            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            // Look for the model files in the app bundle's "models" folder
            let modelConfig = FastVLM.modelConfiguration

            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfig
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.modelInfo = "Loading model: \(Int(progress.fractionCompleted * 100))%"
                }
            }

            modelInfo = "Model loaded"
            evaluationState = .idle
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let container):
            return container
        }
    }

    func load() async {
        do {
            _ = try await _load()
        } catch {
            modelInfo = "Error loading model: \(error)"
            evaluationState = .idle
            print("[FastVLM] Load error: \(error)")
        }
    }

    // MARK: - Inference

    /// Analyze a single frame. Returns immediately if already running or model not ready.
    func analyze(_ image: UIImage) async {
        guard let ciImage = CIImage(image: image) else {
            print("[FastVLM] Failed to create CIImage from UIImage")
            return
        }

        if isRunning {
            return
        }

        isRunning = true
        currentTask?.cancel()

        let task = Task {
            do {
                let modelContainer = try await _load()

                if Task.isCancelled { return }

                let userInput = UserInput(
                    prompt: .text(prompt),
                    images: [.ciImage(ciImage)]
                )

                let result = try await modelContainer.perform { context in
                    Task { @MainActor in
                        self.evaluationState = .processingPrompt
                    }

                    let llmStart = Date()
                    let input = try await context.processor.prepare(input: userInput)

                    var seenFirstToken = false

                    let result = try MLXLMCommon.generate(
                        input: input, parameters: self.generateParameters, context: context
                    ) { tokens in
                        if Task.isCancelled { return .stop }

                        if !seenFirstToken {
                            seenFirstToken = true
                            let duration = Date().timeIntervalSince(llmStart)
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                self.evaluationState = .generatingResponse
                                self.output = text
                                self.ttft = "\(Int(duration * 1000))ms"
                            }
                        }

                        if tokens.count % self.displayEveryNTokens == 0 {
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                self.output = text
                            }
                        }

                        if tokens.count >= self.maxTokens {
                            return .stop
                        }
                        return .more
                    }

                    return result
                }

                if !Task.isCancelled {
                    self.output = result.output
                }
            } catch {
                if !Task.isCancelled {
                    output = "Failed: \(error)"
                    print("[FastVLM] Inference error: \(error)")
                }
            }

            if evaluationState == .generatingResponse {
                evaluationState = .idle
            }
            isRunning = false
        }

        currentTask = task
    }

    /// Called from the video frame pipeline. Throttles to ~1fps and auto-analyzes.
    func analyzeIfReady(image: UIImage) {
        guard isActive else { return }

        let now = Date()
        guard now.timeIntervalSince(lastFrameTime) >= frameInterval else { return }
        lastFrameTime = now

        Task {
            await analyze(image)
        }
    }

    // MARK: - Lifecycle

    func start() async {
        isActive = true
        output = ""
        ttft = ""
        await load()
    }

    func stop() {
        isActive = false
        cancel()
    }

    func cancel() {
        currentTask?.cancel()
        currentTask = nil
        isRunning = false
        output = ""
        ttft = ""
        evaluationState = .idle
    }
}
