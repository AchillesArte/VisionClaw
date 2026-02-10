import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import UIKit

/// On-device vision-language model service using FastVLM (0.5B).
/// Uses a self-aware prompt to detect scene changes and maintain stable descriptions.
@Observable
@MainActor
class FastVLMService {

    // MARK: - Public state

    var isActive = false
    var isRunning = false
    var output = ""
    var ttft = ""
    var modelInfo = ""

    // Scene state
    var sceneLabel: String = ""
    var isSceneStable: Bool = false
    var sceneChangeCount: Int = 0

    enum EvaluationState: String {
        case idle = "Idle"
        case loading = "Loading Model"
        case processingPrompt = "Processing"
        case generatingResponse = "Generating"
    }

    var evaluationState = EvaluationState.idle

    // MARK: - Private

    private enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    private let generateParameters = GenerateParameters(temperature: 0.0)
    private let maxTokens = 80

    private var loadState = LoadState.idle
    private var currentTask: Task<Void, Never>?

    private var lastAnalyzeStart = Date.distantPast
    private var frameCount = 0

    // Scene tracking
    private var previousOutput: String = ""
    private var stableFrameCount: Int = 0
    private let stableConfirmThreshold = 2

    // Stable-mode throttling
    private var lastStableCheckTime = Date.distantPast
    private let stableCheckInterval: TimeInterval = 2.0

    // MARK: - Prompt

    private var currentPrompt: String {
        if previousOutput.isEmpty {
            return "Describe what you see briefly, about 15 words or less."
        } else {
            return """
            Your previous observation: "\(previousOutput)"
            Look at the current image. If the main subject is the same as before, respond with exactly: SAME
            If the scene has meaningfully changed, describe the new scene briefly in 15 words or less.
            """
        }
    }

    // MARK: - Init

    init() {
        FastVLM.register(modelFactory: VLMModelFactory.shared)
    }

    // MARK: - Model loading

    private func _load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            evaluationState = .loading

            MLX.GPU.set(cacheLimit: 256 * 1024 * 1024)

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

    /// Analyze a single frame with scene-aware prompting.
    func analyze(_ image: UIImage) async {
        guard let ciImage = CIImage(image: image) else {
            print("[FastVLM] Failed to create CIImage from UIImage")
            return
        }

        if isRunning {
            return
        }

        let gap = Date().timeIntervalSince(lastAnalyzeStart)
        lastAnalyzeStart = Date()
        frameCount += 1
        let thisFrame = frameCount
        print("[FastVLM] Frame #\(thisFrame) - gap since last: \(Int(gap * 1000))ms")

        isRunning = true
        currentTask?.cancel()

        let frameStart = Date()
        let promptText = currentPrompt

        let task = Task {
            do {
                let modelContainer = try await _load()

                if Task.isCancelled { return }

                let prepareStart = Date()
                let userInput = UserInput(
                    prompt: .text(promptText),
                    images: [.ciImage(ciImage)]
                )

                let result = try await modelContainer.perform { context in
                    Task { @MainActor in
                        self.evaluationState = .processingPrompt
                    }

                    let input = try await context.processor.prepare(input: userInput)
                    let prepareTime = Date().timeIntervalSince(prepareStart)
                    print("[FastVLM] Frame #\(thisFrame) - image prep: \(Int(prepareTime * 1000))ms")

                    let generateStart = Date()
                    var seenFirstToken = false

                    let result = try MLXLMCommon.generate(
                        input: input, parameters: self.generateParameters, context: context
                    ) { tokens in
                        if Task.isCancelled { return .stop }

                        if !seenFirstToken {
                            seenFirstToken = true
                            let ttftDuration = Date().timeIntervalSince(generateStart)
                            Task { @MainActor in
                                self.evaluationState = .generatingResponse
                                self.ttft = "\(Int(ttftDuration * 1000))ms"
                            }
                            print("[FastVLM] Frame #\(thisFrame) - TTFT: \(Int(ttftDuration * 1000))ms")
                        }

                        if tokens.count >= self.maxTokens {
                            return .stop
                        }
                        return .more
                    }

                    let generateTime = Date().timeIntervalSince(generateStart)
                    print("[FastVLM] Frame #\(thisFrame) - generate: \(Int(generateTime * 1000))ms (\(result.output.count) chars)")

                    return result
                }

                if !Task.isCancelled {
                    let response = result.output.trimmingCharacters(in: .whitespacesAndNewlines)
                    let totalTime = Date().timeIntervalSince(frameStart)

                    // Scene change detection
                    if response.uppercased().hasPrefix("SAME") {
                        stableFrameCount += 1
                        if stableFrameCount >= stableConfirmThreshold {
                            isSceneStable = true
                        }
                        output = sceneLabel
                        print("[FastVLM] Frame #\(thisFrame) - SAME (stable: \(stableFrameCount)) - TOTAL: \(Int(totalTime * 1000))ms")
                    } else {
                        stableFrameCount = 0
                        isSceneStable = false
                        sceneLabel = response
                        previousOutput = response
                        sceneChangeCount += 1
                        output = response
                        print("[FastVLM] Frame #\(thisFrame) - SCENE CHANGE #\(sceneChangeCount): \(response) - TOTAL: \(Int(totalTime * 1000))ms")
                    }
                }
            } catch {
                if !Task.isCancelled {
                    output = "Failed: \(error)"
                    print("[FastVLM] Frame #\(thisFrame) inference error: \(error)")
                }
            }

            if evaluationState == .generatingResponse {
                evaluationState = .idle
            }
            isRunning = false
        }

        currentTask = task
    }

    /// Called from the video frame pipeline. Throttles when scene is stable.
    func analyzeIfReady(image: UIImage) {
        guard isActive, !isRunning else { return }

        // When scene is stable, only re-check every stableCheckInterval
        if isSceneStable {
            let now = Date()
            guard now.timeIntervalSince(lastStableCheckTime) >= stableCheckInterval else { return }
            lastStableCheckTime = now
        }

        Task {
            await analyze(image)
        }
    }

    // MARK: - Lifecycle

    func start() async {
        isActive = true
        output = ""
        ttft = ""
        sceneLabel = ""
        previousOutput = ""
        isSceneStable = false
        stableFrameCount = 0
        sceneChangeCount = 0
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
