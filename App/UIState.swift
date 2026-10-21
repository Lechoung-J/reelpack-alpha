import Foundation
#if canImport(Shared)
import Shared
#endif

/// Four capabilities, never an assumed execution order.
enum PresentationCapability: String, CaseIterable, Identifiable, Codable, Sendable {
    case transcription, shots, ocr, music
    var id: String { rawValue }
    var index: Int {
        switch self {
        case .transcription: return 1
        case .shots: return 2
        case .ocr: return 3
        case .music: return 4
        }
    }
    var title: String {
        switch self {
        case .transcription: return "聆听言语"
        case .shots: return "整理镜头"
        case .ocr: return "拾取文字"
        case .music: return "感知节奏"
        }
    }
    var shortName: String {
        switch self {
        case .transcription: return "转录"
        case .shots: return "镜头"
        case .ocr: return "文字识别"
        case .music: return "音频"
        }
    }
}

enum UIWorkflowMode: String, Codable, Sendable {
    case idle, preparing, processing, finalizing, complete, partial, error, cancelling, cancelled
}

/// Independent of SwiftUI and the canvas for a standalone swiftc harness.
enum UIExecutionStatus: String, Codable, Sendable {
    case pending, running, observed, complete
    case notRun = "not_run"
    case failed, unknown, partial, cancelled
    var label: String {
        switch self {
        case .pending: return "未开始"
        case .running: return "运行中"
        case .observed: return "待确认"
        case .complete: return "已完成"
        case .notRun: return "未运行"
        case .failed: return "失败"
        case .unknown: return "状态未知"
        case .partial: return "部分完成"
        case .cancelled: return "已取消"
        }
    }
}

struct UICapabilityState: Equatable, Codable, Sendable {
    var status: UIExecutionStatus = .pending
    var rawStatus: String?
    var outcome: String?
    var displayLabel: String {
        if status == .complete && outcome == "speech_not_detected" { return "未检出语音" }
        if status == .notRun && outcome == "no_audio_track" { return "无音轨" }
        return status.label
    }
    var reason: String?
    var impact: String?
    var remedy: String?
    var lastStage: String?
    var lastMessage: String?
}

struct UIWorkflowIssue: Equatable, Codable, Sendable {
    let stage: String
    let code: String
    let message: String
}

struct UIRawProgress: Equatable, Codable, Sendable {
    let stage: String
    let percent: Int
    let message: String
}

/// App-only adaptation. Neither percent nor a subsequent stage proves success.
struct UIWorkflowState {
    private(set) var mode: UIWorkflowMode = .idle
    private(set) var generation: UUID?
    private(set) var isActive = false
    private(set) var inputURL: URL?
    private(set) var outputDirectory: URL?
    private(set) var capabilities = Dictionary(uniqueKeysWithValues:
        PresentationCapability.allCases.map { ($0, UICapabilityState()) })
    private(set) var artifacts: [OutputArtifact] = []
    private(set) var completionOutput: [String: String]?
    private(set) var completionReceived = false
    private(set) var progressEvents: [UIRawProgress] = []
    private(set) var issues: [UIWorkflowIssue] = []
    private(set) var diagnostics: [String] = []
    private(set) var notice: String?
    private var currentCapability: PresentationCapability?
    private var summaryDiagnostics: [String] = []

    var isBusy: Bool { isActive }
    var hasResults: Bool { !artifacts.isEmpty }
    var hasSavedResults: Bool {
        guard let completionOutput else { return false }
        var outcome = PipelineOutcome()
        outcome.consume(.complete(output: completionOutput))
        return outcome.validSavedResult
    }
    var hasUsableCompletion: Bool { completionReceived && hasSavedResults }
    var allCapabilitiesComplete: Bool {
        PresentationCapability.allCases.allSatisfy { capabilities[$0]?.status == .complete }
    }
    /// Raw multiline errors stay in details; the canvas gets one line.
    var statusText: String {
        if let notice { return Self.singleLine(notice) }
        switch mode {
        case .idle: return "拖入本地视频，或选择视频"
        case .preparing: return "正在准备"
        case .processing: return "正在处理"
        case .finalizing: return "正在保存"
        case .complete: return "完成"
        case .partial:
            let missing = PresentationCapability.allCases.filter { capabilities[$0]?.status != .complete }
            if missing.count == 1, let capability = missing.first, let state = capabilities[capability] {
                return "已保存 · \(capability.shortName)\(state.displayLabel)"
            }
            return "已保存 · 部分结果待确认"
        case .error: return "处理未完成"
        case .cancelling: return "正在取消"
        case .cancelled: return "已取消"
        }
    }

    mutating func begin(input: URL, outputDirectory: URL, generation: UUID = UUID()) {
        self = UIWorkflowState()
        self.generation = generation
        self.inputURL = input
        self.outputDirectory = outputDirectory
        isActive = true
        mode = .preparing
    }
    mutating func showNotice(_ message: String) { notice = message }
    mutating func clearNotice() { notice = nil }
    /// A rejected launch also preserves any previous usable results.
    mutating func failToStart(_ message: String) {
        guard !isActive else { return }
        notice = nil
        issues.append(UIWorkflowIssue(stage: "startup", code: "LAUNCH_FAILED", message: message))
        mode = .error
    }
    mutating func requestCancellation(generation: UUID) {
        guard self.generation == generation, isActive else { return }
        notice = nil
        mode = .cancelling
    }

    /// Tests may supply already validated artifacts; production uses the baseline
    /// OutputArtifact mapping and filesystem checks unchanged.
    mutating func apply(event: IPCMessage, generation: UUID, validatedArtifacts: [OutputArtifact]? = nil) {
        guard self.generation == generation, isActive else { return }
        notice = nil
        switch event {
        case .progress(let stage, let percent, let message):
            progressEvents.append(UIRawProgress(stage: stage, percent: percent, message: message))
            guard mode != .cancelling, !completionReceived else { return }
            let next = Self.capability(stage: stage, message: message)
            leaveCurrentCapability(unless: next)
            if let next {
                currentCapability = next
                capabilities[next]?.status = .running
                capabilities[next]?.lastStage = stage
                capabilities[next]?.lastMessage = message
                mode = .processing
            } else {
                switch stage.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
                case "input", "runtime_cache", "model_preflight", "cache": mode = .preparing
                case "output": mode = .finalizing
                default: mode = .processing
                }
            }
        case .error(let stage, let code, let message):
            // NO_AUDIO and optional errors may be followed by valid saved results.
            issues.append(UIWorkflowIssue(stage: stage, code: code, message: message))
        case .complete(let output):
            completionReceived = true
            completionOutput = output
            artifacts = validatedArtifacts ?? OutputArtifact.fromPipelineOutput(output)
            leaveCurrentCapability(unless: nil)
            parseSummary(output["capability_summary"])
            if let execution = output["execution_status"], execution != "complete" {
                summaryDiagnostics.append("执行状态为 \(execution)，已有产物只代表部分结果。")
            }
            if !hasSavedResults {
                summaryDiagnostics.append("完成消息没有可用的文稿、资产包或 PDF；原视频本身不能证明结果已保存。")
            }
            for key in ["markdown", "package", "video", "pdf"] {
                if let path = output[key], !path.isEmpty,
                   !artifacts.contains(where: { $0.url.standardizedFileURL.path == URL(fileURLWithPath: path).standardizedFileURL.path }) {
                    summaryDiagnostics.append("完成消息中的 \(key) 路径不存在或类型不符：\(path)")
                }
            }
            // Still owned by ProcessRunner until its stream actually ends.
            if mode != .cancelling { mode = .finalizing }
        }
    }

    /// Only call after the runner stream ends, never on a timer or percent=100.
    mutating func finish(generation: UUID, error: String? = nil) {
        guard self.generation == generation, isActive else { return }
        let cancelling = mode == .cancelling
        isActive = false
        notice = nil
        if let error {
            issues.append(UIWorkflowIssue(stage: "process", code: "PROCESS_EXIT", message: error))
        }
        if cancelling {
            mode = .cancelled
            for key in PresentationCapability.allCases where capabilities[key]?.status == .running || capabilities[key]?.status == .observed {
                capabilities[key]?.status = .cancelled
            }
            return
        }
        if hasUsableCompletion {
            mode = allCapabilitiesComplete && issues.isEmpty && summaryDiagnostics.isEmpty && error == nil ? .complete : .partial
        } else {
            mode = .error
            diagnostics.append("流程已结束，但没有收到包含可用保存结果的完成消息。")
            for key in PresentationCapability.allCases {
                switch capabilities[key]?.status {
                case .pending, .running, .observed:
                    capabilities[key]?.status = .unknown
                    capabilities[key]?.reason = "流程结束前未收到该能力的最终摘要。"
                default: break
                }
            }
        }
    }
    mutating func reset() {
        guard !isActive else { return }
        self = UIWorkflowState()
    }

    var detailText: String {
        guard mode != .idle || notice != nil || !issues.isEmpty || !artifacts.isEmpty else { return "" }
        var lines = ["Reelpack · \(statusText)"]
        if let generation { lines.append("任务：\(generation.uuidString)") }
        if let inputURL { lines.append("输入：\(inputURL.path)") }
        if let outputDirectory { lines.append("本次输出目录：\(outputDirectory.path)") }
        if let notice { lines.append("提示：\(notice)") }
        for capability in PresentationCapability.allCases {
            let state = capabilities[capability] ?? UICapabilityState()
            lines.append("\n\(String(format: "%02d", capability.index)) \(capability.title) [\(capability.rawValue)]：\(state.displayLabel)")
            if let rawStatus = state.rawStatus { lines.append("原始状态：\(rawStatus)") }
            if let reason = state.reason { lines.append("原因：\(reason)") }
            if let impact = state.impact { lines.append("影响：\(impact)") }
            if let remedy = state.remedy { lines.append("处理建议：\(remedy)") }
        }
        if !issues.isEmpty {
            lines.append("\n原始错误与提示：")
            lines += issues.map { "[\($0.stage)] \($0.code)：\($0.message)" }
        }
        let notes = diagnostics + summaryDiagnostics
        if !notes.isEmpty { lines.append("\n结果检查："); lines += notes }
        if !progressEvents.isEmpty {
            lines.append("\n原始阶段事件（percent 是原始数据，不代表四项能力完成度）：")
            lines += progressEvents.map { "[\($0.stage)] percent=\($0.percent) \($0.message)" }
        }
        if !artifacts.isEmpty {
            lines.append("\n已有产物：")
            lines += artifacts.map { "\($0.kind.rawValue)：\($0.url.path)" }
        }
        if let completionOutput {
            lines.append("\n完成消息：")
            lines += completionOutput.keys.sorted().map { "\($0)：\(completionOutput[$0] ?? "")" }
        }
        return lines.joined(separator: "\n")
    }

    private mutating func leaveCurrentCapability(unless next: PresentationCapability?) {
        if let currentCapability, currentCapability != next,
           capabilities[currentCapability]?.status == .running {
            capabilities[currentCapability]?.status = .observed
        }
        if currentCapability != next { currentCapability = nil }
    }
    private mutating func parseSummary(_ encoded: String?) {
        summaryDiagnostics = []
        for capability in PresentationCapability.allCases {
            capabilities[capability]?.status = .unknown
            capabilities[capability]?.rawStatus = nil
            capabilities[capability]?.outcome = nil
            capabilities[capability]?.reason = "完成摘要未提供此能力的可识别状态。"
            capabilities[capability]?.impact = nil
            capabilities[capability]?.remedy = nil
        }
        guard let encoded, let data = encoded.data(using: .utf8),
              let report = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = report["capabilities"] as? [[String: Any]] else {
            summaryDiagnostics.append("capability_summary 缺失或格式无法解析；已有产物可查看，各能力保持未知。")
            return
        }
        for capability in PresentationCapability.allCases {
            let matches = items.filter { ($0["name"] as? String) == capability.rawValue }
            guard matches.count == 1, let item = matches.first else {
                summaryDiagnostics.append("能力摘要 \(capability.rawValue) \(matches.isEmpty ? "缺失" : "重复，无法确定状态")。")
                continue
            }
            let raw = item["status"] as? String
            let status: UIExecutionStatus
            switch raw {
            case "complete": status = .complete
            case "not_run": status = .notRun
            case "failed": status = .failed
            case "partial": status = .partial
            default: status = .unknown
            }
            capabilities[capability]?.status = status
            capabilities[capability]?.rawStatus = raw
            capabilities[capability]?.outcome = item["outcome"] as? String
            capabilities[capability]?.reason = item["reason"] as? String
            capabilities[capability]?.impact = item["impact"] as? String
            capabilities[capability]?.remedy = item["remedy"] as? String
            if status == .unknown, capabilities[capability]?.reason == nil {
                capabilities[capability]?.reason = "无法识别的能力状态：\(raw ?? "缺失")"
            }
        }
        // Optional vision/tagging/audio_board stay in the raw summary in detailText;
        // they never overwrite the status of the corresponding base capability.
    }
    private static func capability(stage: String, message: String) -> PresentationCapability? {
        let stage = stage.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch stage {
        case "shots": return .shots
        case "audio_decode", "audio_physical", "audio_tagging", "audio_board", "audio_complete": return .music
        case "transcribe":
            let message = message.lowercased()
            if message.range(of: "\\bocr\\b", options: .regularExpression) != nil || message.contains("文字识别") { return .ocr }
            if message.contains("关键帧") || message.contains("视觉校正") || message.contains("frame") || message.contains("correction") { return nil }
            if message.contains("whisper") || message.contains("语音转录") || message.contains("转录完成") || message.contains("transcrib") { return .transcription }
            return nil
        default: return nil
        }
    }
    private static func singleLine(_ text: String) -> String {
        text.split(whereSeparator: { $0.isNewline }).joined(separator: " ")
    }
}

/// Validation does not probe media, install models or use the network.
enum UIInputValidation {
    static func validate(urls: [URL], fileManager: FileManager = .default) throws -> URL {
        guard urls.count == 1 else { throw Failure("请一次选择或拖入一个本地视频文件。") }
        let url = urls[0]
        guard url.isFileURL, url.host == nil || url.host == "" || url.host == "localhost" else {
            throw Failure("请选择本地 MP4、MOV 或 MKV 视频文件。")
        }
        let local = url.standardizedFileURL
        guard ["mp4", "mov", "mkv"].contains(local.pathExtension.lowercased()) else {
            throw Failure("仅支持本地 MP4、MOV 或 MKV 视频文件。")
        }
        let regular = (try? local.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
        guard regular, fileManager.isReadableFile(atPath: local.path) else {
            throw Failure("找不到可读取的本地视频文件，请检查文件路径。")
        }
        return local
    }
    static func validate(path: String, fileManager: FileManager = .default) throws -> URL {
        let path = path.trimmingCharacters(in: .whitespacesAndNewlines)
        let expanded = (path as NSString).expandingTildeInPath
        guard !path.contains("://"), expanded.hasPrefix("/") else {
            throw Failure("请选择本地视频文件，或使用完整的本地文件路径。")
        }
        return try validate(urls: [URL(fileURLWithPath: expanded)], fileManager: fileManager)
    }
    struct Failure: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
