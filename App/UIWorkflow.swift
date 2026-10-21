import AppKit
import Combine
import Foundation
import Shared
import UniformTypeIdentifiers

/// Owns only App presentation and the baseline runner's lifecycle.
@MainActor
final class UIWorkflowModel: ObservableObject {
    @Published private(set) var state = UIWorkflowState()
    var onCompletion: (([String: String], URL) -> Void)?

    var mode: UIWorkflowMode { state.mode }
    var isBusy: Bool { state.isBusy }
    var statusText: String { state.statusText }
    var artifacts: [OutputArtifact] { state.artifacts }
    var hasResults: Bool { state.hasResults }
    var inputURL: URL? { state.inputURL }
    /// Frozen at acceptance, independent of subsequent output-picker changes.
    var outputDirectory: URL? { state.outputDirectory }
    var detailText: String { state.detailText }
    var nodes: [PathNodePresentation] {
        PresentationCapability.allCases.map { capability in
            let value = state.capabilities[capability] ?? UICapabilityState()
            return PathNodePresentation(
                id: capability.rawValue, name: capability.title,
                status: PathNodeStatus(rawValue: value.status.rawValue) ?? .unknown,
                statusLabel: value.displayLabel,
                reason: [value.reason, value.impact, value.remedy].compactMap { $0 }.joined(separator: "\n")
            )
        }
    }

    private let processRunner = ProcessRunner()
    private var processingTask: Task<Void, Never>?
    private var startedAt: Date?
    private var monotonicStart: TimeInterval?
    private let evidence = UIWorkflowEvidence()

    @discardableResult
    func accept(urls: [URL], outputDirectory: String, useCloudAPI: Bool) -> Bool {
        guard !isBusy else {
            state.showNotice("当前任务仍在处理，请等待结束或先取消。")
            record(kind: "busy_input")
            return false
        }
        do {
            return start(input: try UIInputValidation.validate(urls: urls),
                         outputDirectory: outputDirectory, useCloudAPI: useCloudAPI)
        } catch {
            state.showNotice(error.localizedDescription)
            record(kind: "invalid_input")
            return false
        }
    }

    @discardableResult
    func accept(path: String, outputDirectory: String, useCloudAPI: Bool) -> Bool {
        guard !isBusy else {
            state.showNotice("当前任务仍在处理，请等待结束或先取消。")
            record(kind: "busy_input")
            return false
        }
        do {
            return start(input: try UIInputValidation.validate(path: path),
                         outputDirectory: outputDirectory, useCloudAPI: useCloudAPI)
        } catch {
            state.showNotice(error.localizedDescription)
            record(kind: "invalid_input")
            return false
        }
    }

    func chooseInput(outputDirectory: String, useCloudAPI: Bool) {
        guard !isBusy else {
            state.showNotice("当前任务仍在处理，请等待结束或先取消。")
            record(kind: "busy_input")
            return
        }
        let panel = NSOpenPanel()
        var types: [UTType] = [.mpeg4Movie, .quickTimeMovie]
        if let mkv = UTType(filenameExtension: "mkv") { types.append(mkv) }
        panel.allowedContentTypes = types
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.prompt = "选择视频"
        if panel.runModal() == .OK {
            accept(urls: panel.urls, outputDirectory: outputDirectory, useCloudAPI: useCloudAPI)
        }
    }

    func cancel() {
        guard isBusy, let generation = state.generation else { return }
        state.requestCancellation(generation: generation)
        record(kind: "cancel_requested")
        // Do not cancel the consumer Task: that can end AsyncThrowingStream
        // consumption before waitUntilExit. The baseline runner terminates the
        // process tree and confirms stream termination after the process exits.
        processRunner.cancel()
    }

    func windowDidDisappear() { cancel() }

    func reset() {
        guard !isBusy else { return }
        state.reset()
        startedAt = nil
        monotonicStart = nil
        record(kind: "reset")
    }

    func revealResults() {
        let existing = state.artifacts.filter { artifact in
            var directory: ObjCBool = false
            return FileManager.default.fileExists(atPath: artifact.url.path, isDirectory: &directory)
                && directory.boolValue == (artifact.kind == .package)
        }
        guard !existing.isEmpty else {
            state.showNotice("结果路径已不存在，请在详情中检查原始输出位置。")
            record(kind: "results_missing")
            return
        }
        NSWorkspace.shared.activateFileViewerSelecting(existing.map(\.url))
        if existing.count != state.artifacts.count {
            state.showNotice("已显示现有结果；部分路径已不存在，可在详情中查看。")
        }
        record(kind: "results_revealed")
    }

    func detailText(for nodeID: String?) -> String {
        guard let nodeID, let capability = PresentationCapability(rawValue: nodeID),
              let value = state.capabilities[capability] else { return detailText }
        let heading = "\(capability.title) · \(value.status.label)"
        let reason = [value.reason, value.impact, value.remedy].compactMap { $0 }.joined(separator: "\n")
        return [heading, reason, detailText].filter { !$0.isEmpty }.joined(separator: "\n\n")
    }

    /// Same entry point as the production stream; generation rejects late events.
    func apply(event: IPCMessage, generation: UUID) {
        guard state.generation == generation, state.isActive else { return }
        state.apply(event: event, generation: generation)
        switch event {
        case .progress(let stage, let percent, let message):
            record(kind: "progress", raw: ["stage": stage, "percent": percent, "message": message])
        case .error(let stage, let code, let message):
            record(kind: "error", raw: ["stage": stage, "code": code, "message": message])
        case .complete(let output):
            record(kind: "complete", raw: ["output": output])
        }
    }

    /// Useful to callers that already own the current task context.
    func apply(event: IPCMessage) {
        guard let generation = state.generation else { return }
        apply(event: event, generation: generation)
    }

    private func start(input: URL, outputDirectory: String, useCloudAPI: Bool) -> Bool {
        let runtime = RuntimeConfiguration.resolve()
        do { try runtime.validateForLaunch() }
        catch {
            state.failToStart(error.localizedDescription)
            record(kind: "launch_failed", raw: ["message": error.localizedDescription])
            return false
        }
        let output = URL(fileURLWithPath: (outputDirectory as NSString).expandingTildeInPath,
                         isDirectory: true).standardizedFileURL
        let generation = UUID()
        state.begin(input: input, outputDirectory: output, generation: generation)
        startedAt = Date()
        monotonicStart = ProcessInfo.processInfo.systemUptime
        // Share the CLI's auto OCR strategy; explicit stages and final provenance
        // remain authoritative. Model/cloud semantics and physical audio stay unchanged.
        var arguments = [
            "--input", input.path,
            "--output-dir", output.path,
            "--library-root", output.appendingPathComponent("library", isDirectory: true).path,
            "--ocr-backend", "auto",
            "--rewrite", "off",
            "--model", SettingsStore.whisperModel,
            "--language", "auto",
            "--vision-model", SettingsStore.visionModel,
        ]
        if useCloudAPI { arguments.append("--cloud") }
        record(kind: "start", raw: ["arguments": arguments])
        // Register the process before returning from acceptance. A cancellation
        // in this same actor turn must reach this run rather than an older one.
        let stream = processRunner.run(configuration: runtime, arguments: arguments)
        processingTask = Task { [weak self] in
            guard let self else { return }
            var exitError: String?
            do {
                for try await event in stream {
                    apply(event: event, generation: generation)
                }
            } catch {
                exitError = error.localizedDescription
            }
            guard state.generation == generation, state.isActive else { return }
            state.finish(generation: generation, error: exitError)
            processingTask = nil
            var raw: [String: Any] = [:]
            if let exitError { raw["message"] = exitError }
            record(kind: "end", raw: raw)
            if state.hasUsableCompletion, mode != .cancelled, exitError == nil,
               let output = state.completionOutput,
               output["execution_status"] == nil || output["execution_status"] == "complete" {
                onCompletion?(output, input)
            }
        }
        return true
    }

    private func record(kind: String, raw: [String: Any] = [:]) {
        evidence.record(kind: kind, state: state, raw: raw, startedAt: startedAt,
                        elapsed: monotonicStart.map { max(0, ProcessInfo.processInfo.systemUptime - $0) })
    }
}

/// Explicit opt-in evidence only. This does not alter IPC, preferences, normal UI,
/// runtime arguments, or the process environment. Input hashing belongs to the
/// external acceptance manifest; the UI records the input path it actually used.
private final class UIWorkflowEvidence {
    private let logURL: URL?
    private let snapshotURL: URL?
    private let sessionID = UUID().uuidString

    init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        logURL = Self.explicitPath(environment["REELPACK_UI_EVENT_LOG"])
        snapshotURL = Self.explicitPath(environment["REELPACK_UI_STATE_SNAPSHOT"])
    }

    func record(kind: String, state: UIWorkflowState, raw: [String: Any], startedAt: Date?, elapsed: TimeInterval?) {
        guard logURL != nil || snapshotURL != nil else { return }
        var payload: [String: Any] = [
            "kind": kind, "timestamp": ISO8601DateFormatter().string(from: Date()),
            "session_id": sessionID, "mode": state.mode.rawValue,
            "status_text": state.statusText, "is_busy": state.isBusy,
            "completion_received": state.completionReceived,
            "artifacts": state.artifacts.map { ["kind": $0.kind.rawValue, "path": $0.url.path] },
            "issues": state.issues.map { ["stage": $0.stage, "code": $0.code, "message": $0.message] },
            "raw": raw,
        ]
        payload["capabilities"] = Dictionary(uniqueKeysWithValues: PresentationCapability.allCases.map { capability in
            let value = state.capabilities[capability] ?? UICapabilityState()
            var entry = ["status": value.status.rawValue, "title": capability.title]
            entry["raw_status"] = value.rawStatus
            entry["reason"] = value.reason
            entry["impact"] = value.impact
            entry["remedy"] = value.remedy
            entry["last_stage"] = value.lastStage
            entry["last_message"] = value.lastMessage
            return (capability.rawValue, entry)
        })
        if let generation = state.generation { payload["generation"] = generation.uuidString }
        if let input = state.inputURL { payload["input_file"] = input.path }
        if let output = state.outputDirectory { payload["output_directory"] = output.path }
        if let startedAt { payload["started_at"] = ISO8601DateFormatter().string(from: startedAt) }
        if let elapsed { payload["elapsed_seconds"] = elapsed }
        if let output = state.completionOutput { payload["completion_output"] = output }
        payload["detail_text"] = state.detailText
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) else { return }
        // Both opt-in outputs are JSONL: one compact state per event, including
        // the final state. Escaped raw message newlines stay within their JSON line.
        let line = data + Data([0x0A])
        for url in Set([logURL, snapshotURL].compactMap { $0 }) {
            do {
                if !FileManager.default.fileExists(atPath: url.path) {
                    guard FileManager.default.createFile(atPath: url.path, contents: nil,
                                                         attributes: [.posixPermissions: 0o600]) else { continue }
                }
                let file = try FileHandle(forWritingTo: url)
                defer { try? file.close() }
                try file.seekToEnd()
                try file.write(contentsOf: line)
            } catch { /* Evidence paths never change production processing. */ }
        }
    }

    private static func explicitPath(_ value: String?) -> URL? {
        guard let value, !value.isEmpty, !value.contains("://") else { return nil }
        let expanded = (value as NSString).expandingTildeInPath
        guard expanded.hasPrefix("/") else { return nil }
        return URL(fileURLWithPath: expanded).standardizedFileURL
    }
}
