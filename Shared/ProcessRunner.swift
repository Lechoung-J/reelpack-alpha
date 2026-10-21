import Foundation
import Darwin

/// Reads JSON lines from a subprocess stdout and yields parsed IPCMessage values.
/// Runs on a background Task so the main thread stays responsive.
public final class ProcessRunner: @unchecked Sendable {
    public init() {}
    private var process: Process?
    private var isCancelled = false
    private var generation = 0
    private let lock = NSLock()

    /// Launch with the same resolved interpreter and environment used by diagnostics.
    public func run(configuration: RuntimeConfiguration, arguments: [String]) -> AsyncThrowingStream<IPCMessage, Error> {
        do {
            try configuration.validateForLaunch()
        } catch {
            return AsyncThrowingStream { $0.finish(throwing: error) }
        }
        return run(pythonPath: configuration.pythonPath, scriptPath: configuration.scriptPath!, arguments: arguments,
                   environment: configuration.environment, inheritEnvironment: false,
                   workingDirectory: configuration.paths.coreScript?.deletingLastPathComponent())
    }

    /// Start Python pipeline and yield progress/completion/error messages.
    public func run(
        pythonPath: String = "python3",
        scriptPath: String,
        arguments: [String],
        environment: [String: String] = [:],
        inheritEnvironment: Bool = true,
        workingDirectory: URL? = nil
    ) -> AsyncThrowingStream<IPCMessage, Error> {
        AsyncThrowingStream { continuation in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: pythonPath)
            process.arguments = [scriptPath] + arguments
            var childEnvironment = inheritEnvironment ? ProcessInfo.processInfo.environment : [:]
            for (key, value) in environment { childEnvironment[key] = value }
            process.environment = childEnvironment
            process.currentDirectoryURL = workingDirectory
            process.standardInput = FileHandle.nullDevice

            let stdoutPipe = Pipe()
            process.standardOutput = stdoutPipe
            let stderrPipe = Pipe()
            process.standardError = stderrPipe

            let runGeneration = self.lock.withLock {
                self.generation += 1
                self.isCancelled = false
                self.process = process
                return self.generation
            }

            let stderrLock = NSLock()
            var stderrData = Data()
            stderrPipe.fileHandleForReading.readabilityHandler = { handle in
                let chunk = handle.availableData
                guard !chunk.isEmpty else { return }
                stderrLock.withLock { stderrData.append(chunk) }
            }

            // N.B. stdout reading MUST run off main thread — FileHandle.read blocks.
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    guard !self.lock.withLock({ self.isCancelled || self.generation != runGeneration }) else {
                        continuation.finish()
                        return
                    }

                    try process.run()

                    if self.lock.withLock({ self.isCancelled || self.generation != runGeneration }) {
                        self.terminateProcessTree(rootPID: process.processIdentifier)
                    }

                    let handle = stdoutPipe.fileHandleForReading
                    var buffer = Data()

                    while self.lock.withLock({ !self.isCancelled }) {
                        // On a pipe, read(upToCount:) can wait for the requested
                        // byte count or EOF, delaying small flushed NDJSON messages.
                        // availableData blocks only until bytes are available; keep
                        // the existing byte buffer so split UTF-8/lines still parse.
                        let chunk = handle.availableData
                        guard !chunk.isEmpty else { break } // EOF; then await exit below.
                        buffer.append(chunk)

                        // Parse complete NDJSON lines
                        while let newline = buffer.firstIndex(of: 0x0A) {
                            let lineData = buffer[..<newline]
                            buffer.removeSubrange(...newline)
                            if let line = String(data: lineData, encoding: .utf8),
                               let msg = IPCMessage.parse(line) {
                                continuation.yield(msg)
                            }
                        }
                    }

                    process.waitUntilExit()
                    stderrPipe.fileHandleForReading.readabilityHandler = nil

                    let wasCancelled = self.lock.withLock {
                        self.isCancelled || self.generation != runGeneration
                    }
                    if process.terminationStatus != 0 && !wasCancelled {
                        var detail = ""
                        let captured = stderrLock.withLock { stderrData }
                        if let stderr = String(data: captured, encoding: .utf8), !stderr.isEmpty {
                            detail = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                                .split(separator: "\n").suffix(3).joined(separator: "\n")
                        }
                        continuation.finish(throwing: ProcessError.exitCode(process.terminationStatus, detail: detail))
                    } else {
                        continuation.finish()
                    }
                    self.lock.withLock {
                        if self.generation == runGeneration { self.process = nil }
                    }
                } catch {
                    stderrPipe.fileHandleForReading.readabilityHandler = nil
                    let wasCancelled = self.lock.withLock {
                        let cancelled = self.isCancelled || self.generation != runGeneration
                        if self.generation == runGeneration { self.process = nil }
                        return cancelled
                    }
                    if wasCancelled {
                        continuation.finish()
                    } else {
                        continuation.finish(throwing: error)
                    }
                }
            }

            continuation.onTermination = { @Sendable _ in
                self.cancel()
            }
        }
    }

    public func cancel() {
        let runningProcess = lock.withLock {
            isCancelled = true
            return process
        }
        guard let runningProcess, runningProcess.isRunning else { return }
        terminateProcessTree(rootPID: runningProcess.processIdentifier)
    }

    private func terminateProcessTree(rootPID: pid_t) {
        let descendants = descendantPIDs(of: rootPID)
        for pid in descendants.reversed() { kill(pid, SIGTERM) }
        kill(rootPID, SIGTERM)

        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 1) {
            for pid in descendants where kill(pid, 0) == 0 { kill(pid, SIGKILL) }
            if kill(rootPID, 0) == 0 { kill(rootPID, SIGKILL) }
        }
    }

    private func descendantPIDs(of rootPID: pid_t) -> [pid_t] {
        var result: [pid_t] = []
        var parents = [rootPID]

        while !parents.isEmpty {
            let parent = parents.removeFirst()
            let capacity = proc_listchildpids(parent, nil, 0)
            guard capacity > 0 else { continue }

            var children = [pid_t](repeating: 0, count: Int(capacity))
            let count = children.withUnsafeMutableBytes {
                proc_listchildpids(parent, $0.baseAddress, Int32($0.count))
            }
            guard count > 0 else { continue }

            children = Array(children.prefix(Int(count))).filter { $0 > 0 }
            result.append(contentsOf: children)
            parents.append(contentsOf: children)
        }

        return result
    }
}

public enum ProcessError: LocalizedError {
    case exitCode(Int32, detail: String)

    public var errorDescription: String? {
        switch self {
        case .exitCode(let code, let detail):
            var msg = "处理意外中断 (exit code: \(code))"
            if !detail.isEmpty { msg += "\n\(detail)" }
            return msg
        }
    }
}
