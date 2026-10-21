import Foundation
import Darwin

public struct RuntimeImportCheck: Codable, Sendable {
    public let name: String
    public let available: Bool
    public let version: String?
    public let detail: String?
}

public struct RuntimePythonCheck: Codable, Sendable {
    public let requestedPath: String
    public var executable: String?
    public var architecture: String?
    public var version: String?
    public var imports: [RuntimeImportCheck] = []
    public var coreImport: RuntimeImportCheck?
    public let required: Bool
    public var available = false
    public var detail: String?
    public var exitCode: Int32?
    public var timedOut = false
}

public struct RuntimeToolCheck: Codable, Sendable {
    public let name: String
    public let path: String?
    public let required: Bool
    public let available: Bool
    public let detail: String
    public let exitCode: Int32?
    public let timedOut: Bool
}

public struct RuntimeRootCheck: Codable, Sendable {
    public let path: String
    public let exists: Bool
    public let writable: Bool
    public let writabilityCheck: String
    public let sizeBytes: Int64
    public let sizePartial: Bool
    public let modelsReady: Bool?

    private enum CodingKeys: String, CodingKey {
        case path, exists, writable
        case writabilityCheck = "writability_check"
        case sizeBytes = "size_bytes"
        case sizePartial = "size_partial"
        case modelsReady = "models_ready"
    }
}

public struct RuntimeModelFileCheck: Codable, Sendable {
    public let path: String
    public let exists: Bool
    public let sizeBytes: Int64
    public let ready: Bool?
    public let rows: Int?
    private enum CodingKeys: String, CodingKey {
        case path, exists, ready, rows
        case sizeBytes = "size_bytes"
    }
}

public struct RuntimeModelCheck: Codable, Sendable {
    public let name: String
    public let repository: String
    public let path: String?
    public let ready: Bool?
    public let sizeBytes: Int64
    public let files: [RuntimeModelFileCheck]
    public let verification: String
    public let inference: String
    public let checkpoint: RuntimeModelFileCheck?
    public let labels: RuntimeModelFileCheck?
    public let sed: RuntimeModelFileCheck?
    public let labelsReady: Bool?
    public let sedReady: Bool?
    public let requiredDownloadBytes: Int64?
    public let automaticDownload: Bool?
    public let isOptional: Bool?
    private enum CodingKeys: String, CodingKey {
        case name, repository, path, ready, files, verification, inference, checkpoint, labels, sed
        case sizeBytes = "size_bytes"
        case labelsReady = "labels_ready"
        case sedReady = "sed_ready"
        case requiredDownloadBytes = "required_download_bytes"
        case automaticDownload = "automatic_download"
        case isOptional = "optional"
    }
}

public struct RuntimeExecutionCheck: Codable, Sendable {
    public let status: String
    public let detail: String
}

public struct RuntimeDiagnosticReport: Encodable, Sendable {
    public let schemaVersion = 1
    public let product = "Reelpack"
    public let processArchitecture: String
    public let hostArchitecture: String
    public let runtimeRoot: String
    public let resourceSource: String
    public let resourcesRoot: String?
    public let coreScript: String?
    public let environment: [String: String]
    public let python: RuntimePythonCheck
    public let helpers: [String: RuntimePythonCheck]
    public let tools: [RuntimeToolCheck]
    public let roots: [String: RuntimeRootCheck]
    public let models: [String: RuntimeModelCheck]
    public let jit: RuntimeExecutionCheck
    public let rootInspectionDetail: String?
    public let ready: Bool
    public let issues: [String]
    public let installationHint: String?
    public var exitCode: Int32 { ready ? 0 : 1 }

    public func writeJSON(to handle: FileHandle = .standardOutput) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        handle.write(try encoder.encode(self))
        handle.write(Data("\n".utf8))
    }
}

/// Read-only metadata checks: never prepare caches, import ML libraries, or run JIT.
public enum RuntimeDiagnostics {
    public static func exitIfRequested(arguments: [String] = CommandLine.arguments) {
        guard arguments.contains("--runtime-check") else { return }
        var environment = ProcessInfo.processInfo.environment
        for (argument, variable) in [("--output-dir", "REELPACK_OUTPUT_ROOT"), ("--library-root", "REELPACK_LIBRARY_ROOT")] {
            if let index = arguments.firstIndex(of: argument), index + 1 < arguments.count {
                environment[variable] = arguments[index + 1]
            } else if let value = arguments.first(where: { $0.hasPrefix(argument + "=") }) {
                environment[variable] = String(value.dropFirst(argument.count + 1))
            }
        }
        let report = check(configuration: .resolve(environment: environment))
        do {
            try report.writeJSON()
            exit(report.exitCode)
        } catch {
            FileHandle.standardError.write(Data("Reelpack 诊断 JSON 输出失败：\(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    public static func check(configuration: RuntimeConfiguration = .resolve()) -> RuntimeDiagnosticReport {
        let paths = configuration.paths
        let rootInspection = checkRuntimeRoots(configuration: configuration)
        let python = checkPython(paths.python, modules: ["numpy", "cv2", "requests", "PIL", "faster_whisper"],
                                 coreScript: paths.coreScript, required: true, metadataOnly: true, configuration: configuration)
        let helpers = [
            "audio": checkPython(paths.audioPython, modules: ["numpy", "librosa", "soundfile", "pyloudnorm", "PIL"],
                                 required: true, metadataOnly: true, configuration: configuration),
            "transnet": checkPython(paths.transnetPython, modules: ["numpy", "cv2", "torch", "transnetv2_pytorch"],
                                    required: true, metadataOnly: true, configuration: configuration),
            "ocr": checkPython(paths.ocrPython, modules: ["numpy", "cv2", "onnxruntime", "rapidocr_onnxruntime"],
                               required: true, metadataOnly: true, configuration: configuration),
            // panns_inference may download files during import. Only inspect its module spec.
            "music": checkPython(paths.musicPython, modules: ["numpy", "librosa", "soundfile", "torch", "panns_inference"],
                                 required: false, metadataOnly: true, configuration: configuration),
        ]
        let tools = [checkMediaTool("ffmpeg", configuration: configuration),
                     checkMediaTool("ffprobe", configuration: configuration),
                     checkNativeOCR(configuration: configuration),
                     checkOllama(configuration: configuration)]
        var issues = configuration.launchIssues
        if rootInspection.payload == nil {
            issues.append("运行时路径检查失败：\(rootInspection.detail ?? "无诊断结果")")
        } else if rootInspection.payload?.roots["cache"]?.writable == false {
            issues.append("候选缓存根的目录权限不可写；实际启动将验证并尝试临时回退。")
        }
        if !python.available { issues.append("主 Python 依赖元数据或 Core 语法检查失败：\(python.detail ?? python.requestedPath)") }
        for name in ["audio", "transnet", "ocr"] {
            if let helper = helpers[name], !helper.available {
                if name == "audio" {
                    issues.append("音频环境需要更新，请运行 --repair-env audio；会先备份。\(helper.detail ?? helper.requestedPath)")
                } else {
                    issues.append("\(name) Python 依赖检查失败：\(helper.detail ?? helper.requestedPath)")
                }
            }
        }
        for tool in tools where tool.required && !tool.available {
            issues.append("\(tool.name) 检查失败：\(tool.detail)")
        }
        let ready = issues.isEmpty
        return RuntimeDiagnosticReport(processArchitecture: paths.architecture,
                                       hostArchitecture: RuntimePaths.hostArchitecture,
                                       runtimeRoot: paths.runtimeRoot.path,
                                       resourceSource: paths.resourceSource.rawValue,
                                       resourcesRoot: paths.resourcesRoot?.path, coreScript: configuration.scriptPath,
                                       environment: configuration.diagnosticEnvironment,
                                       python: python, helpers: helpers, tools: tools,
                                       roots: rootInspection.payload?.roots ?? [:], models: rootInspection.payload?.models ?? [:],
                                       jit: rootInspection.payload?.jit ?? RuntimeExecutionCheck(status: "not_run", detail: "只读诊断不执行 JIT 编译。"),
                                       rootInspectionDetail: rootInspection.detail,
                                       ready: ready, issues: issues,
                                       installationHint: ready ? nil : configuration.installationHint)
    }

    private struct RootsPayload: Decodable {
        let roots: [String: RuntimeRootCheck]
        let models: [String: RuntimeModelCheck]
        let jit: RuntimeExecutionCheck
    }

    private static func checkRuntimeRoots(configuration: RuntimeConfiguration) -> (payload: RootsPayload?, detail: String?) {
        guard let script = configuration.paths.coreScript?.deletingLastPathComponent()
            .appendingPathComponent("runtime_environment.py"),
              FileManager.default.isReadableFile(atPath: script.path) else {
            return (nil, "未找到 Core/runtime_environment.py。")
        }
        var environment = configuration.environment
        environment["REELPACK_READONLY_CHECK"] = "1"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        let result = RuntimeCommand.run(executable: configuration.paths.python,
                                       arguments: ["-I", "-B", script.path, "--inspect"],
                                       environment: environment, timeout: 10)
        guard result.exitCode == 0, !result.timedOut,
              let payload = try? JSONDecoder().decode(RootsPayload.self, from: Data(result.output.utf8)) else {
            return (nil, result.detail)
        }
        return (payload, nil)
    }

    private struct PythonPayload: Decodable {
        let executable: String
        let architecture: String
        let version: String
        let imports: [RuntimeImportCheck]
        let coreImport: RuntimeImportCheck?
    }

    private static func checkPython(
        _ executable: URL, modules: [String], coreScript: URL? = nil,
        required: Bool, metadataOnly: Bool = false, configuration: RuntimeConfiguration
    ) -> RuntimePythonCheck {
        var check = RuntimePythonCheck(requestedPath: executable.path, required: required)
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            check.detail = "未安装：\(executable.path)"
            return check
        }
        let moduleJSON = String(data: (try? JSONEncoder().encode(modules)) ?? Data("[]".utf8), encoding: .utf8) ?? "[]"
        var environment = configuration.environment
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["REELPACK_READONLY_CHECK"] = "1"
        let result = RuntimeCommand.run(executable: executable,
                                        arguments: ["-I", "-B", "-c", pythonProbe, moduleJSON, coreScript?.path ?? "",
                                                    metadataOnly ? "spec" : "import"],
                                        environment: environment, timeout: 30)
        check.exitCode = result.exitCode
        check.timedOut = result.timedOut
        let payload = result.output.split(separator: "\n").reversed().compactMap { line -> PythonPayload? in
            try? JSONDecoder().decode(PythonPayload.self, from: Data(line.utf8))
        }.first
        guard let payload else {
            check.detail = result.timedOut ? "导入检查超时（30 秒）" : result.detail
            return check
        }
        check.executable = payload.executable
        check.architecture = payload.architecture
        check.version = payload.version
        check.imports = payload.imports
        check.coreImport = payload.coreImport
        let importsReady = payload.imports.allSatisfy(\.available)
        let coreReady = coreScript == nil || payload.coreImport?.available == true
        let architectureReady = payload.architecture == configuration.paths.architecture
        check.available = result.exitCode == 0 && !result.timedOut && importsReady && coreReady && architectureReady
        if !architectureReady {
            check.detail = "Python 架构 \(payload.architecture) 与应用 \(configuration.paths.architecture) 不匹配。"
        } else {
            let failures = (payload.imports + (payload.coreImport.map { [$0] } ?? []))
                .filter { !$0.available }.map { "\($0.name): \($0.detail ?? "不可用")" }
            check.detail = failures.isEmpty ? (metadataOnly ? "模块元数据/Core 语法已检查；未导入库、执行 JIT 或模型推理。" : "imports 已就绪；未执行 JIT。")
                : failures.joined(separator: "; ")
        }
        return check
    }

    private static func checkMediaTool(_ name: String, configuration: RuntimeConfiguration) -> RuntimeToolCheck {
        guard let executable = configuration.executable(named: name) else {
            return RuntimeToolCheck(name: name, path: nil, required: true, available: false,
                                    detail: "PATH 中未找到 \(name)。请运行 install.sh。", exitCode: nil, timedOut: false)
        }
        let result = RuntimeCommand.run(executable: executable, arguments: ["-version"],
                                        environment: configuration.environment, timeout: 5)
        return RuntimeToolCheck(name: name, path: executable.path, required: true,
                                available: result.exitCode == 0 && !result.timedOut,
                                detail: result.detail, exitCode: result.exitCode, timedOut: result.timedOut)
    }

    private static func checkNativeOCR(configuration: RuntimeConfiguration) -> RuntimeToolCheck {
        guard let executable = configuration.paths.ocrBinary,
              FileManager.default.isExecutableFile(atPath: executable.path) else {
            return RuntimeToolCheck(name: "macos_ocr", path: configuration.paths.ocrBinary?.path, required: true,
                                    available: false, detail: "未找到可执行的原生 OCR helper（macOS 14+）。请运行 install.sh。",
                                    exitCode: nil, timedOut: false)
        }
        // No images or user data: expected usage exit verifies that dyld can load the helper.
        let result = RuntimeCommand.run(executable: executable, arguments: [], environment: configuration.environment, timeout: 5)
        let available = result.exitCode == 2 && !result.timedOut && result.output.contains("用法: macos_ocr")
        return RuntimeToolCheck(name: "macos_ocr", path: executable.path, required: true, available: available,
                                detail: available ? "原生 OCR helper 已启动（macOS 14+；无输入返回用法）。" : result.detail,
                                exitCode: result.exitCode, timedOut: result.timedOut)
    }

    private static func checkOllama(configuration: RuntimeConfiguration) -> RuntimeToolCheck {
        let result = RuntimeCommand.run(executable: URL(fileURLWithPath: "/usr/bin/curl"),
                                        arguments: ["--disable", "--silent", "--show-error", "--noproxy", "*", "--connect-timeout", "2",
                                                    "--max-time", "4", "--output", "/dev/null", "--write-out", "%{http_code}",
                                                    "http://127.0.0.1:11434/api/tags"],
                                        environment: configuration.environment, timeout: 5)
        let available = result.exitCode == 0 && result.output.trimmingCharacters(in: .whitespacesAndNewlines) == "200"
        return RuntimeToolCheck(name: "Ollama", path: "http://127.0.0.1:11434", required: false,
                                available: available, detail: available ? "localhost:11434" : "未连接（可选；连接超时 2 秒，总超时 4 秒）。",
                                exitCode: result.exitCode, timedOut: result.timedOut)
    }

    private static let pythonProbe = #"""
import ast, contextlib, importlib, importlib.metadata, importlib.util, io, json, platform, sys
sys.dont_write_bytecode = True
modules, script, mode = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
results, core_result = [], None

def inspect(name):
    try:
        if mode == 'spec':
            found = importlib.util.find_spec(name) is not None
            distribution = {'cv2': 'opencv-python', 'PIL': 'Pillow', 'panns_inference': 'panns-inference',
                            'faster_whisper': 'faster-whisper', 'transnetv2_pytorch': 'transnetv2-pytorch',
                            'rapidocr_onnxruntime': 'rapidocr-onnxruntime'}.get(name, name)
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = None
            return dict(name=name, available=found, version=version,
                        detail='module metadata only; JIT not run' if found else 'module not installed')
        module = importlib.import_module(name)
        version = getattr(module, '__version__', None)
        return dict(name=name, available=True, version=str(version) if version is not None else None, detail=None)
    except Exception as exc:
        return dict(name=name, available=False, version=None, detail=(type(exc).__name__ + ': ' + str(exc))[:500])

# Libraries may write logs. Keep stdout a single JSON line and never instantiate a model.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    results = [inspect(name) for name in modules]
    if script:
        from pathlib import Path
        try:
            if not Path(script).is_file():
                raise FileNotFoundError(script)
            if mode == 'spec':
                ast.parse(Path(script).read_text(encoding='utf-8'), filename=script)
                detail = 'syntax checked without executing Core or preparing caches; JIT not run'
            else:
                sys.path.insert(0, str(Path(script).parent))
                spec = importlib.util.spec_from_file_location('reelpack_runtime_core', script)
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
                detail = 'import checked; JIT not run'
            core_result = dict(name='core', available=True, version=None, detail=detail)
        except Exception as exc:
            core_result = dict(name='core', available=False, version=None, detail=(type(exc).__name__ + ': ' + str(exc))[:500])
print(json.dumps(dict(executable=sys.executable, architecture=platform.machine(), version=platform.python_version(),
                     imports=results, coreImport=core_result)))
"""#
}

private enum RuntimeCommand {
    struct Result {
        let exitCode: Int32?
        let timedOut: Bool
        let output: String
        let error: String?
        var detail: String {
            if timedOut { return "执行超时" }
            if let error { return error }
            return String(output.split(separator: "\n").first ?? "没有输出")
        }
    }

    private final class OutputBuffer: @unchecked Sendable {
        private let lock = NSLock()
        private var data = Data()
        func append(_ chunk: Data) { lock.withLock { data.append(chunk.prefix(max(0, 65_536 - data.count))) } }
        var string: String { lock.withLock { String(decoding: data, as: UTF8.self) } }
    }

    static func run(executable: URL, arguments: [String], environment: [String: String], timeout: TimeInterval) -> Result {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment
        process.standardInput = FileHandle.nullDevice
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        let output = OutputBuffer()
        let terminated = DispatchSemaphore(value: 0)
        let drained = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in terminated.signal() }
        do {
            try process.run()
        } catch {
            return Result(exitCode: nil, timedOut: false, output: "", error: error.localizedDescription)
        }
        DispatchQueue.global(qos: .utility).async {
            while let chunk = try? pipe.fileHandleForReading.read(upToCount: 4096), !chunk.isEmpty {
                output.append(chunk)
            }
            drained.signal()
        }
        let timedOut = terminated.wait(timeout: .now() + timeout) == .timedOut
        if timedOut {
            if process.isRunning { process.terminate() }
            if terminated.wait(timeout: .now() + 1) == .timedOut {
                if process.isRunning { kill(process.processIdentifier, SIGKILL) }
                _ = terminated.wait(timeout: .now() + 1)
            }
        }
        _ = drained.wait(timeout: .now() + 1)
        return Result(exitCode: process.isRunning ? nil : process.terminationStatus,
                      timedOut: timedOut, output: output.string, error: nil)
    }
}
