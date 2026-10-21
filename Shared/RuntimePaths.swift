import Foundation
import Darwin

/// Installation contract shared by the app, CLI and dependency diagnostics.
/// Resources are immutable; all five Python environments live under runtimeRoot.
public struct RuntimePaths: Sendable {
    public enum ResourceSource: String, Sendable {
        case bundled
        case development
        case missing
    }

    public static var processArchitecture: String {
        #if arch(arm64)
        return "arm64"
        #else
        return "x86_64"
        #endif
    }

    public static var hostArchitecture: String {
        var supportsARM: Int32 = 0
        var size = MemoryLayout<Int32>.size
        if sysctlbyname("hw.optional.arm64", &supportsARM, &size, nil, 0) == 0, supportsARM == 1 {
            return "arm64"
        }
        return "x86_64"
    }

    public let architecture: String
    public let runtimeRoot: URL
    public let resourcesRoot: URL?
    public let resourceSource: ResourceSource
    public let coreScript: URL?
    public let python: URL
    public let audioPython: URL
    public let transnetPython: URL
    public let ocrPython: URL
    public let musicPython: URL
    public let ocrBinary: URL?
    public let configurationIssues: [String]

    public init(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        bundle: Bundle = .main,
        executableURL: URL? = Bundle.main.executableURL,
        currentDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) {
        architecture = Self.processArchitecture
        let defaultRoot = homeDirectory.appendingPathComponent("Library/Application Support/Reelpack/runtime")
            .appendingPathComponent(architecture, isDirectory: true)
        var issues: [String] = []
        if let override = environment["REELPACK_RUNTIME_ROOT"], !override.isEmpty {
            let expanded = (override as NSString).expandingTildeInPath
            if expanded.hasPrefix("/") {
                runtimeRoot = URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL
            } else {
                runtimeRoot = defaultRoot
                issues.append("REELPACK_RUNTIME_ROOT 必须是绝对路径（也可使用 ~）。")
            }
        } else {
            runtimeRoot = defaultRoot
        }
        configurationIssues = issues
        python = Self.python(in: runtimeRoot, environmentName: ".venv")
        audioPython = Self.python(in: runtimeRoot, environmentName: ".venv-audio")
        transnetPython = Self.python(in: runtimeRoot, environmentName: ".venv-transnet")
        ocrPython = Self.python(in: runtimeRoot, environmentName: ".venv-ocr")
        musicPython = Self.python(in: runtimeRoot, environmentName: ".venv-music")

        let executable = executableURL?.standardizedFileURL.resolvingSymlinksInPath()
        let appResources: URL?
        if bundle.bundleURL.pathExtension.lowercased() == "app" {
            appResources = bundle.resourceURL
                ?? bundle.bundleURL.appendingPathComponent("Contents/Resources", isDirectory: true)
        } else if let executable,
                  executable.deletingLastPathComponent().lastPathComponent == "MacOS",
                  executable.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "Contents",
                  executable.deletingLastPathComponent().deletingLastPathComponent()
                    .deletingLastPathComponent().pathExtension.lowercased() == "app" {
            appResources = executable.deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Resources", isDirectory: true)
        } else {
            appResources = nil
        }

        if let appResources {
            // A packaged executable must never borrow Core from an unrelated cwd/source checkout.
            resourcesRoot = appResources
            resourceSource = .bundled
            coreScript = appResources.appendingPathComponent("Core/core.py")
            ocrBinary = appResources.appendingPathComponent("Tools/macos_ocr/macos_ocr")
        } else {
            // Prefer the executable's checkout over cwd, including when launched from Finder.
            let roots = Self.ancestors(of: executable?.deletingLastPathComponent())
                + Self.ancestors(of: currentDirectory.standardizedFileURL)
            let root = roots.first {
                FileManager.default.isReadableFile(atPath: $0.appendingPathComponent("Core/core.py").path)
            }
            resourcesRoot = root
            resourceSource = root == nil ? .missing : .development
            coreScript = root?.appendingPathComponent("Core/core.py")
            ocrBinary = root?.appendingPathComponent("Tools/macos_ocr/macos_ocr")
        }
    }

    private static func python(in root: URL, environmentName: String) -> URL {
        let bin = root.appendingPathComponent(environmentName).appendingPathComponent("bin")
        let python = bin.appendingPathComponent("python")
        if FileManager.default.isExecutableFile(atPath: python.path) { return python }
        let python3 = bin.appendingPathComponent("python3")
        return FileManager.default.isExecutableFile(atPath: python3.path) ? python3 : python
    }

    private static func ancestors(of directory: URL?) -> [URL] {
        guard let directory else { return [] }
        var candidate = directory.standardizedFileURL
        var result: [URL] = []
        while true {
            result.append(candidate)
            // Foundation can turn deleting '/' into '/..' instead of returning '/'.
            if candidate.path.isEmpty || candidate.path == "/" { break }
            let parent = candidate.deletingLastPathComponent().standardizedFileURL
            if parent.path == candidate.path { break }
            candidate = parent
        }
        return result
    }
}

/// Read-only candidates for disposable acceleration caches and persistent models.
/// Python's prepare_runtime_environment verifies writes at launch, including sandbox
/// EPERM, and can replace the candidate with the stable per-user temporary root.
public struct RuntimeCachePaths: Sendable {
    public let root: URL
    public let fallbackRoot: URL
    public let numba: URL
    public let matplotlib: URL
    public let pycache: URL
    public let huggingFaceHome: String
    public let torchHome: String

    public init(environment: [String: String], homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser) {
        let home = Self.absolute(environment["HOME"], homeDirectory: homeDirectory) ?? homeDirectory
        let primary = Self.absolute(environment["REELPACK_CACHE_ROOT"], homeDirectory: home)
            ?? home.appendingPathComponent("Library/Caches/Reelpack", isDirectory: true)
        let temporary = Self.absolute(environment["TMPDIR"], homeDirectory: home)
            ?? URL(fileURLWithPath: "/tmp", isDirectory: true)
        fallbackRoot = temporary.appendingPathComponent("reelpack-cache-\(getuid())", isDirectory: true)
        let children = [primary, primary.appendingPathComponent("numba"), primary.appendingPathComponent("mpl"),
                        primary.appendingPathComponent("pycache")]
        root = children.allSatisfy(Self.writableWithoutProbe) ? primary : fallbackRoot
        numba = root.appendingPathComponent("numba", isDirectory: true)
        matplotlib = root.appendingPathComponent("mpl", isDirectory: true)
        pycache = root.appendingPathComponent("pycache", isDirectory: true)
        let persistent = Self.absolute(environment["XDG_CACHE_HOME"], homeDirectory: home)
            ?? home.appendingPathComponent(".cache", isDirectory: true)
        huggingFaceHome = environment["HF_HOME"].flatMap { $0.isEmpty ? nil : $0 }
            ?? persistent.appendingPathComponent("huggingface", isDirectory: true).path
        torchHome = environment["TORCH_HOME"].flatMap { $0.isEmpty ? nil : $0 }
            ?? persistent.appendingPathComponent("torch", isDirectory: true).path
    }

    private static func absolute(_ value: String?, homeDirectory: URL) -> URL? {
        guard let value, !value.isEmpty else { return nil }
        let expanded = value == "~" ? homeDirectory.path
            : value.hasPrefix("~/") ? homeDirectory.appendingPathComponent(String(value.dropFirst(2))).path
            : (value as NSString).expandingTildeInPath
        guard expanded.hasPrefix("/") else { return nil }
        return URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL
    }

    private static func writableWithoutProbe(_ directory: URL) -> Bool {
        var candidate = directory.standardizedFileURL
        while true {
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDirectory) {
                guard isDirectory.boolValue,
                      let attributes = try? FileManager.default.attributesOfItem(atPath: candidate.path),
                      let permissions = attributes[.posixPermissions] as? NSNumber else { return false }
                return permissions.intValue & 0o222 != 0 && permissions.intValue & 0o111 != 0
                    && access(candidate.path, W_OK | X_OK) == 0
            }
            guard candidate.path != "/", !candidate.path.isEmpty else { return false }
            let parent = candidate.deletingLastPathComponent().standardizedFileURL
            guard parent.path != candidate.path else { return false }
            candidate = parent
        }
    }
}

public struct RuntimeConfiguration: Sendable {
    public let paths: RuntimePaths
    public let environment: [String: String]
    public var pythonPath: String { paths.python.path }
    public var scriptPath: String? { paths.coreScript?.path }

    public static func resolve(
        environment inheritedEnvironment: [String: String] = ProcessInfo.processInfo.environment,
        bundle: Bundle = .main,
        executableURL: URL? = Bundle.main.executableURL,
        currentDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> RuntimeConfiguration {
        let paths = RuntimePaths(environment: inheritedEnvironment, bundle: bundle, executableURL: executableURL,
                                 currentDirectory: currentDirectory, homeDirectory: homeDirectory)
        var environment = inheritedEnvironment
        // Prevent a shell's Python configuration from changing the installed interpreter's imports.
        environment.removeValue(forKey: "PYTHONHOME")
        environment.removeValue(forKey: "PYTHONPATH")
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["REELPACK_RUNTIME_ROOT"] = paths.runtimeRoot.path
        let cache = RuntimeCachePaths(environment: inheritedEnvironment, homeDirectory: homeDirectory)
        environment["REELPACK_CACHE_ROOT"] = cache.root.path
        environment["NUMBA_CACHE_DIR"] = cache.numba.path
        environment["MPLCONFIGDIR"] = cache.matplotlib.path
        environment["PYTHONPYCACHEPREFIX"] = cache.pycache.path
        environment["HF_HOME"] = cache.huggingFaceHome
        environment["TORCH_HOME"] = cache.torchHome
        environment["REELPACK_AUDIO_PYTHON"] = paths.audioPython.path
        environment["REELPACK_TRANSNET_PYTHON"] = paths.transnetPython.path
        environment["REELPACK_OCR_PYTHON"] = paths.ocrPython.path
        environment["REELPACK_MUSIC_PYTHON"] = paths.musicPython.path
        environment["REELPACK_OCR_BIN"] = paths.ocrBinary?.path ?? ""
        let candidates = [paths.runtimeRoot.appendingPathComponent("bin").path, "/opt/homebrew/bin", "/usr/local/bin"]
            + (inheritedEnvironment["PATH"] ?? "").split(separator: ":").map(String.init)
            + ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        var seen = Set<String>()
        // Absolute entries keep PATH independent of Finder's working directory.
        environment["PATH"] = candidates.filter { $0.hasPrefix("/") && seen.insert($0).inserted }.joined(separator: ":")
        return RuntimeConfiguration(paths: paths, environment: environment)
    }

    /// Only the runtime contract is exposed in diagnostics, never arbitrary inherited env/credentials.
    public var diagnosticEnvironment: [String: String] {
        let keys = ["REELPACK_RUNTIME_ROOT", "REELPACK_AUDIO_PYTHON", "REELPACK_TRANSNET_PYTHON",
                    "REELPACK_OCR_PYTHON", "REELPACK_MUSIC_PYTHON", "REELPACK_OCR_BIN", "PATH",
                    "PYTHONNOUSERSITE", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
                    "REELPACK_CACHE_ROOT", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX",
                    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME",
                    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "REELPACK_OUTPUT_ROOT", "REELPACK_LIBRARY_ROOT",
                    "REELPACK_PANNS_CHECKPOINT", "REELPACK_PANNS_SED_CHECKPOINT", "REELPACK_PANNS_LABELS"]
        return environment.filter { keys.contains($0.key) }
    }

    public func executable(named name: String) -> URL? {
        (environment["PATH"] ?? "").split(separator: ":").lazy.map {
            URL(fileURLWithPath: String($0), isDirectory: true).appendingPathComponent(name)
        }.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    public var launchIssues: [String] {
        var issues = paths.configurationIssues
        if !FileManager.default.isExecutableFile(atPath: pythonPath) {
            issues.append("未找到运行环境的 Python：\(pythonPath)")
        }
        if let scriptPath {
            if !FileManager.default.isReadableFile(atPath: scriptPath) {
                issues.append("未找到 Reelpack Core：\(scriptPath)")
            }
        } else {
            issues.append("未找到包内 Resources/Core/core.py 或开发目录 Core/core.py。")
        }
        for name in ["ffmpeg", "ffprobe"] where executable(named: name) == nil {
            issues.append("PATH 中未找到 \(name)。")
        }
        return issues
    }

    public var installationHint: String {
        "请运行发行包中的 install.sh 安装运行环境，然后重新打开 Reelpack。运行环境：\(paths.runtimeRoot.path)"
    }

    public func validateForLaunch() throws {
        let issues = launchIssues
        if !issues.isEmpty { throw RuntimeConfigurationError(issues: issues, installationHint: installationHint) }
    }
}

public struct RuntimeConfigurationError: LocalizedError, Sendable {
    public let issues: [String]
    public let installationHint: String
    public var errorDescription: String? {
        (["Reelpack 运行环境未就绪。", installationHint] + issues).joined(separator: "\n")
    }
}
