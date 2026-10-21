import XCTest
import Darwin
@testable import Shared

final class RuntimePathsTests: XCTestCase {
    private var root: URL!
    private var home: URL!
    private var temporary: URL!
    private var runtime: URL!
    private let fixtureName = "reelpack-runtime-path-tests-" + UUID().uuidString

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath()
            .appendingPathComponent(fixtureName, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        home = root.appendingPathComponent("home", isDirectory: true)
        temporary = root.appendingPathComponent("tmp", isDirectory: true)
        runtime = root.appendingPathComponent("runtime", isDirectory: true)
        for path in [home!, temporary!, runtime!] {
            try FileManager.default.createDirectory(at: path, withIntermediateDirectories: false)
        }
    }

    override func tearDownWithError() throws {
        guard let root else { return }
        let expected = FileManager.default.temporaryDirectory.resolvingSymlinksInPath()
            .appendingPathComponent(fixtureName, isDirectory: true)
        // Verify the known absolute fixture path before deleting anything.
        guard root.standardizedFileURL.path == expected.path,
              root.resolvingSymlinksInPath().path == expected.path else {
            return XCTFail("Unexpected test cleanup path")
        }
        try FileManager.default.removeItem(at: root)
    }

    private var environment: [String: String] {
        ["HOME": home.path, "TMPDIR": temporary.path, "REELPACK_RUNTIME_ROOT": runtime.path,
         "PATH": "/usr/bin:/bin", "PYTHONHOME": "shell configuration", "PYTHONPATH": "shell packages",
         "PRIVATE_TEST_TOKEN": "must never appear in diagnostic JSON"]
    }

    private func resolve(_ overrides: [String: String] = [:]) -> RuntimeConfiguration {
        let inherited = environment.merging(overrides) { _, new in new }
        return RuntimeConfiguration.resolve(environment: inherited,
            executableURL: root.appendingPathComponent("bin/reelpack"), currentDirectory: root, homeDirectory: home)
    }

    private func snapshot() throws -> [String: String] {
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(at: root,
            includingPropertiesForKeys: nil, options: []))
        var records: [String: String] = [:]
        for case let path as URL in enumerator {
            let attributes = try FileManager.default.attributesOfItem(atPath: path.path)
            let relative = String(path.path.dropFirst(root.path.count + 1))
            let modification = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
            if attributes[.type] as? FileAttributeType == .typeSymbolicLink {
                records[relative] = "link:" + (try FileManager.default.destinationOfSymbolicLink(atPath: path.path))
            } else {
                records[relative] = "\(attributes[.size] ?? 0):\(attributes[.posixPermissions] ?? 0):\(modification)"
            }
        }
        return records
    }

    func testResolutionCopiesEnvironmentAndSetsCacheChildrenWithoutCreatingAnything() throws {
        let inherited = environment
        let before = try snapshot()
        let configuration = resolve()
        let cache = home.appendingPathComponent("Library/Caches/Reelpack")
        XCTAssertEqual(configuration.environment["REELPACK_CACHE_ROOT"], cache.path)
        XCTAssertEqual(configuration.environment["NUMBA_CACHE_DIR"], cache.appendingPathComponent("numba").path)
        XCTAssertEqual(configuration.environment["MPLCONFIGDIR"], cache.appendingPathComponent("mpl").path)
        XCTAssertEqual(configuration.environment["PYTHONPYCACHEPREFIX"], cache.appendingPathComponent("pycache").path)
        XCTAssertEqual(configuration.environment["PYTHONDONTWRITEBYTECODE"], "1")
        XCTAssertEqual(configuration.environment["HF_HOME"], home.appendingPathComponent(".cache/huggingface").path)
        XCTAssertEqual(configuration.environment["TORCH_HOME"], home.appendingPathComponent(".cache/torch").path)
        XCTAssertNil(configuration.environment["PYTHONHOME"])
        XCTAssertNil(configuration.environment["PYTHONPATH"])
        XCTAssertEqual(inherited, environment)
        XCTAssertEqual(try snapshot(), before)
    }

    func testExplicitModelLocationsAndOutputRootsRemainIndependentFromAccelerationCache() throws {
        let cache = root.appendingPathComponent("custom cache")
        let model = root.appendingPathComponent("persistent models")
        let torch = root.appendingPathComponent("persistent torch")
        let output = root.appendingPathComponent("user outputs")
        let configuration = resolve(["REELPACK_CACHE_ROOT": cache.path, "HF_HOME": model.path,
                                     "TORCH_HOME": torch.path, "REELPACK_OUTPUT_ROOT": output.path])
        XCTAssertEqual(configuration.environment["REELPACK_CACHE_ROOT"], cache.path)
        XCTAssertEqual(configuration.environment["HF_HOME"], model.path)
        XCTAssertEqual(configuration.environment["TORCH_HOME"], torch.path)
        XCTAssertEqual(configuration.environment["REELPACK_OUTPUT_ROOT"], output.path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: cache.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: model.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
    }

    func testReadOnlyCandidateUsesStableTmpUidFallbackWithoutProbing() throws {
        let candidate = root.appendingPathComponent("read only")
        try FileManager.default.createDirectory(at: candidate, withIntermediateDirectories: false)
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: candidate.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: candidate.path) }
        let before = try snapshot()
        let first = resolve(["REELPACK_CACHE_ROOT": candidate.path])
        let second = resolve(["REELPACK_CACHE_ROOT": candidate.path])
        let fallback = temporary.appendingPathComponent("reelpack-cache-\(getuid())")
        XCTAssertEqual(first.environment["REELPACK_CACHE_ROOT"], fallback.path)
        XCTAssertEqual(second.environment["REELPACK_CACHE_ROOT"], fallback.path)
        XCTAssertEqual(first.environment["MPLCONFIGDIR"], fallback.appendingPathComponent("mpl").path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fallback.path))
        XCTAssertEqual(try snapshot(), before)
    }

    func testReadOnlyChildSelectsFallbackForWholeCacheTree() throws {
        let candidate = root.appendingPathComponent("candidate")
        let child = candidate.appendingPathComponent("numba")
        try FileManager.default.createDirectory(at: child, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: child.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: child.path) }
        let configuration = resolve(["REELPACK_CACHE_ROOT": candidate.path])
        XCTAssertEqual(configuration.environment["REELPACK_CACHE_ROOT"], temporary.appendingPathComponent("reelpack-cache-\(getuid())").path)
        XCTAssertEqual(configuration.environment["HF_HOME"], home.appendingPathComponent(".cache/huggingface").path)
    }

    func testPersistentModelDefaultsHonorXdgInsteadOfTemporaryCache() {
        let persistent = root.appendingPathComponent("xdg")
        let configuration = resolve(["XDG_CACHE_HOME": persistent.path])
        XCTAssertEqual(configuration.environment["HF_HOME"], persistent.appendingPathComponent("huggingface").path)
        XCTAssertEqual(configuration.environment["TORCH_HOME"], persistent.appendingPathComponent("torch").path)
        XCTAssertNotEqual(configuration.environment["HF_HOME"], configuration.environment["REELPACK_CACHE_ROOT"])
    }

    func testDiagnosticEnvironmentIncludesOnlyRuntimeContract() {
        let configuration = resolve()
        XCTAssertNotNil(configuration.diagnosticEnvironment["REELPACK_CACHE_ROOT"])
        XCTAssertNotNil(configuration.diagnosticEnvironment["NUMBA_CACHE_DIR"])
        XCTAssertNotNil(configuration.diagnosticEnvironment["HF_HOME"])
        XCTAssertNil(configuration.diagnosticEnvironment["PRIVATE_TEST_TOKEN"])
        XCTAssertNil(configuration.diagnosticEnvironment["HOME"])
    }

    func testRuntimeDiagnosticReportsRootSizesAndModelsWithoutExecutingCoreOrWritingCaches() throws {
        let search = (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map(String.init)
            + ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]
        let python = try XCTUnwrap(search.lazy.map { URL(fileURLWithPath: $0).appendingPathComponent("python3") }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }, "Python is unavailable")
        let core = root.appendingPathComponent("Core", isDirectory: true)
        try FileManager.default.createDirectory(at: core, withIntermediateDirectories: false)
        // If a diagnostic imports Core, this raises and fails its readiness check.
        try Data("raise RuntimeError('Core must not execute in a read-only diagnostic')\n".utf8)
            .write(to: core.appendingPathComponent("core.py"))
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Core/runtime_environment.py")
        try FileManager.default.copyItem(at: source, to: core.appendingPathComponent("runtime_environment.py"))
        for name in [".venv", ".venv-audio", ".venv-transnet", ".venv-ocr", ".venv-music"] {
            let bin = runtime.appendingPathComponent(name + "/bin")
            try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python"), withDestinationURL: python)
        }
        let model = root.appendingPathComponent("read only HF")
        let modelSnapshot = model.appendingPathComponent("hub/models--Systran--faster-whisper-large-v3/snapshots/preset")
        try FileManager.default.createDirectory(at: modelSnapshot, withIntermediateDirectories: true)
        for name in ["model.bin", "config.json", "tokenizer.json", "vocabulary.json"] {
            try Data("fixture".utf8).write(to: modelSnapshot.appendingPathComponent(name))
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: model.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: model.path) }
        let outputs = root.appendingPathComponent("chosen outputs")
        let library = root.appendingPathComponent("chosen library")
        let labels = root.appendingPathComponent("provided labels.csv")
        let csv = "index,mid,display_name\n" + (0..<527).map { "\($0),/m/\($0),Label \($0)\n" }.joined()
        try Data(csv.utf8).write(to: labels)
        let configuration = resolve(["HF_HOME": model.path, "REELPACK_OUTPUT_ROOT": outputs.path,
                                     "REELPACK_LIBRARY_ROOT": library.path, "REELPACK_PANNS_LABELS": labels.path])
        let before = try snapshot()
        let report = RuntimeDiagnostics.check(configuration: configuration)
        XCTAssertNil(report.rootInspectionDetail)
        XCTAssertEqual(report.python.coreImport?.available, true)
        XCTAssertTrue(report.python.coreImport?.detail?.contains("without executing Core") == true)
        XCTAssertEqual(report.roots["outputs"]?.path, outputs.path)
        XCTAssertEqual(report.roots["library"]?.path, library.path)
        XCTAssertEqual(report.roots["cache"]?.exists, false)
        XCTAssertEqual(report.roots["models"]?.modelsReady, true)
        XCTAssertEqual(report.roots["models"]?.sizeBytes, 28)
        XCTAssertEqual(report.roots["models"]?.writable, false)
        XCTAssertEqual(report.models["whisper"]?.ready, true)
        XCTAssertEqual(report.models["panns"]?.ready, false)
        XCTAssertEqual(report.models["panns"]?.labels?.path, labels.path)
        XCTAssertEqual(report.models["panns"]?.labels?.rows, 527)
        XCTAssertEqual(report.models["panns"]?.labelsReady, true)
        XCTAssertEqual(report.models["panns"]?.automaticDownload, false)
        XCTAssertEqual(report.models["panns"]?.requiredDownloadBytes, 327428481)
        XCTAssertEqual(report.models["transnet"]?.ready, false)
        XCTAssertEqual(report.jit.status, "not_run")
        XCTAssertEqual(try snapshot(), before)
        let json = String(decoding: try JSONEncoder().encode(report), as: UTF8.self)
        XCTAssertFalse(json.contains("must never appear in diagnostic JSON"))
    }
}
