import Foundation
import ArgumentParser
import Shared

@main
struct ReelpackCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "reelpack-cli",
        abstract: "Reelpack：分析本地视频并转写为 MD/PDF",
        version: "0.2.0"
    )

    @Argument(help: "可读取的本地视频文件路径（支持 ~；--runtime-check 时无需输入）")
    var input: String?

    @Option(name: .shortAndLong, help: "输出目录")
    var outputDir: String = "~/Downloads/Reelpack"

    @Option(help: "资产库目录；默认位于输出目录下的 library")
    var libraryRoot: String?

    @Option(help: "转录语言代码，默认 auto 自动识别；例如 ja、en、ko、zh")
    var language: String = "auto"

    @Option(help: "转录模型：small 或 large；默认 large（large-v3）")
    var model: String = "large"

    @Option(help: "OCR 后端：auto、macos、ollama、tesseract、rapidocr")
    var ocrBackend: String = "auto"

    @Option(help: "原片 OCR 最大宽度；0 保留源分辨率")
    var ocrMaxWidth: Int = 1920

    @Option(help: "音乐后端：librosa 或显式 BYO panns；不会自动下载权重")
    var musicBackend: String = "librosa"

    @Option(help: "语义整理：off（默认）；显式short或film开启，原文始终保留")
    var rewrite: String = "off"

    @Option(help: "本地Ollama整理模型，须已准备，不自动下载")
    var rewriteModel: String = "qwen2.5:7b"

    @Option(help: "摘要后端：none、ollama、cloud")
    var summaryBackend: String = "none"

    @Option(help: "转录校正后端：none、ollama、cloud")
    var correctionBackend: String = "none"

    @Option(help: "显式 Ollama 后端的模型名称；不代表默认已经运行")
    var visionModel: String = "internvl3.5:4b"

    @Option(help: "PANNs 曲风候选阈值（0..1）")
    var genreThreshold: Double = 0.05

    @Option(help: "PANNs 曲风候选最多项数（1..10）")
    var genreTopK: Int = 3

    @Flag(help: "跳过模型摘要，交由 Agent 读取证据后回答")
    var skipSummary: Bool = false

    @Flag(help: "相同运行签名也重新分析，覆盖同一资产的分析与文稿")
    var force: Bool = false

    @Flag(help: "输出运行时定位与 Python/import/ffmpeg 检测 JSON，然后退出")
    var runtimeCheck: Bool = false

    @Flag(name: .shortAndLong, help: "使用云端 API 替代本地模型")
    var cloud: Bool = false

    @Flag(name: .shortAndLong, help: "同时输出 PDF")
    var pdf: Bool = false

    mutating func validate() throws {
        if runtimeCheck { return }
        guard ["small", "large"].contains(model) else {
            throw ValidationError("--model must be small or large (allowed: small|large).")
        }
        guard ["short", "film", "off"].contains(rewrite), ["auto", "macos", "ollama", "tesseract", "rapidocr"].contains(ocrBackend),
              ["librosa", "panns"].contains(musicBackend), (0...8192).contains(ocrMaxWidth),
              ["none", "ollama", "cloud"].contains(summaryBackend),
              ["none", "ollama", "cloud"].contains(correctionBackend),
              genreThreshold.isFinite, (0...1).contains(genreThreshold), (1...10).contains(genreTopK) else {
            throw ValidationError("后端、模型、OCR 宽度或曲风阈值参数无效。请查看 --help。")
        }
        let rawPath = (input ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawPath.isEmpty else {
            throw ValidationError("请输入本地视频文件路径；检查安装可使用 --runtime-check。")
        }
        if rawPath.contains("://") || rawPath.lowercased().hasPrefix("www.") {
            throw ValidationError("Reelpack 只处理本地视频文件。请让你的 Agent（或你）先把视频取成本地文件，再传入本地路径。详见 SKILL.md。")
        }
        let path = URL(fileURLWithPath: (rawPath as NSString).expandingTildeInPath).standardizedFileURL.path
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              !isDirectory.boolValue,
              FileManager.default.isReadableFile(atPath: path) else {
            throw ValidationError("找不到可读取的本地视频文件，请检查文件路径。")
        }
        input = path
        outputDir = URL(fileURLWithPath: (outputDir as NSString).expandingTildeInPath, isDirectory: true).standardizedFileURL.path
        if let libraryRoot {
            self.libraryRoot = URL(fileURLWithPath: (libraryRoot as NSString).expandingTildeInPath,
                                   isDirectory: true).standardizedFileURL.path
        } else {
            libraryRoot = URL(fileURLWithPath: outputDir, isDirectory: true).appendingPathComponent("library").path
        }
    }

    mutating func run() async throws {
        if runtimeCheck { RuntimeDiagnostics.exitIfRequested() }
        guard let input, let libraryRoot else { throw ValidationError("请输入本地视频文件路径。") }
        let runtime = RuntimeConfiguration.resolve()
        try runtime.validateForLaunch()
        let runner = ProcessRunner()

        print("Reelpack CLI v0.2.0")
        print("输入: \(input)")
        print("资产库: \(libraryRoot)")
        print("转录: faster-whisper \(model)；OCR: \(ocrBackend)；基础音频: librosa")
        if musicBackend == "panns" { print("可选音乐标签: PANNs（需要 BYO 模型；缺少时会告知成本并保持未运行，不自动下载）") }
        if summaryBackend != "none" || correctionBackend != "none" || cloud {
            print("摘要/校正为显式选项；实际能力状态以完成摘要和 analysis.json 为准。")
        }
        print()

        var outcome = PipelineOutcome()
        var processError: String?
        let cancellation = CLICancellation(runner: runner)
        defer { cancellation.stop() }
        let stream = runner.run(
                configuration: runtime,
                arguments: [
                    "--input", input,
                    "--output-dir", outputDir,
                    "--library-root", libraryRoot,
                    "--ocr-backend", ocrBackend,
                    "--ocr-max-width", String(ocrMaxWidth),
                    "--music-backend", musicBackend,
                    "--model", model,
                    "--language", language,
                    "--vision-model", visionModel,
                    "--rewrite", rewrite,
                    "--rewrite-model", rewriteModel,
                    "--summary-backend", summaryBackend,
                    "--correction-backend", correctionBackend,
                    "--genre-threshold", String(genreThreshold),
                    "--genre-top-k", String(genreTopK),
                    skipSummary ? "--skip-summary" : "",
                    force ? "--force" : "",
                    cloud ? "--cloud" : "",
                    pdf ? "--pdf" : "",
                ].filter { !$0.isEmpty }
            )
        // run() registers a fresh generation and resets its cancellation flag.
        if cancellation.signalNumber != nil { runner.cancel() }
        do {
            for try await msg in stream {
                outcome.consume(msg)
                switch msg {
                case .progress(let stage, let percent, let message):
                    let bar = progressBar(percent: percent)
                    print("\r[\(bar)] \(stage): \(message)", terminator: "")
                    fflush(stdout)
                case .complete:
                    break // Do not announce success until the process exits normally.
                case .error(_, _, let message):
                    print("\n⚠️ 阶段报告: \(message)")
                }
            }
        } catch {
            processError = error.localizedDescription
        }
        // Stream termination is the cancellation cutoff; freeze the result once.
        let cancelSignal = cancellation.finish()
        if let failure = outcome.failureReason(processError: processError, cancelled: cancelSignal != nil) {
            print("\n❌ 处理未完成: \(failure)")
            if let path = outcome.output?["package"] { print("📦 已报告的结果位置（请核对状态）: \(path)") }
            if let number = cancelSignal { throw ExitCode(128 + number) }
            throw ExitCode.failure
        }
        if let output = outcome.output {
            print("\n")
            for line in CapabilitySummary.lines(from: output["capability_summary"]) { print(line) }
            if !outcome.issues.isEmpty { print("⚠️ 已保存结果；处理中有 \(outcome.issues.count) 条阶段提示，请核对能力状态。") }
            if let reuse = output["reuse"] { print("♻️ \(reuse)") }
            if let md = output["markdown"] { print("📄 Markdown: \(md)") }
            if let pdf = output["pdf"] { print("📄 PDF: \(pdf)") }
            if let video = output["video"] { print("🎬 视频: \(video)") }
            if let package = output["package"] { print("📦 资产包: \(package)") }
        }
    }

    private func progressBar(percent: Int) -> String {
        let filled = max(0, min(100, percent)) / 5
        let empty = 20 - filled
        return String(repeating: "█", count: filled) + String(repeating: "░", count: empty)
    }
}

/// Signal handlers only schedule cancellation; ProcessRunner owns child shutdown.
final class CLICancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var received: Int32?
    private var accepting = true
    private var sources: [DispatchSourceSignal] = []
    var signalNumber: Int32? { lock.withLock { received } }

    init(runner: ProcessRunner) {
        for number in [SIGINT, SIGTERM] {
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global(qos: .userInitiated))
            source.setEventHandler { [weak self] in
                guard let self else { return }
                let shouldCancel = self.lock.withLock {
                    guard self.accepting else { return false }
                    if self.received == nil { self.received = number }
                    return true
                }
                if shouldCancel { runner.cancel() }
            }
            sources.append(source)
            source.resume()
        }
    }
    func finish() -> Int32? {
        lock.withLock { accepting = false; return received }
    }
    func stop() {
        for source in sources { source.cancel() }
        sources.removeAll()
        for number in [SIGINT, SIGTERM] { signal(number, SIG_DFL) }
    }
}
