import Foundation
import XCTest
import Shared

final class OutputArtifactTests: XCTestCase {
    private let fileManager = FileManager()
    private var fixtureDirectory: URL!

    override func setUpWithError() throws {
        fixtureDirectory = fileManager.temporaryDirectory.appendingPathComponent(
            "ReelpackOutputArtifactTests-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(at: fixtureDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let fixtureDirectory {
            try fileManager.removeItem(at: fixtureDirectory)
        }
    }

    func testMapsExistingArtifactsInPresentationOrder() throws {
        let markdown = try makeFile("文稿.md")
        let package = try makeDirectory("AI 资产包")
        let video = try makeFile("original.mp4")
        let pdf = try makeFile("文稿.pdf")
        let analysis = try makeFile("analysis.json")

        let artifacts = OutputArtifact.fromPipelineOutput([
            "video": video.path,
            "pdf": pdf.path,
            "analysis": analysis.path,
            "package": package.path,
            "markdown": markdown.path,
            "asset_id": "asset_synthetic",
        ], fileManager: fileManager)

        XCTAssertEqual(artifacts, [
            OutputArtifact(url: markdown, kind: .markdown),
            OutputArtifact(url: package, kind: .package),
            OutputArtifact(url: video, kind: .video),
            OutputArtifact(url: pdf, kind: .pdf),
        ])
        XCTAssertTrue(artifacts[1].url.hasDirectoryPath)
    }

    func testNormalAndCachedCompletionsMapBothGeneratedResults() throws {
        let markdown = try makeFile("analysis.md")
        let package = try makeDirectory("package")
        let paths = ["markdown": markdown.path, "package": package.path]

        for cached in [false, true] {
            var output: [String: Any] = paths
            if cached { output["cached"] = true }
            let data = try JSONSerialization.data(withJSONObject: [
                "type": "complete",
                "output": output,
            ])
            let line = try XCTUnwrap(String(data: data, encoding: .utf8))
            guard case .complete(let parsed)? = IPCMessage.parse(line) else {
                return XCTFail("Expected a complete message for cached=\(cached)")
            }
            XCTAssertEqual(OutputArtifact.fromPipelineOutput(parsed, fileManager: fileManager), [
                OutputArtifact(url: markdown, kind: .markdown),
                OutputArtifact(url: package, kind: .package),
            ])
        }
    }

    func testOmitsMissingAndEmptyPaths() throws {
        let markdown = try makeFile("analysis.md")
        let missing = fixtureDirectory.appendingPathComponent("missing").path

        let artifacts = OutputArtifact.fromPipelineOutput([
            "markdown": markdown.path,
            "package": missing,
            "video": missing,
            "pdf": "",
        ], fileManager: fileManager)

        XCTAssertEqual(artifacts, [OutputArtifact(url: markdown, kind: .markdown)])
    }

    func testAnalysisFileCannotBePresentedAsPackage() throws {
        let markdown = try makeFile("analysis.md")
        let analysis = try makeFile("analysis.json")

        for output in [
            ["markdown": markdown.path, "analysis": analysis.path],
            ["markdown": markdown.path, "analysis": analysis.path, "package": analysis.path],
        ] {
            XCTAssertEqual(OutputArtifact.fromPipelineOutput(output, fileManager: fileManager), [
                OutputArtifact(url: markdown, kind: .markdown),
            ])
        }
    }

    func testDirectoriesAreOnlyAcceptedForPackageArtifacts() throws {
        let directory = try makeDirectory("package")

        let artifacts = OutputArtifact.fromPipelineOutput([
            "markdown": directory.path,
            "package": directory.path,
            "video": directory.path,
            "pdf": directory.path,
        ], fileManager: fileManager)

        XCTAssertEqual(artifacts, [OutputArtifact(url: directory, kind: .package)])
    }

    private func makeFile(_ name: String) throws -> URL {
        let url = fixtureDirectory.appendingPathComponent(name)
        try Data().write(to: url)
        return url
    }

    private func makeDirectory(_ name: String) throws -> URL {
        let url = fixtureDirectory.appendingPathComponent(name, isDirectory: true)
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
