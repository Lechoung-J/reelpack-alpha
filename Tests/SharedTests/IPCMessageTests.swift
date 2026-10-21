import XCTest
@testable import Shared

final class IPCMessageTests: XCTestCase {
    func testCompleteWithCachedBooleanPreservesArtifactPaths() {
        for cached in [true, false] {
            let line = #"{"protocol_version":1,"type":"complete","output":{"asset_id":"asset_synthetic","markdown":"/synthetic/analysis.md","analysis":"/synthetic/analysis.json","package":"/synthetic/package","video":"/synthetic/original.mp4","cached":\#(cached)}}"#
            guard case .complete(let output)? = IPCMessage.parse(line) else {
                return XCTFail("Complete output with cached=\(cached) must parse")
            }
            XCTAssertEqual(output, [
                "asset_id": "asset_synthetic",
                "markdown": "/synthetic/analysis.md",
                "analysis": "/synthetic/analysis.json",
                "package": "/synthetic/package",
                "video": "/synthetic/original.mp4",
            ])
            XCTAssertNil(output["cached"])
        }
    }

    func testCompleteIgnoresNonStringMetadataWithoutDroppingStringArtifacts() {
        let line = #"{"protocol_version":1,"type":"complete","output":{"markdown":"/synthetic/analysis.md","cached":true,"count":2,"optional":null,"details":{"source":"local_file"},"items":["x"]}}"#
        guard case .complete(let output)? = IPCMessage.parse(line) else {
            return XCTFail("Mixed artifact and metadata output must parse")
        }
        XCTAssertEqual(output, ["markdown": "/synthetic/analysis.md"])
    }

    func testCompleteRejectsNonObjectOutput() {
        for value in ["null", "true", "[]", "\"artifact\""] {
            let line = #"{"protocol_version":1,"type":"complete","output":\#(value)}"#
            XCTAssertNil(IPCMessage.parse(line))
        }
    }

    func testProcessRunnerDeliversCompleteWithCachedBoolean() async throws {
        let line = #"{"protocol_version":1,"type":"complete","output":{"markdown":"/synthetic/analysis.md","cached":true}}"#
        let runner = ProcessRunner()
        var received: [IPCMessage] = []
        for try await message in runner.run(
            pythonPath: "/bin/sh",
            scriptPath: "-c",
            arguments: ["printf '%s\\n' '\(line)'"]
        ) {
            received.append(message)
        }
        XCTAssertEqual(received.count, 1)
        guard let message = received.first, case .complete(let output) = message else {
            return XCTFail("Expected a complete message")
        }
        XCTAssertEqual(output, ["markdown": "/synthetic/analysis.md"])
    }
}
