import XCTest
@testable import Shared

final class PipelineOutcomeTests: XCTestCase {
    private func fixture(_ body: (URL) throws -> Void) throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("reelpack-outcome-\(UUID())")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try body(folder)
    }
    func testRecoverableErrorThenSavedCompletionNeedsNormalExit() throws {
        try fixture { dir in
            let file = dir.appendingPathComponent("analysis.md")
            try "# Saved result".write(to: file, atomically: true, encoding: .utf8)
            var state = PipelineOutcome()
            state.consume(.error(stage: "audio", code: "OPTIONAL", message: "unavailable"))
            XCTAssertNotNil(state.failureReason())
            state.consume(.complete(output: ["markdown": file.path]))
            XCTAssertNil(state.failureReason())
            XCTAssertEqual(state.issues.count, 1)
            XCTAssertNotNil(state.failureReason(processError: "exit 7"))
            XCTAssertNotNil(state.failureReason(cancelled: true))
        }
    }
    func testFailedExecutionCannotTurnIntoSuccessBecauseMarkdownExists() throws {
        try fixture { dir in
            let file = dir.appendingPathComponent("analysis.md")
            try "# Partial checkpoint".write(to: file, atomically: true, encoding: .utf8)
            for execution in ["in_progress", "failed", "cancelled"] {
                var state = PipelineOutcome()
                state.consume(.complete(output: ["markdown": file.path, "execution_status": execution]))
                XCTAssertNotNil(state.failureReason())
            }
        }
    }
    func testMissingEmptyOrInputOnlyIsNotSavedCompletion() throws {
        try fixture { dir in
            let file = dir.appendingPathComponent("empty.md")
            try Data().write(to: file)
            for output in [["video": file.path], ["markdown": file.path], ["package": dir.path], ["markdown": dir.appendingPathComponent("absent").path]] {
                var state = PipelineOutcome(); state.consume(.complete(output: output))
                XCTAssertNotNil(state.failureReason())
            }
        }
    }
    func testPartialPackageCanBeSuccessfulButCorruptOrFailedPackageCannot() throws {
        try fixture { dir in
            let file = dir.appendingPathComponent("analysis.json")
            for status in ["partial", "complete", "failed", "unknown"] {
                let data = try JSONSerialization.data(withJSONObject: ["asset_id": "asset_test", "status": status])
                try data.write(to: file)
                var state = PipelineOutcome(); state.consume(.complete(output: ["package": dir.path]))
                XCTAssertEqual(state.failureReason() == nil, ["partial", "complete"].contains(status))
            }
        }
    }
}
