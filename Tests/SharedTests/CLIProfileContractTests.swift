import Foundation
import XCTest

/// The CLI executable is not a SwiftPM test target; this guards its declarative
/// profile allowlist and forwarding. Behavioral validation runs against the
/// built binary in Tests/test_profile_selection.py.
final class CLIProfileContractTests: XCTestCase {
    private var cliSource: String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = root.appendingPathComponent("CLI/main.swift")
        return (try? String(contentsOf: source, encoding: .utf8)) ?? ""
    }

    func testProfileDefaultsToAutoAndIsForwarded() {
        XCTAssertTrue(cliSource.contains("var profile: String = \"auto\""))
        XCTAssertTrue(cliSource.contains("\"--profile\", profile"))
    }

    func testProfileAllowlistMatchesCore() {
        XCTAssertTrue(cliSource.contains("[\"auto\", \"ad\", \"talk\", \"podcast\"].contains(profile)"))
    }

    func testSpeakersOptionRemainsOffByDefault() {
        XCTAssertTrue(cliSource.contains("var speakers: String = \"off\""))
        XCTAssertTrue(cliSource.contains("\"--speakers\", speakers"))
    }
}
