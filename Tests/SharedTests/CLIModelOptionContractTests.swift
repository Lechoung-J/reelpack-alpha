import Foundation
import XCTest

final class CLIModelOptionContractTests: XCTestCase {
    private var cliSource: String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = root.appendingPathComponent("CLI/main.swift")
        return (try? String(contentsOf: source, encoding: .utf8)) ?? ""
    }

    func testModelDefaultsToLarge() {
        XCTAssertTrue(cliSource.contains("var model: String = \"large\""))
        XCTAssertTrue(cliSource.contains("默认 large（large-v3）"))
    }

    func testExplicitSmallIsAllowedAndForwarded() {
        XCTAssertTrue(cliSource.contains("[\"small\", \"large\"].contains(model)"))
        XCTAssertTrue(cliSource.contains("\"--model\", model"))
        XCTAssertTrue(cliSource.contains("allowed: small|large"))
    }
}
