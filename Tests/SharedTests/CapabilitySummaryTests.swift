import Foundation
import XCTest
@testable import Shared

final class CapabilitySummaryTests: XCTestCase {
    func testOptionalFailureAndVisionNotRunAreShownWithoutThrowing() throws {
        let payload: [String: Any] = [
            "complete": 3, "total": 5,
            "capabilities": [
                ["name": "transcription", "status": "complete"],
                ["name": "shots", "status": "complete"],
                ["name": "ocr", "status": "complete"],
                ["name": "music", "status": "failed", "reason": "librosa JIT cache",
                 "impact": "beats unavailable", "remedy": "reelpack enrich asset --only audio"],
                ["name": "vision", "status": "not_run", "reason": "read evidence/*.jpg",
                 "impact": "no semantic captions"],
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        let lines = CapabilitySummary.lines(from: String(decoding: data, as: UTF8.self))
        XCTAssertTrue(lines[0].contains("3/5"))
        XCTAssertTrue(lines.contains { $0.contains("失败: music") && $0.contains("JIT cache") })
        XCTAssertTrue(lines.contains { $0.contains("未运行: vision") && $0.contains("evidence/*.jpg") })
        XCTAssertTrue(lines.contains { $0.contains("下一步:") })
        XCTAssertFalse(lines.joined().contains("internvl"))
    }

    func testSuccessfulEmptySpeechRemainsCompleteAndExplainsContent() throws {
        let payload: [String: Any] = ["complete": 4, "total": 5, "capabilities": [
            ["name": "transcription", "status": "complete", "outcome": "speech_not_detected"]]]
        let data = try JSONSerialization.data(withJSONObject: payload)
        let text = CapabilitySummary.lines(from: String(decoding: data, as: UTF8.self)).joined(separator: "\n")
        XCTAssertTrue(text.contains("未检出语音"))
        XCTAssertTrue(text.contains("不代表没有声音"))
        XCTAssertFalse(text.contains("失败: transcription"))
    }

    func testUnmeasuredBoardAndMissingTaggingDoNotBecomeZeroPercent() throws {
        let payload: [String: Any] = ["complete": 4, "total": 5, "capabilities": [],
            "audio_board": ["status": "not_run", "reason": "beats not measured"],
            "tagging": ["status": "not_run", "reason": "BYO weights missing; no download"]]
        let data = try JSONSerialization.data(withJSONObject: payload)
        let text = CapabilitySummary.lines(from: String(decoding: data, as: UTF8.self)).joined(separator: "\n")
        XCTAssertTrue(text.contains("audio_board"))
        XCTAssertTrue(text.contains("BYO"))
        XCTAssertFalse(text.contains("0%"))
    }
}
