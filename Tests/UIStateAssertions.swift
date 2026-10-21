import Foundation

/// Standalone UI reducer checks; compile with IPCMessage, OutputArtifact and UIState.
@main struct UIStateAssertions {
    @MainActor static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("reelpack-ui-state-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let md = root.appendingPathComponent("analysis.md")
        try "# Saved".write(to: md, atomically: true, encoding: .utf8)
        var assertions = 0
        func expect(_ truth: Bool) { assertions += 1; precondition(truth) }
        func finalState(_ transcription: String, outcome: String, error: String? = nil) throws -> UIWorkflowState {
            var state = UIWorkflowState()
            let id = UUID()
            state.begin(input: root.appendingPathComponent("input.mp4"), outputDirectory: root, generation: id)
            expect(state.capabilities.values.allSatisfy { $0.status == .pending })
            state.apply(event: .progress(stage: "shots", percent: 100, message: "shots"), generation: id)
            expect(state.capabilities[.shots]?.status == .running)
            let caps: [[String: Any]] = PresentationCapability.allCases.map {
                ["name":$0.rawValue, "status": $0 == .transcription ? transcription : "complete",
                 "outcome": $0 == .transcription ? outcome : "detected"]
            }
            let encoded = String(data: try JSONSerialization.data(withJSONObject: ["capabilities": caps]), encoding: .utf8)!
            state.apply(event: .complete(output: ["markdown": md.path,"capability_summary": encoded]), generation: id)
            expect(state.mode == .finalizing)
            state.finish(generation: id, error: error)
            return state
        }
        let silent = try finalState("complete", outcome: "speech_not_detected")
        expect(silent.mode == .complete)
        expect(silent.capabilities[.transcription]?.displayLabel == "未检出语音")
        expect(silent.detailText.contains("未检出语音"))
        let noTrack = try finalState("not_run", outcome: "no_audio_track")
        expect(noTrack.mode == .partial)
        expect(noTrack.capabilities[.transcription]?.displayLabel == "无音轨")
        let failed = try finalState("failed", outcome: "inference_failed", error: "process exit 1")
        expect(failed.mode == .partial && failed.hasResults)
        expect(failed.capabilities[.transcription]?.status == .failed)
        try Data().write(to: md)
        for output in [["markdown": md.path], ["package": root.path]] {
            var empty = UIWorkflowState()
            let id = UUID()
            empty.begin(input: root.appendingPathComponent("input.mp4"), outputDirectory: root, generation: id)
            empty.apply(event: .complete(output: output), generation: id)
            empty.finish(generation: id)
            expect(empty.mode == .error)
            expect(!empty.hasSavedResults)
        }
        print("PASS: \(assertions) UI outcome assertions")
    }
}
