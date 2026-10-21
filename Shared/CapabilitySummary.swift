import Foundation

/// Presentation only; partial/optional failures never change subprocess exit semantics.
public enum CapabilitySummary {
    public static func lines(from encoded: String?) -> [String] {
        guard let encoded, let data = encoded.data(using: .utf8),
              let report = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = report["capabilities"] as? [[String: Any]] else {
            return ["✅ 资产包已保存；请查看 analysis.json 中各能力的状态。"]
        }
        let complete = report["complete"] as? Int ?? 0
        let total = report["total"] as? Int ?? items.count
        var result = ["✅ 完成（\(complete)/\(total) 能力）；可选缺项不影响已有产物"]
        for item in items {
            let name = item["name"] as? String ?? "unknown"
            let status = item["status"] as? String ?? "unknown"
            if status == "complete" {
                if name == "ocr" { result.append("ℹ️ ocr: complete (backend: \(item["backend"] as? String ?? "unknown"))") }
                if let fallback = item["fallback_from"] as? String {
                    result.append("ℹ️ \(name): \(item["backend"] as? String ?? "unknown")（从 \(fallback) 回退；\(item["reason"] as? String ?? "后端错误")）")
                }
                if item["outcome"] as? String == "speech_not_detected" {
                    result.append("ℹ️ transcription: 已完成识别，未检出语音；不代表没有声音。")
                }
                continue
            }
            let label = status == "failed" ? "失败" : status == "not_run" ? "未运行" : "未完成"
            result.append("⚠️ \(label): \(name) → \(item["reason"] as? String ?? status) → 见 analysis.json")
            if let impact = item["impact"] as? String { result.append("   影响: \(impact)") }
            if let remedy = item["remedy"] as? String { result.append("   下一步: \(remedy)") }
        }
        for field in ["audio_board", "tagging", "rewrite"] {
            if let state = report[field] as? [String: Any],
               let status = state["status"] as? String, status != "complete" {
                result.append("⚠️ \(field): \(status) → \(state["reason"] as? String ?? "此可选结果不可用")")
            }
        }
        return result
    }
}
