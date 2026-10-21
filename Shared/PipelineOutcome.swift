import Foundation

/// Final process outcome, independent from intermediate IPC error notifications.
/// Source media alone is not a saved result. A failed run can still retain checkpoints.
public struct PipelineOutcome {
    public private(set) var output: [String: String]?
    public private(set) var issues: [String] = []
    public init() {}

    public mutating func consume(_ message: IPCMessage) {
        switch message {
        case .complete(let value): output = value
        case .error(let stage, let code, let message): issues.append("[\(stage)] \(code)：\(message)")
        case .progress: break
        }
    }

    public var validSavedResult: Bool {
        guard let output else { return false }
        let fm = FileManager.default
        if let path = output["package"] {
            let analysis = URL(fileURLWithPath: path).appendingPathComponent("analysis.json")
            if let data = try? Data(contentsOf: analysis),
               let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let id = object["asset_id"] as? String, !id.isEmpty,
               ["complete", "partial"].contains(object["status"] as? String ?? "") { return true }
        }
        for key in ["markdown", "pdf"] {
            guard let path = output[key] else { continue }
            var directory: ObjCBool = false
            if fm.fileExists(atPath: path, isDirectory: &directory), !directory.boolValue,
               fm.isReadableFile(atPath: path),
               let attributes = try? fm.attributesOfItem(atPath: path),
               let size = attributes[.size] as? NSNumber, size.int64Value > 0 { return true }
        }
        return false
    }

    /// Evaluate only after ProcessRunner finishes, preserving nonzero process exits.
    public func failureReason(processError: String? = nil, cancelled: Bool = false) -> String? {
        if cancelled { return "已取消；已保存的检查点保留，未宣称完成。" }
        if let processError { return processError }
        guard let output else { return "进程结束但未收到完成消息；已保存的检查点可供检查。" }
        if let execution = output["execution_status"], execution != "complete" {
            return "完成消息的执行状态不是 complete：\(execution)；保留已保存的部分结果。"
        }
        guard validSavedResult else { return "完成消息没有可读取的有效保存结果。" }
        return nil
    }
}
