import Foundation

/// NDJSON message types exchanged between Swift and Python via subprocess stdin/stdout.
public enum IPCMessage {
    case progress(stage: String, percent: Int, message: String)
    case complete(output: [String: String])
    case error(stage: String, code: String, message: String)

    /// Parse a single JSON line from Python stdout.
    public static func parse(_ line: String) -> IPCMessage? {
        guard let data = line.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String
        else { return nil }

        switch type {
        case "progress":
            guard let stage = json["stage"] as? String,
                  let percent = json["percent"] as? Int,
                  let message = json["message"] as? String
            else { return nil }
            return .progress(stage: stage, percent: percent, message: message)
        case "complete":
            guard let output = json["output"] as? [String: Any]
            else { return nil }
            // Core also sends metadata such as cached: true alongside string artifact paths.
            return .complete(output: output.compactMapValues { $0 as? String })
        case "error":
            guard let stage = json["stage"] as? String,
                  let code = json["code"] as? String,
                  let message = json["message"] as? String
            else { return nil }
            return .error(stage: stage, code: code, message: message)
        default:
            return nil
        }
    }
}
