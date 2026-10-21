import Foundation

public struct OutputArtifact: Identifiable, Equatable, Sendable {
    public enum Kind: String, Sendable {
        case markdown
        case package
        case video
        case pdf

        public var title: String {
            switch self {
            case .markdown: return "Markdown 文稿 · 给人阅读"
            case .package: return "AI 资产包 · 给 Agent"
            case .video: return "原始视频"
            case .pdf: return "PDF 文稿"
            }
        }

        public var symbol: String {
            switch self {
            case .markdown: return "doc.text.fill"
            case .package: return "folder.fill"
            case .video: return "play.rectangle.fill"
            case .pdf: return "doc.richtext.fill"
            }
        }
    }

    public let url: URL
    public let kind: Kind
    public var id: String { url.path }

    public init(url: URL, kind: Kind) {
        self.url = url
        self.kind = kind
    }

    /// Maps existing files and package directories in presentation order.
    public static func fromPipelineOutput(
        _ output: [String: String],
        fileManager: FileManager = .default
    ) -> [OutputArtifact] {
        let ordered: [(String, Kind)] = [
            ("markdown", .markdown),
            ("package", .package),
            ("video", .video),
            ("pdf", .pdf),
        ]
        return ordered.compactMap { key, kind in
            guard let path = output[key], !path.isEmpty else { return nil }
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: path, isDirectory: &isDirectory),
                  isDirectory.boolValue == (kind == .package) else { return nil }
            return OutputArtifact(
                url: URL(fileURLWithPath: path, isDirectory: kind == .package),
                kind: kind
            )
        }
    }
}
