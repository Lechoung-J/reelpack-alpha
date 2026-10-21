import Foundation
import Combine
import Darwin

/// Persists history and preferences in Reelpack's Application Support directory.
/// Writes use POSIX flock for cross-process safety (App + CLI concurrent access).
public class SettingsStore: ObservableObject {
    public static let whisperModel = "large"
    public static let visionModel = "internvl3.5:4b"

    @Published public var history: [HistoryEntry] = []
    @Published public var preferences = Preferences()

    private let fileURL: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Reelpack")
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base.appendingPathComponent("settings.json")
    }()

    public init() { load() }

    public struct HistoryEntry: Codable, Identifiable {
        public var id: String = UUID().uuidString
        public var sourceFile: String?
        public var title: String
        public var createdAt: Date = Date()
        public var status: String = "completed"
        public var outputVideo: String?
        public var outputMarkdown: String?

        public init(
            sourceFile: String? = nil,
            title: String,
            createdAt: Date = Date(),
            status: String = "completed",
            outputVideo: String? = nil,
            outputMarkdown: String? = nil
        ) {
            self.sourceFile = sourceFile
            self.title = title
            self.createdAt = createdAt
            self.status = status
            self.outputVideo = outputVideo
            self.outputMarkdown = outputMarkdown
        }
    }

    public struct Preferences: Codable {
        public var whisperModel: String = SettingsStore.whisperModel
        public var visionModel: String = SettingsStore.visionModel
        public var useCloudAPI: Bool = false
        public var outputDir: String = "~/Downloads/Reelpack"

        public init(
            whisperModel: String = SettingsStore.whisperModel,
            visionModel: String = SettingsStore.visionModel,
            useCloudAPI: Bool = false,
            outputDir: String = "~/Downloads/Reelpack"
        ) {
            self.whisperModel = whisperModel
            self.visionModel = visionModel
            self.useCloudAPI = useCloudAPI
            self.outputDir = outputDir
        }

        private enum CodingKeys: String, CodingKey {
            case whisperModel, visionModel, useCloudAPI, outputDir
        }

        // Older settings may omit fields; unrelated legacy keys are ignored.
        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            whisperModel = try container.decodeIfPresent(String.self, forKey: .whisperModel) ?? SettingsStore.whisperModel
            visionModel = try container.decodeIfPresent(String.self, forKey: .visionModel) ?? SettingsStore.visionModel
            useCloudAPI = try container.decodeIfPresent(Bool.self, forKey: .useCloudAPI) ?? false
            outputDir = try container.decodeIfPresent(String.self, forKey: .outputDir) ?? "~/Downloads/Reelpack"
        }
    }

    public func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let json = try? JSONDecoder().decode(SettingsJSON.self, from: data)
        else { return }
        history = json.history ?? []
        preferences = json.preferences ?? Preferences()
        preferences.whisperModel = Self.whisperModel
        preferences.visionModel = Self.visionModel
    }

    public func save() {
        let wrapper = SettingsJSON(history: history, preferences: preferences)
        guard let data = try? JSONEncoder().encode(wrapper) else { return }

        // Keep personal history and preferences owner-only, including existing files.
        let fd = open(fileURL.path, O_WRONLY | O_CREAT, 0o600)
        guard fd >= 0 else { return }
        defer { close(fd) }
        fchmod(fd, 0o600)
        flock(fd, LOCK_EX)
        ftruncate(fd, 0)
        write(fd, (data as NSData).bytes, data.count)
        flock(fd, LOCK_UN)
    }

    private struct SettingsJSON: Codable {
        var history: [HistoryEntry]?
        var preferences: Preferences?
    }
}
