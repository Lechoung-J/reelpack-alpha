import Foundation
import Combine
import Shared

/// Uses exactly the interpreter/environment that will run the media pipeline.
class HealthCheck: ObservableObject {
    @Published var items: [HealthItem] = []
    @Published var installationHint: String?
    @Published var runtimeRoot = ""
    private var isChecking = false

    struct HealthItem: Identifiable {
        var id: String { name }
        var name: String
        var available: Bool
        var version: String?
        var detail: String?
        var required = true
    }

    func check(configuration: RuntimeConfiguration = .resolve()) {
        guard !isChecking else { return }
        isChecking = true
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let report = RuntimeDiagnostics.check(configuration: configuration)
            let pythonDetail = report.python.available
                ? "\(report.python.executable ?? report.python.requestedPath)\n主要 imports 与 Core 已就绪"
                : report.python.detail
            var results = [HealthItem(name: "Python / Core", available: report.python.available,
                                      version: report.python.version, detail: pythonDetail)]
            for (key, title) in [("audio", "音频分析"), ("transnet", "TransNetV2"), ("ocr", "RapidOCR"), ("music", "音乐语义（可选）")] {
                if let python = report.helpers[key] {
                    results.append(HealthItem(name: title, available: python.available,
                                              version: python.available ? (python.required ? "imports 已就绪" : "模块已安装 · 权重自备") : nil,
                                              detail: python.detail, required: python.required))
                }
            }
            results += report.tools.map {
                HealthItem(name: $0.name, available: $0.available,
                           version: $0.available ? $0.detail : nil, detail: $0.detail, required: $0.required)
            }
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.runtimeRoot = report.runtimeRoot
                self.installationHint = report.installationHint
                self.items = results
                self.isChecking = false
            }
        }
    }
}
