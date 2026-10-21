import AppKit
import SwiftUI
import Shared
import UniformTypeIdentifiers

extension Notification.Name {
    static let uiChooseVideo = Notification.Name("Reelpack.UI.ChooseVideo")
    static let uiChooseOutput = Notification.Name("Reelpack.UI.ChooseOutput")
    static let uiCancel = Notification.Name("Reelpack.UI.Cancel")
    static let uiShowDetails = Notification.Name("Reelpack.UI.ShowDetails")
    static let uiShowSettings = Notification.Name("Reelpack.UI.ShowSettings")
}

struct ContentView: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var health: HealthCheck
    @StateObject private var workflow = UIWorkflowModel()
    @State private var dropHover = false
    @State private var showDetails = false
    @State private var showSettings = false
    @State private var nodeDetails: PathNodePresentation?

    var body: some View {
        PathCanvasView(
            nodes: workflow.nodes,
            stateText: workflow.statusText,
            dropHover: dropHover,
            isBusy: workflow.isBusy,
            isPreparing: isPreparing,
            showResults: workflow.hasResults,
            outputPath: effectiveOutputDirectory,
            logo: BrandArtwork.image,
            showsDetails: !workflow.detailText.isEmpty,
            onChoose: chooseInput,
            onOutput: chooseOutput,
            onCancel: { workflow.cancel() },
            onResults: { workflow.revealResults() },
            onDetails: { nodeDetails = nil; showDetails = true },
            onNodeDetails: { node in nodeDetails = node; showDetails = true }
        )
        .ignoresSafeArea()
        .onDrop(of: [.fileURL], isTargeted: $dropHover, perform: receiveDrop)
        .sheet(isPresented: $showDetails) { detailsSheet }
        .sheet(isPresented: $showSettings) {
            SettingsPanel(isPresented: $showSettings)
                .environmentObject(settings)
                .environmentObject(health)
        }
        .onAppear {
            workflow.onCompletion = { output, source in addHistory(output: output, source: source) }
            health.check(configuration: RuntimeConfiguration.resolve())
            captureLive("idle")
            if let trial = UITrialHarness.input {
                workflow.accept(urls: [trial],
                                outputDirectory: effectiveOutputDirectory,
                                useCloudAPI: settings.preferences.useCloudAPI)
            }
            if let seconds = UITrialHarness.cancelAfter {
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                    workflow.cancel()
                }
            }
        }
        .onChange(of: workflow.mode) {
            captureLive(workflow.mode.rawValue)
            UITrialCapture.render(name: workflow.mode.rawValue, nodes: workflow.nodes,
                                  stateText: workflow.statusText, isBusy: workflow.isBusy,
                                  isPreparing: isPreparing, showResults: workflow.hasResults,
                                  showsDetails: !workflow.detailText.isEmpty,
                                  outputPath: effectiveOutputDirectory)
        }
        .onReceive(NotificationCenter.default.publisher(for: .uiChooseVideo)) { _ in chooseInput() }
        .onReceive(NotificationCenter.default.publisher(for: .uiChooseOutput)) { _ in chooseOutput() }
        .onReceive(NotificationCenter.default.publisher(for: .uiCancel)) { _ in workflow.cancel() }
        .onReceive(NotificationCenter.default.publisher(for: .uiShowDetails)) { _ in nodeDetails = nil; showDetails = true }
        .onReceive(NotificationCenter.default.publisher(for: .uiShowSettings)) { _ in showSettings = true }
        .onExitCommand { if workflow.isBusy { workflow.cancel() } }
        .onDisappear { workflow.windowDidDisappear() }
    }

    private var detailsSheet: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(nodeDetails.map { "\($0.name) · \($0.statusLabel)" } ?? "任务详情")
                .font(.headline)
            ScrollView {
                Text(detailsText.isEmpty ? "尚无处理记录。将本地视频拖入窗口即可开始。" : detailsText)
                    .font(.system(size: 12))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack {
                Button("复制详情") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(detailsText, forType: .string)
                }
                Spacer()
                if workflow.hasResults { Button("打开结果") { workflow.revealResults() } }
                Button("关闭") { showDetails = false }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 560, height: 360)
    }

    /// Preparation is a receipt, never a substitute for a capability stage.
    private var isPreparing: Bool {
        guard workflow.isBusy, !workflow.state.completionReceived else { return false }
        switch workflow.mode {
        case .preparing:
            return true
        case .processing:
            // Unknown setup progress may arrive before the first reliable stage.
            // Once a capability was observed, never restart the receipt feedback.
            return workflow.state.capabilities.values.allSatisfy { $0.lastStage == nil }
        default:
            return false
        }
    }

    /// Trial override never persists; the saved preference stays authoritative.
    private var effectiveOutputDirectory: String {
        UITrialHarness.outputOverride ?? settings.preferences.outputDir
    }

    private var detailsText: String {
        [nodeDetails?.reason, workflow.detailText].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: "\n\n")
    }

    private func chooseInput() {
        workflow.chooseInput(outputDirectory: effectiveOutputDirectory,
                             useCloudAPI: settings.preferences.useCloudAPI)
    }

    private func chooseOutput() {
        let panel = NSOpenPanel()
        panel.title = "选择输出位置"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: (effectiveOutputDirectory as NSString).expandingTildeInPath)
        if panel.runModal() == .OK, let chosen = panel.url {
            if UITrialHarness.outputOverride == nil {
                settings.preferences.outputDir = chosen.path
                settings.save()
            }
        }
    }

    private func receiveDrop(_ providers: [NSItemProvider]) -> Bool {
        // Never accept URLs/webpages or silently take the first of several files.
        guard !providers.isEmpty else { return false }
        guard providers.count == 1 else {
            workflow.accept(urls: [], outputDirectory: effectiveOutputDirectory,
                            useCloudAPI: settings.preferences.useCloudAPI)
            return false
        }
        guard let provider = providers.first, provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) else { return false }
        let output = effectiveOutputDirectory
        let useCloud = settings.preferences.useCloudAPI
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            let url: URL?
            if let data = item as? Data { url = URL(dataRepresentation: data, relativeTo: nil) }
            else if let value = item as? URL { url = value }
            else if let value = item as? NSURL { url = value as URL }
            else { url = nil }
            Task { @MainActor in
                workflow.accept(urls: url.map { [$0] } ?? [], outputDirectory: output, useCloudAPI: useCloud)
            }
        }
        return true
    }

    private func addHistory(output: [String: String], source: URL) {
        var entry = SettingsStore.HistoryEntry(sourceFile: source.path, title: source.lastPathComponent)
        entry.outputVideo = output["video"]
        entry.outputMarkdown = output["markdown"]
        entry.status = "completed"
        settings.history.insert(entry, at: 0)
        settings.save()
    }

    private func captureLive(_ name: String) {
        guard UITrialHarness.captureDirectory != nil else { return }
        let window = NSApp.windows.first { $0.isVisible && $0.contentView?.bounds.width ?? 0 > 1 }
        UITrialCapture.liveWindow(named: name, window: window)
    }
}
