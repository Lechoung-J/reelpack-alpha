import SwiftUI
import Shared

/// One entry point: diagnostics exit before SwiftUI constructs settings or windows.
@main
enum ReelpackEntryPoint {
    static func main() {
        RuntimeDiagnostics.exitIfRequested()
        ReelpackApp.main()
    }
}

struct ReelpackApp: App {
    @StateObject private var settings = SettingsStore()
    @StateObject private var health = HealthCheck()

    var body: some Scene {
        WindowGroup("Reelpack") {
            ContentView()
                .frame(minWidth: 700, minHeight: 560)
                .environmentObject(settings)
                .environmentObject(health)
                .background(PathWindowStyle())
                .onAppear { NSApp.activate(ignoringOtherApps: true) }
        }
        .defaultSize(width: 760, height: 600)
        .windowResizability(.contentMinSize)
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("选择本地视频…") { NotificationCenter.default.post(name: .uiChooseVideo, object: nil) }
                    .keyboardShortcut("o")
                Button("选择输出位置…") { NotificationCenter.default.post(name: .uiChooseOutput, object: nil) }
                    .keyboardShortcut("o", modifiers: [.command, .shift])
            }
            CommandGroup(replacing: .appSettings) {
                Button("设置与运行诊断…") { NotificationCenter.default.post(name: .uiShowSettings, object: nil) }
                    .keyboardShortcut(",")
            }
            CommandMenu("任务") {
                Button("取消当前任务") { NotificationCenter.default.post(name: .uiCancel, object: nil) }
                    .keyboardShortcut(".")
                Button("查看任务详情…") { NotificationCenter.default.post(name: .uiShowDetails, object: nil) }
                    .keyboardShortcut("i", modifiers: [.command, .option])
                Divider()
                Button("复位画布") { NotificationCenter.default.post(name: .pathCanvasResetTransform, object: nil) }
                    .keyboardShortcut("0")
            }
        }
    }
}
