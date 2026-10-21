import AppKit
import SwiftUI
import Shared

struct SettingsPanel: View {
    @Binding var isPresented: Bool
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var health: HealthCheck

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Reelpack 设置")
                    .font(.system(size: 18, weight: .semibold))
                Spacer()
                Button("完成") { isPresented = false }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 20)
            .frame(height: 54)
            .background(.bar)
            .overlay(alignment: .bottom) { Divider() }

            Form {
                Section("输出") {
                    LabeledContent("保存位置") {
                        HStack(spacing: 8) {
                            Text(shortPath(settings.preferences.outputDir))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.head)
                            Button("更改…", action: chooseOutputDir)
                        }
                    }
                    Text("本地视频的 Markdown 文稿保存在上方目录。")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Section("处理模型") {
                    LabeledContent("Whisper") {
                        Text("Large v3 · 最高精度")
                            .foregroundStyle(.secondary)
                    }
                    LabeledContent("视觉模型") {
                        Text("InternVL 3.5 4B")
                            .foregroundStyle(.secondary)
                    }
                    Toggle("使用云端 API", isOn: $settings.preferences.useCloudAPI)
                }

                Section("运行环境") {
                    if health.items.isEmpty {
                        HStack {
                            ProgressView().controlSize(.small)
                            Text("正在检查…").foregroundStyle(.secondary)
                        }
                    } else {
                        ForEach(health.items) { item in
                            HStack {
                                Label {
                                    Text(item.name)
                                } icon: {
                                    Image(systemName: item.available ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                        .foregroundStyle(item.available ? .green : .yellow)
                                }
                                Spacer()
                                Text(item.available ? item.version ?? "已就绪" : (item.required ? "需要安装" : "可选 · 未就绪"))
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            .help(item.detail ?? item.name)
                        }
                    }
                    if !health.runtimeRoot.isEmpty {
                        Text("运行环境：\(shortPath(health.runtimeRoot))")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    if let hint = health.installationHint {
                        Text(hint)
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    Button("重新检查") { health.check() }
                }

                if !settings.history.isEmpty {
                    Section("最近处理") {
                        ForEach(settings.history.prefix(5)) { entry in
                            HStack {
                                Image(systemName: "doc.text")
                                    .foregroundStyle(.secondary)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(entry.title).lineLimit(1)
                                    Text(entry.createdAt, style: .relative)
                                        .font(.system(size: 10))
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if let path = entry.outputMarkdown {
                                    Button {
                                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                                    } label: {
                                        Image(systemName: "arrow.right.circle")
                                    }
                                    .buttonStyle(.borderless)
                                }
                            }
                        }
                    }
                }
            }
            .formStyle(.grouped)
        }
        .frame(width: 480, height: 520)
        .onDisappear { settings.save() }
    }

    private func chooseOutputDir() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.directoryURL = URL(fileURLWithPath: (settings.preferences.outputDir as NSString).expandingTildeInPath)
        panel.prompt = "选择保存位置"
        if panel.runModal() == .OK, let url = panel.url {
            settings.preferences.outputDir = url.path
            settings.save()
        }
    }

    private func shortPath(_ path: String) -> String {
        let expanded = (path as NSString).expandingTildeInPath
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return expanded.replacingOccurrences(of: home, with: "~")
    }
}
