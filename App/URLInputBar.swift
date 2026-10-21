import AppKit
import SwiftUI

struct LocalFileInputBar: View {
    @Binding var filePathText: String
    let isEnabled: Bool
    let hasSelectedFile: Bool
    let outputDirectory: String
    let buttonTitle: String
    let onChooseOutputDirectory: () -> Void
    let onProcess: () -> Void

    @State private var editCommand: FilePathTextFieldCommand?
    @State private var editCommandID = 0

    private var canProcess: Bool {
        isEnabled && (hasSelectedFile || !filePathText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    var body: some View {
        VStack(spacing: 9) {
            HStack(spacing: 7) {
                Image(systemName: "folder")
                    .foregroundStyle(.secondary)
                Text("保存到")
                    .foregroundStyle(.secondary)
                Button(shortPath(outputDirectory), action: onChooseOutputDirectory)
                    .buttonStyle(.link)
                    .lineLimit(1)
                    .truncationMode(.head)
                Spacer()
                Text("完成后可将文件直接拖到 Finder")
                    .foregroundStyle(.tertiary)
            }
            .font(.system(size: 11))

            HStack(spacing: 10) {
                HStack(spacing: 7) {
                    Image(systemName: "doc")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)

                    NativeFilePathTextField(
                        text: $filePathText,
                        placeholder: hasSelectedFile ? "已选择本地视频" : "输入或粘贴本地视频文件路径…",
                        isEnabled: isEnabled && !hasSelectedFile,
                        command: editCommand,
                        commandID: editCommandID
                    )
                    .frame(height: 26)

                    if !filePathText.isEmpty && !hasSelectedFile && isEnabled {
                        Button {
                            sendEditCommand(.clear)
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.tertiary)
                        }
                        .buttonStyle(.plain)
                        .help("清除路径")
                    } else if !hasSelectedFile && isEnabled {
                        Button(action: pastePath) {
                            Image(systemName: "doc.on.clipboard")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help("粘贴路径（⌘V）")
                    }
                }
                .padding(.horizontal, 10)
                .frame(height: 34)
                .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 7))
                .overlay {
                    RoundedRectangle(cornerRadius: 7)
                        .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1)
                }

                Button(buttonTitle, action: onProcess)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(!canProcess)
                    .keyboardShortcut(.return, modifiers: [])
            }
        }
        .padding(.horizontal, 24)
        .padding(.top, 11)
        .padding(.bottom, 16)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }

    private func pastePath() {
        sendEditCommand(.paste)
    }

    private func sendEditCommand(_ command: FilePathTextFieldCommand) {
        editCommand = command
        editCommandID += 1
    }

    private func shortPath(_ path: String) -> String {
        let expanded = (path as NSString).expandingTildeInPath
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return expanded.replacingOccurrences(of: home, with: "~")
    }
}

private enum FilePathTextFieldCommand {
    case paste
    case clear
}

/// AppKit-backed field keeps standard macOS edit commands, including Cmd+V.
private struct NativeFilePathTextField: NSViewRepresentable {
    @Binding var text: String
    let placeholder: String
    let isEnabled: Bool
    let command: FilePathTextFieldCommand?
    let commandID: Int

    func makeCoordinator() -> Coordinator { Coordinator(text: $text) }

    func makeNSView(context: Context) -> NSTextField {
        let field = NSTextField()
        field.delegate = context.coordinator
        field.isEditable = true
        field.isSelectable = true
        field.isBezeled = false
        field.drawsBackground = false
        field.focusRingType = .none
        field.font = .systemFont(ofSize: 12)
        field.lineBreakMode = .byTruncatingTail
        field.maximumNumberOfLines = 1
        return field
    }

    func updateNSView(_ field: NSTextField, context: Context) {
        if field.stringValue != text { field.stringValue = text }
        field.placeholderString = placeholder
        field.isEnabled = isEnabled
        field.textColor = isEnabled ? .labelColor : .secondaryLabelColor

        guard context.coordinator.lastCommandID != commandID, let command else { return }
        DispatchQueue.main.async {
            guard field.window?.makeFirstResponder(field) == true,
                  let editor = field.currentEditor() as? NSTextView else { return }
            context.coordinator.lastCommandID = commandID
            switch command {
            case .paste:
                NSApp.sendAction(#selector(NSText.paste(_:)), to: nil, from: field)
            case .clear:
                editor.selectAll(nil)
                editor.insertText("", replacementRange: editor.selectedRange)
            }
        }
    }

    final class Coordinator: NSObject, NSTextFieldDelegate {
        @Binding var text: String
        var lastCommandID = 0

        init(text: Binding<String>) { _text = text }

        func controlTextDidChange(_ notification: Notification) {
            guard let field = notification.object as? NSTextField else { return }
            text = field.stringValue
        }
    }
}
