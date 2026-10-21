import AppKit
import QuickLookThumbnailing
import SwiftUI
import Shared
import UniformTypeIdentifiers

struct DropZone: View {
    @Binding var droppedFile: URL?
    @Binding var errorMessage: String?
    let artifacts: [OutputArtifact]
    let isProcessing: Bool
    let progressMessage: String
    let progressPercent: Int
    let sourceDescription: String?
    let onChooseFile: () -> Void
    let onCancel: () -> Void
    let onClearResults: () -> Void
    let onRevealResults: () -> Void

    @State private var isDragOver = false

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor).opacity(0.54))
                .overlay {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(
                            isDragOver ? Color.accentColor.opacity(0.75) : Color.secondary.opacity(0.16),
                            style: StrokeStyle(lineWidth: isDragOver ? 2 : 1, dash: artifacts.isEmpty ? [7, 5] : [])
                        )
                }

            Group {
                if !artifacts.isEmpty {
                    completedContent
                } else if isProcessing {
                    processingContent
                } else if let file = droppedFile {
                    selectedFileContent(file)
                } else {
                    emptyContent
                }
            }
            .padding(32)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 18)
        .onDrop(of: [.fileURL], isTargeted: $isDragOver, perform: handleDrop)
        .animation(.easeInOut(duration: 0.18), value: isDragOver)
        .animation(.easeInOut(duration: 0.2), value: artifacts)
    }

    private var emptyContent: some View {
        VStack(spacing: 14) {
            Image(systemName: "arrow.down.doc.fill")
                .font(.system(size: 42, weight: .light))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(isDragOver ? Color.accentColor : Color.secondary)

            VStack(spacing: 5) {
                Text(isDragOver ? "松开以添加视频" : "将视频拖到这里")
                    .font(.system(size: 16, weight: .semibold))
                Text("支持 MP4、MOV 和 MKV")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }

            Button("选择视频…", action: onChooseFile)
                .buttonStyle(.bordered)
                .controlSize(.regular)
        }
    }

    private func selectedFileContent(_ file: URL) -> some View {
        VStack(spacing: 14) {
            ZStack(alignment: .topTrailing) {
                FinderThumbnail(url: file, size: CGSize(width: 300, height: 176))

                Button {
                    droppedFile = nil
                    errorMessage = nil
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 19))
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(.white, Color.black.opacity(0.55))
                }
                .buttonStyle(.plain)
                .offset(x: 9, y: -9)
                .help("移除视频")
            }

            VStack(spacing: 3) {
                Text(file.lastPathComponent)
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(1)
                Text(fileMetadata(file))
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            Label("已就绪，点击下方“导出资产包与文稿”开始", systemImage: "checkmark.circle.fill")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        }
    }

    private var processingContent: some View {
        VStack(spacing: 18) {
            Image(systemName: "waveform.and.magnifyingglass")
                .font(.system(size: 40, weight: .light))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(Color.accentColor)

            VStack(spacing: 6) {
                Text(progressMessage.isEmpty ? "正在准备…" : progressMessage)
                    .font(.system(size: 14, weight: .medium))
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                if let sourceDescription {
                    Text(sourceDescription)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }

            VStack(spacing: 7) {
                ProgressView(value: Double(progressPercent), total: 100)
                    .progressViewStyle(.linear)
                    .frame(maxWidth: 320)
                Text("\(progressPercent)%")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }

            Button("取消", action: onCancel)
                .buttonStyle(.bordered)
        }
    }

    private var completedContent: some View {
        ScrollView {
            VStack(spacing: 18) {
                Label("处理完成", systemImage: "checkmark.circle.fill")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.green)

                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 176, maximum: 200), alignment: .top)],
                    spacing: 16
                ) {
                    ForEach(artifacts) { artifact in
                        ArtifactCard(artifact: artifact)
                    }
                }
                .frame(maxWidth: 620)

                Text("Markdown 便于阅读，资产包用于 Agent 查询。\n结果已保存在所选目录，可双击打开或拖到 Finder。")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                HStack(spacing: 8) {
                    Button("在 Finder 中显示", action: onRevealResults)
                        .buttonStyle(.bordered)
                    Button("处理新视频", action: onClearResults)
                        .buttonStyle(.borderedProminent)
                }
            }
            .frame(maxWidth: .infinity)
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard !isProcessing, let provider = providers.first else { return false }
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
            guard let url else { return }
            let ext = url.pathExtension.lowercased()
            guard ["mp4", "mov", "mkv"].contains(ext) else {
                DispatchQueue.main.async { errorMessage = "不支持的文件格式，请选择 MP4、MOV 或 MKV。" }
                return
            }
            DispatchQueue.main.async {
                droppedFile = url
                errorMessage = nil
            }
        }
        return true
    }

    private func fileMetadata(_ url: URL) -> String {
        guard let values = try? url.resourceValues(forKeys: [.fileSizeKey]), let size = values.fileSize else {
            return url.pathExtension.uppercased()
        }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return "\(url.pathExtension.uppercased()) · \(formatter.string(fromByteCount: Int64(size)))"
    }
}

private struct ArtifactCard: View {
    let artifact: OutputArtifact

    var body: some View {
        VStack(spacing: 9) {
            ZStack(alignment: .bottomTrailing) {
                FinderThumbnail(url: artifact.url, size: CGSize(width: 156, height: 104))
                Image(systemName: artifact.kind.symbol)
                    .font(.system(size: 14))
                    .foregroundStyle(.white)
                    .padding(6)
                    .background(.black.opacity(0.58), in: Circle())
                    .padding(7)
            }

            VStack(spacing: 2) {
                Text(artifact.url.lastPathComponent)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 170)
                Text(artifact.kind.title)
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .contentShape(RoundedRectangle(cornerRadius: 10))
        .onTapGesture(count: 2) { NSWorkspace.shared.open(artifact.url) }
        .onDrag {
            if artifact.kind == .package {
                return NSItemProvider(object: artifact.url as NSURL)
            }
            return NSItemProvider(contentsOf: artifact.url) ?? NSItemProvider(object: artifact.url as NSURL)
        }
        .contextMenu {
            Button("打开") { NSWorkspace.shared.open(artifact.url) }
            Button("在 Finder 中显示") {
                NSWorkspace.shared.activateFileViewerSelecting([artifact.url])
            }
        }
        .help("拖到 Finder，或双击打开")
    }
}

struct FinderThumbnail: View {
    let url: URL
    let size: CGSize
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(Color(nsColor: .textBackgroundColor))
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .padding(5)
            } else {
                Image(nsImage: NSWorkspace.shared.icon(forFile: url.path))
                    .resizable()
                    .scaledToFit()
                    .padding(24)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: size.width, height: size.height)
        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(Color.secondary.opacity(0.14), lineWidth: 1)
        }
        .shadow(color: .black.opacity(0.08), radius: 7, y: 3)
        .onAppear(perform: loadThumbnail)
        .onChange(of: url) { _ in loadThumbnail() }
    }

    private func loadThumbnail() {
        let scale = NSScreen.main?.backingScaleFactor ?? 2
        let request = QLThumbnailGenerator.Request(
            fileAt: url,
            size: size,
            scale: scale,
            representationTypes: .all
        )
        QLThumbnailGenerator.shared.generateBestRepresentation(for: request) { representation, _ in
            guard let representation else { return }
            DispatchQueue.main.async { image = representation.nsImage }
        }
    }
}
