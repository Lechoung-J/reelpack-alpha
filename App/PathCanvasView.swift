import AppKit
import SwiftUI

/// Presentation facts supplied by the workflow adapter, never inferred from a percentage.
enum PathNodeStatus: String, CaseIterable, Equatable {
    case pending
    case observed
    case running
    case complete
    case notRun = "not_run"
    case failed
    case unknown
    case partial
    case cancelled

    var defaultLabel: String {
        switch self {
        case .pending: return "未开始"
        case .observed: return "等待确认"
        case .running: return "运行中"
        case .complete: return "已完成"
        case .notRun: return "未运行"
        case .failed: return "失败"
        case .unknown: return "状态未知"
        case .partial: return "部分完成"
        case .cancelled: return "已取消"
        }
    }

    var needsVisibleLabel: Bool {
        switch self {
        case .notRun, .failed, .unknown, .partial, .cancelled: return true
        case .pending, .observed, .running, .complete: return false
        }
    }
}

struct PathNodePresentation: Identifiable, Equatable {
    let id: String
    let name: String
    let status: PathNodeStatus
    let statusLabel: String
    let reason: String?

    init(
        id: String,
        name: String,
        status: PathNodeStatus = .pending,
        statusLabel: String? = nil,
        reason: String? = nil
    ) {
        self.id = id
        self.name = name
        self.status = status
        self.statusLabel = statusLabel ?? status.defaultLabel
        self.reason = reason
    }

    static let defaults = [
        PathNodePresentation(id: "transcription", name: "聆听言语"),
        PathNodePresentation(id: "shots", name: "整理镜头"),
        PathNodePresentation(id: "ocr", name: "拾取文字"),
        PathNodePresentation(id: "music", name: "感知节奏"),
    ]
}

extension Notification.Name {
    /// A parent menu/keyboard command can reset the canvas without owning its transform.
    static let pathCanvasResetTransform = Notification.Name("Reelpack.pathCanvasResetTransform")
}

/// Path layout in a 760 × 600 pt coordinate system.
/// Circular arc lengths also define the nine decorative leaves; no browser is involved.
enum PathCanvasGeometry {
    static let size = CGSize(width: 760, height: 600)
    static let center = CGPoint(x: 380, y: 300)
    static let nodeCenters = [
        CGPoint(x: 220, y: 220), CGPoint(x: 450, y: 220),
        CGPoint(x: 450, y: 350), CGPoint(x: 220, y: 350),
    ]
    static let firstArcLength = CGFloat.pi * 55 / 2
    static let topLength: CGFloat = 475
    static let returnArcLength = CGFloat.pi * 65
    static let bottomLength: CGFloat = 460
    static let length = firstArcLength + topLength + returnArcLength + bottomLength

    static func point(at distance: CGFloat) -> CGPoint {
        let d = min(length, max(0, distance))
        if d <= firstArcLength {
            let angle = CGFloat.pi + d / 55
            return CGPoint(x: 135 + 55 * cos(angle), y: 275 + 55 * sin(angle))
        }
        if d <= firstArcLength + topLength {
            return CGPoint(x: 135 + d - firstArcLength, y: 220)
        }
        if d <= firstArcLength + topLength + returnArcLength {
            let angle = -CGFloat.pi / 2 + (d - firstArcLength - topLength) / 65
            return CGPoint(x: 610 + 65 * cos(angle), y: 285 + 65 * sin(angle))
        }
        return CGPoint(x: 610 - (d - firstArcLength - topLength - returnArcLength), y: 350)
    }

    static let satellites: [CGPoint] = {
        var candidates: [CGPoint] = []
        var distance: CGFloat = 0
        while distance <= length {
            let point = point(at: distance)
            if !nodeCenters.contains(where: { hypot($0.x - point.x, $0.y - point.y) < 36 }) {
                candidates.append(point)
            }
            distance += 57
        }
        let removed: Set<Int> = [0, 1, 4, 7, 9, 10, 12, 15, 17]
        return candidates.enumerated().compactMap { removed.contains($0.offset) ? nil : $0.element }
    }()

    static var track: Path {
        var path = Path()
        path.move(to: CGPoint(x: 80, y: 275))
        path.addArc(
            center: CGPoint(x: 135, y: 275), radius: 55,
            startAngle: .degrees(180), endAngle: .degrees(270), clockwise: false
        )
        path.addLine(to: CGPoint(x: 610, y: 220))
        path.addArc(
            center: CGPoint(x: 610, y: 285), radius: 65,
            startAngle: .degrees(-90), endAngle: .degrees(90), clockwise: false
        )
        path.addLine(to: CGPoint(x: 150, y: 350))
        return path
    }
}

private struct PathCanvasPalette {
    let background: Color
    let ink: Color
    let track: Color
    let purple: Color
    let stem: Color
    let secondary: Color

    init(scheme: ColorScheme) {
        // Paired semantic colors from the frozen tokens; the artwork is never inverted.
        if scheme == .dark {
            background = Color(red: 25 / 255, green: 25 / 255, blue: 27 / 255)
            ink = Color(red: 242 / 255, green: 242 / 255, blue: 240 / 255)
            track = Color(red: 118 / 255, green: 118 / 255, blue: 123 / 255)
            purple = Color(red: 169 / 255, green: 140 / 255, blue: 255 / 255)
            stem = track
            secondary = Color(red: 176 / 255, green: 176 / 255, blue: 182 / 255)
        } else {
            background = .white
            ink = Color(red: 23 / 255, green: 23 / 255, blue: 25 / 255)
            track = Color(red: 147 / 255, green: 147 / 255, blue: 151 / 255)
            purple = Color(red: 113 / 255, green: 70 / 255, blue: 232 / 255)
            stem = Color(red: 154 / 255, green: 154 / 255, blue: 159 / 255)
            secondary = Color(red: 101 / 255, green: 101 / 255, blue: 108 / 255)
        }
    }
}

private struct PathCanvasTransform {
    var scale: CGFloat = 1
    var translation = CGSize.zero

    mutating func zoom(by factor: CGFloat, around point: CGPoint, in viewport: CGSize) {
        guard factor.isFinite, factor > 0 else { return }
        let next = min(1.8, max(0.45, scale * factor))
        let anchor = CGPoint(x: point.x - viewport.width / 2, y: point.y - viewport.height / 2)
        translation = CGSize(
            width: anchor.x - (anchor.x - translation.width) * next / scale,
            height: anchor.y - (anchor.y - translation.height) * next / scale
        )
        scale = next
    }

    func canvasPoint(from point: CGPoint, viewport: CGSize) -> CGPoint {
        CGPoint(
            x: PathCanvasGeometry.center.x + (point.x - viewport.width / 2 - translation.width) / scale,
            y: PathCanvasGeometry.center.y + (point.y - viewport.height / 2 - translation.height) / scale
        )
    }
}

/// Native presentation only. The parent owns file drops, workflow, output selection and details.
struct PathCanvasView: View {
    let nodes: [PathNodePresentation]
    let stateText: String
    let dropHover: Bool
    let isBusy: Bool
    let isPreparing: Bool
    let showResults: Bool
    let outputPath: String
    let logo: NSImage?
    let showsDetails: Bool
    let onChoose: () -> Void
    let onOutput: () -> Void
    let onCancel: () -> Void
    let onResults: () -> Void
    let onDetails: () -> Void
    let onNodeDetails: (PathNodePresentation) -> Void

    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.controlActiveState) private var controlActiveState
    @StateObject private var motion = PathCanvasMotion()
    @State private var transform = PathCanvasTransform()
    @State private var isPanning = false
    @State private var isWindowVisible = true

    init(
        nodes: [PathNodePresentation],
        stateText: String,
        dropHover: Bool,
        isBusy: Bool,
        isPreparing: Bool = false,
        showResults: Bool,
        outputPath: String,
        logo: NSImage?,
        showsDetails: Bool = false,
        onChoose: @escaping () -> Void,
        onOutput: @escaping () -> Void,
        onCancel: @escaping () -> Void,
        onResults: @escaping () -> Void,
        onDetails: @escaping () -> Void,
        onNodeDetails: @escaping (PathNodePresentation) -> Void = { _ in }
    ) {
        self.nodes = nodes
        self.stateText = stateText
        self.dropHover = dropHover
        self.isBusy = isBusy
        self.isPreparing = isPreparing
        self.showResults = showResults
        self.outputPath = outputPath
        self.logo = logo
        self.showsDetails = showsDetails
        self.onChoose = onChoose
        self.onOutput = onOutput
        self.onCancel = onCancel
        self.onResults = onResults
        self.onDetails = onDetails
        self.onNodeDetails = onNodeDetails
    }

    private var orderedNodes: [PathNodePresentation] {
        PathNodePresentation.defaults.map { fallback in
            nodes.first(where: { $0.id == fallback.id }) ?? fallback
        }
    }

    private var motionEnabled: Bool { !reduceMotion && !isPanning && !dropHover && isWindowVisible }
    // This gate belongs only to the running-node ring; leaf motion is unchanged.
    private var nodeBreathingEnabled: Bool {
        isBusy && !reduceMotion && isWindowVisible && controlActiveState == .key
    }
    private var showsPreparation: Bool { isBusy && isPreparing }
    private var hasStatusRow: Bool { !stateText.isEmpty || isBusy || showResults || showsDetails }

    var body: some View {
        GeometryReader { proxy in
            let palette = PathCanvasPalette(scheme: colorScheme)
            ZStack(alignment: .topLeading) {
                palette.background
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)

                PathCanvasInputSurface(
                    onHover: { point in
                        guard let point else { motion.releasePointer(); return }
                        let canvasPoint = transform.canvasPoint(from: point, viewport: proxy.size)
                        let overNode = PathCanvasGeometry.nodeCenters.contains {
                            hypot($0.x - canvasPoint.x, $0.y - canvasPoint.y) <= 20
                        }
                        let overFolder = CGRect(x: proxy.size.width - 63, y: proxy.size.height - 62, width: 40, height: 40).contains(point)
                        let rowWidth = max(1, min(540, proxy.size.width - 160))
                        let overStatus = hasStatusRow && CGRect(
                            x: (proxy.size.width - rowWidth) / 2,
                            y: proxy.size.height / 2 + 151,
                            width: rowWidth, height: 26
                        ).contains(point)
                        if overNode || overFolder || overStatus { motion.releasePointer() }
                        else { motion.setPointer(canvasPoint) }
                    },
                    onPan: { delta in
                        transform.translation.width += delta.width
                        transform.translation.height += delta.height
                    },
                    onZoom: { factor, anchor in
                        motion.releasePointer()
                        transform.zoom(by: factor, around: anchor, in: proxy.size)
                    },
                    onPanning: { panning in isPanning = panning },
                    onVisibility: { visible in isWindowVisible = visible }
                )
                .accessibilityHidden(true)

                drawing(palette: palette)
                    .frame(width: PathCanvasGeometry.size.width, height: PathCanvasGeometry.size.height)
                    .scaleEffect(transform.scale, anchor: .center)
                    .position(
                        x: proxy.size.width / 2 + transform.translation.width,
                        y: proxy.size.height / 2 + transform.translation.height
                    )

                Text("Reelpack")
                    .font(titleFont)
                    .tracking(-0.6)
                    .foregroundStyle(palette.ink)
                    .frame(height: 28)
                    .position(x: proxy.size.width / 2, y: 31)
                    .allowsHitTesting(false)
                    .accessibilityAddTraits(.isHeader)

                if let logo {
                    Image(nsImage: logo)
                        .resizable()
                        .interpolation(.high)
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 57, height: 64)
                        .position(x: 57.5, y: proxy.size.height - 59)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                }

                Button(action: onOutput) {
                    Image(systemName: "folder")
                        .font(.system(size: 20, weight: .regular))
                        .foregroundStyle(palette.ink)
                        .frame(width: 40, height: 40)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .position(x: proxy.size.width - 43, y: proxy.size.height - 42)
                .help(outputPath.isEmpty ? "选择输出位置" : "选择输出位置 · \(outputPath)")
                .accessibilityLabel("选择输出位置")
                .accessibilityValue(outputPath)
                .accessibilityIdentifier("pathCanvas.outputFolder")

                if hasStatusRow {
                    statusRow(palette: palette)
                        .frame(width: max(1, min(540, proxy.size.width - 160)), height: 26)
                        .position(x: proxy.size.width / 2, y: proxy.size.height / 2 + 164)
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .clipped()
            .contextMenu {
                Button("选择视频…", action: onChoose).disabled(isBusy)
                Button("复位画布", action: resetTransform)
                Divider()
                Menu("能力状态") {
                    ForEach(orderedNodes) { node in
                        Button("\(node.name) · \(node.statusLabel)") { onNodeDetails(node) }
                    }
                }
                if isBusy { Button("取消处理", action: onCancel) }
                if showResults { Button("打开结果", action: onResults) }
                if showsDetails { Button("查看详情", action: onDetails) }
            }
        }
        .ignoresSafeArea()
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Reelpack 视频处理画布")
        .accessibilityAction(named: Text("选择视频")) { if !isBusy { onChoose() } }
        .accessibilityAction(named: Text("复位画布"), resetTransform)
        .onReceive(NotificationCenter.default.publisher(for: .pathCanvasResetTransform)) { _ in resetTransform() }
        .onAppear { motion.setActive(motionEnabled) }
        .onChange(of: motionEnabled) { _, enabled in motion.setActive(enabled) }
        .onDisappear { motion.setActive(false) }
    }

    private func drawing(palette: PathCanvasPalette) -> some View {
        ZStack(alignment: .topLeading) {
            Canvas { context, _ in
                context.stroke(
                    PathCanvasGeometry.track,
                    with: .color(dropHover ? palette.purple.opacity(0.65) : palette.track),
                    lineWidth: 0.8
                )
                for (index, point) in [CGPoint(x: 80, y: 275), CGPoint(x: 150, y: 350)].enumerated() {
                    let color = index == 0 && showsPreparation ? palette.purple : palette.ink
                    context.fill(Path(ellipseIn: CGRect(x: point.x - 2.5, y: point.y - 2.5, width: 5, height: 5)), with: .color(color))
                }
                for (index, origin) in PathCanvasGeometry.satellites.enumerated() {
                    let offset = motion.displacements[index]
                    context.fill(
                        Path(ellipseIn: CGRect(x: origin.x + offset.width - 8, y: origin.y + offset.height - 8, width: 16, height: 16)),
                        with: .color(palette.ink)
                    )
                }
                for center in PathCanvasGeometry.nodeCenters {
                    var stem = Path()
                    stem.move(to: CGPoint(x: center.x, y: center.y + 19))
                    stem.addLine(to: CGPoint(x: center.x, y: center.y + 46))
                    context.stroke(stem, with: .color(palette.stem), style: StrokeStyle(lineWidth: 1, lineCap: .round, dash: [0.1, 2.4]))
                    context.fill(
                        Path(ellipseIn: CGRect(x: center.x - 1.5, y: center.y + 45.5, width: 3, height: 3)),
                        with: .color(palette.ink)
                    )
                }
            }
            .allowsHitTesting(false)
            .accessibilityHidden(true)

            if showsPreparation {
                Circle()
                    .stroke(palette.purple.opacity(0.28), lineWidth: 1)
                    .frame(width: 15, height: 15)
                    .position(x: 80, y: 275)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)

                Text("已接收 · 准备中")
                    .font(.system(size: 11))
                    .foregroundStyle(palette.purple)
                    .fixedSize(horizontal: true, vertical: true)
                    .frame(width: 120, height: 18, alignment: .leading)
                    .position(x: 140, y: 297)
                    .allowsHitTesting(false)
                    .accessibilityLabel("视频已接收，正在准备")
                    .accessibilityIdentifier("pathCanvas.preparing")
            }

            ForEach(Array(orderedNodes.enumerated()), id: \.element.id) { index, node in
                let center = PathCanvasGeometry.nodeCenters[index]
                nodeButton(node, number: index + 1, palette: palette)
                    .position(center)

                Text(node.name)
                    .font(stageFont)
                    .tracking(1.4)
                    .foregroundStyle(palette.ink)
                    .fixedSize(horizontal: true, vertical: true)
                    .frame(width: 174, height: 26.6, alignment: .leading)
                    .position(x: center.x - 13 + 87, y: center.y + 58 + 13.3)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)

                if node.status.needsVisibleLabel || (node.status == .complete && node.statusLabel != node.status.defaultLabel) {
                    Text(node.statusLabel)
                        .font(.system(size: 10, weight: .regular))
                        .foregroundStyle(palette.secondary)
                        .lineLimit(1)
                        .frame(width: 174, height: 14, alignment: .leading)
                        .position(x: center.x - 13 + 87, y: center.y + 92)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
            }
        }
    }

    private func nodeButton(_ node: PathNodePresentation, number: Int, palette: PathCanvasPalette) -> some View {
        Button { onNodeDetails(node) } label: {
            ZStack {
                if node.status == .running {
                    PathRunningNodeRing(color: palette.purple, isAnimating: nodeBreathingEnabled)
                }
                Circle()
                    .fill(node.status == .running || node.status == .complete ? palette.purple : palette.ink)
                    .frame(width: 26, height: 26)
                if dropHover && node.status != .running && node.status != .complete {
                    Circle().stroke(palette.purple.opacity(0.55), lineWidth: 1).frame(width: 26, height: 26)
                }
                Text(String(format: "%02d", number))
                    .font(.system(size: 10, weight: .medium))
                    .tracking(0.3)
                    // Dark neutral nodes use light ink, so their numerals need dark ink.
                    .foregroundStyle(colorScheme == .dark ? palette.background : .white)
            }
            .frame(width: 40, height: 40)
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .help(nodeHelp(node))
        .accessibilityLabel("\(String(format: "%02d", number))，\(node.name)")
        .accessibilityValue(node.reason.map { "\(node.statusLabel)，\($0)" } ?? node.statusLabel)
        .accessibilityHint("查看能力状态与原因")
        .accessibilityIdentifier("pathCanvas.node.\(node.id)")
        .contextMenu {
            Text("\(node.name) · \(node.statusLabel)")
            if let reason = node.reason, !reason.isEmpty { Text(reason) }
            Button("查看详情") { onNodeDetails(node) }
        }
    }

    private func statusRow(palette: PathCanvasPalette) -> some View {
        HStack(spacing: 10) {
            if !stateText.isEmpty || isBusy {
                Text(stateText.isEmpty ? "正在处理" : stateText)
                    .foregroundStyle(palette.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .help(stateText)
                    .accessibilityIdentifier("pathCanvas.globalStatus")
            }
            if isBusy { Button("取消", action: onCancel).accessibilityLabel("取消处理") }
            if showResults { Button("打开结果", action: onResults) }
            if showsDetails { Button("查看详情", action: onDetails) }
        }
        .font(.system(size: 12))
        .buttonStyle(.borderless)
        .tint(palette.purple)
        .fixedSize(horizontal: false, vertical: true)
    }

    private var titleFont: Font {
        if let font = NSFont(name: "Georgia", size: 21) { return Font(font) }
        return .system(size: 21, weight: .regular, design: .serif)
    }

    private var stageFont: Font {
        for name in ["Songti SC", "STSong"] {
            if let font = NSFont(name: name, size: 19) { return Font(font) }
        }
        return .system(size: 19, weight: .regular, design: .serif)
    }

    private func nodeHelp(_ node: PathNodePresentation) -> String {
        let status = "\(node.name) · \(node.statusLabel)"
        guard let reason = node.reason, !reason.isEmpty else { return status }
        return "\(status)\n\(reason)"
    }

    private func resetTransform() {
        transform = PathCanvasTransform()
        motion.reset()
    }
}

/// Only the ring redraws. The timeline exists solely for a visible, active running
/// node; idle, completed, hidden and reduced-motion presentations have no schedule.
private struct PathRunningNodeRing: View {
    let color: Color
    let isAnimating: Bool

    var body: some View {
        Group {
            if isAnimating {
                TimelineView(.animation(minimumInterval: 1.0 / 30, paused: !isAnimating)) { context in
                    let phase = (1 - cos(context.date.timeIntervalSinceReferenceDate * 2 * .pi / 2.6)) / 2
                    ring(opacity: 0.22 + 0.12 * phase, scale: 0.98 + 0.04 * phase)
                }
            } else {
                ring(opacity: 0.28, scale: 1)
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private func ring(opacity: Double, scale: Double) -> some View {
        Circle()
            .stroke(color.opacity(opacity), lineWidth: 1)
            .frame(width: 37, height: 37)
            .scaleEffect(scale)
    }
}
