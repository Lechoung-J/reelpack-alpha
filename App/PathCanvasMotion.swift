import AppKit
import SwiftUI

/// A spring runs only while one of the nine leaves is approaching a new target.
/// A stationary pointer, an idle canvas and a settled return to origin have no timer.
@MainActor
final class PathCanvasMotion: ObservableObject {
    @Published private(set) var displacements = Array(repeating: CGSize.zero, count: PathCanvasGeometry.satellites.count)

    private var velocities = Array(repeating: CGSize.zero, count: PathCanvasGeometry.satellites.count)
    private var pointer: CGPoint?
    private var active = false
    private var timer: Timer?
    private var lastTime: TimeInterval = 0

    func setActive(_ enabled: Bool) {
        active = enabled
        if !enabled { reset() }
    }

    func setPointer(_ position: CGPoint?) {
        guard active else { return }
        pointer = position
        wakeIfNeeded()
    }

    func releasePointer() {
        pointer = nil
        wakeIfNeeded()
    }

    func reset() {
        stopTimer()
        pointer = nil
        velocities = Array(repeating: .zero, count: velocities.count)
        let origin = Array(repeating: CGSize.zero, count: displacements.count)
        if displacements != origin { displacements = origin }
    }

    private func target(for origin: CGPoint) -> CGSize {
        guard active, let pointer else { return .zero }
        let dx = origin.x - pointer.x
        let dy = origin.y - pointer.y
        let distance = hypot(dx, dy)
        guard distance < 66 else { return .zero }
        let strength = 22 * pow(1 - distance / 66, 1.4)
        if distance < 0.5 { return CGSize(width: 0, height: -strength) }
        return CGSize(width: dx / distance * strength, height: dy / distance * strength)
    }

    private func wakeIfNeeded() {
        guard active, timer == nil else { return }
        let unsettled = PathCanvasGeometry.satellites.enumerated().contains { index, origin in
            let target = target(for: origin)
            let displacement = displacements[index]
            let velocity = velocities[index]
            return hypot(target.width - displacement.width, target.height - displacement.height) >= 0.015
                || hypot(velocity.width, velocity.height) >= 0.04
        }
        guard unsettled else { return }
        lastTime = 0
        let next = Timer(timeInterval: 1 / 60, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
        timer = next
        RunLoop.main.add(next, forMode: .common)
    }

    private func tick() {
        guard active else { reset(); return }
        let now = ProcessInfo.processInfo.systemUptime
        let dt = lastTime == 0 ? 1 / 60 : min(max(0, now - lastTime), 1 / 30)
        lastTime = now
        let h = CGFloat(dt / 2)
        var positions = displacements
        var moving = false
        for (index, origin) in PathCanvasGeometry.satellites.enumerated() {
            let target = target(for: origin)
            var displacement = positions[index]
            var velocity = velocities[index]
            for _ in 0..<2 {
                velocity.width += (110 * (target.width - displacement.width) - 21 * velocity.width) * h
                velocity.height += (110 * (target.height - displacement.height) - 21 * velocity.height) * h
                displacement.width += velocity.width * h
                displacement.height += velocity.height * h
            }
            if hypot(displacement.width - target.width, displacement.height - target.height) < 0.015,
               hypot(velocity.width, velocity.height) < 0.04 {
                displacement = target
                velocity = .zero
            } else {
                moving = true
            }
            positions[index] = displacement
            velocities[index] = velocity
        }
        if positions != displacements { displacements = positions }
        if !moving { stopTimer() }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
        lastTime = 0
    }

    deinit { timer?.invalidate() }
}

/// This native background receives only empty-space mouse events. SwiftUI buttons
/// remain above it. It intentionally registers no drag types, allowing the parent's
/// whole-view onDrop handler to own external file drags.
struct PathCanvasInputSurface: NSViewRepresentable {
    let onHover: (CGPoint?) -> Void
    let onPan: (CGSize) -> Void
    let onZoom: (CGFloat, CGPoint) -> Void
    let onPanning: (Bool) -> Void
    let onVisibility: (Bool) -> Void

    func makeNSView(context: Context) -> PathCanvasInputNSView {
        let view = PathCanvasInputNSView()
        configure(view)
        return view
    }

    func updateNSView(_ view: PathCanvasInputNSView, context: Context) {
        configure(view)
    }

    static func dismantleNSView(_ view: PathCanvasInputNSView, coordinator: ()) {
        view.detach()
    }

    private func configure(_ view: PathCanvasInputNSView) {
        view.onHover = onHover
        view.onPan = onPan
        view.onZoom = onZoom
        view.onPanning = onPanning
        view.onVisibility = onVisibility
    }
}

final class PathCanvasInputNSView: NSView {
    var onHover: (CGPoint?) -> Void = { _ in }
    var onPan: (CGSize) -> Void = { _ in }
    var onZoom: (CGFloat, CGPoint) -> Void = { _, _ in }
    var onPanning: (Bool) -> Void = { _ in }
    var onVisibility: (Bool) -> Void = { _ in }

    private var tracking: NSTrackingArea?
    private var lastDragPoint: CGPoint?
    private var observers: [NSObjectProtocol] = []
    private var lastVisibility: Bool?

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let tracking { removeTrackingArea(tracking) }
        let area = NSTrackingArea(
            rect: .zero,
            options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        tracking = area
        addTrackingArea(area)
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        detachObservers()
        lastVisibility = nil
        window?.acceptsMouseMovedEvents = true
        let center = NotificationCenter.default
        if let window {
            for name in [
                NSWindow.didMiniaturizeNotification, NSWindow.didDeminiaturizeNotification,
                NSWindow.didChangeOcclusionStateNotification, NSWindow.willCloseNotification,
            ] {
                observers.append(center.addObserver(forName: name, object: window, queue: .main) { [weak self] notification in
                    if notification.name == NSWindow.willCloseNotification {
                        self?.publishVisibility(false)
                    } else {
                        self?.updateVisibility()
                    }
                })
            }
        }
        for name in [NSApplication.didHideNotification, NSApplication.didUnhideNotification] {
            observers.append(center.addObserver(forName: name, object: NSApp, queue: .main) { [weak self] _ in
                self?.updateVisibility()
            })
        }
        updateVisibility()
    }

    override func viewDidHide() {
        super.viewDidHide()
        updateVisibility()
    }

    override func viewDidUnhide() {
        super.viewDidUnhide()
        updateVisibility()
    }

    override func mouseEntered(with event: NSEvent) { mouseMoved(with: event) }

    override func mouseMoved(with event: NSEvent) {
        guard lastDragPoint == nil else { return }
        onHover(convert(event.locationInWindow, from: nil))
    }

    override func mouseExited(with event: NSEvent) { onHover(nil) }

    override func mouseDown(with event: NSEvent) {
        lastDragPoint = convert(event.locationInWindow, from: nil)
        onHover(nil)
        onPanning(true)
        NSCursor.closedHand.push()
    }

    override func mouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        guard let previous = lastDragPoint else { return }
        onPan(CGSize(width: point.x - previous.x, height: point.y - previous.y))
        lastDragPoint = point
    }

    override func mouseUp(with event: NSEvent) { finishPan() }

    override func scrollWheel(with event: NSEvent) {
        onHover(nil)
        let anchor = convert(event.locationInWindow, from: nil)
        if !event.modifierFlags.intersection([.command, .control, .option]).isEmpty {
            onZoom(CGFloat(exp(-Double(event.scrollingDeltaY) * 0.002)), anchor)
        } else {
            let multiplier: CGFloat = event.hasPreciseScrollingDeltas ? 1 : 12
            onPan(CGSize(width: event.scrollingDeltaX * multiplier, height: event.scrollingDeltaY * multiplier))
        }
    }

    override func magnify(with event: NSEvent) {
        onHover(nil)
        onZoom(max(0.01, 1 + event.magnification), convert(event.locationInWindow, from: nil))
    }

    private func finishPan() {
        guard lastDragPoint != nil else { return }
        lastDragPoint = nil
        NSCursor.pop()
        onPanning(false)
    }

    private func updateVisibility() {
        let visible = window.map {
            $0.isVisible && !$0.isMiniaturized && $0.occlusionState.contains(.visible) && !NSApp.isHidden && !isHiddenOrHasHiddenAncestor
        } ?? false
        publishVisibility(visible)
    }

    private func publishVisibility(_ visible: Bool) {
        guard visible != lastVisibility else { return }
        lastVisibility = visible
        if !visible {
            finishPan()
            onHover(nil)
        }
        // Attaching a represented view can occur during SwiftUI's update pass.
        let callback = onVisibility
        DispatchQueue.main.async { callback(visible) }
    }

    private func detachObservers() {
        for observer in observers { NotificationCenter.default.removeObserver(observer) }
        observers.removeAll()
    }

    func detach() {
        finishPan()
        onHover(nil)
        publishVisibility(false)
        detachObservers()
    }

    deinit { detachObservers() }
}
