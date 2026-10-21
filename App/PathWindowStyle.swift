import AppKit
import SwiftUI

/// A presentation-only window accessor. Native controls and resize behavior remain.
struct PathWindowStyle: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView { Accessor() }
    func updateNSView(_ nsView: NSView, context: Context) {}

    private final class Accessor: NSView {
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard let window else { return }
            window.titleVisibility = .hidden
            window.titlebarAppearsTransparent = true
            window.styleMask.insert(.fullSizeContentView)
            window.isMovableByWindowBackground = false
            window.contentMinSize = NSSize(width: 700, height: 560)
        }
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }
}
