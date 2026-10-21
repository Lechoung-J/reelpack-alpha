import AppKit
import SwiftUI

/// Trial-only affordances that are inert unless the tester sets an environment
/// variable. They add no processing behavior and change no Core/CLI/IPC contract.
///
/// `REELPACK_UI_TRIAL_INPUT`   – local video path handed to the same `accept` entry
///                               a real drop uses (never a synthetic event).
/// `REELPACK_UI_TRIAL_CANCEL_AFTER` – seconds after start to request real cancellation.
/// `REELPACK_UI_CAPTURE_DIR`   – where native renders of the real view are written.
enum UITrialHarness {
    static var input: URL? {
        guard let path = ProcessInfo.processInfo.environment["REELPACK_UI_TRIAL_INPUT"], !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }

    static var cancelAfter: Double? {
        guard let raw = ProcessInfo.processInfo.environment["REELPACK_UI_TRIAL_CANCEL_AFTER"], let value = Double(raw), value > 0 else { return nil }
        return value
    }

    static var captureDirectory: URL? {
        guard let path = ProcessInfo.processInfo.environment["REELPACK_UI_CAPTURE_DIR"], !path.isEmpty else { return nil }
        let url = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Trial output directory. It never writes back to the saved preferences.
    static var outputOverride: String? {
        guard let path = ProcessInfo.processInfo.environment["REELPACK_UI_TRIAL_OUTPUT"], !path.isEmpty else { return nil }
        return path
    }
}

/// Renders the real presentation view natively (no screen-recording permission and
/// no browser). A live-window bitmap is used when the window is on screen so the
/// capture also proves what the user actually sees.
@MainActor
enum UITrialCapture {
    static func liveWindow(named name: String, window: NSWindow?) {
        guard let directory = UITrialHarness.captureDirectory else { return }
        guard let view = window?.contentView, view.bounds.width > 1, view.bounds.height > 1 else { return }
        write(view: view, size: view.bounds.size, appearance: window?.effectiveAppearance,
              to: directory.appendingPathComponent("live-\(name).png"))
    }

    static func render(
        name: String,
        nodes: [PathNodePresentation],
        stateText: String,
        dropHover: Bool = false,
        isBusy: Bool = false,
        isPreparing: Bool = false,
        showResults: Bool = false,
        showsDetails: Bool = false,
        outputPath: String,
        sizes: [(String, CGSize)] = [("default", CGSize(width: 760, height: 600)),
                                     ("minimum", CGSize(width: 700, height: 560))],
        appearances: [(String, NSAppearance.Name)] = [("light", .aqua), ("dark", .darkAqua)]
    ) {
        guard let directory = UITrialHarness.captureDirectory else { return }
        for (sizeName, size) in sizes {
            for (appearanceName, appearance) in appearances {
                let view = NSHostingView(rootView: PathCanvasView(
                    nodes: nodes,
                    stateText: stateText,
                    dropHover: dropHover,
                    isBusy: isBusy,
                    isPreparing: isPreparing,
                    showResults: showResults,
                    outputPath: outputPath,
                    logo: BrandArtwork.image,
                    showsDetails: showsDetails,
                    onChoose: {}, onOutput: {}, onCancel: {}, onResults: {}, onDetails: {}, onNodeDetails: { _ in }
                ))
                view.frame = CGRect(origin: .zero, size: size)
                write(view: view, size: size, appearance: NSAppearance(named: appearance),
                      to: directory.appendingPathComponent("\(name)-\(sizeName)-\(appearanceName).png"))
            }
        }
    }

    private static func write(view: NSView, size: CGSize, appearance: NSAppearance?, to url: URL) {
        view.appearance = appearance
        view.layoutSubtreeIfNeeded()
        guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        guard let data = rep.representation(using: .png, properties: [:]) else { return }
        try? data.write(to: url)
    }
}
