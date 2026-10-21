import Foundation
import Vision
import AppKit
import CoreVideo

struct FrameResult {
    let file: String
    let text: String
    let error: String?
}

func recognizeText(in path: String) -> FrameResult {
    guard let image = NSImage(contentsOfFile: path),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return FrameResult(file: path, text: "", error: "无法读取图片")
    }

    // Supply owned, explicitly laid-out BGRA storage. Passing a CGImage lets
    // Vision choose its internal 420f conversion, which fails on some hosts.
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, cgImage.width, cgImage.height,
        kCVPixelFormatType_32BGRA,
        [kCVPixelBufferCGImageCompatibilityKey: true,
         kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary, &buffer)
    guard status == kCVReturnSuccess, let pixelBuffer = buffer else {
        return FrameResult(file: path, text: "", error: "CVPixelBufferCreate BGRA failed: \(status)")
    }
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    guard let context = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer),
        width: cgImage.width, height: cgImage.height, bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue) else {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        return FrameResult(file: path, text: "", error: "Cannot construct BGRA bitmap context")
    }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
    CVPixelBufferUnlockBaseAddress(pixelBuffer, [])

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = CommandLine.arguments.contains("--fast") ? .fast : .accurate
    request.usesCPUOnly = !CommandLine.arguments.contains("--allow-gpu")
    if CommandLine.arguments.contains("--revision1") { request.revision = VNRecognizeTextRequestRevision1 }
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    request.usesLanguageCorrection = true

    do {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])
        let lines = (request.results ?? []).compactMap { observation in
            observation.topCandidates(1).first?.string
        }
        return FrameResult(file: path, text: lines.joined(separator: "\n"), error: nil)
    } catch {
        return FrameResult(file: path, text: "", error: "OCR 失败: \(error.localizedDescription) [\((error as NSError).domain):\((error as NSError).code)]")
    }
}

func jsonObject(for result: FrameResult) -> [String: Any] {
    [
        "file": result.file,
        "text": result.text,
        "error": result.error ?? NSNull()
    ]
}

let arguments = Array(CommandLine.arguments.dropFirst())
let jsonMode = arguments.contains("--json")
let flags: Set<String> = ["--json", "--fast", "--allow-gpu", "--revision1"]
let paths = arguments.filter { !flags.contains($0) }

if paths.isEmpty {
    fputs("用法: macos_ocr [--json] [--fast] [--allow-gpu] [--revision1] <图片路径>...\n", stderr)
    exit(2)
}

let start = Date()
var results: [FrameResult] = []
results.reserveCapacity(paths.count)

for path in paths {
    autoreleasepool {
        results.append(recognizeText(in: path))
    }
}

let elapsed = Date().timeIntervalSince(start)

if jsonMode {
    let payload: [String: Any] = [
        "frames": results.map(jsonObject),
        "count": results.count,
        "seconds": elapsed
    ]
    do {
        let data = try JSONSerialization.data(withJSONObject: payload, options: [])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    } catch {
        fputs("JSON 输出失败: \(error.localizedDescription)\n", stderr)
        exit(1)
    }
} else {
    for result in results {
        if !result.text.isEmpty {
            print(result.text)
        }
        if let error = result.error {
            fputs("\(result.file): \(error)\n", stderr)
        }
    }
    print("--- recognized \(results.reduce(0) { $0 + ($1.text.isEmpty ? 0 : $1.text.split(separator: "\n").count) }) lines in \(String(format: "%.3f", elapsed))s ---")
}
