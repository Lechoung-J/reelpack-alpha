// Minimal allocation-only reproduction: no JPEG, CGImage, OCR or model involved.
import Foundation
import CoreVideo
for (name, format) in [("BGRA", kCVPixelFormatType_32BGRA), ("420f", kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)] {
    for surface in [false, true] {
        var buffer: CVPixelBuffer?
        let attrs = surface ? [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary : nil
        let status = CVPixelBufferCreate(kCFAllocatorDefault, 640, 1138, format, attrs, &buffer)
        print("format=\(name) IOSurface=\(surface) CVReturn=\(status) allocated=\(buffer != nil)")
    }
}
