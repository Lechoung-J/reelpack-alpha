# macOS Vision OCR helper

Build a native helper for macOS 14 or later:

```sh
ARCH=arm64 sh Tools/macos_ocr/build.sh
# Intel:
ARCH=x86_64 sh Tools/macos_ocr/build.sh
```

Recognize visible text in one or more images:

```sh
Tools/macos_ocr/macos_ocr --json /path/to/frame.jpg
```

JSON contains `count`, `seconds` and per-frame `file`, `text`, `error` fields. Check each frame's error even when the process exits successfully. An OCR failure is different from a successful image with no visible text.

The default uses accurate recognition, CPU processing and Chinese/English language settings. It supplies owned BGRA pixel-buffer storage. Diagnostic flags `--fast`, `--allow-gpu` and `--revision1` change recognition behavior and should not silently replace the default.

Vision requires access to macOS image and IOSurface services. If recognition fails in a restricted runtime, test the same image on the host and retain the actual error. The helper does not bypass system restrictions.

The optional allocation diagnostic can be built separately:

```sh
swiftc -target arm64-apple-macosx14.0 \
  Tools/macos_ocr/diagnose.swift -o /tmp/reelpack-ocr-diagnose -framework CoreVideo
/tmp/reelpack-ocr-diagnose
```
