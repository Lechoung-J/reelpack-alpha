#!/bin/sh
set -eu
cd "$(dirname "$0")"
arch="${ARCH:-$(uname -m)}"
case "$arch" in
    arm64|x86_64) ;;
    *) echo "Unsupported macOS architecture: $arch" >&2; exit 1 ;;
esac
module_cache="${TMPDIR:-/tmp}/reelpack-swift-module-cache-${arch}"
mkdir -p "$module_cache"
swiftc -O -target "${arch}-apple-macosx14.0" -module-cache-path "$module_cache" ocr_main.swift -o macos_ocr -framework Vision -framework AppKit
