"""Pixel and normalized OCR regions, without decoding pixel data.

Backends that expose bounding boxes (Tesseract words) report pixel rectangles.
This module keeps that rectangle in the saved contract as
``region = {x, y, w, h}`` (plus the historical width/height aliases) and derives
a 0-1 ``region_norm`` when the OCR frame size is known.  Items produced by
backends that only return text keep ``region=None`` and ``region_norm=None``;
old packages are migrated the same way and never raise.

``region=None`` does not mean the run failed: it means the selected backend
does not (or, for RapidOCR, is not yet verified to) expose coordinates.  The
saved ``region_support`` declaration tells consumers which backends can produce
regions, so a null region is read as "backend did not provide it" rather than
"implementation failure".  RapidOCR coordinates are never synthesized here; its
support stays ``"unverified"`` until measured on the bundled ONNX worker.
"""
import math

# Declared per-backend region capability.  ``True`` = coordinates are saved,
# ``False`` = text only, ``"unverified"`` = not yet confirmed.  Keep this in
# sync with the backends wired in ``core._recognize_ocr``.
REGION_SUPPORT = {
    "tesseract": True,
    "rapidocr": "unverified",
    "macos": False,
    "ollama": False,
}

# JPEG Start-Of-Frame markers that carry the image dimensions.
_SOF_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def image_size(data):
    """Return ``(width, height)`` for JPEG/PNG bytes, or None if unreadable."""
    if not isinstance(data, (bytes, bytearray)):
        return None
    data = bytes(data)
    if data.startswith(b'\x89PNG\r\n\x1a\n') and len(data) >= 24:
        width = int.from_bytes(data[16:20], 'big')
        height = int.from_bytes(data[20:24], 'big')
        return (width, height) if width and height else None
    if data[:2] != b'\xff\xd8':
        return None
    index = 2
    while index + 3 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker == 0xFF:
            index += 1
            continue
        if marker == 0xD8 or marker == 0xD9 or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        length = int.from_bytes(data[index + 2:index + 4], 'big')
        if length < 2:
            return None
        if marker in _SOF_MARKERS:
            if index + 9 > len(data):
                return None
            height = int.from_bytes(data[index + 5:index + 7], 'big')
            width = int.from_bytes(data[index + 7:index + 9], 'big')
            return (width, height) if width and height else None
        index += 2 + length
    return None


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _size(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = value
        if _number(width) and _number(height) and width > 0 and height > 0:
            return float(width), float(height)
    return None


def ensure_item_region(item, frame_size=None):
    """Guarantee ``region``/``region_norm`` keys on one OCR item.

    ``region`` keeps x/y/w/h in the pixels of the OCR frame (historical
    width/height aliases are preserved).  ``region_norm`` is only computed when
    both a usable rectangle and the OCR frame size are available.
    """
    if not isinstance(item, dict):
        return item
    region = item.get('region')
    if not isinstance(region, dict):
        item['region'] = None
        item.setdefault('region_norm', None)
        return item
    region = dict(region)
    if 'w' not in region and 'width' in region:
        region['w'] = region.get('width')
    if 'h' not in region and 'height' in region:
        region['h'] = region.get('height')
    item['region'] = region
    if item.get('region_norm') is not None:
        return item
    item['region_norm'] = None
    x, y = region.get('x'), region.get('y')
    w, h = region.get('w'), region.get('h')
    if not all(_number(value) for value in (x, y, w, h)):
        return item
    size = _size(frame_size)
    if size is None:
        return item
    width, height = size
    item['region_norm'] = {
        'x': round(float(x) / width, 6),
        'y': round(float(y) / height, 6),
        'w': round(float(w) / width, 6),
        'h': round(float(h) / height, 6),
    }
    return item


def ensure_ocr_regions(items, frame_sizes=None):
    """Apply :func:`ensure_item_region` to every item; never raises on gaps."""
    frame_sizes = frame_sizes or {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        timestamp = item.get('timestamp', item.get('start'))
        size = None
        try:
            size = frame_sizes.get(round(float(timestamp), 6))
        except (TypeError, ValueError, AttributeError):
            size = None
        ensure_item_region(item, size)
    return items


def migrate_ocr_regions(analysis):
    """Read-time migration: old items gain explicit null region fields."""
    if not isinstance(analysis, dict):
        return analysis
    ocr = analysis.get('ocr')
    if isinstance(ocr, dict):
        ocr.setdefault('region_support', dict(REGION_SUPPORT))
        if isinstance(ocr.get('items'), list):
            ocr['items'] = [ensure_item_region(dict(item)) if isinstance(item, dict) else item
                            for item in ocr['items']]
    analysis.setdefault('ocr_conflicts', [])
    return analysis
