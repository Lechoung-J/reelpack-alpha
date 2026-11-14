"""Mark OCR/transcript text disagreements without rewriting either side."""
import math
from difflib import SequenceMatcher

DEFAULT_THRESHOLD = 0.6
DEFAULT_MIN_CHARS = 8


def _bounds(item):
    if not isinstance(item, dict):
        return None
    start = item.get('start', item.get('timestamp'))
    end = item.get('end', start)
    try:
        start = float(start)
        end = float(end) if end is not None else start
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    return (min(start, end), max(start, end))


def _overlap(left, right):
    """Overlap interval, treating a zero-length item as a point observation."""
    lo, hi = max(left[0], right[0]), min(left[1], right[1])
    if hi > lo:
        return (lo, hi)
    if left[0] == left[1] and right[0] <= left[0] <= right[1]:
        return (left[0], left[0])
    if right[0] == right[1] and left[0] <= right[0] <= left[1]:
        return (right[0], right[0])
    return None


def find_conflicts(ocr_items, segments, *, threshold=DEFAULT_THRESHOLD,
                   min_chars=DEFAULT_MIN_CHARS):
    """Return one record per overlapping OCR/transcript pair that disagrees.

    Similarity is :class:`difflib.SequenceMatcher` over the stripped texts.
    Only pairs whose similarity is below ``threshold`` and where both texts
    have at least ``min_chars`` characters are recorded.  Inputs are never
    modified.
    """
    if not (isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
            and math.isfinite(threshold) and 0 <= threshold <= 1):
        raise ValueError('threshold 必须在 0..1')
    if not isinstance(min_chars, int) or isinstance(min_chars, bool) or min_chars < 0:
        raise ValueError('min_chars 必须为非负整数')
    conflicts = []
    for ocr in ocr_items or []:
        if not isinstance(ocr, dict):
            continue
        ocr_bounds = _bounds(ocr)
        ocr_text = ocr.get('text')
        if ocr_bounds is None or not isinstance(ocr_text, str):
            continue
        if len(ocr_text.strip()) < min_chars:
            continue
        for segment in segments or []:
            if not isinstance(segment, dict):
                continue
            segment_bounds = _bounds(segment)
            segment_text = segment.get('text')
            if segment_bounds is None or not isinstance(segment_text, str):
                continue
            if len(segment_text.strip()) < min_chars:
                continue
            overlap = _overlap(ocr_bounds, segment_bounds)
            if overlap is None:
                continue
            similarity = SequenceMatcher(None, ocr_text.strip(), segment_text.strip()).ratio()
            if similarity >= threshold:
                continue
            conflicts.append({
                'start': round(overlap[0], 3),
                'end': round(overlap[1], 3),
                'ocr_text': ocr_text,
                'transcript_text': segment_text,
                'similarity': round(similarity, 4),
            })
    conflicts.sort(key=lambda item: (item['start'], item['end'], item['similarity']))
    return conflicts
