"""Subtitle-band change trigger.

Instead of OCRing every sampled frame, a coarse pass reads only the bottom
``band`` of the frame.  Each band image is downscaled to a small grayscale
signature (64x16 by default); a change in that signature schedules a short
densified window around the later timestamp for the second OCR pass.

The trigger is *visual* on purpose.  Subtitle templates that swap one word or
one digit ("slow line 0" -> "slow line 3") differ by only a few characters, so
an edit-distance threshold misses them.  OCR text is compared in
:func:`duplicate_text_timestamps` for de-duplication only: it can suppress a
densified pass when both sides read as the same sentence, but it never decides
that something changed.

The module contains no OCR backend: callers supply images and texts.  This
keeps the trigger deterministic and testable on synthetic media.
"""
import io
import math
import subprocess
import unicodedata
from difflib import SequenceMatcher

DEFAULT_BAND = 0.15
DEFAULT_SIGNATURE_SIZE = (64, 16)
# Mean absolute grey-level difference, normalized to 0..1 (> 2/255).
DEFAULT_MEAN_ABS_DIFF_THRESHOLD = 2.0 / 255.0
# Per-pixel grey delta that counts as a changed pixel, and how many changed
# pixels count as a visual change (a Hamming-like signal on the downscaled band).
DEFAULT_CHANGED_PIXEL_DELTA = 8
DEFAULT_CHANGED_PIXEL_COUNT = 2


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _media_duration(media):
    """Probe media duration with ffprobe; raises when it cannot be read."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=nw=1:nk=1', str(media)],
        capture_output=True, text=True, timeout=30)
    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError):
        detail = (result.stderr or '').strip()
        raise RuntimeError('无法读取媒体时长' + (f': {detail}' if detail else '')) from None
    return duration if math.isfinite(duration) and duration > 0 else 0.0


def subtitle_band_times(media, *, interval=30.0, band=DEFAULT_BAND, duration=None):
    """Coarse timestamps for the subtitle band pass.

    Samples start at 0 and advance by ``interval`` seconds.  A near-end sample
    is added when the tail would otherwise be left uncovered or when a short
    clip would produce a single point (two are needed to observe a change).
    Only the bottom ``band`` of the frame is intended to be read by the caller.
    ``duration`` may be supplied to avoid probing the media.
    """
    if not _number(interval) or interval <= 0:
        raise ValueError('interval 必须为正数')
    if not _number(band) or not 0 < band <= 1:
        raise ValueError('band 必须在 (0, 1] 内')
    duration = _media_duration(media) if duration is None else float(duration)
    if not math.isfinite(duration) or duration <= 0:
        return []
    times = []
    current = 0.0
    while current < duration - 1e-9:
        times.append(round(current, 6))
        current += interval
    tail = round(duration - 0.5, 6)
    if duration > 0.5 and tail > (times[-1] if times else -1.0) and (
            len(times) < 2 or duration - times[-1] > interval / 2):
        times.append(tail)
    return times


def normalize_text(value):
    """Full-width to half-width, lower-case, and drop whitespace/punctuation."""
    text = unicodedata.normalize('NFKC', str(value or '')).casefold()
    return ''.join(ch for ch in text
                   if not ch.isspace() and not unicodedata.category(ch).startswith(('P', 'Z', 'C')))


def changed(prev, cur, threshold=0.15):
    """True when normalized edit-distance ratio exceeds ``threshold``.

    Retained for text-only callers; the OCR pipeline no longer uses text as a
    trigger signal (see :func:`signature_changed`).
    """
    if not _number(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold 必须在 0..1')
    before, after = normalize_text(prev), normalize_text(cur)
    if before == after:
        return False
    ratio = SequenceMatcher(None, before, after).ratio()
    return (1.0 - ratio) > threshold


def densify(t, *, window=1.0, step=0.5):
    """Symmetric sampling points around ``t``, inclusive of ``t`` itself."""
    if not _number(t):
        raise ValueError('t 必须为有限秒数')
    if not _number(window) or window < 0:
        raise ValueError('window 必须非负')
    if not _number(step) or step <= 0:
        raise ValueError('step 必须为正数')
    count = int(round(window / step))
    return [round(float(t) + index * step, 6) for index in range(-count, count + 1)]


# ── Subtitle-band image signatures ──

def _signature_size(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = value
        if all(isinstance(part, int) and not isinstance(part, bool) and part > 0
               for part in (width, height)):
            return int(width), int(height)
    raise ValueError('size 必须是两个正整数')


def band_signature(image, *, size=DEFAULT_SIGNATURE_SIZE):
    """Downscale one subtitle-band image to a small grayscale pixel signature.

    Returns ``width * height`` bytes (row-major, 0-255).  The signature lives
    only in memory; it is a change detector, not a package field.  OpenCV is
    preferred for area downscaling, Pillow is the fallback.
    """
    width, height = _signature_size(size)
    data = bytes(image or b'')
    if not data:
        raise ValueError('字幕带图像为空')
    try:
        import cv2
        import numpy as np
    except ImportError:
        cv2 = None
    if cv2 is not None:
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError('无法解码字幕带图像')
        return cv2.resize(decoded, (width, height), interpolation=cv2.INTER_AREA).tobytes()
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError('画面变化触发需要 opencv-python 或 Pillow') from None
    with Image.open(io.BytesIO(data)) as raw:
        return raw.convert('L').resize((width, height), Image.BOX).tobytes()


def band_signatures(frames, *, size=DEFAULT_SIGNATURE_SIZE):
    """Turn ``(timestamp, image_bytes)`` frames into ``(timestamp, signature)``."""
    ordered = sorted(((float(timestamp), image) for timestamp, image in frames or ()),
                     key=lambda item: item[0])
    return [(timestamp, band_signature(image, size=size)) for timestamp, image in ordered]


def signature_difference(prev, cur, *, pixel_delta=DEFAULT_CHANGED_PIXEL_DELTA):
    """Mean absolute grey difference and changed-pixel count for two signatures."""
    if not isinstance(pixel_delta, int) or isinstance(pixel_delta, bool) or pixel_delta < 0:
        raise ValueError('pixel_delta 必须为非负整数')
    before, after = bytes(prev or b''), bytes(cur or b'')
    if not before or len(before) != len(after):
        raise ValueError('签名必须非空且长度相同')
    total = 0
    changed_pixels = 0
    for left, right in zip(before, after):
        difference = abs(left - right)
        total += difference
        if difference > pixel_delta:
            changed_pixels += 1
    return {'mean_abs_diff': total / len(before) / 255.0,
            'changed_pixels': changed_pixels}


def signature_changed(prev, cur, *, mean_abs_threshold=DEFAULT_MEAN_ABS_DIFF_THRESHOLD,
                      pixel_delta=DEFAULT_CHANGED_PIXEL_DELTA,
                      changed_pixel_count=DEFAULT_CHANGED_PIXEL_COUNT):
    """True when the band signature moved enough to warrant a densified pass.

    Either signal is sufficient: mean absolute difference above
    ``mean_abs_threshold`` (default 2/255) or more than ``changed_pixel_count``
    pixels whose grey level moved by more than ``pixel_delta``.
    """
    if not _number(mean_abs_threshold) or not 0 <= mean_abs_threshold <= 1:
        raise ValueError('mean_abs_threshold 必须在 0..1')
    if not isinstance(changed_pixel_count, int) or isinstance(changed_pixel_count, bool) \
            or changed_pixel_count < 0:
        raise ValueError('changed_pixel_count 必须为非负整数')
    difference = signature_difference(prev, cur, pixel_delta=pixel_delta)
    return (difference['mean_abs_diff'] > float(mean_abs_threshold)
            or difference['changed_pixels'] > changed_pixel_count)


def signature_change_pairs(signatures, *, mean_abs_threshold=DEFAULT_MEAN_ABS_DIFF_THRESHOLD,
                           pixel_delta=DEFAULT_CHANGED_PIXEL_DELTA,
                           changed_pixel_count=DEFAULT_CHANGED_PIXEL_COUNT):
    """``(previous_timestamp, timestamp)`` pairs whose band images differ."""
    ordered = sorted(((float(timestamp), bytes(signature)) for timestamp, signature in signatures or ()),
                     key=lambda item: item[0])
    return [(previous_timestamp, timestamp)
            for (previous_timestamp, previous), (timestamp, current)
            in zip(ordered, ordered[1:])
            if signature_changed(previous, current,
                                 mean_abs_threshold=mean_abs_threshold,
                                 pixel_delta=pixel_delta,
                                 changed_pixel_count=changed_pixel_count)]


def count_signature_changes(signatures, *, mean_abs_threshold=DEFAULT_MEAN_ABS_DIFF_THRESHOLD,
                            pixel_delta=DEFAULT_CHANGED_PIXEL_DELTA,
                            changed_pixel_count=DEFAULT_CHANGED_PIXEL_COUNT):
    """Number of consecutive coarse frames judged as a visual subtitle change.

    This is the observability counterpart of
    :func:`triggered_times_from_signatures`; it counts image transitions only,
    independently of how many densified points (if any) were scheduled.
    """
    return len(signature_change_pairs(signatures,
                                      mean_abs_threshold=mean_abs_threshold,
                                      pixel_delta=pixel_delta,
                                      changed_pixel_count=changed_pixel_count))


def duplicate_text_timestamps(signatures, texts, *,
                              mean_abs_threshold=DEFAULT_MEAN_ABS_DIFF_THRESHOLD,
                              pixel_delta=DEFAULT_CHANGED_PIXEL_DELTA,
                              changed_pixel_count=DEFAULT_CHANGED_PIXEL_COUNT):
    """Later timestamps of image changes whose OCR reads as the same sentence.

    Text comparison is used here *only* for de-duplication: the image
    signature already decided that a visual change occurred, and identical
    normalized OCR text on both sides means there is no new sentence to record.
    Empty OCR on either side is deliberately not a duplicate -- unreadable
    subtitles must still be densified.
    """
    duplicates = set()
    for previous_timestamp, timestamp in signature_change_pairs(
            signatures, mean_abs_threshold=mean_abs_threshold,
            pixel_delta=pixel_delta, changed_pixel_count=changed_pixel_count):
        before = normalize_text((texts or {}).get(round(previous_timestamp, 6), ''))
        after = normalize_text((texts or {}).get(round(timestamp, 6), ''))
        if before and before == after:
            duplicates.add(round(timestamp, 6))
    return duplicates


def triggered_times_from_signatures(signatures, *,
                                    mean_abs_threshold=DEFAULT_MEAN_ABS_DIFF_THRESHOLD,
                                    pixel_delta=DEFAULT_CHANGED_PIXEL_DELTA,
                                    changed_pixel_count=DEFAULT_CHANGED_PIXEL_COUNT,
                                    window=1.0, step=0.5, duration=None, existing=(),
                                    skip_timestamps=()):
    """New timestamps to OCR after a coarse band-signature pass.

    ``signatures`` is an iterable of ``(timestamp, signature)`` pairs.  A
    signature change schedules :func:`densify` around the later timestamp;
    points already in ``existing``, listed in ``skip_timestamps`` (text
    de-duplication), or outside ``[0, duration)`` are omitted.
    """
    seen = set()
    for value in existing or ():
        try:
            seen.add(round(float(value), 6))
        except (TypeError, ValueError):
            continue
    skipped = set()
    for value in skip_timestamps or ():
        try:
            skipped.add(round(float(value), 6))
        except (TypeError, ValueError):
            continue
    scheduled = set()
    for _, timestamp in signature_change_pairs(
            signatures, mean_abs_threshold=mean_abs_threshold,
            pixel_delta=pixel_delta, changed_pixel_count=changed_pixel_count):
        if round(timestamp, 6) in skipped:
            continue
        for point in densify(timestamp, window=window, step=step):
            if point < 0 or (duration is not None and point >= float(duration)):
                continue
            if round(point, 6) in seen:
                continue
            scheduled.add(round(point, 6))
    return sorted(scheduled)


# ── Legacy text-only trigger helpers ──
# Kept for old callers/evidence scripts.  The OCR pipeline now uses band-image
# signatures; text is compared only by ``duplicate_text_timestamps`` for dedup.

def _observation(value):
    if isinstance(value, dict):
        return float(value.get('timestamp', value.get('start'))), str(value.get('text') or '')
    timestamp, text = value
    return float(timestamp), str(text or '')


def count_changes(observations, *, threshold=0.15):
    """Deprecated text-only change count (not used by the OCR pipeline)."""
    if not _number(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold 必须在 0..1')
    ordered = sorted((_observation(value) for value in observations or ()), key=lambda item: item[0])
    return sum(1 for (_, previous), (_, current) in zip(ordered, ordered[1:])
               if changed(previous, current, threshold))


def triggered_times(observations, *, threshold=0.15, window=1.0, step=0.5,
                    duration=None, existing=()):
    """Deprecated text-only densify scheduling (not used by the OCR pipeline)."""
    seen = set()
    for value in existing or ():
        try:
            seen.add(round(float(value), 6))
        except (TypeError, ValueError):
            continue
    ordered = sorted((_observation(value) for value in observations or ()), key=lambda item: item[0])
    scheduled = set()
    for (_, previous), (timestamp, current) in zip(ordered, ordered[1:]):
        if not changed(previous, current, threshold):
            continue
        for point in densify(timestamp, window=window, step=step):
            if point < 0 or (duration is not None and point >= float(duration)):
                continue
            if round(point, 6) in seen:
                continue
            scheduled.add(round(point, 6))
    return sorted(scheduled)


def extract_band_frames(media, times, *, band=DEFAULT_BAND, max_width=1920, timeout=60,
                        dropped=None):
    """Decode one bottom-band JPEG per timestamp via ffmpeg.

    The crop happens inside ffmpeg, so the full frame never reaches the OCR
    backend.  Individual failures are never silent: when ``dropped`` is a list,
    one ``{"timestamp": t, "reason": "ffmpeg_timeout"|"ffmpeg_failed"}`` entry is
    appended for every sampling point that produced no frame.  The call raises
    only when no frame at all could be decoded (the attached ``dropped`` and
    ``attempted`` attributes still describe the failed pass).
    """
    if not _number(band) or not 0 < band <= 1:
        raise ValueError('band 必须在 (0, 1] 内')
    if max_width is not None and (not _number(max_width) or max_width < 0):
        raise ValueError('max_width 必须非负')
    times = [float(timestamp) for timestamp in times]
    frames, errors, drops = [], [], []
    focus = f'{float(band):g}'
    offset = f'{1.0 - float(band):g}'
    filters = [f'crop=iw:ih*{focus}:0:ih*{offset}']
    if max_width:
        filters.append(f'scale=min(iw\\,{int(max_width)}):-2')

    def record_drop(timestamp, reason):
        entry = {'timestamp': round(float(timestamp), 6), 'reason': reason}
        drops.append(entry)
        if dropped is not None:
            dropped.append(entry)

    for timestamp in times:
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(timestamp), '-i', str(media),
                   '-frames:v', '1', '-an', '-vf', ','.join(filters),
                   '-f', 'image2pipe', '-c:v', 'mjpeg', '-q:v', '4', 'pipe:1']
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout)
            if result.returncode or not result.stdout:
                detail = (result.stderr or b'').decode(errors='replace').strip()
                raise RuntimeError(detail or f'ffmpeg exit {result.returncode}')
            frames.append((float(timestamp), result.stdout))
        except subprocess.TimeoutExpired as exc:
            record_drop(timestamp, 'ffmpeg_timeout')
            errors.append(f'{timestamp}: timeout after {exc.timeout}s')
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            record_drop(timestamp, 'ffmpeg_failed')
            errors.append(f'{timestamp}: {exc}')
    if times and not frames:
        error = RuntimeError('字幕带帧抽取失败: ' + '; '.join(errors))
        error.dropped = list(drops)
        error.attempted = len(times)
        raise error
    return frames
