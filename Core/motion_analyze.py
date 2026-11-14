#!/usr/bin/env python3
"""Quantitative image-motion evidence for a video interval or supplied shot intervals.

This module measures the dominant global image-plane similarity transform (translation,
scale, rotation) between sampled frames with Lucas-Kanade optical flow and a RANSAC fit.
It reports *evidence*, not conclusions: velocities, accelerations, zoom/rotation rates,
fit quality and residual local motion. The downstream multimodal Agent interprets it.

The global-motion measurement is adapted from JuneYaooo/rea-video
(rea_video/features.py, commit 3974260fb52476b453b5992f667e7dc1dd83c6eb, Apache-2.0,
Copyright 2026 JuneYaooo). Adapted and changed for timestamped quantitative evidence;
see third_party/rea-video/UPSTREAM.md and docs/motion-evidence.md.

Public API
----------
analyze_motion(media_path, *, start=0.0, end=None, shots=None, sample_fps=8.0,
               analysis_width=480, max_seconds=120.0, ...) -> dict

Direct CLI
----------
python3 Core/motion_analyze.py INPUT --start 0 --end 3 --sample-fps 8 \
        --analysis-width 480 --max-seconds 120

The module imports cv2/numpy lazily so ``--help`` and the missing-dependency status work
on an interpreter without OpenCV. No dependencies beyond OpenCV, NumPy, the standard
library and ffmpeg/ffprobe are used. Inputs are read-only; no cache is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

SCHEMA = "reelpack.motion-evidence"
SCHEMA_VERSION = 1
BACKEND = "opencv-lk-ransac-similarity-v1"
MODULE_VERSION = "1.0.0"

DEFAULT_SAMPLE_FPS = 8.0
DEFAULT_ANALYSIS_WIDTH = 480
DEFAULT_MAX_SECONDS = 120.0
DEFAULT_SMOOTHING_WINDOW = 5
MAX_SMOOTHING_WINDOW = 100000
DEFAULT_HEAD_TAIL_SECONDS = 1.0
MAX_FRAME_CAP = 4000
MAX_ANALYSIS_WIDTH = 1920
MIN_ANALYSIS_WIDTH = 64

# The decode wall deadline covers process startup, pipe reads and process exit.
# It is deliberately generous by default; tests lower these private values so
# stalled-decoder regressions finish in seconds instead of minutes.
DECODE_TIMEOUT_MIN_SECONDS = 60.0
DECODE_TIMEOUT_SECONDS_PER_MEDIA_SECOND = 20.0
DECODE_TIMEOUT_MAX_SECONDS = 1800.0
DECODE_CLEANUP_SECONDS = 5.0
DECODE_FRAME_QUEUE_SIZE = 2

MIN_TRACKS = 12              # tracked corners needed before a RANSAC fit is attempted
MIN_INLIER_COUNT = 10        # inliers needed for a pair to count as valid
MIN_INLIER_FRACTION = 0.40   # inlier share of tracked points needed for validity
FB_MAX_ERROR_PX = 1.0        # forward/backward track consistency limit
RANSAC_REPROJ_PX = 1.0
RANSAC_MAX_ITERS = 2000
FEATURE_MAX_CORNERS = 300
FEATURE_QUALITY = 0.01
FEATURE_MIN_DISTANCE = 7
FEATURE_BLOCK_SIZE = 7
LK_WINDOW = 21
LK_MAX_LEVEL = 3
FLAT_STD = 2.0               # below this a frame carries no usable luma structure
FLAT_GRADIENT = 0.5          # mean Sobel magnitude below this is flat
LOCAL_RESIDUAL_PX = 1.5      # residual considered local (foreground-ish) motion
PHOTOMETRIC_MARGIN = 0.08    # ignore 8% of each border in the compensated residual

_PTS_TIME_RE = re.compile(r"pts_time:\s*(-?[0-9]+(?:\.[0-9]+)?)")

UNITS = {
    "time": "seconds on the source presentation timeline used by ffmpeg -ss",
    "vx_frame_width_s": "frame-widths per second: (center dx in analysis px) / frame_width / delta_t; +x = image content moves right",
    "vy_frame_height_s": "frame-heights per second: (center dy in analysis px) / frame_height / delta_t; +y = image content moves down",
    "speed_normalized_frame_s": "normalized frame units per second: sqrt(vx^2 + vy^2) (each axis divided by its own frame dimension)",
    "speed_px_s": "analysis-frame pixels per second at the recorded analysis_width x analysis_height",
    "zoom_log_scale_s": "natural-log scale rate: log(scale) / delta_t; positive = image content is expanding (zoom in)",
    "rotation_deg_s": "degrees per second; positive = clockwise in display coordinates (x right, y down)",
    "acceleration": "frame-widths/s^2 in x, frame-heights/s^2 in y; magnitude = sqrt(ax^2 + ay^2) in normalized frame units/s^2",
    "translation_px": "pixels in the analysis frame; raw affine translation is at the image origin, center displacement is measured at the frame center",
    "residual_px": "pixels in the analysis frame",
    "inlier_fraction": "fraction of forward/backward-valid tracked corners accepted by the RANSAC similarity fit",
}

SIGN_CONVENTION = (
    "Image coordinates: x increases right, y increases down, as displayed after applying source "
    "rotation metadata. Translation, velocity and acceleration describe IMAGE CONTENT movement, "
    "not camera travel: content moving right gives positive vx. The similarity transform is fitted "
    "as prev -> next. Center displacement removes the component that a zoom/rotation about the "
    "frame center produces at the image origin, so centered zoom does not appear as lateral motion. "
    "Positive rotation is clockwise on screen. Positive zoom rate means content expands."
)

LIMITATIONS = [
    "Measured in the image plane only: values are not physical camera speed, focal length, depth or a calibrated probability.",
    "local_motion/residual fields are evidence relative to the fitted dominant similarity transform, not foreground/background segmentation.",
    "No actor identity, object category, scene semantics or camera-travel direction is inferred.",
    "Times come from ffmpeg's presentation timeline. Variable-frame-rate sources are resampled by the documented fps filter; the recorded sample timestamps and spacing are the resampled values mapped back to source seconds.",
    "Translation is measured in analysis-frame pixels at the recorded analysis_width/height, then normalized by displayed width/height.",
    "Square pixels are assumed: anamorphic sample-aspect-ratio sources are scaled to the coded aspect ratio after rotation metadata is applied.",
    "Automatic shot/cut detection is not performed. No-cut-crossing guarantees apply only to caller-supplied shot boundaries and to invalid gaps represented in the sampled pairs; a cut inside an unsplit interval can still produce a cross-cut pair and is an input limitation, not evidence of continuity.",
    "Coverage is reported at two levels: decoded/sampled coverage and measured pair coverage. The final sampling quantum before the requested end is not claimed as observed; exact measured start/end/duration are in coverage.measured.",
]

# ffprobe fields requested in one call. Rotation is read from display-matrix side data and tags.
_PROBE_ENTRIES = (
    "stream=index,codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,start_time,duration,"
    "nb_frames,sample_aspect_ratio:stream_side_data=rotation:stream_tags=rotate"
)
_PROBE_FORMAT = "format=format_name,duration,start_time,size"


class MotionError(Exception):
    """An input or media error that becomes a structured status, never an exception."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------------------
# Small numeric / text helpers
# --------------------------------------------------------------------------------------

def _finite(value):
    """Return a finite float or None. JSON must never contain NaN/Infinity."""
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    # Keep request metadata JSON-serializable even for adversarial huge integers.
    if not -(1 << 63) <= number <= (1 << 63) - 1:
        return None
    return number


def _fraction(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return _finite(float(numerator) / float(denominator))


def _decode_timeout_seconds(duration):
    """Wall-clock allowance for one decode: startup, pipe reads and process exit."""
    duration = _finite(duration)
    if duration is None or duration < 0:
        duration = 0.0
    return max(DECODE_TIMEOUT_MIN_SECONDS,
               min(DECODE_TIMEOUT_MAX_SECONDS,
                   DECODE_TIMEOUT_MIN_SECONDS
                   + duration * DECODE_TIMEOUT_SECONDS_PER_MEDIA_SECOND))


def _parse_rate(text):
    """Parse ffprobe 'num/den' or plain rate text into a float or None."""
    if text in (None, "", "0/0"):
        return None
    try:
        if "/" in str(text):
            num, den = str(text).split("/", 1)
            if float(den) == 0:
                return None
            return _finite(float(num) / float(den))
        return _finite(float(text))
    except (TypeError, ValueError):
        return None


def _median(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None
    values.sort()
    middle = len(values) // 2
    if len(values) % 2:
        return _finite(values[middle])
    return _finite((values[middle - 1] + values[middle]) / 2.0)


def _percentile(values, fraction):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None
    values.sort()
    if len(values) == 1:
        return _finite(values[0])
    position = max(0.0, min(1.0, fraction)) * (len(values) - 1)
    lower = int(math.floor(position))
    upper = min(len(values) - 1, lower + 1)
    weight = position - lower
    return _finite(values[lower] * (1.0 - weight) + values[upper] * weight)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize(value):
    """Recursively make a structure JSON-safe: finite numbers or None, no numpy scalars."""
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _warn(result, code, message):
    result["warnings"].append({"code": code, "message": message})


def _error(result, code, message, status="failed"):
    result["status"] = status
    result["reason"] = code
    result["errors"].append({"code": code, "message": message})


# --------------------------------------------------------------------------------------
# Result scaffolding
# --------------------------------------------------------------------------------------

def _empty_coverage(start, end):
    return {
        "requested": {"start": _finite(start), "end": _finite(end),
                      "duration": _finite((end - start) if start is not None and end is not None else None)},
        "clip_to_media": None,
        "budget": {"max_seconds": None, "max_frames": MAX_FRAME_CAP,
                   "max_pairs": MAX_FRAME_CAP - 1},
        "analyzed": {"start": None, "end": None, "duration": None},
        "decoded": {"start": None, "end": None, "duration": None,
                    "envelope_duration": None, "sample_count": 0,
                    "complete": False, "unobserved_tail_seconds": None,
                    "reasons": []},
        "measured": {"start": None, "end": None, "duration": None,
                     "envelope_duration": None, "ranges": [],
                     "pair_count": 0, "valid_pair_count": 0, "complete": False,
                     "unobserved_tail_seconds": None, "reasons": []},
        "measurement_gaps": [],
        "components": [],
        "incomplete_reasons": [],
        "fully_covered": False,
        "truncation_reasons": [],
    }


def _empty_timeline():
    return {
        "timestamp_source": None,
        "analysis_fps": None,
        "frame_width": None,
        "frame_height": None,
        "sample_count": 0,
        "frame_times_seconds": [],
        "timestamp_validation": {"status": "not_run", "max_deviation_seconds": None,
                                 "pts_samples": 0, "frames_decoded": 0},
    }


COVERAGE_INTERVAL_EPSILON = 1e-6


def _merge_interval_union(intervals):
    """Return sorted, non-overlapping finite positive-duration intervals."""
    clean = []
    for start, end in intervals:
        start, end = _finite(start), _finite(end)
        if start is None or end is None or end - start <= COVERAGE_INTERVAL_EPSILON:
            continue
        clean.append((start, end))
    clean.sort()
    merged = []
    for start, end in clean:
        if merged and start <= merged[-1][1] + COVERAGE_INTERVAL_EPSILON:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _clip_interval_union(intervals, start, end):
    """Clip and merge intervals to one component/request scope."""
    scope_start, scope_end = _finite(start), _finite(end)
    if (scope_start is None or scope_end is None
            or scope_end - scope_start <= COVERAGE_INTERVAL_EPSILON):
        return []
    clipped = []
    for item_start, item_end in intervals:
        clipped_start = max(item_start, scope_start)
        clipped_end = min(item_end, scope_end)
        if clipped_end - clipped_start > COVERAGE_INTERVAL_EPSILON:
            clipped.append((clipped_start, clipped_end))
    return _merge_interval_union(clipped)


def _interval_union_duration(intervals):
    if not intervals:
        return 0.0
    return _finite(sum(end - start for start, end in intervals))


def _interval_gaps(intervals, start, end):
    """Return uncovered [start, end] ranges inside one measured scope."""
    scope_start, scope_end = _finite(start), _finite(end)
    if (scope_start is None or scope_end is None
            or scope_end - scope_start <= COVERAGE_INTERVAL_EPSILON):
        return []
    gaps = []
    cursor = scope_start
    for item_start, item_end in intervals:
        if item_start > cursor + COVERAGE_INTERVAL_EPSILON:
            gaps.append((cursor, item_start))
        if item_end > cursor:
            cursor = item_end
    if scope_end > cursor + COVERAGE_INTERVAL_EPSILON:
        gaps.append((cursor, scope_end))
    return gaps


def _serialize_intervals(intervals):
    return [{"start": _finite(start), "end": _finite(end),
             "duration": _finite(end - start)} for start, end in intervals]


def _base_result(path, request):
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND,
        "backend_version": MODULE_VERSION,
        "status": "failed",
        "reason": None,
        "source": {
            "path": str(path) if path is not None else None,
            "resolved_path": None,
            "digest_sha256": None,
            "size_bytes": None,
            "duration_seconds": None,
            "start_time": None,
            "width": None,
            "height": None,
            "rotation_degrees": 0,
            "display_width": None,
            "display_height": None,
            "codec": None,
            "pix_fmt": None,
            "format_name": None,
            "fps": {"r_frame_rate": None, "avg_frame_rate": None,
                    "nominal": None, "variable": None},
        },
        "request": request,
        "parameters": {
            "sample_fps": None,
            "analysis_width": None,
            "analysis_height": None,
            "max_seconds": None,
            "max_frames": MAX_FRAME_CAP,
            "max_pairs": MAX_FRAME_CAP - 1,
            "min_tracks": MIN_TRACKS,
            "feature_detector": {"max_corners": FEATURE_MAX_CORNERS,
                                 "quality_level": FEATURE_QUALITY,
                                 "min_distance": FEATURE_MIN_DISTANCE,
                                 "block_size": FEATURE_BLOCK_SIZE},
            "optical_flow": {"window_px": LK_WINDOW, "max_level": LK_MAX_LEVEL,
                             "forward_backward_max_error_px": FB_MAX_ERROR_PX},
            "ransac": {"model": "similarity", "reprojection_threshold_px": RANSAC_REPROJ_PX,
                       "max_iterations": RANSAC_MAX_ITERS,
                       "min_inlier_count": MIN_INLIER_COUNT,
                       "min_inlier_fraction": MIN_INLIER_FRACTION},
            "smoothing": {"method": "centered_moving_average",
                          "window_samples": DEFAULT_SMOOTHING_WINDOW,
                          "scope": "within valid contiguous samples only; no gap bridging"},
            "acceleration": {"method": "central_difference",
                             "scope": "interior of valid contiguous samples only"},
            "shot_head_tail_seconds": DEFAULT_HEAD_TAIL_SECONDS,
        },
        "units": UNITS,
        "sign_convention": SIGN_CONVENTION,
        "coverage": _empty_coverage(None, None),
        "timeline": _empty_timeline(),
        "pairs": [],
        "series": [],
        "segments": [],
        "shots": None,
        "summary": {},
        "limitations": list(LIMITATIONS),
        "warnings": [],
        "errors": [],
    }


# --------------------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------------------

def _probe_media(path):
    try:
        probe = subprocess.run(
            [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", _PROBE_ENTRIES, "-show_entries", _PROBE_FORMAT,
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise MotionError("ffprobe_missing", "ffprobe was not found on PATH")
    except subprocess.SubprocessError as exc:
        raise MotionError("probe_failed", f"ffprobe failed: {exc}")
    if probe.returncode != 0:
        detail = (probe.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        raise MotionError("probe_failed", f"ffprobe could not read the media: {detail[0][:500]}")
    try:
        payload = json.loads(probe.stdout or "{}")
    except json.JSONDecodeError:
        raise MotionError("probe_failed", "ffprobe returned unparseable metadata")
    streams = payload.get("streams") or []
    if not streams:
        raise MotionError("no_video_stream", "the file contains no video stream")
    stream = streams[0]
    width = _int_or_none(stream.get("width"))
    height = _int_or_none(stream.get("height"))
    if not width or not height:
        raise MotionError("invalid_media", "video dimensions are missing or zero")

    rotation = 0
    for side_data in stream.get("side_data_list") or []:
        if side_data.get("rotation") is not None:
            try:
                rotation = int(round(float(side_data["rotation"])))
            except (TypeError, ValueError):
                rotation = 0
            break
    if not rotation and stream.get("tags", {}).get("rotate"):
        try:
            rotation = int(round(float(stream["tags"]["rotate"])))
        except (TypeError, ValueError):
            rotation = 0
    rotated = abs(rotation) % 180 == 90
    display_width, display_height = (height, width) if rotated else (width, height)

    r_rate = _parse_rate(stream.get("r_frame_rate"))
    avg_rate = _parse_rate(stream.get("avg_frame_rate"))
    nominal = avg_rate or r_rate
    variable = None
    if r_rate and avg_rate and avg_rate > 0:
        variable = bool(abs(r_rate - avg_rate) / avg_rate > 0.01)

    fmt = payload.get("format") or {}
    start_time = _finite(stream.get("start_time"))
    if start_time is None:
        start_time = _finite(fmt.get("start_time")) or 0.0
    duration = _finite(stream.get("duration"))
    if duration is None or duration <= 0:
        duration = _finite(fmt.get("duration"))
    if duration is None or duration <= 0:
        raise MotionError("invalid_media", "the video duration is missing or zero")

    return {
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "format_name": fmt.get("format_name"),
        "width": width,
        "height": height,
        "rotation_degrees": rotation,
        "display_width": display_width,
        "display_height": display_height,
        "start_time": start_time,
        "duration": duration,
        "fps": {"r_frame_rate": stream.get("r_frame_rate"),
                "avg_frame_rate": stream.get("avg_frame_rate"),
                "nominal": nominal, "variable": variable},
        "nb_frames": _int_or_none(stream.get("nb_frames")),
        "size_bytes": _int_or_none(fmt.get("size")),
    }


def _analysis_size(display_width, display_height, analysis_width):
    width = max(MIN_ANALYSIS_WIDTH, int(analysis_width))
    height = max(2, int(round(display_height * width / float(display_width))))
    return width, height


# --------------------------------------------------------------------------------------
# Frame statistics and pair measurement (adapted from rea-video global_motion)
# --------------------------------------------------------------------------------------

def _gray_stats(gray, cv2, np):
    mean = float(gray.mean())
    std = float(gray.std())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = float(np.mean(np.hypot(gx, gy)))
    return {"mean": _finite(mean), "std": _finite(std), "gradient_mean": _finite(gradient)}


def _pair_skeleton(index, from_frame, to_frame):
    return {
        "pair_index": index,
        "shot_index": None,
        "segment_index": None,
        "from_frame": from_frame,
        "to_frame": to_frame,
        "start_time": None,
        "end_time": None,
        "center_time": None,
        "delta_t": None,
        "status": "invalid",
        "reason": None,
        "transform": None,
        "center_displacement_px": None,
        "velocity": None,
        "tracking": {
            "detected_corners": None,
            "forward_valid": None,
            "forward_backward_valid": None,
            "used_points": None,
            "inlier_count": None,
            "inlier_fraction": None,
            "residual_median_px": None,
            "residual_p95_px": None,
            "flow_median_px": None,
            "flow_p90_px": None,
        },
        "quality": {
            "inlier_fraction": None,
            "residual_median_px": None,
            "local_motion_ratio": None,
            "photometric_mad": None,
            "photometric_block_p98": None,
        },
        "frames": {"from": None, "to": None},
        "local_motion_note": (
            "Residual and local-motion values are relative to the fitted dominant similarity "
            "transform; they are not foreground/background segmentation."
        ),
    }


def _invalidate(pair, reason):
    pair["status"] = "invalid"
    pair["reason"] = reason
    return pair


def _measure_pair(prev, cur, prev_stats, cur_stats, cv2, np, width, height):
    """Fit one prev->next similarity transform; never turn a failure into zero motion."""
    pair = _pair_skeleton(None, None, None)
    pair["frames"] = {"from": prev_stats, "to": cur_stats}

    flat = ((prev_stats["std"] is None or prev_stats["std"] < FLAT_STD)
            or (cur_stats["std"] is None or cur_stats["std"] < FLAT_STD))
    low_texture = ((prev_stats["gradient_mean"] is None or prev_stats["gradient_mean"] < FLAT_GRADIENT)
                   or (cur_stats["gradient_mean"] is None or cur_stats["gradient_mean"] < FLAT_GRADIENT))
    if flat or low_texture:
        return _invalidate(pair, "flat_or_low_texture_frame")

    corners = cv2.goodFeaturesToTrack(
        prev, maxCorners=FEATURE_MAX_CORNERS, qualityLevel=FEATURE_QUALITY,
        minDistance=FEATURE_MIN_DISTANCE, blockSize=FEATURE_BLOCK_SIZE)
    detected = 0 if corners is None else int(len(corners))
    pair["tracking"]["detected_corners"] = detected
    if corners is None or detected < MIN_TRACKS:
        return _invalidate(pair, "too_few_features")

    nxt, status, _err = cv2.calcOpticalFlowPyrLK(
        prev, cur, corners, None, winSize=(LK_WINDOW, LK_WINDOW), maxLevel=LK_MAX_LEVEL)
    if nxt is None or status is None:
        return _invalidate(pair, "forward_flow_failed")

    forward_ok = status.reshape(-1).astype(bool)
    tracked = nxt.reshape(-1, 2).astype(np.float32)
    origins = corners.reshape(-1, 2).astype(np.float32)
    finite_tracks = np.isfinite(tracked).all(axis=1)
    forward_ok = forward_ok & finite_tracks
    pair["tracking"]["forward_valid"] = int(forward_ok.sum())
    if int(forward_ok.sum()) < MIN_TRACKS:
        return _invalidate(pair, "too_few_forward_tracks")

    # Backward tracking is only run on points that forward tracking reported as valid.
    # (The upstream implementation tracked backward before checking `nxt_pts`, carrying
    # lost-track garbage into the fit.)
    origins_f = origins[forward_ok].reshape(-1, 1, 2)
    tracked_f = tracked[forward_ok].reshape(-1, 1, 2)
    back, back_status, _err = cv2.calcOpticalFlowPyrLK(
        cur, prev, tracked_f, None, winSize=(LK_WINDOW, LK_WINDOW), maxLevel=LK_MAX_LEVEL)
    if back is None or back_status is None:
        return _invalidate(pair, "backward_flow_failed")
    back_points = back.reshape(-1, 2).astype(np.float32)
    back_ok = back_status.reshape(-1).astype(bool) & np.isfinite(back_points).all(axis=1)
    fb_error = np.linalg.norm(origins_f.reshape(-1, 2) - back_points, axis=1)
    consistent = back_ok & (fb_error < FB_MAX_ERROR_PX)
    pair["tracking"]["forward_backward_valid"] = int(consistent.sum())
    if int(consistent.sum()) < MIN_TRACKS:
        return _invalidate(pair, "forward_backward_inconsistent")

    p0 = origins_f.reshape(-1, 2)[consistent]
    p1 = tracked_f.reshape(-1, 2)[consistent]
    pair["tracking"]["used_points"] = int(len(p0))
    flow = np.linalg.norm(p1 - p0, axis=1)
    pair["tracking"]["flow_median_px"] = _finite(float(np.median(flow)))
    pair["tracking"]["flow_p90_px"] = _finite(float(np.percentile(flow, 90)))

    matrix, inliers = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_PX,
        maxIters=RANSAC_MAX_ITERS, confidence=0.995)
    if matrix is None:
        return _invalidate(pair, "ransac_failed")
    if not np.isfinite(matrix).all():
        return _invalidate(pair, "degenerate_transform")

    predicted = p0 @ matrix[:, :2].T + matrix[:, 2]
    residual = np.linalg.norm(predicted - p1, axis=1)
    if inliers is not None:
        inlier_mask = inliers.reshape(-1).astype(bool)
        inlier_count = int(inlier_mask.sum())
    else:  # OpenCV should always return a mask for RANSAC; residual fallback keeps evidence.
        inlier_mask = residual < RANSAC_REPROJ_PX
        inlier_count = int(inlier_mask.sum())
    inlier_fraction = _fraction(inlier_count, len(p0))
    pair["tracking"]["inlier_count"] = inlier_count
    pair["tracking"]["inlier_fraction"] = inlier_fraction
    pair["tracking"]["residual_median_px"] = _finite(float(np.median(residual)))
    pair["tracking"]["residual_p95_px"] = _finite(float(np.percentile(residual, 95)))
    pair["quality"]["inlier_fraction"] = inlier_fraction
    pair["quality"]["residual_median_px"] = pair["tracking"]["residual_median_px"]

    if inlier_count < MIN_INLIER_COUNT or (inlier_fraction is not None and inlier_fraction < MIN_INLIER_FRACTION):
        return _invalidate(pair, "low_inlier_support")

    a = float(matrix[0, 0])
    b = float(matrix[1, 0])
    tx = float(matrix[0, 2])
    ty = float(matrix[1, 2])
    scale = math.hypot(a, b)
    rotation_deg = math.degrees(math.atan2(b, a))
    if not (0.5 <= scale <= 2.0) or not math.isfinite(rotation_deg):
        return _invalidate(pair, "implausible_transform")

    # Component of the affine map at the frame center. The raw (tx, ty) translation is
    # measured at the image origin and is contaminated by a centered zoom/rotation.
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    dx_center = (a - 1.0) * center_x - b * center_y + tx
    dy_center = b * center_x + (a - 1.0) * center_y + ty

    pair["transform"] = {
        "model": "similarity",
        "matrix_2x3": [[a, -b, tx], [b, a, ty]],
        "scale": _finite(scale),
        "rotation_deg": _finite(rotation_deg),
        "translation_origin_px": {"x": _finite(tx), "y": _finite(ty)},
        "center_px": {"x": _finite(center_x), "y": _finite(center_y)},
    }
    pair["center_displacement_px"] = {"x": _finite(dx_center), "y": _finite(dy_center)}

    # Residual local motion: tracked corners that the dominant similarity does not explain.
    pair["quality"]["local_motion_ratio"] = _fraction(int((residual > LOCAL_RESIDUAL_PX).sum()), len(p0))

    # Photometric residual after compensating the dominant transform.
    warped = cv2.warpAffine(prev, matrix, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    difference = cv2.absdiff(warped, cur)
    margin_y = max(1, int(height * PHOTOMETRIC_MARGIN))
    margin_x = max(1, int(width * PHOTOMETRIC_MARGIN))
    inner = difference[margin_y:height - margin_y, margin_x:width - margin_x]
    if inner.size:
        pair["quality"]["photometric_mad"] = _finite(float(inner.mean()))
        block_h, block_w = inner.shape[0] // 16, inner.shape[1] // 16
        if block_h and block_w:
            blocks = inner[:block_h * 16, :block_w * 16].astype(np.float32)
            blocks = blocks.reshape(block_h, 16, block_w, 16).mean(axis=(1, 3))
            pair["quality"]["photometric_block_p98"] = _finite(float(np.percentile(blocks, 98)))

    pair["status"] = "valid"
    pair["reason"] = None
    return pair


# --------------------------------------------------------------------------------------
# Timestamps, velocity, segments, smoothing and acceleration
# --------------------------------------------------------------------------------------

def _attach_velocity(pair, start_time, end_time, width, height):
    pair["start_time"] = _finite(start_time)
    pair["end_time"] = _finite(end_time)
    if start_time is None or end_time is None:
        return _invalidate(pair, "missing_timestamp")
    pair["center_time"] = _finite((start_time + end_time) / 2.0)
    delta = end_time - start_time
    pair["delta_t"] = _finite(delta)
    if not math.isfinite(delta) or delta <= 0:
        if pair["status"] == "valid":
            return _invalidate(pair, "nonpositive_delta_t")
        return pair
    if pair["status"] != "valid" or pair["center_displacement_px"] is None:
        return pair
    dx = pair["center_displacement_px"]["x"]
    dy = pair["center_displacement_px"]["y"]
    if dx is None or dy is None:
        return _invalidate(pair, "nonfinite_displacement")
    vx = dx / float(width) / delta
    vy = dy / float(height) / delta
    zoom_rate = math.log(pair["transform"]["scale"]) / delta
    rotation_rate = pair["transform"]["rotation_deg"] / delta
    pair["velocity"] = {
        "vx_frame_width_s": _finite(vx),
        "vy_frame_height_s": _finite(vy),
        "speed_normalized_frame_s": _finite(math.hypot(vx, vy)),
        "speed_px_s": _finite(math.hypot(dx, dy) / delta),
        "zoom_log_scale_s": _finite(zoom_rate),
        "rotation_deg_s": _finite(rotation_rate),
    }
    return pair


def _velocity_none():
    return {
        "vx_frame_width_s": None,
        "vy_frame_height_s": None,
        "speed_normalized_frame_s": None,
        "speed_px_s": None,
        "zoom_log_scale_s": None,
        "rotation_deg_s": None,
    }


def _mean_velocity(velocities):
    keys = list(_velocity_none().keys())
    mean = {}
    for key in keys:
        values = [v[key] for v in velocities if v is not None and v.get(key) is not None]
        mean[key] = _finite(sum(values) / len(values)) if values else None
    return mean


def _build_series(pairs, smoothing_window):
    """Build contiguous-valid segments, smoothed velocities, and acceleration.

    Invalid pairs and shot/cut boundaries terminate segments. Smoothing and
    differentiation never bridge a gap and never fill zeros.
    """
    half = max(0, int(smoothing_window) // 2)
    segments = []
    segment = None
    previous = None
    for pair in pairs:
        contiguous = (
            previous is not None
            and previous["status"] == "valid"
            and pair["status"] == "valid"
            and previous["end_time"] is not None
            and pair["start_time"] is not None
            and abs(pair["start_time"] - previous["end_time"]) <= 1e-3
            and previous["shot_index"] == pair["shot_index"]
        )
        if pair["status"] == "valid":
            if not contiguous:
                segment = {
                    "segment_index": len(segments),
                    "shot_index": pair["shot_index"],
                    "first_pair": pair["pair_index"],
                    "last_pair": pair["pair_index"],
                    "pair_count": 0,
                    "start_time": pair["start_time"],
                    "end_time": pair["end_time"],
                }
                segments.append(segment)
            else:
                segment["last_pair"] = pair["pair_index"]
                segment["end_time"] = pair["end_time"]
            segment["pair_count"] += 1
            pair["segment_index"] = segment["segment_index"]
        else:
            pair["segment_index"] = None
            segment = None
        previous = pair

    by_index = {pair["pair_index"]: pair for pair in pairs}
    series = []
    for pair in pairs:
        global_index = pair["pair_index"]
        entry = {
            "pair_index": global_index,
            "shot_index": pair["shot_index"],
            "segment_index": pair["segment_index"],
            "start_time": pair["start_time"],
            "end_time": pair["end_time"],
            "center_time": pair["center_time"],
            "delta_t": pair["delta_t"],
            "status": pair["status"],
            "reason": pair["reason"],
            "velocity_raw": None,
            "velocity_smoothed": None,
            "smoothing": None,
            "acceleration_raw": None,
            "acceleration_smoothed": None,
            "differentiation": {"status": "not_available", "reason": "invalid_pair"},
        }
        if pair["status"] != "valid" or pair["velocity"] is None:
            entry["differentiation"]["reason"] = pair["reason"] or "invalid_pair"
            series.append(entry)
            continue

        entry["velocity_raw"] = dict(pair["velocity"])
        if pair["segment_index"] is None:
            entry["smoothing"] = {"method": "centered_moving_average",
                                  "window_samples": int(smoothing_window),
                                  "used_samples": 0, "quality": "not_available"}
        else:
            segment = segments[pair["segment_index"]]
            window_start = max(segment["first_pair"], global_index - half)
            window_end = min(segment["last_pair"], global_index + half)
            window_pairs = [by_index[index] for index in range(window_start, window_end + 1)
                            if index in by_index and by_index[index]["status"] == "valid"]
            entry["velocity_smoothed"] = _mean_velocity([p["velocity"] for p in window_pairs])
            entry["smoothing"] = {
                "method": "centered_moving_average",
                "window_samples": int(smoothing_window),
                "used_samples": len(window_pairs),
                "quality": "full" if len(window_pairs) >= int(smoothing_window) else "reduced",
            }

        previous_pair = by_index.get(global_index - 1)
        next_pair = by_index.get(global_index + 1)
        interior = (
            previous_pair is not None and next_pair is not None
            and previous_pair["status"] == "valid" and next_pair["status"] == "valid"
            and previous_pair["segment_index"] == pair["segment_index"]
            and next_pair["segment_index"] == pair["segment_index"]
            and previous_pair["center_time"] is not None and next_pair["center_time"] is not None
        )
        if not interior:
            entry["differentiation"] = {"status": "not_available", "reason": "segment_boundary"}
        else:
            span = next_pair["center_time"] - previous_pair["center_time"]
            if not math.isfinite(span) or span <= 0:
                entry["differentiation"] = {"status": "not_available", "reason": "nonpositive_span"}
            else:
                entry["differentiation"] = {"method": "central_difference",
                                            "time_span_seconds": _finite(span)}

                def _accel(before, after):
                    if before is None or after is None:
                        return None
                    components = ("vx_frame_width_s", "vy_frame_height_s")
                    if any(before[key] is None or after[key] is None for key in components):
                        return None
                    ax = (after["vx_frame_width_s"] - before["vx_frame_width_s"]) / span
                    ay = (after["vy_frame_height_s"] - before["vy_frame_height_s"]) / span
                    return {"ax": _finite(ax), "ay": _finite(ay),
                            "magnitude": _finite(math.hypot(ax, ay))}

                if (entry["velocity_raw"] is not None
                        and previous_pair["velocity"] is not None and next_pair["velocity"] is not None):
                    entry["acceleration_raw"] = _accel(previous_pair["velocity"], next_pair["velocity"])
                # The smoothed neighbour values are computed from their valid windows directly
                # so this pass does not depend on entry construction order.
                if pair["segment_index"] is not None:
                    seg = segments[pair["segment_index"]]

                    def _smoothed(index):
                        lo = max(seg["first_pair"], index - half)
                        hi = min(seg["last_pair"], index + half)
                        candidates = [by_index[k]["velocity"] for k in range(lo, hi + 1)
                                      if k in by_index and by_index[k]["status"] == "valid"]
                        return _mean_velocity(candidates) if candidates else None

                    entry["acceleration_smoothed"] = _accel(
                        _smoothed(global_index - 1), _smoothed(global_index + 1))
        series.append(entry)

    # A final segment bookkeeping pass for times/counts.
    for segment in segments:
        members = [pair for pair in pairs if pair["segment_index"] == segment["segment_index"]]
        segment["pair_count"] = len(members)
        segment["start_time"] = _finite(members[0]["start_time"]) if members else None
        segment["end_time"] = _finite(members[-1]["end_time"]) if members else None
        segment["duration"] = _finite((members[-1]["end_time"] - members[0]["start_time"])
                                      if members else None)
    return series, segments


def _summarize(pairs, series_by_index):
    valid = [pair for pair in pairs if pair["status"] == "valid" and pair["velocity"] is not None]
    summary = {
        "status": "measured" if valid else "insufficient_evidence",
        "pair_count": len(pairs),
        "valid_pair_count": len(valid),
        "valid_fraction": _fraction(len(valid), len(pairs)),
        "invalid_reasons": {},
        "velocity": None,
        "zoom": None,
        "rotation": None,
        "acceleration": None,
        "quality": None,
    }
    reasons = {}
    for pair in pairs:
        if pair["status"] != "valid":
            reasons[pair["reason"] or "unknown"] = reasons.get(pair["reason"] or "unknown", 0) + 1
    summary["invalid_reasons"] = reasons
    if not valid:
        return summary

    speed = [pair["velocity"]["speed_normalized_frame_s"] for pair in valid]
    velocity = {
        "median_vx_frame_width_s": _median([pair["velocity"]["vx_frame_width_s"] for pair in valid]),
        "median_vy_frame_height_s": _median([pair["velocity"]["vy_frame_height_s"] for pair in valid]),
        "median_speed_normalized_frame_s": _median(speed),
        "p90_speed_normalized_frame_s": _percentile(speed, 0.90),
        "median_speed_px_s": _median([pair["velocity"]["speed_px_s"] for pair in valid]),
    }
    zoom = {
        "median_log_scale_s": _median([pair["velocity"]["zoom_log_scale_s"] for pair in valid]),
        "median_abs_log_scale_s": _median([abs(pair["velocity"]["zoom_log_scale_s"]) for pair in valid]),
    }
    rotation = {
        "median_deg_s": _median([pair["velocity"]["rotation_deg_s"] for pair in valid]),
        "median_abs_deg_s": _median([abs(pair["velocity"]["rotation_deg_s"]) for pair in valid]),
    }
    accelerations = [series_by_index[pair["pair_index"]]["acceleration_smoothed"]
                     for pair in valid
                     if series_by_index.get(pair["pair_index"], {}).get("acceleration_smoothed")]
    acceleration = None
    if accelerations:
        acceleration = {
            "sample_count": len(accelerations),
            "median_ax": _median([item["ax"] for item in accelerations]),
            "median_ay": _median([item["ay"] for item in accelerations]),
            "median_magnitude": _median([item["magnitude"] for item in accelerations]),
            "p90_magnitude": _percentile([item["magnitude"] for item in accelerations], 0.90),
        }
    quality = {
        "median_inlier_fraction": _median([pair["tracking"]["inlier_fraction"] for pair in valid]),
        "median_residual_px": _median([pair["tracking"]["residual_median_px"] for pair in valid]),
        "median_used_points": _median([pair["tracking"]["used_points"] for pair in valid]),
        "median_local_motion_ratio": _median([pair["quality"]["local_motion_ratio"] for pair in valid]),
        "median_photometric_mad": _median([pair["quality"]["photometric_mad"] for pair in valid]),
    }
    summary.update({"velocity": velocity, "zoom": zoom, "rotation": rotation,
                    "acceleration": acceleration, "quality": quality})
    return summary


# --------------------------------------------------------------------------------------
# Decoding one interval
# --------------------------------------------------------------------------------------

def _drain_stderr(stream, sink):
    """Read ffmpeg stderr without letting a full pipe block decoding."""
    tail = deque(maxlen=120)
    pts = []
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace")
            if "Parsed_showinfo" in line:
                match = _PTS_TIME_RE.search(line)
                if match:
                    try:
                        pts.append(float(match.group(1)))
                    except ValueError:
                        pass
            else:
                tail.append(line.rstrip())
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass
    sink["pts"] = pts
    sink["tail"] = list(tail)


def _queue_put(frame_queue, item, stop_event):
    """Put without ever blocking cleanup indefinitely."""
    while not stop_event.is_set():
        try:
            frame_queue.put(item, timeout=0.05)
            return True
        except queue.Full:
            continue
    return False


def _read_frame_pipe(stream, frame_size, frame_queue, sink, stop_event):
    """Read fixed-size raw frames off the decoder pipe until EOF, error or stop."""
    try:
        while not stop_event.is_set():
            buffer = stream.read(frame_size)
            if not buffer:
                sink["eof"] = True
                _queue_put(frame_queue, ("eof", None), stop_event)
                return
            if len(buffer) != frame_size:
                sink["partial"] = True
                _queue_put(frame_queue, ("partial", len(buffer)), stop_event)
                return
            if not _queue_put(frame_queue, ("frame", buffer), stop_event):
                return
    except (OSError, ValueError) as exc:
        if not stop_event.is_set():
            sink["read_error"] = str(exc)
            _queue_put(frame_queue, ("error", str(exc)), stop_event)


def _decode_interval(resolved_path, start, duration, sample_fps, analysis_width,
                     display_width, display_height, cv2, np, max_frames):
    """Decode a bounded interval streaming, return pair measurements without timestamps.

    The wall deadline starts before process creation and covers pipe reads, frame
    measurement and process exit. A stalled decoder is terminated, then killed if
    needed, and both pipe readers are joined within DECODE_CLEANUP_SECONDS. A cap
    reached by this module is recorded separately from a decoder-initiated failure.
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    frame_width, frame_height = _analysis_size(display_width, display_height, analysis_width)
    frames_planned = int(math.floor(duration * sample_fps)) + 1
    filter_chain = (f"fps={sample_fps:.6f},scale={frame_width}:{frame_height}:flags=bicubic,"
                    f"setsar=1,showinfo")
    command = [
        ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info",
        "-ss", f"{start:.6f}", "-i", str(resolved_path), "-t", f"{duration:.6f}",
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", filter_chain, "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    deadline = time.monotonic() + _decode_timeout_seconds(duration)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sink = {}
    frame_queue = queue.Queue(maxsize=DECODE_FRAME_QUEUE_SIZE)
    stop_event = threading.Event()
    stdout_reader = threading.Thread(
        target=_read_frame_pipe,
        args=(process.stdout, frame_width * frame_height, frame_queue, sink, stop_event),
        daemon=True)
    stderr_reader = threading.Thread(
        target=_drain_stderr, args=(process.stderr, sink), daemon=True)
    stdout_reader.start()
    stderr_reader.start()

    frame_size = frame_width * frame_height
    pairs = []
    previous = None
    previous_stats = None
    frames_decoded = 0
    partial_frame = False
    cap_hit = False
    timed_out = False
    reader_error = None
    returncode = None
    terminated = False
    killed = False
    try:
        while True:
            if frames_decoded >= max_frames:
                cap_hit = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                kind, payload = frame_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if kind == "eof":
                break
            if kind == "partial":
                partial_frame = True
                break
            if kind == "error":
                reader_error = payload
                break
            buffer = payload
            if len(buffer) != frame_size:
                partial_frame = True
                break
            gray = np.frombuffer(buffer, dtype=np.uint8).reshape(frame_height, frame_width)
            stats = _gray_stats(gray, cv2, np)
            frame_index = frames_decoded
            if previous is not None:
                pair = _measure_pair(previous, gray, previous_stats, stats, cv2, np,
                                     frame_width, frame_height)
                pair["pair_index"] = len(pairs)
                pair["from_frame"] = frame_index - 1
                pair["to_frame"] = frame_index
                pairs.append(pair)
            previous = gray
            previous_stats = stats
            frames_decoded += 1
            if frames_decoded >= max_frames:
                cap_hit = True
                break

        # After a clean EOF the decoder still has to exit; that wait is inside the
        # same wall deadline. Cap/partial/error paths terminate in cleanup instead.
        if not (cap_hit or partial_frame or timed_out or reader_error):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        if process.poll() is None:
            try:
                process.terminate()
                terminated = True
            except OSError:
                pass
            try:
                returncode = process.wait(timeout=DECODE_CLEANUP_SECONDS)
            except subprocess.TimeoutExpired:
                killed = True
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    returncode = process.wait(timeout=DECODE_CLEANUP_SECONDS)
                except subprocess.TimeoutExpired:
                    returncode = process.poll()
        else:
            returncode = process.poll()
        stop_event.set()
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        stdout_reader.join(timeout=DECODE_CLEANUP_SECONDS)
        stderr_reader.join(timeout=DECODE_CLEANUP_SECONDS)

    readers_finished = not stdout_reader.is_alive() and not stderr_reader.is_alive()
    module_terminated = terminated or killed
    decoder_exit_nonzero = bool(
        not module_terminated and not timed_out and returncode not in (0, None))
    decoder_failed = bool(decoder_exit_nonzero or reader_error)
    intentional_termination = bool(cap_hit and module_terminated)

    pts = list(sink.get("pts") or [])
    stderr_tail = "\n".join(sink.get("tail") or [])
    timestamp_source = "ffmpeg-fps-filter-cfr+showinfo-pts"
    timestamp_validation = {"status": "ok", "max_deviation_seconds": None,
                            "pts_samples": len(pts), "frames_decoded": frames_decoded}
    local_times = None
    if len(pts) >= frames_decoded and frames_decoded:
        expected = [index / sample_fps for index in range(frames_decoded)]
        deviations = [abs(pts[index] - expected[index]) for index in range(frames_decoded)]
        max_deviation = max(deviations) if deviations else 0.0
        timestamp_validation["max_deviation_seconds"] = _finite(max_deviation)
        if pts[0] > duration + max(1.0, duration):
            # Timestamps were not rebased after the input seek; index times are unambiguous.
            local_times = expected
            timestamp_source = "index-derived-after-unrebased-pts"
            timestamp_validation["status"] = "fallback"
        else:
            local_times = pts[:frames_decoded]
            if max_deviation > 0.5 / sample_fps:
                timestamp_validation["status"] = "irregular"
    else:
        local_times = [index / sample_fps for index in range(frames_decoded)]
        timestamp_source = "index-derived-fallback"
        timestamp_validation["status"] = "missing_showinfo_pts"

    for index, pair in enumerate(pairs):
        if index + 1 < len(local_times):
            _attach_velocity(pair, start + local_times[index], start + local_times[index + 1],
                             frame_width, frame_height)
        else:
            _invalidate(pair, "missing_end_timestamp")

    duration_analyzed = None
    if local_times and frames_decoded:
        duration_analyzed = local_times[frames_decoded - 1] - local_times[0]
    truncated = bool(partial_frame or timed_out or decoder_failed
                     or (frames_decoded + 1 < frames_planned)
                     or (cap_hit and frames_decoded < frames_planned))
    decode = {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frames_planned": frames_planned,
        "frames_decoded": frames_decoded,
        "duration_planned": _finite(duration),
        "duration_analyzed": _finite(duration_analyzed),
        "ffmpeg_returncode": returncode,
        "cap_hit": cap_hit,
        "partial_frame": partial_frame,
        "timed_out": timed_out,
        "decoder_exit_nonzero": decoder_exit_nonzero,
        "decoder_failed": decoder_failed,
        "module_terminated": module_terminated,
        "intentional_termination": intentional_termination,
        "read_error": reader_error,
        "truncated": truncated,
        "cleanup": {
            "terminated": terminated,
            "killed": killed,
            "stdout_reader_alive": stdout_reader.is_alive(),
            "stderr_reader_alive": stderr_reader.is_alive(),
            "readers_finished": readers_finished,
        },
        "stderr_tail": stderr_tail[-2000:] if stderr_tail else None,
    }
    return {
        "pairs": pairs,
        "frame_times_absolute": [start + value for value in local_times],
        "decode": decode,
        "timeline": {
            "timestamp_source": timestamp_source,
            "analysis_fps": _finite(sample_fps),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "sample_count": frames_decoded,
            "frame_times_seconds": [_finite(start + value) for value in local_times],
            "timestamp_validation": timestamp_validation,
        },
    }


# --------------------------------------------------------------------------------------
# Shot helpers
# --------------------------------------------------------------------------------------

def _window_pairs(pairs, start, end):
    return [pair for pair in pairs
            if pair["center_time"] is not None and start - 1e-6 <= pair["center_time"] <= end + 1e-6]


def _window(pairs, start, end, series_by_index):
    chosen = _window_pairs(pairs, start, end)
    return {
        "start": _finite(start),
        "end": _finite(end),
        "duration": _finite(end - start),
        "pair_indices": [pair["pair_index"] for pair in chosen],
        "pair_count": len(chosen),
        "valid_pair_count": sum(1 for pair in chosen if pair["status"] == "valid"),
        "summary": _summarize(chosen, series_by_index),
    }


def _shot_intervals(shots, start, end, media_duration):
    """Validate shots and intersect them with the optional overall clip."""
    if not isinstance(shots, (list, tuple)) or not shots:
        raise MotionError("invalid_shots", "shots must be a non-empty list of {start, end}")
    intervals = []
    for index, shot in enumerate(shots):
        if isinstance(shot, dict):
            shot_start, shot_end = shot.get("start"), shot.get("end")
        elif isinstance(shot, (list, tuple)) and len(shot) == 2:
            shot_start, shot_end = shot
        else:
            raise MotionError("invalid_shots", f"shot {index} must be {{start, end}} seconds")
        shot_start, shot_end = _finite(shot_start), _finite(shot_end)
        if shot_start is None or shot_end is None:
            raise MotionError("invalid_shots", f"shot {index} has a non-finite start/end")
        if shot_start < 0:
            raise MotionError("invalid_shots", f"shot {index} starts before 0")
        if shot_end <= shot_start:
            raise MotionError("invalid_shots", f"shot {index} end must be greater than start")
        clip_start = max(shot_start, start)
        clip_end = min(shot_end, end, media_duration)
        intervals.append({
            "shot_index": index,
            "requested_start": shot_start,
            "requested_end": shot_end,
            "start": clip_start,
            "end": clip_end,
            "invalid": clip_end <= clip_start,
        })
    return intervals


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def analyze_motion(media_path, *, start=0.0, end=None, shots=None, sample_fps=DEFAULT_SAMPLE_FPS,
                   analysis_width=DEFAULT_ANALYSIS_WIDTH, max_seconds=DEFAULT_MAX_SECONDS,
                   smoothing_window=DEFAULT_SMOOTHING_WINDOW,
                   head_tail_seconds=DEFAULT_HEAD_TAIL_SECONDS):
    """Measure image motion for one interval or for supplied shot intervals.

    Parameters
    ----------
    media_path : str | os.PathLike
        Local video file. Opened read-only; never modified.
    start, end : float | None
        Interval in seconds on the source presentation timeline. ``end=None`` means the
        media duration. When ``shots`` is supplied, this interval clips every shot.
    shots : list | None
        Shot intervals as dicts ``{"start": sec, "end": sec}`` (``index`` is accepted but
        ignored) or ``(start, end)`` pairs. Each supplied shot is decoded separately, so
        no frame pair crosses a supplied boundary or invalid gap. Automatic cut
        detection is not performed; an unsplit interval may still contain a cut.
    sample_fps : float
        Sampling rate of the analysis timeline (default 8). Frames are resampled by
        ffmpeg's ``fps`` filter and their real output timestamps are recorded.
    analysis_width : int
        Width in pixels of the grayscale analysis frames (default 480). Height preserves
        the displayed aspect ratio after rotation metadata is applied.
    max_seconds : float
        Cumulative analyzed-duration budget (default 120 s). Truncation is reported.
    smoothing_window : int
        Centered moving-average window in samples for smoothed velocity (default 5;
        accepted range 1-100000).
    head_tail_seconds : float
        Per-shot head/tail window length, clipped to the measured shot (default 1.0 s).

    Returns
    -------
    dict
        A JSON-safe evidence document. ``status`` is one of ``complete``, ``partial``,
        ``insufficient_evidence``, ``invalid_input``, ``unavailable`` or ``failed``.
        Every numeric value is finite or ``None``.
    """
    request = {
        "start": _finite(start),
        "end": _finite(end),
        "shots": bool(shots) if shots is not None else False,
        "sample_fps": _finite(sample_fps),
        "analysis_width": _int_or_none(analysis_width),
        "max_seconds": _finite(max_seconds),
        "smoothing_window": _int_or_none(smoothing_window),
        "head_tail_seconds": _finite(head_tail_seconds),
    }
    result = _base_result(media_path, request)

    try:
        if media_path is None:
            raise MotionError("invalid_input", "media_path is required")
        resolved = _resolve_file(media_path)
    except MotionError as exc:
        _error(result, exc.code, exc.message, status="invalid_input")
        return _sanitize(result)

    result["source"]["resolved_path"] = str(resolved)

    # Parameter validation.
    try:
        start = _finite(start)
        if start is None or start < 0:
            raise MotionError("invalid_interval", "start must be a finite number >= 0")
        if end is not None:
            end = _finite(end)
            if end is None:
                raise MotionError("invalid_interval", "end must be a finite number or null")
            if end <= start:
                raise MotionError("invalid_interval", "end must be greater than start")
        sample_fps = _finite(sample_fps)
        if sample_fps is None or not 0 < sample_fps <= 240:
            raise MotionError("invalid_parameters", "sample_fps must be in (0, 240]")
        analysis_width = _int_or_none(analysis_width)
        if analysis_width is None or not MIN_ANALYSIS_WIDTH <= analysis_width <= MAX_ANALYSIS_WIDTH:
            raise MotionError("invalid_parameters",
                              f"analysis_width must be in [{MIN_ANALYSIS_WIDTH}, {MAX_ANALYSIS_WIDTH}]")
        max_seconds = _finite(max_seconds)
        if max_seconds is None or max_seconds <= 0:
            raise MotionError("invalid_parameters", "max_seconds must be a finite positive number")
        smoothing_window = _int_or_none(smoothing_window)
        if smoothing_window is None or not 1 <= smoothing_window <= MAX_SMOOTHING_WINDOW:
            raise MotionError(
                "invalid_parameters",
                f"smoothing_window must be an integer in [1, {MAX_SMOOTHING_WINDOW}]")
        if smoothing_window % 2 == 0:
            smoothing_window += 1
            _warn(result, "smoothing_window_adjusted",
                  f"even smoothing_window rounded up to {smoothing_window} for a centered window")
        head_tail_seconds = _finite(head_tail_seconds)
        if head_tail_seconds is None or head_tail_seconds <= 0:
            raise MotionError("invalid_parameters", "head_tail_seconds must be a finite positive number")
    except MotionError as exc:
        _error(result, exc.code, exc.message, status="invalid_input")
        return _sanitize(result)

    try:
        import numpy as np  # noqa: F401 - lazy dependency
        import cv2
    except ImportError as exc:
        _error(result, "dependencies_missing",
               f"OpenCV and NumPy are required for motion analysis: {exc}", status="unavailable")
        return _sanitize(result)

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        _error(result, "dependencies_missing",
               "ffmpeg and ffprobe must be available on PATH for motion analysis",
               status="unavailable")
        return _sanitize(result)

    try:
        info = _probe_media(resolved)
    except MotionError as exc:
        if exc.code == "ffprobe_missing":
            status = "unavailable"
        elif exc.code in ("no_video_stream", "invalid_media"):
            status = "invalid_input"
        else:
            status = "failed"
        _error(result, exc.code, exc.message, status=status)
        return _sanitize(result)

    source = result["source"]
    source.update({
        "digest_sha256": _sha256(resolved),
        "size_bytes": info["size_bytes"] if info["size_bytes"] is not None else resolved.stat().st_size,
        "duration_seconds": _finite(info["duration"]),
        "start_time": _finite(info["start_time"]),
        "width": info["width"],
        "height": info["height"],
        "rotation_degrees": info["rotation_degrees"],
        "display_width": info["display_width"],
        "display_height": info["display_height"],
        "codec": info["codec"],
        "pix_fmt": info["pix_fmt"],
        "format_name": info["format_name"],
        "fps": info["fps"],
    })
    if info["start_time"] not in (None, 0, 0.0) and abs(info["start_time"]) > 1e-6:
        _warn(result, "non_zero_media_start_time",
              f"media start_time is {info['start_time']} s; interval seconds follow ffmpeg -ss semantics")

    media_duration = info["duration"]
    media_end = media_duration
    truncation_reasons = []
    effective_end = media_end if end is None else min(end, media_end)
    if end is not None and end > media_end:
        _warn(result, "end_clipped_to_media",
              f"requested end {end:.6f}s exceeds media duration {media_end:.6f}s")
        truncation_reasons.append("requested_end_beyond_media")
    if start >= media_end or effective_end <= start:
        _error(result, "interval_outside_media",
               f"interval [{start}, {effective_end}] does not overlap media [0, {media_end}]",
               status="invalid_input")
        return _sanitize(result)

    shot_mode = shots is not None
    try:
        if shot_mode:
            intervals = _shot_intervals(shots, start, effective_end, media_end)
        else:
            intervals = [{
                "shot_index": None,
                "requested_start": start,
                "requested_end": media_end if end is None else end,
                "start": start,
                "end": effective_end,
                "invalid": False,
            }]
    except MotionError as exc:
        _error(result, exc.code, exc.message, status="invalid_input")
        return _sanitize(result)

    total_requested_duration = sum(max(0.0, interval["requested_end"] - interval["requested_start"])
                                   for interval in intervals)
    coverage = result["coverage"]
    coverage["requested"] = {
        "start": _finite(start),
        "end": _finite(media_end if end is None else end),
        "duration": _finite((media_end if end is None else end) - start),
        "shots_total_duration": _finite(total_requested_duration) if shot_mode else None,
    }
    coverage["clip_to_media"] = {
        "start": _finite(start), "end": _finite(effective_end),
        "duration": _finite(effective_end - start),
    }
    coverage["budget"] = {"max_seconds": _finite(max_seconds), "max_frames": MAX_FRAME_CAP,
                          "max_pairs": MAX_FRAME_CAP - 1}

    result["parameters"].update({
        "sample_fps": _finite(sample_fps),
        "analysis_width": analysis_width,
        "analysis_height": None,
        "max_seconds": _finite(max_seconds),
        "max_frames": MAX_FRAME_CAP,
        "max_pairs": MAX_FRAME_CAP - 1,
        "smoothing": {"method": "centered_moving_average",
                      "window_samples": smoothing_window,
                      "scope": "within valid contiguous samples only; no gap bridging"},
        "shot_head_tail_seconds": _finite(head_tail_seconds),
    })

    remaining_seconds = max_seconds
    remaining_frames = MAX_FRAME_CAP
    all_pairs = []
    all_series = []
    all_segments = []
    all_frame_times = []
    combined_decode_failed = []
    shot_reports = []
    pair_offset = 0
    segment_offset = 0
    timeline_sources = set()
    timeline_widths = set()
    timeline_heights = set()
    timestamp_validation = {"status": "ok", "max_deviation_seconds": None,
                            "pts_samples": 0, "frames_decoded": 0}
    total_analyzed_duration = 0.0
    fatal = False

    for interval in intervals:
        report = {
            "shot_index": interval["shot_index"],
            "requested": {"start": _finite(interval["requested_start"]),
                          "end": _finite(interval["requested_end"]),
                          "duration": _finite(interval["requested_end"] - interval["requested_start"])},
            "analyzed": {"start": None, "end": None, "duration": None},
            "status": "pending",
            "reason": None,
            "truncated": False,
            "pair_range": None,
            "timeline": _empty_timeline(),
            "head_window": None,
            "tail_window": None,
            "summary": _summarize([], {}),
        }
        if interval["invalid"]:
            report["status"] = "invalid_interval"
            report["reason"] = "interval_outside_requested_or_media_range"
            report["summary"]["status"] = "insufficient_evidence"
            shot_reports.append(report)
            continue
        duration = interval["end"] - interval["start"]
        if remaining_seconds <= 1.0 / sample_fps:
            report["status"] = "skipped_budget"
            report["reason"] = "max_seconds_budget_exhausted"
            report["truncated"] = True
            report["summary"]["status"] = "insufficient_evidence"
            truncation_reasons.append("max_seconds_budget_exhausted")
            shot_reports.append(report)
            continue
        analyzed_duration = min(duration, remaining_seconds)
        if analyzed_duration < duration:
            report["truncated"] = True
            truncation_reasons.append("max_seconds")
            _warn(result, "max_seconds_truncated",
                  f"shot {interval['shot_index']} clipped to {analyzed_duration:.3f}s by max_seconds")
        frames_needed = int(math.floor(analyzed_duration * sample_fps)) + 1
        if frames_needed > remaining_frames:
            analyzed_duration = max(0.0, (remaining_frames - 1) / sample_fps)
            frames_needed = remaining_frames
            report["truncated"] = True
            truncation_reasons.append("max_frames")
            _warn(result, "max_frames_truncated",
                  f"shot {interval['shot_index']} clipped to {analyzed_duration:.3f}s by the frame cap")
            if analyzed_duration <= 0:
                report["status"] = "skipped_budget"
                report["reason"] = "frame_cap_exhausted"
                report["summary"]["status"] = "insufficient_evidence"
                shot_reports.append(report)
                continue

        try:
            decoded = _decode_interval(
                resolved, interval["start"], analyzed_duration, sample_fps, analysis_width,
                info["display_width"], info["display_height"], cv2, np,
                max_frames=min(remaining_frames, frames_needed + 1))
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
            report["status"] = "failed"
            report["reason"] = "decode_failed"
            combined_decode_failed.append(str(exc))
            _error(result, "decode_failed",
                   f"shot {interval['shot_index']} could not be decoded: {exc}", status="failed")
            fatal = True
            shot_reports.append(report)
            break

        pairs = decoded["pairs"]
        decode = decoded["decode"]
        if decode["frames_decoded"] == 0:
            if decode["timed_out"]:
                report["status"] = "failed"
                report["reason"] = "decode_timeout"
                _error(result, "decode_timeout",
                       f"shot {interval['shot_index']} exceeded the decode deadline "
                       f"before producing a frame", status="failed")
                fatal = True
            elif decode["decoder_failed"] or decode["ffmpeg_returncode"] not in (0, None):
                report["status"] = "failed"
                report["reason"] = "decode_no_frames"
                _error(result, "decode_no_frames",
                       f"shot {interval['shot_index']} produced no frames "
                       f"(ffmpeg exit {decode['ffmpeg_returncode']}): "
                       f"{(decode['stderr_tail'] or '')[-300:]}", status="failed")
                fatal = True
            else:
                report["status"] = "insufficient_evidence"
                report["reason"] = "no_frames_decoded"
            shot_reports.append(report)
            continue

        remaining_seconds -= analyzed_duration
        remaining_frames -= decode["frames_decoded"]
        total_analyzed_duration += decode["duration_analyzed"] or 0.0
        interval_duration_analyzed = decode["duration_analyzed"]

        unique_shot = 0 if interval["shot_index"] is None else interval["shot_index"]
        # Segment continuity uses a stable integer shot key while series are built; public
        # pair_index/segment_index stay interval-local until construction is complete.
        for pair in pairs:
            pair["shot_index"] = unique_shot
        ordered = sorted(pairs, key=lambda item: item["pair_index"])
        series, segments = _build_series(ordered, smoothing_window)

        # Shift interval-local indices/segment ids into the global arrays exactly once.
        for pair in ordered:
            pair["pair_index"] += pair_offset
            if pair["segment_index"] is not None:
                pair["segment_index"] += segment_offset
            pair["shot_index"] = interval["shot_index"]
        for entry in series:
            entry["pair_index"] += pair_offset
            if entry["segment_index"] is not None:
                entry["segment_index"] += segment_offset
            entry["shot_index"] = interval["shot_index"]
        for segment in segments:
            segment["segment_index"] += segment_offset
            segment["first_pair"] += pair_offset
            segment["last_pair"] += pair_offset
            segment["shot_index"] = interval["shot_index"]
        ordered = sorted(ordered, key=lambda item: item["pair_index"])
        series_by_index = {entry["pair_index"]: entry for entry in series}

        analyzed_start = decoded["frame_times_absolute"][0] if decoded["frame_times_absolute"] else None
        analyzed_end = decoded["frame_times_absolute"][-1] if decoded["frame_times_absolute"] else None
        report["analyzed"] = {
            "start": _finite(analyzed_start),
            "end": _finite(analyzed_end),
            "duration": _finite(interval_duration_analyzed),
        }
        report["pair_range"] = {
            "first": ordered[0]["pair_index"] if ordered else None,
            "last": ordered[-1]["pair_index"] if ordered else None,
            "count": len(ordered),
            "valid_count": sum(1 for pair in ordered if pair["status"] == "valid"),
        }
        report["timeline"] = decoded["timeline"]
        report["timeline"]["frame_times_seconds"] = [
            _finite(value) for value in decoded["timeline"]["frame_times_seconds"]]

        if decoded["decode"]["truncated"]:
            if decode["timed_out"]:
                _error(result, "decode_timeout",
                       f"shot {interval['shot_index']} hit the decode deadline after "
                       f"{decode['frames_decoded']} of ~{decode['frames_planned']} planned "
                       f"frames; partial evidence retained", status="partial")
                report["truncated"] = True
                truncation_reasons.append("decode_timeout")
            elif decode["decoder_failed"]:
                _error(result, "decode_error",
                       f"shot {interval['shot_index']} decoded "
                       f"{decode['frames_decoded']} of ~{decode['frames_planned']} planned "
                       f"frames; ffmpeg exit {decode['ffmpeg_returncode']}"
                       f"{'; ' + decode['read_error'] if decode['read_error'] else ''}: "
                       f"{(decode['stderr_tail'] or '')[-300:]}", status="partial")
                report["truncated"] = True
                truncation_reasons.append("decode_error")
            elif decode["cap_hit"] and decode["frames_decoded"] < decode["frames_planned"]:
                truncation_reasons.append("frame_cap")
            else:
                _warn(result, "media_ended_before_planned_end",
                      f"shot {interval['shot_index']} received {decode['frames_decoded']} of "
                      f"~{decode['frames_planned']} planned frames")
                report["truncated"] = True
                truncation_reasons.append("media_ended")
        if decode["partial_frame"]:
            _warn(result, "partial_frame_payload",
                  f"shot {interval['shot_index']} ended with an incomplete raw frame")

        report["summary"] = _summarize(ordered, series_by_index)
        if report["summary"]["valid_pair_count"]:
            report["status"] = "partial" if report["truncated"] else "measured"
        else:
            report["status"] = "insufficient_evidence"
            report["reason"] = _dominant_reason(ordered)

        # Head/tail windows reference real pair indices inside this shot.
        if analyzed_start is not None and analyzed_end is not None and analyzed_end > analyzed_start:
            head_end = min(analyzed_start + head_tail_seconds, analyzed_end)
            tail_start = max(analyzed_end - head_tail_seconds, analyzed_start)
            report["head_window"] = _window(ordered, analyzed_start, head_end, series_by_index)
            report["tail_window"] = _window(ordered, tail_start, analyzed_end, series_by_index)

        all_pairs.extend(ordered)
        all_series.extend(series)
        all_segments.extend(segments)
        all_frame_times.extend(decoded["timeline"]["frame_times_seconds"])
        pair_offset += len(ordered)
        segment_offset += len(segments)
        timeline_sources.add(decoded["timeline"]["timestamp_source"])
        timeline_widths.add(decode["frame_width"])
        timeline_heights.add(decode["frame_height"])
        validation = decoded["timeline"]["timestamp_validation"]
        timestamp_validation["pts_samples"] += validation.get("pts_samples") or 0
        timestamp_validation["frames_decoded"] += validation.get("frames_decoded") or 0
        if validation.get("status") not in (None, "ok"):
            timestamp_validation["status"] = validation.get("status")
        deviation = validation.get("max_deviation_seconds")
        if deviation is not None:
            previous = timestamp_validation.get("max_deviation_seconds") or 0.0
            timestamp_validation["max_deviation_seconds"] = _finite(max(previous, deviation))
        shot_reports.append(report)

    result["pairs"] = all_pairs
    result["series"] = all_series
    result["segments"] = all_segments
    result["timeline"] = {
        "timestamp_source": (",".join(sorted(timeline_sources)) if timeline_sources else None),
        "analysis_fps": _finite(sample_fps),
        "frame_width": next(iter(timeline_widths)) if len(timeline_widths) == 1 else None,
        "frame_height": next(iter(timeline_heights)) if len(timeline_heights) == 1 else None,
        "sample_count": len(all_frame_times),
        "frame_times_seconds": [_finite(value) for value in sorted(all_frame_times)],
        "timestamp_validation": timestamp_validation,
    }
    if len(timeline_widths) == 1 and timeline_widths:
        result["parameters"]["analysis_width"] = next(iter(timeline_widths))
    if len(timeline_heights) == 1 and timeline_heights:
        result["parameters"]["analysis_height"] = next(iter(timeline_heights))
    coverage["analyzed"] = {
        "start": _finite(min(all_frame_times)) if all_frame_times else None,
        "end": _finite(max(all_frame_times)) if all_frame_times else None,
        "duration": _finite(total_analyzed_duration) if all_frame_times else None,
    }

    # Coverage has two distinct meanings. Decoded coverage says which source span was
    # actually sampled. Measured coverage is the union of usable valid pair intervals;
    # an envelope that spans an internal invalid region is not measured coverage.
    pairs_by_shot = {}
    for pair in all_pairs:
        pairs_by_shot.setdefault(pair["shot_index"], []).append(pair)
    sampling_reasons = set()
    measurement_reasons = set()
    measurement_gaps = []
    component_measurements = []
    component_intervals = []
    sample_quantum = 1.0 / sample_fps
    for report in shot_reports:
        shot_key = report["shot_index"]
        component_pairs = pairs_by_shot.get(shot_key, [])
        timed_pairs = [pair for pair in component_pairs
                       if pair["start_time"] is not None and pair["end_time"] is not None]
        valid_pairs = [pair for pair in timed_pairs if pair["status"] == "valid"]
        invalid_pairs = [pair for pair in timed_pairs if pair["status"] != "valid"]
        sample_times = [value for value in
                        (report["timeline"].get("frame_times_seconds") or [])
                        if _finite(value) is not None]
        requested_duration = report["requested"]["duration"] or 0.0
        # The effective selected component after the overall interval and media
        # intersection. Completeness, end reach and tails use this scope, never the
        # unsupplied space before, between or after caller-selected shots.
        selected_start = _finite(max(report["requested"]["start"], start))
        selected_end = _finite(min(report["requested"]["end"], effective_end))
        selected_duration = (_finite(selected_end - selected_start)
                             if selected_start is not None and selected_end is not None
                             and selected_end > selected_start else 0.0)
        component_issues = []
        if report["status"] in ("invalid_interval", "skipped_budget", "failed", "pending"):
            component_issues.append(report["reason"] or report["status"])
        elif not sample_times:
            component_issues.append("no_frames_decoded")
        if report["truncated"]:
            component_issues.append("truncated_decode")

        scope_start = _finite(sample_times[0]) if sample_times else None
        scope_end = _finite(sample_times[-1]) if sample_times else None
        decoded_component_tail = 0.0
        if selected_duration > COVERAGE_INTERVAL_EPSILON:
            if scope_end is None:
                decoded_component_tail = selected_duration
            else:
                decoded_component_tail = _finite(max(0.0, selected_end - scope_end))
            if shot_mode and decoded_component_tail > sample_quantum + 1e-6:
                component_issues.append("sampled_span_does_not_reach_requested_end")
        for issue in set(component_issues):
            sampling_reasons.add(issue)

        covered = _clip_interval_union(
            ((pair["start_time"], pair["end_time"]) for pair in valid_pairs),
            scope_start, scope_end)
        component_intervals.extend(covered)
        gap_ranges = _interval_gaps(covered, scope_start, scope_end)
        component_reasons = set(component_issues)
        measured_component_tail = 0.0
        if selected_duration > COVERAGE_INTERVAL_EPSILON:
            if covered:
                measured_component_tail = _finite(max(0.0, selected_end - covered[-1][1]))
            else:
                measured_component_tail = selected_duration

        if requested_duration > 0 and not valid_pairs:
            if selected_duration > COVERAGE_INTERVAL_EPSILON:
                gap_start, gap_end = selected_start, selected_end
            else:
                # The selected shot is empty after intersection; keep its requested
                # bounds as the record without treating them as selected evidence.
                gap_start = _finite(report["requested"]["start"])
                gap_end = _finite(report["requested"]["end"])
            measurement_gaps.append({
                "shot_index": shot_key,
                "kind": "no_valid_pairs",
                "start": gap_start,
                "end": gap_end,
                "duration": _finite(gap_end - gap_start)
                            if gap_start is not None and gap_end is not None else None,
                "requested_start": gap_start,
                "requested_end": gap_end,
                "pair_count": len(component_pairs),
                "valid_pair_count": 0,
                "invalid_pair_count": len(invalid_pairs),
                "invalid_pair_reasons": sorted({pair["reason"] or "invalid_pair"
                                                for pair in invalid_pairs}),
                "reason": report["reason"] or "no_valid_pairs",
            })
        elif valid_pairs:
            for gap_start, gap_end in gap_ranges:
                if gap_start <= scope_start + COVERAGE_INTERVAL_EPSILON:
                    kind = "leading"
                elif gap_end >= scope_end - COVERAGE_INTERVAL_EPSILON:
                    kind = "trailing"
                else:
                    kind = "internal"
                overlapping = [
                    pair for pair in invalid_pairs
                    if min(pair["end_time"], gap_end) - max(pair["start_time"], gap_start)
                    > COVERAGE_INTERVAL_EPSILON]
                reason = f"{kind}_invalid_pairs"
                component_reasons.add(reason)
                measurement_reasons.add(f"measurement_gap_{kind}")
                measurement_gaps.append({
                    "shot_index": shot_key,
                    "kind": kind,
                    "start": _finite(gap_start),
                    "end": _finite(gap_end),
                    "duration": _finite(gap_end - gap_start),
                    "requested_start": report["requested"]["start"],
                    "requested_end": report["requested"]["end"],
                    "pair_count": len(component_pairs),
                    "valid_pair_count": len(valid_pairs),
                    "invalid_pair_count": len(overlapping),
                    "invalid_pair_reasons": sorted({pair["reason"] or "invalid_pair"
                                                    for pair in overlapping}),
                    "reason": reason,
                })
        if component_issues:
            for issue in component_issues:
                measurement_reasons.add(issue)
        component_complete = bool(
            not component_issues and bool(valid_pairs) and covered
            and scope_start is not None and scope_end is not None
            and scope_end - scope_start > COVERAGE_INTERVAL_EPSILON
            and not gap_ranges)
        component_measurements.append({
            "shot_index": shot_key,
            "selected_start": selected_start,
            "selected_end": selected_end,
            "sampled_start": scope_start,
            "sampled_end": scope_end,
            "sample_count": len(sample_times),
            "pair_count": len(component_pairs),
            "valid_pair_count": len(valid_pairs),
            "measured_start": _finite(covered[0][0]) if covered else None,
            "measured_end": _finite(covered[-1][1]) if covered else None,
            "measured_duration": _interval_union_duration(covered),
            "measured_envelope_duration": (
                _finite(scope_end - scope_start)
                if scope_start is not None and scope_end is not None else None),
            "covered_ranges": _serialize_intervals(covered),
            "gaps": _serialize_intervals(gap_ranges),
            "decoded_unobserved_tail_seconds": _finite(decoded_component_tail),
            "measured_unobserved_tail_seconds": _finite(measured_component_tail),
            "complete": component_complete,
            "reasons": sorted(component_reasons),
        })

    decoded_start = _finite(min(all_frame_times)) if all_frame_times else None
    decoded_end = _finite(max(all_frame_times)) if all_frame_times else None
    decoded_envelope_duration = (
        _finite(decoded_end - decoded_start)
        if decoded_start is not None and decoded_end is not None else None)
    measured_ranges = _merge_interval_union(component_intervals)
    measured_start = _finite(measured_ranges[0][0]) if measured_ranges else None
    measured_end = _finite(measured_ranges[-1][1]) if measured_ranges else None
    measured_duration = _interval_union_duration(measured_ranges)
    measured_envelope_duration = (_finite(measured_end - measured_start)
                                  if measured_start is not None and measured_end is not None
                                  else None)
    clip_end = _finite(effective_end)
    clip_duration = _finite(max(0.0, effective_end - start))
    if shot_mode:
        # Selected-shot scope: completeness and tails come from each effective
        # component, not from the outer request span. Unsupplied media outside the
        # selected shots is never counted as missing evidence.
        sampled_ranges = _merge_interval_union(
            (component["sampled_start"], component["sampled_end"])
            for component in component_measurements
            if component["sampled_start"] is not None
            and component["sampled_end"] is not None)
        decoded_duration = _interval_union_duration(sampled_ranges)
        if decoded_duration <= COVERAGE_INTERVAL_EPSILON:
            sampling_reasons.add("sampled_span_does_not_reach_requested_end")
        decoded_tail = max((component["decoded_unobserved_tail_seconds"]
                            for component in component_measurements), default=None)
        measured_tail = max((component["measured_unobserved_tail_seconds"]
                             for component in component_measurements), default=None)
        decoded_complete = bool(not fatal and decoded_duration > COVERAGE_INTERVAL_EPSILON
                                and not sampling_reasons and not truncation_reasons)
    else:
        decoded_duration = decoded_envelope_duration
        if decoded_start is None or decoded_end is None or \
                decoded_end <= decoded_start or \
                decoded_end < effective_end - sample_quantum - 1e-6:
            sampling_reasons.add("sampled_span_does_not_reach_requested_end")
        decoded_tail = (_finite(max(0.0, clip_end - decoded_end))
                        if decoded_end is not None and clip_end is not None else clip_duration)
        measured_tail = (_finite(max(0.0, clip_end - measured_end))
                         if measured_end is not None and clip_end is not None else clip_duration)
        decoded_complete = bool(not fatal and all_frame_times and not sampling_reasons
                                and not truncation_reasons)
    tail_quantum_ok = (decoded_tail is not None and measured_tail is not None
                       and decoded_tail <= sample_quantum + 1e-6
                       and measured_tail <= sample_quantum + 1e-6)
    measured_complete = bool(decoded_complete and tail_quantum_ok and measured_ranges
                             and not measurement_gaps and not measurement_reasons
                             and all(component["complete"]
                                     for component in component_measurements))
    coverage["decoded"] = {
        "start": decoded_start,
        "end": decoded_end,
        "duration": decoded_duration,
        "envelope_duration": decoded_envelope_duration,
        "sample_count": len(all_frame_times),
        "complete": decoded_complete,
        "unobserved_tail_seconds": decoded_tail,
        "reasons": sorted(sampling_reasons),
    }
    coverage["measured"] = {
        "start": measured_start,
        "end": measured_end,
        "duration": measured_duration,
        "envelope_duration": measured_envelope_duration,
        "ranges": _serialize_intervals(measured_ranges),
        "pair_count": len(all_pairs),
        "valid_pair_count": sum(1 for pair in all_pairs if pair["status"] == "valid"),
        "complete": measured_complete,
        "unobserved_tail_seconds": measured_tail,
        "reasons": sorted(measurement_reasons),
    }
    coverage["measurement_gaps"] = measurement_gaps
    coverage["components"] = component_measurements
    raw_incomplete = list(truncation_reasons) + sorted(sampling_reasons | measurement_reasons)
    if measurement_gaps:
        raw_incomplete.append("measurement_gap")
    coverage["incomplete_reasons"] = sorted(set(raw_incomplete))
    fully_covered = bool(decoded_complete and measured_complete)
    coverage["fully_covered"] = fully_covered
    coverage["truncation_reasons"] = sorted(set(truncation_reasons))

    series_by_index = {entry["pair_index"]: entry for entry in all_series}
    overall_summary = _summarize(all_pairs, series_by_index)
    overall_summary["shot_count"] = len(shot_reports) if shot_mode else None
    overall_summary["analyzed_duration_seconds"] = _finite(total_analyzed_duration)
    result["summary"] = overall_summary

    if shot_mode:
        result["shots"] = shot_reports
    else:
        # In single-interval mode expose the same report shape for convenience.
        result["shots"] = None

    valid_pairs = overall_summary["valid_pair_count"]
    if fatal:
        pass  # status/error already set
    elif not all_pairs:
        if result["status"] == "partial" and result["errors"]:
            # A timeout/decoder error is more specific than "no pairs measured".
            # Keep the explicit failure and the honest partial status.
            pass
        elif result["errors"]:
            result["status"] = "partial"
            result["reason"] = result["reason"] or "decode_error"
        else:
            result["status"] = "insufficient_evidence"
            result["reason"] = "no_pairs_measured"
    elif valid_pairs == 0:
        result["status"] = "insufficient_evidence"
        result["reason"] = _dominant_reason(all_pairs)
    elif not fully_covered:
        result["status"] = "partial"
        result["reason"] = "+".join(coverage["incomplete_reasons"]) or "partial_coverage"
    else:
        result["status"] = "complete"
        result["reason"] = None

    if combined_decode_failed:
        result["errors"].append({"code": "decode_failures",
                                 "message": "; ".join(combined_decode_failed)[:1000]})
    return _sanitize(result)


def _dominant_reason(pairs):
    counts = {}
    for pair in pairs:
        if pair["status"] != "valid":
            reason = pair["reason"] or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _resolve_file(path):
    try:
        from pathlib import Path
        candidate = Path(path).expanduser()
    except (TypeError, ValueError):
        raise MotionError("invalid_input", "media_path must be a filesystem path")
    if not candidate.exists():
        raise MotionError("file_not_found", f"file does not exist: {candidate}")
    if not candidate.is_file():
        raise MotionError("not_a_file", f"path is not a regular file: {candidate}")
    resolved = candidate.resolve()
    try:
        with open(resolved, "rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise MotionError("unreadable_file", f"cannot read {resolved}: {exc}")
    return resolved


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _load_shots(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "shots" in payload:
        payload = payload["shots"]
    if not isinstance(payload, list):
        raise ValueError("shots JSON must be a list or {\"shots\": [...]}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure quantitative image motion (translation, zoom, rotation) for a "
                    "video interval or supplied shot intervals. Writes one JSON document to "
                    "stdout; diagnostics go to stderr.",
        epilog="Exit codes: 0 measured; 2 invalid input/arguments; 3 missing dependencies; "
               "4 analysis/decode failure. Status 'insufficient_evidence' exits 0 because the "
               "measurement succeeded but the picture did not support a valid fit.")
    parser.add_argument("input", help="video file (read-only)")
    parser.add_argument("--start", type=float, default=0.0, help="interval start in seconds (default 0)")
    parser.add_argument("--end", type=float, default=None, help="interval end in seconds (default media end)")
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS,
                        help=f"analysis sampling rate (default {DEFAULT_SAMPLE_FPS:g})")
    parser.add_argument("--analysis-width", type=int, default=DEFAULT_ANALYSIS_WIDTH,
                        help=f"analysis frame width (default {DEFAULT_ANALYSIS_WIDTH})")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                        help=f"cumulative duration budget (default {DEFAULT_MAX_SECONDS:g})")
    parser.add_argument("--shots-json", default=None,
                        help="JSON file with a list of {start, end} shot intervals")
    parser.add_argument("--head-tail", type=float, default=DEFAULT_HEAD_TAIL_SECONDS,
                        help=f"per-shot head/tail window seconds (default {DEFAULT_HEAD_TAIL_SECONDS:g})")
    parser.add_argument("--output", default=None,
                        help="optional path to also write the JSON document (stdout is always written)")
    args = parser.parse_args(argv)

    shots = None
    try:
        if args.shots_json:
            shots = _load_shots(args.shots_json)
        started = time.time()
        result = analyze_motion(
            args.input, start=args.start, end=args.end, shots=shots,
            sample_fps=args.sample_fps, analysis_width=args.analysis_width,
            max_seconds=args.max_seconds, head_tail_seconds=args.head_tail)
    except (OSError, ValueError) as exc:
        result = _base_result(args.input, {
            "start": _finite(args.start), "end": _finite(args.end),
            "shots": bool(args.shots_json), "sample_fps": _finite(args.sample_fps),
            "analysis_width": _int_or_none(args.analysis_width),
            "max_seconds": _finite(args.max_seconds),
        })
        _error(result, "cli_input_error", str(exc), status="invalid_input")
        started = time.time()
    except Exception as exc:  # Keep a structured JSON contract even on unexpected failures.
        result = _base_result(args.input, None)
        _error(result, "unexpected_failure", f"{type(exc).__name__}: {exc}", status="failed")

    result = _sanitize(result)
    document = json.dumps(result, indent=2, ensure_ascii=False)
    write_failed = False
    if args.output:
        try:
            from pathlib import Path
            Path(args.output).expanduser().write_text(document + "\n", encoding="utf-8")
        except OSError as exc:
            write_failed = True
            print(f"motion_analyze: cannot write --output: {exc}", file=sys.stderr)
    print(document)

    elapsed = time.time() - started
    status = result.get("status")
    print(f"motion_analyze: status={status} reason={result.get('reason')} "
          f"pairs={len(result.get('pairs') or [])} "
          f"valid={result.get('summary', {}).get('valid_pair_count')} "
          f"elapsed={elapsed:.2f}s", file=sys.stderr)
    for warning in result.get("warnings") or []:
        print(f"motion_analyze: warning {warning['code']}: {warning['message']}", file=sys.stderr)
    for error in result.get("errors") or []:
        print(f"motion_analyze: error {error['code']}: {error['message']}", file=sys.stderr)

    if write_failed:
        return 2
    return {"invalid_input": 2, "unavailable": 3}.get(status, 4 if status == "failed" else 0)


if __name__ == "__main__":
    sys.exit(main())
