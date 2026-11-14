#!/usr/bin/env python3
"""Installed motion-evidence adapter for an existing asset package.

This module only resolves an existing package, its archived original and its
saved shot intervals.  It never creates a library, cache or lock in read-only
mode and never changes motion mathematics: the measurement is delegated to the
``motion_analyze.analyze_motion`` API.

Runtime output may contain absolute input paths.  Persisted engine measurement
documents are made portable by ``asset_supplements.save_engine_measurement``.
"""
from __future__ import annotations

import math
from pathlib import Path

from asset_supplements import (
    SupplementError,
    analysis_duration,
    library_root,
    media_reference,
    package_dir,
    read_analysis,
    read_asset,
    save_engine_measurement,
    validate_asset_id,
)


MOTION_SCHEMA = "reelpack.motion-evidence"
MOTION_SCHEMA_VERSION = 1
USABLE_STATUSES = ("complete", "partial", "insufficient_evidence")
EXIT_CODES = {"invalid_input": 2, "unavailable": 3, "failed": 4}


class MotionCommandError(RuntimeError):
    def __init__(self, code, message, status="invalid_input", asset_scope=None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.asset_scope = asset_scope


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def validate_range(start, end):
    """Both bounds are mandatory finite numbers; no implicit whole-video range."""
    start, end = _finite(start), _finite(end)
    if start is None or end is None:
        raise MotionCommandError("invalid_interval", "motion 需要有限的 --start 与 --end")
    if start < 0:
        raise MotionCommandError("invalid_interval", "--start 必须 >= 0")
    if end <= start:
        raise MotionCommandError("invalid_interval", "--end 必须大于 --start")
    return start, end


def failure_result(error, *, asset_scope=None, cuts=None):
    scope = asset_scope if asset_scope is not None else error.asset_scope
    return {
        "schema": MOTION_SCHEMA,
        "schema_version": MOTION_SCHEMA_VERSION,
        "command": "motion",
        "status": error.status,
        "reason": str(error),
        "asset_scope": scope,
        "cuts": cuts or {"status": "not_evaluated",
                         "reason": "measurement was not attempted"},
        "errors": [{"code": error.code, "message": str(error)}],
    }


def _context(root, asset_id):
    root = library_root(root)
    asset_id = validate_asset_id(asset_id)
    package, asset = read_asset(root, asset_id)
    analysis, analysis_sha256 = read_analysis(package)
    media_path, media_relative, media_status = media_reference(root, asset)
    scope = {
        "asset_id": asset_id,
        "package_relative_path": f"packages/{asset_id}",
        "analysis_relative_path": f"packages/{asset_id}/analysis.json",
        "analysis_sha256": analysis_sha256,
        "analysis_present": analysis is not None,
        "source_content_sha256": asset["content"]["sha256"],
        "media_relative_path": media_relative,
        "media_path_base": "library_root",
        "media_status": media_status,
        "available_video_duration_seconds": analysis_duration(analysis),
    }
    return root, asset_id, package, asset, analysis, analysis_sha256, media_path, scope


def saved_cut_intervals(analysis, start, end):
    """Return saved shot intervals clipped to the request and explicit cut status."""
    if analysis is None:
        return None, {
            "status": "unavailable",
            "reason": "analysis_missing",
            "note": "没有已保存的镜头证据；切点边界不可用，未推断或编造任何切点。",
        }
    shots = analysis.get("shots")
    if not isinstance(shots, dict):
        return None, {
            "status": "unavailable",
            "reason": "shot_evidence_absent",
            "note": "没有已保存的镜头证据；切点边界不可用，未推断或编造任何切点。",
        }
    items = shots.get("items")
    if not isinstance(items, list):
        raise MotionCommandError("corrupt_shots", "analysis.shots.items 不是数组",
                                 status="failed")
    source_status = shots.get("status")
    if not items:
        return None, {
            "status": "unavailable",
            "reason": "saved_shot_list_empty",
            "source_status": source_status,
            "note": ("已保存的镜头列表为空；不得把请求范围静默当作已测量。"
                     "本次仍执行单区间测量并报告实际覆盖，切点边界不可用。"),
        }

    intervals = []
    for index, shot in enumerate(items):
        if not isinstance(shot, dict):
            raise MotionCommandError("corrupt_shots", f"shots.items[{index}] 不是 object",
                                     status="failed")
        shot_start, shot_end = _finite(shot.get("start")), _finite(shot.get("end"))
        if shot_start is None or shot_end is None or shot_end <= shot_start:
            raise MotionCommandError(
                "corrupt_shots",
                f"shots.items[{index}] 的 start/end 缺失、非有限或顺序无效",
                status="failed")
        clipped_start = max(shot_start, start)
        clipped_end = min(shot_end, end)
        if clipped_end > clipped_start:
            intervals.append({"start": clipped_start, "end": clipped_end,
                              "shot_id": shot.get("shot_id")})
    if not intervals:
        return None, {
            "status": "unavailable",
            "reason": "no_saved_shot_overlaps_requested_range",
            "source_status": source_status,
            "note": "没有已保存镜头与请求范围相交；本次按单区间测量，切点边界未被应用。",
        }
    return [{"start": item["start"], "end": item["end"]} for item in intervals], {
        "status": "applied",
        "source_status": source_status,
        "backend": shots.get("backend"),
        "shot_count": len(intervals),
        "intervals": intervals,
        "note": ("按已保存镜头区间裁剪并逐镜解码；未跨越已知切点生成帧对。"
                 "切点本身来自已保存分析，不是本次自动检测。"),
    }


def _motion_error_from_supplement(exc):
    message = str(exc)
    if "找不到已有资产库" in message:
        status, code = "invalid_input", "library_missing"
    elif "资产包不存在" in message or "无效 asset_id" in message:
        status, code = "invalid_input", "asset_not_found"
    elif "符号链接" in message or "越出" in message or "不安全" in message:
        status, code = "invalid_input", "unsafe_path"
    elif "媒体" in message and "offline" not in message:
        status, code = "failed", "metadata_error"
    else:
        status, code = "invalid_input", "package_error"
    return MotionCommandError(code, message, status=status)


def run_motion(root, asset_id, *, start, end, sample_fps=None, analysis_width=None,
               max_seconds=None, smoothing_window=None, head_tail_seconds=None,
               save=False):
    """Measure one explicit interval; return ``(result, exit_code)``."""
    try:
        start, end = validate_range(start, end)
    except MotionCommandError as exc:
        return failure_result(exc), EXIT_CODES[exc.status]

    try:
        (root, asset_id, package, asset, analysis, analysis_sha256,
         media_path, scope) = _context(root, asset_id)
    except SupplementError as exc:
        error = _motion_error_from_supplement(exc)
        return failure_result(error), EXIT_CODES[error.status]

    if scope["media_status"] != "available":
        error = MotionCommandError(
            "media_offline",
            f"归档原片不可用：{scope['media_relative_path']}（library_root 相对）。未修改任何已保存分析。",
            status="failed", asset_scope=scope)
        return failure_result(error), EXIT_CODES[error.status]
    if save and analysis is None:
        error = MotionCommandError(
            "analysis_missing",
            "资产尚未有 analysis.json，无法保存带来源摘要的引擎测量。可先只读测量。",
            status="invalid_input", asset_scope=scope)
        return failure_result(error), EXIT_CODES[error.status]

    try:
        intervals, cuts = saved_cut_intervals(analysis, start, end)
    except MotionCommandError as exc:
        exc.asset_scope = scope
        return failure_result(exc), EXIT_CODES[exc.status]

    from motion_analyze import analyze_motion

    options = {
        "sample_fps": sample_fps,
        "analysis_width": analysis_width,
        "max_seconds": max_seconds,
        "smoothing_window": smoothing_window,
        "head_tail_seconds": head_tail_seconds,
    }
    supplied = {name: value for name, value in options.items() if value is not None}
    result = analyze_motion(str(media_path), start=start, end=end,
                            shots=intervals, **supplied)
    result["command"] = "motion"
    result["asset_scope"] = scope
    result["cuts"] = cuts

    status = result.get("status")
    code = EXIT_CODES.get(status, 0 if status in USABLE_STATUSES else 4)
    if save:
        if status not in USABLE_STATUSES:
            result["save"] = {
                "status": "failed",
                "reason": f"只保存可用测量；当前状态 {status!r}",
            }
        else:
            try:
                result["save"] = save_engine_measurement(
                    root, asset_id, result,
                    expected_analysis_sha256=analysis_sha256)
                result["save"]["status"] = "saved"
            except SupplementError as exc:
                result["save"] = {"status": "failed", "reason": str(exc)}
                code = code or 2
    return result, code
