#!/usr/bin/env python3
"""Validation and construction helpers for the Agent Video v1 result contract."""
from __future__ import annotations

import math
import re
from typing import Any

from asset_store import ANALYSIS_VERSION, SCHEMA_VERSION
from ocr_regions import REGION_SUPPORT
from pipeline_state import engine_provenance


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "analysis_version",
    "asset_id",
    "status",
    "source",
    "video",
    "capabilities",
    "transcript",
    "ocr",
    "shots",
    "evidence",
}
VALID_STATUS = {"complete", "partial", "failed", "unknown"}
VALID_CAPABILITY_STATUS = {"complete", "partial", "failed", "not_run", "unknown"}
SPEAKER_LABEL = re.compile(r"^S[1-9]\d*$")


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def migrate_speakers(analysis):
    """Read-time migration: old packages gain an explicit not_run speakers block.

    This never rewrites the package on disk; only the returned analysis object
    is completed, mirroring ``ocr_regions.migrate_ocr_regions``.
    """
    if isinstance(analysis, dict):
        analysis.setdefault("speakers", {"status": "not_run"})
    return analysis


def empty_analysis(asset_id: str, source: dict[str, Any], video: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "engine": engine_provenance(),
        "asset_id": asset_id,
        "status": "partial",
        "source": source,
        "video": video or {"duration_seconds": None, "width": None, "height": None, "fps": None},
        "capabilities": {
            "transcription": {"status": "not_run", "backend": None, "model": None},
            "ocr": {"status": "not_run", "backend": None, "model": None},
            "shots": {"status": "not_run", "backend": None, "model": None},
            "vision": {"status": "not_run", "backend": None, "model": None},
            "music": {"status": "not_run", "backend": None, "model": None, "experimental": True},
        },
        "transcript": {
            "status": "not_run",
            "raw_segments": [],
            "raw_text": "",
            "corrected_text": None,
            "correction": {"status": "not_run", "edits": []},
        },
        "ocr": {"status": "not_run", "sampling": {}, "items": [], "reading_text": "",
                "region_support": dict(REGION_SUPPORT)},
        "ocr_conflicts": [],
        "shots": {"status": "not_run", "items": []},
        "content_hints": {
            "shots_per_min": None,
            "speech_ratio": None,
            "ocr_change_rate": None,
            "frame_board": None,
            "duration_seconds": None,
        },
        "evidence": [],
        "summary": {"status": "not_run", "text": ""},
        "uncertainties": [],
    }


def _validate_speakers(speakers: Any, errors: list[str]) -> None:
    """Optional experimental block: complete runs carry labelled speech segments."""
    if not isinstance(speakers, dict):
        errors.append("speakers 必须是 object")
        return
    status = speakers.get("status")
    if status not in VALID_CAPABILITY_STATUS:
        errors.append(f"speakers.status 必须是 {sorted(VALID_CAPABILITY_STATUS)} 之一")
    for field in ("backend", "model"):
        if speakers.get(field) is not None and not isinstance(speakers.get(field), str):
            errors.append(f"speakers.{field} 必须是 string 或 null")
    if "experimental" in speakers and not isinstance(speakers["experimental"], bool):
        errors.append("speakers.experimental 必须是 boolean")
    if status == "failed" and not speakers.get("reason"):
        errors.append("speakers failed 缺少 reason")
    confidence = speakers.get("confidence")
    if confidence is not None and not isinstance(confidence, dict):
        errors.append("speakers.confidence 必须是 object")
    segments = speakers.get("segments")
    if segments is None:
        if status == "complete":
            errors.append("speakers.segments 在 complete 时必须是 array")
        return
    if not isinstance(segments, list):
        errors.append("speakers.segments 必须是 array")
        return
    for index, item in enumerate(segments):
        if not isinstance(item, dict):
            errors.append(f"speakers.segments[{index}] 必须是 object")
            continue
        start, end = item.get("start"), item.get("end")
        if not _finite_number(start) or not _finite_number(end):
            errors.append(f"speakers.segments[{index}] 缺少有效的 start/end")
        elif float(end) < float(start):
            errors.append(f"speakers.segments[{index}] end 不能小于 start")
        speaker = item.get("speaker")
        if speaker is not None and (not isinstance(speaker, str) or not SPEAKER_LABEL.match(speaker)):
            errors.append(f"speakers.segments[{index}].speaker 必须是 S 编号字符串或 null")
        if not isinstance(item.get("text", ""), str):
            errors.append(f"speakers.segments[{index}].text 必须是 string")


def validate_analysis(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["analysis 顶层必须是 object"]
    missing = sorted(REQUIRED_TOP_LEVEL - set(value))
    if missing:
        errors.append(f"缺少顶层字段: {', '.join(missing)}")
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {SCHEMA_VERSION}")
    if not value.get("analysis_version"):
        errors.append("analysis_version 不能为空")
    if value.get("status") not in VALID_STATUS:
        errors.append(f"status 必须是 {sorted(VALID_STATUS)} 之一")
    if not value.get("asset_id"):
        errors.append("asset_id 不能为空")

    for section in ("transcript", "ocr", "shots"):
        if not isinstance(value.get(section), dict):
            errors.append(f"{section} 必须是 object")
    if not isinstance(value.get("capabilities"), dict):
        errors.append("capabilities 必须是 object")
    if not isinstance(value.get("evidence"), list):
        errors.append("evidence 必须是 array")

    evidence_ids: set[str] = set()
    evidence = value.get("evidence")
    if isinstance(evidence, list):
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                errors.append(f"evidence[{index}] 必须是 object")
                continue
            evidence_id = item.get("evidence_id")
            if not evidence_id:
                errors.append(f"evidence[{index}] 缺少 evidence_id")
            elif evidence_id in evidence_ids:
                errors.append(f"evidence[{index}] evidence_id 重复: {evidence_id}")
            else:
                evidence_ids.add(evidence_id)
            if item.get("relative_path") is not None and not isinstance(item.get("relative_path"), str):
                errors.append(f"evidence[{index}].relative_path 必须是 string")

    transcript = value.get("transcript")
    if isinstance(transcript, dict):
        if not isinstance(transcript.get("raw_segments", []), list):
            errors.append("transcript.raw_segments 必须是 array")
        for index, item in enumerate(transcript.get("raw_segments", [])):
            if not isinstance(item, dict):
                errors.append(f"transcript.raw_segments[{index}] 必须是 object")
                continue
            if item.get("speaker") is not None and not isinstance(item["speaker"], str):
                errors.append(f"transcript.raw_segments[{index}].speaker 必须是 string 或 null")
            if item.get("start") is None or item.get("end") is None:
                errors.append(f"transcript.raw_segments[{index}] 缺少 start/end")
            if not isinstance(item.get("text", ""), str):
                errors.append(f"transcript.raw_segments[{index}].text 必须是 string")

    speakers = value.get("speakers")
    if speakers is not None:
        _validate_speakers(speakers, errors)

    ocr = value.get("ocr")
    if isinstance(ocr, dict):
        if not isinstance(ocr.get("items", []), list):
            errors.append("ocr.items 必须是 array")
        for index, item in enumerate(ocr.get("items", [])):
            if not isinstance(item, dict):
                errors.append(f"ocr.items[{index}] 必须是 object")
                continue
            # Per-frame backfill retains empty successful frames and explicit failures.
            if not isinstance(item.get("text"), str) or (not item.get("text") and item.get("status") not in {"complete", "failed"}):
                errors.append(f"ocr.items[{index}] 缺少 text")
            if item.get("status") == "failed" and not item.get("reason"):
                errors.append(f"ocr.items[{index}] failed 缺少 reason")
            if item.get("start") is None and item.get("timestamp") is None:
                errors.append(f"ocr.items[{index}] 缺少 timestamp 或 start")
            if item.get("region") is not None and not isinstance(item.get("region"), dict):
                errors.append(f"ocr.items[{index}].region 必须是 object 或 null")
            if item.get("region_norm") is not None and not isinstance(item.get("region_norm"), dict):
                errors.append(f"ocr.items[{index}].region_norm 必须是 object 或 null")
        region_support = ocr.get("region_support")
        if region_support is not None and not isinstance(region_support, dict):
            errors.append("ocr.region_support 必须是 object")
        sampling = ocr.get("sampling")
        if isinstance(sampling, dict):
            dropped = sampling.get("dropped")
            if dropped is not None:
                if not isinstance(dropped, list):
                    errors.append("ocr.sampling.dropped 必须是 array")
                else:
                    for index, item in enumerate(dropped):
                        if not isinstance(item, dict) or item.get("timestamp") is None or not item.get("reason"):
                            errors.append(f"ocr.sampling.dropped[{index}] 必须有 timestamp 和 reason")
                    if sampling.get("dropped_count") != len(dropped):
                        errors.append("ocr.sampling.dropped_count 必须等于 dropped 长度")
            count_keys = ("coarse_scans", "densified_frames", "total_ocr_frames", "dropped_count")
            if all(isinstance(sampling.get(key), int) and not isinstance(sampling.get(key), bool)
                   for key in count_keys):
                if sampling["total_ocr_frames"] != (sampling["coarse_scans"]
                                                    + sampling["densified_frames"]
                                                    - sampling["dropped_count"]):
                    errors.append("ocr.sampling 帧数不自洽: total != coarse + densified - dropped")

    conflicts = value.get("ocr_conflicts")
    if conflicts is not None:
        if not isinstance(conflicts, list):
            errors.append("ocr_conflicts 必须是 array")
        else:
            for index, item in enumerate(conflicts):
                if not isinstance(item, dict):
                    errors.append(f"ocr_conflicts[{index}] 必须是 object")
                    continue
                for key in ("start", "end", "ocr_text", "transcript_text", "similarity"):
                    if key not in item:
                        errors.append(f"ocr_conflicts[{index}] 缺少 {key}")

    shots = value.get("shots")
    if isinstance(shots, dict):
        if not isinstance(shots.get("items", []), list):
            errors.append("shots.items 必须是 array")
        seen: set[str] = set()
        for index, item in enumerate(shots.get("items", [])):
            if not isinstance(item, dict):
                errors.append(f"shots.items[{index}] 必须是 object")
                continue
            shot_id = item.get("shot_id")
            if not shot_id:
                errors.append(f"shots.items[{index}] 缺少 shot_id")
            elif shot_id in seen:
                errors.append(f"shots.items[{index}] shot_id 重复: {shot_id}")
            else:
                seen.add(shot_id)
            if item.get("start") is None or item.get("end") is None:
                errors.append(f"shots.items[{index}] 缺少 start/end")

    for section_name, items_key in (("transcript", "raw_segments"), ("ocr", "items"), ("shots", "items")):
        section = value.get(section_name)
        if isinstance(section, dict):
            for index, item in enumerate(section.get(items_key, [])):
                if not isinstance(item, dict):
                    continue
                refs = item.get("evidence_ids", [])
                if refs is not None and not isinstance(refs, list):
                    errors.append(f"{section_name}.{items_key}[{index}].evidence_ids 必须是 array")
                elif isinstance(refs, list):
                    for evidence_id in refs:
                        if evidence_id not in evidence_ids:
                            errors.append(f"{section_name}.{items_key}[{index}] 引用了不存在的 evidence_id: {evidence_id}")

    for name, capability in (value.get("capabilities") or {}).items():
        if isinstance(capability, dict) and capability.get("status") not in VALID_CAPABILITY_STATUS:
            errors.append(f"capabilities.{name}.status 无效")
    return errors


def assert_valid_analysis(value: dict[str, Any]) -> None:
    errors = validate_analysis(value)
    if errors:
        raise ValueError("资产分析契约校验失败: " + "; ".join(errors))


def transcript_segment(segment_id, start, end, text, speaker=None):
    """Speaker is an optional stable diarization label; None means not determined."""
    return {"segment_id": segment_id, "start": start, "end": end, "text": text, "speaker": speaker}
