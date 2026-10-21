#!/usr/bin/env python3
"""Validation and construction helpers for the Agent Video v1 result contract."""
from __future__ import annotations

from typing import Any

from asset_store import ANALYSIS_VERSION, SCHEMA_VERSION


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


def empty_analysis(asset_id: str, source: dict[str, Any], video: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
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
        "ocr": {"status": "not_run", "sampling": {}, "items": [], "reading_text": ""},
        "shots": {"status": "not_run", "items": []},
        "evidence": [],
        "summary": {"status": "not_run", "text": ""},
        "uncertainties": [],
    }


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
