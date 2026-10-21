#!/usr/bin/env python3
"""Persistent, portable storage for Reelpack Agent Video assets.

The store deliberately uses only the Python standard library.  It keeps the
structured JSON result as the source of truth and derives the Markdown view
from that result.  Persisted paths are relative to ``library_root``; the
resolved local path is only added to query responses at runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
ANALYSIS_VERSION = "agent-video-v1.0"
INDEX_VERSION = 1


class AssetStoreError(RuntimeError):
    """A user-actionable storage or contract error."""


class AssetNotFoundError(AssetStoreError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def asset_id_for_hash(content_sha256: str) -> str:
    return f"asset_{content_sha256[:24]}"


def _safe_filename(value: str, fallback: str = "media") -> str:
    name = Path(value).name or fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return (name or fallback)[:120]


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _overlaps(item: dict[str, Any], start: float, end: float) -> bool:
    timestamp = item.get("timestamp")
    item_start = item.get("start", timestamp)
    item_end = item.get("end", timestamp)
    if item_start is None and item_end is None:
        return False
    try:
        item_start = float(item_start if item_start is not None else item_end)
        item_end = float(item_end if item_end is not None else item_start)
    except (TypeError, ValueError):
        return False
    return item_end >= start and item_start <= end


def _safe_relative_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise AssetStoreError(f"发现越界相对路径: {relative_path}")
    return candidate


class AssetStore:
    """A local library containing packages, media and rebuildable cache."""

    def __init__(self, library_root: str | Path):
        self.root = Path(library_root).expanduser().resolve()
        self.packages_dir = self.root / "packages"
        self.media_dir = self.root / "media" / "originals"
        self.cache_dir = self.root / "cache"
        self.index_path = self.root / "index.json"
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"index_version": INDEX_VERSION, "schema_version": SCHEMA_VERSION, "assets": {}}
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssetStoreError(f"资产索引损坏，无法读取: {self.index_path}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("assets", {}), dict):
            raise AssetStoreError(f"资产索引格式无效: {self.index_path}")
        return value

    def _package_dir(self, asset_id: str) -> Path:
        return self.packages_dir / asset_id

    def _asset_path(self, asset_id: str) -> Path:
        return self._package_dir(asset_id) / "asset.json"

    def _analysis_path(self, asset_id: str) -> Path:
        return self._package_dir(asset_id) / "analysis.json"

    def _markdown_path(self, asset_id: str) -> Path:
        return self._package_dir(asset_id) / "analysis.md"

    def import_file(self, source_path: str | Path, *, run_signature=None, force=False) -> dict[str, Any]:
        """Register and archive a local media file, deduplicating exact bytes.

        A different transcode or crop has a different byte hash and therefore a
        different asset_id.  The source file is never deleted or modified.
        """
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise AssetStoreError(f"媒体文件不存在或不是普通文件: {source}")
        try:
            size = source.stat().st_size
        except OSError as exc:
            raise AssetStoreError(f"无法读取媒体文件大小: {source}") from exc
        content_sha256 = sha256_file(source)
        asset_id = asset_id_for_hash(content_sha256)
        index = self._load_index()
        existing = index["assets"].get(asset_id)
        package_dir = self._package_dir(asset_id)
        package_dir.mkdir(parents=True, exist_ok=True)

        if existing:
            asset = self._read_json(self._asset_path(asset_id), "asset")
            relative_media = asset.get("media", {}).get("relative_path")
            if not relative_media:
                raise AssetStoreError(f"资产缺少媒体相对路径: {asset_id}")
            media_path = _safe_relative_path(self.root, relative_media)
            if not media_path.exists():
                usage = shutil.disk_usage(self.root)
                if size > usage.free:
                    raise AssetStoreError(f"空间不足，无法恢复媒体（需要 {size} bytes，可用 {usage.free} bytes）")
                self._copy_and_verify(source, media_path, content_sha256)
            is_new = False
        else:
            filename = _safe_filename(source.name, fallback=f"{asset_id}.media")
            relative_media = str(Path("media") / "originals" / asset_id / filename)
            media_path = _safe_relative_path(self.root, relative_media)
            usage = shutil.disk_usage(self.root)
            if size > usage.free:
                raise AssetStoreError(f"空间不足，无法归档媒体（需要 {size} bytes，可用 {usage.free} bytes）")
            self._copy_and_verify(source, media_path, content_sha256)
            asset = {
                "schema_version": SCHEMA_VERSION,
                "asset_id": asset_id,
                "created_at": utc_now(),
                "source": {
                    "kind": "local_file",
                    "display_name": source.name,
                    "original_extension": source.suffix.lower(),
                },
                "content": {
                    "sha256": content_sha256,
                    "size_bytes": size,
                },
                "media": {
                    "relative_path": relative_media,
                    "status": "archived",
                },
                "analysis": {
                    "status": "not_run",
                    "analysis_version": ANALYSIS_VERSION,
                },
            }
            _atomic_json_write(self._asset_path(asset_id), asset)
            index["assets"][asset_id] = {
                "asset_id": asset_id,
                "content_sha256": content_sha256,
                "created_at": asset["created_at"],
                "package_relative_path": str(Path("packages") / asset_id),
            }
            _atomic_json_write(self.index_path, index)
            is_new = True

        analysis = self._read_json(self._analysis_path(asset_id), "analysis", optional=True)
        if run_signature is not None:
            from pipeline_state import reusable
            complete = not force and reusable(analysis, run_signature)
        else:
            # Legacy inspection/import callers have no request options. This flag
            # is never used by the pipeline without an explicit request signature.
            complete = bool(analysis and analysis.get("status") == "complete") and not force
        if complete:
            asset["analysis"] = {
                "status": analysis.get("status", "partial"),
                "analysis_version": analysis.get("analysis_version", ANALYSIS_VERSION),
            }
            _atomic_json_write(self._asset_path(asset_id), asset)
        return {
            "asset_id": asset_id,
            "is_new": is_new,
            "analysis_complete": complete,
            "media_path": str(media_path),
            "package_dir": str(package_dir),
            "package_relative_path": str(Path("packages") / asset_id),
        }

    def save_analysis(self, asset_id: str, analysis: dict[str, Any]) -> dict[str, str]:
        """Persist a structured result and derive Markdown from the same data."""
        asset = self._read_json(self._asset_path(asset_id), "asset")
        if not isinstance(analysis, dict):
            raise AssetStoreError("analysis 必须是 JSON object")
        payload = dict(analysis)
        if payload.get("asset_id") not in (None, asset_id):
            raise AssetStoreError(f"analysis.asset_id 与目标资产不一致: {payload.get('asset_id')} != {asset_id}")
        payload["asset_id"] = asset_id
        payload.setdefault("saved_at", utc_now())
        try:
            from asset_contract import assert_valid_analysis
            assert_valid_analysis(payload)
        except ValueError as exc:
            raise AssetStoreError(str(exc)) from exc
        _atomic_json_write(self._analysis_path(asset_id), payload)
        markdown = self._render_markdown(asset, payload)
        self._markdown_path(asset_id).write_text(markdown, encoding="utf-8")
        asset["analysis"] = {
            "status": payload["status"],
            "analysis_version": payload["analysis_version"],
            "saved_at": payload["saved_at"],
        }
        _atomic_json_write(self._asset_path(asset_id), asset)
        return {
            "analysis_path": str(self._analysis_path(asset_id)),
            "markdown_path": str(self._markdown_path(asset_id)),
            "package_dir": str(self._package_dir(asset_id)),
        }

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        asset = self._read_json(self._asset_path(asset_id), "asset")
        analysis = self._read_json(self._analysis_path(asset_id), "analysis", optional=True)
        relative_media = asset.get("media", {}).get("relative_path")
        media_path = _safe_relative_path(self.root, relative_media) if relative_media else None
        result = {
            "schema_version": SCHEMA_VERSION,
            "asset": asset,
            "analysis": analysis,
            "paths": {
                "package_relative": str(Path("packages") / asset_id),
                "markdown_relative": str(Path("packages") / asset_id / "analysis.md"),
                "media_relative": relative_media,
            },
            "media": {
                "status": "available" if media_path and media_path.exists() else "offline",
                "local_path": str(media_path) if media_path and media_path.exists() else None,
            },
        }
        return result

    def query_time(self, asset_id: str, start: float, end: float) -> dict[str, Any]:
        if end < start:
            raise AssetStoreError("时间范围结束值不能小于开始值")
        result = self.get_asset(asset_id)
        analysis = result.get("analysis") or {}
        transcript = analysis.get("transcript") or {}
        ocr = analysis.get("ocr") or {}
        shots = analysis.get("shots") or {}
        result["query"] = {"type": "time_range", "start": start, "end": end}
        result["analysis"] = {
            "schema_version": analysis.get("schema_version", SCHEMA_VERSION),
            "analysis_version": analysis.get("analysis_version", ANALYSIS_VERSION),
            "status": analysis.get("status", "unknown"),
            "transcript": {
                **{k: v for k, v in transcript.items() if k not in {"raw_segments", "segments"}},
                "raw_segments": [
                    item for item in transcript.get("raw_segments", [])
                    if isinstance(item, dict) and _overlaps(item, start, end)
                ],
            },
            "ocr": {
                **{k: v for k, v in ocr.items() if k != "items"},
                "items": [
                    item for item in ocr.get("items", [])
                    if isinstance(item, dict) and _overlaps(item, start, end)
                ],
            },
            "shots": {
                **{k: v for k, v in shots.items() if k != "items"},
                "items": [
                    item for item in shots.get("items", [])
                    if isinstance(item, dict) and _overlaps(item, start, end)
                ],
            },
            "evidence": [
                item for item in analysis.get("evidence", [])
                if isinstance(item, dict) and _overlaps(item, start, end)
            ],
        }
        result["evidence"] = result["analysis"]["evidence"]
        result["images"] = [item for item in result["evidence"] if item.get("kind") in {"frame", "image"}]
        return result

    def query_shot(self, asset_id: str, shot_id: str) -> dict[str, Any]:
        analysis = self.get_asset(asset_id).get("analysis") or {}
        shot = next((s for s in analysis.get("shots", {}).get("items", []) if s.get("shot_id") == shot_id), None)
        if shot is None:
            raise AssetStoreError(f"镜头不存在: {shot_id}")
        result = self.query_time(asset_id, shot["start"], shot["end"])
        result["query"] = {"type": "shot", "shot_id": shot_id, "start": shot["start"], "end": shot["end"]}
        result["analysis"]["shots"]["items"] = [shot]
        # Do not present whole-video reading text as this shot's transcript/OCR.
        transcript = result["analysis"]["transcript"]
        transcript["raw_segments"] = [s for s in transcript["raw_segments"]
                                      if s["start"] < shot["end"] and s["end"] > shot["start"]]
        transcript["raw_text"] = "\n".join(s.get("text", "") for s in transcript["raw_segments"])
        transcript["corrected_text"] = None
        transcript["correction"] = {"status": "not_run", "reason": "shot query returns timed raw segments"}
        ocr = result["analysis"]["ocr"]
        ocr["items"] = [item for item in ocr["items"]
                        if item.get("start", item.get("timestamp", shot["start"])) < shot["end"]]
        ocr["reading_text"] = "\n".join(item.get("text", "") for item in ocr["items"])
        ocr.pop("reading_reconstruction", None)
        result["evidence"] = [e for e in result["evidence"] if e.get("shot_id") == shot_id or
                              (not e.get("shot_id") and shot["start"] <= e.get("timestamp", shot["start"]) < shot["end"])]
        result["analysis"]["evidence"] = result["evidence"]
        result["images"] = [e for e in result["evidence"] if e.get("kind") in {"frame", "image"}]
        return result

    def set_summary(self, asset_id: str, text: str) -> dict[str, str]:
        analysis = self.get_asset(asset_id).get("analysis")
        if analysis is None:
            raise AssetStoreError("资产尚未有 analysis.json")
        analysis["summary"] = {"status": "complete" if text.strip() else "not_run",
                               "text": text.strip(), "source": "downstream_agent", "updated_at": utc_now()}
        return self.save_analysis(asset_id, analysis)

    def list_assets(self) -> list[dict[str, Any]]:
        index = self._load_index()
        output = []
        for asset_id in sorted(index["assets"]):
            try:
                result = self.get_asset(asset_id)
            except AssetStoreError:
                continue
            asset = result["asset"]
            analysis = result.get("analysis") or {}
            output.append({
                "asset_id": asset_id,
                "display_name": asset.get("source", {}).get("display_name"),
                "content_sha256": asset.get("content", {}).get("sha256"),
                "analysis_status": analysis.get("status", asset.get("analysis", {}).get("status", "unknown")),
                "media_status": result["media"]["status"],
                "package_relative": result["paths"]["package_relative"],
            })
        return output

    def _read_json(self, path: Path, label: str, optional: bool = False) -> dict[str, Any] | None:
        if not path.exists():
            if optional:
                return None
            raise AssetNotFoundError(f"未找到{label}: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssetStoreError(f"{label} JSON 无法读取: {path}") from exc
        if not isinstance(value, dict):
            raise AssetStoreError(f"{label} JSON 顶层必须是 object: {path}")
        return value

    @staticmethod
    def _copy_and_verify(source: Path, destination: Path, expected_sha256: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(f".{destination.name}.copying")
        try:
            shutil.copy2(source, temp)
            actual = sha256_file(temp)
            if actual != expected_sha256:
                raise AssetStoreError("媒体复制校验失败，源文件可能在复制过程中发生变化")
            os.replace(temp, destination)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _render_markdown(asset: dict[str, Any], analysis: dict[str, Any]) -> str:
        source = asset.get("source", {})
        name = source.get("display_name") or asset.get("asset_id", "asset")
        transcript = analysis.get("transcript") or {}
        lines = [
            f"# {name}",
            "",
            f"- asset_id: `{asset.get('asset_id', '')}`",
            f"- schema_version: `{analysis.get('schema_version', SCHEMA_VERSION)}`",
            f"- analysis_version: `{analysis.get('analysis_version', ANALYSIS_VERSION)}`",
            f"- status: `{analysis.get('status', 'unknown')}`",
            "",
            "## 摘要",
            "",
            str((analysis.get("summary") or {}).get("text", "(未生成)")),
            "",
            "## 原始转录",
            "",
            f"语言: `{transcript.get('language') or 'unknown'}` (probability: `{transcript.get('language_probability')}`)",
            "",
        ]
        raw_segments = transcript.get("raw_segments") or []
        if raw_segments:
            for item in raw_segments:
                start = item.get("start")
                end = item.get("end")
                text = item.get("text", "")
                lines.append(f"[{start:.3f}-{end:.3f}] {text}" if isinstance(start, (int, float)) and isinstance(end, (int, float)) else str(text))
        else:
            lines.append(str(transcript.get("raw_text", "(无转录)")))
        from readable_sections import rewrite_section, screen_text_section
        lines.extend(['', rewrite_section(transcript.get('rewrite')), '', screen_text_section(analysis.get('ocr')), ''])
        lines.extend(["", "## 镜头", "", "镜头分析状态: " + str((analysis.get("shots") or {}).get("status", "not_run")), ""])
        if (analysis.get("capabilities", {}).get("vision") or {}).get("status") != "complete":
            lines.extend(["本包无预生成镜头语义描述（vision 未运行或未完成）。请直接阅读 `evidence/*.jpg` 判断画面内容。", ""])
        shots = (analysis.get("shots") or {}).get("items") or []
        if shots:
            for shot in shots:
                lines.append(f"- `{shot.get('shot_id', '?')}` {shot.get('start', '?')}–{shot.get('end', '?')}: {shot.get('description', '')}")
        else:
            lines.append("(首阶段尚未运行镜头划分)")
        lines.extend(["", "## 能力状态", ""])
        for key, value in (analysis.get("capabilities") or {}).items():
            state = value.get("status", "unknown")
            lines.append(f"- {key}: `{state}`")
            if state != "complete":
                for field in ("reason", "impact", "remedy"):
                    if value.get(field): lines.append(f"  - {field}: {value[field]}")
        music = analysis.get("music") or {}
        descriptors = music.get("descriptors") or music
        lines.extend(["", "## 音频测量（全混音 / full mix）", "",
                      f"基础测量状态: `{descriptors.get('status', 'not_run')}`"])
        if descriptors.get("status") in ("complete", "partial"):
            if descriptors.get("tempo_bpm") is not None:
                lines.append(f"检出的节奏: {descriptors['tempo_bpm']:.3f} BPM")
            lines.append(f"检出的拍点数: {len(descriptors.get('beat_times_seconds', []))}")
        alignment = descriptors.get("cut_alignment") or music.get("cut_alignment") or {}
        ratio = alignment.get("aligned_ratio")
        if alignment.get("status") == "measured" and ratio is not None:
            lines.append(f"切点/拍点对齐: {alignment.get('aligned_cut_count')}/{alignment.get('cut_count')} ({ratio:.2%})；容差 ±{alignment.get('tolerance_seconds')} s")
        else:
            lines.append("切点/拍点对齐: **未测得（not_measured）**。")
            if alignment.get("reason"): lines.append(f"原因: {alignment['reason']}")
        tagging = music.get("tagging") or {"status": "not_run"}
        lines.append(f"可选曲风/情绪标签: `{tagging.get('status', 'not_run')}`")
        for key in ("reason", "impact", "remedy"):
            if tagging.get(key): lines.append(f"- {key}: {tagging[key]}")
        board = music.get("audio_board") or {}
        if board.get("status") == "complete" and board.get("path"):
            lines.extend(["", f"![音频证据板：全混音 / full mix]({board['path']})", ""])
        else:
            lines.append(f"音频证据图: `{board.get('status', 'not_run')}`；{board.get('reason', '未生成')}")
        lines.append("图和数字仅描述全混音测量，不做人声分离；不支持曲风、情绪、作者意图、独立 BGM 或乐器识别结论。")
        return "\n".join(lines) + "\n"
