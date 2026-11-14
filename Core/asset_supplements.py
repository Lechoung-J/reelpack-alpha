#!/usr/bin/env python3
"""Immutable, portable supplement documents for an existing asset package.

Two supplement kinds exist and stay separate:

``agent_annotations``
    Versioned, attributed Agent interpretations (voice -> person, person -> name,
    person -> role).  They are interpretation records, never engine facts.

``engine_measurement``
    A preserved output of an engine measurement module (currently motion).

Documents live at ``packages/<asset_id>/supplements/<supplement_id>.json``.
There is no mutable index: listing validates the files and sorts the IDs.
Published files are immutable and created through a same-directory temporary
file plus an exclusive hard link, so a failure leaves either the complete
document or no new document.

Path rules: persisted references are package- or library-relative with an
explicit base.  Absolute machine paths are never persisted.  Writes take the
same per-asset ``.locks/<asset_id>.lock`` exclusive lock as ``AssetAttempt``.
Read-only operations never create a lock directory, cache, or library.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
from contextlib import contextmanager

from asset_store import sha256_file, utc_now


SUPPLEMENTS_DIRNAME = "supplements"
SUPPLEMENT_SCHEMA = "reelpack.supplement"
SUPPLEMENT_SCHEMA_VERSION = 1
ANNOTATION_SCHEMA = "reelpack.agent-annotations"
ANNOTATION_SCHEMA_VERSION = 1
ENGINE_MEASUREMENT_SCHEMA = "reelpack.engine-measurement"
ENGINE_MEASUREMENT_SCHEMA_VERSION = 1
ANNOTATION_SOURCE = "downstream_agent"
ENGINE_SOURCE = "engine"

ASSET_ID_RE = re.compile(r"^asset_[A-Za-z0-9_-]+$")
SUPPLEMENT_ID_RE = re.compile(r"^sup_[A-Za-z0-9_]+$")
SPEAKER_RE = re.compile(r"^S[1-9]\d*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ANNOTATION_KINDS = ("voice_person", "person_identity", "person_role")
ANNOTATION_STATUSES = ("supported", "tentative", "ambiguous", "unknown")
USABLE_MEASUREMENT_STATUSES = ("complete", "partial", "insufficient_evidence")


class SupplementError(RuntimeError):
    """A validated, user-actionable supplement error."""


# --------------------------------------------------------------------------------------
# Read-only package helpers
# --------------------------------------------------------------------------------------

def _finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def _require(condition, message):
    if not condition:
        raise SupplementError(message)


def library_root(root):
    """Resolve an existing library root without creating anything."""
    path = Path(root).expanduser()
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise SupplementError(f"找不到已有资产库：{root}。请先处理本地视频，或指定 --library-root。") from exc
    _require(path.is_dir(), f"资产库根路径不是目录：{path}")
    packages = path / "packages"
    _require(not packages.is_symlink(), f"拒绝符号链接资产目录：{packages}")
    _require(packages.is_dir(), f"找不到已有资产库：{path}。未创建新库。")
    return path


def validate_asset_id(asset_id):
    _require(isinstance(asset_id, str) and ASSET_ID_RE.fullmatch(asset_id),
             f"无效 asset_id：{asset_id!r}；只接受 asset_ 开头的库内标识")
    return asset_id


def package_dir(root, asset_id):
    root = Path(root)
    validate_asset_id(asset_id)
    package = root / "packages" / asset_id
    _require(not package.is_symlink(), f"拒绝符号链接资产包：{package}")
    _require(package.is_dir(), f"资产包不存在：{package}")
    try:
        resolved = package.resolve(strict=True)
    except OSError as exc:
        raise SupplementError(f"资产包无法解析：{package}") from exc
    _require(resolved == package, f"资产包路径不是规范路径：{package}")
    return package


def read_json_object(path, label):
    path = Path(path)
    _require(not path.is_symlink(), f"拒绝符号链接{label}：{path}")
    _require(path.is_file(), f"未找到{label}：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplementError(f"{label} JSON 无法读取：{path}") from exc
    _require(isinstance(value, dict), f"{label} JSON 顶层必须是 object：{path}")
    return value


def read_asset(root, asset_id):
    root = Path(root)
    package = package_dir(root, asset_id)
    asset = read_json_object(package / "asset.json", "资产元数据")
    _require(asset.get("asset_id") == asset_id, "资产元数据 asset_id 与包目录不一致")
    content = asset.get("content") or {}
    digest = content.get("sha256")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest),
             "资产元数据缺少有效的源内容 SHA256")
    return package, asset


def read_analysis(package):
    """Return ``(analysis, sha256)`` or ``(None, None)``; read-only."""
    path = Path(package) / "analysis.json"
    _require(not path.is_symlink(), f"拒绝符号链接分析文件：{path}")
    if not path.exists():
        return None, None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SupplementError(f"分析 JSON 无法读取：{path}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupplementError(f"分析 JSON 损坏，无法读取：{path}") from exc
    _require(isinstance(value, dict), f"分析 JSON 顶层必须是 object：{path}")
    return value, digest


def media_reference(root, asset):
    """Return ``(absolute_path, relative_path, status)`` without writing."""
    root = Path(root)
    media = asset.get("media") or {}
    relative = media.get("relative_path")
    _require(isinstance(relative, str) and relative,
             "资产元数据缺少媒体相对路径")
    relay = Path(relative)
    _require(not relay.is_absolute() and ".." not in relay.parts,
             f"不安全的媒体相对路径：{relative}")
    current = root
    for part in relay.parts:
        current = current / part
        _require(not current.is_symlink(), f"拒绝符号链接媒体路径：{current}")
    candidate = (root / relay).resolve()
    _require(candidate == root or root in candidate.parents,
             f"媒体路径越出资产库：{relative}")
    status = "available" if candidate.is_file() else "offline"
    return candidate, relative, status


def analysis_duration(analysis):
    if not isinstance(analysis, dict):
        return None
    video = analysis.get("video") or {}
    duration = video.get("duration_seconds")
    if _finite_number(duration) and float(duration) > 0:
        return float(duration)
    return None


def known_speaker_labels(analysis):
    labels = set()
    if not isinstance(analysis, dict):
        return labels
    speakers = analysis.get("speakers") or {}
    for segment in speakers.get("segments") or []:
        if isinstance(segment, dict):
            label = segment.get("speaker")
            if isinstance(label, str) and SPEAKER_RE.fullmatch(label):
                labels.add(label)
    transcript = analysis.get("transcript") or {}
    for segment in transcript.get("raw_segments") or []:
        if isinstance(segment, dict):
            label = segment.get("speaker")
            if isinstance(label, str) and SPEAKER_RE.fullmatch(label):
                labels.add(label)
    return labels


def source_evidence_ids(analysis):
    ids = set()
    if isinstance(analysis, dict):
        for item in analysis.get("evidence") or []:
            if isinstance(item, dict):
                evidence_id = item.get("evidence_id")
                if isinstance(evidence_id, str) and evidence_id:
                    ids.add(evidence_id)
    return ids


def source_evidence_paths(analysis):
    """Map each source evidence ID to its optional package-relative path.

    Validation is deliberately lazy: only IDs actually referenced by an
    annotation are checked, and the path is validated for structure and
    containment without requiring the historical evidence file to exist.
    """
    paths = {}
    if isinstance(analysis, dict):
        for item in analysis.get("evidence") or []:
            if isinstance(item, dict):
                evidence_id = item.get("evidence_id")
                if isinstance(evidence_id, str) and evidence_id:
                    paths[evidence_id] = item.get("relative_path")
    return paths


# --------------------------------------------------------------------------------------
# Annotation validation
# --------------------------------------------------------------------------------------

def _check_keys(value, allowed, required, label):
    _require(isinstance(value, dict), f"{label} 必须是 object")
    missing = sorted(set(required) - set(value))
    extra = sorted(set(value) - set(allowed))
    _require(not missing, f"{label} 缺少字段：{', '.join(missing)}")
    _require(not extra, f"{label} 含未知字段：{', '.join(extra)}")


def _validate_evidence_path(package, relative_path, label, evidence_id):
    """Reject a referenced evidence path that is not safely package-relative.

    Historical evidence bytes are not required to exist: this validates the
    recorded path's structure and containment only.
    """
    _require(_nonempty_text(relative_path),
             f"{label} 引用的证据 {evidence_id} 的 relative_path 必须是非空文本")
    try:
        path = Path(relative_path)
        parts = path.parts
    except (OSError, ValueError, TypeError) as exc:
        raise SupplementError(
            f"{label} 引用的证据 {evidence_id} 的 relative_path 无效：{relative_path!r}") from exc
    _require(not path.is_absolute(),
             f"{label} 引用的证据 {evidence_id} 不允许绝对路径：{relative_path}")
    _require(parts and all(part not in ("", ".", "..") for part in parts),
             f"{label} 引用的证据 {evidence_id} 不允许跳出资产包：{relative_path}")
    package = Path(package)
    candidate = package
    for part in parts:
        candidate = candidate / part
        try:
            is_link = candidate.is_symlink()
        except (OSError, ValueError) as exc:
            raise SupplementError(
                f"{label} 引用的证据 {evidence_id} 的 relative_path 无法检查：{relative_path}") from exc
        _require(not is_link,
                 f"{label} 引用的证据 {evidence_id} 的路径组件不得是符号链接：{relative_path}")
    try:
        resolved = candidate.resolve(strict=False)
        package_resolved = package.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SupplementError(
            f"{label} 引用的证据 {evidence_id} 的 relative_path 无法解析：{relative_path}") from exc
    _require(resolved == package_resolved or package_resolved in resolved.parents,
             f"{label} 引用的证据 {evidence_id} 的 relative_path 越出资产包：{relative_path}")


def _validate_locator(locator, index, evidence_ids, duration, package=None,
                      evidence_paths=None):
    label = f"assertions[{index}].evidence_refs"
    _require(isinstance(locator, dict), f"{label} 的 locator 必须是 object")
    kind = locator.get("kind")
    if kind == "package_evidence":
        _check_keys(locator, {"kind", "evidence_id"}, {"kind", "evidence_id"}, label)
        evidence_id = locator.get("evidence_id")
        _require(_nonempty_text(evidence_id), f"{label} package_evidence 缺少 evidence_id")
        _require(evidence_id in evidence_ids,
                 f"{label} 引用了源分析中不存在的 evidence_id：{evidence_id}")
        if evidence_paths is not None and package is not None:
            recorded_path = evidence_paths.get(evidence_id)
            if recorded_path is not None:
                _validate_evidence_path(package, recorded_path, label, evidence_id)
        return {"kind": "package_evidence", "evidence_id": evidence_id}
    if kind == "media_range":
        _check_keys(locator, {"kind", "start", "end"}, {"kind", "start", "end"}, label)
        start, end = locator.get("start"), locator.get("end")
        _require(_finite_number(start) and _finite_number(end),
                 f"{label} media_range start/end 必须是有限数")
        start, end = float(start), float(end)
        _require(start >= 0, f"{label} media_range start 必须 >= 0")
        _require(end > start, f"{label} media_range end 必须大于 start")
        _require(duration is not None,
                 f"{label} media_range 无法校验：源分析未记录可用视频时长")
        _require(end <= duration + 1e-9,
                 f"{label} media_range end {end} 超出可用视频时长 {duration}")
        return {"kind": "media_range", "start": start, "end": end}
    if kind == "source_text":
        _check_keys(locator, {"kind", "source_label", "locator", "excerpt"},
                    {"kind", "source_label", "locator", "excerpt"}, label)
        for field in ("source_label", "locator", "excerpt"):
            _require(_nonempty_text(locator.get(field)),
                     f"{label} source_text.{field} 必须是非空文本")
        # Attributed Agent-supplied text only: never read the locator as a path
        # or request it over the network.
        return {"kind": "source_text", "source_label": locator["source_label"],
                "locator": locator["locator"], "excerpt": locator["excerpt"]}
    raise SupplementError(f"{label} 含未知 locator kind：{kind!r}")


def _validate_assertion(assertion, index, evidence_ids, speaker_labels, duration,
                        recorded=False, package=None, evidence_paths=None):
    label = f"assertions[{index}]"
    _require(isinstance(assertion, dict), f"{label} 必须是 object")
    kind = assertion.get("kind")
    _require(kind in ANNOTATION_KINDS, f"{label}.kind 必须是 {list(ANNOTATION_KINDS)} 之一")
    required = {"assertion_id", "kind", "status", "person_id", "value", "reasoning",
                "uncertainty", "confidence", "evidence_refs"}
    allowed = set(required)
    if kind == "voice_person":
        required.add("speaker")
        allowed.add("speaker")
    _check_keys(assertion, allowed, required, label)

    assertion_id = assertion.get("assertion_id")
    _require(_nonempty_text(assertion_id), f"{label}.assertion_id 必须是非空文本")

    status = assertion.get("status")
    _require(status in ANNOTATION_STATUSES,
             f"{label}.status 必须是 {list(ANNOTATION_STATUSES)} 之一")

    person_id = assertion.get("person_id")
    _require(person_id is None or _nonempty_text(person_id),
             f"{label}.person_id 必须是录音内人物标识或 null")

    speaker = assertion.get("speaker")
    if kind == "voice_person":
        _require(isinstance(speaker, str) and SPEAKER_RE.fullmatch(speaker),
                 f"{label}.speaker 必须是现有录音内 S 编号")
        _require(speaker in speaker_labels,
                 f"{label}.speaker 不是源分析中已知的说话人编号：{speaker}")

    value = assertion.get("value")
    if kind == "voice_person":
        _require(value is None, f"{label}.value 在 voice_person 时必须是 null")
    else:
        if status in ("supported", "tentative"):
            _require(_nonempty_text(value),
                     f"{label}.value 在 {status} 时不能为空；不要用空值伪造支持")
        else:
            _require(value is None or _nonempty_text(value),
                     f"{label}.value 必须是文本或 null")

    reasoning = assertion.get("reasoning")
    _require(_nonempty_text(reasoning), f"{label}.reasoning 必须是非空归因理由")
    uncertainty = assertion.get("uncertainty")
    _require(_nonempty_text(uncertainty), f"{label}.uncertainty 必须说明剩余不确定性")

    confidence = assertion.get("confidence")
    if confidence is not None:
        _require(_finite_number(confidence) and 0.0 <= float(confidence) <= 1.0,
                 f"{label}.confidence 必须是 [0,1] 内有限数或 null")
        confidence = float(confidence)

    refs = assertion.get("evidence_refs")
    _require(isinstance(refs, list) and refs, f"{label}.evidence_refs 必须是非空数组")
    canonical_refs = [
        _validate_locator(locator, index, evidence_ids, duration,
                          package=package, evidence_paths=evidence_paths)
        for locator in refs
    ]

    canonical = {
        "assertion_id": assertion_id,
        "kind": kind,
        "status": status,
        "person_id": person_id,
        "value": value,
        "reasoning": reasoning,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "evidence_refs": canonical_refs,
    }
    if kind == "voice_person":
        canonical = {"assertion_id": assertion_id, "kind": kind, "status": status,
                     "person_id": person_id, "speaker": speaker, "value": None,
                     "reasoning": reasoning, "uncertainty": uncertainty,
                     "confidence": confidence, "evidence_refs": canonical_refs}
    return canonical


def validate_annotation_shape(value, *, asset_id, evidence_ids, speaker_labels, duration,
                              package=None, evidence_paths=None):
    """Validate the versioned Agent annotation object and return canonical fields.

    ``package``/``evidence_paths`` are supplied only when validating fresh input
    against the current source analysis. Stored supplement documents pass
    neither, so historical evidence paths are never re-resolved.
    """
    allowed = {"schema", "schema_version", "asset_id", "author", "supersedes", "assertions"}
    required = {"schema", "schema_version", "asset_id", "author", "assertions"}
    _check_keys(value, allowed, required, "annotations")
    _require(value.get("schema") == ANNOTATION_SCHEMA,
             f"schema 必须是 {ANNOTATION_SCHEMA}")
    _require(value.get("schema_version") == ANNOTATION_SCHEMA_VERSION,
             f"schema_version 必须是 {ANNOTATION_SCHEMA_VERSION}")
    _require(value.get("asset_id") == asset_id,
             f"asset_id 与目标资产不一致：{value.get('asset_id')!r} != {asset_id!r}")
    _require(_nonempty_text(value.get("author")), "author 必须是非空 harness/名称")

    supersedes = value.get("supersedes")
    if supersedes is not None:
        _require(isinstance(supersedes, str) and SUPPLEMENT_ID_RE.fullmatch(supersedes),
                 "supersedes 必须是已有 supplement ID")

    assertions = value.get("assertions")
    _require(isinstance(assertions, list) and assertions, "assertions 必须是非空数组")
    canonical_assertions = []
    seen = set()
    for index, assertion in enumerate(assertions):
        canonical = _validate_assertion(assertion, index, evidence_ids,
                                        speaker_labels, duration,
                                        package=package, evidence_paths=evidence_paths)
        assertion_id = canonical["assertion_id"]
        _require(assertion_id not in seen,
                 f"assertion_id 在本次输入内重复：{assertion_id}")
        seen.add(assertion_id)
        canonical_assertions.append(canonical)

    canonical = {
        "schema": ANNOTATION_SCHEMA,
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "asset_id": asset_id,
        "author": value["author"],
        "assertions": canonical_assertions,
    }
    if supersedes is not None:
        canonical["supersedes"] = supersedes
    return canonical


def _validate_annotation_against_analysis(value, asset_id, analysis, package):
    return validate_annotation_shape(
        value, asset_id=asset_id, evidence_ids=source_evidence_ids(analysis),
        speaker_labels=known_speaker_labels(analysis),
        duration=analysis_duration(analysis),
        package=package, evidence_paths=source_evidence_paths(analysis))


# --------------------------------------------------------------------------------------
# Publishing, locks and lookup
# --------------------------------------------------------------------------------------

def generate_supplement_id(kind):
    stamp = utc_now().replace("-", "").replace(":", "")
    prefix = "sup_em_" if kind == "engine_measurement" else "sup_an_"
    return f"{prefix}{stamp}_{secrets.token_hex(6)}"


def supplements_dir(root, asset_id, create=False):
    package = package_dir(root, asset_id)
    directory = package / SUPPLEMENTS_DIRNAME
    _require(not directory.is_symlink(), f"拒绝符号链接补充目录：{directory}")
    if directory.exists():
        _require(directory.is_dir(), f"补充目录不是目录：{directory}")
        return directory
    if not create:
        return None
    directory.mkdir(mode=0o700)
    return directory


@contextmanager
def asset_lock(root, asset_id):
    """Exclusive non-blocking lock using the existing per-asset lock protocol."""
    root = Path(root)
    locks = root / ".locks"
    _require(not locks.is_symlink(), f"拒绝符号链接锁目录：{locks}")
    if locks.exists():
        _require(locks.is_dir(), f"锁路径不是目录：{locks}")
    else:
        locks.mkdir(mode=0o700)
    lock_path = locks / f"{asset_id}.lock"
    _require(not lock_path.is_symlink(), f"拒绝符号链接锁文件：{lock_path}")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        _require(stat.S_ISREG(os.fstat(fd).st_mode), "资产锁不是普通文件")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupplementError(
                f"资产 {asset_id} 正在被其他写入操作占用（重分析/保存/注释）；本次未写入") from exc
        yield
    finally:
        os.close(fd)


def _write_document(directory, document):
    supplement_id = document["supplement_id"]
    destination = directory / f"{supplement_id}.json"
    _require(not destination.is_symlink(), f"拒绝符号链接目标：{destination}")
    fd, temporary = tempfile.mkstemp(prefix=f".{supplement_id}.", suffix=".tmp",
                                     dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Exclusive hard link: either the complete document appears or the
            # existing file is left untouched.  os.replace would overwrite.
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise SupplementError(
                f"补充文档已存在，拒绝覆盖：{supplement_id}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return destination


def _immutable_buffer(document, asset_id, directory):
    document["asset_id"] = asset_id
    destination = _write_document(directory, document)
    return {
        "supplement_id": document["supplement_id"],
        "path_relative": f"packages/{asset_id}/{SUPPLEMENTS_DIRNAME}/{destination.name}",
        "sha256": sha256_file(destination),
    }


def _provenance(asset_id, asset, analysis_sha256, media_relative, analysis, analysis_present):
    return {
        "source_content_sha256": asset["content"]["sha256"],
        "source_analysis_sha256": analysis_sha256,
        "analysis_present": bool(analysis_present),
        "analysis_relative_path": f"packages/{asset_id}/analysis.json",
        "media_relative_path": media_relative,
        "path_base": "library_root",
        "available_video_duration_seconds": analysis_duration(analysis),
        "source_evidence_ids": sorted(source_evidence_ids(analysis)),
        "known_speaker_labels": sorted(known_speaker_labels(analysis)),
    }


def _supersedes_document(root, asset_id, supplement_id):
    document = load_supplement(root, asset_id, supplement_id)
    _require(document.get("kind") == "agent_annotations",
             f"supersedes 只能指向同一资产的 Agent 注释：{supplement_id}")
    _require(document.get("asset_id") == asset_id,
             f"supersedes 跨资产引用被拒绝：{supplement_id}")
    return document


def save_annotations(root, asset_id, payload):
    """Validate and persist one immutable Agent annotation record."""
    root = library_root(root)
    asset_id = validate_asset_id(asset_id)
    package, asset = read_asset(root, asset_id)
    analysis, analysis_sha256 = read_analysis(package)
    _require(analysis is not None, "资产尚未有 analysis.json，无法记录证据引用")
    # First validation happens before any lock directory/file is created.
    canonical = _validate_annotation_against_analysis(payload, asset_id, analysis, package)
    _, media_relative, _ = media_reference(root, asset)
    # Reject a non-directory or symlinked supplement path and an invalid
    # supersedes target before the lock directory exists, so invalid input
    # leaves the library untouched.
    supplements_dir(root, asset_id, create=False)
    if canonical.get("supersedes") is not None:
        _supersedes_document(root, asset_id, canonical["supersedes"])

    with asset_lock(root, asset_id):
        package, asset = read_asset(root, asset_id)
        analysis, analysis_sha256 = read_analysis(package)
        _require(analysis is not None, "资产正在重分析且尚无 analysis.json；本次未写入")
        canonical = _validate_annotation_against_analysis(payload, asset_id, analysis, package)
        if canonical.get("supersedes") is not None:
            _supersedes_document(root, asset_id, canonical["supersedes"])

        supplement_id = generate_supplement_id("agent_annotations")
        document = {
            "schema": SUPPLEMENT_SCHEMA,
            "schema_version": SUPPLEMENT_SCHEMA_VERSION,
            "supplement_id": supplement_id,
            "kind": "agent_annotations",
            "asset_id": asset_id,
            "created_at": utc_now(),
            "source": ANNOTATION_SOURCE,
            "author": canonical["author"],
            "supersedes": canonical.get("supersedes"),
            "input": canonical,
            "provenance": _provenance(asset_id, asset, analysis_sha256,
                                      media_relative, analysis, True),
        }
        directory = supplements_dir(root, asset_id, create=True)
        published = _immutable_buffer(document, asset_id, directory)
        published["relationship"] = relationship_for(document, analysis_sha256)
        return published


def save_engine_measurement(root, asset_id, measurement, expected_analysis_sha256):
    """Persist one usable engine measurement as a portable immutable document."""
    root = library_root(root)
    asset_id = validate_asset_id(asset_id)
    _require(isinstance(measurement, dict), "measurement 必须是 object")
    status = measurement.get("status")
    _require(status in USABLE_MEASUREMENT_STATUSES,
             f"只保存可用测量（{list(USABLE_MEASUREMENT_STATUSES)}）；当前状态：{status!r}")

    package, asset = read_asset(root, asset_id)
    analysis, current_sha = read_analysis(package)
    _require(analysis is not None, "资产尚未有 analysis.json，无法保存测量来源")
    _require(expected_analysis_sha256 == current_sha,
             "测量期间 analysis.json 已变化，拒绝把旧测量标记为当前分析")
    media_path, media_relative, media_status = media_reference(root, asset)
    _require(media_status == "available", "归档原片不可用，拒绝保存测量")
    supplements_dir(root, asset_id, create=False)
    source = measurement.get("source") or {}
    measured_digest = source.get("digest_sha256")
    _require(isinstance(measured_digest, str) and SHA256_RE.fullmatch(measured_digest),
             "测量缺少有效的源内容 SHA256")
    _require(measured_digest == asset["content"]["sha256"],
             "测量源内容与资产记录的归档内容不一致，拒绝保存")

    with asset_lock(root, asset_id):
        package, asset = read_asset(root, asset_id)
        analysis, current_sha = read_analysis(package)
        _require(analysis is not None, "资产分析在保存前消失；本次未写入")
        _require(expected_analysis_sha256 == current_sha,
                 "测量期间 analysis.json 已变化，拒绝把旧测量标记为当前分析")
        _require(measurement["source"]["digest_sha256"] == asset["content"]["sha256"],
                 "保存前源内容摘要与资产记录不一致")

        supplement_id = generate_supplement_id("engine_measurement")
        portable = portable_measurement(measurement, root, media_path, media_relative)
        document = {
            "schema": SUPPLEMENT_SCHEMA,
            "schema_version": SUPPLEMENT_SCHEMA_VERSION,
            "supplement_id": supplement_id,
            "kind": "engine_measurement",
            "asset_id": asset_id,
            "created_at": utc_now(),
            "source": ENGINE_SOURCE,
            "author": "reelpack.motion_analyze",
            "backend": {
                "name": measurement.get("backend"),
                "version": measurement.get("backend_version"),
            },
            "request": copy.deepcopy(measurement.get("request")),
            "parameters": copy.deepcopy(measurement.get("parameters")),
            "measurement": portable,
            "provenance": _provenance(asset_id, asset, current_sha, media_relative,
                                      analysis, True),
        }
        directory = supplements_dir(root, asset_id, create=True)
        published = _immutable_buffer(document, asset_id, directory)
        published["relationship"] = relationship_for(document, current_sha)
        return published


# --------------------------------------------------------------------------------------
# Portable runtime output
# --------------------------------------------------------------------------------------

def _portable_string(value, root, media_path, media_relative):
    """Rewrite library/media prefixes, including paths embedded in messages."""
    if not isinstance(value, str):
        return value
    media_text = str(media_path)
    root_text = str(root)
    if media_text in value or root_text in value:
        # Media is beneath the root, so replace the longer prefix first.  The
        # exact-path cases become the same relative forms as the branches below.
        return value.replace(media_text, media_relative).replace(root_text, "library_root")
    try:
        absolute = Path(value)
    except (OSError, ValueError, TypeError):
        return value
    if not absolute.is_absolute():
        return value
    return "<absolute_path_removed>"


def _scrub(value, root, media_path, media_relative):
    if isinstance(value, str):
        return _portable_string(value, root, media_path, media_relative)
    if isinstance(value, list):
        return [_scrub(item, root, media_path, media_relative) for item in value]
    if isinstance(value, dict):
        return {key: _scrub(item, root, media_path, media_relative)
                for key, item in value.items()}
    return value


def portable_measurement(measurement, root, media_path, media_relative):
    """Return a JSON-safe copy with library-relative, portable references."""
    portable = copy.deepcopy(measurement)
    source = portable.get("source")
    if isinstance(source, dict):
        source.pop("path", None)
        source.pop("resolved_path", None)
        source["media_relative_path"] = media_relative
        source["path_base"] = "library_root"
    portable = _scrub(portable, root, media_path, media_relative)
    # The scrub above can only introduce the portable forms; never leave an
    # absolute machine path behind in a persisted reference.
    return portable


# --------------------------------------------------------------------------------------
# Readback and relationship
# --------------------------------------------------------------------------------------

def relationship_for(document, current_analysis_sha256):
    recorded = ((document.get("provenance") or {}).get("source_analysis_sha256"))
    if current_analysis_sha256 is None:
        status, reason = "unknown", "current_analysis_missing"
    elif recorded == current_analysis_sha256:
        status, reason = "current", "recorded_against_current_analysis"
    else:
        status, reason = "stale", "current_analysis_differs_from_recorded_source"
    return {
        "status": status,
        "reason": reason,
        "recorded_source_analysis_sha256": recorded,
        "current_source_analysis_sha256": current_analysis_sha256,
        "evidence_resolution": ("not_resolved" if status != "current"
                                else "recorded_refs_belong_to_current_analysis"),
    }


def _validate_provenance(provenance):
    _check_keys(provenance,
                {"source_content_sha256", "source_analysis_sha256", "analysis_present",
                 "analysis_relative_path", "media_relative_path", "path_base",
                 "available_video_duration_seconds", "source_evidence_ids",
                 "known_speaker_labels"},
                {"source_content_sha256", "source_analysis_sha256", "analysis_present",
                 "analysis_relative_path", "media_relative_path", "path_base",
                 "available_video_duration_seconds", "source_evidence_ids",
                 "known_speaker_labels"}, "provenance")
    for field in ("source_content_sha256", "source_analysis_sha256"):
        _require(isinstance(provenance.get(field), str)
                 and SHA256_RE.fullmatch(provenance[field]),
                 f"provenance.{field} 必须是 SHA256")
    _require(provenance.get("path_base") == "library_root",
             "provenance.path_base 必须显式为 library_root")
    _require(isinstance(provenance.get("analysis_relative_path"), str),
             "provenance.analysis_relative_path 必须是文本")
    _require(provenance.get("media_relative_path") is None
             or isinstance(provenance.get("media_relative_path"), str),
             "provenance.media_relative_path 必须是文本或 null")
    duration = provenance.get("available_video_duration_seconds")
    _require(duration is None or (_finite_number(duration) and float(duration) > 0),
             "provenance.available_video_duration_seconds 必须是有限正数或 null")
    _require(isinstance(provenance.get("source_evidence_ids"), list)
             and all(isinstance(item, str) for item in provenance["source_evidence_ids"]),
             "provenance.source_evidence_ids 必须是字符串数组")
    _require(isinstance(provenance.get("known_speaker_labels"), list)
             and all(isinstance(item, str) and SPEAKER_RE.fullmatch(item)
                     for item in provenance["known_speaker_labels"]),
             "provenance.known_speaker_labels 必须是 S 编号数组")
    _require(isinstance(provenance.get("analysis_present"), bool),
             "provenance.analysis_present 必须是 boolean")


def validate_supplement_document(document, asset_id=None):
    """Validate one stored supplement without consulting a newer analysis."""
    _check_keys(document,
                {"schema", "schema_version", "supplement_id", "kind", "asset_id",
                 "created_at", "source", "author", "supersedes", "input",
                 "backend", "request", "parameters", "measurement", "provenance"},
                {"schema", "schema_version", "supplement_id", "kind", "asset_id",
                 "created_at", "source", "author", "provenance"}, "supplement")
    _require(document.get("schema") == SUPPLEMENT_SCHEMA,
             f"补充文档 schema 必须是 {SUPPLEMENT_SCHEMA}")
    _require(document.get("schema_version") == SUPPLEMENT_SCHEMA_VERSION,
             f"补充文档 schema_version 必须是 {SUPPLEMENT_SCHEMA_VERSION}")
    supplement_id = document.get("supplement_id")
    _require(isinstance(supplement_id, str) and SUPPLEMENT_ID_RE.fullmatch(supplement_id),
             "补充文档 supplement_id 无效")
    _require(asset_id is None or document.get("asset_id") == asset_id,
             "补充文档跨资产引用被拒绝")
    _require(isinstance(document.get("asset_id"), str)
             and ASSET_ID_RE.fullmatch(document["asset_id"]),
             "补充文档 asset_id 无效")
    _require(isinstance(document.get("created_at"), str) and document["created_at"].strip(),
             "补充文档缺少 created_at")
    provenance = document.get("provenance")
    _validate_provenance(provenance)

    kind = document.get("kind")
    if kind == "agent_annotations":
        _require(document.get("source") == ANNOTATION_SOURCE,
                 "Agent 注释 source 必须是 downstream_agent")
        _require(_nonempty_text(document.get("author")), "Agent 注释缺少 author")
        _check_keys(document,
                    {"schema", "schema_version", "supplement_id", "kind", "asset_id",
                     "created_at", "source", "author", "supersedes", "input", "provenance"},
                    {"schema", "schema_version", "supplement_id", "kind", "asset_id",
                     "created_at", "source", "author", "supersedes", "input", "provenance"},
                    "supplement")
        canonical = validate_annotation_shape(
            document.get("input"), asset_id=document["asset_id"],
            evidence_ids=set(provenance["source_evidence_ids"]),
            speaker_labels=set(provenance["known_speaker_labels"]),
            duration=provenance["available_video_duration_seconds"])
        _require(document.get("supersedes") == canonical.get("supersedes"),
                 "补充文档 supersedes 与 canonical input 不一致")
        return document
    if kind == "engine_measurement":
        _require(document.get("source") == ENGINE_SOURCE,
                 "引擎测量 source 必须是 engine")
        _check_keys(document,
                    {"schema", "schema_version", "supplement_id", "kind", "asset_id",
                     "created_at", "source", "author", "backend", "request",
                     "parameters", "measurement", "provenance"},
                    {"schema", "schema_version", "supplement_id", "kind", "asset_id",
                     "created_at", "source", "author", "backend", "request",
                     "parameters", "measurement", "provenance"}, "supplement")
        _require(_nonempty_text(document.get("author")), "引擎测量缺少 author")
        _require(isinstance(document.get("backend"), dict), "引擎测量缺少 backend")
        _require(isinstance(document.get("request"), dict), "引擎测量缺少 request")
        _require(isinstance(document.get("parameters"), dict), "引擎测量缺少 parameters")
        measurement = document.get("measurement")
        _require(isinstance(measurement, dict), "引擎测量缺少 measurement")
        _require(measurement.get("status") in USABLE_MEASUREMENT_STATUSES,
                 "引擎测量只允许可用状态")
        source = measurement.get("source") or {}
        _require(source.get("digest_sha256") == provenance["source_content_sha256"],
                 "引擎测量源摘要与 provenance 不一致")
        _require(source.get("media_relative_path") == provenance["media_relative_path"],
                 "引擎测量媒体相对路径与 provenance 不一致")
        _require(source.get("path_base") == "library_root",
                 "引擎测量缺少显式 library_root 基址")
        return document
    raise SupplementError(f"未知补充文档 kind：{kind!r}")


def load_supplement(root, asset_id, supplement_id):
    root = library_root(root)
    asset_id = validate_asset_id(asset_id)
    _require(isinstance(supplement_id, str) and SUPPLEMENT_ID_RE.fullmatch(supplement_id),
             f"无效 supplement ID：{supplement_id!r}")
    directory = supplements_dir(root, asset_id, create=False)
    _require(directory is not None, f"资产没有补充目录：{asset_id}")
    path = directory / f"{supplement_id}.json"
    _require(not path.is_symlink(), f"拒绝符号链接补充文件：{path}")
    _require(path.is_file(), f"补充文档不存在：{supplement_id}")
    document = read_json_object(path, "补充文档")
    validate_supplement_document(document, asset_id=asset_id)
    _require(document.get("supplement_id") == supplement_id,
             "补充文档 ID 与文件名不一致")
    return document


def list_supplements(root, asset_id, supplement_id=None):
    root = library_root(root)
    asset_id = validate_asset_id(asset_id)
    package, asset = read_asset(root, asset_id)
    analysis, current_sha = read_analysis(package)
    media_path, media_relative, media_status = media_reference(root, asset)

    if supplement_id is not None:
        document = load_supplement(root, asset_id, supplement_id)
        result = {
            "schema": "reelpack.supplement-readback",
            "schema_version": 1,
            "asset_id": asset_id,
            "media_status": media_status,
            "supplement": document,
            "relationship": relationship_for(document, current_sha),
        }
        if result["relationship"]["status"] == "stale":
            result["warnings"] = [{
                "code": "stale_source_analysis",
                "message": "该补充记录针对旧分析保存；其 evidence_id 不会在当前分析中被重新解析。",
            }]
        return result

    directory = supplements_dir(root, asset_id, create=False)
    entries, warnings = [], []
    if directory is not None:
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.name.startswith(".") or path.suffix != ".json":
                continue
            if path.is_symlink() or not path.is_file():
                warnings.append({"file": path.name, "reason": "不是普通文件；已跳过"})
                continue
            try:
                document = read_json_object(path, "补充文档")
                validate_supplement_document(document, asset_id=asset_id)
                _require(document.get("supplement_id") == path.stem,
                         "补充文档 ID 与文件名不一致")
            except (SupplementError, OSError, ValueError) as exc:
                warnings.append({"file": path.name, "reason": str(exc)})
                continue
            relationship = relationship_for(document, current_sha)
            entry = {
                "supplement_id": document["supplement_id"],
                "kind": document["kind"],
                "source": document["source"],
                "author": document["author"],
                "created_at": document["created_at"],
                "schema": document["schema"],
                "schema_version": document["schema_version"],
                "supersedes": document.get("supersedes"),
                "recorded_source_content_sha256":
                    document["provenance"]["source_content_sha256"],
                "recorded_source_analysis_sha256":
                    document["provenance"]["source_analysis_sha256"],
                "relationship": relationship,
            }
            if document["kind"] == "agent_annotations":
                entry["assertion_count"] = len(document["input"]["assertions"])
                entry["assertions"] = document["input"]["assertions"]
            else:
                entry["measurement_status"] = document["measurement"].get("status")
            entries.append(entry)
    entries.sort(key=lambda entry: (entry["created_at"], entry["supplement_id"]))
    return {
        "schema": "reelpack.supplement-list",
        "schema_version": 1,
        "asset_id": asset_id,
        "media_status": media_status,
        "count": len(entries),
        "supplements": entries,
        "warnings": warnings,
        "analysis_present": analysis is not None,
        "recorded_analysis_sha256": current_sha,
        "path_base": "library_root",
        "media_relative_path": media_relative,
    }
