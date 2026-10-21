"""Isolated per-asset attempts; publish whole packages with a macOS exchange."""
from __future__ import annotations

import copy
import ctypes
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import uuid

from asset_contract import assert_valid_analysis
from asset_store import AssetStoreError, _atomic_json_write, utc_now


class AssetAttempt:
    """Hold an asset lock until exit, retaining failed and superseded packages.

    ``save_analysis`` is an AssetStore adapter. It never promotes a finished
    attempt implicitly: call ``promote`` or ``save(..., final=True)``. Only an
    asset with no analysis at entry receives public checkpoint snapshots.
    """

    def __init__(self, store, archive):
        self.store = store
        self.root = Path(store.root)
        self.archive = copy.deepcopy(archive)
        self.asset_id = archive.get("asset_id", "")
        if not isinstance(self.asset_id, str) or not re.fullmatch(r"asset_[A-Za-z0-9_-]+", self.asset_id):
            raise AssetStoreError("Invalid asset id")
        self.canonical_dir = self.root / "packages" / self.asset_id
        supplied = Path(archive["package_dir"])
        self._check(supplied)
        if supplied != self.canonical_dir:
            raise AssetStoreError("archive.package_dir must be the canonical package")
        self.attempt_dir = self.root / ".attempts" / self.asset_id / uuid.uuid4().hex
        self.package_dir = self.attempt_dir / "package"
        self.previous_dir = None
        self._lock_fd = None
        self._active = False
        self._promoted = False
        self._failed = False
        self._started_at = utc_now()
        self._publish_checkpoints = False

    def _check(self, path, *, tree=False):
        """Reject lexical escapes and symlinks before filesystem operations."""
        path = Path(path)
        if not path.is_absolute() or ".." in path.parts:
            raise AssetStoreError(f"Unsafe path: {path}")
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise AssetStoreError(f"Path outside library: {path}") from exc
        for part in (path, *path.parents):
            if part.is_symlink():
                raise AssetStoreError(f"Symlink rejected: {part}")
        if path.resolve() != path:
            raise AssetStoreError(f"Non-canonical path: {path}")
        if path.exists() and not (path.is_file() or path.is_dir()):
            raise AssetStoreError(f"Special file rejected: {path}")
        if tree and path.is_dir():
            for directory, dirs, files in os.walk(path, followlinks=False):
                for name in dirs + files:
                    self._check(Path(directory) / name)
        return path

    def _require_active(self):
        if not self._active or self._promoted:
            raise RuntimeError("AssetAttempt is not an active writable attempt")
        self._check(self.package_dir, tree=True)
        self._check(self.canonical_dir, tree=True)

    def __enter__(self):
        if self._active or self._lock_fd is not None or self.attempt_dir.exists():
            raise RuntimeError("AssetAttempt cannot be entered twice")
        locks = self._check(self.root / ".locks")
        locks.mkdir(parents=True, exist_ok=True)
        lock_path = self._check(locks / f"{self.asset_id}.lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise AssetStoreError("Asset lock is not a regular file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"An attempt is already active for {self.asset_id}") from exc
            self._lock_fd = fd
            self._check(self.canonical_dir, tree=True)
            asset = self._read_json(self.canonical_dir / "asset.json")
            if asset.get("asset_id") != self.asset_id:
                raise AssetStoreError("Canonical asset metadata id mismatch")
            old_analysis = self.canonical_dir / "analysis.json"
            self._publish_checkpoints = not old_analysis.exists()
            if old_analysis.exists():
                old = self._read_json(old_analysis)
                self._validate(old)
            self._check(self.package_dir)
            self.package_dir.mkdir(parents=True)
            _atomic_json_write(self._check(self.package_dir / "asset.json"), copy.deepcopy(asset))
            self._active = True
            self._manifest("in_progress")
            return self
        except BaseException:
            self._active = False
            self._lock_fd = None
            os.close(fd)
            raise

    def _read_json(self, path):
        self._check(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AssetStoreError(f"Cannot read package JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise AssetStoreError(f"Package JSON must be an object: {path}")
        return payload

    def _validate(self, payload):
        try:
            assert_valid_analysis(payload)
        except (ValueError, TypeError, AttributeError) as exc:
            raise AssetStoreError(str(exc)) from exc
        if payload.get("asset_id") != self.asset_id:
            raise AssetStoreError("Analysis asset id mismatch")
        execution = payload.get("execution")
        if execution is not None and not isinstance(execution, dict):
            raise AssetStoreError("Analysis execution must be an object")
        for evidence in payload.get("evidence", []):
            relative = evidence.get("relative_path")
            if relative is not None:
                rel = Path(relative)
                if rel.is_absolute() or ".." in rel.parts:
                    raise AssetStoreError("Unsafe evidence relative path")
                self._check(self.package_dir / rel)

    def _render_markdown(self, asset, analysis):
        return self.store._render_markdown(asset, analysis)

    @staticmethod
    def _paths(directory):
        return {"analysis_path": str(directory / "analysis.json"),
                "markdown_path": str(directory / "analysis.md"),
                "package_dir": str(directory)}

    def save_analysis(self, asset_id, analysis):
        self._require_active()
        if asset_id != self.asset_id or not isinstance(analysis, dict):
            raise AssetStoreError("Invalid target asset or analysis")
        payload = copy.deepcopy(analysis)
        if payload.get("asset_id") not in (None, self.asset_id):
            raise AssetStoreError("Analysis asset id mismatch")
        payload["asset_id"] = self.asset_id
        payload.setdefault("saved_at", utc_now())
        self._validate(payload)
        execution = payload.setdefault("execution", {})
        if execution is None:
            raise AssetStoreError("Analysis execution must be an object")
        execution.setdefault("attempt_id", self.attempt_dir.name)
        asset = self._read_json(self.package_dir / "asset.json")
        if asset.get("asset_id") != self.asset_id:
            raise AssetStoreError("Staged asset metadata id mismatch")
        markdown = self._render_markdown(asset, payload)
        _atomic_json_write(self._check(self.package_dir / "analysis.json"), payload)
        # This file is private until the containing directory is exchanged.
        self._check(self.package_dir / "analysis.md").write_text(markdown, encoding="utf-8")
        asset["analysis"] = {"status": payload["status"],
                             "analysis_version": payload["analysis_version"],
                             "saved_at": payload["saved_at"]}
        _atomic_json_write(self._check(self.package_dir / "asset.json"), asset)
        execution = payload.get("execution") or {}
        if execution.get("status") == "failed":
            self._failed = True
        if self._publish_checkpoints:
            self._checkpoint()
        return self._paths(self.package_dir)

    def save(self, analysis, final=False):
        paths = self.save_analysis(self.asset_id, analysis)
        return self.promote() if final else paths

    def _exchange(self, source, destination):
        self._check(source, tree=True)
        self._check(destination, tree=True)
        if not source.is_dir() or not destination.is_dir():
            raise AssetStoreError("Exchange requires two existing package directories")
        if sys.platform != "darwin":
            raise AssetStoreError("Atomic package exchange requires macOS; originals retained")
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            exchange = libc.renameatx_np
        except AttributeError as exc:
            raise AssetStoreError("Atomic package exchange is unavailable") from exc
        exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        exchange.restype = ctypes.c_int
        if exchange(-2, os.fsencode(source), -2, os.fsencode(destination), 2) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(source))

    def _checkpoint(self):
        self._require_active()
        snapshot = self._check(self.attempt_dir / f"checkpoint-{uuid.uuid4().hex}")
        shutil.copytree(self.package_dir, snapshot, symlinks=True)
        self._check(snapshot, tree=True)
        # Even a final payload is not cacheable until explicit promotion.
        payload = self._read_json(snapshot / "analysis.json")
        execution = payload.setdefault("execution", {})
        if execution.get("status") != "failed":
            execution["status"] = "in_progress"
        _atomic_json_write(self._check(snapshot / "analysis.json"), payload)
        asset = self._read_json(snapshot / "asset.json")
        self._check(snapshot / "analysis.md").write_text(self._render_markdown(asset, payload), encoding="utf-8")
        self._exchange(snapshot, self.canonical_dir)
        # The old canonical snapshot is deliberately retained for recovery.

    def promote(self):
        self._require_active()
        payload = self._read_json(self.package_dir / "analysis.json")
        self._validate(payload)
        if self._failed or (payload.get("execution") or {}).get("status") != "complete":
            raise AssetStoreError("Only an execution-complete attempt can be promoted")
        previous = self.package_dir
        self._exchange(previous, self.canonical_dir)
        self.previous_dir = previous
        self.package_dir = self.canonical_dir
        self._promoted = True
        return self._paths(self.canonical_dir)

    def failure(self, analysis=None):
        self._require_active()
        self._failed = True
        if analysis is not None:
            payload = copy.deepcopy(analysis)
            payload.setdefault("execution", {})["status"] = "failed"
            return self.save_analysis(self.asset_id, payload)
        return self._paths(self.package_dir)

    def _manifest(self, outcome, error=None):
        manifest = {"asset_id": self.asset_id, "attempt_id": self.attempt_dir.name, "outcome": outcome,
                    "started_at": self._started_at, "updated_at": utc_now(),
                    "source_path": self.archive.get("media_path"),
                    "canonical_package_dir": str(self.canonical_dir),
                    "staged_package_dir": str(self.attempt_dir / "package"),
                    "previous_package_dir": str(self.previous_dir) if self.previous_dir else None}
        if error is not None:
            manifest["error"] = str(error)
        _atomic_json_write(self._check(self.attempt_dir / "attempt.json"), manifest)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._active and self._lock_fd is None:
            return False
        try:
            outcome = "complete" if self._promoted else "failed" if exc_type or self._failed else "incomplete"
            try:
                self._manifest(outcome, exc_value)
            except Exception:
                if self._promoted:
                    logging.getLogger(__name__).warning(
                        "Package promotion succeeded but attempt manifest could not be saved: %s",
                        self.attempt_dir, exc_info=True)
                elif exc_type is None:
                    raise
        finally:
            self._active = False
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
        return False
