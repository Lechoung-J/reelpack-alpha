"""Standard-library runtime paths, cache preparation and read-only diagnostics.

Call ``os.environ.update(prepare_runtime_environment())`` before importing workers.
Resolution and inspection never import an ML library or download a model.  The
permission-only result in diagnostics is deliberately not a sandbox write test.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import stat
import tempfile


_CACHE_CHILDREN = {
    "NUMBA_CACHE_DIR": "numba",
    "MPLCONFIGDIR": "mpl",
    "PYTHONPYCACHEPREFIX": "pycache",
}
_WHISPER_REPOSITORY = "models--Systran--faster-whisper-large-v3"
_SIZE_ENTRY_LIMIT = 4096


def _environment(environment):
    return dict(os.environ if environment is None else environment)


def _home(environment):
    return Path(environment.get("HOME") or str(Path.home())).absolute()


def _path(value, environment):
    text = os.fspath(value)
    if text == "~":
        return _home(environment)
    if text.startswith("~/"):
        return _home(environment) / text[2:]
    return Path(os.path.abspath(os.path.expanduser(text)))


def _cache_candidates(environment):
    primary = _path(environment.get("REELPACK_CACHE_ROOT") or
                    str(_home(environment) / "Library/Caches/Reelpack"), environment)
    # Do not use tempfile.gettempdir(): its first call can itself write a probe.
    temporary = _path(environment.get("TMPDIR") or "/tmp", environment)
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return primary, temporary / ("reelpack-cache-" + str(uid))


def _writable_without_probe(path):
    """Estimate permission to use/create a directory, without changing anything."""
    candidate = path
    while True:
        try:
            attributes = candidate.stat()
        except FileNotFoundError:
            parent = candidate.parent
            if parent == candidate:
                return False
            candidate = parent
            continue
        except OSError:
            return False
        return (stat.S_ISDIR(attributes.st_mode)
                and bool(attributes.st_mode & 0o222)
                and bool(attributes.st_mode & 0o111)
                and os.access(candidate, os.W_OK | os.X_OK))


def _cache_tree_writable_without_probe(root):
    return all(_writable_without_probe(path)
               for path in (root, *(root / name for name in _CACHE_CHILDREN.values())))


def _probe_directory(path):
    """Use exclusive creation, then remove only the exact file we just created."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    fd, filename = tempfile.mkstemp(prefix=".reelpack-write-probe-", dir=str(root))
    probe = Path(filename)
    owned = os.fstat(fd)
    try:
        os.write(fd, b"reelpack\n")
    finally:
        os.close(fd)
        # Check the resolved absolute parent and the inode before any removal.
        # A replacement/symlink is never followed or removed.
        if probe.parent.resolve(strict=True) != root or not probe.name.startswith(".reelpack-write-probe-"):
            raise OSError("cache probe cleanup path changed")
        observed = probe.lstat()
        if (not stat.S_ISREG(observed.st_mode) or observed.st_dev != owned.st_dev
                or observed.st_ino != owned.st_ino):
            raise OSError("cache probe ownership changed")
        probe.unlink()


def _prepare_cache_tree(root, environment):
    resolved = root.resolve()
    if resolved in (Path("/"), _home(environment).resolve()):
        raise OSError("unsafe Reelpack cache root")
    for path in (root, *(root / name for name in _CACHE_CHILDREN.values())):
        _probe_directory(path)


def _model_defaults(environment):
    persistent = _path(environment.get("XDG_CACHE_HOME") or str(_home(environment) / ".cache"), environment)
    # Explicit model locations stay persistent and are never rewritten to the
    # temporary acceleration cache. No model directories are created here.
    if not environment.get("HF_HOME"):
        environment["HF_HOME"] = str(persistent / "huggingface")
    if not environment.get("TORCH_HOME"):
        environment["TORCH_HOME"] = str(persistent / "torch")
    return environment


def _hub_root(environment):
    return _path(environment.get("HF_HUB_CACHE") or environment.get("HUGGINGFACE_HUB_CACHE")
                 or str(_path(environment["HF_HOME"], environment) / "hub"), environment)


def _file_metadata(path):
    try:
        attributes = path.stat()
        exists = stat.S_ISREG(attributes.st_mode)
        return {"path": str(path), "exists": exists,
                "size_bytes": attributes.st_size if exists else 0}
    except OSError:
        return {"path": str(path), "exists": False, "size_bytes": 0}


def _whisper_model(environment):
    """Inspect known HF snapshot filenames; never open/hash model weights."""
    hub = _hub_root(environment)
    repository = hub / _WHISPER_REPOSITORY
    snapshots = repository / "snapshots"
    try:
        candidates = sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.name)
    except OSError:
        candidates = []
    best = None
    for snapshot in candidates:
        vocabulary = snapshot / "vocabulary.json"
        if not vocabulary.is_file():
            vocabulary = snapshot / "vocabulary.txt"
        files = [_file_metadata(snapshot / name)
                 for name in ("model.bin", "config.json", "tokenizer.json")]
        files.append(_file_metadata(vocabulary))
        ready = all(item["exists"] and item["size_bytes"] > 0 for item in files)
        model = {"name": "large-v3", "repository": str(repository), "path": str(snapshot),
                 "ready": ready, "size_bytes": sum(item["size_bytes"] for item in files), "files": files,
                 "verification": "filenames_and_sizes_only", "inference": "not_run"}
        if best is None or model["size_bytes"] > best["size_bytes"]:
            best = model
        if ready:
            return model
    return best or {"name": "large-v3", "repository": str(repository), "path": None,
                    "ready": False, "size_bytes": 0, "files": [],
                    "verification": "filenames_and_sizes_only", "inference": "not_run"}


def _panns_labels(path):
    labels = _file_metadata(path)
    labels.update(ready=False, rows=0, verification="527_nonempty_csv_labels")
    if not labels["exists"] or not 0 < labels["size_bytes"] <= 1024 * 1024:
        return labels
    try:
        # Only the small labels CSV is read. Checkpoints remain stat-only.
        with path.open(encoding="utf-8", newline="") as source:
            rows = csv.DictReader(source)
            valid = True
            for row in rows:
                labels["rows"] += 1
                valid = valid and bool(row.get("display_name"))
                if labels["rows"] > 527:
                    break
            labels["ready"] = valid and labels["rows"] == 527
    except (OSError, ValueError, csv.Error):
        pass
    return labels


def _panns_models(environment):
    data = _home(environment) / "panns_data"
    checkpoint = _file_metadata(_path(environment.get("REELPACK_PANNS_CHECKPOINT")
                                     or str(data / "Cnn14_mAP=0.431.pth"), environment))
    labels = _panns_labels(_path(environment.get("REELPACK_PANNS_LABELS")
                                or str(data / "class_labels_indices.csv"), environment))
    sed = _file_metadata(_path(environment.get("REELPACK_PANNS_SED_CHECKPOINT")
                              or str(data / "Cnn14_DecisionLevelMax_mAP=0.385.pth"), environment))
    checkpoint_ready = checkpoint["exists"] and checkpoint["size_bytes"] >= 300000000
    sed_ready = sed["exists"] and sed["size_bytes"] >= 300000000 and labels["ready"]
    common = {"repository": str(Path(checkpoint["path"]).parent), "verification": "checkpoint_sizes_and_527_csv_labels",
              "inference": "not_run", "labels": labels, "automatic_download": False}
    panns = {**common, "name": "PANNs Cnn14", "path": checkpoint["path"],
             "ready": checkpoint_ready and labels["ready"], "checkpoint": checkpoint,
             "labels_ready": labels["ready"], "sed": sed, "sed_ready": sed_ready,
             "files": [checkpoint, labels], "size_bytes": checkpoint["size_bytes"] + labels["size_bytes"],
             "required_download_bytes": 327428481, "optional": True}
    sed_model = {**common, "name": "PANNs Cnn14 DecisionLevelMax", "path": sed["path"],
                 "ready": sed_ready, "checkpoint": sed, "files": [sed, labels],
                 "size_bytes": sed["size_bytes"] + labels["size_bytes"], "optional": True}
    return panns, sed_model


def _transnet_model(environment):
    runtime = environment.get("REELPACK_RUNTIME_ROOT")
    files = []
    if runtime:
        root = _path(runtime, environment)
        try:
            weights = root.glob(".venv-transnet/lib/python*/site-packages/transnetv2_pytorch/transnetv2-pytorch-weights.pth")
            files = [_file_metadata(path) for path in sorted(weights)]
        except OSError:
            pass
    return {"name": "TransNetV2", "repository": str(_path(runtime, environment)) if runtime else "",
            "path": files[0]["path"] if files else None,
            "ready": any(item["exists"] and item["size_bytes"] > 0 for item in files) if runtime else None,
            "files": files, "size_bytes": sum(item["size_bytes"] for item in files),
            "verification": "bundled_weight_filename_and_size_only", "inference": "not_run", "automatic_download": False}


def prepare_runtime_environment(environment=None, create=True):
    """Return a copied environment with usable runtime cache paths.

    ``create=True`` verifies every cache directory with a real exclusive write
    probe; EACCES, EPERM (including sandbox denial) and other filesystem errors
    try the stable ``$TMPDIR/reelpack-cache-<uid>`` fallback. ``create=False``
    only reads metadata: it never mkdirs, probes, or writes a cache.  This function
    does not mutate ``os.environ``; callers explicitly apply its return value.
    """
    prepared = _model_defaults(_environment(environment))
    primary, fallback = _cache_candidates(prepared)
    root = primary
    if create:
        errors = []
        for candidate in dict.fromkeys((primary, fallback)):
            try:
                _prepare_cache_tree(candidate, prepared)
            except OSError as exc:
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            else:
                root = candidate
                break
        else:
            raise OSError("No writable Reelpack cache root; " + "; ".join(errors))
    elif not _cache_tree_writable_without_probe(primary):
        root = fallback
    prepared["REELPACK_CACHE_ROOT"] = str(root)
    for key, child in _CACHE_CHILDREN.items():
        prepared[key] = str(root / child)
    prepared["PYTHONDONTWRITEBYTECODE"] = "1"
    if prepared.get("REELPACK_RUNTIME_ROOT"):
        runtime = _path(prepared["REELPACK_RUNTIME_ROOT"], prepared)
        for name in ("audio", "transnet", "ocr", "music"):
            prepared.setdefault("REELPACK_" + name.upper() + "_PYTHON",
                                str(runtime / (".venv-" + name) / "bin/python"))
    model = _whisper_model(prepared)
    if model["ready"] and any(not _writable_without_probe(path) for path in
                              (_path(prepared["HF_HOME"], prepared), _hub_root(prepared),
                               Path(model["repository"]), Path(model["path"]))):
        prepared["HF_HUB_OFFLINE"] = "1"
        prepared["TRANSFORMERS_OFFLINE"] = "1"
    return prepared


def prepare_pipeline_environment(environment=None, create=True):
    """Keep optional acceleration-cache failure from aborting existing core work.

    Workers still use the strict preparer and return an independent failed
    capability if neither allowed cache is writable. No third root is invented.
    """
    try:
        prepared = prepare_runtime_environment(environment, create=create)
        prepared.pop('REELPACK_CACHE_PREPARE_ERROR', None)
        return prepared
    except OSError as exc:
        prepared = prepare_runtime_environment(environment, create=False)
        prepared['REELPACK_CACHE_PREPARE_ERROR'] = str(exc)
        return prepared


def _metadata_size(path):
    """Bound directory metadata work; read no file contents and follow no dirs."""
    try:
        attributes = path.stat()
    except OSError:
        return 0, False
    if stat.S_ISREG(attributes.st_mode):
        return attributes.st_size, False
    if not stat.S_ISDIR(attributes.st_mode):
        return 0, False
    pending, size, entries, partial = [path], 0, 0, False
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as children:
                for child in children:
                    entries += 1
                    if entries > _SIZE_ENTRY_LIMIT:
                        return size, True
                    try:
                        if child.is_dir(follow_symlinks=False):
                            pending.append(Path(child.path))
                        elif child.is_file(follow_symlinks=True):
                            size += child.stat(follow_symlinks=True).st_size
                    except OSError:
                        partial = True
        except OSError:
            partial = True
    return size, partial


def _root_info(path, models_ready=None):
    try:
        exists = path.is_dir()
    except OSError:
        exists = False
    size, partial = _metadata_size(path)
    return {"path": str(path), "exists": exists, "writable": _writable_without_probe(path),
            "writability_check": "permissions_only_not_write_verified",
            "size_bytes": size, "size_partial": partial, "models_ready": models_ready}


def inspect_runtime_roots(environment=None, output_dir=None, library_root=None):
    """Return JSON-compatible, read-only root/model metadata.

    Roots have path/exists/writable/size_bytes/size_partial/models_ready fields.
    Writability is a permissions estimate (for missing roots, their nearest
    existing parent), never evidence of a successful sandbox write.  Whisper
    readiness means only the expected local nonempty files exist, not inference.
    Caller-specified output/library roots stay separate from every cache root.
    """
    inherited = _environment(environment)
    primary, fallback = _cache_candidates(inherited)
    prepared = prepare_runtime_environment(inherited, create=False)
    cache = _path(prepared["REELPACK_CACHE_ROOT"], prepared)
    output = _path(output_dir or prepared.get("REELPACK_OUTPUT_ROOT")
                   or str(_home(prepared) / "Downloads/Reelpack"), prepared)
    library = _path(library_root or prepared.get("REELPACK_LIBRARY_ROOT") or str(output / "library"), prepared)
    model = _whisper_model(prepared)
    panns, panns_sed = _panns_models(prepared)
    transnet = _transnet_model(prepared)
    roots = {"cache": _root_info(cache), "numba": _root_info(cache / "numba"),
             "mpl": _root_info(cache / "mpl"), "pycache": _root_info(cache / "pycache"),
             "models": _root_info(_path(prepared["HF_HOME"], prepared), model["ready"]),
             "hf_hub": _root_info(_hub_root(prepared), model["ready"]),
             "torch": _root_info(_path(prepared["TORCH_HOME"], prepared), transnet["ready"]),
             "panns": _root_info(Path(panns["checkpoint"]["path"]).parent, panns["ready"]),
             "panns_labels": _root_info(Path(panns["labels"]["path"]).parent, panns["labels"]["ready"]),
             "panns_sed": _root_info(Path(panns_sed["checkpoint"]["path"]).parent, panns_sed["ready"]),
             "outputs": _root_info(output), "library": _root_info(library)}
    if prepared.get("REELPACK_RUNTIME_ROOT"):
        roots["runtime"] = _root_info(_path(prepared["REELPACK_RUNTIME_ROOT"], prepared))
    return {"schema_version": 1, "read_only": True, "roots": roots,
            "models": {"whisper": model, "panns": panns, "panns_sed": panns_sed, "transnet": transnet},
            "cache_primary": str(primary), "cache_fallback": str(fallback), "cache_uses_fallback": cache == fallback,
            "jit": {"status": "not_run", "detail": "Read-only metadata inspection does not validate JIT compilation."},
            "bytecode": {"enabled": False, "configured_root": prepared["PYTHONPYCACHEPREFIX"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-env", action="store_true")
    mode.add_argument("--inspect", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_env:
        result = prepare_runtime_environment(create=os.environ.get("REELPACK_READONLY_CHECK") != "1")
        os.environ.update(result)
    else:
        result = inspect_runtime_roots()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
