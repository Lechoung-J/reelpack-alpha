"""Focused runtime cache tests: no models, media, installation or network."""

import errno
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "Core/runtime_environment.py"
SPEC = importlib.util.spec_from_file_location("runtime_environment", MODULE_PATH)
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


class RuntimeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="reelpack-runtime-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.home = self.root / "home"
        self.home.mkdir()
        self.temporary = self.root / "temporary"
        self.temporary.mkdir()
        self.environment = {"HOME": str(self.home), "TMPDIR": str(self.temporary), "PATH": "/usr/bin:/bin"}

    def test_python_only_enrichment_gets_workers_from_runtime_root(self):
        environment = dict(self.environment, REELPACK_RUNTIME_ROOT=str(self.root / 'runtime'))
        environment['REELPACK_MUSIC_PYTHON'] = '/explicit/music/python'
        prepared = RUNTIME.prepare_runtime_environment(environment, create=False)
        for name in ('audio', 'transnet', 'ocr'):
            self.assertEqual(prepared['REELPACK_' + name.upper() + '_PYTHON'],
                             str(self.root / 'runtime' / ('.venv-' + name) / 'bin/python'))
        self.assertEqual(prepared['REELPACK_MUSIC_PYTHON'], '/explicit/music/python')
        self.assertFalse((self.root / 'runtime').exists())

    def snapshot(self):
        return {str(p.relative_to(self.root)): (p.stat().st_mode, p.stat().st_mtime_ns,
                                              p.read_bytes() if p.is_file() else None)
                for p in self.root.rglob("*")}

    def make_whisper(self, hf_root):
        snapshot = hf_root / "hub/models--Systran--faster-whisper-large-v3/snapshots/preset"
        snapshot.mkdir(parents=True)
        for filename in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
            (snapshot / filename).write_bytes(b"fixture " + filename.encode())
        return snapshot

    def test_default_cache_tree_is_real_writable_and_copy_does_not_mutate_environment(self):
        before = dict(os.environ)
        original = dict(self.environment)
        prepared = RUNTIME.prepare_runtime_environment(self.environment)
        self.assertEqual(self.environment, original)
        self.assertEqual(dict(os.environ), before)
        cache = self.home / "Library/Caches/Reelpack"
        self.assertEqual(prepared["REELPACK_CACHE_ROOT"], str(cache))
        for variable, child in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "mpl"), ("PYTHONPYCACHEPREFIX", "pycache")):
            self.assertEqual(prepared[variable], str(cache / child))
            self.assertTrue((cache / child).is_dir())
        self.assertEqual(list(cache.rglob(".reelpack-write-probe-*")), [])
        self.assertEqual(prepared["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertFalse((self.home / ".cache").exists())
        self.assertEqual(prepared["HF_HOME"], str(self.home / ".cache/huggingface"))
        self.assertEqual(prepared["TORCH_HOME"], str(self.home / ".cache/torch"))

    def test_explicit_cache_and_model_roots_are_distinct_and_model_overrides_survive(self):
        cache = self.root / "custom acceleration cache"
        hf, torch = self.root / "persistent HF", self.root / "persistent torch"
        environment = {**self.environment, "REELPACK_CACHE_ROOT": str(cache), "HF_HOME": str(hf), "TORCH_HOME": str(torch),
                       "NUMBA_CACHE_DIR": "inherited-unused", "MPLCONFIGDIR": "inherited-unused"}
        prepared = RUNTIME.prepare_runtime_environment(environment)
        self.assertEqual(prepared["HF_HOME"], str(hf))
        self.assertEqual(prepared["TORCH_HOME"], str(torch))
        self.assertEqual(prepared["NUMBA_CACHE_DIR"], str(cache / "numba"))
        self.assertFalse(hf.exists())
        self.assertFalse(torch.exists())
        outputs, library = self.root / "user outputs", self.root / "user library"
        inspection = RUNTIME.inspect_runtime_roots(prepared, output_dir=outputs, library_root=library)
        self.assertEqual(inspection["roots"]["outputs"]["path"], str(outputs))
        self.assertEqual(inspection["roots"]["library"]["path"], str(library))
        self.assertFalse(outputs.exists())
        self.assertFalse(library.exists())

    def test_read_only_resolution_and_inspection_do_not_mkdir_probe_or_write(self):
        original = self.snapshot()
        with patch.object(Path, "mkdir", side_effect=AssertionError("mkdir called")), \
                patch.object(RUNTIME.tempfile, "mkstemp", side_effect=AssertionError("probe called")), \
                patch.object(RUNTIME.tempfile, "gettempdir", side_effect=AssertionError("implicit temporary probe")), \
                patch.object(os, "open", side_effect=AssertionError("write open called")):
            prepared = RUNTIME.prepare_runtime_environment(self.environment, create=False)
            inspection = RUNTIME.inspect_runtime_roots(self.environment)
        self.assertEqual(self.snapshot(), original)
        self.assertFalse(Path(prepared["REELPACK_CACHE_ROOT"]).exists())
        self.assertFalse(inspection["roots"]["cache"]["exists"])
        self.assertEqual(inspection["jit"]["status"], "not_run")
        self.assertEqual(inspection["roots"]["cache"]["writability_check"], "permissions_only_not_write_verified")
        json.dumps(inspection)

    def test_sandbox_eperm_real_probe_uses_stable_uid_fallback_across_runs(self):
        primary = self.root / "sandbox denied"
        environment = {**self.environment, "REELPACK_CACHE_ROOT": str(primary)}
        real_probe = RUNTIME.tempfile.mkstemp
        attempts = []

        def sandbox_probe(*args, **kwargs):
            attempts.append(Path(kwargs["dir"]))
            if Path(kwargs["dir"]) == primary:
                raise PermissionError(errno.EPERM, "Operation not permitted", str(primary))
            return real_probe(*args, **kwargs)

        with patch.object(RUNTIME.tempfile, "mkstemp", side_effect=sandbox_probe):
            first = RUNTIME.prepare_runtime_environment(environment)
            second = RUNTIME.prepare_runtime_environment(environment)
        fallback = self.temporary / f"reelpack-cache-{os.getuid()}"
        self.assertEqual(first["REELPACK_CACHE_ROOT"], str(fallback))
        self.assertEqual(second["REELPACK_CACHE_ROOT"], str(fallback))
        self.assertIn(primary, attempts)
        self.assertIn(fallback, attempts)
        self.assertEqual(first["HF_HOME"], str(self.home / ".cache/huggingface"))
        self.assertEqual(list(self.root.rglob(".reelpack-write-probe-*")), [])

    def test_sandbox_eperm_mkdir_also_falls_back(self):
        primary = self.root / "denied mkdir"
        original_mkdir = Path.mkdir

        def sandbox_mkdir(path, *args, **kwargs):
            if path == primary:
                raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
            return original_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", sandbox_mkdir):
            prepared = RUNTIME.prepare_runtime_environment({**self.environment, "REELPACK_CACHE_ROOT": str(primary)})
        self.assertTrue(prepared["REELPACK_CACHE_ROOT"].endswith(f"reelpack-cache-{os.getuid()}"))
        self.assertFalse(primary.exists())

    def test_permission_only_resolution_selects_fallback_without_creating_it(self):
        cache = self.root / "read only"
        cache.mkdir()
        cache.chmod(0o555)
        self.addCleanup(cache.chmod, 0o700)
        environment = {**self.environment, "REELPACK_CACHE_ROOT": str(cache)}
        before = self.snapshot()
        prepared = RUNTIME.prepare_runtime_environment(environment, create=False)
        report = RUNTIME.inspect_runtime_roots(environment)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(prepared["REELPACK_CACHE_ROOT"], str(self.temporary / f"reelpack-cache-{os.getuid()}"))
        self.assertTrue(report["cache_uses_fallback"])
        self.assertFalse(Path(prepared["REELPACK_CACHE_ROOT"]).exists())

    def test_read_only_cache_child_causes_whole_tree_fallback(self):
        cache = self.root / "cache with blocked child"
        (cache / "mpl").mkdir(parents=True)
        (cache / "mpl").chmod(0o555)
        self.addCleanup((cache / "mpl").chmod, 0o700)
        environment = {**self.environment, "REELPACK_CACHE_ROOT": str(cache)}
        prepared = RUNTIME.prepare_runtime_environment(environment, create=False)
        self.assertEqual(prepared["MPLCONFIGDIR"], str(self.temporary / f"reelpack-cache-{os.getuid()}" / "mpl"))

    def test_read_only_preset_whisper_hits_using_metadata_and_sets_offline(self):
        hf = self.root / "preset HF"
        snapshot = self.make_whisper(hf)
        hf.chmod(0o555)
        self.addCleanup(hf.chmod, 0o700)
        environment = {**self.environment, "HF_HOME": str(hf), "TORCH_HOME": str(self.root / "models torch")}
        before = self.snapshot()
        with patch.object(Path, "open", side_effect=AssertionError("model content opened")):
            prepared = RUNTIME.prepare_runtime_environment(environment, create=False)
            inspection = RUNTIME.inspect_runtime_roots(environment)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(prepared["HF_HOME"], str(hf))
        self.assertEqual(prepared["HF_HUB_OFFLINE"], "1")
        self.assertEqual(prepared["TRANSFORMERS_OFFLINE"], "1")
        model = inspection["models"]["whisper"]
        self.assertTrue(model["ready"])
        self.assertEqual(model["path"], str(snapshot))
        self.assertEqual(model["size_bytes"], sum(p.stat().st_size for p in snapshot.iterdir()))
        self.assertEqual(model["inference"], "not_run")
        self.assertTrue(inspection["roots"]["models"]["models_ready"])
        self.assertFalse(inspection["roots"]["models"]["writable"])

    def test_hf_hub_override_and_snapshot_symlink_weights_are_supported(self):
        hub = self.root / "specific HF hub"
        hf = self.root / "other HF home"
        snapshot = self.make_whisper(hub.parent / "fixture")
        hub.symlink_to(snapshot.parents[2], target_is_directory=True)
        weight = snapshot / "model.bin"
        blob = snapshot.parent / "blob"
        blob.write_bytes(weight.read_bytes())
        self.assertEqual(weight.resolve(), snapshot / "model.bin")
        weight.unlink()
        weight.symlink_to(blob)
        report = RUNTIME.inspect_runtime_roots({**self.environment, "HF_HOME": str(hf), "HF_HUB_CACHE": str(hub)})
        self.assertTrue(report["models"]["whisper"]["ready"])
        self.assertEqual(report["roots"]["hf_hub"]["path"], str(hub))

    def test_incomplete_snapshot_is_not_ready_and_does_not_force_offline(self):
        hf = self.root / "incomplete HF"
        snapshot = self.make_whisper(hf)
        (snapshot / "model.bin").write_bytes(b"")
        prepared = RUNTIME.prepare_runtime_environment({**self.environment, "HF_HOME": str(hf)}, create=False)
        report = RUNTIME.inspect_runtime_roots(prepared)
        self.assertFalse(report["models"]["whisper"]["ready"])
        self.assertNotIn("HF_HUB_OFFLINE", prepared)

    def test_xdg_default_models_never_follow_acceleration_cache(self):
        persistent = self.root / "persistent xdg"
        report = RUNTIME.prepare_runtime_environment({**self.environment, "XDG_CACHE_HOME": str(persistent)}, create=False)
        self.assertEqual(report["HF_HOME"], str(persistent / "huggingface"))
        self.assertEqual(report["TORCH_HOME"], str(persistent / "torch"))
        self.assertNotIn(str(self.temporary), report["HF_HOME"])
        self.assertFalse(persistent.exists())

    def test_inspect_cli_and_read_only_prepare_cli_create_nothing(self):
        before = self.snapshot()
        environment = {**self.environment, "REELPACK_READONLY_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        for flag in ("--inspect", "--prepare-env"):
            process = subprocess.run([sys.executable, "-B", str(MODULE_PATH), flag], capture_output=True, text=True, env=environment)
            self.assertEqual(process.returncode, 0, process.stderr)
            json.loads(process.stdout)
        self.assertEqual(self.snapshot(), before)

    def test_panns_models_respect_explicit_csv_and_stat_only_large_checkpoints(self):
        models = self.root / "provided model weights"
        models.mkdir()
        checkpoint, sed = models / "Cnn14.pth", models / "SED.pth"
        for weight in (checkpoint, sed):
            with weight.open("wb") as output:
                output.truncate(327428481)  # Sparse fixture: no weights are loaded.
        labels = self.root / "provided labels.csv"
        labels.write_text("index,mid,display_name\n" + "".join(f"{i},/m/{i},Label {i}\n" for i in range(527)), encoding="utf-8")
        environment = {**self.environment, "REELPACK_PANNS_CHECKPOINT": str(checkpoint),
                       "REELPACK_PANNS_SED_CHECKPOINT": str(sed), "REELPACK_PANNS_LABELS": str(labels)}
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path in (checkpoint, sed):
                self.fail("diagnostic opened a large model checkpoint")
            return original_open(path, *args, **kwargs)

        before = {path: path.stat().st_mtime_ns for path in (models, checkpoint, sed, labels)}
        with patch.object(Path, "open", guarded_open):
            report = RUNTIME.inspect_runtime_roots(environment)
        panns = report["models"]["panns"]
        self.assertTrue(panns["ready"])
        self.assertTrue(panns["sed_ready"])
        self.assertTrue(report["models"]["panns_sed"]["ready"])
        self.assertEqual(panns["labels"]["path"], str(labels))
        self.assertEqual(panns["labels"]["rows"], 527)
        self.assertEqual(panns["checkpoint"]["size_bytes"], 327428481)
        self.assertEqual(panns["required_download_bytes"], 327428481)
        self.assertFalse(panns["automatic_download"])
        self.assertEqual(panns["inference"], "not_run")
        self.assertEqual(report["roots"]["panns"]["size_bytes"], 2 * 327428481)
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in before})
        self.assertFalse((self.home / "panns_data").exists())

    def test_invalid_panns_labels_never_mark_a_present_checkpoint_ready(self):
        weight = self.root / "local checkpoint.pth"
        with weight.open("wb") as output:
            output.truncate(300000000)
        labels = self.root / "invalid labels.csv"
        labels.write_text("index,display_name\n0,Only one label\n", encoding="utf-8")
        report = RUNTIME.inspect_runtime_roots({**self.environment, "REELPACK_PANNS_CHECKPOINT": str(weight),
                                                "REELPACK_PANNS_LABELS": str(labels)})
        self.assertFalse(report["models"]["panns"]["ready"])
        self.assertFalse(report["models"]["panns"]["labels_ready"])
        self.assertEqual(report["models"]["panns"]["labels"]["rows"], 1)
        self.assertFalse(report["models"]["panns_sed"]["ready"])

    def test_transnet_bundled_weight_readiness_uses_only_stat_and_is_unknown_without_runtime(self):
        without_runtime = RUNTIME.inspect_runtime_roots(self.environment)
        self.assertIsNone(without_runtime["models"]["transnet"]["ready"])
        self.assertIsNone(without_runtime["roots"]["torch"]["models_ready"])
        runtime = self.root / "runtime"
        package = runtime / ".venv-transnet/lib/python3.12/site-packages/transnetv2_pytorch"
        package.mkdir(parents=True)
        weight = package / "transnetv2-pytorch-weights.pth"
        weight.write_bytes(b"bundled model fixture")
        with patch.object(Path, "open", side_effect=AssertionError("checkpoint content opened")):
            report = RUNTIME.inspect_runtime_roots({**self.environment, "REELPACK_RUNTIME_ROOT": str(runtime)})
        self.assertTrue(report["models"]["transnet"]["ready"])
        self.assertTrue(report["roots"]["torch"]["models_ready"])
        self.assertEqual(report["models"]["transnet"]["path"], str(weight))
        self.assertEqual(report["models"]["transnet"]["size_bytes"], weight.stat().st_size)
        self.assertFalse(report["models"]["transnet"]["automatic_download"])

    def test_pipeline_caller_preserves_persistent_roots_when_all_cache_probes_fail(self):
        environment = dict(self.environment, HF_HOME='/persistent/read-only/hf', TORCH_HOME='/persistent/torch')
        with patch.object(RUNTIME.tempfile, 'mkstemp', side_effect=PermissionError(errno.EPERM, 'denied')):
            prepared = RUNTIME.prepare_pipeline_environment(environment)
        self.assertIn('No writable Reelpack cache root', prepared['REELPACK_CACHE_PREPARE_ERROR'])
        self.assertEqual(prepared['HF_HOME'], '/persistent/read-only/hf')
        self.assertEqual(prepared['TORCH_HOME'], '/persistent/torch')
        self.assertNotEqual(prepared['HF_HOME'], prepared['REELPACK_CACHE_ROOT'])

    def test_all_probe_failures_are_reported_without_an_unstable_third_location(self):
        with patch.object(RUNTIME.tempfile, "mkstemp", side_effect=PermissionError(errno.EPERM, "denied")):
            with self.assertRaisesRegex(OSError, "No writable Reelpack cache root"):
                RUNTIME.prepare_runtime_environment(self.environment)


if __name__ == "__main__":
    unittest.main()
