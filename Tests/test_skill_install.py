"""Exercise only the installer Python helper in a temporary mocked user home."""

import ast
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"
INSTALLER_TEXT = INSTALLER.read_text(encoding="utf-8")
HELPER = INSTALLER_TEXT.split("py_task() {", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n}", 1)[0]
# Import and compile the real helper functions without the installer dispatcher,
# platform checks, downloads, runtime creation or GUI launch.
TREE = ast.parse(HELPER)
FUNCTIONS = ast.Module(
    body=[node for node in TREE.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))],
    type_ignores=[],
)
NAMESPACE = {}
exec(compile(FUNCTIONS, str(INSTALLER), "exec"), NAMESPACE)
INTEGRATE = NAMESPACE["integrate_skills"]


class SkillInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="reelpack-skill-tests-")
        self.addCleanup(self.temporary.cleanup)
        # macOS /var is a symlink; start from the resolved fixture directory.
        self.root = Path(self.temporary.name).resolve()
        self.home = self.root / "mock-home"
        self.home.mkdir()
        self.app = self.root / "Applications with space" / "Reelpack.app"
        self.resources = self.app / "Contents/Resources"
        (self.resources / "Core").mkdir(parents=True)
        self.canonical = "# Reelpack 原文\r\n\r\n本地视频与证据帧。\r\n".encode("utf-8")
        self.source = self.resources / "SKILL.md"
        self.source.write_bytes(self.canonical)
        self.adapter = self.resources / "Core/agent_entry.py"
        self.adapter.write_text(
            "import json, os, sys\n"
            "print(json.dumps({'argv': sys.argv[1:], 'runtime': os.environ.get('REELPACK_RUNTIME_ROOT'), 'path': os.environ.get('PATH')}, ensure_ascii=False))\n",
            encoding="utf-8",
        )
        self.runtime = self.root / "runtime with space"
        self.runtime.mkdir()
        self.output = ""

    def run_integration(self, **kwargs):
        output = io.StringIO()
        with patch.object(Path, "home", return_value=self.home), redirect_stdout(output):
            result = INTEGRATE(self.app, self.runtime, **kwargs)
        self.output = output.getvalue()
        return result

    def configure(self, *names):
        for name in names:
            (self.home / name).mkdir(parents=True, exist_ok=True)

    def snapshot(self, root, ignore_report=False):
        records = {}
        for p in root.rglob("*"):
            if ignore_report and p.name == "skill-integration.json":
                continue
            if p.is_symlink():
                records[str(p.relative_to(root))] = ("symlink", os.readlink(p))
            elif p.is_file():
                records[str(p.relative_to(root))] = ("file", p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_mode)
            else:
                records[str(p.relative_to(root))] = ("directory", p.stat().st_mtime_ns)
        return records

    def test_native_agents_receive_frontmatter_original_bytes_and_hashes(self):
        self.configure(".claude", ".codex", ".cursor")
        sensitive = {self.home / ".claude/settings.json", self.home / ".codex/auth.json", self.home / ".cursor/settings.json"}
        for p in sensitive:
            p.write_text("private configuration must not be read", encoding="utf-8")
        original_open = Path.open

        def guarded_open(p, *args, **kwargs):
            if p in sensitive:
                self.fail(f"installer tried to read agent configuration: {p.name}")
            return original_open(p, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            result = self.run_integration()
        self.assertEqual([agent["status"] for agent in result["agents"]], ["installed"] * 3)
        self.assertTrue((self.home / ".agents/skills/reelpack/SKILL.md").is_file())
        self.assertFalse((self.home / ".codex/skills").exists())
        for agent in result["agents"]:
            skill = Path(agent["path"])
            native = skill.read_bytes()
            self.assertTrue(native.startswith(b"---\nname: reelpack\ndescription: "))
            self.assertIn(b"\n---\n", native)
            self.assertTrue(native.endswith(self.canonical))
            self.assertIn(str(self.runtime / "bin/reelpack").encode(), native)
            metadata = json.loads((skill.parent / ".metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["canonical_sha256"], hashlib.sha256(self.canonical).hexdigest())
            self.assertEqual(metadata["skill_sha256"], hashlib.sha256(native).hexdigest())
        self.assertEqual((self.runtime / "share/reelpack/SKILL.md").read_bytes(), self.canonical)
        self.assertEqual(self.source.read_bytes(), self.canonical)
        report = json.loads((self.runtime / "skill-integration.json").read_text(encoding="utf-8"))
        self.assertEqual(report, result)
        for record in report["files"]:
            self.assertEqual(hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest(), record["sha256"])
        self.assertNotIn("private configuration", self.output)

    def test_codex_detects_agents_directory_without_codex(self):
        self.configure(".agents")
        result = self.run_integration()
        self.assertEqual(result["agents"][1]["status"], "installed")
        self.assertFalse((self.home / ".codex").exists())

    def test_no_agents_still_gets_canonical_cli_and_manual_instructions(self):
        result = self.run_integration()
        self.assertEqual([agent["status"] for agent in result["agents"]], ["not_detected"] * 3)
        for agent in result["agents"]:
            self.assertFalse(Path(agent["path"]).parent.exists())
            self.assertIn(agent["agent"], self.output)
        self.assertIn(f"ask the agent to read {self.source}", self.output)
        self.assertTrue((self.runtime / "share/reelpack/SKILL.md").is_file())
        self.assertTrue((self.runtime / "bin/reelpack").is_file())
        self.assertTrue((self.home / ".local/bin/reelpack").is_file())
        self.assertIn("export PATH=", self.output)

    def test_opt_out_skips_all_user_home_writes(self):
        self.configure(".claude", ".codex", ".cursor")
        before = self.snapshot(self.home)
        result = self.run_integration(register=False)
        self.assertEqual(self.snapshot(self.home), before)
        self.assertEqual([agent["status"] for agent in result["agents"]], ["disabled"] * 3)
        self.assertEqual(result["launchers"][1]["status"], "disabled")
        self.assertTrue((self.runtime / "bin/reelpack").is_file())
        self.assertTrue((self.runtime / "share/reelpack/SKILL.md").is_file())

    def test_same_content_is_reused_without_rewriting_skills_or_launchers(self):
        self.configure(".claude", ".agents", ".cursor")
        self.run_integration()
        before_home = self.snapshot(self.home)
        before_runtime = self.snapshot(self.runtime, ignore_report=True)
        result = self.run_integration()
        self.assertEqual([agent["status"] for agent in result["agents"]], ["reused"] * 3)
        self.assertEqual([entry["status"] for entry in result["launchers"]], ["reused"] * 2)
        self.assertEqual(self.snapshot(self.home), before_home)
        self.assertEqual(self.snapshot(self.runtime, ignore_report=True), before_runtime)

    def test_existing_custom_skill_is_preserved_and_not_given_metadata(self):
        target = self.home / ".claude/skills/reelpack"
        target.mkdir(parents=True)
        skill = target / "SKILL.md"
        skill.write_bytes(b"user custom skill")
        before = self.snapshot(target)
        result = self.run_integration()
        self.assertEqual(result["agents"][0]["status"], "conflict")
        self.assertEqual(self.snapshot(target), before)
        self.assertFalse((target / ".metadata.json").exists())
        self.assertIn("existing content differs", self.output)
        self.assertIn("manual integration", self.output)

    def test_occupied_skill_directory_without_skill_is_preserved(self):
        target = self.home / ".cursor/skills/reelpack"
        target.mkdir(parents=True)
        (target / "custom.txt").write_bytes(b"keep")
        before = self.snapshot(target)
        result = self.run_integration()
        self.assertEqual(result["agents"][2]["status"], "conflict")
        self.assertEqual(self.snapshot(target), before)

    def test_custom_metadata_is_preserved_even_when_skill_matches(self):
        self.configure(".claude")
        self.run_integration()
        directory = self.home / ".claude/skills/reelpack"
        (directory / ".metadata.json").write_bytes(b"user custom metadata")
        before = self.snapshot(directory)
        result = self.run_integration()
        self.assertEqual(result["agents"][0]["status"], "conflict")
        self.assertEqual(self.snapshot(directory), before)

    def test_existing_custom_runtime_skill_is_preserved(self):
        destination = self.runtime / "share/reelpack/SKILL.md"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"custom canonical reference")
        result = self.run_integration()
        self.assertEqual(result["canonical_skill"]["status"], "conflict")
        self.assertEqual(destination.read_bytes(), b"custom canonical reference")
        self.assertIn(str(self.source), self.output)

    def test_existing_custom_launchers_are_preserved_and_direct_adapter_is_shown(self):
        for destination in (self.runtime / "bin/reelpack", self.home / ".local/bin/reelpack"):
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"custom command")
        result = self.run_integration()
        self.assertEqual([entry["status"] for entry in result["launchers"]], ["conflict"] * 2)
        for entry in result["launchers"]:
            self.assertEqual(Path(entry["path"]).read_bytes(), b"custom command")
        command_line = next(line for line in self.output.splitlines() if "Command to use:" in line)
        self.assertIn("agent_entry.py", command_line)
        self.assertNotIn("bin/reelpack", command_line)

    def test_matching_but_non_executable_launcher_is_not_reported_ready(self):
        self.run_integration(register=False)
        entry = self.runtime / "bin/reelpack"
        entry.chmod(0o600)
        result = self.run_integration(register=False)
        self.assertEqual(result["launchers"][0]["status"], "conflict")
        self.assertEqual(entry.stat().st_mode & 0o777, 0o600)

    def test_symlink_agent_paths_are_preserved(self):
        for component in (".claude", ".claude/skills", ".claude/skills/reelpack", ".claude/skills/reelpack/SKILL.md", ".claude/skills/reelpack/.metadata.json"):
            with self.subTest(component=component):
                self.home = self.root / ("home-" + str(len(list(self.root.glob("home-*")))))
                self.home.mkdir()
                outside = self.root / (self.home.name + "-outside")
                outside.mkdir()
                (outside / "sentinel").write_bytes(b"keep outside")
                link = self.home / component
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.suffix in (".md", ".json"):
                    link.symlink_to(outside / "sentinel")
                else:
                    link.symlink_to(outside, target_is_directory=True)
                before = self.snapshot(outside)
                result = self.run_integration()
                self.assertEqual(result["agents"][0]["status"], "unsafe")
                self.assertTrue(link.is_symlink())
                self.assertEqual(self.snapshot(outside), before)

    def test_symlink_launcher_directories_and_files_are_preserved(self):
        outside = self.root / "outside-launcher"
        outside.mkdir()
        (outside / "reelpack").write_bytes(b"custom")
        (self.runtime / "bin").symlink_to(outside, target_is_directory=True)
        local = self.home / ".local/bin"
        local.mkdir(parents=True)
        (local / "reelpack").symlink_to(outside / "reelpack")
        before = self.snapshot(outside)
        result = self.run_integration()
        self.assertEqual([entry["status"] for entry in result["launchers"]], ["unsafe"] * 2)
        self.assertEqual(self.snapshot(outside), before)

    def test_runtime_symlink_is_rejected_before_writes(self):
        link = self.root / "runtime-link"
        link.symlink_to(self.runtime, target_is_directory=True)
        before = self.snapshot(self.runtime)
        with patch.object(Path, "home", return_value=self.home), self.assertRaisesRegex(ValueError, "symlink"):
            INTEGRATE(self.app, link)
        self.assertEqual(self.snapshot(self.runtime), before)

    def test_report_symlink_and_unmanaged_report_are_preserved(self):
        outside = self.root / "outside-report.json"
        outside.write_bytes(b"custom report")
        report = self.runtime / "skill-integration.json"
        report.symlink_to(outside)
        self.run_integration()
        self.assertTrue(report.is_symlink())
        self.assertEqual(outside.read_bytes(), b"custom report")
        self.assertIn("report preserved/not written", self.output)
        # Verify the exact target before deleting this fixture symlink.
        self.assertEqual(report.absolute(), self.runtime / "skill-integration.json")
        report.unlink()
        report.write_bytes(b'{"schema_version":1,"managed_by":"user"}')
        before = report.read_bytes()
        self.run_integration()
        self.assertEqual(report.read_bytes(), before)
        self.assertIn("unmanaged integration report preserved", self.output)

    def test_read_only_missing_integration_creates_nothing(self):
        self.configure(".claude", ".codex", ".cursor")
        before = self.snapshot(self.root)
        result = self.run_integration(check_only=True)
        self.assertEqual(self.snapshot(self.root), before)
        self.assertEqual([agent["status"] for agent in result["agents"]], ["missing"] * 3)
        self.assertIn("Read-only integration check", self.output)
        self.assertFalse((self.runtime / "skill-integration.json").exists())

    def test_read_only_existing_integration_does_not_refresh_report(self):
        self.configure(".claude", ".agents", ".cursor")
        self.run_integration()
        before = self.snapshot(self.root)
        result = self.run_integration(check_only=True)
        self.assertEqual([agent["status"] for agent in result["agents"]], ["ready"] * 3)
        self.assertEqual(self.snapshot(self.root), before)

    def test_launchers_preserve_argv_runtime_and_native_path_with_shell_characters(self):
        self.runtime = self.root / "runtime 'quote' $dollar; & space"
        (self.runtime / ".venv/bin").mkdir(parents=True)
        (self.runtime / ".venv/bin/python").symlink_to(Path(sys.executable).resolve())
        result = self.run_integration()
        arguments = ["视频 with spaces $(touch nope).mp4", "--output-dir", str(self.root / "output 'quoted' $x"), "--flag=literal; value"]
        for entry in result["launchers"]:
            syntax = subprocess.run(["/bin/sh", "-n", entry["path"]], capture_output=True, text=True)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            process = subprocess.run([entry["path"], *arguments], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
            self.assertEqual(process.returncode, 0, process.stderr)
            observed = json.loads(process.stdout)
            self.assertEqual(observed["argv"], arguments)
            self.assertEqual(observed["runtime"], str(self.runtime))
            self.assertEqual(observed["path"], f"{self.runtime}/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/bin:/bin")
        self.assertFalse((self.root / "nope").exists())

    def configure_cache_adapter(self):
        (self.runtime / ".venv/bin").mkdir(parents=True, exist_ok=True)
        (self.runtime / ".venv/bin/python").symlink_to(Path(sys.executable).resolve())
        module = INSTALLER.parent / "Core/runtime_environment.py"
        (self.resources / "Core/runtime_environment.py").write_bytes(module.read_bytes())
        self.adapter.write_text(
            "import json, os, subprocess, sys\n"
            "from runtime_environment import prepare_runtime_environment\n"
            "os.environ.update(prepare_runtime_environment(create=os.environ.get('REELPACK_READONLY_CHECK') != '1'))\n"
            "keys = ('REELPACK_CACHE_ROOT', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR', 'PYTHONPYCACHEPREFIX', 'PYTHONDONTWRITEBYTECODE', 'HF_HOME', 'TORCH_HOME', 'HF_HUB_OFFLINE')\n"
            "child_code = 'import json, os; print(json.dumps(dict(os.environ)))'\n"
            "child = json.loads(subprocess.check_output([sys.executable, '-B', '-c', child_code], text=True))\n"
            "print(json.dumps({'environment': {key: os.environ.get(key) for key in keys}, 'child': {key: child.get(key) for key in keys}, 'argv': sys.argv[1:]}))\n",
            encoding="utf-8",
        )

    def run_cache_launcher(self, environment=None, arguments=()):
        result = self.run_integration(register=False)
        launcher = result["launchers"][0]["path"]
        child_environment = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "TMPDIR": str(self.root / "tmp")}
        (self.root / "tmp").mkdir(exist_ok=True)
        child_environment.update(environment or {})
        process = subprocess.run([launcher, *arguments], capture_output=True, text=True, env=child_environment)
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def test_launcher_prepares_default_cache_without_manual_environment_and_workers_inherit(self):
        self.configure_cache_adapter()
        outputs = self.root / "separate user outputs"
        observed = self.run_cache_launcher(arguments=("--output-dir", str(outputs)))
        environment = observed["environment"]
        cache = self.home / "Library/Caches/Reelpack"
        self.assertEqual(environment["REELPACK_CACHE_ROOT"], str(cache))
        for variable, child in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "mpl"), ("PYTHONPYCACHEPREFIX", "pycache")):
            self.assertEqual(environment[variable], str(cache / child))
            self.assertTrue((cache / child).is_dir())
        self.assertEqual(observed["child"], environment)
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(environment["HF_HOME"], str(self.home / ".cache/huggingface"))
        self.assertEqual(environment["TORCH_HOME"], str(self.home / ".cache/torch"))
        self.assertFalse(outputs.exists())
        self.assertFalse((self.home / ".cache").exists())
        self.assertEqual(list((self.resources / "Core").rglob("__pycache__")), [])

    def test_launcher_read_only_candidate_falls_back_to_same_tmp_uid_cache_across_runs(self):
        self.configure_cache_adapter()
        cache = self.root / "read only candidate"
        cache.mkdir()
        cache.chmod(0o555)
        self.addCleanup(cache.chmod, 0o700)
        environment = {"REELPACK_CACHE_ROOT": str(cache)}
        first = self.run_cache_launcher(environment)
        second = self.run_cache_launcher(environment)
        fallback = self.root / "tmp" / f"reelpack-cache-{os.getuid()}"
        self.assertEqual(first["environment"]["REELPACK_CACHE_ROOT"], str(fallback))
        self.assertEqual(second["environment"]["REELPACK_CACHE_ROOT"], str(fallback))
        self.assertEqual(first["child"], first["environment"])
        self.assertEqual(list(fallback.rglob(".reelpack-write-probe-*")), [])

    def test_launcher_preserves_explicit_persistent_model_roots(self):
        self.configure_cache_adapter()
        cache, hf, torch = self.root / "acceleration", self.root / "models HF", self.root / "models torch"
        observed = self.run_cache_launcher({"REELPACK_CACHE_ROOT": str(cache), "HF_HOME": str(hf), "TORCH_HOME": str(torch)})
        self.assertEqual(observed["environment"]["REELPACK_CACHE_ROOT"], str(cache))
        self.assertEqual(observed["environment"]["HF_HOME"], str(hf))
        self.assertEqual(observed["environment"]["TORCH_HOME"], str(torch))
        self.assertFalse(hf.exists())
        self.assertFalse(torch.exists())

    def test_launcher_read_only_check_never_prepares_cache_directories(self):
        self.configure_cache_adapter()
        observed = self.run_cache_launcher({"REELPACK_READONLY_CHECK": "1"}, arguments=("--runtime-check",))
        cache = self.home / "Library/Caches/Reelpack"
        self.assertEqual(observed["environment"]["REELPACK_CACHE_ROOT"], str(cache))
        self.assertFalse(cache.exists())
        self.assertFalse((self.home / "Library").exists())
        self.assertEqual(observed["child"], observed["environment"])

    def test_missing_or_empty_canonical_and_missing_adapter_fail_before_writes(self):
        self.source.write_bytes(b"")
        before = self.snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "canonical App SKILL"):
            self.run_integration()
        self.assertEqual(self.snapshot(self.root), before)
        self.source.write_bytes(self.canonical)
        # Verify this controlled temporary target before removing it.
        self.assertEqual(self.adapter.absolute(), self.resources / "Core/agent_entry.py")
        self.adapter.unlink()
        before = self.snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "agent_entry.py"):
            self.run_integration()
        self.assertEqual(self.snapshot(self.root), before)

    def test_old_managed_upgrade_backs_up_skill_and_metadata_together(self):
        self.configure(".claude", ".agents", ".cursor")
        old = self.run_integration()
        old_bytes = {item["path"]: Path(item["path"]).read_bytes() for item in old["files"]}
        self.source.write_bytes(b"# New official skill\nExplicit model preparation.\n")
        result = self.run_integration()
        self.assertEqual(result["canonical_skill"]["status"], "upgraded")
        self.assertEqual([item["status"] for item in result["agents"]], ["upgraded"] * 3)
        for item in result["files"]:
            if item["status"] == "upgraded":
                self.assertEqual(Path(item["backup"]).read_bytes(), old_bytes[item["path"]])
                self.assertEqual(item["previous_sha256"], hashlib.sha256(old_bytes[item["path"]]).hexdigest())
        for agent in result["agents"]:
            backup = Path(agent["backup"])
            self.assertEqual({p.name for p in backup.iterdir()}, {"SKILL.md", ".metadata.json"})
        history = result["upgrades"]
        self.assertEqual(self.run_integration()["upgrades"], history)

    def test_leftover_skill_without_app_or_runtime_recovers_from_metadata(self):
        self.configure(".claude")
        self.run_integration()
        # Remove only these verified fixture paths, leaving the native home skill behind.
        self.assertEqual(self.app, self.root / "Applications with space/Reelpack.app")
        self.assertEqual(self.runtime, self.root / "runtime with space")
        shutil.rmtree(self.app)
        shutil.rmtree(self.runtime)
        (self.resources / "Core").mkdir(parents=True)
        self.source.write_bytes(b"# Replacement release\n")
        self.adapter.write_bytes(b"# replacement adapter\n")
        self.runtime = self.root / "replacement-runtime"
        self.runtime.mkdir()
        result = self.run_integration()
        self.assertEqual(result["agents"][0]["status"], "upgraded")
        self.assertEqual(result["canonical_skill"]["status"], "installed")
        self.assertIn(str(self.runtime).encode(), Path(result["agents"][0]["path"]).read_bytes())

    def test_app_and_runtime_path_change_checks_old_report_before_launcher_upgrade(self):
        self.run_integration()
        old_home_launcher = (self.home / ".local/bin/reelpack").read_bytes()
        old_app = self.app
        new_app = self.root / "new-app/Reelpack.app"
        new_app.parent.mkdir()
        self.assertEqual(old_app, self.root / "Applications with space/Reelpack.app")
        old_app.rename(new_app)
        self.app = new_app
        moved = self.run_integration()
        self.assertEqual([entry["status"] for entry in moved["launchers"]], ["upgraded", "upgraded"])
        self.assertEqual((Path(moved["launchers"][1]["backup"]) / "reelpack").read_bytes(), old_home_launcher)
        self.runtime = self.root / "moved-runtime"
        self.runtime.mkdir()
        result = self.run_integration()
        self.assertEqual(result["launchers"][1]["status"], "upgraded")
        self.assertIn(str(self.runtime).encode(), (self.home / ".local/bin/reelpack").read_bytes())

    def test_modified_managed_skill_and_marker_only_launcher_stay_conflicts(self):
        self.configure(".claude")
        self.run_integration()
        skill = self.home / ".claude/skills/reelpack/SKILL.md"
        skill.write_bytes(skill.read_bytes() + b"user customization")
        launcher = self.runtime / "bin/reelpack"
        launcher.write_bytes(launcher.read_bytes() + b"# user custom\n")
        canonical = self.runtime / "share/reelpack/SKILL.md"
        canonical.write_bytes(b"# user canonical\n")
        self.source.write_bytes(b"new official release")
        before = self.snapshot(self.home / ".claude")
        result = self.run_integration()
        self.assertEqual(result["agents"][0]["status"], "conflict")
        self.assertEqual(result["launchers"][0]["status"], "conflict")
        self.assertEqual(result["canonical_skill"]["status"], "conflict")
        self.assertEqual(self.snapshot(self.home / ".claude"), before)
        self.assertTrue(launcher.read_bytes().endswith(b"# user custom\n"))

    def test_check_only_upgrade_has_zero_write_syscalls(self):
        self.configure(".claude", ".agents", ".cursor")
        self.run_integration()
        self.source.write_bytes(b"new official release")
        before = self.snapshot(self.root)
        with patch.object(os, "open", side_effect=AssertionError("write open")), patch.object(os, "replace", side_effect=AssertionError("replace")), patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")), patch.object(Path, "write_bytes", side_effect=AssertionError("write")), patch.object(Path, "unlink", side_effect=AssertionError("unlink")):
            result = self.run_integration(check_only=True)
        self.assertEqual(result["canonical_skill"]["status"], "update_available")
        self.assertEqual([a["status"] for a in result["agents"]], ["update_available"] * 3)
        self.assertEqual(self.snapshot(self.root), before)

    def test_canonical_only_existing_copy_restores_runtime_entry_without_home_writes(self):
        canonical = self.runtime / "share/reelpack/SKILL.md"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(self.canonical)
        before = self.snapshot(self.home)
        result = self.run_integration(register=False)
        self.assertEqual(result["canonical_skill"]["status"], "reused")
        self.assertEqual(result["launchers"][0]["status"], "installed")
        self.assertEqual(self.snapshot(self.home), before)

    def test_no_register_upgrade_preserves_home_and_future_ownership(self):
        self.configure(".claude")
        self.run_integration()
        self.source.write_bytes(b"new canonical")
        before = self.snapshot(self.home)
        result = self.run_integration(register=False)
        self.assertEqual(result["canonical_skill"]["status"], "upgraded")
        self.assertEqual(self.snapshot(self.home), before)
        self.assertEqual(self.run_integration()["agents"][0]["status"], "upgraded")

    def test_skill_metadata_transaction_rolls_back_on_second_replace_failure(self):
        self.configure(".claude")
        self.run_integration()
        directory = self.home / ".claude/skills/reelpack"
        saved = {p.name: p.read_bytes() for p in directory.iterdir()}
        self.source.write_bytes(b"new release")
        replace = os.replace
        def fail_metadata(source, destination):
            if Path(destination) == directory / ".metadata.json" and Path(source).parent.name.startswith(".reelpack-update-"):
                raise OSError("injected metadata failure")
            return replace(source, destination)
        with patch.object(os, "replace", side_effect=fail_metadata):
            result = self.run_integration()
        self.assertEqual(result["agents"][0]["status"], "unsafe")
        self.assertEqual({p.name: p.read_bytes() for p in directory.iterdir()}, saved)

    def test_report_marker_without_file_hash_does_not_authorize_upgrade(self):
        self.run_integration(register=False)
        (self.runtime / "skill-integration.json").write_text('{"schema_version": 1, "managed_by": "reelpack-install"}')
        self.source.write_bytes(b"new release")
        result = self.run_integration(register=False)
        self.assertEqual(result["canonical_skill"]["status"], "conflict")

    def test_app_validator_requires_skill_release_notes_and_adapter(self):
        info = {"CFBundleExecutable": "Reelpack", "CFBundleIdentifier": "com.reelpack.app", "CFBundleIconFile": "AppIcon", "LSMinimumSystemVersion": "14.0"}
        (self.app / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
        (self.resources / "AppIcon.icns").write_bytes(b"icns")
        for name in ("runtime_environment.py", "core.py", "audio_analyze.py", "shot_detect_transnet.py", "optional_backend_worker.py", "requirements-arm64.txt", "requirements-audio.txt", "requirements-transnet.txt", "constraints-transnet.txt", "requirements-ocr.txt", "requirements-music.txt"):
            (self.resources / "Core" / name).write_bytes(b"fixture")
        for relative in ("Contents/MacOS/Reelpack", "Contents/Resources/Tools/macos_ocr/macos_ocr"):
            p = self.app / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fixture")
            p.chmod(0o700)
        notes = self.resources / "RELEASE_NOTES.md"
        notes.write_bytes(b"Release notes")
        for missing in (self.source, notes, self.adapter):
            with self.subTest(missing=missing.name):
                saved = missing.read_bytes()
                missing.write_bytes(b"")
                # Adapter existence is sufficient; remove its controlled fixture instead.
                if missing == self.adapter:
                    self.assertEqual(missing.absolute(), self.resources / "Core/agent_entry.py")
                    missing.unlink()
                errors = io.StringIO()
                with patch.object(sys, "argv", [str(INSTALLER), "app", str(self.app), "14.0"]), redirect_stdout(io.StringIO()), redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                    exec(compile(HELPER, str(INSTALLER), "exec"), {})
                self.assertEqual(raised.exception.code, 1)
                self.assertIn(missing.name, errors.getvalue())
                missing.write_bytes(saved)
        with patch.object(sys, "argv", [str(INSTALLER), "app", str(self.app), "14.0"]), redirect_stdout(io.StringIO()):
            exec(compile(HELPER, str(INSTALLER), "exec"), {})


if __name__ == "__main__":
    unittest.main()
