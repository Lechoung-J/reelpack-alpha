"""CLI profile contract and cache-signature consequences.

The Swift CLI is not a SwiftPM test target. Validation is exercised through a
built binary when one exists; forwarding is observed by launching the binary
against a fake runtime interpreter that records its argv.
"""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI_SOURCE = (ROOT / "CLI/main.swift").read_text(encoding="utf-8")
sys.path.insert(0, str(ROOT / "Core"))

from pipeline_state import reusable, run_signature


def cli_binary():
    candidates = [ROOT / ".build/debug/reelpack-cli", *ROOT.glob(".build/*/debug/reelpack-cli")]
    return next((path for path in candidates if path.is_file()), None)


BINARY = cli_binary()


class ProfileSignatureTests(unittest.TestCase):
    def test_auto_profile_keeps_legacy_signature_byte_identical(self):
        base = run_signature("a" * 64, {})
        self.assertNotIn("profile", base["options"])
        self.assertEqual(base, run_signature("a" * 64, {"profile": "auto"}))

    def test_each_explicit_profile_changes_signature(self):
        digests = {}
        for profile in ("ad", "talk", "podcast"):
            with self.subTest(profile=profile):
                signature = run_signature("a" * 64, {"profile": profile})
                self.assertEqual(signature["options"]["profile"], profile)
                self.assertNotEqual(signature["digest"], run_signature("a" * 64, {})["digest"])
                digests[profile] = signature["digest"]
        self.assertEqual(len(set(digests.values())), 3)

    def test_cached_ocr_result_is_not_reused_across_profiles(self):
        base = run_signature("a" * 64, {})
        analysis = {
            "status": "complete", "run_signature": base,
            "capabilities": {name: {"status": "complete"}
                             for name in ("transcription", "shots", "ocr")},
            "transcript": {"status": "complete"}, "shots": {"status": "complete"},
            "ocr": {"status": "complete"}, "execution": {"status": "complete"},
        }
        self.assertTrue(reusable(analysis, base))
        talk = run_signature("a" * 64, {"profile": "talk"})
        self.assertFalse(reusable(analysis, talk))


class ProfileCLIContractTests(unittest.TestCase):
    def test_profile_option_and_forwarding_are_declared(self):
        self.assertIn('var profile: String = "auto"', CLI_SOURCE)
        self.assertIn('["auto", "ad", "talk", "podcast"].contains(profile)', CLI_SOURCE)
        self.assertIn('"--profile", profile', CLI_SOURCE)
        self.assertIn("画像", CLI_SOURCE)
        self.assertIn('["off", "auto"].contains(speakers)', CLI_SOURCE)
        self.assertIn('"--speakers", speakers', CLI_SOURCE)

    @unittest.skipUnless(BINARY, "build the CLI first: swift build")
    def test_invalid_profile_is_rejected_before_launch(self):
        result = subprocess.run([str(BINARY), "/nonexistent/clip.mp4", "--profile", "movie"],
                                capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("画像", result.stderr + result.stdout)
        help_result = subprocess.run([str(BINARY), "--help"], capture_output=True,
                                     text=True, timeout=60)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--profile", help_result.stdout)
        self.assertIn("auto", help_result.stdout)

    def run_with_fake_runtime(self, extra_arguments):
        temp = tempfile.TemporaryDirectory(prefix="reelpack-profile-cli-")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        home = root / "home"
        home.mkdir()
        runtime = root / "runtime"
        (runtime / ".venv/bin").mkdir(parents=True)
        capture = root / "captured-args.txt"
        fake_python = runtime / ".venv/bin/python"
        fake_python.write_text(
            "#!/bin/sh\n: > \"$REELPACK_TEST_CAPTURE\"\n"
            "for argument in \"$@\"; do printf '%s\\n' \"$argument\" >> \"$REELPACK_TEST_CAPTURE\"; done\n"
            "exit 0\n", encoding="utf-8")
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
        source = root / "local clip.mp4"
        source.write_bytes(b"fixture bytes; the fake interpreter never decodes")
        environment = dict(os.environ)
        environment.update({
            "HOME": str(home),
            "REELPACK_RUNTIME_ROOT": str(runtime),
            "REELPACK_TEST_CAPTURE": str(capture),
        })
        result = subprocess.run(
            [str(BINARY), str(source), "--output-dir", str(root / "output dir"),
             *extra_arguments],
            capture_output=True, text=True, timeout=60, env=environment)
        self.assertTrue(capture.is_file(), result.stderr)
        return result.returncode, capture.read_text(encoding="utf-8").splitlines()

    @unittest.skipUnless(BINARY, "build the CLI first: swift build")
    def test_explicit_profile_is_forwarded_to_core(self):
        code, arguments = self.run_with_fake_runtime(["--profile", "podcast"])
        # No completion message from the fake interpreter, so the CLI reports
        # failure; the argv it launched is what this test observes.
        self.assertNotEqual(code, 0)
        index = arguments.index("--profile")
        self.assertEqual(arguments[index + 1], "podcast")
        self.assertEqual(arguments[arguments.index("--speakers") + 1], "off")

    @unittest.skipUnless(BINARY, "build the CLI first: swift build")
    def test_default_profile_is_auto_and_forwarded(self):
        _, arguments = self.run_with_fake_runtime([])
        self.assertEqual(arguments[arguments.index("--profile") + 1], "auto")


if __name__ == "__main__":
    unittest.main()
