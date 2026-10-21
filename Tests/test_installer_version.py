"""Offline tests of real installer helpers; never run the full installer."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from Tests.test_skill_install import HELPER, INSTALLER, INSTALLER_TEXT, NAMESPACE


class InstallerVersionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="reelpack-version-tests-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.manifest = self.root / "manifest.json"
        self.data = {"schema_version": 1, "version": "0.1.1", "release_tag": "v0.1.1", "repository": "Lechoung-J/Reelpack-release", "assets": {"arm64": {"filename": "Reelpack-0.1.1-arm64.zip", "sha256": "a" * 64}}}

    def helper(self, command, *args):
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with patch.object(sys, "argv", [str(INSTALLER), command, *map(str, args)]), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                exec(compile(HELPER, str(INSTALLER), "exec"), {})
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def manifest_result(self, requested=""):
        self.manifest.write_text(json.dumps(self.data))
        return self.helper("manifest", self.manifest, "arm64", self.data["repository"], requested)

    def test_tag_matrix_and_legacy_manifest(self):
        for value, expected in [("0.1.1", "v0.1.1"), ("v0.1.1", "v0.1.1"), ("preview-r2", "preview-r2"), ("01.1.1", "01.1.1"), ("0.1.1-rc.1", "v0.1.1-rc.1")]:
            self.assertEqual(NAMESPACE["release_tag"](value), expected)
        for requested in ("", "0.1.1", "v0.1.1"):
            code, out, err = self.manifest_result(requested)
            self.assertEqual(code, 0, err)
            self.assertIn("version\t0.1.1\nrelease_tag\tv0.1.1\n", out)
        del self.data["release_tag"]
        self.assertIn("release_tag\tv0.1.1", self.manifest_result()[1])
        self.data["release_tag"] = "custom-r2"
        self.assertEqual(self.manifest_result("custom-r2")[0], 0)
        self.assertNotEqual(self.manifest_result("0.1.1")[0], 0)

    def test_explicit_url_tag_filename_hash_binding(self):
        asset = self.data["assets"]["arm64"]
        asset["url"] = "https://github.com/Lechoung-J/Reelpack-release/releases/download/v0.1.1/Reelpack-0.1.1-arm64.zip"
        self.assertEqual(self.manifest_result("0.1.1")[0], 0)
        for bad in ("v0.1.2/Reelpack-0.1.1-arm64.zip", "v0.1.1/other.zip"):
            asset["url"] = "https://github.com/Lechoung-J/Reelpack-release/releases/download/" + bad
            self.assertNotEqual(self.manifest_result()[0], 0)
        asset["url"] = "https://mirror.example/releases/Reelpack-0.1.1-arm64.zip"
        self.assertEqual(self.manifest_result()[0], 0)
        asset["sha256"] = "invalid"
        self.assertNotEqual(self.manifest_result()[0], 0)

    def test_manifest_mismatches_rejected(self):
        self.assertNotEqual(self.manifest_result("v0.1.2")[0], 0)
        self.data["version"] = "v0.1.1"
        self.assertNotEqual(self.manifest_result()[0], 0)
        self.data["version"] = "0.1.2"
        self.assertNotEqual(self.manifest_result()[0], 0)

    def test_download_uses_tag_explicit_url_and_optional_sidecar(self):
        self.assertIn('VERSION=$MANIFEST_RELEASE_TAG', INSTALLER_TEXT)
        self.assertIn('asset_url=${MANIFEST_ASSET_URL:-"${GITHUB_BASE%/}/$VERSION/$ASSET_NAME"}', INSTALLER_TEXT)
        self.assertIn('public_download "${GITHUB_BASE%/}/$VERSION/$checksum_name"', INSTALLER_TEXT)
        self.assertNotIn('VERSION=$MANIFEST_VERSION', INSTALLER_TEXT)
        digest = "b" * 64
        checksum = self.root / "asset.sha256"
        checksum.write_text(digest + "  Reelpack-0.1.1-arm64.zip\n")
        self.assertEqual(self.helper("checksum", checksum, "Reelpack-0.1.1-arm64.zip")[:2], (0, digest + "\n"))
        self.assertNotEqual(self.helper("checksum", checksum, "another.zip")[0], 0)

    def test_invalid_arguments_exit_before_platform_or_runtime_writes(self):
        # Execute only the real argument-parser prefix, with a tripwire before platform/Python discovery.
        prefix = INSTALLER_TEXT.split('[ "$(/usr/bin/uname -s)" = Darwin ]', 1)[0]
        for arguments in (("--manifest-url", "https://example.com/release.json"), ("--version", "bad/tag"), ("--asset-name", "../bad.zip"), ("--repair-env", "unknown"), ("--github-base", "http://example.com"), ("--check", "--replace-app")):
            process = subprocess.run(["/bin/bash", "-c", prefix + '\nexit 97\n', "installer", *arguments], env={"HOME": str(self.root), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}, capture_output=True, text=True)
            self.assertEqual(process.returncode, 2, (arguments, process.stderr))
            self.assertEqual(list(self.root.iterdir()), [])
        process = subprocess.run(["/bin/bash", "-c", prefix + '\nprintf "%s" "$VERSION"\n', "installer", "--public-release", "--version", "0.1.1"], env={"HOME": str(self.root), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "REELPACK_NO_BANNER": "1"}, capture_output=True, text=True)
        self.assertEqual((process.returncode, process.stdout), (0, "v0.1.1"), process.stderr)

    def test_success_cleans_only_exact_empty_promoted_stage(self):
        target = self.root / "Reelpack.app"
        target.mkdir()
        own = self.root / ".reelpack-stage.current"
        other = self.root / ".reelpack-stage.someone-else"
        own.mkdir()
        other.mkdir()
        self.assertEqual(self.helper("clean-app-stage", own, self.root, target)[0], 0)
        self.assertFalse(own.exists())
        self.assertTrue(other.exists())
        own.mkdir()
        (own / "failure-evidence").write_text("keep")
        self.assertNotEqual(self.helper("clean-app-stage", own, self.root, target)[0], 0)
        self.assertEqual((own / "failure-evidence").read_text(), "keep")
        outside = self.root / "unrelated"
        outside.mkdir()
        self.assertNotEqual(self.helper("clean-app-stage", outside, self.root, target)[0], 0)
        self.assertTrue(outside.exists())
        self.assertIn('App staging preserved for diagnosis: $APP_STAGE', INSTALLER_TEXT)
        self.assertLess(INSTALLER_TEXT.index('py_task move "$APP_STAGE/Reelpack.app"'), INSTALLER_TEXT.index('py_task clean-app-stage "$APP_STAGE"'))


if __name__ == "__main__":
    unittest.main()
