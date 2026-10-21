"""Fast shell-level acceptance coverage for installer tiers and presentation."""
from pathlib import Path
import os
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install.sh"


def run_installer(*arguments, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["/bin/bash", str(INSTALLER), *arguments],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )


class InstallerTierTests(unittest.TestCase):
    def test_help_lists_tier_and_noninteractive_controls(self):
        result = run_installer("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--tier", "--quiet", "--non-interactive", "--dry-run"):
            self.assertIn(option, result.stdout)

    def test_lite_dry_run_is_nonblocking_and_reports_small(self):
        result = run_installer("--tier", "lite", "--dry-run", "--non-interactive")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lite", result.stdout)
        self.assertIn("small", result.stdout)
        self.assertIn("five isolated Python environments", result.stdout)
        self.assertIn("no download will run", result.stdout)

    def test_banner_respects_quiet_environment_and_narrow_width(self):
        quiet = run_installer("--dry-run", "--quiet")
        self.assertEqual(quiet.returncode, 0, quiet.stderr)
        self.assertNotIn("the second eye", quiet.stdout)

        env_quiet = run_installer("--dry-run", extra_env={"REELPACK_NO_BANNER": "1"})
        self.assertNotIn("the second eye", env_quiet.stdout)

        normal = run_installer("--dry-run")
        self.assertEqual(normal.returncode, 0, normal.stderr)
        self.assertIn("Reelpack", normal.stdout)
        self.assertIn("the second eye for your scroll · v0.2.0", normal.stdout)

        narrow = run_installer("--dry-run", extra_env={"COLUMNS": "40"})
        self.assertEqual(narrow.returncode, 0, narrow.stderr)
        first_line = narrow.stdout.splitlines()[0]
        self.assertIn("the second eye for your scroll", first_line)
        self.assertNotIn("A " + "second eye", first_line)
        self.assertIn("Reelpack 0.2.0", first_line)

    def test_product_version_matches_repository_version(self):
        installer_text = INSTALLER.read_text()
        match = re.search(r'^PRODUCT_VERSION="([^"]+)"', installer_text, re.MULTILINE)
        self.assertIsNotNone(match, "install.sh must define PRODUCT_VERSION")
        repository_version = (ROOT / "VERSION").read_text().strip()
        self.assertEqual(match.group(1), repository_version)

    def test_missing_model_guidance_has_size_quality_and_no_silent_download(self):
        result = run_installer("--dry-run", "--non-interactive")
        self.assertIn("3.1 GB", result.stdout)
        self.assertIn("small is faster and smaller", result.stdout)
        self.assertIn("No model is downloaded automatically", result.stdout)
        self.assertIn("WhisperModel", result.stdout)
        self.assertIn("large-v3", result.stdout)


if __name__ == "__main__":
    unittest.main()
