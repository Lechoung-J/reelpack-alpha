import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseIdentityTests(unittest.TestCase):
    def test_product_versions_are_consistent(self):
        version = (ROOT/'VERSION').read_text().strip()
        self.assertRegex(version, r'^\d+\.\d+\.\d+$')
        cli = (ROOT/'CLI/main.swift').read_text()
        self.assertIn(f'version: "{version}"', cli)
        self.assertIn(f'Reelpack CLI v{version}', cli)
        for name in ['README.md', 'README.zh-CN.md', 'RELEASE_NOTES.md', 'SKILL.md']:
            self.assertIn(version, (ROOT/name).read_text(), name)
        builder = (ROOT/'Build/build_app.py').read_text()
        ast.parse(builder)
        for field in ['source_commit', 'source_dirty', 'release_tag']:
            self.assertIn(field, builder)
        self.assertIn('release builds require a clean committed source tree', builder)
        self.assertIn("reelpack-cli', '--version'", builder)

    def test_public_instructions_have_explicit_versioned_source_and_checksums(self):
        version = (ROOT/'VERSION').read_text().strip()
        for name in ['README.md', 'README.zh-CN.md']:
            text = (ROOT/name).read_text()
            self.assertIn('releases/download/v0.2.0', text)
            self.assertIn('--version v0.2.0 --public-release', text)
            self.assertIn('--archive Reelpack-0.2.0-arm64.zip', text)
            self.assertIn('shasum -a 256 -c install.sh.sha256', text)
            self.assertIn(f'releases/tag/v{version}', text)
            self.assertIn('docs/cli-install.md', text)
            self.assertEqual(text.count('```') % 2, 0)
        instructions = (ROOT/'docs/cli-install.md').read_text()
        self.assertIn(f'releases/download/v{version}', instructions)
        self.assertIn(f'--archive Reelpack-CLI-{version}-arm64.zip', instructions)
        self.assertIn('shasum -a 256 -c install-cli.sh.sha256', instructions)
        self.assertNotIn('/releases/latest', instructions)


if __name__ == '__main__': unittest.main()
