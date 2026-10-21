"""Supplementary contract checks for the Swift CLI model option.

The CLI is not a SwiftPM test target, so this checks its allowlist and argument
forwarding without changing Package.swift or duplicating production logic.
"""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI_SOURCE = (ROOT / "CLI/main.swift").read_text(encoding="utf-8")


class CLIModelOptionTests(unittest.TestCase):
    def test_model_default_remains_large(self):
        self.assertRegex(CLI_SOURCE, r'var model:\s*String\s*=\s*"large"')
        self.assertRegex(CLI_SOURCE, r'转录模型：small 或 large；默认 large（large-v3）')

    def test_explicit_small_is_allowed_and_forwarded_to_core(self):
        self.assertRegex(CLI_SOURCE, r'guard\s*\["small",\s*"large"\]\.contains\(model\)')
        arguments = CLI_SOURCE.split("arguments: [", 1)[1].split("].filter", 1)[0]
        self.assertRegex(arguments, r'"--model",\s*model')
        self.assertIn('"--model must be small or large (allowed: small|large)."', CLI_SOURCE)

    def test_unsupported_base_is_not_in_the_allowlist(self):
        match = re.search(r'guard\s*\[("small",\s*"large")\]\.contains\(model\)', CLI_SOURCE)
        self.assertIsNotNone(match)
        self.assertNotIn('"base"', match.group(1))
        self.assertIn("allowed: small|large", CLI_SOURCE)


if __name__ == "__main__":
    unittest.main()
