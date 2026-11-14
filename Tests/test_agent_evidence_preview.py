"""Installed `reelpack preview`: whole-video board before ASR/import.

Real ffmpeg fixtures, real subprocess entry, PIL read-back of the board. No
analysis, model, network or library directory may be created or touched.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "Core"
FFMPEG = shutil.which("ffmpeg")


def encode_video(path, seconds=8):
    subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=10", "-t", str(seconds),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)],
        check=True, capture_output=True, text=True, timeout=120,
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root):
    records = {}
    for path in Path(root).rglob("*"):
        if path.is_file():
            records[str(path.relative_to(root))] = ("file", path.read_bytes())
        elif path.is_dir():
            records[str(path.relative_to(root))] = ("dir",)
    return records


def run_preview(arguments, home, extra_env=None):
    environment = dict(os.environ)
    environment["HOME"] = str(home)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["REELPACK_CACHE_ROOT"] = str(Path(home) / "cache")
    environment.pop("REELPACK_LIBRARY_ROOT", None)
    environment.pop("REELPACK_OUTPUT_ROOT", None)
    environment.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-B", str(CORE / "agent_entry.py"), *arguments],
        capture_output=True, text=True, timeout=180, env=environment,
    )


@unittest.skipUnless(FFMPEG, "ffmpeg is required for the preview fixture")
class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_temp = tempfile.TemporaryDirectory(prefix="reelpack-preview-template-")
        template = Path(cls.template_temp.name)
        cls.video = template / "media dir with spaces" / "clip with spaces.mp4"
        cls.video.parent.mkdir()
        encode_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        cls.template_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reelpack-preview-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.output = self.root / "output dir with spaces" / "board"

    def test_preview_builds_readable_board_and_metadata_without_side_effects(self):
        before = digest(self.video)
        result = run_preview(["preview", str(self.video), "--output", str(self.output),
                              "--frames", "12"], self.home)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["command"], "preview")
        self.assertEqual(payload["source"]["path"], str(self.video.resolve()))
        self.assertEqual(payload["source"]["size_bytes"], self.video.stat().st_size)
        self.assertEqual(payload["requested_frames"], 12)
        self.assertEqual(payload["grid"]["count"], 12)
        self.assertEqual(payload["grid"]["cols"], 4)
        self.assertEqual(payload["grid"]["rows"], 3)
        self.assertEqual(len(payload["frames"]), 12)
        timestamps = [frame["timestamp"] for frame in payload["frames"]]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertGreater(timestamps[0], 0)
        self.assertLess(timestamps[-1], 8)
        self.assertAlmostEqual(payload["duration_seconds"], 8.0, delta=0.2)
        self.assertFalse(payload["analysis_run"])
        self.assertFalse(payload["network_used"])
        self.assertFalse(payload["model_used"])
        self.assertFalse(payload["content_classified"])
        self.assertFalse(payload["profile_selected"])

        board = Path(payload["board"]["path"])
        self.assertEqual(board, (self.output / "preview-board.jpg").resolve())
        self.assertTrue(board.is_file())
        self.assertGreater(board.stat().st_size, 0)
        from PIL import Image
        with Image.open(board) as image:
            self.assertEqual(image.size, (payload["grid"]["frame_width"],
                                          payload["grid"]["frame_height"]))
            image.verify()

        metadata = self.output / "preview.json"
        self.assertTrue(metadata.is_file())
        self.assertEqual(json.loads(metadata.read_text(encoding="utf-8")), payload)
        self.assertEqual(digest(self.video), before)
        self.assertFalse((self.home / "Downloads").exists())
        self.assertFalse((self.home / "cache").exists())

    def test_missing_and_corrupt_media_fail_without_leaving_output(self):
        missing = self.root / "missing.mp4"
        result = run_preview(["preview", str(missing), "--output", str(self.output)], self.home)
        self.assertEqual(result.returncode, 2)
        self.assertIn("源视频", result.stderr)
        self.assertFalse(self.output.exists())

        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"not a real video")
        corrupt_before = corrupt.read_bytes()
        result = run_preview(["preview", str(corrupt), "--output", str(self.output)], self.home)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.strip())
        self.assertFalse(self.output.exists())
        self.assertEqual(corrupt.read_bytes(), corrupt_before)

    def test_output_conflicts_are_refused_without_overwriting(self):
        self.output.mkdir(parents=True)
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep me", encoding="utf-8")
        before = snapshot(self.output)
        result = run_preview(["preview", str(self.video), "--output", str(self.output)], self.home)
        self.assertEqual(result.returncode, 2)
        self.assertIn("非空", result.stderr)
        self.assertEqual(snapshot(self.output), before)

        result = run_preview(["preview", str(self.video), "--output", str(self.video)], self.home)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.strip())

        library = self.root / "library"
        result = run_preview(["preview", str(self.video), "--output", str(library / "preview")],
                             self.home, {"REELPACK_LIBRARY_ROOT": str(library)})
        self.assertEqual(result.returncode, 2)
        self.assertIn("资产库", result.stderr)
        self.assertFalse(library.exists())

    def test_help_creates_nothing(self):
        result = run_preview(["preview", "--help"], self.home)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)
        self.assertFalse((self.home / "Downloads").exists())
        self.assertFalse((self.home / "cache").exists())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
