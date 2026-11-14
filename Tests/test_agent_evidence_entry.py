"""Installed `reelpack` evidence retrieval: real fixture, real subprocess entry.

No ASR/model is imported or run. Small H.264 fixtures are encoded with the
local ffmpeg; review OCR is stubbed only where the optional macOS Vision binary
is unavailable, while routing, storage, sampling and decoding stay real.
"""
import contextlib
import importlib.abc
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "Core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from asset_contract import empty_analysis
from asset_store import AssetStore

FFMPEG = shutil.which("ffmpeg")


def encode_video(path, seconds=6):
    subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
         "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", "-y", str(path)],
        check=True, capture_output=True, text=True, timeout=120,
    )


def extract_frame(video, timestamp, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(video),
         "-frames:v", "1", "-y", str(destination)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return destination


def snapshot(root):
    records = {}
    for path in Path(root).rglob("*"):
        if path.is_file():
            records[str(path.relative_to(root))] = ("file", path.read_bytes())
        elif path.is_dir():
            records[str(path.relative_to(root))] = ("dir",)
    return records


def run_entry(arguments, *, home=None, extra_env=None, timeout=180):
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if home is not None:
        environment["HOME"] = str(home)
        environment.pop("REELPACK_LIBRARY_ROOT", None)
        environment.pop("REELPACK_OUTPUT_ROOT", None)
    environment["REELPACK_CACHE_ROOT"] = str(Path(environment["HOME"]) / "cache")
    environment.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-B", str(CORE / "agent_entry.py"), *arguments],
        capture_output=True, text=True, timeout=timeout, env=environment,
    )


@unittest.skipUnless(FFMPEG, "ffmpeg is required for the encoded evidence fixture")
class EvidenceEntrySubprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_temp = tempfile.TemporaryDirectory(prefix="reelpack-evidence-template-")
        template = Path(cls.template_temp.name)
        cls.video = template / "source clip.mp4"
        encode_video(cls.video)
        library = template / "library"
        store = AssetStore(library)
        imported = store.import_file(cls.video)
        cls.asset_id = imported["asset_id"]
        package = Path(imported["package_dir"])
        frames = [
            extract_frame(cls.video, 0.5, package / "evidence/frame_1.jpg"),
            extract_frame(cls.video, 2.5, package / "evidence/frame_2.jpg"),
            extract_frame(cls.video, 5.0, package / "evidence/frame_3.jpg"),
        ]
        cls.evidence = [
            {"evidence_id": f"frame_{index + 1}", "kind": "image", "timestamp": timestamp,
             "relative_path": f"evidence/frame_{index + 1}.jpg", "shot_id": shot_id,
             "purpose": "shot_representative"}
            for index, (timestamp, shot_id) in enumerate([(0.5, "shot_0001"),
                                                          (2.5, "shot_0001"),
                                                          (5.0, "shot_0002")])
        ]
        analysis = empty_analysis(cls.asset_id, {"kind": "local_file"})
        analysis["status"] = "partial"
        analysis["video"] = {"duration_seconds": 6.0, "width": 320, "height": 240,
                             "fps": 10.0, "audio_track": {"status": "complete", "present": True}}
        analysis["capabilities"]["transcription"] = {"status": "not_run", "backend": None, "model": None}
        analysis["capabilities"]["ocr"] = {"status": "complete", "backend": "stub", "model": "stub"}
        analysis["capabilities"]["shots"] = {"status": "complete", "backend": "stub", "model": "stub"}
        analysis["shots"] = {"status": "complete", "backend": "stub", "items": [
            {"shot_id": "shot_0001", "start": 0.0, "end": 3.0, "boundary": {"uncertain": False},
             "representative_frame": "frame_1", "evidence_ids": ["frame_1", "frame_2"]},
            {"shot_id": "shot_0002", "start": 3.0, "end": 6.0, "boundary": {"uncertain": False},
             "representative_frame": "frame_3", "evidence_ids": ["frame_3"]},
        ]}
        analysis["evidence"] = cls.evidence
        analysis["transcript"] = {
            "status": "not_run", "raw_text": "",
            "raw_segments": [{"segment_id": "seg_1", "start": 0.5, "end": 2.0,
                              "text": "第一句 参考"}],
        }
        analysis["ocr"] = {
            "status": "complete", "reading_text": "字幕 KEY",
            "items": [{"ocr_id": "ocr_1", "start": 1.0, "end": 1.2, "timestamp": 1.0,
                       "text": "字幕 KEY", "status": "complete",
                       "evidence_ids": ["frame_1"]}],
        }
        store.save_analysis(cls.asset_id, analysis)
        assert len(frames) == 3
        cls.template_library = library

    @classmethod
    def tearDownClass(cls):
        cls.template_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reelpack-evidence-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.library = self.root / "library"
        shutil.copytree(self.template_library, self.library)
        self.home = self.root / "home"
        self.home.mkdir()

    def test_frames_and_montage_route_through_installed_entry(self):
        result = run_entry(["frames", self.asset_id, "--start", "0", "--end", "3",
                            "--view", "overview", "--grid", "9",
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["asset_id"], self.asset_id)
        self.assertEqual(payload["view"], "overview")
        self.assertEqual(payload["grid"], 9)
        self.assertEqual(len(payload["pages"]), 1)
        page = payload["pages"][0]
        self.assertTrue(Path(page["absolute_path"]).is_file())
        self.assertGreater(Path(page["absolute_path"]).stat().st_size, 0)
        self.assertEqual([item["evidence_id"] for item in payload["frames"]],
                         ["frame_1", "frame_2"])
        self.assertTrue(all(0 <= item["timestamp"] <= 3 for item in payload["frames"]))
        self.assertIn("timestamp", payload["frames"][0])
        self.assertIn("relative_path", payload["frames"][0])

        alias = run_entry(["montage", self.asset_id, "--shot", "shot_0002",
                           "--library-root", str(self.library)], home=self.home)
        self.assertEqual(alias.returncode, 0, alias.stderr)
        alias_payload = json.loads(alias.stdout)
        self.assertEqual(alias_payload["asset_id"], self.asset_id)
        self.assertEqual([item["evidence_id"] for item in alias_payload["frames"]], ["frame_3"])

    def test_detail_frame_returns_existing_path_resolution_and_source(self):
        result = run_entry(["frames", self.asset_id, "--start", "0", "--end", "3",
                            "--view", "detail", "--timestamp", "2.5",
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["view"], "detail")
        self.assertEqual(payload["source"], "proxy")
        self.assertEqual(payload["resolution"], "320x240")
        self.assertTrue(Path(payload["absolute_path"]).is_file())
        self.assertGreaterEqual(payload["width"], 1)
        self.assertGreaterEqual(payload["height"], 1)
        frame = payload["frames"][0]
        self.assertEqual(frame["timestamp"], 2.5)
        self.assertEqual(frame["resolution"], "320x240")

    def test_review_routes_and_returns_requested_range_artifacts(self):
        output = self.root / "review output with spaces"
        result = run_entry(["review", self.asset_id, "--start", "1", "--end", "2",
                            "--interval", "0.5", "--padding", "0",
                            "--output", str(output), "--library-root", str(self.library)],
                           home=self.home)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["asset_id"], self.asset_id)
        self.assertEqual(payload["range"], [1.0, 2.0])
        self.assertTrue(Path(payload["audio"]).is_file())
        review_json = output / "review.json"
        self.assertTrue(review_json.is_file())
        self.assertEqual(json.loads(review_json.read_text(encoding="utf-8")), payload)
        self.assertTrue((output / "review.txt").is_file())
        self.assertEqual([item["timestamp"] for item in payload["items"]], [1.0, 1.5, 2.0])
        for item in payload["items"]:
            self.assertTrue(Path(item["path"]).is_file(), item)
            # The optional macOS Vision OCR binary is not installed in this fixture.
            self.assertEqual(item["status"], "failed")
            self.assertTrue(item["reason"])
        self.assertEqual(payload["status"], "failed")

    def test_review_with_injected_ocr_returns_complete_items(self):
        # Replace only the OCR backend seam; entry routing, decoding, sampling,
        # audio export and package reads all remain the real implementation.
        import asset_enrich
        defaults = list(asset_enrich.review.__defaults__)
        defaults[-1] = lambda path: "字幕候选"
        output = self.root / "review-complete"
        with mock.patch.object(asset_enrich.review, "__defaults__", tuple(defaults)):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                code = __import__("agent_entry").main([
                    "review", self.asset_id, "--start", "1", "--end", "2",
                    "--interval", "0.5", "--padding", "0",
                    "--output", str(output), "--library-root", str(self.library)])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "complete")
        self.assertTrue(all(item["status"] == "complete" for item in payload["items"]))
        self.assertEqual({item["text"] for item in payload["items"]}, {"字幕候选"})
        self.assertEqual([entry["text"] for entry in payload["candidate_subtitles"]],
                         ["字幕候选", "字幕候选", "字幕候选"])

    def test_help_creates_no_default_library(self):
        result = run_entry(["frames", "--help"], home=self.home)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("asset_id", result.stdout)
        self.assertFalse((self.home / "Downloads").exists())
        self.assertFalse((self.home / "cache").exists())

    def test_top_level_help_and_query_create_no_library_or_cache(self):
        result = run_entry(["--help"], home=self.home)
        # Without a native CLI the entry still prints its own help, then reports
        # the missing forwarder; the read-only path must not create anything.
        self.assertIn("preview", result.stdout)
        self.assertIn("frames", result.stdout)
        self.assertIn("review", result.stdout)
        self.assertFalse((self.home / "Downloads").exists())
        self.assertFalse((self.home / "cache").exists())

        output = self.root / "unused output"
        result = run_entry(["query", "--output-dir", str(output), "text"], home=self.home)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.strip())
        self.assertFalse(output.exists())

    def test_missing_library_and_asset_fail_without_creating_files(self):
        missing = self.root / "missing library"
        result = run_entry(["frames", self.asset_id, "--start", "0", "--end", "3",
                            "--library-root", str(missing)], home=self.home)
        self.assertEqual(result.returncode, 2)
        self.assertIn("找不到已有资产库", result.stderr)
        self.assertFalse(missing.exists())

        before = snapshot(self.library)
        result = run_entry(["frames", "asset_absent", "--start", "0", "--end", "3",
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.strip())
        self.assertEqual(before, snapshot(self.library))

    def test_invalid_ranges_fail_explicitly_and_create_no_output(self):
        before = snapshot(self.library)
        result = run_entry(["frames", self.asset_id, "--start", "5", "--end", "2",
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.strip())
        self.assertEqual(before, snapshot(self.library))

        output = self.root / "never-created"
        result = run_entry(["review", self.asset_id, "--start", "3", "--end", "2",
                            "--output", str(output), "--library-root", str(self.library)],
                           home=self.home)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())

    def test_evidence_commands_do_not_import_analysis_engine(self):
        import agent_entry
        blocked = {"core", "faster_whisper", "torch", "cv2", "transformers", "numpy"}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise AssertionError(f"evidence command imported {name}")
                return None

        blocker = Blocker()
        sys.meta_path.insert(0, blocker)
        self.addCleanup(sys.meta_path.remove, blocker)
        for module in list(sys.modules):
            if module.split(".")[0] in blocked:
                self.skipTest(f"test process already imported {module}")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = agent_entry.main(["frames", self.asset_id, "--start", "0", "--end", "3",
                                     "--library-root", str(self.library)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["asset_id"], self.asset_id)


if __name__ == "__main__":
    unittest.main()
