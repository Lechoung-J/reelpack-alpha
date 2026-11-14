"""Installed entry for motion, annotate and supplements.

The subprocess uses the real ``Core/agent_entry.py`` and a real small H.264
fixture.  No ASR, model download or network access is involved.  Read-only
commands are checked against a byte snapshot so routing cannot silently create
libraries, caches, lock files or supplements.
"""
import contextlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "Core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from asset_contract import empty_analysis
from asset_store import AssetStore

FFMPEG = shutil.which("ffmpeg")


def encode_video(path):
    subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10",
         "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
         "-y", str(path)],
        check=True, capture_output=True, text=True, timeout=120,
    )


def snapshot(root):
    records = {}
    for path in Path(root).rglob("*"):
        if path.is_file():
            records[str(path.relative_to(root))] = ("file", path.read_bytes())
        elif path.is_dir():
            records[str(path.relative_to(root))] = ("dir",)
    return records


def run_entry(arguments, *, home, timeout=180):
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HOME"] = str(home)
    environment.pop("REELPACK_LIBRARY_ROOT", None)
    environment.pop("REELPACK_OUTPUT_ROOT", None)
    environment["REELPACK_CACHE_ROOT"] = str(Path(home) / "cache")
    return subprocess.run(
        [sys.executable, "-B", str(CORE / "agent_entry.py"), *arguments],
        capture_output=True, text=True, timeout=timeout, env=environment,
    )


@unittest.skipUnless(FFMPEG, "ffmpeg is required for the encoded fixture")
class SupplementEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_temp = tempfile.TemporaryDirectory(prefix="reelpack-supplement-template-")
        template = Path(cls.template_temp.name)
        cls.video = template / "source clip.mp4"
        encode_video(cls.video)
        library = template / "library"
        store = AssetStore(library)
        imported = store.import_file(cls.video)
        cls.asset_id = imported["asset_id"]
        analysis = empty_analysis(cls.asset_id, {"kind": "local_file"})
        analysis["status"] = "complete"
        analysis["video"] = {"duration_seconds": 3.0, "width": 320, "height": 240, "fps": 10.0}
        analysis["speakers"] = {
            "status": "complete", "backend": "fixture", "experimental": True,
            "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "one"},
                {"start": 1.0, "end": 2.0, "speaker": "S2", "text": "two"},
            ],
        }
        analysis["evidence"] = [
            {"evidence_id": "frame_1", "kind": "image", "timestamp": 0.5,
             "relative_path": "evidence/frame_1.jpg"},
        ]
        analysis["shots"] = {"status": "complete", "backend": "fixture", "items": [
            {"shot_id": "shot_1", "start": 0.0, "end": 3.0},
        ]}
        store.save_analysis(cls.asset_id, analysis)
        cls.template_library = library

    @classmethod
    def tearDownClass(cls):
        cls.template_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reelpack-supplement-entry-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.library = self.root / "library"
        shutil.copytree(self.template_library, self.library)
        self.home = self.root / "home"
        self.home.mkdir()

    def annotation(self, **overrides):
        value = {
            "schema": "reelpack.agent-annotations",
            "schema_version": 1,
            "asset_id": self.asset_id,
            "author": "entry-test",
            "assertions": [{
                "assertion_id": "a_voice",
                "kind": "voice_person",
                "status": "tentative",
                "person_id": "P1",
                "speaker": "S1",
                "value": None,
                "reasoning": "S1 is active while the left figure is visible in frame_1.",
                "uncertainty": "No lip-sync measurement; identity remains an interpretation.",
                "confidence": 0.55,
                "evidence_refs": [
                    {"kind": "package_evidence", "evidence_id": "frame_1"},
                    {"kind": "media_range", "start": 0.0, "end": 2.0},
                ],
            }],
        }
        value.update(overrides)
        return value

    def write_input(self, payload, name="annotations.json"):
        path = self.root / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_motion_read_only_does_not_touch_the_library(self):
        before = snapshot(self.library)
        result = run_entry(["motion", self.asset_id, "--start", "0", "--end", "1",
                            "--sample-fps", "10", "--library-root", str(self.library)],
                           home=self.home)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["command"], "motion")
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["cuts"]["status"], "applied")
        self.assertEqual(snapshot(self.library), before)
        self.assertFalse((self.library / ".locks").exists())
        self.assertFalse((self.library / "packages" / self.asset_id / "supplements").exists())

    def test_motion_save_and_supplements_readback(self):
        analysis_before = (self.library / "packages" / self.asset_id / "analysis.json").read_bytes()
        result = run_entry(["motion", self.asset_id, "--start", "0", "--end", "1",
                            "--sample-fps", "10", "--save",
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["save"]["status"], "saved")
        saved = self.library / payload["save"]["path_relative"]
        self.assertTrue(saved.is_file())
        self.assertNotIn(str(self.library), saved.read_text(encoding="utf-8"))
        self.assertEqual(
            (self.library / "packages" / self.asset_id / "analysis.json").read_bytes(),
            analysis_before)

        listing = run_entry(["supplements", self.asset_id,
                             "--library-root", str(self.library)], home=self.home)
        self.assertEqual(listing.returncode, 0, listing.stderr)
        readback = json.loads(listing.stdout)
        self.assertEqual(readback["count"], 1)
        entry = readback["supplements"][0]
        self.assertEqual(entry["kind"], "engine_measurement")
        self.assertEqual(entry["relationship"]["status"], "current")
        self.assertEqual(entry["source"], "engine")

        one = run_entry(["supplements", self.asset_id, "--id", entry["supplement_id"],
                         "--library-root", str(self.library)], home=self.home)
        self.assertEqual(one.returncode, 0, one.stderr)
        self.assertEqual(json.loads(one.stdout)["supplement"]["kind"], "engine_measurement")

    def test_annotate_writes_attributed_record_preserving_input_bytes(self):
        annotation_path = self.write_input(self.annotation())
        input_bytes = annotation_path.read_bytes()
        analysis_before = (self.library / "packages" / self.asset_id / "analysis.json").read_bytes()
        result = run_entry(["annotate", self.asset_id, "--input", str(annotation_path),
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["relationship"]["status"], "current")
        document = json.loads((self.library / payload["path_relative"]).read_text(encoding="utf-8"))
        self.assertEqual(document["source"], "downstream_agent")
        self.assertEqual(document["author"], "entry-test")
        self.assertEqual(document["input"]["assertions"][0]["speaker"], "S1")
        self.assertEqual(annotation_path.read_bytes(), input_bytes)
        self.assertEqual(
            (self.library / "packages" / self.asset_id / "analysis.json").read_bytes(),
            analysis_before)

        listing = run_entry(["supplements", self.asset_id,
                             "--library-root", str(self.library)], home=self.home)
        self.assertEqual(listing.returncode, 0, listing.stderr)
        readback = json.loads(listing.stdout)
        self.assertEqual(readback["count"], 1)
        self.assertEqual(readback["supplements"][0]["kind"], "agent_annotations")
        self.assertEqual(readback["supplements"][0]["assertion_count"], 1)

    def test_malformed_referenced_evidence_paths_reject_before_any_write(self):
        package = self.library / "packages" / self.asset_id
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "outside-sentinel.jpg"
        sentinel.write_bytes(b"outside sentinel unchanged")
        evidence_dir = package / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "link.jpg").symlink_to(sentinel)
        (package / "linked-dir").symlink_to(outside, target_is_directory=True)

        # Establish prior supplement bytes that a rejected write must preserve.
        prior_path = self.write_input(self.annotation(), "prior.json")
        prior = run_entry(["annotate", self.asset_id, "--input", str(prior_path),
                           "--library-root", str(self.library)], home=self.home)
        self.assertEqual(prior.returncode, 0, prior.stderr)
        prior_document = self.library / json.loads(prior.stdout)["path_relative"]
        prior_bytes = prior_document.read_bytes()

        analysis_path = package / "analysis.json"
        cases = [
            ("traversal", "../../../outside-sentinel.jpg"),
            ("absolute", str(sentinel)),
            ("symlink-file", "evidence/link.jpg"),
            ("symlink-ancestor", "linked-dir/outside-sentinel.jpg"),
        ]
        store = AssetStore(self.library)
        for label, relative_path in cases:
            with self.subTest(case=label):
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                analysis["evidence"] = [{
                    "evidence_id": "frame_1", "kind": "image", "timestamp": 0.5,
                    "relative_path": relative_path,
                }]
                store.save_analysis(self.asset_id, analysis)
                analysis_before = analysis_path.read_bytes()
                before = snapshot(self.library)
                input_path = self.write_input(self.annotation(), f"{label}.json")
                result = run_entry(["annotate", self.asset_id, "--input", str(input_path),
                                    "--library-root", str(self.library)], home=self.home)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("错误", result.stderr)
                self.assertEqual(snapshot(self.library), before)
                self.assertEqual(analysis_path.read_bytes(), analysis_before)
                self.assertEqual(sentinel.read_bytes(), b"outside sentinel unchanged")
                self.assertEqual(prior_document.read_bytes(), prior_bytes)

        # A normal package-relative path remains valid even though the image
        # bytes are absent; historical evidence availability is not required.
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis["evidence"] = [{
            "evidence_id": "frame_1", "kind": "image", "timestamp": 0.5,
            "relative_path": "evidence/frame_1.jpg",
        }]
        store.save_analysis(self.asset_id, analysis)
        input_path = self.write_input(self.annotation(), "normal.json")
        result = run_entry(["annotate", self.asset_id, "--input", str(input_path),
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["relationship"]["status"], "current")

    def test_invalid_annotation_and_range_create_no_lock_or_supplement(self):
        bad = self.annotation()
        bad["assertions"][0]["speaker"] = "S9"
        bad_path = self.write_input(bad, "bad.json")
        before = snapshot(self.library)
        result = run_entry(["annotate", self.asset_id, "--input", str(bad_path),
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 2)
        self.assertIn("错误", result.stderr)
        self.assertEqual(snapshot(self.library), before)
        self.assertFalse((self.library / ".locks").exists())

        result = run_entry(["motion", self.asset_id, "--start", "2", "--end", "1",
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "invalid_input")
        self.assertEqual(payload["errors"][0]["code"], "invalid_interval")
        self.assertEqual(snapshot(self.library), before)

    def test_missing_input_and_missing_asset_are_explicit(self):
        before = snapshot(self.library)
        result = run_entry(["annotate", self.asset_id,
                            "--input", str(self.root / "absent.json"),
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 2)
        self.assertIn("注释输入不是普通文件", result.stderr)
        self.assertEqual(snapshot(self.library), before)

        result = run_entry(["motion", "asset_absent", "--start", "0", "--end", "1",
                            "--library-root", str(self.library)], home=self.home)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["errors"][0]["code"], "asset_not_found")
        self.assertEqual(snapshot(self.library), before)

    def test_help_and_missing_library_create_nothing(self):
        for command in ("motion", "annotate", "supplements"):
            with self.subTest(command=command):
                result = run_entry([command, "--help"], home=self.home)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("asset_id", result.stdout)
        self.assertFalse((self.home / "Downloads").exists())
        self.assertFalse((self.home / "cache").exists())

        missing = self.root / "missing library"
        for arguments in (["motion", self.asset_id, "--start", "0", "--end", "1"],
                          ["supplements", self.asset_id],
                          ["annotate", self.asset_id, "--input", str(self.write_input(self.annotation()))]):
            with self.subTest(arguments=arguments[0]):
                result = run_entry([*arguments, "--library-root", str(missing)], home=self.home)
                self.assertEqual(result.returncode, 2)
                self.assertIn("找不到已有资产库", result.stderr)
                self.assertFalse(missing.exists())

        top = run_entry(["--help"], home=self.home)
        self.assertIn("motion", top.stdout)
        self.assertIn("annotate", top.stdout)
        self.assertIn("supplements", top.stdout)

    def test_annotate_and_supplements_do_not_import_analysis_or_media_engine(self):
        import agent_entry
        blocked = {"core", "faster_whisper", "torch", "cv2", "transformers", "numpy",
                   "motion_analyze", "asset_attempt"}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise AssertionError(f"supplement command imported {name}")
                return None

        blocker = Blocker()
        sys.meta_path.insert(0, blocker)
        self.addCleanup(sys.meta_path.remove, blocker)
        for module in list(sys.modules):
            if module.split(".")[0] in blocked:
                self.skipTest(f"test process already imported {module}")

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            code = agent_entry.main(["supplements", self.asset_id,
                                     "--library-root", str(self.library)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["count"], 0)


if __name__ == "__main__":
    unittest.main()
