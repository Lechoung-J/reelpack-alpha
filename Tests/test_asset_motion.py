"""Installed motion adapter over real analytic fixtures.

Measurement mathematics stays in ``motion_analyze``; these tests only cover the
adapter contract: explicit finite ranges, saved-shot cut boundaries, structured
failure, read-only behaviour, portable saved measurements and immutability of
prior supplement documents.  Fixtures are real H.264 files from
``Tests/test_motion_analyze.py`` and require the same optional media toolchain.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "Core"
TESTS = Path(__file__).resolve().parent
for entry in (CORE, TESTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import asset_supplements
import asset_motion
from asset_contract import empty_analysis
from asset_store import AssetStore

import test_motion_analyze as motion_fixtures

HAVE_MEDIA = motion_fixtures.HAVE_MEDIA


def snapshot(root):
    root = Path(root)
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


@unittest.skipUnless(HAVE_MEDIA, "requires opencv, ffmpeg and ffprobe")
class MotionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pan = motion_fixtures.Fixtures.build(
            "adapter_pan_320x240_10fps.mp4",
            motion_fixtures._translation_fixture(320, 240, 10, seconds=3.0,
                                                 vx=0.12, vy=-0.05, seed=11))
        cls.pan_other = motion_fixtures.Fixtures.build(
            "adapter_pan_other_320x240_10fps.mp4",
            motion_fixtures._translation_fixture(320, 240, 10, seconds=3.0,
                                                 vx=-0.08, vy=0.04, seed=12))
        cls.two_shot = motion_fixtures.Fixtures.build(
            "adapter_two_shot_320x240_10fps.mp4",
            motion_fixtures._two_shot_fixture(320, 240, 10, seconds=3.0,
                                              speed=0.18, cut=1.0))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reelpack-motion-adapter-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()

    def library(self, fixture, *, analysis=True, shots=None, name="library"):
        root = self.base / name
        store = AssetStore(root)
        imported = store.import_file(fixture)
        asset_id = imported["asset_id"]
        self.asset_id = asset_id
        self.root = root
        self.store = store
        self.media = Path(imported["media_path"])
        if analysis:
            value = empty_analysis(asset_id, {"kind": "local_file",
                                              "display_name": Path(fixture).name})
            value["status"] = "complete"
            value["video"] = {"duration_seconds": 3.0, "width": 320, "height": 240, "fps": 10.0}
            if shots is not None:
                value["shots"] = {"status": "complete", "backend": "fixture",
                                  "items": shots}
            store.save_analysis(asset_id, value)
        return root, asset_id

    def run_motion(self, root, asset_id, **options):
        options.setdefault("start", 0.0)
        options.setdefault("end", 2.0)
        options.setdefault("sample_fps", 10)
        return asset_motion.run_motion(root, asset_id, **options)

    def test_explicit_range_measures_signed_translation(self):
        root, asset_id = self.library(self.pan, shots=[
            {"shot_id": "shot_1", "start": 0.0, "end": 3.0}])
        result, code = self.run_motion(root, asset_id)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["command"], "motion")
        self.assertEqual(result["asset_scope"]["asset_id"], asset_id)
        self.assertEqual(result["asset_scope"]["package_relative_path"],
                         f"packages/{asset_id}")
        self.assertEqual(result["cuts"]["status"], "applied")
        velocity = result["summary"]["velocity"]
        self.assertAlmostEqual(velocity["median_vx_frame_width_s"], 0.12, delta=0.02)
        self.assertAlmostEqual(velocity["median_vy_frame_height_s"], -0.05, delta=0.02)
        self.assertGreater(velocity["median_speed_px_s"], 10.0)
        self.assertGreater(result["coverage"]["analyzed"]["duration"], 1.5)
        self.assertLessEqual(result["coverage"]["analyzed"]["end"], 2.0)
        self.assertTrue(all(entry["status"] == "valid" for entry in result["pairs"]))

    def test_read_only_measurement_does_not_write_the_library(self):
        root, asset_id = self.library(self.pan, shots=[
            {"shot_id": "shot_1", "start": 0.0, "end": 3.0}])
        before = snapshot(root)
        result, code = self.run_motion(root, asset_id)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(snapshot(root), before)
        self.assertFalse((root / ".locks").exists())
        self.assertFalse((root / "packages" / asset_id / "supplements").exists())

    def test_saved_shot_list_keeps_cut_and_never_crosses_it(self):
        root, asset_id = self.library(self.two_shot, shots=[
            {"shot_id": "shot_1", "start": 0.0, "end": 1.0},
            {"shot_id": "shot_2", "start": 1.0, "end": 2.0}])
        result, code = self.run_motion(root, asset_id)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["cuts"]["status"], "applied")
        self.assertEqual([round(item["start"], 6) for item in result["cuts"]["intervals"]],
                         [0.0, 1.0])
        self.assertEqual([round(item["end"], 6) for item in result["cuts"]["intervals"]],
                         [1.0, 2.0])
        for pair in result["pairs"]:
            crossed = pair["start_time"] < 1.0 < pair["end_time"]
            self.assertFalse(crossed, f"pair crosses the saved cut: {pair['pair_index']}")
        per_shot = [shot["summary"]["velocity"]["median_vx_frame_width_s"]
                    for shot in result["shots"]]
        self.assertGreater(per_shot[0], 0.12)
        self.assertLess(per_shot[1], -0.12)
        self.assertEqual(result["summary"]["shot_count"], 2)

    def test_range_outside_saved_shots_is_measured_without_inventing_cuts(self):
        root, asset_id = self.library(self.pan, shots=[
            {"shot_id": "shot_1", "start": 0.0, "end": 1.0}])
        result, code = self.run_motion(root, asset_id, start=1.5, end=2.5)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["cuts"]["status"], "unavailable")
        self.assertEqual(result["cuts"]["reason"], "no_saved_shot_overlaps_requested_range")
        self.assertGreater(result["summary"]["pair_count"], 0)
        self.assertTrue(all(pair["start_time"] >= 1.5 for pair in result["pairs"]))
        self.assertTrue(all(pair["end_time"] <= 2.5 for pair in result["pairs"]))

    def test_empty_saved_shot_list_is_explicit_and_does_not_silently_claim_cuts(self):
        root, asset_id = self.library(self.pan, shots=[])
        result, code = self.run_motion(root, asset_id)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["cuts"]["status"], "unavailable")
        self.assertEqual(result["cuts"]["reason"], "saved_shot_list_empty")
        self.assertGreater(result["summary"]["valid_pair_count"], 0)

    def test_missing_analysis_allows_read_only_measurement_but_not_save(self):
        root, asset_id = self.library(self.pan, analysis=False)
        before = snapshot(root)
        result, code = self.run_motion(root, asset_id)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["cuts"]["reason"], "analysis_missing")
        self.assertFalse(result["asset_scope"]["analysis_present"])
        self.assertEqual(snapshot(root), before)

        blocked, blocked_code = self.run_motion(root, asset_id, save=True)
        self.assertEqual(blocked_code, 2)
        self.assertEqual(blocked["status"], "invalid_input")
        self.assertEqual(blocked["errors"][0]["code"], "analysis_missing")
        self.assertEqual(snapshot(root), before)

    def test_invalid_ranges_fail_without_writes(self):
        root, asset_id = self.library(self.pan, shots=[])
        before = snapshot(root)
        for start, end in [(-1.0, 2.0), (2.0, 2.0), (2.0, 1.0),
                           (float("nan"), 2.0), (0.0, float("inf"))]:
            with self.subTest(start=start, end=end):
                result, code = self.run_motion(root, asset_id, start=start, end=end)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "invalid_input")
                self.assertEqual(result["errors"][0]["code"], "invalid_interval")
                self.assertEqual(result["cuts"]["status"], "not_evaluated")
        self.assertEqual(snapshot(root), before)
        self.assertFalse((root / ".locks").exists())

    def test_offline_media_fails_with_structured_error_and_no_write(self):
        root, asset_id = self.library(self.pan, shots=[])
        shutil.rmtree(root / "media")
        before = snapshot(root)
        result, code = self.run_motion(root, asset_id, save=True)
        self.assertEqual(code, 4)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["errors"][0]["code"], "media_offline")
        self.assertEqual(snapshot(root), before)
        self.assertFalse((root / "packages" / asset_id / "supplements").exists())

    def test_save_publishes_portable_measurement_and_reads_back_offline(self):
        root, asset_id = self.library(self.two_shot, shots=[
            {"shot_id": "shot_1", "start": 0.0, "end": 1.0},
            {"shot_id": "shot_2", "start": 1.0, "end": 2.0}])
        result, code = self.run_motion(root, asset_id, save=True)
        self.assertEqual(code, 0)
        self.assertEqual(result["save"]["status"], "saved")
        path = root / result["save"]["path_relative"]
        text = path.read_text(encoding="utf-8")
        self.assertNotIn(str(root), text)
        self.assertNotIn(str(self.media), text)
        document = json.loads(text)
        self.assertEqual(document["kind"], "engine_measurement")
        self.assertEqual(document["source"], "engine")
        self.assertEqual(document["measurement"]["status"], "complete")
        self.assertEqual(document["measurement"]["cuts"]["status"], "applied")
        self.assertEqual(document["measurement"]["asset_scope"]["analysis_sha256"],
                         document["provenance"]["source_analysis_sha256"])
        self.assertEqual(document["measurement"]["source"]["media_relative_path"],
                         document["provenance"]["media_relative_path"])

        shutil.rmtree(root / "media")
        readback = asset_supplements.list_supplements(root, asset_id)
        self.assertEqual(readback["media_status"], "offline")
        self.assertEqual(readback["count"], 1)
        self.assertEqual(readback["supplements"][0]["kind"], "engine_measurement")
        self.assertEqual(readback["supplements"][0]["relationship"]["status"], "current")
        show = asset_supplements.list_supplements(
            root, asset_id, result["save"]["supplement_id"])
        self.assertEqual(show["supplement"]["measurement"]["status"], "complete")

    def test_failed_save_preserves_the_prior_document_bytes(self):
        root, asset_id = self.library(self.pan, shots=[])
        first, code = self.run_motion(root, asset_id, save=True)
        self.assertEqual(code, 0)
        first_path = root / first["save"]["path_relative"]
        first_bytes = first_path.read_bytes()
        before_listing = asset_supplements.list_supplements(root, asset_id)
        self.assertEqual(before_listing["count"], 1)

        # Swap the archived media for a different valid clip.  Measurement
        # succeeds, but the measured digest no longer matches asset metadata, so
        # the save must be refused and the earlier immutable document retained.
        shutil.copyfile(self.pan_other, self.media)
        second, second_code = self.run_motion(root, asset_id, save=True)
        self.assertEqual(second["status"], "complete")
        self.assertNotEqual(second_code, 0)
        self.assertEqual(second["save"]["status"], "failed")
        self.assertIn("不一致", second["save"]["reason"])
        self.assertEqual(first_path.read_bytes(), first_bytes)
        after_listing = asset_supplements.list_supplements(root, asset_id)
        self.assertEqual(after_listing["count"], 1)
        self.assertEqual(after_listing["supplements"][0]["supplement_id"],
                         first["save"]["supplement_id"])

    def test_nonusable_status_is_not_saved(self):
        root, asset_id = self.library(self.pan, shots=[])
        before = snapshot(root)
        synthetic = {
            "schema": "reelpack.motion-evidence", "schema_version": 1,
            "backend": "test-synthetic", "backend_version": "0",
            "status": "failed", "reason": "synthetic engine failure",
            "source": {"digest_sha256": "0" * 64, "duration_seconds": 3.0},
            "request": {"start": 0.0, "end": 2.0}, "parameters": {},
            "pairs": [], "segments": [], "shots": [], "warnings": [],
        }
        with mock.patch("motion_analyze.analyze_motion", return_value=dict(synthetic)):
            result, code = self.run_motion(root, asset_id, save=True)
        self.assertEqual(code, 4)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["save"]["status"], "failed")
        self.assertIn("只保存可用测量", result["save"]["reason"])
        self.assertFalse((root / "packages" / asset_id / "supplements").exists())

    def test_corrupt_saved_shots_fail_before_measurement(self):
        root, asset_id = self.library(self.pan, shots=[
            {"shot_id": "shot_1", "start": 1.0, "end": 1.0}])
        before = snapshot(root)
        result, code = self.run_motion(root, asset_id)
        self.assertEqual(code, 4)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["errors"][0]["code"], "corrupt_shots")
        self.assertEqual(snapshot(root), before)


if __name__ == "__main__":
    unittest.main()
