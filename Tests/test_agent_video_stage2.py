import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Core"))

from asset_contract import empty_analysis, assert_valid_analysis
from asset_store import AssetStore
from ocr_sampling import merge_observations, sampling_candidates
from shot_detect import detect_shots


class Stage2UnitTests(unittest.TestCase):
    def test_ocr_sampling_is_independent_and_exact_merge_is_conservative(self):
        sampled = sampling_candidates(6, interval=2, change_timestamps=[1.0])
        self.assertTrue(sampled["independent_sampling"])
        self.assertIn(1.0, [x["timestamp"] for x in sampled["candidates"]])
        merged = merge_observations([
            {"start": 0, "end": 1, "text": "API v2", "region": {"x": 0, "y": 0}, "shot_id": "shot_1"},
            {"start": 1.2, "end": 2, "text": "API v2", "region": {"x": 0, "y": 0}, "shot_id": "shot_1"},
            {"start": 3, "end": 4, "text": "API v3", "region": {"x": 0, "y": 0}, "shot_id": "shot_1"},
            {"start": 5, "end": 6, "text": "API v2", "region": {"x": 0, "y": 0}, "shot_id": "shot_2"},
        ])["items"]
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["end"], 2)
        self.assertEqual(merged[1]["text"], "API v3")
        self.assertEqual(merged[2]["shot_id"], "shot_2")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_shot_detection_registers_jpeg_evidence_for_synthetic_cut(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "cuts.avi"
            package = root / "packages" / "asset_test"
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=10:d=1",
                "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=10:d=1",
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]", "-map", "[out]",
                "-c:v", "mjpeg", "-q:v", "3", str(video).replace(".mp4", ".avi"),
            ], check=True, stdout=subprocess.DEVNULL)
            result, evidence = detect_shots(video, package, threshold=0.15,
                                             representative_interval=0.5, backend='custom')
            self.assertIn(result["status"], {"complete", "partial"})
            self.assertGreaterEqual(len(result["items"]), 2)
            self.assertTrue(evidence)
            for item in evidence:
                self.assertEqual(item["kind"], "image")
                self.assertTrue((package / item["relative_path"]).is_file())
                self.assertTrue(item["relative_path"].endswith(".jpg"))
            self.assertTrue(all(item["representative_frame"] in {e["evidence_id"] for e in evidence} for item in result["items"]))

    def test_evidence_query_and_contract(self):
        with tempfile.TemporaryDirectory() as td:
            store = AssetStore(Path(td) / "library")
            source = Path(td) / "x.mp4"
            source.write_bytes(b"video")
            imported = store.import_file(source)
            analysis = empty_analysis(imported["asset_id"], {"kind": "local_file"})
            analysis["shots"] = {"status": "complete", "items": [{"shot_id": "shot_0001", "start": 0, "end": 2, "boundary": {"uncertain": True}, "representative_frame": "frame_1", "evidence_ids": ["frame_1"]}]}
            analysis["evidence"] = [{"evidence_id": "frame_1", "kind": "image", "timestamp": 1, "relative_path": "evidence/frame_1.jpg", "shot_id": "shot_0001", "purpose": "shot_representative"}]
            assert_valid_analysis(analysis)
            store.save_analysis(imported["asset_id"], analysis)
            queried = store.query_time(imported["asset_id"], 0.5, 1.5)
            self.assertEqual(queried["images"][0]["evidence_id"], "frame_1")
            self.assertEqual(queried["analysis"]["shots"]["items"][0]["shot_id"], "shot_0001")


if __name__ == "__main__":
    unittest.main()
