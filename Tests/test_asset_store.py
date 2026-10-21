import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Core"))

from asset_cli import main as asset_cli_main
from asset_contract import assert_valid_analysis, empty_analysis, validate_analysis
from asset_store import AssetStore, AssetStoreError, sha256_file


class AssetStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "library"
        self.source = Path(self.temp.name) / "sample.mp4"
        self.source.write_bytes(b"same bytes, different names are the same exact asset")
        self.store = AssetStore(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_import_is_stable_and_deduplicates_exact_bytes(self):
        first = self.store.import_file(self.source)
        self.assertTrue(first["is_new"])
        other = Path(self.temp.name) / "renamed.mov"
        other.write_bytes(self.source.read_bytes())
        second = self.store.import_file(other)
        self.assertFalse(second["is_new"])
        self.assertEqual(first["asset_id"], second["asset_id"])
        self.assertEqual(len(self.store.list_assets()), 1)
        self.assertEqual(sha256_file(self.source), self.store.get_asset(first["asset_id"])["asset"]["content"]["sha256"])
        analysis = empty_analysis(first["asset_id"], {"kind": "local_file"})
        analysis["status"] = "complete"
        self.store.save_analysis(first["asset_id"], analysis)
        cached = self.store.import_file(other)
        self.assertTrue(cached["analysis_complete"])

    def test_different_bytes_are_not_deduplicated(self):
        first = self.store.import_file(self.source)
        other = Path(self.temp.name) / "transcoded.mp4"
        other.write_bytes(self.source.read_bytes() + b"\ntranscode")
        second = self.store.import_file(other)
        self.assertNotEqual(first["asset_id"], second["asset_id"])
        self.assertEqual(len(self.store.list_assets()), 2)

    def test_structured_analysis_and_markdown_survive_restart_and_media_offline(self):
        imported = self.store.import_file(self.source)
        analysis = empty_analysis(
            imported["asset_id"],
            {"kind": "local_file", "display_name": self.source.name},
            {"duration_seconds": 30.0, "width": 1920, "height": 1080, "fps": 30.0},
        )
        analysis["status"] = "partial"
        analysis["transcript"].update({
            "status": "complete",
            "raw_segments": [{"segment_id": "seg_1", "start": 1.0, "end": 4.0, "text": "API v2"}],
            "raw_text": "API v2",
            "corrected_text": "API v2",
        })
        analysis["ocr"] = {
            "status": "complete",
            "sampling": {"strategy": "independent", "interval_seconds": 1.0},
            "items": [{"ocr_id": "ocr_1", "start": 2.0, "end": 3.0, "text": "API v2", "region": {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.1}, "evidence_ids": ["ev_1"]}],
            "reading_text": "API v2",
        }
        analysis["evidence"] = [{"evidence_id": "ev_1", "kind": "frame", "timestamp": 2.5, "relative_path": "evidence/ev_1.jpg", "purpose": "ocr"}]
        self.store.save_analysis(imported["asset_id"], analysis)

        restarted = AssetStore(self.root)
        asset_id = imported["asset_id"]
        queried = restarted.query_time(asset_id, 2.0, 3.0)
        self.assertEqual(queried["analysis"]["transcript"]["raw_segments"][0]["text"], "API v2")
        self.assertEqual(queried["analysis"]["ocr"]["items"][0]["text"], "API v2")
        self.assertEqual(queried["analysis"]["evidence"][0]["evidence_id"], "ev_1")
        self.assertIn("API v2", (self.root / "packages" / asset_id / "analysis.md").read_text())

        (self.root / "media" / "originals" / asset_id / self.source.name).unlink()
        offline = restarted.get_asset(asset_id)
        self.assertEqual(offline["media"]["status"], "offline")
        self.assertEqual(offline["analysis"]["transcript"]["raw_segments"][0]["text"], "API v2")

    def test_query_cli_accepts_library_option_after_subcommand(self):
        imported = self.store.import_file(self.source)
        analysis = empty_analysis(imported["asset_id"], {"kind": "local_file"})
        analysis["transcript"]["raw_segments"] = [{"start": 1, "end": 2, "text": "CLI query"}]
        self.store.save_analysis(imported["asset_id"], analysis)
        self.assertEqual(asset_cli_main(["query", imported["asset_id"], "--start", "1", "--end", "2", "--library-root", str(self.root)]), 0)

    def test_invalid_time_range_and_contract_are_rejected(self):
        imported = self.store.import_file(self.source)
        with self.assertRaises(AssetStoreError):
            self.store.query_time(imported["asset_id"], 5, 1)
        analysis = empty_analysis(imported["asset_id"], {"kind": "local_file"})
        analysis["shots"]["items"] = [{"shot_id": "shot_1", "start": 0, "end": 1}]
        assert_valid_analysis(analysis)
        bad = dict(analysis)
        bad.pop("evidence")
        self.assertTrue(validate_analysis(bad))

    def test_save_analysis_rejects_incomplete_contract_and_unknown_evidence(self):
        imported = self.store.import_file(self.source)
        analysis = empty_analysis(imported["asset_id"], {"kind": "local_file"})
        analysis["ocr"]["items"] = [{"ocr_id": "ocr_1", "timestamp": 1, "text": "x", "evidence_ids": ["missing"]}]
        with self.assertRaises(AssetStoreError):
            self.store.save_analysis(imported["asset_id"], analysis)
        analysis["ocr"]["items"] = []
        analysis["asset_id"] = "asset_wrong"
        with self.assertRaises(AssetStoreError):
            self.store.save_analysis(imported["asset_id"], analysis)

    def test_images_and_evidence_are_queryable_without_media(self):
        imported = self.store.import_file(self.source)
        analysis = empty_analysis(imported["asset_id"], {"kind": "local_file"})
        analysis["evidence"] = [{"evidence_id": "frame_1", "kind": "image", "timestamp": 2, "relative_path": "evidence/frame_1.jpg", "purpose": "shot representative"}]
        analysis["shots"]["items"] = [{"shot_id": "shot_1", "start": 1, "end": 3, "evidence_ids": ["frame_1"]}]
        self.store.save_analysis(imported["asset_id"], analysis)
        result = self.store.query_time(imported["asset_id"], 2, 2)
        self.assertEqual(result["images"][0]["evidence_id"], "frame_1")
        self.assertEqual(result["analysis"]["shots"]["items"][0]["shot_id"], "shot_1")


if __name__ == "__main__":
    unittest.main()
