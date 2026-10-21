import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Core"))

from asset_attempt import AssetAttempt
from asset_contract import empty_analysis
from asset_store import AssetStore, AssetStoreError


class AssetAttemptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        source = self.base / "video.mp4"
        source.write_bytes(b"test media")
        self.store = AssetStore(self.base / "library")
        self.archive = self.store.import_file(source)
        self.asset_id = self.archive["asset_id"]
        self.canonical = Path(self.archive["package_dir"])

    def tearDown(self):
        self.temp.cleanup()

    def analysis(self, state="in_progress", text="new"):
        value = empty_analysis(self.asset_id, {"kind": "local_file"})
        value["execution"] = {"status": state}
        value["transcript"]["raw_text"] = text
        value["transcript"]["raw_segments"] = [{"start": 0, "end": 1, "text": text}]
        return value

    def old_package(self, legacy=False):
        value = self.analysis("complete", "old")
        value["status"] = "complete"
        if legacy:
            value.pop("execution")
        self.store.save_analysis(self.asset_id, value)
        evidence = self.canonical / "evidence"
        evidence.mkdir()
        (evidence / "old.jpg").write_bytes(b"old pixels")
        asset_path = self.canonical / "asset.json"
        asset = json.loads(asset_path.read_text())
        asset["custom_metadata"] = {"preserve": True}
        asset_path.write_text(json.dumps(asset))
        return self.bytes(self.canonical)

    @staticmethod
    def bytes(directory):
        return {str(p.relative_to(directory)): p.read_bytes()
                for p in directory.rglob("*") if p.is_file()}

    @unittest.skipUnless(sys.platform == "darwin", "native macOS atomic exchange")
    def test_complete_promotion_exchanges_whole_package(self):
        before = self.old_package()
        archive = copy.deepcopy(self.archive)
        with AssetAttempt(self.store, self.archive) as attempt:
            stage = attempt.package_dir
            self.assertEqual(set(self.bytes(stage)), {"asset.json"})
            (stage / "evidence").mkdir()
            (stage / "evidence" / "new.jpg").write_bytes(b"new pixels")
            value = self.analysis("complete")
            value["evidence"] = [{"evidence_id": "new", "relative_path": "evidence/new.jpg"}]
            paths = attempt.save(value, final=True)
            self.assertEqual(paths["package_dir"], str(self.canonical))
            self.assertEqual(self.bytes(attempt.previous_dir), before)
            self.assertFalse((self.canonical / "evidence" / "old.jpg").exists())
            self.assertEqual((self.canonical / "evidence" / "new.jpg").read_bytes(), b"new pixels")
            self.assertTrue(self.store.get_asset(self.asset_id)["asset"]["custom_metadata"]["preserve"])
        self.assertEqual(self.archive, archive)
        self.assertEqual(json.loads((attempt.attempt_dir / "attempt.json").read_text())["outcome"], "complete")

    def test_old_package_is_byte_identical_after_failure_including_legacy(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                if not (self.canonical / "evidence").exists():
                    self.old_package(legacy=legacy)
                if legacy:
                    value = self.analysis("complete", "old")
                    value.pop("execution")
                    self.store.save_analysis(self.asset_id, value)
                before = self.bytes(self.canonical)
                with self.assertRaisesRegex(RuntimeError, "pipeline failed"):
                    with AssetAttempt(self.store, self.archive) as attempt:
                        attempt.save(self.analysis())
                        (attempt.package_dir / "failed-frame.jpg").write_bytes(b"new failure evidence")
                        attempt.failure(self.analysis())
                        raise RuntimeError("pipeline failed")
                self.assertEqual(self.bytes(self.canonical), before)
                self.assertTrue((attempt.package_dir / "failed-frame.jpg").exists())
                manifest = json.loads((attempt.attempt_dir / "attempt.json").read_text())
                self.assertEqual(manifest["outcome"], "failed")
                self.assertEqual(manifest["error"], "pipeline failed")

    @unittest.skipUnless(sys.platform == "darwin", "native macOS atomic exchange")
    def test_first_checkpoint_is_queryable_and_failure_remains_visible(self):
        with AssetAttempt(self.store, self.archive) as attempt:
            value = self.analysis("complete")
            paths = attempt.save_analysis(self.asset_id, value)
            self.assertEqual(paths["package_dir"], str(attempt.package_dir))
            query = self.store.query_time(self.asset_id, 0, 1)
            self.assertEqual(query["analysis"]["transcript"]["raw_segments"][0]["text"], "new")
            self.assertEqual(self.store.get_asset(self.asset_id)["analysis"]["execution"]["status"], "in_progress")
            self.assertEqual(value["execution"]["status"], "complete")
            saved = json.loads((attempt.package_dir / "analysis.json").read_text())
            self.assertEqual(saved["execution"]["status"], "complete")
            self.assertEqual(saved["execution"]["attempt_id"], attempt.attempt_dir.name)
            attempt.failure(self.analysis())
            self.assertEqual(self.store.get_asset(self.asset_id)["analysis"]["execution"]["status"], "failed")
            self.assertTrue(self.store.list_assets())
        self.assertTrue(attempt.package_dir.exists())

    def test_lock_conflict_and_release(self):
        with AssetAttempt(self.store, self.archive):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with AssetAttempt(self.store, self.archive):
                    self.fail("Concurrent attempt acquired the lock")
        with AssetAttempt(self.store, self.archive):
            pass

    def test_exception_before_promote_preserves_old_bytes(self):
        before = self.old_package()
        with self.assertRaises(KeyboardInterrupt):
            with AssetAttempt(self.store, self.archive) as attempt:
                attempt.save(self.analysis("complete"))
                raise KeyboardInterrupt()
        self.assertEqual(self.bytes(self.canonical), before)
        self.assertTrue((attempt.package_dir / "analysis.json").is_file())
        with AssetAttempt(self.store, self.archive):
            pass

    def test_exchange_failure_preserves_canonical_and_stage(self):
        before = self.old_package()
        with self.assertRaises(OSError):
            with AssetAttempt(self.store, self.archive) as attempt:
                attempt.save(self.analysis("complete"))
                with patch.object(attempt, "_exchange", side_effect=OSError("injected before swap")):
                    attempt.promote()
        self.assertEqual(self.bytes(self.canonical), before)
        self.assertEqual(json.loads((attempt.package_dir / "analysis.json").read_text())["transcript"]["raw_text"], "new")

    def test_corrupt_existing_analysis_fails_without_mutation_and_releases_lock(self):
        self.old_package()
        (self.canonical / "analysis.json").write_text("{bad JSON")
        before = self.bytes(self.canonical)
        for _ in range(2):
            with self.assertRaises(AssetStoreError):
                with AssetAttempt(self.store, self.archive):
                    pass
        self.assertEqual(self.bytes(self.canonical), before)

    def test_symlink_and_traversal_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        archive = dict(self.archive, package_dir=str(outside))
        with self.assertRaises(AssetStoreError):
            AssetAttempt(self.store, archive)
        with self.assertRaises(AssetStoreError):
            AssetAttempt(self.store, dict(self.archive, asset_id="../escape"))
        (self.canonical / "evidence").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(AssetStoreError):
            with AssetAttempt(self.store, self.archive):
                pass
        self.assertEqual(list(outside.iterdir()), [])

    def test_stage_symlink_rejected_without_touching_target(self):
        self.old_package()
        outside = self.base / "outside.json"
        outside.write_bytes(b"untouched")
        with AssetAttempt(self.store, self.archive) as attempt:
            (attempt.package_dir / "analysis.json").symlink_to(outside)
            with self.assertRaises(AssetStoreError):
                attempt.save(self.analysis())
        self.assertEqual(outside.read_bytes(), b"untouched")

    def test_invalid_payload_and_incomplete_promotion_rejected(self):
        before = self.old_package()
        with AssetAttempt(self.store, self.archive) as attempt:
            with self.assertRaises(AssetStoreError):
                attempt.save({"asset_id": self.asset_id})
            value = self.analysis()
            value["evidence"] = [{"evidence_id": "bad", "relative_path": "../../outside"}]
            with self.assertRaises(AssetStoreError):
                attempt.save(value)
            attempt.save(self.analysis())
            with self.assertRaises(AssetStoreError):
                attempt.promote()
        self.assertEqual(self.bytes(self.canonical), before)

    @unittest.skipUnless(sys.platform == "darwin", "native macOS atomic exchange")
    def test_manifest_failure_after_promotion_logs_without_reporting_failure(self):
        self.old_package()
        attempt = AssetAttempt(self.store, self.archive)
        attempt.__enter__()
        attempt.save(self.analysis("complete"), final=True)
        with patch.object(attempt, "_manifest", side_effect=OSError("manifest unavailable")):
            with self.assertLogs("asset_attempt", level="WARNING") as logs:
                self.assertFalse(attempt.__exit__(None, None, None))
        self.assertIn("promotion succeeded", logs.output[0])
        self.assertEqual(self.store.get_asset(self.asset_id)["analysis"]["execution"]["status"], "complete")
        with AssetAttempt(self.store, self.archive):
            pass

    def test_exit_after_failed_enter_is_safe_and_does_not_create_attempt(self):
        with AssetAttempt(self.store, self.archive):
            blocked = AssetAttempt(self.store, self.archive)
            with self.assertRaises(RuntimeError):
                blocked.__enter__()
            self.assertFalse(blocked.__exit__(None, None, None))
            self.assertFalse(blocked.attempt_dir.exists())

    def test_staged_metadata_corruption_is_rejected(self):
        before = self.old_package()
        with AssetAttempt(self.store, self.archive) as attempt:
            metadata = attempt.package_dir / "asset.json"
            metadata.write_text('{"asset_id":"asset_wrong"}')
            with self.assertRaisesRegex(AssetStoreError, "metadata id mismatch"):
                attempt.save(self.analysis())
        self.assertEqual(self.bytes(self.canonical), before)

    def test_non_macos_exchange_fails_safely(self):
        before = self.old_package()
        with AssetAttempt(self.store, self.archive) as attempt:
            attempt.save(self.analysis("complete"))
            with patch("asset_attempt.sys.platform", "linux"):
                with self.assertRaisesRegex(AssetStoreError, "requires macOS"):
                    attempt.promote()
        self.assertEqual(self.bytes(self.canonical), before)


if __name__ == "__main__":
    unittest.main()
