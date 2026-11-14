"""Supplement history survives whole-package AssetAttempt promotion.

Promotion exchanges the package directory, so these tests exercise the adapter
copy that keeps already-published supplement bytes.  Old documents must keep
their recorded analysis digest and read back as ``stale`` after reanalysis; they
are never rewritten to look current.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Core"))

import asset_supplements
from asset_attempt import AssetAttempt
from asset_contract import empty_analysis
from asset_store import AssetStore


class AssetAttemptSupplementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reelpack-attempt-supplements-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        source = self.base / "video.mp4"
        source.write_bytes(b"attempt supplement fixture media")
        self.store = AssetStore(self.base / "library")
        self.archive = self.store.import_file(source)
        self.asset_id = self.archive["asset_id"]
        self.canonical = Path(self.archive["package_dir"])
        self.root = self.store.root

    def analysis(self, text="old", state="complete"):
        value = empty_analysis(self.asset_id, {"kind": "local_file"})
        value["status"] = "complete"
        value["execution"] = {"status": state}
        value["video"] = {"duration_seconds": 3.0, "width": 320, "height": 240, "fps": 10.0}
        value["speakers"] = {"status": "complete", "backend": "fixture", "experimental": True,
                             "segments": [{"start": 0.0, "end": 1.0, "speaker": "S1",
                                           "text": text}]}
        value["evidence"] = [{"evidence_id": "frame_1", "kind": "image",
                              "timestamp": 0.5, "relative_path": "evidence/frame_1.jpg"}]
        value["transcript"]["raw_text"] = text
        value["transcript"]["raw_segments"] = [{"start": 0.0, "end": 1.0, "text": text}]
        return value

    def annotation(self):
        return {
            "schema": "reelpack.agent-annotations",
            "schema_version": 1,
            "asset_id": self.asset_id,
            "author": "attempt-fixture",
            "assertions": [{
                "assertion_id": "a_voice", "kind": "voice_person", "status": "tentative",
                "person_id": "P1", "speaker": "S1", "value": None,
                "reasoning": "S1 is active while the visible figure is tracked.",
                "uncertainty": "No lip-sync measurement.",
                "confidence": 0.5,
                "evidence_refs": [
                    {"kind": "package_evidence", "evidence_id": "frame_1"},
                    {"kind": "media_range", "start": 0.0, "end": 1.0},
                ],
            }],
        }

    def seed_supplement(self):
        self.store.save_analysis(self.asset_id, self.analysis("old"))
        evidence = self.canonical / "evidence"
        evidence.mkdir()
        (evidence / "old.jpg").write_bytes(b"old pixels")
        saved = asset_supplements.save_annotations(self.root, self.asset_id, self.annotation())
        path = self.root / saved["path_relative"]
        return saved, path, path.read_bytes()

    @staticmethod
    def files(directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in Path(directory).rglob("*") if path.is_file()}

    @unittest.skipUnless(sys.platform == "darwin", "native macOS atomic exchange")
    def test_promotion_keeps_immutable_supplement_history_but_drops_old_evidence(self):
        saved, path, original = self.seed_supplement()
        with AssetAttempt(self.store, self.archive) as attempt:
            stage = attempt.package_dir
            staged_supplement = stage / "supplements" / path.name
            self.assertTrue(staged_supplement.is_file())
            self.assertEqual(staged_supplement.read_bytes(), original)
            self.assertFalse((stage / "evidence" / "old.jpg").exists())

            value = self.analysis("new")
            attempt.save(value, final=True)

            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((self.canonical / "evidence" / "old.jpg").exists())
            self.assertEqual((self.canonical / "supplements" / path.name).read_bytes(), original)
            self.assertEqual(
                (attempt.previous_dir / "supplements" / path.name).read_bytes(), original)
            self.assertTrue((attempt.previous_dir / "analysis.json").is_file())

        listing = asset_supplements.list_supplements(self.root, self.asset_id)
        self.assertEqual(listing["count"], 1)
        entry = listing["supplements"][0]
        self.assertEqual(entry["supplement_id"], saved["supplement_id"])
        self.assertEqual(entry["relationship"]["status"], "stale")
        self.assertEqual(entry["relationship"]["evidence_resolution"], "not_resolved")
        show = asset_supplements.list_supplements(
            self.root, self.asset_id, saved["supplement_id"])
        self.assertEqual(show["supplement"]["provenance"]["source_analysis_sha256"],
                         entry["recorded_source_analysis_sha256"])
        self.assertNotEqual(entry["recorded_source_analysis_sha256"],
                            listing["recorded_analysis_sha256"])

    def test_failed_attempt_preserves_supplement_bytes_in_canonical_package(self):
        saved, path, original = self.seed_supplement()
        before = self.files(self.canonical)
        with self.assertRaisesRegex(RuntimeError, "pipeline failed"):
            with AssetAttempt(self.store, self.archive) as attempt:
                self.assertEqual(
                    (attempt.package_dir / "supplements" / path.name).read_bytes(), original)
                attempt.save(self.analysis("new", state="in_progress"))
                attempt.failure(self.analysis("new", state="failed"))
                raise RuntimeError("pipeline failed")
        self.assertEqual(self.files(self.canonical), before)
        self.assertTrue((attempt.package_dir / "supplements" / path.name).is_file())
        manifest = json.loads((attempt.attempt_dir / "attempt.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["outcome"], "failed")

    def test_hidden_publication_residue_is_not_promoted(self):
        saved, path, original = self.seed_supplement()
        hidden = path.parent / ".sup_an_interrupted.tmp"
        hidden.write_bytes(b"private temp")
        with AssetAttempt(self.store, self.archive) as attempt:
            self.assertFalse((attempt.package_dir / "supplements" / hidden.name).exists())
            self.assertEqual(
                (attempt.package_dir / "supplements" / path.name).read_bytes(), original)
        self.assertEqual(hidden.read_bytes(), b"private temp")

    def test_symlinked_supplement_directory_rejected_without_touching_target(self):
        self.store.save_analysis(self.asset_id, self.analysis("old"))
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "keep.json").write_bytes(b"outside bytes")
        (self.canonical / "supplements").symlink_to(outside, target_is_directory=True)
        before = self.files(self.canonical)
        with self.assertRaisesRegex(Exception, "Symlink rejected|符号链接"):
            with AssetAttempt(self.store, self.archive):
                pass
        self.assertEqual(self.files(self.canonical), before)
        self.assertEqual((outside / "keep.json").read_bytes(), b"outside bytes")

    def test_supplement_writes_use_the_same_asset_lock_as_attempts(self):
        saved, path, original = self.seed_supplement()
        with AssetAttempt(self.store, self.archive) as attempt:
            with self.assertRaisesRegex(asset_supplements.SupplementError, "占用"):
                asset_supplements.save_annotations(self.root, self.asset_id, self.annotation())
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(asset_supplements.list_supplements(
                self.root, self.asset_id)["count"], 1)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
