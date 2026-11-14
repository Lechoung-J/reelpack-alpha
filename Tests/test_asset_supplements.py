"""Supplement persistence, validation and stale readback (no model, no network).

The fixtures are ordinary local packages created with the real AssetStore.  No
motion/media decode is required for annotation persistence, so these tests also
prove that readback works with offline media and after moving the library.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "Core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import asset_supplements
from asset_contract import empty_analysis
from asset_store import AssetStore, sha256_file


class SupplementFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reelpack-supplements-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.library = self.root / "library"
        self.store = AssetStore(self.library)
        self.source = self.root / "clip.mp4"
        self.source.write_bytes(b"supplement fixture media bytes")
        imported = self.store.import_file(self.source)
        self.asset_id = imported["asset_id"]
        self.package = Path(imported["package_dir"])
        analysis = empty_analysis(self.asset_id, {"kind": "local_file", "display_name": "clip.mp4"})
        analysis["status"] = "complete"
        analysis["video"] = {"duration_seconds": 10.0, "width": 640, "height": 480, "fps": 10.0}
        analysis["speakers"] = {
            "status": "complete", "backend": "fixture", "experimental": True,
            "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "one"},
                {"start": 2.0, "end": 3.0, "speaker": "S2", "text": "two"},
            ],
        }
        analysis["evidence"] = [
            {"evidence_id": "frame_1", "kind": "image", "timestamp": 0.5,
             "relative_path": "evidence/frame_1.jpg"},
        ]
        analysis["shots"] = {"status": "complete", "backend": "fixture", "items": [
            {"shot_id": "shot_1", "start": 0.0, "end": 5.0},
            {"shot_id": "shot_2", "start": 5.0, "end": 10.0},
        ]}
        self.store.save_analysis(self.asset_id, analysis)

    def annotation(self, **overrides):
        value = {
            "schema": "reelpack.agent-annotations",
            "schema_version": 1,
            "asset_id": self.asset_id,
            "author": "fixture-harness",
            "assertions": [
                {
                    "assertion_id": "a_voice",
                    "kind": "voice_person",
                    "status": "tentative",
                    "person_id": "P1",
                    "speaker": "S1",
                    "value": None,
                    "reasoning": "S1 overlaps the left-seated speaker in the sampled frames.",
                    "uncertainty": "No lip-sync measurement; the cut changes framing.",
                    "confidence": 0.6,
                    "evidence_refs": [
                        {"kind": "package_evidence", "evidence_id": "frame_1"},
                        {"kind": "media_range", "start": 0.0, "end": 2.0},
                    ],
                },
                {
                    "assertion_id": "a_name",
                    "kind": "person_identity",
                    "status": "supported",
                    "person_id": "P1",
                    "value": "Sample Name",
                    "reasoning": "Attributed Agent-supplied source text, not engine truth.",
                    "uncertainty": "No known gap beyond the attributed source label.",
                    "confidence": 0.8,
                    "evidence_refs": [
                        {"kind": "source_text", "source_label": "attributed title",
                         "locator": "source description", "excerpt": "Sample Name"},
                    ],
                },
                {
                    "assertion_id": "a_role",
                    "kind": "person_role",
                    "status": "unknown",
                    "person_id": None,
                    "value": None,
                    "reasoning": "The analysed interval does not establish a host/guest role.",
                    "uncertainty": "Role is unknown; no inference is made.",
                    "confidence": None,
                    "evidence_refs": [
                        {"kind": "source_text", "source_label": "attributed description",
                         "locator": "source page", "excerpt": "conversation"},
                    ],
                },
            ],
        }
        value.update(overrides)
        return value

    def snapshot(self):
        return {str(path.relative_to(self.library)): path.read_bytes()
                for path in self.library.rglob("*") if path.is_file()}

    def save(self, payload=None):
        return asset_supplements.save_annotations(self.library, self.asset_id,
                                                  payload or self.annotation())


class AnnotationRoundTripTests(SupplementFixture):
    def test_all_three_kinds_round_trip_and_survive_move_and_offline_media(self):
        before_analysis = (self.package / "analysis.json").read_bytes()
        result = self.save()
        self.assertEqual(result["relationship"]["status"], "current")
        path = self.library / result["path_relative"]
        self.assertTrue(path.is_file())
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["source"], "downstream_agent")
        self.assertEqual(document["kind"], "agent_annotations")
        self.assertEqual(document["provenance"]["path_base"], "library_root")
        self.assertNotIn(str(self.library), path.read_text(encoding="utf-8"))
        self.assertEqual([item["kind"] for item in document["input"]["assertions"]],
                         ["voice_person", "person_identity", "person_role"])
        self.assertIsNone(document["input"]["assertions"][0]["value"])
        self.assertIsNone(document["input"]["assertions"][2]["value"])

        moved = self.root / "moved library"
        shutil.move(str(self.library), str(moved))
        shutil.rmtree(moved / "media")
        listing = asset_supplements.list_supplements(moved, self.asset_id)
        self.assertEqual(listing["count"], 1)
        self.assertEqual(listing["media_status"], "offline")
        entry = listing["supplements"][0]
        self.assertEqual(entry["relationship"]["status"], "current")
        readback = asset_supplements.list_supplements(
            moved, self.asset_id, document["supplement_id"])
        self.assertEqual(readback["supplement"]["input"], document["input"])
        self.assertEqual(readback["media_status"], "offline")
        self.assertEqual((Path(moved) / "packages" / self.asset_id / "analysis.json"
                          ).read_bytes(), before_analysis)

    def test_unreferenced_or_text_only_paths_are_never_resolved(self):
        # A malformed path on evidence the annotation does not reference must not
        # impose a universal "every evidence image must be safe/exist" rule.
        analysis = json.loads((self.package / "analysis.json").read_text(encoding="utf-8"))
        analysis["evidence"].append(
            {"evidence_id": "unreferenced", "kind": "image", "timestamp": 1.0,
             "relative_path": "../../../outside.jpg"})
        self.store.save_analysis(self.asset_id, analysis)

        payload = self.annotation()
        payload["assertions"] = [payload["assertions"][1], payload["assertions"][2]]
        result = self.save(payload)
        document = json.loads((self.library / result["path_relative"]).read_text(encoding="utf-8"))
        self.assertEqual([item["kind"] for item in document["input"]["assertions"]],
                         ["person_identity", "person_role"])
        self.assertTrue(all(
            ref["kind"] == "source_text"
            for item in document["input"]["assertions"] for ref in item["evidence_refs"]))

        # The referenced frame_1 path is package-relative but its image is absent;
        # structure-only validation must accept it (offline/missing bytes are fine).
        self.assertEqual(self.save(self.annotation())["relationship"]["status"], "current")

    def test_unknown_and_ambiguous_are_preserved_without_inventing_names(self):
        payload = self.annotation()
        payload["assertions"][1]["status"] = "ambiguous"
        payload["assertions"][1]["value"] = None
        payload["assertions"][1]["person_id"] = None
        result = self.save(payload)
        document = json.loads((self.library / result["path_relative"]).read_text(encoding="utf-8"))
        assertion = document["input"]["assertions"][1]
        self.assertEqual(assertion["status"], "ambiguous")
        self.assertIsNone(assertion["value"])
        self.assertIsNone(assertion["person_id"])

    def test_supersedes_keeps_earlier_bytes(self):
        first = self.save()
        first_path = self.library / first["path_relative"]
        first_bytes = first_path.read_bytes()
        payload = self.annotation(supersedes=first["supplement_id"])
        payload["assertions"] = [dict(payload["assertions"][1], status="tentative")]
        payload["assertions"][0]["assertion_id"] = "a_name_2"
        second = self.save(payload)
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual(first["supplement_id"], self.show(first["supplement_id"])
                         ["supplement"]["supplement_id"])
        document = self.show(second["supplement_id"])["supplement"]
        self.assertEqual(document["supersedes"], first["supplement_id"])
        self.assertEqual(document["input"]["supersedes"], first["supplement_id"])

    def show(self, supplement_id):
        return asset_supplements.list_supplements(self.library, self.asset_id, supplement_id)

    def test_source_text_is_attributed_text_not_a_file_read(self):
        secret = self.root / "unrelated.txt"
        secret.write_text("do not read me", encoding="utf-8")
        payload = self.annotation()
        payload["assertions"][1]["evidence_refs"] = [{
            "kind": "source_text", "source_label": "attributed title",
            "locator": str(secret), "excerpt": "Sample Name is an attribution, not a read",
        }]
        result = self.save(payload)
        self.assertEqual(secret.read_text(encoding="utf-8"), "do not read me")
        document = self.show(result["supplement_id"])["supplement"]
        self.assertEqual(document["input"]["assertions"][1]["evidence_refs"][0]["locator"],
                         str(secret))

    def test_engine_measurement_is_separate_and_portable(self):
        analysis = json.loads((self.package / "analysis.json").read_text(encoding="utf-8"))
        asset = json.loads((self.package / "asset.json").read_text(encoding="utf-8"))
        media_relative = asset["media"]["relative_path"]
        media_path = (self.library / media_relative).resolve()
        measurement = {
            "schema": "reelpack.motion-evidence",
            "schema_version": 1,
            "backend": "opencv-lk-ransac-similarity-v1",
            "backend_version": "1.0.0",
            "status": "complete",
            "source": {"path": str(media_path), "resolved_path": str(media_path),
                       "digest_sha256": asset["content"]["sha256"],
                       "duration_seconds": 10.0},
            "request": {"start": 0.0, "end": 1.0},
            "parameters": {"sample_fps": 8.0},
            "pairs": [],
            "warnings": [{"code": "runtime_path", "message": f"opened {media_path}"}],
        }
        digest = hashlib.sha256((self.package / "analysis.json").read_bytes()).hexdigest()
        result = asset_supplements.save_engine_measurement(
            self.library, self.asset_id, measurement, expected_analysis_sha256=digest)
        path = self.library / result["path_relative"]
        text = path.read_text(encoding="utf-8")
        self.assertNotIn(str(self.library), text)
        self.assertNotIn(str(media_path), text)
        document = json.loads(text)
        self.assertEqual(document["kind"], "engine_measurement")
        self.assertEqual(document["source"], "engine")
        self.assertEqual(document["measurement"]["source"]["media_relative_path"], media_relative)
        self.assertEqual(document["measurement"]["source"]["path_base"], "library_root")
        self.assertEqual(document["measurement"]["warnings"][0]["message"],
                         f"opened {media_relative}")

        annotation = self.save()
        listing = asset_supplements.list_supplements(self.library, self.asset_id)
        kinds = {entry["supplement_id"]: entry["kind"] for entry in listing["supplements"]}
        self.assertEqual(kinds[result["supplement_id"]], "engine_measurement")
        self.assertEqual(kinds[annotation["supplement_id"]], "agent_annotations")


class AnnotationRejectionTests(SupplementFixture):
    def assert_rejected(self, payload, pattern, *, asset_id=None):
        before = self.snapshot()
        lock_dir = self.library / ".locks"
        lock_before = lock_dir.exists()
        with self.assertRaisesRegex(asset_supplements.SupplementError, pattern):
            asset_supplements.save_annotations(self.library, asset_id or self.asset_id, payload)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(lock_dir.exists(), lock_before)

    def test_invalid_annotations_fail_before_writing_or_locking(self):
        cases = []
        missing_author = self.annotation()
        missing_author.pop("author")
        cases.append((missing_author, "缺少字段.*author"))
        bad_schema = self.annotation(schema="other")
        cases.append((bad_schema, "schema 必须是"))
        unknown_speaker = self.annotation()
        unknown_speaker["assertions"][0]["speaker"] = "S9"
        cases.append((unknown_speaker, "不是源分析中已知的说话人"))
        unknown_evidence = self.annotation()
        unknown_evidence["assertions"][0]["evidence_refs"][0]["evidence_id"] = "absent"
        cases.append((unknown_evidence, "不存在的 evidence_id"))
        out_of_duration = self.annotation()
        out_of_duration["assertions"][0]["evidence_refs"].append(
            {"kind": "media_range", "start": 9.0, "end": 11.0})
        cases.append((out_of_duration, "超出可用视频时长"))
        reversed_range = self.annotation()
        reversed_range["assertions"][0]["evidence_refs"].append(
            {"kind": "media_range", "start": 2.0, "end": 2.0})
        cases.append((reversed_range, "必须大于 start"))
        nan_confidence = self.annotation()
        nan_confidence["assertions"][0]["confidence"] = float("nan")
        cases.append((nan_confidence, "confidence 必须是"))
        duplicate = self.annotation()
        duplicate["assertions"][1]["assertion_id"] = "a_voice"
        cases.append((duplicate, "重复"))
        extra_field = self.annotation()
        extra_field["assertions"][0]["unexpected"] = True
        cases.append((extra_field, "未知字段"))
        speaker_on_name = self.annotation()
        speaker_on_name["assertions"][1]["speaker"] = "S1"
        cases.append((speaker_on_name, "未知字段.*speaker"))
        voice_value = self.annotation()
        voice_value["assertions"][0]["value"] = "Alice"
        cases.append((voice_value, "voice_person 时必须是 null"))
        empty_refs = self.annotation()
        empty_refs["assertions"][0]["evidence_refs"] = []
        cases.append((empty_refs, "非空数组"))
        malformed_locator = self.annotation()
        malformed_locator["assertions"][1]["evidence_refs"] = [{"kind": "source_text",
                                                                 "source_label": "x"}]
        cases.append((malformed_locator, "缺少字段"))
        for payload, pattern in cases:
            with self.subTest(pattern=pattern):
                self.assert_rejected(payload, pattern)

    def test_referenced_evidence_paths_fail_before_writing_or_locking(self):
        outside = self.root / "outside-dir"
        outside.mkdir()
        sentinel = outside / "outside-sentinel.jpg"
        sentinel.write_bytes(b"outside sentinel unchanged")
        evidence_dir = self.package / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "link.jpg").symlink_to(sentinel)
        (self.package / "linked-dir").symlink_to(outside, target_is_directory=True)
        cases = [
            ("parent traversal", "../../../outside-sentinel.jpg", "跳出资产包"),
            ("absolute", str(sentinel), "绝对路径"),
            ("symlink file", "evidence/link.jpg", "符号链接"),
            ("symlink ancestor", "linked-dir/outside-sentinel.jpg", "符号链接"),
        ]
        analysis_path = self.package / "analysis.json"
        for label, relative_path, pattern in cases:
            with self.subTest(case=label):
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                analysis["evidence"][0]["relative_path"] = relative_path
                self.store.save_analysis(self.asset_id, analysis)
                analysis_before = analysis_path.read_bytes()
                self.assert_rejected(self.annotation(), pattern)
                self.assertEqual(analysis_path.read_bytes(), analysis_before)
                self.assertEqual(sentinel.read_bytes(), b"outside sentinel unchanged")
                self.assertFalse((self.package / "supplements").exists())

    def test_malformed_top_level_and_cross_asset_fail(self):
        with self.assertRaisesRegex(asset_supplements.SupplementError, "无效 asset_id"):
            asset_supplements.save_annotations(self.library, "../escape", self.annotation())
        other = self.annotation(asset_id="asset_other")
        self.assert_rejected(other, "asset_id 与目标资产不一致")
        with self.assertRaisesRegex(asset_supplements.SupplementError, "必须是 object"):
            asset_supplements.save_annotations(self.library, self.asset_id, [])

    def test_cross_asset_supersedes_rejected(self):
        first = self.save()
        other_source = self.root / "other.mp4"
        other_source.write_bytes(b"different media bytes for another asset")
        other = self.store.import_file(other_source)
        analysis = empty_analysis(other["asset_id"], {"kind": "local_file"})
        analysis["status"] = "complete"
        analysis["video"] = {"duration_seconds": 5.0}
        analysis["speakers"] = {"status": "complete", "segments": [
            {"start": 0, "end": 1, "speaker": "S1", "text": "x"}]}
        analysis["evidence"] = [{"evidence_id": "frame_1", "timestamp": 0.5}]
        self.store.save_analysis(other["asset_id"], analysis)
        payload = {
            "schema": "reelpack.agent-annotations", "schema_version": 1,
            "asset_id": other["asset_id"], "author": "fixture", "supersedes": first["supplement_id"],
            "assertions": [{
                "assertion_id": "a", "kind": "person_role", "status": "unknown",
                "person_id": None, "value": None, "reasoning": "unknown",
                "uncertainty": "no known gap", "confidence": None,
                "evidence_refs": [{"kind": "package_evidence", "evidence_id": "frame_1"}],
            }],
        }
        before = self.snapshot()
        with self.assertRaisesRegex(asset_supplements.SupplementError,
                                    "补充目录|补充文档不存在|跨资产"):
            asset_supplements.save_annotations(self.library, other["asset_id"], payload)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((Path(other["package_dir"]) / "supplements").exists())

    def test_symlink_supplement_directory_and_target_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.package / "supplements").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(asset_supplements.SupplementError, "符号链接"):
            self.save()
        self.assertEqual(list(outside.iterdir()), [])

    def test_existing_target_symlink_is_not_followed(self):
        directory = asset_supplements.supplements_dir(self.library, self.asset_id, create=True)
        outside = self.root / "outside.json"
        outside.write_bytes(b"untouched")
        with mock.patch.object(asset_supplements, "generate_supplement_id",
                               return_value="sup_an_fixed_id"):
            (directory / "sup_an_fixed_id.json").symlink_to(outside)
            with self.assertRaisesRegex(asset_supplements.SupplementError, "符号链接目标"):
                self.save()
        self.assertEqual(outside.read_bytes(), b"untouched")

    def test_lock_contention_is_explicit(self):
        with asset_supplements.asset_lock(self.library, self.asset_id):
            with self.assertRaisesRegex(asset_supplements.SupplementError, "占用"):
                self.save()
        self.assertEqual(asset_supplements.list_supplements(
            self.library, self.asset_id)["count"], 0)


class StaleRelationTests(SupplementFixture):
    def test_changed_analysis_marks_old_annotation_stale_without_resolving_refs(self):
        result = self.save()
        path = self.library / result["path_relative"]
        original = path.read_bytes()
        analysis = json.loads((self.package / "analysis.json").read_text(encoding="utf-8"))
        analysis["summary"] = {"status": "complete", "text": "changed analysis"}
        self.store.save_analysis(self.asset_id, analysis)

        self.assertEqual(path.read_bytes(), original)
        listing = asset_supplements.list_supplements(self.library, self.asset_id)
        entry = listing["supplements"][0]
        self.assertEqual(entry["relationship"]["status"], "stale")
        self.assertEqual(entry["relationship"]["evidence_resolution"], "not_resolved")
        readback = asset_supplements.list_supplements(
            self.library, self.asset_id, result["supplement_id"])
        self.assertEqual(readback["relationship"]["status"], "stale")
        self.assertEqual(readback["warnings"][0]["code"], "stale_source_analysis")
        self.assertNotIn("resolved_evidence", readback)
        # The recorded references are returned verbatim; they are not rewritten
        # to point at the newer analysis.
        self.assertEqual(readback["supplement"]["input"]["assertions"][0]["evidence_refs"][0],
                         {"kind": "package_evidence", "evidence_id": "frame_1"})

    def test_stale_readback_does_not_revalidate_new_analysis_evidence_paths(self):
        result = self.save()
        document = json.loads((self.library / result["path_relative"]).read_text(encoding="utf-8"))
        analysis = json.loads((self.package / "analysis.json").read_text(encoding="utf-8"))
        analysis["summary"] = {"status": "complete", "text": "reanalysed"}
        analysis["evidence"][0]["relative_path"] = "../../../outside-sentinel.jpg"
        self.store.save_analysis(self.asset_id, analysis)

        listing = asset_supplements.list_supplements(self.library, self.asset_id)
        self.assertEqual(listing["supplements"][0]["relationship"]["status"], "stale")
        readback = asset_supplements.list_supplements(
            self.library, self.asset_id, result["supplement_id"])
        self.assertEqual(readback["relationship"]["evidence_resolution"], "not_resolved")
        self.assertEqual(readback["supplement"]["input"], document["input"])
        self.assertEqual(readback["supplement"]["input"]["assertions"][0]["evidence_refs"][0],
                         {"kind": "package_evidence", "evidence_id": "frame_1"})

    def test_missing_current_analysis_is_unknown(self):
        result = self.save()
        (self.package / "analysis.json").unlink()
        listing = asset_supplements.list_supplements(self.library, self.asset_id)
        self.assertEqual(listing["supplements"][0]["relationship"]["status"], "unknown")
        show = asset_supplements.list_supplements(
            self.library, self.asset_id, result["supplement_id"])
        self.assertEqual(show["relationship"]["status"], "unknown")

    def test_corrupt_entry_is_warned_not_silently_resolved(self):
        self.save()
        directory = self.package / "supplements"
        (directory / "sup_an_broken.json").write_text("{bad json", encoding="utf-8")
        listing = asset_supplements.list_supplements(self.library, self.asset_id)
        self.assertEqual(listing["count"], 1)
        self.assertEqual(listing["warnings"][0]["file"], "sup_an_broken.json")
        with self.assertRaisesRegex(asset_supplements.SupplementError, "无法读取|损坏"):
            asset_supplements.load_supplement(self.library, self.asset_id, "sup_an_broken")


if __name__ == "__main__":
    unittest.main()
