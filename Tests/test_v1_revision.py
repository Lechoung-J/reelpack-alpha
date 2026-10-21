import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from frame_sampling import shot_frame_positions, mark_speech_frames
from asset_store import AssetStore, AssetStoreError
from asset_contract import empty_analysis
from frame_board import frame_board
from asset_cli import main
import core


class RevisionTests(unittest.TestCase):
    def test_short_and_long_sampling(self):
        self.assertEqual(shot_frame_positions(30, 90, 30), [30, 59, 89])
        for duration in [3, 13, 20]:
            positions = shot_frame_positions(0, duration*30, 30, 5)
            self.assertIn((duration*30-1)//2, positions)
            self.assertEqual(positions[-1], duration*30-1)
            self.assertLessEqual(max(b-a for a,b in zip(positions, positions[1:])), 60)
        self.assertEqual(shot_frame_positions(0, 1, 30), [0])

    def test_speech(self):
        frames = [{'kind': 'image', 'timestamp': t} for t in [0, 1, 2]]
        mark_speech_frames(frames, [{'start': 1, 'end': 2, 'text': 'hello'}])
        self.assertEqual([f['keyframe'] for f in frames], [False, True, False])
        self.assertIn('speech_keyframe', frames[1]['purpose'])

    def test_summary_failures(self):
        with patch.object(core, '_ollama_generate', return_value='[校正失败: connection refused]'):
            self.assertEqual(core.generate_summary('text'), '')
        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            self.assertEqual(core.generate_summary('text', use_cloud=True), '')
        with patch.object(core, '_ollama_generate', return_value='valid summary'):
            self.assertEqual(core.generate_summary('text'), 'valid summary')

    def test_store_board_and_cli(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'video.mp4'; source.write_bytes(b'fixture')
            store = AssetStore(root / 'library')
            imported = store.import_file(source); aid = imported['asset_id']
            package = Path(imported['package_dir'])
            (package / 'evidence').mkdir()
            analysis = empty_analysis(aid, {})
            analysis['shots']['items'] = [{'shot_id': 'shot_0001', 'start': 0, 'end': 13}, {'shot_id': 'shot_0002', 'start': 13, 'end': 14}]
            analysis['transcript']['raw_segments'] = [{'start': 1, 'end': 2, 'text': 'hello'}]
            analysis['ocr']['items'] = [{'timestamp': 1, 'text': 'caption'}]
            for i in range(14):
                relative = f'evidence/{i}.jpg'
                Image.new('RGB', (640, 360), 'red').save(package / relative)
                analysis['evidence'].append({'evidence_id': f'f{i}', 'kind': 'image', 'timestamp': i,
                                            'relative_path': relative, 'shot_id': 'shot_0001' if i<13 else 'shot_0002', 'keyframe': i==1})
            store.save_analysis(aid, analysis)
            result = store.query_shot(aid, 'shot_0001')
            self.assertEqual(len(result['images']), 13)
            self.assertEqual(result['analysis']['transcript']['raw_segments'][0]['text'], 'hello')
            self.assertEqual(len(result['analysis']['shots']['items']), 1)
            with self.assertRaises(AssetStoreError): store.query_shot(aid, 'missing')
            store.set_summary(aid, 'Agent summary')
            self.assertEqual(store.get_asset(aid)['analysis']['summary']['text'], 'Agent summary')
            self.assertIn('Agent summary', (package/'analysis.md').read_text())
            for grid in [9, 12]:
                board = frame_board(store, aid, shot='shot_0001', grid=grid)
                self.assertEqual(len(board['pages']), 2)
                self.assertEqual(len(board['frames']), 13)
                with Image.open(board['pages'][0]['absolute_path']) as image:
                    self.assertEqual(image.size, (1152 if grid==9 else 1536, 720))
            detail = frame_board(store, aid, shot='shot_0001', view='detail')
            self.assertEqual(detail['frames'][0]['timestamp'], 1)
            detail = frame_board(store, aid, shot='shot_0001', view='detail', timestamp=10.1)
            self.assertEqual(detail['frames'][0]['timestamp'], 10)
            self.assertEqual(main(['query', aid, '--shot', 'shot_0001', '--library-root', str(store.root)]), 0)
            self.assertEqual(main(['set-summary', aid, '--text', '', '--library-root', str(store.root)]), 0)
            self.assertEqual(store.get_asset(aid)['analysis']['summary']['status'], 'not_run')
            analysis['evidence'][0]['relative_path'] = '../escape.jpg'
            store.save_analysis(aid, analysis)
            with self.assertRaises(AssetStoreError): frame_board(store, aid, shot='shot_0001', view='detail', timestamp=0)

    def test_language_prompt_and_failure_fallback(self):
        self.assertIn("不得翻译", core.CORRECT_PROMPT)
        self.assertIn("原语言", core.SUMMARY_PROMPT)
        with patch.object(core, "_ollama_generate", return_value="[校正失败: unavailable]"):
            self.assertEqual(core.correct_transcript("こんにちは", "Nike", language="ja"), "こんにちは")
        with patch.object(core, "_ollama_generate", return_value="こんにちは。") as generate:
            self.assertEqual(core.correct_transcript("こんにちは", "Nike", language="ja"), "こんにちは。")
            self.assertIn("ja", generate.call_args.args[0])

    def test_transcription_language_is_saved(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            store = AssetStore(Path(td) / "library")
            source = Path(td) / "video.mp4"; source.write_bytes(b"fixture")
            imported = store.import_file(source)
            args = SimpleNamespace(model="large", ocr_backend="macos", vision_model="vision")
            core._save_pipeline_analysis(store, imported, str(source), str(source),
                [{"start": 0, "end": 1, "text": "こんにちは"}], "こんにちは", "こんにちは", "", "", args,
                transcription_info={"language": "ja", "language_probability": 0.985,
                                    "params": {"vad_filter": True, "condition_on_previous_text": False},
                                    "timestamp_bounds": {"duration_seconds": 1, "clipped_segments": 1},
                                    "warnings": ["fixture warning"]})
            analysis = store.get_asset(imported["asset_id"])["analysis"]
            self.assertEqual(analysis["capabilities"]["transcription"]["language"], "ja")
            self.assertEqual(analysis["transcript"]["language_probability"], 0.985)
            self.assertEqual(analysis["capabilities"]["transcription"]["timestamp_bounds"]["clipped_segments"], 1)
            self.assertEqual(analysis["capabilities"]["transcription"]["warnings"], ["fixture warning"])
            self.assertEqual(analysis["capabilities"]["transcription"]["params"],
                             {"vad_filter": True, "condition_on_previous_text": False})

    def test_detail_full_resolution_and_offline_proxy_fallback(self):
        from PIL import Image
        import shutil
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); store = AssetStore(root / "library")
            source = root / "source.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=640x360:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(source)], check=True)
            imported = store.import_file(source); aid = imported["asset_id"]
            package = Path(imported["package_dir"]); (package / "evidence").mkdir()
            Image.new("RGB", (160, 90), "blue").save(package / "evidence/proxy.jpg")
            analysis = empty_analysis(aid, {})
            analysis["shots"]["items"] = [{"shot_id": "shot_1", "start": 0, "end": 1}]
            analysis["evidence"] = [{"evidence_id": "frame_1", "kind": "image", "timestamp": .5, "relative_path": "evidence/proxy.jpg", "shot_id": "shot_1", "keyframe": True}]
            store.save_analysis(aid, analysis)
            detail = frame_board(store, aid, view="detail", shot="shot_1", timestamp=.5, full_res=True)
            self.assertEqual(detail["resolution"], "640x360")
            self.assertEqual(detail["source"], "original")
            Path(imported["media_path"]).unlink()
            fallback = frame_board(store, aid, view="detail", shot="shot_1", timestamp=.5, full_res=True)
            self.assertEqual(fallback["resolution"], "160x90")
            self.assertEqual(fallback["source"], "proxy")


if __name__ == '__main__': unittest.main()
