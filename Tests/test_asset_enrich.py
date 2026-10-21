import copy
import json
import sys
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Core'))
from asset_contract import empty_analysis, transcript_segment, validate_analysis
from asset_store import AssetStore
from asset_enrich import enrich, parse_interval, review_times, review

class EnrichTests(unittest.TestCase):
    def fixture(self, td):
        store = AssetStore(Path(td)/'library')
        source = Path(td)/'clip.mp4'
        source.write_bytes(b'fake')
        aid = store.import_file(source)['asset_id']
        analysis = empty_analysis(aid, {'kind': 'local_file'})
        analysis['transcript']['raw_text'] = 'raw'
        analysis['transcript']['corrected_text'] = 'corrected'
        analysis['transcript']['raw_segments'] = [transcript_segment('s', 0, 1, 'hello')]
        analysis['music'] = {'status': 'complete', 'items': ['keep']}
        analysis['boards'] = ['keep']
        analysis['evidence'] = [{'evidence_id': 'e1', 'kind': 'image', 'timestamp': 1, 'relative_path': 'evidence/1.jpg'}]
        store.save_analysis(aid, analysis)
        root = store.root/'packages'/aid
        (root/'evidence').mkdir(); (root/'evidence/1.jpg').write_bytes(b'frame')
        return store, aid, root

    def test_idempotent_and_preserve(self):
        with tempfile.TemporaryDirectory() as td:
            store, aid, root = self.fixture(td)
            before = store.get_asset(aid)['analysis']
            metadata = (root/'asset.json').read_bytes()
            markdown = (root/'analysis.md').read_bytes()
            texts = []
            enrich(store, aid, ocr=lambda _: '字幕', summary=lambda t: texts.append(t) or '摘要')
            after = store.get_asset(aid)['analysis']
            for key in before:
                if key not in ('ocr', 'summary', 'capabilities'):
                    self.assertEqual(before[key], after[key])
            self.assertEqual(texts, ['corrected'])
            data = (root/'analysis.json').read_bytes()
            self.assertEqual(enrich(store, aid, ocr=lambda _: self.fail(), summary=lambda _: self.fail())['changed'], [])
            self.assertEqual(data, (root/'analysis.json').read_bytes())
            self.assertEqual(metadata, (root/'asset.json').read_bytes())
            self.assertEqual(markdown, (root/'analysis.md').read_bytes())

    def test_only_failure_retry(self):
        with tempfile.TemporaryDirectory() as td:
            store, aid, root = self.fixture(td)
            before = store.get_asset(aid)['analysis']
            def fail(_): raise RuntimeError('sandbox only')
            enrich(store, aid, only='ocr', ocr=fail)
            after = store.get_asset(aid)['analysis']
            self.assertEqual(after['ocr']['status'], 'failed')
            self.assertEqual(after['summary'], before['summary'])
            enrich(store, aid, only='ocr', ocr=lambda _: 'ok')
            self.assertEqual(store.get_asset(aid)['analysis']['ocr']['status'], 'complete')
            enrich(store, aid, only='summary', summary=fail)
            self.assertEqual(store.get_asset(aid)['analysis']['summary']['reason'], 'sandbox only')

    def test_empty_frame_and_missing_evidence_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            store, aid, root = self.fixture(td)
            (root/'evidence/1.jpg').rename(root/'evidence/old.jpg')
            with patch('asset_enrich.extract_frame') as extract:
                enrich(store, aid, only='ocr', ocr=lambda _: '')
            extract.assert_called_once()
            after = store.get_asset(aid)['analysis']
            self.assertEqual(after['ocr']['status'], 'complete')
            self.assertEqual(after['ocr']['items'][0]['text'], '')
            self.assertFalse(validate_analysis(after))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe unavailable')
    def test_review_real_audio_and_frames_mock_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td)/'synthetic.mp4'
            subprocess.run(['ffmpeg', '-nostdin', '-n', '-f', 'lavfi', '-i',
                            'color=c=blue:s=320x180:r=10:d=3', '-f', 'lavfi', '-i',
                            'sine=frequency=440:duration=3', '-c:v', 'mpeg4', '-c:a', 'aac',
                            '-shortest', str(source)], check=True, capture_output=True)
            store = AssetStore(Path(td)/'library')
            aid = store.import_file(source)['asset_id']
            store.save_analysis(aid, empty_analysis(aid, {'kind':'local_file'}, {'duration_seconds':3}))
            result = review(store, aid, .5, 1.5, Path(td)/'review', padding=.25, interval=.5,
                            ocr=lambda _: 'mock subtitle')
            self.assertTrue(all(Path(i['path']).is_file() for i in result['items']))
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                    '-of', 'json', result['audio']], check=True, capture_output=True, text=True)
            self.assertAlmostEqual(float(json.loads(probe.stdout)['format']['duration']), 1, places=2)

    def test_interval(self):
        self.assertEqual(parse_interval('56', '61'), (56., 61.))
        self.assertEqual(review_times(56, 61)[0::16], [54.5, 62.5])
        self.assertEqual(review_times(1, 2, interval=.5, frames=2), [0, .5, 1, 1.5, 2, 2.5, 3])
        for a,b in [(2,1),(-1,2),(0,float('nan')),(0,float('inf'))]:
            with self.assertRaises(ValueError): parse_interval(a,b)
        with self.assertRaises(ValueError): review_times(0,2,interval=0)
        with self.assertRaises(ValueError): review_times(0,2,duration=1)

    def test_speaker(self):
        self.assertIsNone(transcript_segment('s',0,1,'hi')['speaker'])
        analysis = empty_analysis('a', {})
        analysis['transcript']['raw_segments']=[dict(start=0,end=1,text='old')]
        self.assertFalse(validate_analysis(analysis))
        analysis['transcript']['raw_segments'][0]['speaker']=4
        self.assertTrue(validate_analysis(analysis))

    def test_review_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            store, aid, root = self.fixture(td)
            before = (root/'analysis.json').read_bytes()
            with patch('asset_enrich.subprocess.run'), patch('asset_enrich.extract_frame'):
                result = review(store,aid,0,1,Path(td)/'review',ocr=lambda _: '候选')
            self.assertTrue(result['candidate_subtitles'])
            self.assertTrue((Path(td)/'review/review.txt').exists())
            self.assertEqual(before,(root/'analysis.json').read_bytes())
            with self.assertRaises(ValueError): review(store,aid,0,1,root/'review')

if __name__ == '__main__': unittest.main()
