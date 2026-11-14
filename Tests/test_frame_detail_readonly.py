"""Real detail extraction preserves stored analysis and supplement provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Core'))
from asset_contract import empty_analysis
from asset_store import AssetStore
from asset_supplements import list_supplements, save_annotations
from frame_board import frame_board


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for real frame extraction')
class DetailReadOnlyTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temporary = tempfile.TemporaryDirectory(prefix='reelpack detail ')
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / 'local video.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=s=640x360:r=5:d=2', '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', '-y', str(self.source)], check=True)
        self.source_digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.store = AssetStore(root / 'library')
        imported = self.store.import_file(self.source)
        self.asset_id = imported['asset_id']
        self.media = Path(imported['media_path'])
        self.package = Path(imported['package_dir'])
        (self.package / 'evidence').mkdir()
        Image.new('RGB', (160, 90), 'blue').save(self.package / 'evidence/proxy.jpg')
        analysis = empty_analysis(self.asset_id, {}, {'duration_seconds': 2})
        analysis['shots']['items'] = [{'shot_id': 'shot_1', 'start': 0, 'end': 2}]
        analysis['evidence'] = [{'evidence_id': 'frame_1', 'kind': 'image', 'timestamp': .5,
                                 'relative_path': 'evidence/proxy.jpg', 'shot_id': 'shot_1',
                                 'keyframe': True}]
        self.store.save_analysis(self.asset_id, analysis)
        save_annotations(self.store.root, self.asset_id, {
            'schema': 'reelpack.agent-annotations', 'schema_version': 1,
            'asset_id': self.asset_id, 'author': 'fixture-agent',
            'assertions': [{'assertion_id': 'role-unknown', 'kind': 'person_role',
                'status': 'unknown', 'person_id': None, 'value': None,
                'reasoning': 'The frame does not establish a role.',
                'uncertainty': 'Role remains unknown.', 'confidence': None,
                'evidence_refs': [{'kind': 'media_range', 'start': 0, 'end': 1}]}]})
        self.protected = [self.package / name for name in ('analysis.json', 'analysis.md', 'asset.json')]
        self.protected += list((self.package / 'supplements').glob('*.json'))
        self.before = {path: path.read_bytes() for path in self.protected}

    def assert_preserved(self):
        self.assertEqual({path: path.read_bytes() for path in self.protected}, self.before)
        raw = json.loads((self.package / 'analysis.json').read_text())
        self.assertNotIn('speakers', raw)
        self.assertNotIn('detail_frames', raw)
        listing = list_supplements(self.store.root, self.asset_id)
        self.assertEqual(listing['count'], 1)
        self.assertEqual(listing['supplements'][0]['relationship']['status'], 'current')
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_digest)

    def detail(self, full_res):
        result = frame_board(self.store, self.asset_id, view='detail', shot='shot_1',
                             timestamp=.5, full_res=full_res)
        path = Path(result['absolute_path'])
        self.assertTrue(path.is_file())
        self.assertEqual(result['frames'][0]['timestamp'], .5)
        self.assertEqual(result['frames'][0]['source'], result['source'])
        self.assertEqual(result['frames'][0]['resolution'], result['resolution'])
        self.assert_preserved()
        return result, path

    def test_proxy_and_original_detail_preserve_package_and_current_supplement(self):
        proxy, proxy_path = self.detail(False)
        self.assertEqual(proxy['source'], 'proxy')
        self.assertEqual(proxy['resolution'], '160x90')
        self.assertEqual(proxy_path.parent.name, 'evidence')
        original, detail_path = self.detail(True)
        self.assertEqual(original['source'], 'original')
        self.assertEqual(original['resolution'], '640x360')
        self.assertEqual(detail_path.parent, self.package / 'boards')
        self.assertGreater(detail_path.stat().st_size, 0)

    def test_offline_media_detail_falls_back_without_persisting_migrations(self):
        self.media.unlink()
        fallback, path = self.detail(True)
        self.assertEqual(fallback['source'], 'proxy')
        self.assertEqual(fallback['resolution'], '160x90')
        self.assertEqual(path.parent.name, 'evidence')

    def test_decoder_failure_detail_falls_back_without_analysis_mutation(self):
        self.media.write_bytes(b'not a decodable video')
        fallback, path = self.detail(True)
        self.assertEqual(fallback['source'], 'proxy')
        self.assertEqual(fallback['resolution'], '160x90')
        self.assertEqual(path.parent.name, 'evidence')
        self.assertEqual(list((self.package / 'boards').glob('*.jpg')), [])


if __name__ == '__main__':
    unittest.main()
