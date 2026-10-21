"""Skill command compatibility without media analysis, networking or asset writes."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
import agent_entry


class AgentEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.library = self.root / 'library'
        self.package = self.library / 'packages/asset_example'
        self.package.mkdir(parents=True)
        self.analysis = {
            'status': 'partial',
            'transcript': {'raw_text': 'A local reference film', 'raw_segments': [
                {'segment_id': 'seg_1', 'start': 1.0, 'end': 2.0, 'text': 'A local reference film'}]},
            'ocr': {'reading_text': '字幕 KEYWORD', 'items': [
                {'ocr_id': 'ocr_1', 'start': 3.0, 'end': 3.0, 'text': '字幕 KEYWORD'}]},
            'summary': {'text': 'Saved summary'}, 'shots': {'items': [{'shot_id': 'shot_1'}]},
        }
        (self.package / 'analysis.json').write_text(json.dumps(self.analysis))
        (self.package / 'asset.json').write_text(json.dumps({'source': {'display_name': 'sample.mp4'}}))

    def test_query_reads_saved_timed_text_without_network_or_engine(self):
        before = {p: p.read_bytes() for p in self.library.rglob('*') if p.is_file()}
        with patch.object(agent_entry.subprocess, 'call', side_effect=AssertionError('process launched')):
            transcript = agent_entry.query_library(self.library, 'REFERENCE')
            ocr = agent_entry.query_library(self.library, 'keyword')
        self.assertEqual(transcript['count'], 1)
        self.assertEqual(transcript['results'][0]['transcript_matches'][0]['start'], 1.0)
        self.assertEqual(ocr['results'][0]['ocr_matches'][0]['text'], '字幕 KEYWORD')
        self.assertFalse(ocr['network_used'])
        self.assertFalse(ocr['analysis_run'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.library.rglob('*') if p.is_file()})

    def test_asset_id_and_empty_listing_work(self):
        self.assertEqual(agent_entry.query_library(self.library, 'asset_example')['count'], 1)
        self.assertEqual(agent_entry.query_library(self.library, '')['count'], 1)
        self.assertEqual(agent_entry.query_library(self.library, 'absent term')['count'], 0)

    def test_query_output_directory_matches_analysis_library_layout(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = agent_entry.main(['query', '--output-dir', str(self.root), 'keyword'])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(Path(payload['library_root']), self.library.resolve())
        self.assertEqual(payload['results'][0]['asset_id'], 'asset_example')

    def test_missing_library_is_not_created(self):
        missing = self.root / 'does-not-exist'
        with self.assertRaises(FileNotFoundError):
            agent_entry.query_library(missing, 'text')
        self.assertFalse(missing.exists())

    def test_corrupt_package_is_reported_without_mutation(self):
        broken = self.library / 'packages/asset_broken'
        broken.mkdir()
        file = broken / 'analysis.json'
        file.write_text('broken json')
        result = agent_entry.query_library(self.library, '')
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['warnings'][0]['asset_id'], 'asset_broken')
        self.assertEqual(file.read_text(), 'broken json')

    def test_analysis_arguments_forward_unchanged_to_native_cli(self):
        executable = Path('/synthetic/Reelpack.app/Contents/MacOS/reelpack-cli')
        arguments = ['/local/video.mp4', '--output-dir', '/output with spaces']
        with patch.object(agent_entry, 'native_cli', return_value=executable), \
             patch.object(agent_entry.subprocess, 'call', return_value=0) as launch:
            self.assertEqual(agent_entry.main(arguments), 0)
        self.assertEqual(launch.call_args.args[0], [str(executable), *arguments])


if __name__ == '__main__':
    unittest.main()
