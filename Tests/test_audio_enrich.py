import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Core'))
from asset_store import AssetStore
from asset_contract import empty_analysis
from asset_enrich import enrich
import core


class AudioEnrichTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        source=self.root/'input.mp4';source.write_bytes(b'synthetic archived bytes')
        self.store=AssetStore(self.root/'library');self.archive=self.store.import_file(source)
        self.asset_id=self.archive['asset_id'];self.folder=Path(self.archive['package_dir'])
        a=empty_analysis(self.asset_id,{'kind':'local_file'})
        a['transcript'].update(status='complete',raw_segments=[{'segment_id':'s1','start':0,'end':1,'text':'preserve me','speaker':None}])
        a['shots'].update(status='complete',items=[{'shot_id':'shot_1','start':0,'end':2,'evidence_ids':['f1'],'representative_frame':'f1'}])
        a['evidence']=[{'evidence_id':'f1','kind':'image','timestamp':.5,'relative_path':'evidence/f1.jpg'}]
        a['music']={'status':'failed','error':'before cache error'}
        (self.folder/'evidence').mkdir();(self.folder/'evidence/f1.jpg').write_bytes(b'original evidence')
        self.store.save_analysis(self.asset_id,a)
        self.before=json.loads((self.folder/'analysis.json').read_text())
        self.asset_before=(self.folder/'asset.json').read_bytes()
        self.music={'status':'complete','backend':'librosa+soundfile+pyloudnorm','model':None,
                    'tempo_bpm':120,'beat_times_seconds':[.5,1.0], 'genre':{'status':'not_run'},'mood':{'status':'not_run'},
                    'cut_alignment':{'status':'measured','cut_count':1,'beat_count':2,'aligned_cut_count':0,'aligned_ratio':0.0,'tolerance_seconds':.15},
                    'audio_board':{'status':'not_run','reason':'synthetic render target'}}

    def assert_protected(self):
        after=json.loads((self.folder/'analysis.json').read_text())
        for field in ('transcript','shots','evidence'):
            self.assertEqual(json.dumps(after[field],ensure_ascii=False),json.dumps(self.before[field],ensure_ascii=False))
        self.assertEqual((self.folder/'evidence/f1.jpg').read_bytes(),b'original evidence')
        self.assertEqual((self.folder/'asset.json').read_bytes(),self.asset_before)
        return after

    def test_audio_only_never_calls_asr_shots_or_frames(self):
        audio=Mock(return_value=copy.deepcopy(self.music))
        with patch.object(core,'transcribe_audio',side_effect=AssertionError('ASR')) as asr, \
             patch.object(core,'detect_shots',side_effect=AssertionError('shots')) as shots, \
             patch.object(core,'extract_frames',side_effect=AssertionError('frames')) as frames:
            result=enrich(self.store,self.asset_id,only='audio',audio=audio)
        for spy in (asr,shots,frames):spy.assert_not_called()
        audio.assert_called_once()
        self.assertEqual(audio.call_args.args[0],self.archive['media_path'])
        self.assertTrue(result['protected_sections_unchanged'])
        after=self.assert_protected();self.assertEqual(after['music']['tempo_bpm'],120)
        self.assertEqual(after['capabilities']['music']['status'],'complete')
        self.assertIn('全混音',(self.folder/'analysis.md').read_text())

    def test_repeated_audio_update_keeps_one_backup_and_no_stacking(self):
        first=Mock(return_value=copy.deepcopy(self.music))
        enrich(self.store,self.asset_id,only='audio',audio=first)
        before=(self.folder/'analysis.json').read_bytes()
        backup=(self.folder/'analysis.pre-audio.json').read_bytes()
        enrich(self.store,self.asset_id,only='audio',audio=first)
        self.assertEqual((self.folder/'analysis.json').read_bytes(),before)
        self.assertEqual((self.folder/'analysis.pre-audio.json').read_bytes(),backup)
        self.assertEqual(len(list(self.folder.glob('analysis.pre-audio*'))),1)
        self.assert_protected()

    def test_offline_media_has_specific_status_without_updating_package(self):
        media=Path(self.archive['media_path'])
        self.assertTrue(self.store.media_dir in media.parents)
        media.unlink()
        before=(self.folder/'analysis.json').read_bytes();audio=Mock()
        result=enrich(self.store,self.asset_id,only='audio',audio=audio)
        self.assertEqual(result['status'],'media_offline');self.assertEqual(result['changed'],[])
        for key in ('reason','impact','remedy'):self.assertTrue(result[key])
        audio.assert_not_called();self.assertEqual((self.folder/'analysis.json').read_bytes(),before)

    def test_redraw_from_json_does_not_call_audio_or_need_media(self):
        audio=Mock(side_effect=AssertionError('decode'))
        with patch('audio_service.ensure_audio_board',return_value={'status':'complete','path':'evidence/audio_board.png'}) as draw:
            result=enrich(self.store,self.asset_id,only='audio',audio=audio,redraw_board=True)
        audio.assert_not_called();draw.assert_called_once()
        self.assertEqual(result['results']['music']['audio_board']['status'],'complete')
        self.assert_protected()

    def test_enrichment_cannot_write_during_an_active_attempt(self):
        from asset_attempt import AssetAttempt
        before=(self.folder/'analysis.json').read_bytes();audio=Mock()
        with AssetAttempt(self.store,self.archive):
            with self.assertRaisesRegex(RuntimeError,'活动写入'):
                enrich(self.store,self.asset_id,only='audio',audio=audio)
        audio.assert_not_called();self.assertEqual((self.folder/'analysis.json').read_bytes(),before)

    def test_in_progress_checkpoint_is_not_overwritten_by_enrichment(self):
        path=self.folder/'analysis.json'
        data=json.loads(path.read_text());data['execution']={'status':'in_progress','stage':'audio'}
        path.write_text(json.dumps(data));before=path.read_bytes();audio=Mock()
        with self.assertRaisesRegex(RuntimeError,'检查点'):
            enrich(self.store,self.asset_id,only='audio',audio=audio)
        audio.assert_not_called();self.assertEqual(path.read_bytes(),before)

    def test_audio_failure_is_explicit_and_preserves_video_results(self):
        result=enrich(self.store,self.asset_id,only='audio',audio=Mock(return_value={'status':'failed','error':'synthetic failure'}))
        self.assertEqual(result['status'],'failed')
        after=self.assert_protected()
        for key in ('reason','impact','remedy'):self.assertTrue(after['capabilities']['music'][key])


if __name__=='__main__':unittest.main()
