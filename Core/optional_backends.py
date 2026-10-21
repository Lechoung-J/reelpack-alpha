"""Standard-library-only adapters for explicitly selected, external CPU backends."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent

PANNS_COST = {'approximate_checkpoint_mb': 327, 'minimum_checkpoint_bytes': 300000000,
              'automatic_download': False, 'device': 'cpu', 'optional': True}
PANNS_COST_NOTICE = 'Optional PANNs Cnn14 weights are approximately 327 MB; provide them explicitly. No automatic download.'


def _runtime_environment():
    environment = dict(os.environ)
    try:
        from runtime_environment import prepare_runtime_environment
    except ImportError:
        return environment
    return prepare_runtime_environment(environment, create=environment.get('REELPACK_READONLY_CHECK') != '1')


def _with_cost(result, backend):
    if backend == 'panns':
        result.setdefault('cost', dict(PANNS_COST))
        result.setdefault('cost_notice', PANNS_COST_NOTICE)
        if result.get('status') != 'complete':
            result.setdefault('impact', 'Semantic candidates unavailable; physical tempo, beats, alignment and timbre are retained.')
            result.setdefault('remedy', 'Provide local REELPACK_PANNS_CHECKPOINT and REELPACK_PANNS_LABELS, and an explicit music Python; retry without downloading weights automatically.')
    return result


def invoke(backend, arguments, runner=None):
    base = dict(status='not_run', backend=backend, model='Cnn14' if backend == 'panns' else 'RapidOCR',
                params={}, experimental=True, items=[], errors=[])
    env = 'music' if backend == 'panns' else 'ocr'
    try:
        environment = _runtime_environment()
        python = Path(environment.get(f'REELPACK_{env.upper()}_PYTHON', str(ROOT / f'.venv-{env}/bin/python'))).expanduser()
        if not python.is_file():
            return _with_cost(dict(base, reason=f'optional python unavailable: {python}'), backend)
        run = (runner or subprocess.run)([str(python), str(ROOT / 'Core/optional_backend_worker.py'), backend, *arguments],
                                        env=environment, capture_output=True, text=True, timeout=1800, check=True)
        result = json.loads(run.stdout.strip().splitlines()[-1])
        if not isinstance(result, dict) or result.get('status') not in ('complete', 'partial', 'failed', 'not_run'):
            raise ValueError('invalid backend response')
        return _with_cost(dict(base, **result), backend)
    except Exception as exc:
        return _with_cost(dict(base, status='failed', reason=f'{type(exc).__name__}: {getattr(exc, "stderr", "") or exc}'), backend)


def recognize_frames(frames, evidence_ids=None, runner=None):
    with tempfile.TemporaryDirectory(prefix='reelpack-rapidocr-') as td:
        manifest = []
        for i, (timestamp, image) in enumerate(frames):
            path = Path(td) / f'{i}.jpg'
            path.write_bytes(image)
            manifest.append(dict(path=str(path), timestamp=timestamp,
                                 evidence_ids=(evidence_ids or {}).get(timestamp, [])))
        path = Path(td) / 'manifest.json'
        path.write_text(json.dumps(manifest))
        return invoke('rapidocr', [str(path)], runner)


def analyze_music(media, descriptors, runner=None, genre_threshold=.05, genre_top_k=3, shots=None):
    args = [str(media), str(genre_threshold), str(genre_top_k)]
    if shots is not None:
        args.append(json.dumps(shots, ensure_ascii=False))
    semantic = invoke('panns', args, runner)
    # Preserve physical descriptors and their own status independently of tagging failure.
    result = dict(descriptors)
    result['descriptors'] = {k: v for k, v in descriptors.items() if k not in ('genre', 'mood')}
    # The top-level status/backend describe the physical analysis that produced BPM,
    # beats and loudness. Optional tags have their own provenance and status.
    provenance = ('status', 'backend', 'model', 'params', 'reason', 'error', 'errors', 'experimental', 'impact', 'remedy', 'cost', 'cost_notice')
    result['tagging'] = {key: semantic[key] for key in provenance if key in semantic}
    result.update({k: v for k, v in semantic.items() if k not in (*provenance, 'items')})
    result.setdefault('status', 'unknown')
    result.setdefault('backend', None)
    result.setdefault('model', None)
    result.setdefault('params', {})
    for key in ('genre', 'mood', 'events'):
        result.setdefault(key, dict(status=semantic['status'], backend='panns', model='Cnn14',
                                    experimental=True, params={}, items=[], reason=semantic.get('reason')))
        if semantic['status'] != 'complete':
            result[key] = dict(status=semantic['status'], backend='panns', model='Cnn14',
                               experimental=True, params={}, items=[], reason=semantic.get('reason'),
                                impact=semantic.get('impact'), remedy=semantic.get('remedy'),
                                cost=semantic.get('cost'), cost_notice=semantic.get('cost_notice'))
    genre_empty = not result.get('genre', {}).get('items')
    mood_empty = not result.get('mood', {}).get('items')
    tags = result.get('top_tags', [])
    music_tag = next((x for x in tags if x.get('label', '').lower() == 'music'), None)
    if genre_empty and mood_empty and music_tag:
        result['unclassified'] = True
        result['fallback_hint'] = (f"检测到音乐（Music {music_tag['score']:.2f}），但曲风/情绪不在可分类集合；top_tags："
            + '、'.join(f"{x['label']} {x['score']:.2f}" for x in tags[:5])
            + '。可能是管弦/氛围类，请结合 top_tags、描述与画面推断。')
    return result
