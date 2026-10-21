"""Shared standard-library orchestration for main and incremental audio analysis."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

from audio_alignment import align_cuts_to_beats
from audio_board import (IMAGE_PATH, METRICS_PATH, deterministic_json, make_board_payload,
                         payload_sha256, render_from_json, validate_payload)

ROOT = Path(__file__).resolve().parent.parent
PHYSICAL_TIMEOUT_SECONDS = 900
RENDER_TIMEOUT_SECONDS = 60


def _runtime_environment():
    """Both workers inherit the same explicitly prepared cache/temp roots."""
    environment = dict(os.environ)
    try:
        from runtime_environment import prepare_runtime_environment
    except ImportError:
        return environment
    return prepare_runtime_environment(environment, create=environment.get('REELPACK_READONLY_CHECK') != '1')


def _audio_python(environment):
    return Path(environment.get('REELPACK_AUDIO_PYTHON', str(ROOT / '.venv-audio/bin/python'))).expanduser()


def _emit(progress, stage, percent, message):
    if progress is not None:
        try:
            progress(stage, percent, message)
        except Exception:
            # Reporting must not discard measurements or abort the main pipeline.
            pass


@contextmanager
def _waiting(progress, stage, percent, message, interval=15):
    stopped = threading.Event()
    started = time.monotonic()
    def heartbeat():
        while not stopped.wait(interval):
            _emit(progress, stage, percent,
                  f'{message}，已等待 {int(time.monotonic() - started)} s；尚未返回结果，不按视频时长推算完成时间')
    thread = threading.Thread(target=heartbeat, daemon=True)
    if progress is not None: thread.start()
    try:
        yield
    finally:
        stopped.set()
        if progress is not None: thread.join(timeout=1)


def _semantic_not_run(name):
    return {'status': 'not_run', 'backend': None, 'model': None, 'params': {}, 'items': [],
            'reason': f'{name} semantic backend was not explicitly selected',
            'impact': f'Physical audio does not establish {name} labels.',
            'remedy': 'Explicitly select PANNs with locally provided weights and labels; no automatic download.'}


def _physical_unavailable(status, reason):
    return {'status': status, 'backend': 'librosa+soundfile+pyloudnorm', 'model': None, 'params': {},
            'experimental': True, 'reason': reason, 'error': reason,
            'impact': 'Physical audio measurements and cut/beat alignment are unavailable.',
            'remedy': 'Check local media, ffmpeg and REELPACK_AUDIO_PYTHON; repair the audio environment and retry.',
            'tempo_bpm': None, 'beat_times_seconds': [], 'beat_count': 0, 'onset_times_seconds': [],
            'genre': _semantic_not_run('genre'), 'mood': _semantic_not_run('mood'),
            'timbre': {'status': 'failed' if status == 'failed' else 'not_run', 'backend': 'librosa_spectral', 'model': 'spectral-descriptors-v1',
                       'params': {}, 'experimental': True, 'scope': 'whole_audio_mix', 'result': None,
                       'reason': reason, 'impact': 'No independent spectral timbre description was measured.',
                       'remedy': 'Repair the physical audio environment and rerun.'}}



def _annotate_states(music):
    if music.get('status') in ('failed', 'not_run'):
        music.setdefault('reason', music.get('error') or 'physical audio measurement did not run')
        music.setdefault('impact', 'Physical audio measurements and cut/beat alignment are unavailable.')
        music.setdefault('remedy', 'Check local media, ffmpeg and explicit audio interpreter; repair locally and retry.')
    for name in ('genre', 'mood', 'events', 'timbre', 'loudness', 'beats', 'tagging'):
        state = music.get(name)
        if not isinstance(state, dict) or state.get('status') not in ('failed', 'not_run', 'not_measured'):
            continue
        state.setdefault('reason', music.get('error') or f'{name} was not measured by the selected backend')
        state.setdefault('impact', f'No {name} conclusion is established; other measured audio fields retain their own states.')
        state.setdefault('remedy', 'Inspect the local backend/dependencies and explicitly rerun this measure; do not infer a missing result.')
    return music


def _last_json(stdout):
    lines = stdout.strip().splitlines()
    if not lines:
        raise ValueError('worker returned no JSON')
    result = json.loads(lines[-1])
    if not isinstance(result, dict) or result.get('status') not in ('complete', 'partial', 'failed', 'not_run'):
        raise ValueError('invalid worker response')
    return result


def _physical_analysis(media, progress):
    try:
        environment = _runtime_environment()
        python = _audio_python(environment)
        if not python.is_file():
            return _physical_unavailable('not_run', f'audio python unavailable: {python}')
        with _waiting(progress, 'audio_physical', 36, '基础音频测量仍在运行'):
            run = subprocess.run([str(python), str(ROOT / 'Core/audio_analyze.py'), str(media)],
                                 env=environment, capture_output=True, text=True,
                                 timeout=PHYSICAL_TIMEOUT_SECONDS, check=True)
        for line in (run.stderr or '').splitlines():
            try:
                event = json.loads(line).get('audio_progress')
                if isinstance(event, dict):
                    _emit(progress, event['stage'], event['percent'], event['message'])
            except (ValueError, TypeError, KeyError, AttributeError):
                pass
        return _last_json(run.stdout)
    except Exception as exc:
        detail = getattr(exc, 'stderr', None) or str(exc)
        if isinstance(detail, bytes):
            detail = detail.decode('utf-8', errors='replace')
        return _physical_unavailable('failed', f'{type(exc).__name__}: {detail[-2000:]}')


def _cut_times(shots):
    if isinstance(shots, dict):
        shots = shots.get('items', [])
    values = []
    for shot in shots or []:
        try:
            value = float(shot['start']) if shot.get('start') is not None else None
        except (TypeError, ValueError, AttributeError, KeyError):
            continue
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def analyze_audio(media, shots, package_dir, backend='librosa', genre_threshold=.05,
                  genre_top_k=3, progress=None):
    """Return a music dict even on physical/tagging/renderer failure.

    Cuts follow the legacy shots.start list, including 0. package_dir=None measures
    without writing a board. The main and incremental callers share this contract.
    """
    if backend == 'panns':
        checkpoint = Path(os.environ.get('REELPACK_PANNS_CHECKPOINT', str(Path.home()/'panns_data/Cnn14_mAP=0.431.pth'))).expanduser()
        labels = Path(os.environ.get('REELPACK_PANNS_LABELS', str(Path.home()/'panns_data/class_labels_indices.csv'))).expanduser()
        try:
            ready = checkpoint.is_file() and checkpoint.stat().st_size >= 300000000 and labels.is_file()
        except OSError:
            ready = False
        if not ready:
            _emit(progress, 'audio_tagging', 31,
                  '可选曲风/情绪模型尚未准备：权重约 327 MB，下载时间受网络影响；不会自动下载。先保留基础测量，准备模型前请让用户决定。')
    _emit(progress, 'audio_decode', 32, '准备音频环境并解码本地全混音')
    if backend not in ('librosa', 'panns'):
        music = _physical_unavailable('failed', f'unsupported audio backend: {backend}')
    else:
        music = _physical_analysis(media, progress)
    music.setdefault('genre', _semantic_not_run('genre'))
    music.setdefault('mood', _semantic_not_run('mood'))
    music.setdefault('tempo_bpm', None)
    beats = music.get('beat_times_seconds') or []
    if not isinstance(beats, (list, tuple)) or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in beats):
        beats = []
        music['beats'] = {'status': 'not_measured', 'reason': 'invalid beat times returned by audio worker'}
        if music.get('status') == 'complete':
            music['status'] = 'partial'
    music['beat_times_seconds'] = list(beats)
    music['beat_count'] = len(beats)
    music.setdefault('timbre', _physical_unavailable('not_run' if music.get('status') == 'not_run' else 'failed',
                         music.get('reason') or music.get('error') or 'audio worker returned no timbre')['timbre'])
    music['experimental'] = True
    cuts = _cut_times(shots)
    _emit(progress, 'audio_physical', 42, '按 shots.start 统计最近拍点偏移（包括媒体起点）')
    try:
        alignment_beats = music['beat_times_seconds'] if music.get('status') in ('complete', 'partial') else []
        music['cut_alignment'] = align_cuts_to_beats(cuts, alignment_beats, .15)
        if music.get('status') in ('failed', 'not_run'):
            music['cut_alignment']['reason'] = 'physical audio measurement ' + music['status'] + '; alignment not measured'
    except (TypeError, ValueError) as exc:
        music['cut_alignment'] = align_cuts_to_beats(cuts, [], .15)
        music['cut_alignment']['reason'] = f'invalid beat measurements: {exc}'
    duration = music.get('duration_seconds')
    if duration is None:
        data = music.get('board_data')
        duration = data.get('duration_seconds') if isinstance(data, dict) else None
    music['cut_alignment']['duration_seconds'] = duration
    if isinstance(music.get('board_data'), dict) and music['board_data'].get('status') == 'complete':
        music['board_data'] = dict(music['board_data'], cut_times_seconds=cuts)
    if backend == 'panns':
        _emit(progress, 'audio_tagging', 44, '检查显式提供的 PANNs 权重与标签；不会自动下载')
        try:
            from optional_backends import analyze_music
            with _waiting(progress, 'audio_tagging', 44, '已显式选择的本地标签计算仍在运行；不下载权重'):
                music = analyze_music(media, music, genre_threshold=genre_threshold,
                                      genre_top_k=genre_top_k, shots=shots)
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            music['tagging'] = {'status': 'failed', 'backend': 'panns', 'model': 'Cnn14',
                                'params': {}, 'reason': reason,
                                'impact': 'Semantic candidates unavailable; physical measures are preserved.',
                                'remedy': 'Repair explicitly selected local PANNs environment and retry.'}
            for name in ('genre', 'mood', 'events'):
                music[name] = dict(_semantic_not_run(name), status='failed', reason=reason)
    else:
        _emit(progress, 'audio_tagging', 44, '默认不运行 genre/mood 语义模型；保留独立 timbre 描述')
    _emit(progress, 'audio_board', 48, '从存盘聚合测量 JSON 生成四面板证据图')
    music['audio_board'] = ensure_audio_board(music, package_dir)
    _annotate_states(music)
    _emit(progress, 'audio_complete', 50, f"音频测量完成：physical={music.get('status')}，board={music['audio_board']['status']}")
    return music


def _board_not_run(reason, metadata_path=None):
    return {'status': 'not_run', 'backend': 'Pillow', 'method': 'four_panel_physical_audio_measurements',
            'params': {}, 'path': None, 'metadata_path': metadata_path, 'reason': reason,
            'impact': 'Audio board unavailable; physical tempo, beats and alignment remain in music JSON.',
            'remedy': 'Repair local Pillow/audio environment or board metrics, then call ensure_audio_board(force=True); no second decode is needed when metrics exist.'}


def _render_in_audio_environment(metadata, image):
    environment = _runtime_environment()
    python = _audio_python(environment)
    if not python.is_file():
        raise FileNotFoundError(f'Pillow unavailable in main interpreter and audio python unavailable: {python}')
    run = subprocess.run([str(python), str(ROOT / 'Core/audio_board.py'), '--metrics', str(metadata),
                          '--output', str(image)], env=environment, capture_output=True, text=True,
                         timeout=RENDER_TIMEOUT_SECONDS, check=True)
    result = _last_json(run.stdout)
    if result['status'] != 'complete':
        raise RuntimeError(result.get('reason', 'audio board CLI did not complete'))
    return result


def ensure_audio_board(music, package_dir, force=False):
    """Persist deterministic metrics, reuse a verified PNG, or redraw only from JSON."""
    if package_dir is None:
        return _board_not_run('no package target')
    if music.get('status') in ('failed', 'not_run') and not (isinstance(music.get('board_data'), dict) and music['board_data'].get('status') == 'complete'):
        return _board_not_run('current physical audio measurement ' + music['status'] + '; previous metrics are not evidence for this run')
    metadata_relative = None
    try:
        root = Path(package_dir).expanduser().resolve()
        image, metadata = root / IMAGE_PATH, root / METRICS_PATH
        if not image.resolve().is_relative_to(root) or not metadata.resolve().is_relative_to(root):
            raise ValueError('audio board target escapes the package directory')
        if music.get('board_data') is not None:
            payload = make_board_payload(music)
        elif metadata.is_file():
            payload = json.loads(metadata.read_text(encoding='utf-8'))
        else:
            return _board_not_run('bounded board_data and persisted metrics unavailable; no second decode attempted')
        validate_payload(payload)
        encoded = deterministic_json(payload)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        if not metadata.is_file() or metadata.read_text(encoding='utf-8') != encoded:
            metadata.write_text(encoded, encoding='utf-8')
        metadata_relative = METRICS_PATH
        previous = music.get('audio_board') or {}
        if (not force and previous.get('status') == 'complete' and previous.get('data_sha256') == payload_sha256(payload)
                and image.is_file() and previous.get('sha256') == hashlib.sha256(image.read_bytes()).hexdigest()):
            return dict(previous)
        try:
            result = render_from_json(metadata, image)
        except (ImportError, ModuleNotFoundError):
            result = _render_in_audio_environment(metadata, image)
        if result.get('status') != 'complete' or not image.is_file():
            raise ValueError('audio board renderer did not produce a complete image')
        result = dict(result, path=IMAGE_PATH, metadata_path=METRICS_PATH,
                      sha256=hashlib.sha256(image.read_bytes()).hexdigest(), data_sha256=payload_sha256(payload))
        return result
    except Exception as exc:
        return _board_not_run(f'{type(exc).__name__}: {exc}', metadata_relative)
