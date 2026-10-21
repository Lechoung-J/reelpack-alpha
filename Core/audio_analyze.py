#!/usr/bin/env python3
"""Isolated full-mix audio measurements; bounded board data, never raw WAV JSON."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from music_timbre import analyze_timbre

FULL_MIX_LABEL = 'Full mix (all audio, mono decode; no source separation)'
BACKEND = 'librosa+soundfile+pyloudnorm'


def semantic_not_run(field):
    return {'status': 'not_run', 'backend': None, 'model': None, 'params': {}, 'items': [],
            'reason': f'{field} semantic backend was not explicitly selected',
            'impact': f'Physical measurements do not establish {field} labels.',
            'remedy': 'Explicitly select PANNs with locally provided weights and labels; no automatic download.'}


def _progress(progress, stage, percent, message):
    if progress is not None:
        progress(stage, percent, message)


def extract_audio(path: str, sr: int = 22050):
    import numpy as np
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(path),
                        '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(sr), tmp.name],
                       check=True, timeout=900)
        y, rate = sf.read(tmp.name, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y), rate


def build_board_data(y, sr, *, source_hash=None, max_times=600, max_freqs=128,
                     n_fft=2048, hop_length=512, frequency_scale='log', db_floor=-80.):
    """Aggregate waveform and real Hann STFT in linear power before converting to dB.

    Frequency rebinning uses overlaps of FFT-bin Hz intervals with target Hz bands.
    Log bands omit DC/sub-minimum frequencies explicitly, rather than relabeling a
    linearly spaced image as logarithmic. STFT batches limit intermediate memory.
    """
    import numpy as np
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all() or sr <= 0:
        raise ValueError('board requires finite, non-empty mono audio and positive sample rate')
    if not 1 <= max_times <= 600 or not 1 <= max_freqs <= 128 or n_fft < 4 or n_fft % 2 or hop_length < 1:
        raise ValueError('invalid board bounds or STFT parameters')
    if frequency_scale not in ('log', 'linear') or not math.isfinite(db_floor) or db_floor >= 0:
        raise ValueError('invalid frequency scale or dB floor')
    duration = len(y) / sr
    count = min(max_times, len(y))
    sample_edges = np.linspace(0, len(y), count + 1, dtype=np.int64)
    slices = [y[a:b] for a, b in zip(sample_edges[:-1], sample_edges[1:])]
    lo = [float(x.min()) for x in slices]
    hi = [float(x.max()) for x in slices]
    rms = np.asarray([np.sqrt(np.mean(x * x)) for x in slices])
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))
    global_rms = float(np.sqrt(np.mean(y * y)))
    waveform = {
        'status': 'measured', 'method': 'sample_bucket_min_max_and_rms',
        'params': {'max_time_bins': max_times, 'rms_db_floor': -120., 'db_reference_amplitude': 1.,
                   'mean_db_aggregation': 'sample_count_weighted_mean_of_bucket_rms_dbfs'},
        'units': {'amplitude': 'normalized PCM', 'rms': 'normalized PCM', 'rms_db': 'dBFS', 'time': 'seconds'},
        'time_edges_seconds': (sample_edges / sr).tolist(),
        'times_seconds': ((sample_edges[:-1] + sample_edges[1:]) / (2 * sr)).tolist(),
        'min': lo, 'max': hi, 'rms': rms.tolist(), 'rms_dbfs': rms_db.tolist(),
        'rms_dbfs_mean': float(np.average(rms_db, weights=np.diff(sample_edges))),
        'rms_linear_global': global_rms, 'rms_dbfs_global': float(20 * np.log10(max(global_rms, 1e-6))),
        'sample_count': int(len(y)),
    }
    fft_hz = np.fft.rfftfreq(n_fft, 1 / sr)
    source_edges = np.r_[0., (fft_hz[:-1] + fft_hz[1:]) / 2, sr / 2]
    if frequency_scale == 'log':
        minimum = min(20., sr / 4.)
        freq_edges = np.geomspace(minimum, sr / 2, max_freqs + 1)
        freq_centers = np.sqrt(freq_edges[:-1] * freq_edges[1:])
    else:
        freq_edges = np.linspace(0., sr / 2, max_freqs + 1)
        freq_centers = (freq_edges[:-1] + freq_edges[1:]) / 2
    overlap = np.maximum(0., np.minimum(freq_edges[1:, None], source_edges[None, 1:]) -
                         np.maximum(freq_edges[:-1, None], source_edges[None, :-1]))
    weights = overlap / np.diff(freq_edges)[:, None]
    frame_count = 1 + len(y) // hop_length
    time_count = min(max_times, frame_count)
    time_edges = np.linspace(0., duration, time_count + 1)
    sums = np.zeros((time_count, max_freqs))
    counts = np.zeros(time_count, dtype=np.int64)
    window = np.hanning(n_fft)
    padded = np.pad(y, (n_fft // 2, n_fft // 2))
    indices = np.arange(n_fft)
    for first in range(0, frame_count, 128):
        frame_ids = np.arange(first, min(frame_count, first + 128))
        samples = padded[frame_ids[:, None] * hop_length + indices[None, :]]
        power = (np.abs(np.fft.rfft(samples * window, axis=1)) / window.sum()) ** 2
        band_power = power @ weights.T
        buckets = np.minimum(time_count - 1, (frame_ids * hop_length / len(y) * time_count).astype(int))
        np.add.at(sums, buckets, band_power)
        np.add.at(counts, buckets, 1)
    if np.any(counts == 0):
        raise ValueError('STFT time aggregation contains uncovered intervals')
    power = sums / counts[:, None]
    db = np.maximum(db_floor, 10 * np.log10(np.maximum(power, 10 ** (db_floor / 10))))
    spectrum = {
        'status': 'measured', 'method': 'hann_stft_band_mean_linear_power_then_db',
        'params': {'n_fft': n_fft, 'hop_length': hop_length, 'window': 'numpy.hanning (symmetric Hann)',
                   'center': True, 'padding': 'constant_zero', 'max_time_bins': max_times,
                   'max_frequency_bins': max_freqs, 'frequency_scale': frequency_scale,
                   'frequency_rebin': 'Hz_interval_overlap_weighted_mean_linear_power',
                   'time_rebin': 'frame_center_bucket_mean_linear_power',
                   'power_normalization': 'abs(rfft(window * samples) / window.sum()) ** 2',
                   'db_reference_power': 1., 'db_floor': db_floor,
                   'excluded_below_hz': float(freq_edges[0])},
        'units': {'time': 'seconds', 'frequency': 'Hz', 'values': 'dB re normalized STFT power 1'},
        'time_edges_seconds': time_edges.tolist(),
        'times_seconds': ((time_edges[:-1] + time_edges[1:]) / 2).tolist(),
        'frequency_edges_hz': freq_edges.tolist(), 'frequencies_hz': freq_centers.tolist(),
        'frame_counts': counts.tolist(), 'values_db': np.round(db, 4).tolist(),
        'matrix_order': 'time_by_frequency',
    }
    return {'status': 'complete', 'schema_version': 1, 'sample_rate': int(sr), 'duration_seconds': duration,
            'source_sha256': source_hash, 'scope': 'whole_audio_mix', 'full_mix_label': FULL_MIX_LABEL,
            'waveform': waveform, 'spectrogram': spectrum,
            'boundary_disclaimer': 'Full-mix physical measurements only; no genre, emotion, authorship, instrument or BGM identity conclusions. Beat/cut proximity does not prove editing intent.'}


def analyze_array(y, sr: int, *, hop_seconds=.5, source_hash=None, progress=None):
    import numpy as np
    import librosa
    import pyloudnorm as pyln
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all() or sr <= 0:
        raise ValueError('empty, non-finite or non-mono audio')
    if not math.isfinite(hop_seconds) or hop_seconds <= 0:
        raise ValueError('RMS segment seconds must be positive')
    _progress(progress, 'audio_physical', 36, '测量全混音 RMS、onset 和拍点')
    hop = max(1, int(sr * hop_seconds))
    frame = max(1, int(sr))
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop).tolist()
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    tempo_val = float(np.asarray(tempo).reshape(-1)[0])
    if not math.isfinite(tempo_val):
        tempo_val = None
    beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    if not beats:
        # An empty detector result cannot establish a zero BPM tempo.
        tempo_val = None
    loudness_error = None
    try:
        loudness = float(pyln.Meter(sr).integrated_loudness(y))
        if math.isfinite(loudness):
            loudness_status = 'complete'
        else:
            loudness, loudness_status, loudness_error = None, 'not_measured', 'non-finite loudness (e.g. silence)'
    except Exception as exc:
        loudness, loudness_status, loudness_error = None, 'failed', str(exc)
    result = {
        'status': 'partial' if loudness_status != 'complete' else 'complete', 'backend': BACKEND,
        'model': None, 'scope': 'whole_audio_mix', 'full_mix_label': FULL_MIX_LABEL,
        'sample_rate': int(sr), 'duration_seconds': len(y) / sr,
        'units': {'tempo_bpm': 'beats/minute', 'time': 'seconds', 'energy': 'normalized PCM RMS'},
        'params': {'sample_rate': sr, 'channels': 1, 'rms_segment_seconds': hop_seconds,
                   'beat_tracking': 'librosa.beat.beat_track', 'onset': 'librosa.onset.onset_detect',
                   'loudness': 'ITU-R BS.1770 via pyloudnorm'},
        'tempo_bpm': tempo_val, 'beat_times_seconds': beats, 'beat_count': len(beats),
        'beats': {'status': 'measured' if beats else 'not_measured', 'method': 'librosa.beat.beat_track',
                  'params': {'hop_length': 512}, 'units': 'seconds',
                  **({'reason': 'no detected beats; tempo/beat alignment not established'} if not beats else {})},
        'onset_times_seconds': onset_times, 'onset_count': len(onset_times),
        'energy': {'type': 'rms', 'segment_seconds': hop_seconds, 'times_seconds': rms_times,
                   'values': rms.astype(float).tolist(), 'units': 'normalized PCM RMS'},
        'loudness': {'status': loudness_status, 'integrated_lufs': loudness, 'units': 'LUFS',
                     **({'reason': loudness_error} if loudness_error else {})},
        'genre': semantic_not_run('genre'), 'mood': semantic_not_run('mood'),
        'timbre': analyze_timbre(y, sr),
    }
    _progress(progress, 'audio_physical', 40, '聚合波形与真实 STFT 频谱供证据板重绘')
    try:
        result['board_data'] = build_board_data(y, sr, source_hash=source_hash)
    except Exception as exc:
        result['board_data'] = {'status': 'not_run', 'reason': f'{type(exc).__name__}: {exc}',
                                'impact': 'Audio board unavailable; physical tempo and beats are retained.',
                                'remedy': 'Repair board DSP parameters or dependencies and rerun audio measurements.'}
    for name in ('timbre', 'loudness', 'beats'):
        field = result[name]
        if field.get('status') in ('not_run', 'failed', 'not_measured'):
            field.setdefault('impact', f'No {name} conclusion is established; other full-mix measures are independent.')
            field.setdefault('remedy', 'Inspect the local audio signal/dependencies and rerun the measure; do not infer missing results.')
    return result


def analyze(path: str, *, progress=None):
    _progress(progress, 'audio_decode', 32, '解码本地全混音为单声道测量输入')
    y, sr = extract_audio(path)
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return analyze_array(y, sr, source_hash=digest.hexdigest(), progress=progress)


def _cli_progress(stage, percent, message):
    print(json.dumps({'audio_progress': {'stage': stage, 'percent': percent, 'message': message}},
                     ensure_ascii=False), file=sys.stderr, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    args = ap.parse_args()
    try:
        import os
        try:
            from runtime_environment import prepare_runtime_environment
        except ImportError:
            prepare_runtime_environment = None
        if prepare_runtime_environment is not None:
            os.environ.update(prepare_runtime_environment(create=os.environ.get('REELPACK_READONLY_CHECK') != '1'))
        print(json.dumps(analyze(args.input, progress=_cli_progress), ensure_ascii=False, allow_nan=False, sort_keys=True))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'backend': BACKEND, 'model': None, 'params': {},
                          'error': f'{type(exc).__name__}: {exc}', 'reason': f'{type(exc).__name__}: {exc}',
                          'impact': 'Physical audio measures and beat alignment are unavailable.',
                          'remedy': 'Check the local media, ffmpeg and explicit audio interpreter; repair locally and retry.',
                          'tempo_bpm': None, 'beat_times_seconds': [], 'beat_count': 0,
                          'genre': semantic_not_run('genre'), 'mood': semantic_not_run('mood')},
                         ensure_ascii=False, allow_nan=False, sort_keys=True))
