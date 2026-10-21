"""Weight-free experimental spectral timbre descriptors, NOT genre/mood labels."""
import math


def describe_timbre(centroid_hz, flatness, *, duration_seconds, sample_rate):
    base = {'status': 'complete', 'backend': 'librosa_spectral', 'model': 'spectral-descriptors-v1',
            'experimental': True, 'params': {'sample_rate': sample_rate, 'n_fft': 2048,
            'hop_length': 512, 'aggregation': 'median', 'weights': None},
            'scope': 'whole_audio_mix', 'time_range': [0, duration_seconds],
            'inference_note': '描述全混音的频谱，不确认乐器、音乐存在、genre 或情绪；人声和噪声会影响结果。'}
    if not all(math.isfinite(x) for x in (centroid_hz, flatness, duration_seconds)):
        return dict(base, status='failed', reason='non-finite spectral descriptor', result=None)
    base['result'] = {'spectral_centroid_hz': centroid_hz, 'spectral_flatness': flatness,
                      'brightness_hint': 'relative descriptor; no categorical threshold',
                      'tonality_hint': 'lower flatness suggests more tonal spectrum; not an instrument label'}
    return base


def analyze_timbre(y, sr):
    base = {'backend': 'librosa_spectral', 'model': 'spectral-descriptors-v1',
            'experimental': True, 'params': {'sample_rate': sr, 'n_fft': 2048, 'hop_length': 512,
            'aggregation': 'median', 'weights': None}, 'scope': 'whole_audio_mix'}
    try:
        import numpy as np
        import librosa
        if len(y) == 0 or not np.isfinite(y).all():
            raise ValueError('empty or non-finite audio')
        if float(np.max(np.abs(y))) < 1e-8:
            return dict(base, status='not_run', reason='silent audio; no meaningful timbre', result=None)
        spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        return describe_timbre(float(np.median(librosa.feature.spectral_centroid(S=spectrum, sr=sr))),
            float(np.median(librosa.feature.spectral_flatness(S=spectrum))),
            duration_seconds=len(y)/sr, sample_rate=sr)
    except Exception as exc:
        return dict(base, status='failed', reason=f'{type(exc).__name__}: {exc}', result=None)
