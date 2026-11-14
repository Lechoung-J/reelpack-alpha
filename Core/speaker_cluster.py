#!/usr/bin/env python3
"""Weight-free experimental speaker clustering from per-window MFCC statistics.

This is a self-contained algorithm implementation. It groups speech-energy windows by acoustic
similarity only: it does not name speakers, bind them to faces, or participate
in the main media pipeline. Heavy dependencies (numpy/librosa/scikit-learn) are
imported lazily so standard-library callers can import this module cheaply.

Failure contract: `cluster_speakers` never raises; it returns a dict whose
`status` is `complete`, `not_run` (nothing measurable) or `failed`. Non-complete
states always carry a `reason`; a `complete` result with fewer than three
windows also carries one and sets `confidence.low_window_count`.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

SAMPLE_RATE = 16000
BACKEND = 'librosa_mfcc_stats+kmeans'

FRAME_LENGTH = 400                 # 25 ms at 16 kHz
HOP_LENGTH = 160                   # 10 ms at 16 kHz
GAP_MERGE_SECONDS = 0.30           # close active runs separated by a short pause
MIN_ACTIVE_SECONDS = 0.40          # discard shorter energy blips
MIN_WINDOW_SECONDS = 1.0
MAX_WINDOW_SECONDS = 3.0
MAX_WINDOWS = 600                  # bound silhouette/feature memory on long media
MAX_AUTO_CLUSTERS = 4
N_MFCC = 13
N_MELS = 40
N_FFT = 512

SILHOUETTE_THRESHOLD = 0.55        # below this, prefer the single-cluster null
SILHOUETTE_TIE_TOLERANCE = 0.02    # prefer the smaller k when scores are near equal
MIN_CENTROID_DISTANCE = 1.2        # in z-scored feature units; blocks duplicate splits
RANDOM_STATE = 42
MIN_DBFS = -80.0                   # below this the whole file is treated as silent

LOW_WINDOW_REASON = '样本过短，未做区分'
LOW_WINDOW_TRUST_REASON = '样本过短，聚类结果可信度低'


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


def _params(*, max_seconds=None, analyzed_seconds=None):
    safe_max_seconds = None
    if max_seconds is not None:
        try:
            candidate = float(max_seconds)
            safe_max_seconds = candidate if math.isfinite(candidate) else None
        except (TypeError, ValueError):
            safe_max_seconds = None
    return {
        'sample_rate': SAMPLE_RATE,
        'channels': 1,
        'max_seconds': safe_max_seconds,
        'analyzed_seconds': analyzed_seconds,
        'frame_seconds': FRAME_LENGTH / SAMPLE_RATE,
        'hop_seconds': HOP_LENGTH / SAMPLE_RATE,
        'gap_merge_seconds': GAP_MERGE_SECONDS,
        'min_active_seconds': MIN_ACTIVE_SECONDS,
        'min_window_seconds': MIN_WINDOW_SECONDS,
        'max_window_seconds': MAX_WINDOW_SECONDS,
        'window_cap': MAX_WINDOWS,
        'n_mfcc': N_MFCC,
        'n_mels': N_MELS,
        'n_fft': N_FFT,
        'feature': 'mfcc_mean_and_std',
        'standardize': 'zscore',
        'cluster_algorithm': 'kmeans',
        'auto_k_range': [2, MAX_AUTO_CLUSTERS],
        'silhouette_threshold': SILHOUETTE_THRESHOLD,
        'silhouette_tie_tolerance': SILHOUETTE_TIE_TOLERANCE,
        'min_centroid_distance': MIN_CENTROID_DISTANCE,
        'random_state': RANDOM_STATE,
        'weights': None,
        'experimental': True,
    }


def _blank(status, reason, params):
    return {'status': status, 'backend': BACKEND, 'params': params, 'windows': [],
            'clusters': {}, 'confidence': {}, 'window_count': 0,
            'duration_seconds': params.get('analyzed_seconds'), 'reason': reason}


def decode_audio(path, sr=SAMPLE_RATE):
    """Decode any ffmpeg-readable file to mono float32 at `sr`.

    Raises on missing files and decode failures; `cluster_speakers` converts
    those exceptions into its `failed` result.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'audio file not found: {source}')
    import numpy as np
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
        subprocess.run(
            ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source),
             '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(sr), tmp.name],
            check=True, capture_output=True, text=True, timeout=900)
        y, rate = sf.read(tmp.name, dtype='float32')
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if rate != sr:
        raise ValueError(f'decoded sample rate {rate} != requested {sr}')
    return y, sr


def _frame_runs(mask):
    """Return inclusive (first, last) frame indices for each True run."""
    runs = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def detect_windows(y, sr=SAMPLE_RATE):
    """Cut 1-3 s analysis windows from energy-active speech regions.

    Returns ``(windows, reason)`` where ``windows`` is a list of
    ``(start_seconds, end_seconds)`` tuples. The threshold is relative to the
    file's own percentile floor and peak, so quiet recordings still work; a
    constant-level recording falls back to a peak-relative threshold instead of
    producing an empty mask.
    """
    import numpy as np
    import librosa

    audio = np.asarray(y, dtype=np.float32)
    duration = len(audio) / sr
    if duration < MIN_WINDOW_SECONDS:
        return [], f'audio shorter than the minimum {MIN_WINDOW_SECONDS:g} s window'
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH,
                              hop_length=HOP_LENGTH, center=False)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    if not np.isfinite(db).all():
        return [], 'non-finite frame energy'
    peak = float(np.percentile(db, 95))
    if peak < MIN_DBFS:  # sparse audible content below the 95th percentile
        peak = float(np.max(db))
    if peak < MIN_DBFS:
        return [], 'audio is silent'
    floor = float(np.percentile(db, 10))
    threshold = min(peak - 10.0, max(floor + 8.0, peak - 35.0))
    active = db > threshold
    runs = _frame_runs(active)

    gap_frames = int(round(GAP_MERGE_SECONDS * sr / HOP_LENGTH))
    merged = []
    for first, last in runs:
        if merged and first - merged[-1][1] - 1 <= gap_frames:
            merged[-1] = (merged[-1][0], last)
        else:
            merged.append((first, last))

    windows = []
    for first, last in merged:
        start = first * HOP_LENGTH / sr
        end = min(duration, (last * HOP_LENGTH + FRAME_LENGTH) / sr)
        span = end - start
        if span < MIN_ACTIVE_SECONDS:
            continue
        if span < MIN_WINDOW_SECONDS:
            pad = (MIN_WINDOW_SECONDS - span) / 2.0
            start, end = max(0.0, start - pad), min(duration, end + pad)
            if end - start < MIN_WINDOW_SECONDS:
                if start <= 0.0:
                    end = min(duration, start + MIN_WINDOW_SECONDS)
                else:
                    start = max(0.0, end - MIN_WINDOW_SECONDS)
            if end - start < MIN_WINDOW_SECONDS:
                continue
        if end - start > MAX_WINDOW_SECONDS:
            count = int(math.ceil((end - start) / MAX_WINDOW_SECONDS))
            while count > 1 and (end - start) / count < MIN_WINDOW_SECONDS:
                count -= 1
            edges = np.linspace(start, end, count + 1)
            windows.extend((float(edges[i]), float(edges[i + 1])) for i in range(count))
        else:
            windows.append((float(start), float(end)))

    if len(windows) > MAX_WINDOWS:
        pick = np.linspace(0, len(windows) - 1, MAX_WINDOWS)
        windows = [windows[int(round(index))] for index in pick]
    if not windows:
        return [], 'no speech-like energy window was detected'
    return windows, None


def _window_features(y, sr, windows):
    """MFCC mean/std per window; windows too short to featurize are dropped."""
    import numpy as np
    import librosa

    kept, rows = [], []
    for start, end in windows:
        first = max(0, int(round(start * sr)))
        last = min(len(y), int(round(end * sr)))
        segment = np.asarray(y[first:last], dtype=np.float32)
        if segment.size < FRAME_LENGTH:
            continue
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                                    hop_length=HOP_LENGTH, n_mels=N_MELS)
        rows.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
        kept.append((start, end))
    if not rows:
        return [], None
    return kept, np.vstack(rows)


def _standardize(features):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(features)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def _kmeans(features, k):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(features)
    return labels, model.cluster_centers_


def _silhouette(features, labels):
    import numpy as np
    from sklearn.metrics import silhouette_score
    distinct = len(np.unique(labels))
    if distinct < 2 or distinct >= len(features):
        return None
    return float(silhouette_score(features, labels))


def _separation(centers):
    import numpy as np
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) < 2:
        return 0.0
    distances = [float(np.linalg.norm(centers[i] - centers[j]))
                 for i in range(len(centers)) for j in range(i + 1, len(centers))]
    return min(distances) if distances else 0.0


def _candidate(features, k):
    import numpy as np
    labels, centers = _kmeans(features, k)
    silhouette = _silhouette(features, labels)
    separation = _separation(centers)
    accepted = (len(np.unique(labels)) == k and silhouette is not None
                and silhouette >= SILHOUETTE_THRESHOLD
                and separation >= MIN_CENTROID_DISTANCE)
    return labels, silhouette, separation, accepted


def _choose_clusters(features):
    """Silhouette-selected k in 2..4, gated by cluster separation.

    Identical or near-identical windows can make KMeans produce two groups
    with a perfect silhouette; the centroid-distance gate keeps such duplicate
    material in one cluster.

    Rows that collapse to fewer distinct feature vectors than ``k`` cannot form
    ``k`` groups; that is recorded as an explicit ``skipped`` candidate rather
    than by swallowing a KMeans error. Any other candidate error propagates to
    ``cluster_speakers`` and becomes a top-level ``failed`` result.
    """
    candidates = []
    accepted = []
    distinct_rows = len({tuple(float(value) for value in row) for row in features})
    limit = min(MAX_AUTO_CLUSTERS, len(features) - 1)
    for k in range(2, limit + 1):
        if distinct_rows < k:
            candidates.append({'k': k, 'silhouette': None,
                               'min_centroid_distance': None, 'accepted': False,
                               'skipped': 'fewer distinct feature rows than k'})
            continue
        labels, silhouette, separation, admissible = _candidate(features, k)
        candidates.append({'k': k, 'silhouette': _round(silhouette),
                           'min_centroid_distance': _round(separation),
                           'accepted': bool(admissible)})
        if admissible:
            accepted.append((k, silhouette, labels))
    if not accepted:
        return None, candidates
    preferred = max(score for _, score, _ in accepted) - SILHOUETTE_TIE_TOLERANCE
    k, silhouette, labels = min((item for item in accepted if item[1] >= preferred),
                                key=lambda item: item[0])
    return (labels, silhouette, k), candidates


def _relabel_by_time(windows, labels):
    """Renumber clusters 0..n-1 by their first window start for stable output."""
    ordered = []
    for cluster in sorted(set(int(label) for label in labels)):
        starts = [windows[index][0] for index, label in enumerate(labels)
                  if int(label) == cluster]
        ordered.append((min(starts), cluster))
    ordered.sort()
    mapping = {old: new for new, (_, old) in enumerate(ordered)}
    return [mapping[int(label)] for label in labels], len(mapping)


def _summarize(windows, labels, count):
    clusters = {}
    for cluster in range(count):
        indices = [index for index, label in enumerate(labels) if label == cluster]
        if not indices:
            continue
        clusters[str(cluster)] = {
            'window_count': len(indices),
            'duration_seconds': _round(sum(windows[i][1] - windows[i][0] for i in indices), 3),
            'first_start_seconds': _round(windows[indices[0]][0], 3),
            'last_end_seconds': _round(windows[indices[-1]][1], 3),
        }
    return clusters


def cluster_speakers(audio_path, *, k=None, max_seconds=None) -> dict:
    """Cluster speech-energy windows of `audio_path` into speaker groups.

    ``k`` fixed keeps exactly that many clusters (clamped to the window count);
    ``k=None`` selects 2..4 by silhouette and falls back to one cluster when no
    split clears the quality gates. ``max_seconds`` limits the analyzed prefix.
    """
    params = _params(max_seconds=max_seconds)
    try:
        if max_seconds is not None:
            if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
                    or not math.isfinite(max_seconds) or max_seconds <= 0):
                return _blank('failed', 'max_seconds must be a positive finite number', params)
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            return _blank('failed', 'k must be a positive integer or None', params)

        audio, sr = decode_audio(audio_path)
        duration = len(audio) / sr
        if max_seconds is not None and duration > max_seconds:
            audio = audio[:max(0, int(round(max_seconds * sr)))]
            duration = len(audio) / sr
        params = _params(max_seconds=max_seconds, analyzed_seconds=_round(duration, 3))

        windows, reason = detect_windows(audio, sr)
        if not windows:
            return _blank('not_run', reason, params)

        windows, features = _window_features(audio, sr, windows)
        if not windows or features is None:
            return _blank('not_run', 'no window could be featurized', params)
        scaled = _standardize(features)

        if k is not None:
            effective = min(k, len(windows))
            if effective == 1 or len(windows) < 2:
                labels, count = [0] * len(windows), 1
                confidence = {'method': 'fixed_k', 'selected_k': 1,
                              'requested_k': k, 'silhouette': None}
            else:
                labels, centers = _kmeans(scaled, effective)
                labels, count = _relabel_by_time(windows, labels)
                confidence = {
                    'method': 'fixed_k', 'selected_k': count, 'requested_k': k,
                    'silhouette': _round(_silhouette(scaled, labels)),
                    'min_centroid_distance': _round(_separation(centers)),
                    'silhouette_threshold': SILHOUETTE_THRESHOLD,
                    'min_centroid_distance_threshold': MIN_CENTROID_DISTANCE,
                }
            if k > len(windows):
                confidence['warning'] = (f'k={k} exceeds the {len(windows)} available '
                                         f'windows; using {count} cluster(s)')
            if len(windows) < 3:
                confidence['low_window_count'] = True
                confidence['reason'] = (LOW_WINDOW_REASON if count == 1
                                        else LOW_WINDOW_TRUST_REASON)
        else:
            if len(windows) < 3:
                labels, count = [0] * len(windows), 1
                confidence = {'method': 'silhouette', 'selected_k': 1,
                              'silhouette': None, 'candidates': [],
                              'low_window_count': True,
                              'reason': LOW_WINDOW_REASON}
            else:
                best, candidates = _choose_clusters(scaled)
                if best is None:
                    labels, count = [0] * len(windows), 1
                    confidence = {
                        'method': 'silhouette', 'selected_k': 1,
                        'silhouette': None, 'candidates': candidates,
                        'silhouette_threshold': SILHOUETTE_THRESHOLD,
                        'min_centroid_distance_threshold': MIN_CENTROID_DISTANCE,
                        'reason': 'no candidate split cleared the silhouette/separation gates',
                    }
                else:
                    labels, silhouette, _selected = best
                    labels, count = _relabel_by_time(windows, labels)
                    confidence = {
                        'method': 'silhouette', 'selected_k': count,
                        'silhouette': _round(silhouette), 'candidates': candidates,
                        'silhouette_threshold': SILHOUETTE_THRESHOLD,
                        'min_centroid_distance_threshold': MIN_CENTROID_DISTANCE,
                    }

        result = {
            'status': 'complete',
            'backend': BACKEND,
            'params': params,
            'windows': [{'start': _round(start, 3), 'end': _round(end, 3),
                         'cluster': int(label)}
                        for (start, end), label in zip(windows, labels)],
            'clusters': _summarize(windows, labels, count),
            'confidence': confidence,
            'window_count': len(windows),
            'duration_seconds': _round(duration, 3),
        }
        if confidence.get('low_window_count'):
            result['reason'] = confidence['reason']
        return result
    except Exception as exc:  # noqa: BLE001 - the caller contract forbids raising
        detail = getattr(exc, 'stderr', None) or str(exc)
        if isinstance(detail, bytes):
            detail = detail.decode('utf-8', errors='replace')
        return _blank('failed', f'{type(exc).__name__}: {str(detail)[-800:]}', params)
