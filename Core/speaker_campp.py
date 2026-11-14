#!/usr/bin/env python3
"""Experimental CAM++ speaker clustering backed by sherpa-onnx.

This is an opt-in experimental backend next to ``speaker_cluster`` (the MFCC backend). It cuts
fixed 1.5 s windows with a 0.75 s hop from the speech-energy regions of a file,
embeds each window with a CAM++ speaker-embedding model through the
``sherpa-onnx`` package, L2-normalises the vectors and groups them with KMeans.
It does not name speakers, bind them to faces, or participate in the main media
pipeline, and it never downloads a model.

Failure contract mirrors ``speaker_cluster.cluster_speakers``: this function
never raises; it returns a dict whose ``status`` is ``complete``, ``not_run``
(nothing measurable, including a missing model) or ``failed``. Non-complete
states always carry a ``reason``; a ``complete`` result with fewer than three
windows also carries one and sets ``confidence.low_window_count``.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

SAMPLE_RATE = 16000
BACKEND = 'campp_sherpa_onnx+kmeans'
MODEL_ENV = 'REELPACK_SPEAKER_MODEL'

WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.75
ENERGY_THRESHOLD = 0.005          # per-window RMS below this is treated as silence
RELATIVE_ENERGY_FACTOR = 0.20     # also drop windows >14 dB below the loud decile
MIN_WINDOWS_FOR_RELATIVE_GATE = 8  # too few windows make the loud decile unreliable
MAX_WINDOWS = 600                 # bound embedding memory/time on long media
MAX_AUTO_CLUSTERS = 4
NUM_THREADS = 2

SILHOUETTE_THRESHOLD = 0.35       # below this, prefer the single-cluster null
SILHOUETTE_TIE_TOLERANCE = 0.02   # prefer the smaller k when scores are near equal
MIN_CENTROID_DISTANCE = 0.30      # cosine distance between centroids; blocks duplicate splits
K2_PREFERENCE_MARGIN = 0.05       # prefer k=2 over a higher k this close in silhouette
K2_MIN_CLUSTER_SHARE = 0.05       # each k=2 cluster must hold at least this share of windows
RANDOM_STATE = 42

MODEL_MISSING_REASON = 'model_missing'
LOW_WINDOW_REASON = '样本过短，未做区分'
LOW_WINDOW_TRUST_REASON = '样本过短，聚类结果可信度低'


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


def _params(*, max_seconds=None, analyzed_seconds=None, model_name=None):
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
        'window_seconds': WINDOW_SECONDS,
        'hop_seconds': HOP_SECONDS,
        'energy_threshold_rms': ENERGY_THRESHOLD,
        'relative_energy_factor': RELATIVE_ENERGY_FACTOR,
        'window_cap': MAX_WINDOWS,
        'embedding': 'campp',
        'embedding_model': model_name,
        'standardize': 'l2_normalize',
        'cluster_algorithm': 'kmeans',
        'distance': 'cosine',
        'auto_k_range': [2, MAX_AUTO_CLUSTERS],
        'silhouette_threshold': SILHOUETTE_THRESHOLD,
        'silhouette_tie_tolerance': SILHOUETTE_TIE_TOLERANCE,
        'k2_preference_margin': K2_PREFERENCE_MARGIN,
        'k2_min_cluster_share': K2_MIN_CLUSTER_SHARE,
        'min_centroid_distance': MIN_CENTROID_DISTANCE,
        'random_state': RANDOM_STATE,
        'num_threads': NUM_THREADS,
        'weights': None,
        'experimental': True,
    }


def _blank(status, reason, params):
    return {'status': status, 'backend': BACKEND, 'params': params, 'windows': [],
            'clusters': {}, 'confidence': {}, 'window_count': 0,
            'duration_seconds': params.get('analyzed_seconds'), 'reason': reason}


def _model_file(candidate):
    try:
        path = Path(candidate).expanduser()
    except (TypeError, ValueError):
        return None
    return path if path.is_file() else None


def resolve_model(model_path=None):
    """Resolve the CAM++ model without ever downloading one.

    An explicit ``model_path`` wins over ``REELPACK_SPEAKER_MODEL``; if the
    explicit path does not name a file the result is ``None`` rather than a
    silent fallback to the environment variable.
    """
    if model_path is not None:
        return _model_file(model_path)
    return _model_file(os.environ[MODEL_ENV]) if os.environ.get(MODEL_ENV) else None


def decode_audio(path, max_seconds=None):
    """Decode any ffmpeg-readable file to mono float32 at 16 kHz via a pipe.

    Raises on missing files and decode failures; ``cluster_speakers`` converts
    those exceptions into its ``failed`` result.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'audio file not found: {source}')
    import numpy as np
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source)]
    if max_seconds is not None:
        command += ['-t', f'{float(max_seconds):.3f}']
    command += ['-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE),
                '-f', 'f32le', '-']
    completed = subprocess.run(command, check=True, capture_output=True, timeout=900)
    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    return np.ascontiguousarray(audio), SAMPLE_RATE


def detect_windows(audio, sr=SAMPLE_RATE):
    """Cut fixed 1.5 s windows (0.75 s hop) whose RMS clears the energy floor.

    Two thresholds are applied. The absolute floor rejects digital silence and
    quiet hiss. A relative gate then rejects windows more than ~14 dB below the
    loudest decile of the remaining windows: real recordings often end in a low
    room-tone tail that sits above the absolute floor, and embedding those
    windows creates a parallel "loud speech vs. quiet tail" split that competes
    with, and usually beats, the speaker split. The relative gate is skipped
    when fewer than ``MIN_WINDOWS_FOR_RELATIVE_GATE`` windows remain because the
    loud decile is not yet meaningful.

    Returns ``(windows, segments, reason)`` where ``windows`` is a list of
    ``(start_seconds, end_seconds)`` and ``segments`` holds the matching
    samples. Long media are subsampled to ``MAX_WINDOWS`` windows evenly.
    """
    import numpy as np

    signal = np.asarray(audio, dtype=np.float32)
    frame = int(round(WINDOW_SECONDS * sr))
    hop = int(round(HOP_SECONDS * sr))
    candidates = []
    for start in range(0, max(len(signal) - frame, 0), hop):
        segment = signal[start:start + frame]
        rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
        if not math.isfinite(rms) or rms < ENERGY_THRESHOLD:
            continue
        candidates.append((start, segment, rms))
    if not candidates:
        return [], [], 'no speech-like energy window was detected'
    if len(candidates) >= MIN_WINDOWS_FOR_RELATIVE_GATE:
        loud_level = float(np.percentile([rms for _, _, rms in candidates], 90))
        gate = max(ENERGY_THRESHOLD, RELATIVE_ENERGY_FACTOR * loud_level)
        candidates = [item for item in candidates if item[2] >= gate]
        if not candidates:
            return [], [], 'no window cleared the relative energy gate'

    windows = [(start / sr, (start + frame) / sr) for start, _, _ in candidates]
    segments = [segment for _, segment, _ in candidates]
    if len(windows) > MAX_WINDOWS:
        picked = []
        for index in np.linspace(0, len(windows) - 1, MAX_WINDOWS):
            resolved = int(round(float(index)))
            if not picked or resolved != picked[-1]:
                picked.append(resolved)
        windows = [windows[index] for index in picked]
        segments = [segments[index] for index in picked]
    return windows, segments, None


def _extractor(model_path):
    import sherpa_onnx
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path), num_threads=NUM_THREADS)
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def embed_windows(windows, segments, model_path):
    """Embed each window; returns the embedded windows and an L2-normalised matrix.

    Windows whose stream never becomes ready or produces a zero/non-finite
    embedding are dropped. Extractor errors propagate to the caller.
    """
    import numpy as np

    extractor = _extractor(model_path)
    kept, rows = [], []
    for window, segment in zip(windows, segments):
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, segment)
        stream.input_finished()
        if not extractor.is_ready(stream):
            continue
        vector = np.asarray(extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            continue
        kept.append(window)
        rows.append(vector / norm)
    if not rows:
        return [], None
    return kept, np.vstack(rows)


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
    return float(silhouette_score(features, labels, metric='cosine'))


def _separation(centers):
    """Minimum pairwise cosine distance between L2-normalised centroids."""
    import numpy as np
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) < 2:
        return 0.0
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    normalized = centers / np.maximum(norms, 1e-12)
    distances = [float(1.0 - normalized[i] @ normalized[j])
                 for i in range(len(normalized)) for j in range(i + 1, len(normalized))]
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


def _cluster_shares(labels, k):
    import numpy as np
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=k)
    return [float(count) / len(labels) for count in counts]


def _choose_clusters(features):
    """Silhouette-selected k in 2..4, gated by centroid separation.

    Duplicate material (for example one segment pasted repeatedly) yields
    near-identical embeddings; the centroid-distance gate keeps such material
    in one cluster. A k=2 split is preferred over a higher k whenever it is
    admissible, both of its clusters hold at least ``K2_MIN_CLUSTER_SHARE`` of
    the windows, and its silhouette is within ``K2_PREFERENCE_MARGIN`` of the
    best score. That breaks the tie in favour of the conversational case (two
    people) instead of an extra content/background cluster, while a lone
    outlier window cannot force a split. Rows that collapse to fewer distinct
    vectors than ``k`` are recorded as an explicit ``skipped`` candidate. Any
    other candidate error propagates to ``cluster_speakers`` and becomes a
    top-level ``failed``.
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
        shares = _cluster_shares(labels, k)
        entry = {'k': k, 'silhouette': _round(silhouette),
                 'min_centroid_distance': _round(separation),
                 'accepted': bool(admissible)}
        if k == 2:
            entry['cluster_shares'] = [_round(share) for share in shares]
            entry['balanced'] = bool(min(shares) >= K2_MIN_CLUSTER_SHARE)
        candidates.append(entry)
        if admissible:
            accepted.append({'k': k, 'silhouette': silhouette, 'labels': labels,
                             'balanced': entry.get('balanced', True)})
    if not accepted:
        return None, candidates
    best = max(item['silhouette'] for item in accepted)
    two = next((item for item in accepted if item['k'] == 2), None)
    if (two is not None and two['balanced']
            and best - two['silhouette'] < K2_PREFERENCE_MARGIN):
        chosen = two
    else:
        preferred = best - SILHOUETTE_TIE_TOLERANCE
        chosen = min((item for item in accepted if item['silhouette'] >= preferred),
                     key=lambda item: item['k'])
    for entry in candidates:
        if entry['k'] == chosen['k']:
            entry['selected'] = True
    return (chosen['labels'], chosen['silhouette'], chosen['k']), candidates


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


def cluster_speakers(audio_path, *, k=None, max_seconds=None, model_path=None) -> dict:
    """Cluster fixed CAM++ speech windows of `audio_path` into speaker groups.

    ``k`` fixed keeps exactly that many clusters (clamped to the window count);
    ``k=None`` selects 2..4 by cosine silhouette and falls back to one cluster
    when no split clears the quality gates. ``max_seconds`` limits the analyzed
    prefix. The model comes from ``model_path``, else ``REELPACK_SPEAKER_MODEL``;
    when neither names a file the result is ``not_run`` / ``model_missing``.
    """
    params = _params(max_seconds=max_seconds)
    try:
        if max_seconds is not None:
            if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
                    or not math.isfinite(max_seconds) or max_seconds <= 0):
                return _blank('failed', 'max_seconds must be a positive finite number', params)
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            return _blank('failed', 'k must be a positive integer or None', params)

        model = resolve_model(model_path)
        params = _params(max_seconds=max_seconds,
                         model_name=model.name if model else None)
        if model is None:
            return _blank('not_run', MODEL_MISSING_REASON, params)

        audio, sr = decode_audio(audio_path, max_seconds=max_seconds)
        duration = len(audio) / sr
        params = _params(max_seconds=max_seconds,
                         analyzed_seconds=_round(duration, 3),
                         model_name=model.name)

        windows, segments, reason = detect_windows(audio, sr)
        if not windows:
            return _blank('not_run', reason, params)

        windows, features = embed_windows(windows, segments, model)
        if not windows or features is None:
            return _blank('not_run', 'no window could be embedded', params)

        if k is not None:
            effective = min(k, len(windows))
            if effective == 1 or len(windows) < 2:
                labels, count = [0] * len(windows), 1
                confidence = {'method': 'fixed_k', 'selected_k': 1,
                              'requested_k': k, 'silhouette': None}
            else:
                labels, centers = _kmeans(features, effective)
                labels, count = _relabel_by_time(windows, labels)
                confidence = {
                    'method': 'fixed_k', 'selected_k': count, 'requested_k': k,
                    'silhouette': _round(_silhouette(features, labels)),
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
                best, candidates = _choose_clusters(features)
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
