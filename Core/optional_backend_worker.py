"""Optional process entry point. Never installs dependencies or downloads weights."""
from contextlib import contextmanager
import csv
import os
import types
import json
import math
from pathlib import Path
import sys


PANNS_COST = {'approximate_checkpoint_mb': 327, 'minimum_checkpoint_bytes': 300000000,
              'automatic_download': False, 'device': 'cpu', 'optional': True}
PANNS_COST_NOTICE = 'Optional PANNs Cnn14 weights are approximately 327 MB; supply local weights explicitly. No automatic download.'


@contextmanager
def _no_model_downloads():
    """The upstream PANNs package uses os.system(wget); disallow it in this worker."""
    original = os.system
    def blocked(command):
        raise RuntimeError('PANNs automatic download disabled; provide explicit local weights/labels')
    os.system = blocked
    try:
        yield
    finally:
        os.system = original


def _install_local_panns_config(labels, ids):
    """Use validated explicit labels without importing upstream home/download config."""
    config = types.ModuleType('panns_inference.config')
    config.sample_rate = 32000
    config.labels, config.ids, config.classes_num = list(labels), list(ids), len(labels)
    config.lb_to_ix = {label: i for i, label in enumerate(labels)}
    config.ix_to_lb = dict(enumerate(labels))
    config.id_to_ix = {label_id: i for i, label_id in enumerate(ids)}
    config.ix_to_id = dict(enumerate(ids))
    sys.modules['panns_inference.config'] = config

GENRES = {'Pop music', 'Hip hop music', 'Rock music', 'Heavy metal', 'Punk rock', 'Grunge',
          'Progressive rock', 'Rock and roll', 'Psychedelic rock', 'Rhythm and blues', 'Soul music',
          'Reggae', 'Country', 'Funk', 'Folk music', 'Jazz', 'Disco', 'Classical music',
          'Opera', 'Electronic music', 'House music', 'Techno', 'Dubstep', 'Drum and bass',
          'Electronica', 'Electronic dance music', 'Ambient music', 'Trance music',
          'Music of Latin America', 'Salsa music', 'Flamenco', 'Blues', 'New-age music',
          'Gospel music', 'Music of Africa', 'Afrobeat', 'Christian music', 'Bluegrass'}
MOODS = {'Funny music', 'Happy music', 'Sad music', 'Tender music', 'Exciting music', 'Angry music', 'Scary music'}


def classify(labels, scores, music_labels=(), genre_threshold=.05, genre_top_k=3):
    if not math.isfinite(genre_threshold) or not 0 <= genre_threshold <= 1 or not 1 <= genre_top_k <= 10:
        raise ValueError("genre threshold must be 0..1 and top-k 1..10")
    tags = sorted([dict(label=l, score=float(s)) for l, s in zip(labels, scores)
                   if math.isfinite(float(s))], key=lambda x: x['score'], reverse=True)
    def field(names=None, threshold=.1, max_items=10):
        items = [x for x in tags if x['score'] >= threshold and
                 (x['label'] in names if names is not None else x['label'] not in GENRES | MOODS | set(music_labels)
                  and 'music' not in x['label'].lower())][:max_items]
        return dict(status='complete', backend='panns', model='Cnn14', experimental=True,
                    params={'threshold': threshold, 'max_items': max_items}, items=items)
    genre, mood, events = field(GENRES, genre_threshold, genre_top_k), field(MOODS), field()
    return dict(genre=genre, mood=mood, events=events, top_tags=tags[:10],
                description='全混音 AudioSet 候选标签（非确定结论）：' +
                '、'.join(f"{x['label']} {x['score']:.2f}" for x in tags[:5]) + '。')


def _timeline_windows(duration, shots, window_seconds=5.0):
    windows = []
    for shot in shots or []:
        if shot.get('start') is None or shot.get('end') is None:
            continue
        start, end = max(0., float(shot['start'])), min(duration, float(shot['end']))
        if math.isfinite(start) and math.isfinite(end) and end > start:
            windows.append((start, end))
    return windows or [(float(start), min(duration, float(start) + window_seconds))
                       for start in __import__('numpy').arange(0, duration, window_seconds)]


def _aggregate_intervals(labels, scores, intervals, shots, duration, genre_threshold, genre_top_k):
    """Overlap-weighted means; intervals always describe actual, unpadded audio."""
    import numpy as np
    arr = np.asarray(scores)
    intervals = np.asarray(intervals)
    if arr.ndim != 2 or arr.shape[1] != len(labels) or not len(arr):
        raise ValueError('unexpected or empty timeline output shape')
    if intervals.shape != (len(arr), 2) or not np.isfinite(arr).all():
        raise ValueError('invalid timeline intervals or scores')
    items = []
    for start, end in _timeline_windows(duration, shots):
        weights = np.maximum(0, np.minimum(intervals[:, 1], end) - np.maximum(intervals[:, 0], start))
        if not weights.sum():
            raise ValueError('timeline interval has no inference coverage')
        mean = np.average(arr, axis=0, weights=weights)
        c = classify(labels, mean, labels[137:283], genre_threshold, genre_top_k)
        items.append(dict(start=round(start, 3), end=round(end, 3),
                          genre=c['genre']['items'], mood=c['mood']['items'],
                          events=c['events']['items'], label_energy_rms=float(np.sqrt(np.mean(mean * mean)))))
    return dict(status='complete', backend='panns', items=items, experimental=True,
                params={'threshold': genre_threshold, 'top_k': genre_top_k},
                description='全混音候选标签的时间聚合；label_energy_rms 仅为未校准标签分数 RMS。')


def _aggregate_framewise(labels, framewise, sample_rate, shots, duration,
                         genre_threshold=.05, genre_top_k=3, intervals=None):
    import numpy as np
    arr = np.asarray(framewise)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or not len(arr):
        raise ValueError('unexpected framewise_output shape')
    step = duration / len(arr)
    uniform = intervals is None
    if intervals is None:
        edges = np.linspace(0, duration, len(arr) + 1)
        intervals = np.column_stack((edges[:-1], edges[1:]))
    result = _aggregate_intervals(labels, arr, intervals, shots, duration, genre_threshold, genre_top_k)
    result.update(method='framewise', model='Cnn14_DecisionLevelMax')
    result['params'].update(aggregation='mean_framewise_per_shot',
                            frame_seconds=step if uniform else None,
                            window_seconds=5.0 if not shots else None, sample_rate=sample_rate)
    return result


def _rms(audio):
    import numpy as np
    audio = np.asarray(audio, dtype=float)
    return float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0


def _audio_features(audio, sr, intervals, label_scores, onset_std_multiplier=2.5):
    """Cheap, deterministic features used only for boundary detection."""
    import numpy as np
    rows = []
    for (start, end), tags in zip(intervals, label_scores):
        a = np.asarray(audio[int(start * sr):max(int(end * sr), int(start * sr) + 1)], dtype=float)
        if not a.size: a = np.zeros(1)
        rms = _rms(a)
        spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
        freqs = np.fft.rfftfreq(len(a), 1 / sr)
        total = float(spec.sum()) or 1.0
        centroid = float((freqs * spec).sum() / total) / max(sr / 2, 1)
        flux = float(np.mean(np.abs(np.diff(spec)))) / (float(np.mean(spec)) + 1e-8)
        onset_density = float(np.sum(np.abs(np.diff(a)) > (np.std(a) * onset_std_multiplier + 1e-8))) / max(end - start, .001)
        rows.append(np.r_[rms, flux, centroid, onset_density, np.asarray(tags, dtype=float)])
    return np.asarray(rows, dtype=float)


def _novelty_peaks(features, times, threshold=.65, min_distance=10.0,
                   min_segment_seconds=4., duration=None, min_novelty=.1,
                   normalization_percentile=95., std_floor=1e-6, details=False, cutoff_percentile=75.):
    """Group-balanced novelty; tuple output retained for existing callers."""
    import numpy as np
    if len(features) < 2: return []
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('features must be a finite matrix')
    times = np.asarray(times, dtype=float)
    if len(times) != len(x) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('times must be finite and strictly increasing')
    x = (x - x.mean(0)) / np.maximum(x.std(0), std_floor)
    delta = np.diff(x, axis=0)
    groups = {'energy': [0], 'rhythm': [3] if x.shape[1] > 3 else []}
    contributions = {k: np.r_[0., np.mean(delta[:, cols] ** 2, axis=1)]
                     for k, cols in groups.items() if cols}
    # Each spectral scalar and the mean label-vector change is one timbre component.
    # Repeating labels cannot drown out energy/rhythm or dilute spectral evidence.
    timbre_parts = [delta[:, i] ** 2 for i in (1, 2) if i < x.shape[1]]
    if x.shape[1] > 4:
        timbre_parts.append(np.mean(delta[:, 4:] ** 2, axis=1))
    if timbre_parts:
        contributions['timbre'] = np.r_[0., np.mean(timbre_parts, axis=0)]
    novelty = np.sqrt(sum(contributions.values()))
    raw = np.r_[0., np.linalg.norm(delta, axis=1)]
    scale = float(np.percentile(novelty, cutoff_percentile))
    cutoff = max(min_novelty, float(np.mean(novelty) + threshold * (scale - np.mean(novelty))))
    normalizer = max(float(np.percentile(novelty, normalization_percentile)), min_novelty, std_floor)
    candidates = [i for i in range(1, len(novelty) - 1)
                  if novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]
                  and novelty[i] >= cutoff and times[i] >= min_segment_seconds
                  and (duration is None or duration - times[i] >= min_segment_seconds)]
    chosen = []
    for i in sorted(candidates, key=lambda j: novelty[j], reverse=True):
        if all(abs(times[i] - times[j]) >= max(min_distance, min_segment_seconds) for j in chosen):
            chosen.append(i)
    peaks = []
    for i in sorted(chosen):
        total = sum(v[i] for v in contributions.values())
        shares = {k: float(v[i] / total) for k, v in contributions.items()}
        peaks.append(dict(time=float(times[i]), magnitude=float(np.clip(novelty[i] / normalizer, 0, 1)),
                          magnitude_raw=float(raw[i]), novelty=float(novelty[i]),
                          contributions=shares, type=max(shares, key=shares.get)))
    return peaks if details else [(p['time'], p['magnitude']) for p in peaks]


def _build_two_layer_timeline(audio, sr, labels, frame_scores, duration, shots,
                              genre_threshold=.05, genre_top_k=3, bucket_seconds=25.,
                              window_seconds=1., min_turn_seconds=10., turn_threshold=.65,
                              min_segment_seconds=4., min_novelty=.1,
                              normalization_percentile=95., std_floor=1e-6, frame_intervals=None,
                              cutoff_percentile=75., onset_std_multiplier=2.5):
    import numpy as np
    params = dict(bucket_seconds=bucket_seconds, window_seconds=window_seconds,
                  min_turn_seconds=min_turn_seconds, threshold=turn_threshold,
                  min_segment_seconds=min_segment_seconds, min_novelty=min_novelty,
                  normalization_percentile=normalization_percentile, std_floor=std_floor,
                  cutoff_percentile=cutoff_percentile, onset_std_multiplier=onset_std_multiplier)
    if any(not math.isfinite(v) or v <= 0 for k, v in params.items() if k != 'threshold') or not math.isfinite(turn_threshold) or turn_threshold < 0 or normalization_percentile > 100 or cutoff_percentile > 100:
        raise ValueError('invalid structure parameters')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be positive')
    params.update(magnitude_method='clip(group_balanced_novelty / full_track_percentile, 0, 1)',
                  magnitude_raw_method='legacy_standardized_feature_euclidean_norm',
                  contribution_method='group_mean_squared_delta_share_timbre_three_components',
                  feature_groups={'energy': 'audio_rms', 'timbre': 'spectral_flux+centroid+panns_tags',
                                  'rhythm': 'onset_density_proxy'})
    n = max(1, int(math.ceil(duration / window_seconds)))
    edges = np.linspace(0, duration, n + 1)
    intervals = list(zip(edges[:-1], edges[1:]))
    score = np.asarray(frame_scores, dtype=float)
    frame_edges = np.linspace(0, duration, len(score) + 1)
    source_intervals = np.asarray(frame_intervals) if frame_intervals is not None else np.column_stack((frame_edges[:-1], frame_edges[1:]))
    tags = [np.average(score, axis=0, weights=np.maximum(0, np.minimum(source_intervals[:, 1], e) - np.maximum(source_intervals[:, 0], s)))
            if score.ndim == 2 and len(score) else np.zeros(len(labels)) for s, e in intervals]
    features = _audio_features(audio, sr, intervals, tags, onset_std_multiplier)
    peaks = _novelty_peaks(features, [s for s, _ in intervals], turn_threshold, min_turn_seconds,
                           min_segment_seconds, duration, min_novelty, normalization_percentile,
                           std_floor, details=True, cutoff_percentile=cutoff_percentile)
    boundaries = [0.] + [p['time'] for p in peaks] + [duration]
    def segment(start, end):
        weights = np.array([max(0., min(e, end) - max(s, start)) for s, e in intervals])
        mean = np.average(tags, axis=0, weights=weights)
        c = classify(labels, mean, labels[137:283], genre_threshold, genre_top_k)
        energy = _rms(audio[int(start * sr):max(int(end * sr), int(start * sr) + 1)])
        return dict(start=round(start, 3), end=round(end, 3), genre=c['genre']['items'], mood=c['mood']['items'],
                    events=c['events']['items'], energy=energy, energy_type='audio_rms')
    overview = [segment(s, min(duration, s + bucket_seconds)) for s in np.arange(0, duration, bucket_seconds)]
    segments = [segment(a, b) for a, b in zip(boundaries[:-1], boundaries[1:])]
    turning = []
    for i, peak in enumerate(peaks):
        before, after = segments[i:i + 2]
        changed = [(before[k][0]['label'], after[k][0]['label']) for k in ('genre', 'mood')
                   if before[k] and after[k] and before[k][0]['label'] != after[k][0]['label']]
        turning.append(dict(peak, time=round(peak['time'], 3),
                            type='semantic' if changed else peak['type'],
                            physical_type=peak['type'],
                            **{'from': changed[0][0] if changed else None,
                               'to': changed[0][1] if changed else None},
                            status='complete', backend='librosa+panns',
                            method='group_balanced_novelty', params=params.copy(), experimental=True))
    return dict(overview=dict(status='complete', backend='librosa+panns', method='coarse_audio_buckets',
                              params={'bucket_seconds': bucket_seconds}, experimental=True, items=overview),
                structure=dict(status='complete', backend='librosa+panns', method='novelty_peak_segmentation',
                               params=params, experimental=True, segments=segments, no_significant_turns=not bool(turning)),
                turning_points=turning, no_significant_turns=not bool(turning))


def _sliding_timeline(model, audio, sr, labels, shots, genre_threshold=.05, genre_top_k=3,
                      window_seconds=5., step_seconds=2.):
    import numpy as np
    scores, intervals = [], []
    for offset in range(0, len(audio), int(sr * step_seconds)):
        chunk = audio[offset:offset + int(sr * window_seconds)]
        intervals.append((offset / sr, (offset + len(chunk)) / sr))
        chunk = np.pad(chunk, (0, max(0, sr - len(chunk))))
        clip, _ = model.inference(chunk[None, :])
        scores.append(clip[0])
    result = _aggregate_intervals(labels, scores, intervals, shots, len(audio) / sr,
                                  genre_threshold, genre_top_k)
    result.update(method='sliding_window_clipwise', model='Cnn14')
    result['params'].update(aggregation='sliding_window_clipwise', window_seconds=window_seconds,
                            step_seconds=step_seconds, sample_rate=sr)
    return result


def _music_timeline(audio, sr, labels, shots, clip_model, sed_checkpoint,
                    genre_threshold=.05, genre_top_k=3, structure_params=None):
    import numpy as np
    structure_params = structure_params or {}
    reasons = []
    try:
        if not sed_checkpoint.is_file() or sed_checkpoint.stat().st_size < 300000000:
            raise FileNotFoundError('SED checkpoint missing or incomplete; no download attempted')
        with _no_model_downloads():
            from panns_inference import SoundEventDetection
            sed = SoundEventDetection(checkpoint_path=str(sed_checkpoint), device='cpu')
        model_sr = getattr(sed.model, 'sample_rate', sr)
        if model_sr != sr:
            raise ValueError(f'SED sample rate {model_sr} differs from audio {sr}')
        frames, intervals, frame_steps = [], [], []
        for offset in range(0, len(audio), sr * 10):
            chunk = audio[offset:offset + sr * 10]
            actual_end = (offset + len(chunk)) / sr
            padded = np.pad(chunk, (0, max(0, sr - len(chunk))))
            fw = np.asarray(sed.inference(padded[None, :]))
            if fw.ndim != 3 or fw.shape[0] != 1 or fw.shape[2] != len(labels) or not fw.shape[1]:
                raise ValueError('unexpected SED framewise_output shape')
            # Each chunk has its own time grid: never stretch a padded tail over the whole video.
            frame_steps.append(len(padded) / sr / fw.shape[1])
            edges = offset / sr + np.arange(fw.shape[1] + 1) * frame_steps[-1]
            keep = edges[:-1] < actual_end
            frames.append(fw[0][keep])
            intervals.extend(zip(edges[:-1][keep], np.minimum(edges[1:][keep], actual_end)))
        frame_scores = np.concatenate(frames)
        result = _aggregate_framewise(labels, frame_scores, sr, shots, len(audio) / sr,
                                      genre_threshold, genre_top_k, intervals=intervals)
        result['params'].update(chunk_seconds=10, frame_seconds_per_chunk=frame_steps,
                                time_mapping='chunk_local_uniform_grid_padding_clipped')
        result['items'] = [dict(x, energy=_rms(audio[int(x['start']*sr):max(int(x['end']*sr), int(x['start']*sr)+1)]), energy_type='audio_rms') for x in result['items']]
        result['two_layer'] = _build_two_layer_timeline(audio, sr, labels, frame_scores, len(audio) / sr, shots,
                                                        genre_threshold, genre_top_k, frame_intervals=intervals, **structure_params)
        for layer in ('overview', 'structure'):
            result['two_layer'][layer]['params']['label_source'] = 'framewise'
        return result
    except Exception as exc:
        reasons.append(f'SED: {type(exc).__name__}: {exc}')
    try:
        result = _sliding_timeline(clip_model, audio, sr, labels, shots, genre_threshold, genre_top_k)
        result['fallback_reason'] = '; '.join(reasons)
        # Sliding scores are still sufficient for honest coarse/structure metadata.
        scores = []
        for offset in range(0, len(audio), int(sr * 2.)):
            chunk = audio[offset:offset + int(sr * 5.)]
            padded = np.pad(chunk, (0, max(0, sr - len(chunk))))
            scores.append(clip_model.inference(padded[None, :])[0][0])
        result['two_layer'] = _build_two_layer_timeline(audio, sr, labels, scores, len(audio) / sr, shots,
                                                        genre_threshold, genre_top_k,
                                                        frame_intervals=[(o / sr, min(len(audio) / sr, o / sr + 5.)) for o in range(0, len(audio), int(sr * 2.))],
                                                        **structure_params)
        for layer in ('overview', 'structure'):
            result['two_layer'][layer]['params']['label_source'] = 'sliding_window_clipwise'
        return result
    except Exception as exc:
        reasons.append(f'sliding window: {type(exc).__name__}: {exc}')
        return dict(status='not_run', backend='panns', method=None, items=[], experimental=True,
                    reason='; '.join(reasons))

def _trend(timeline):
    parts = []
    for x in timeline:
        labels = [i['label'] for i in (x.get('genre') or [])[:1] + (x.get('mood') or [])[:1]]
        if not labels: labels = ['未分类音乐'] if (x.get('events') or []) else ['无显著标签']
        parts.append(f"{x['start']:.0f}–{x['end']:.0f}s " + '+'.join(labels))
    return ' → '.join(parts)


def _trend_from_segments(segments):
    parts = []
    for x in segments:
        labels = [i['label'] for i in (x.get('genre') or [])[:1] + (x.get('mood') or [])[:1]]
        parts.append('+'.join(labels) if labels else ('未分类音乐' if x.get('events') else '无显著标签'))
    return ' → '.join(parts)

def panns(path, genre_threshold=.05, genre_top_k=3, shots_json=None):
    import os
    checkpoint = Path(os.environ.get('REELPACK_PANNS_CHECKPOINT', str(Path.home() / 'panns_data/Cnn14_mAP=0.431.pth'))).expanduser()
    labels_path = Path(os.environ.get('REELPACK_PANNS_LABELS', str(Path.home() / 'panns_data/class_labels_indices.csv'))).expanduser()
    # panns_inference itself downloads missing files, so guard before importing it.
    if not checkpoint.is_file() or checkpoint.stat().st_size < 300000000 or not labels_path.is_file():
        reason = 'PANNs checkpoint/labels missing or incomplete; no download attempted'
        return dict(status='not_run', reason=reason, trend=None,
                    impact='No PANNs semantic candidates are measured; physical descriptors are retained.',
                    remedy='Provide explicit local REELPACK_PANNS_CHECKPOINT (~327 MB) and REELPACK_PANNS_LABELS (527 AudioSet rows), then retry.',
                    cost=dict(PANNS_COST), cost_notice=PANNS_COST_NOTICE,
                    timeline=dict(status='not_run', backend='panns', method=None,
                                  reason=reason, items=[], overview={'status':'not_run'}, structure={'status':'not_run'},
                                  turning_points=[], experimental=True))
    with labels_path.open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    labels = [r['display_name'] for r in rows]
    ids = [r.get('mid', str(i)) for i, r in enumerate(rows)]
    if len(labels) != 527 or any(not label for label in labels):
        raise ValueError('expected 527 non-empty AudioSet labels')
    _install_local_panns_config(labels, ids)
    with _no_model_downloads():
        import torch
        from panns_inference import AudioTagging
    from audio_analyze import extract_audio
    torch.set_num_threads(2)
    audio, sr = extract_audio(path, 32000)
    if len(audio) == 0:
        raise ValueError('empty audio')
    with _no_model_downloads():
        model = AudioTagging(checkpoint_path=str(checkpoint), device='cpu')
    # Bounded CPU memory: non-overlapping 10s chunks. Preserve whole-mix clipwise output.
    import numpy as np
    scores = []
    with torch.no_grad():
        for offset in range(0, len(audio), sr * 10):
            chunk = audio[offset:offset + sr * 10]
            chunk = np.pad(chunk, (0, max(0, sr - len(chunk))))
            clip, _ = model.inference(chunk[None, :])
            scores.append(clip[0])
    result = dict(status='complete', backend='panns', model='Cnn14', experimental=True,
                 cost=dict(PANNS_COST), cost_notice=PANNS_COST_NOTICE,
                params={'sample_rate': sr, 'device': 'cpu', 'chunk_seconds': 10,
                        'aggregation': 'mean_clipwise', 'scope': 'whole_mix',
                        'scores_calibrated': False, 'labels_path': str(labels_path),
                         'checkpoint_path': str(checkpoint), 'automatic_download': False}, **classify(labels, np.mean(scores, axis=0), labels[137:283], genre_threshold, genre_top_k))
    sed_checkpoint = Path(os.environ.get('REELPACK_PANNS_SED_CHECKPOINT',
        str(Path.home() / 'panns_data/Cnn14_DecisionLevelMax_mAP=0.385.pth')))
    shots = json.loads(shots_json) if shots_json else []
    if isinstance(shots, dict):
        shots = shots.get('items', [])
    structure_params = json.loads(os.environ.get('REELPACK_MUSIC_STRUCTURE_PARAMS', '{}'))
    tl = _music_timeline(audio, sr, labels, shots, model, sed_checkpoint, genre_threshold, genre_top_k, structure_params)
    result['timeline'] = tl
    result['trend'] = _trend_from_segments(tl.get('two_layer', {}).get('structure', {}).get('segments', [])) if tl['status'] == 'complete' else None
    if tl.get('two_layer'):
        result['timeline']['overview'] = tl['two_layer']['overview']
        result['timeline']['structure'] = tl['two_layer']['structure']
        result['timeline']['segments'] = tl['two_layer']['structure']['segments']
        result['turning_points'] = tl['two_layer']['turning_points']
        result['timeline']['turning_points'] = tl['two_layer']['turning_points']
        result['timeline']['no_significant_turns'] = tl['two_layer']['no_significant_turns']
    result['params']['framewise'] = tl.get('method') == 'framewise'
    return result


def rapidocr(manifest):
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    items, errors = [], []
    frames = json.loads(Path(manifest).read_text())
    for frame in frames:
        try:
            rows, _ = engine(frame['path'])
            for box, text, confidence in rows or []:
                points = [[float(x), float(y)] for x, y in box]
                xs, ys = zip(*points)
                ts = frame['timestamp']
                items.append(dict(timestamp=ts, start=ts, end=ts, text=text,
                    region=dict(x=min(xs), y=min(ys), width=max(xs)-min(xs), height=max(ys)-min(ys),
                                polygon=points, units='pixels', origin='top_left'),
                    confidence=float(confidence), confidence_calibrated=False,
                    evidence_ids=frame['evidence_ids'], source='rapidocr', uncertain=float(confidence)<.6))
        except Exception as exc:
            errors.append(dict(timestamp=frame['timestamp'], reason=str(exc)))
    status = ('failed' if len(errors) == len(frames) else 'partial') if errors else 'complete'
    return dict(status=status, backend='rapidocr', model='RapidOCR bundled ONNX',
                params={'device': 'cpu'}, experimental=True, items=items, errors=errors,
                **({'reason': '; '.join(x['reason'] for x in errors)} if errors else {}))


if __name__ == '__main__':
    try:
        try:
            from runtime_environment import prepare_runtime_environment
        except ImportError:
            prepare_runtime_environment = None
        if prepare_runtime_environment is not None:
            os.environ.update(prepare_runtime_environment(create=os.environ.get('REELPACK_READONLY_CHECK') != '1'))
        print(json.dumps((panns(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else .05, int(sys.argv[4]) if len(sys.argv) > 4 else 3, sys.argv[5] if len(sys.argv) > 5 else None) if sys.argv[1] == 'panns' else rapidocr(sys.argv[2])), ensure_ascii=False, allow_nan=False))
    except (ImportError, ModuleNotFoundError) as exc:
        print(json.dumps(dict(status='not_run', reason=f'{type(exc).__name__}: {exc}')))
    except Exception as exc:
        print(json.dumps(dict(status='failed', reason=f'{type(exc).__name__}: {exc}')))
