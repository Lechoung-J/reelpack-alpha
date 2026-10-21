"""Cut/beat measurements. Positive offset means the cut follows the beat."""
import math


def align_cuts_to_beats(cuts, beats, tolerance=.15):
    """Preserve the original nearest-beat calculation, including a cut at 0 s."""
    cuts, beats = [float(x) for x in cuts], [float(x) for x in beats]
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and non-negative')
    if not all(math.isfinite(x) for x in cuts + beats):
        raise ValueError('cut and beat times must be finite')
    offsets = [min((cut - beat for beat in beats), key=abs) for cut in cuts] if beats else []
    measured = bool(beats) and bool(cuts)
    aligned = sum(abs(x) <= tolerance for x in offsets) if beats else None
    result = {
        'status': 'measured' if measured else 'not_measured',
        'method': 'nearest_detected_beat',
        'params': {'tolerance_seconds': tolerance, 'offset_definition': 'cut_time - nearest_beat_time',
                   'tie_break': 'first beat in supplied order', 'boundary_inclusive': True},
        'units': {'times': 'seconds', 'offsets': 'seconds', 'aligned_ratio': 'fraction'},
        'cut_count': len(cuts), 'beat_count': len(beats), 'tolerance_seconds': tolerance,
        'aligned_cut_count': aligned, 'aligned_ratio': aligned / len(cuts) if measured else None,
        'mean_signed_offset_seconds': sum(offsets) / len(offsets) if offsets else None,
        'mean_absolute_offset_seconds': sum(map(abs, offsets)) / len(offsets) if offsets else None,
        'offsets_seconds': offsets,
        'includes_media_start': any(x == 0 for x in cuts),
        'cut_source': 'shots.start, including media_start when supplied',
        'data': '统计来自 shots.start 与最近拍点；沿用包括媒体起点的计数口径。',
        'inference': '对齐比例只描述时间接近程度，不证明创作意图或音乐身份。',
    }
    if not measured:
        result.update(reason='未测得拍点 (beats not measured); no detected beats; alignment not measured' if not beats else 'no supplied cuts; alignment not measured',
                      impact='No cut/beat synchrony ratio can be inferred.',
                      remedy='Inspect beat detection or supply cut times; do not replace missing measurements with zero.')
    return result
