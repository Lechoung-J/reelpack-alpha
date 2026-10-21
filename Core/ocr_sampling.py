"""Backend-independent OCR sampling and exact, region-aware occurrence merging.

No fuzzy text normalization: digits, versions and code remain byte-for-byte intact.
Unknown regions are never deduplicated. Reading reconstruction is conservative;
scrolling/character animation detection is not claimed without backend observations.
"""
import copy
import math


def sampling_candidates(duration, segments=(), interval=1.0, change_timestamps=()):
    if not math.isfinite(duration) or duration < 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError('invalid sampling range')
    candidates = {}
    def add(t, reason):
        t = float(t)
        if math.isfinite(t) and 0 <= t < duration:
            candidates.setdefault(round(t, 6), set()).add(reason)
    for index in range(math.ceil(duration / interval)):
        add(index * interval, 'independent_interval')
    for segment in segments:
        start, end = float(segment['start']), float(segment['end'])
        for t in (start, (start + end) / 2, end):
            add(t, 'transcript_candidate')
    for t in change_timestamps:
        add(t, 'text_change_candidate')
    return {"strategy": "independent_interval+transcript+supplied_text_changes", "interval_seconds": interval,
            "independent_sampling": True, "text_change_detection": "not_run",
            "candidates": [{"timestamp": t, "reasons": sorted(reasons)} for t, reasons in sorted(candidates.items())]}


def merge_observations(observations, max_gap=1.5):
    """Merge only adjacent exact observations in the same known region AND segment.

    Keep each observation's timestamp and evidence refs in changes/occurrences.
    Cross-segment reappearances remain separate records, including identical text.
    """
    output = []
    for original in sorted(observations, key=lambda x: x.get('start', x.get('timestamp', 0))):
        item = copy.deepcopy(original)
        start = item.get('start', item.get('timestamp'))
        item['start'] = start
        item.setdefault('end', start)
        item['occurrences'] = [{"start": start, "end": item['end'], "evidence_ids": item.get('evidence_ids', [])}]
        prior = output[-1] if output else None
        if (prior and item.get('region') is not None and prior.get('region') == item['region']
                and prior['text'] == item['text'] and prior.get('segment_id') == item.get('segment_id')
                and prior.get('shot_id') == item.get('shot_id') and 0 <= start - prior['end'] <= max_gap):
            prior['end'] = max(prior['end'], item['end'])
            prior['occurrences'].extend(item['occurrences'])
            prior['evidence_ids'] = list(dict.fromkeys(prior.get('evidence_ids', []) + item.get('evidence_ids', [])))
        else:
            item.setdefault('ocr_id', f'ocr_{len(output)+1:04d}')
            output.append(item)
    return {"items": output, "reading_text": '\n'.join(i['text'] for i in output),
            "reading_reconstruction": {"status": "partial", "strategy": "exact_observation_sequence",
                                       "scrolling_alignment": "not_run", "changes_preserved": True}}
