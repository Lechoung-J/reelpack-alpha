"""Shared shot sampling; preserve endpoints/midpoint without visual deduplication."""
import math


def shot_frame_positions(first, last, fps, interval=2.0):
    if fps <= 0 or interval <= 0 or last <= first:
        raise ValueError('invalid shot sampling bounds')
    # last is exclusive: never sample the next shot's first frame.
    anchors = sorted(set((first, (first + last - 1) // 2, last - 1)))
    step = max(1, math.floor(min(interval, 2.0) * fps))
    positions = set(anchors)
    if (last - first) / fps > 2:
        for left, right in zip(anchors, anchors[1:]):
            positions.update(range(left, right, step))
    return sorted(positions)


def mark_speech_frames(evidence, segments):
    for frame in evidence:
        ts = frame.get('timestamp')
        if frame.get('kind') in {'frame', 'image'} and ts is not None:
            speech = any(s.get('text', '').strip() and s['start'] <= ts < s['end'] for s in segments)
            frame['keyframe'] = speech
            if speech:
                frame['purpose'] = 'shot_representative speech_keyframe'
