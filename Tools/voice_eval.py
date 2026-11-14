#!/usr/bin/env python3
"""Known-answer speaker-clustering evaluation on two audio files.

Each input is cut into fixed-length chunks, the chunks are interleaved as
A/B/A/B..., and the resulting sample is clustered. Because the source of every
chunk is known, the tool reports cluster purity and pairwise accuracy in one
command:

    python Tools/voice_eval.py speaker_a.wav speaker_b.wav
    python Tools/voice_eval.py speaker_a.wav speaker_b.wav --out-dir report-dir

Output is a single JSON object on stdout; ``--json-out PATH`` writes it to an
exact file and ``--out-dir DIR`` writes ``DIR/voice_eval.json`` (both may be
combined). Exit code is 0 only when the clustering ran and produced a complete
result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Core'))
from speaker_cluster import SAMPLE_RATE, cluster_speakers, decode_audio


def _assign_truth(windows, truth):
    """Label each clustered window by the truth interval it overlaps most."""
    labels = []
    for window in windows:
        best, best_overlap = None, 0.0
        for item in truth:
            overlap = max(0.0, min(window['end'], item['end'])
                          - max(window['start'], item['start']))
            if overlap > best_overlap:
                best, best_overlap = item['source'], overlap
        labels.append(best)
    return labels


def _metrics(predicted, truth):
    pairs = [(prediction, label) for prediction, label in zip(predicted, truth)
             if label is not None]
    if not pairs:
        return {'purity': None, 'pairwise_accuracy': None, 'pairwise_precision': None,
                'pairwise_recall': None, 'pairwise_f1': None,
                'pair_counts': {'same_same': 0, 'same_different': 0,
                                'different_same': 0, 'different_different': 0}}
    groups = {}
    for prediction, label in pairs:
        groups.setdefault(prediction, Counter())[label] += 1
    purity = sum(max(counts.values()) for counts in groups.values()) / len(pairs)

    same_same = same_different = different_same = different_different = 0
    for left in range(len(pairs)):
        for right in range(left + 1, len(pairs)):
            same_predicted = pairs[left][0] == pairs[right][0]
            same_actual = pairs[left][1] == pairs[right][1]
            if same_actual and same_predicted:
                same_same += 1
            elif same_actual:
                different_same += 1
            elif same_predicted:
                same_different += 1
            else:
                different_different += 1
    total = same_same + same_different + different_same + different_different
    precision = same_same / (same_same + same_different) if same_same + same_different else None
    recall = same_same / (same_same + different_same) if same_same + different_same else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {
        'purity': round(purity, 4),
        'pairwise_accuracy': round((same_same + different_different) / total, 4) if total else None,
        'pairwise_precision': None if precision is None else round(precision, 4),
        'pairwise_recall': None if recall is None else round(recall, 4),
        'pairwise_f1': None if f1 is None else round(f1, 4),
        'pair_counts': {'same_same': same_same, 'same_different': same_different,
                        'different_same': different_same, 'different_different': different_different},
    }


def _build_sample(first, second, *, chunk_seconds, gap_seconds, max_chunks, lead_seconds=0.3):
    import numpy as np

    if not 1.0 <= chunk_seconds <= 3.0:
        raise ValueError('chunk_seconds must be between 1.0 and 3.0 to match analysis windows')
    if gap_seconds < 0.35:
        raise ValueError('gap_seconds must be at least 0.35 so utterances stay separate')
    if max_chunks < 2:
        raise ValueError('max_chunks must be at least 2 to compare speaker groups')
    audio_a, sample_rate = decode_audio(first)
    audio_b, _ = decode_audio(second)
    shorter_seconds = min(len(audio_a), len(audio_b)) / sample_rate
    if shorter_seconds < 1.0:
        raise ValueError('each input must contain at least 1.0 s of audio')
    effective_chunk_seconds = min(chunk_seconds, shorter_seconds)
    chunk = min(int(round(effective_chunk_seconds * sample_rate)),
                len(audio_a), len(audio_b))
    effective_chunk_seconds = chunk / sample_rate
    available = min(len(audio_a) // chunk, len(audio_b) // chunk)
    rounds = min(max_chunks, max(2, available))
    gap = np.zeros(int(round(gap_seconds * sample_rate)), dtype=np.float32)
    lead = np.zeros(int(round(lead_seconds * sample_rate)), dtype=np.float32)

    parts, truth = [lead], []
    cursor = len(lead)
    for round_index in range(rounds):
        source_chunk = round_index % available
        for label, audio in (('A', audio_a), ('B', audio_b)):
            start = cursor / sample_rate
            parts.append(audio[source_chunk * chunk:(source_chunk + 1) * chunk])
            cursor += chunk
            truth.append({'source': label, 'start': start, 'end': cursor / sample_rate})
            parts.append(gap)
            cursor += len(gap)
    return np.concatenate(parts), sample_rate, truth, rounds, round(effective_chunk_seconds, 3)


def evaluate(first, second, *, chunk_seconds=1.8, gap_seconds=0.4, max_chunks=8, k=None):
    """Build the known-answer sample, cluster it and return a JSON-safe report."""
    try:
        with tempfile.TemporaryDirectory(prefix='reelpack-voice-eval-') as directory:
            import soundfile as sf
            audio, sample_rate, truth, rounds, effective_chunk = _build_sample(
                first, second, chunk_seconds=chunk_seconds,
                gap_seconds=gap_seconds, max_chunks=max_chunks)
            sample = Path(directory) / 'known-answer.wav'
            sf.write(str(sample), audio, sample_rate, subtype='PCM_16')
            result = cluster_speakers(str(sample), k=k)

        base = {
            'inputs': {'first': str(first), 'second': str(second)},
            'params': {'chunk_seconds': chunk_seconds,
                       'effective_chunk_seconds': effective_chunk,
                       'gap_seconds': gap_seconds, 'rounds': rounds,
                       'max_chunks': max_chunks, 'k': k, 'sample_rate': SAMPLE_RATE},
            'backend': result.get('backend'),
        }
        if result.get('status') != 'complete':
            return dict(base, status=result.get('status', 'failed'),
                        reason=result.get('reason', 'clustering did not complete'),
                        window_count=0, cluster_count=0, purity=None,
                        pairwise_accuracy=None)
        predicted = [window['cluster'] for window in result['windows']]
        labels = _assign_truth(result['windows'], truth)
        metrics = _metrics(predicted, labels)
        return dict(
            base, status='complete',
            window_count=len(predicted),
            matched_window_count=sum(label is not None for label in labels),
            cluster_count=len(result['clusters']),
            silhouette=result.get('confidence', {}).get('silhouette'),
            **metrics,
            windows=[dict(window, source=label) for window, label in zip(result['windows'], labels)],
        )
    except Exception as exc:  # noqa: BLE001 - CLI tool reports instead of raising
        return {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}',
                'inputs': {'first': str(first), 'second': str(second)},
                'window_count': 0, 'cluster_count': 0,
                'purity': None, 'pairwise_accuracy': None}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Known-answer speaker-clustering evaluation (purity / pairwise accuracy).')
    parser.add_argument('first', help='audio file treated as speaker A')
    parser.add_argument('second', help='audio file treated as speaker B')
    parser.add_argument('--chunk-seconds', type=float, default=1.8,
                        help='utterance length in seconds (1.0-3.0, default 1.8)')
    parser.add_argument('--gap-seconds', type=float, default=0.4,
                        help='silence between utterances (>=0.35, default 0.4)')
    parser.add_argument('--max-chunks', type=int, default=8,
                        help='maximum number of chunks taken from each input (default 8)')
    parser.add_argument('--k', type=int, default=None, help='fixed cluster count; default auto')
    parser.add_argument('--json-out', help='also write the JSON report to this path')
    parser.add_argument('--out-dir', metavar='DIR',
                        help='also write the JSON report to DIR/voice_eval.json '
                             '(may be combined with --json-out)')
    arguments = parser.parse_args(argv)

    report = evaluate(arguments.first, arguments.second,
                      chunk_seconds=arguments.chunk_seconds,
                      gap_seconds=arguments.gap_seconds,
                      max_chunks=arguments.max_chunks, k=arguments.k)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if arguments.json_out:
        Path(arguments.json_out).write_text(text + '\n', encoding='utf-8')
    if arguments.out_dir:
        out_dir = Path(arguments.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'voice_eval.json').write_text(text + '\n', encoding='utf-8')
    return 0 if report.get('status') == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
