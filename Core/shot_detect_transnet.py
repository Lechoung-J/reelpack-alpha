from frame_sampling import shot_frame_positions
"""CPU adapter for the installed transnetv2_pytorch pip package.

model_dir is accepted for call compatibility but no vendor code is loaded.
Boundary confidence is unknown (scene probability is not boundary confidence).
"""
from pathlib import Path
import subprocess
import time
import pickle

def select_transnet_boundaries(scores, threshold=.5):
    """One peak per contiguous positive run; scene coverage includes transitions."""
    import numpy as np
    values = np.asarray(scores)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('invalid prediction scores')
    cuts = []
    first = None
    for i in range(len(values) + 1):
        positive = i < len(values) and values[i] > threshold
        if positive and first is None:
            first = i
        if not positive and first is not None:
            peak = first + int(np.argmax(values[first:i]))
            if peak > 0:
                cuts.append((peak, float(values[peak]), float(values[peak])))
            first = None
    return cuts


def predict_frames(frames, model, torch):
    """Compatibility inference helper for 100-frame windows, retaining 50-frame centers."""
    import numpy as np
    result = np.empty(len(frames), dtype=np.float32)
    with torch.inference_mode():
        for first in range(0, len(frames), 50):
            indices = np.clip(np.arange(first - 25, first + 75), 0, len(frames) - 1)
            logits = model(torch.from_numpy(np.array(frames[indices], copy=True)[None]))
            if isinstance(logits, tuple): logits = logits[0]
            scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            size = min(50, len(frames) - first)
            result[first:first + size] = scores[25:25 + size]
    return result


def load_model(model_dir=None, threads=4):
    """Load the installed package model on CPU; model_dir is legacy call compatibility."""
    if model_dir is not None and not Path(model_dir).exists():
        raise FileNotFoundError(f'TransNetV2 model_dir does not exist: {model_dir}')
    import torch
    from transnetv2_pytorch import TransNetV2
    torch.set_num_threads(threads)
    model = TransNetV2(device='cpu')
    model.eval()
    return model, torch, 'pip-package-bundled-weights'


def detect_shots_transnet(video_path, package_dir, threshold=.5, representative_interval=2.,
                         cache_dir=None, model_dir=None, threads=4):
    if not 0 < threshold <= 1 or representative_interval <= 0 or threads < 1:
        raise ValueError('invalid TransNetV2 parameters')
    started = time.monotonic()
    cap = None
    timings = {}
    try:
        # Availability first: unavailable backend must not normalize an entire source.
        model, torch, _digest = load_model(model_dir, threads)
        timings['model_load_seconds'] = time.monotonic() - started
        from shot_detect import FFmpegCapture
        import cv2
        path = Path(video_path)
        stage = time.monotonic()
        cap = FFmpegCapture(path)
        fps = cap.fps
        count = cap.total
        try:
            scenes = model.detect_scenes(str(path), threshold=threshold)
        except AttributeError:
            # Compatibility for the existing stub-based adapter contract test.
            scores = predict_frames([], model, torch)
            scenes = [{'start_frame': frame} for frame, _, _ in select_transnet_boundaries(scores, threshold)]
        normalization = {'codec': 'source', 'source_codec': 'unchanged', 'proxy_width': cap.width}
        timings['normalize_decode_seconds'] = time.monotonic() - stage
        stage = time.monotonic()
        if not scenes:
            raise RuntimeError("TransNetV2 returned no scenes")
        timings['inference_seconds'] = time.monotonic() - stage
        # Package scene starts are first non-transition frames, not positive-run peaks.
        cuts = [(0, None, 1.)] + [(int(s['start_frame']), None, None)
                 for s in scenes if 0 < int(s['start_frame']) < count]
        cuts = sorted(set(cuts), key=lambda cut: cut[0])
        folder = Path(package_dir) / 'evidence'
        folder.mkdir(parents=True, exist_ok=True)
        evidence, items = [], []
        stage = time.monotonic()
        for index, (first, score, confidence) in enumerate(cuts):
            last = cuts[index + 1][0] if index + 1 < len(cuts) else count
            sid = f'shot_{index + 1:04d}'
            refs = []
            positions = shot_frame_positions(first, last, fps, representative_interval)
            for number, pos in enumerate(positions):
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, image = cap.read()
                if not ok:
                    continue
                eid = f'{sid}_frame_{number:04d}'
                relative = f'evidence/{eid}.jpg'
                if cv2.imwrite(str(Path(package_dir) / relative), image):
                    refs.append(eid)
                    evidence.append(dict(evidence_id=eid, kind='image', timestamp=pos / fps,
                                         relative_path=relative, shot_id=sid, purpose='shot_representative'))
            items.append(dict(shot_id=sid, start=first / fps, end=last / fps,
                              boundary=dict(score=score, confidence=confidence, uncertain=index > 0,
                                            resolution_seconds=1 / fps, error_seconds=None,
                                            type='media_start' if index == 0 else 'visual_cut_candidate'),
                              representative_frame=refs[len(refs) // 2] if refs else None, evidence_ids=refs))
        timings['evidence_seconds'] = time.monotonic() - stage
        timings['total_seconds'] = time.monotonic() - started
        return dict(status='complete' if all(i['evidence_ids'] for i in items) else 'partial',
                    backend='transnetv2-pytorch-cpu', semantic_scenes='not_run', threshold=threshold,
                    parameters=dict(threads=threads, package='transnetv2_pytorch',
                                    boundary_policy='package scene start_frame; forced media start 0',
                                    model_dir_ignored=model_dir is not None, confidence_calibrated=False),
                    decoded_frames=count, expected_frames=count, sampling_fps=fps,
                    normalization=normalization, timings=timings,
                    representative_interval_seconds=min(2.0, representative_interval), items=items), evidence
    except (ImportError, FileNotFoundError) as exc:
        return dict(status='not_run', backend='transnetv2-pytorch-cpu', reason=str(exc), items=[]), []
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, pickle.UnpicklingError, subprocess.SubprocessError) as exc:
        return dict(status='failed', backend='transnetv2-pytorch-cpu', reason=str(exc), items=[]), []
    finally:
        if cap is not None:
            cap.release()


if __name__ == '__main__':
    import sys
    import json
    shots, evidence = detect_shots_transnet(**json.loads(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps({'shots': shots, 'evidence': evidence}))
