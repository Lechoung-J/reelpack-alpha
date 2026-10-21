"""CPU OCR with explicit Chinese language selection and pixel word regions.

No downloads or silent language/backend fallback. TSV confidence is uncalibrated.
"""
import csv
import io
import os
import subprocess
import tempfile
from pathlib import Path


def parse_tsv(tsv, timestamp, evidence_ids=()):
    items = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter='\t'):
        if row.get('level') != '5' or not (row.get('text') or '').strip():
            continue
        confidence = float(row['conf'])
        if confidence < 0:
            continue
        items.append({'timestamp': timestamp, 'start': timestamp, 'end': timestamp,
                      'text': row['text'], 'region': {'x': int(row['left']), 'y': int(row['top']),
                      'width': int(row['width']), 'height': int(row['height']), 'units': 'pixels',
                      'origin': 'top_left'}, 'confidence': confidence / 100,
                      'confidence_calibrated': False, 'evidence_ids': list(evidence_ids),
                      'source': 'tesseract_word', 'uncertain': confidence < 60})
    return items


def recognize_frames(frames, evidence_ids=None, runner=None):
    runner = runner or subprocess.run
    language = os.environ.get('REELPACK_TESSERACT_LANG', 'chi_sim+eng')
    binary = os.environ.get('REELPACK_TESSERACT_BIN', 'tesseract')
    result = {'status': 'complete', 'backend': 'tesseract', 'model': language,
              'params': {'language': language, 'psm': 11, 'oem': 1}, 'items': [], 'errors': []}
    with tempfile.TemporaryDirectory(prefix='reelpack-tesseract-') as td:
        for index, (timestamp, image) in enumerate(frames):
            try:
                path = Path(td) / f'{index}.jpg'
                path.write_bytes(image)
                run = runner([binary, str(path), 'stdout', '-l', language, '--oem', '1',
                              '--psm', '11', 'tsv'], capture_output=True, text=True,
                             check=True, timeout=120)
                result['items'].extend(parse_tsv(run.stdout, timestamp,
                    (evidence_ids or {}).get(timestamp, [])))
            except Exception as exc:
                detail = getattr(exc, 'stderr', '') or str(exc)
                result['errors'].append({'timestamp': timestamp, 'reason': f'{type(exc).__name__}: {detail}'})
    if result['errors']:
        result['status'] = 'failed' if len(result['errors']) == len(frames) else 'partial'
        result['reason'] = '; '.join(e['reason'] for e in result['errors'])
    return result
