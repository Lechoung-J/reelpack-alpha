"""Probe-driven, offline OCR selection with bounded per-frame error fallback."""
import json
from pathlib import Path
import subprocess
import tempfile

from ocr_probe import PROBE_TIMEOUT_SECONDS, normalize_probe, run_probe
from ocr_probe_ollama import OllamaOCR

STRATEGY_VERSION = 'rapid-probe-v2'
NATIVE_TIMEOUT_SECONDS = 300
FALLBACK_TIMEOUT_SECONDS = 120
NATIVE_MODEL = 'Apple Vision'
BACKEND_ORDER = ('rapidocr', 'macos', 'ollama')


def _frame(timestamp, refs, status, reason=None):
    return dict(timestamp=timestamp, status=status,
                evidence_ids=list((refs or {}).get(timestamp, [])), reason=reason)


def _item(timestamp, refs, text, backend, model, original=None):
    item = dict(original or {})
    item.update(timestamp=timestamp, start=timestamp, end=timestamp, text=text,
                evidence_ids=list((refs or {}).get(timestamp, [])), status='complete',
                backend=backend, model=model, source=backend, fallback_from=None, reason=None)
    return item


def _status(frames):
    statuses = [frame['status'] for frame in frames]
    if all(status == 'complete' for status in statuses):
        return 'complete'
    if 'complete' in statuses:
        return 'partial'
    return 'not_run' if all(status == 'not_run' for status in statuses) else 'failed'


def _native(frames, refs, ocr_bin, runner, timeout=NATIVE_TIMEOUT_SECONDS):
    attempt = dict(backend='macos', model=NATIVE_MODEL,
                   params={'timeout_seconds': timeout}, status='complete', frames=[], errors=[], reason=None)
    items, stderr = [], ''
    try:
        if not ocr_bin:
            raise FileNotFoundError('macOS OCR helper unavailable')
        with tempfile.TemporaryDirectory(prefix='reelpack-native-ocr-') as directory:
            paths = []
            for index, (_, image) in enumerate(frames):
                path = Path(directory) / f'frame_{index:03d}.jpg'
                path.write_bytes(image)
                paths.append(str(path))
            result = (runner or subprocess.run)([str(ocr_bin), '--json', *paths], capture_output=True,
                                                text=True, timeout=timeout)
            stderr = result.stderr.strip()
            if result.returncode:
                raise RuntimeError(f'macOS Vision OCR exit {result.returncode}: {stderr}')
            start = result.stdout.find('{')
            payload = json.loads(result.stdout[start:] if start >= 0 else result.stdout)
            rows = payload.get('frames') if isinstance(payload, dict) else None
            if not isinstance(rows, list) or len(rows) != len(frames):
                raise ValueError('macOS Vision OCR response frame count mismatch')
            if 'count' in payload and payload['count'] != len(frames):
                raise ValueError('macOS Vision OCR response count mismatch')
            for path, row in zip(paths, rows):
                if not isinstance(row, dict) or row.get('file') != path:
                    raise ValueError('macOS Vision OCR response frame path mismatch')
                if not row.get('error') and not isinstance(row.get('text'), str):
                    raise ValueError('macOS Vision OCR response text must be a string')
            for (timestamp, _), row in zip(frames, rows):
                reason = str(row['error']) if row.get('error') else None
                attempt['frames'].append(_frame(timestamp, refs, 'failed' if reason else 'complete', reason))
                if reason:
                    attempt['errors'].append(dict(backend='macos', timestamp=timestamp, reason=reason))
                else:
                    items.append(_item(timestamp, refs, row['text'], 'macos', NATIVE_MODEL))
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
        detail = getattr(exc, 'stderr', None) or stderr
        if detail and str(detail) not in reason:
            reason += f'; stderr: {detail}'
        status = 'not_run' if isinstance(exc, FileNotFoundError) else 'failed'
        attempt['frames'] = [_frame(ts, refs, status, reason) for ts, _ in frames]
        attempt['errors'] = [dict(backend='macos', reason=reason)]
        items = []
    attempt['status'] = _status(attempt['frames'])
    attempt['reason'] = '; '.join(error['reason'] for error in attempt['errors']) or None
    return items, attempt


def _response(frames, refs, callback, backend, timeout):
    try:
        response = callback(frames, refs, timeout)
        if not isinstance(response, dict) or response.get('status') not in ('complete', 'partial', 'failed', 'not_run'):
            raise ValueError(f'invalid {backend} response status')
        rows, errors = response.get('items', []), response.get('errors', [])
        if not isinstance(rows, list) or not isinstance(errors, list):
            raise ValueError(f'invalid {backend} response items/errors')
        timestamps = {ts for ts, _ in frames}
        for row in rows:
            if not isinstance(row, dict) or row.get('timestamp') not in timestamps or not isinstance(row.get('text'), str):
                raise ValueError(f'invalid {backend} response frame/text')
    except Exception as exc:
        response = dict(status='not_run' if isinstance(exc, (FileNotFoundError, ImportError)) else 'failed',
                        reason=f'{type(exc).__name__}: {exc}', items=[], errors=[])
        rows, errors = [], []
    model = response.get('model') or ('RapidOCR' if backend == 'rapidocr' else 'Ollama')
    reason = response.get('reason')
    normalized_errors = [dict(error, backend=backend) if isinstance(error, dict)
                         else dict(backend=backend, reason=str(error)) for error in errors]
    if response['status'] != 'complete' and not reason:
        reason = '; '.join(str(error.get('reason', error)) for error in normalized_errors) or f"{backend} {response['status']}"
    if reason and not normalized_errors:
        normalized_errors.append(dict(backend=backend, reason=reason))
    errors_by_ts = {error['timestamp']: str(error.get('reason') or reason or f'{backend} frame failed')
                    for error in normalized_errors if error.get('timestamp') in {ts for ts, _ in frames}}
    mapped_partial = response['status'] == 'partial' and normalized_errors and all(
        error.get('timestamp') in errors_by_ts for error in normalized_errors)
    attempt = dict(backend=backend, model=model, params=dict(response.get('params') or {}, timeout_seconds=timeout),
                   status=response['status'], frames=[], errors=normalized_errors, reason=reason)
    items = []
    for timestamp, _ in frames:
        frame_reason = errors_by_ts.get(timestamp)
        if frame_reason:
            status = 'failed'
        elif response['status'] == 'complete' or mapped_partial:
            status = 'complete'
        else:
            status = 'not_run' if response['status'] == 'not_run' else 'failed'
            frame_reason = reason
        attempt['frames'].append(_frame(timestamp, refs, status, frame_reason))
        if status == 'complete':
            for row in [row for row in rows if row['timestamp'] == timestamp] or [dict(text='')]:
                items.append(_item(timestamp, refs, row['text'], backend, model, row))
    attempt['status'] = _status(attempt['frames'])
    return items, attempt


def _rapid_default(frames, refs, timeout):
    from optional_backends import recognize_frames
    def bounded_runner(command, **kwargs):
        kwargs['timeout'] = timeout
        kwargs['env'] = {**kwargs.get('env', {}), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        command = list(command)
        if len(command) > 1 and Path(command[1]).name == 'optional_backend_worker.py':
            command[1] = str(Path(__file__).with_name('ocr_probe_worker.py'))
        return subprocess.run(command, **kwargs)
    return recognize_frames(frames, refs, runner=bounded_runner)


def recognize_auto(frames, refs, ocr_bin, runner=None, rapid=None, *, ocr_model=None,
                   ollama_base=None, ollama=None, probe=None, backend_order=BACKEND_ORDER) -> dict:
    """Probe once per candidate/run, select rapidocr→macos→local Ollama.

    Positional arguments remain compatible. Optional ``rapid``/``ollama`` are
    callable(frames, refs); ``probe(backend)`` and ``backend_order`` are test seams.
    Production probes run the same adapters on the built-in PNG with 10s budgets.
    A successful empty material result is final. Only unsuccessful frames move
    to another backend, at most one material attempt per backend. A temporarily
    failed probe can receive one material retry after a selected backend fails;
    missing backends and an all-failed probe run are never retried in a loop.
    """
    frames = list(frames)
    order = tuple(dict.fromkeys(backend_order))
    if not order or any(name not in BACKEND_ORDER for name in order):
        raise ValueError('invalid OCR backend order')
    params = dict(strategy_version=STRATEGY_VERSION, native_timeout_seconds=NATIVE_TIMEOUT_SECONDS,
                  fallback_timeout_seconds=FALLBACK_TIMEOUT_SECONDS, probe_timeout_seconds=PROBE_TIMEOUT_SECONDS,
                  fallback_on='frame_error_only', automatic_download=False, backend_order=list(order))
    base = dict(status='complete', backend='auto', model=None, params=params, items=[], errors=[],
                fallback_from=None, reason=None, attempts=[], probes=[])
    if not frames:
        return base
    local_ollama = OllamaOCR(ocr_model, ollama_base)
    rapid_call = (lambda selected, references, timeout: rapid(selected, references)) if rapid is not None else _rapid_default
    ollama_call = (lambda selected, references, timeout: ollama(selected, references)) if ollama is not None else local_ollama
    adapters = {
        'macos': lambda selected, references, timeout: _native(selected, references, ocr_bin, runner, timeout),
        'rapidocr': lambda selected, references, timeout: _response(selected, references, rapid_call, 'rapidocr', timeout),
        'ollama': lambda selected, references, timeout: _response(selected, references, ollama_call, 'ollama', timeout),
    }
    probes = {}
    def check(backend):
        if backend not in probes:
            if probe is None:
                record = run_probe(backend, adapters[backend])
            else:
                try:
                    record = normalize_probe(backend, probe(backend))
                except Exception as exc:
                    record = dict(backend=backend, status='failed', reason=f'{type(exc).__name__}: {exc}',
                                  text='', availability='available')
            probes[backend] = record
            base['probes'].append(record)
        return probes[backend]
    selected = next((name for name in order if check(name)['status'] == 'complete'), None)
    if selected is None:
        base.update(status='not_run' if all(p['status'] == 'not_run' for p in probes.values()) else 'failed',
                    reason='; '.join(f"{name} probe: {p.get('reason') or p['status']}" for name, p in probes.items()),
                    errors=[dict(backend=name, phase='probe', reason=p.get('reason') or p['status']) for name, p in probes.items()])
        return base
    pending, history, outcomes = frames, {}, {}
    candidates = [selected] + [name for name in order if name != selected]
    for backend in candidates:
        if not pending:
            break
        record = check(backend)
        if record['status'] == 'not_run' or record.get('availability') == 'missing':
            continue
        timeout = {'rapidocr': FALLBACK_TIMEOUT_SECONDS, 'macos': NATIVE_TIMEOUT_SECONDS, 'ollama': 60}[backend]
        items, attempt = adapters[backend](pending, refs, timeout)
        attempt['probe_status'] = record['status']
        attempt['retry_after_probe_failure'] = record['status'] != 'complete'
        base['attempts'].append(attempt)
        for item in items:
            prior = history.get(item['timestamp'], [])
            item['fallback_from'] = prior[-1]['backend'] if prior else None
            item['reason'] = '; '.join(entry['reason'] for entry in prior if entry['reason']) or None
        base['items'].extend(items)
        for outcome in attempt['frames']:
            timestamp = outcome['timestamp']
            previous = outcomes.get(timestamp)
            # An executed failure is not erased by a later unavailable adapter.
            if previous and previous['status'] == 'failed' and outcome['status'] == 'not_run':
                continue
            outcomes[timestamp] = outcome
            if outcome['status'] != 'complete':
                history.setdefault(timestamp, []).append(dict(backend=backend, reason=outcome.get('reason')))
        pending = [(ts, image) for ts, image in pending if outcomes[ts]['status'] != 'complete']
    base['status'] = _status(list(outcomes.values()))
    completed_backends = {item['backend'] for item in base['items']}
    base['backend'] = 'mixed' if len(completed_backends) > 1 else next(iter(completed_backends), selected)
    models = {attempt['backend']: attempt['model'] for attempt in base['attempts']}
    base['model'] = models if base['backend'] == 'mixed' else models.get(base['backend'])
    base['errors'] = [error for attempt in base['attempts'] for error in attempt['errors']]
    reasons = [attempt['reason'] for attempt in base['attempts'] if attempt.get('reason')]
    base['reason'] = '; '.join(reasons) or None
    base['fallback_from'] = selected if len(base['attempts']) > 1 else None
    positions = {ts: index for index, (ts, _) in enumerate(frames)}
    base['items'].sort(key=lambda item: positions[item['timestamp']])
    return base
