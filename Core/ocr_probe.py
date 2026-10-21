"""Deterministic in-memory OCR probe, without Pillow or an external asset."""
import struct
import zlib

PROBE_TIMEOUT_SECONDS = 10
PROBE_TIMESTAMP = -1.0


def probe_png():
    """Render REELPACK 1234 in large black glyphs on a white PNG."""
    glyphs = {
        'R': ['11110','10001','10001','11110','10100','10010','10001'],
        'E': ['11111','10000','10000','11110','10000','10000','11111'],
        'L': ['10000','10000','10000','10000','10000','10000','11111'],
        'P': ['11110','10001','10001','11110','10000','10000','10000'],
        'A': ['01110','10001','10001','11111','10001','10001','10001'],
        'C': ['01111','10000','10000','10000','10000','10000','01111'],
        'K': ['10001','10010','10100','11000','10100','10010','10001'],
        '1': ['00100','01100','00100','00100','00100','00100','01110'],
        '2': ['01110','10001','00001','00010','00100','01000','11111'],
        '3': ['11110','00001','00001','01110','00001','00001','11110'],
        '4': ['00010','00110','01010','10010','11111','00010','00010'],
        ' ': ['00000'] * 7,
    }
    text, scale, margin = 'REELPACK 1234', 6, 24
    width, height = len(text) * 6 * scale + 2 * margin, 7 * scale + 2 * margin
    pixels = bytearray([255]) * (width * height)
    for index, character in enumerate(text):
        for y, row in enumerate(glyphs[character]):
            for x, value in enumerate(row):
                if value == '1':
                    for dy in range(scale):
                        start = (margin + y * scale + dy) * width + margin + (index * 6 + x) * scale
                        pixels[start:start + scale] = bytes(scale)
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    raw = b''.join(b'\x00' + pixels[y * width:(y + 1) * width] for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 0, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def run_probe(backend, adapter):
    """Use the material adapter; only nonempty successful text passes."""
    try:
        items, attempt = adapter([(PROBE_TIMESTAMP, probe_png())], {}, PROBE_TIMEOUT_SECONDS)
        text = '\n'.join(item['text'] for item in items if item.get('text', '').strip())
        status, reason = attempt['status'], attempt.get('reason')
        if status == 'complete' and not text.strip():
            status, reason = 'failed', 'probe returned empty text'
        return dict(backend=backend, status=status, text=text, reason=reason,
                    model=attempt.get('model'), availability='missing' if status == 'not_run' else 'available',
                    timeout_seconds=PROBE_TIMEOUT_SECONDS, attempt=attempt)
    except Exception as exc:
        status = 'not_run' if isinstance(exc, (FileNotFoundError, ImportError)) else 'failed'
        return dict(backend=backend, status=status, text='', reason=f'{type(exc).__name__}: {exc}',
                    availability='missing' if status == 'not_run' else 'available',
                    timeout_seconds=PROBE_TIMEOUT_SECONDS)


def normalize_probe(backend, record):
    record = dict(record, backend=backend)
    if record.get('status') == 'complete' and not str(record.get('text') or '').strip():
        record.update(status='failed', reason='probe returned empty text')
    if record.get('status') not in ('complete', 'failed', 'not_run'):
        record.update(status='failed', reason=record.get('reason') or 'invalid probe status')
    record.setdefault('availability', 'missing' if record['status'] == 'not_run' else 'available')
    return record
