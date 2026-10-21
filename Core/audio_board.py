#!/usr/bin/env python3
"""Pillow-only audio evidence renderer. Reads persisted measurements; never decodes."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

IMAGE_PATH = 'evidence/audio_board.png'
METRICS_PATH = 'evidence/audio_board.metrics.json'
WIDTH, HEIGHT = 1800, 1220
RENDERER_VERSION = 3
CJK_FONT_CANDIDATES = (
    ('systemPingFang', '/System/Library/Fonts/PingFang.ttc'),
    ('systemPingFang', '/System/Library/Fonts/Supplemental/PingFang.ttc'),
    ('systemSTHeiti', '/System/Library/Fonts/STHeiti Light.ttc'),
    ('systemHiraginoGB', '/System/Library/Fonts/Hiragino Sans GB.ttc'),
)
DISCLAIMER = ('Full mix physical measures only. No genre, emotion, author, instrument or BGM identity conclusions. '
              'Beat/cut proximity does not establish editing intent.')



def _cjk_font_metadata():
    # Read existing system fonts only; never embed, install or download a font.
    for source, path in CJK_FONT_CANDIDATES:
        if Path(path).is_file():
            return {'cjk_font_source': source, 'cjk_font_path': path, 'cjk_fallback': 'English'}
    return {'cjk_font_source': 'English fallback', 'cjk_font_path': None, 'cjk_fallback': 'English'}


def deterministic_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')) + '\n'


def payload_sha256(payload):
    return hashlib.sha256(deterministic_json(payload).encode('utf-8')).hexdigest()


def make_board_payload(music):
    """Select only physical evidence; semantic tags never enter the board."""
    data = music.get('board_data')
    if not isinstance(data, dict) or data.get('status') != 'complete':
        raise ValueError('bounded audio board_data unavailable')
    beats = list(music.get('beat_times_seconds') or [])
    onsets = list(music.get('onset_times_seconds') or [])
    cuts = list(data.get('cut_times_seconds') or [])
    alignment = dict(music.get('cut_alignment') or {})
    waveform = data.get('waveform') or {}
    measured = bool(beats) and alignment.get('status') in ('measured', 'complete')
    offset_histogram = dict(histogram(alignment.get('offsets_seconds') or [], bins=31),
                            status='measured', method='nearest_offset_counts', units={'edges': 'seconds', 'counts': 'cuts'},
                            params={'bins': 31, 'tolerance_seconds': alignment.get('tolerance_seconds', .15)}) if measured else {
        'status': 'not_measured', 'method': 'nearest_offset_counts', 'counts': None, 'edges_seconds': None,
        'reason': alignment.get('reason', 'alignment not measured'),
        'params': {'bins': 31, 'tolerance_seconds': alignment.get('tolerance_seconds', .15)}}
    return {
        'schema_version': 1, 'scope': 'whole_audio_mix',
        'full_mix_label': data.get('full_mix_label', 'Full mix / mono decode'),
        'boundary_disclaimer': DISCLAIMER,
        'renderer': {'method': 'Pillow', 'version': RENDERER_VERSION, 'width': WIDTH, 'height': HEIGHT,
                     'font': 'Pillow bundled default', 'histogram_bins': 31, **_cjk_font_metadata()},
        'board_data': data,
        'metrics': {'physical_status': music.get('status'), 'tempo_bpm': music.get('tempo_bpm'),
                    'beat_count': len(beats), 'onset_count': len(onsets), 'cut_count': len(cuts),
                    'aligned_cut_count': alignment.get('aligned_cut_count'),
                    'aligned_ratio': alignment.get('aligned_ratio'),
                    'alignment_status': alignment.get('status', 'not_measured'),
                    'duration_seconds': data.get('duration_seconds'), 'sample_rate': data.get('sample_rate'),
                    'source_sha256': data.get('source_sha256'),
                    'rms_dbfs_mean': waveform.get('rms_dbfs_mean'),
                    'rms_dbfs_global': waveform.get('rms_dbfs_global'),
                    'includes_media_start': any(float(x) == 0 for x in cuts)},
        'timeline': {'status': 'measured' if beats or onsets or cuts else 'not_measured', 'method': 'detected_audio_events_and_supplied_shots_start',
                     'params': {'beats': 'librosa.beat.beat_track', 'onsets': 'librosa.onset.onset_detect',
                                'cuts': 'shots.start, media_start included when supplied'},
                     'event_states': {'beats': 'measured' if beats else 'not_measured',
                                      'onsets': 'measured' if onsets else 'not_measured',
                                      'cuts': 'measured' if cuts else 'not_measured'},
                     'units': 'seconds', 'beats': beats, 'onsets': onsets, 'cuts': cuts,
                     'cut_source': 'shots.start including media_start when supplied'},
        'alignment': alignment, 'offset_histogram': offset_histogram,
    }


def _finite(values, label):
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
        raise ValueError(f'{label} must contain finite numbers')


def validate_payload(payload):
    data = payload.get('board_data') or {}
    duration = data.get('duration_seconds')
    _finite([duration], 'duration')
    if duration <= 0:
        raise ValueError('duration must be positive')
    wave = data.get('waveform') or {}
    times = wave.get('times_seconds') or []
    if not 1 <= len(times) <= 600:
        raise ValueError('waveform must have 1..600 time bins')
    for key in ('times_seconds', 'min', 'max', 'rms', 'rms_dbfs'):
        values = wave.get(key) or []
        if len(values) != len(times):
            raise ValueError('waveform arrays must have matching lengths')
        _finite(values, f'waveform {key}')
    spec = data.get('spectrogram') or {}
    rows = spec.get('values_db') or []
    freqs = spec.get('frequencies_hz') or []
    if not 1 <= len(rows) <= 600 or not 1 <= len(freqs) <= 128:
        raise ValueError('spectrogram exceeds bounded matrix limits')
    if len(spec.get('times_seconds') or []) != len(rows):
        raise ValueError('spectrogram time axis must match rows')
    for row in rows:
        if len(row) != len(freqs):
            raise ValueError('spectrogram frequency axis must match rows')
        _finite(row, 'spectrogram')
    _finite(freqs, 'frequency')
    edges = spec.get('frequency_edges_hz') or []
    _finite(edges, 'frequency edges')
    if len(edges) != len(freqs) + 1 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError('invalid frequency edges')
    if spec.get('params', {}).get('frequency_scale') == 'log' and edges[0] <= 0:
        raise ValueError('log frequency axis must be positive')
    for name in ('beats', 'onsets', 'cuts'):
        _finite(payload.get('timeline', {}).get(name) or [], name)
    _finite(payload.get('alignment', {}).get('offsets_seconds') or [], 'offsets')
    return data


def histogram(offsets, bins=31, extent=None):
    """Include every nearest offset; tolerance band is marked separately."""
    extent = max(.3, max(map(abs, offsets), default=0.)) if extent is None else extent
    counts = [0] * bins
    for value in offsets:
        i = min(bins - 1, max(0, int((value + extent) / (2 * extent) * bins)))
        counts[i] += 1
    return {'edges_seconds': [-extent + 2 * extent * i / bins for i in range(bins + 1)],
            'counts': counts, 'extent_seconds': extent}


def _format(value, digits=2, suffix=''):
    return 'not_measured' if value is None else f'{value:.{digits}f}{suffix}'


def render_board(payload, output_path):
    """Render entirely from JSON-safe board payload; Pillow is the only dependency."""
    from PIL import Image, ImageDraw, ImageFont
    data = validate_payload(payload)
    duration = float(data['duration_seconds'])
    image = Image.new('RGB', (WIDTH, HEIGHT), '#eef2f7')
    draw = ImageDraw.Draw(image)
    fonts = {n: ImageFont.load_default(size=n) for n in (14, 16, 18, 20, 24, 32)}
    cjk_fonts = {}
    cjk_path = payload['renderer'].get('cjk_font_path')
    if cjk_path:
        try:
            cjk_fonts = {n: ImageFont.truetype(cjk_path, n) for n in (18, 24)}
        except OSError:
            cjk_fonts = {}
    ink, muted, grid = '#16283d', '#556579', '#dce4ee'
    def text(xy, value, size=16, color=ink):
        draw.text(xy, str(value), font=fonts[size], fill=color)
    def cjk_text(xy, bilingual, english, size=18, color=ink):
        draw.text(xy, bilingual if cjk_fonts else english,
                  font=cjk_fonts.get(size, fonts[size]), fill=color)
    text((36, 24), 'AUDIO MEASUREMENT BOARD', 32)
    cjk_text((36, 66), '全混音 / full mix (all audio, mono decode; no source separation)',
             payload['full_mix_label'], 18, muted)
    metrics = payload['metrics']
    text((36, 98), f"Physical state: {metrics['physical_status']}  |  Duration {_format(duration, 3)} s  |  "
         f"Sample rate {data['sample_rate']} Hz  |  Tempo {_format(metrics['tempo_bpm'])} BPM  |  "
         f"Beats {metrics['beat_count']}  |  Onsets {metrics['onset_count']}", 18)
    if metrics.get('source_sha256'):
        text((36, 130), 'Source SHA256: ' + metrics['source_sha256'], 14, muted)
    panels = [(30, 164, 885, 650), (915, 164, 1770, 650),
              (30, 674, 885, 1130), (915, 674, 1770, 1130)]
    titles = ['1  Full-mix waveform + RMS dBFS', '2  Real frequency spectrogram',
              '3  Beats / onsets / shots.start timeline', '4  Nearest cut-to-beat offset histogram']
    plots = []
    for box, title in zip(panels, titles):
        draw.rounded_rectangle(box, radius=12, fill='white', outline='#cfdae7', width=2)
        text((box[0] + 18, box[1] + 16), title, 20)
        plots.append((box[0] + 64, box[1] + 106, box[2] - 50, box[3] - 84))
    def time_axes(box, y_lines=4):
        left, top, right, bottom = box
        for i in range(6):
            x = left + (right - left) * i / 5
            draw.line((x, top, x, bottom), fill=grid)
            text((x - 13, bottom + 10), f'{duration * i / 5:.1f}', 14, muted)
        for i in range(y_lines + 1):
            y = top + (bottom - top) * i / y_lines
            draw.line((left, y, right, y), fill=grid)
        text((right - 70, bottom + 34), 'Time (s)', 14, muted)
    wave, box = data['waveform'], plots[0]
    time_axes(box)
    left, top, right, bottom = box
    zero = (top + bottom) / 2
    amplitude_max = max(1., max(map(abs, wave['min'] + wave['max'])))
    ay = lambda x: zero - x / amplitude_max * (bottom - top) / 2
    tx = lambda x: left + x / duration * (right - left)
    for t, low, high in zip(wave['times_seconds'], wave['min'], wave['max']):
        draw.line((tx(t), ay(low), tx(t), ay(high)), fill='#328cb3', width=2)
    points = [(tx(t), bottom - (max(-120., min(0., db)) + 120.) / 120. * (bottom - top))
              for t, db in zip(wave['times_seconds'], wave['rms_dbfs'])]
    if len(points) > 1:
        draw.line(points, fill='#dd762a', width=3)
    else:
        x, y = points[0]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='#dd762a')
    mean_y = bottom - (max(-120., min(0., wave['rms_dbfs_mean'])) + 120.) / 120. * (bottom - top)
    for dash in range(left, right, 18):
        draw.line((dash, mean_y, min(right, dash + 10), mean_y), fill='#905632', width=2)
    for amplitude in (-amplitude_max, 0., amplitude_max):
        text((left - 44, ay(amplitude) - 8), f'{amplitude:.1f}', 14, muted)
    for db in (-120, -60, 0):
        text((right + 7, bottom - (db + 120) / 120 * (bottom - top) - 8), str(db), 14, '#bc5d19')
    text((panels[0][0] + 18, panels[0][1] + 52),
         f"State: measured  |  RMS dBFS mean {_format(wave['rms_dbfs_mean'])}  |  Global RMS {_format(wave['rms_dbfs_global'])} dBFS", 18)
    text((panels[0][0] + 18, panels[0][1] + 78), 'Blue: bucket min/max PCM    Orange: RMS dBFS; dashed: mean (right axis)', 14, muted)
    text((panels[0][0] + 18, panels[0][3] - 28), 'Mean: sample-weighted bucket dBFS; reference amplitude 1; floor -120 dBFS.', 14, muted)

    spec, box = data['spectrogram'], plots[1]
    left, top, right, bottom = box
    right -= 30
    rows, freqs = spec['values_db'], spec['frequencies_hz']
    floor = float(spec['params']['db_floor'])
    upper = max(floor + 1., max(max(row) for row in rows))
    stops = [(12, 20, 52), (37, 66, 125), (33, 139, 167), (237, 186, 73), (252, 246, 193)]
    def heat(value):
        pos = max(0., min(1., (value - floor) / (upper - floor))) * (len(stops) - 1)
        j = min(len(stops) - 2, int(pos))
        mix = pos - j
        return tuple(round(a + (b - a) * mix) for a, b in zip(stops[j], stops[j + 1]))
    pixels = [heat(rows[t][f]) for f in reversed(range(len(freqs))) for t in range(len(rows))]
    raster = Image.new('RGB', (len(rows), len(freqs)))
    raster.putdata(pixels)
    raster = raster.resize((right - left, bottom - top), resample=Image.Resampling.NEAREST)
    image.paste(raster, (left, top))
    for row in range(bottom - top + 1):
        value = upper - row / (bottom - top) * (upper - floor)
        draw.line((right + 12, top + row, right + 24, top + row), fill=heat(value))
    for i in range(3):
        value = upper - i / 2 * (upper - floor)
        text((right + 27, top + i / 2 * (bottom - top) - 8), f'{value:.0f}', 14, muted)
    for i in range(6):
        x = left + (right - left) * i / 5
        text((x - 13, bottom + 10), f'{duration * i / 5:.1f}', 14, muted)
    edges = spec['frequency_edges_hz']
    scale = spec['params']['frequency_scale']
    for hz in sorted(set([edges[0], edges[-1]] + [x for x in (100., 1000., 5000.) if edges[0] < x < edges[-1]])):
        fraction = (math.log(hz / edges[0]) / math.log(edges[-1] / edges[0]) if scale == 'log'
                    else (hz - edges[0]) / (edges[-1] - edges[0]))
        y = bottom - fraction * (bottom - top)
        label = f'{hz / 1000:.1f}k' if hz >= 1000 else f'{hz:.0f}'
        text((left - 51, y - 8), label, 14, muted)
        draw.line((left - 4, y, left, y), fill=ink)
    text((right - 75, bottom + 34), 'Time (s)', 14, muted)
    text((panels[1][0] + 18, panels[1][1] + 52),
         f"State: measured  |  Hann STFT n_fft={spec['params']['n_fft']} hop={spec['params']['hop_length']}  |  {scale} Hz axis", 18)
    text((panels[1][0] + 18, panels[1][1] + 78),
         f"Power dB ref=1; display {floor:.0f}..{upper:.1f} dB; Hz bands averaged in linear power", 14, muted)
    text((panels[1][0] + 18, panels[1][3] - 28),
         f"{len(rows)} time x {len(freqs)} frequency bins; range {edges[0]:.0f}..{edges[-1]:.0f} Hz; zero-centered padding.", 14, muted)

    timeline, box = payload['timeline'], plots[2]
    left, top, right, bottom = box
    time_axes(box, 3)
    colors = {'beats': '#dd762a', 'onsets': '#398faf', 'cuts': '#76539d'}
    for i, name in enumerate(('beats', 'onsets', 'cuts')):
        y = top + (i + .5) / 3 * (bottom - top)
        text((left - 53, y - 8), name, 14, colors[name])
        for timestamp in timeline[name]:
            if 0 <= timestamp <= duration:
                x = left + timestamp / duration * (right - left)
                draw.line((x, y - 21, x, y + 21), fill=colors[name], width=2)
    text((panels[2][0] + 18, panels[2][1] + 52),
         f"Beats {len(timeline['beats'])} | Onsets {len(timeline['onsets'])} | Starts {len(timeline['cuts'])} | IncludedMediaStart: {'yes' if metrics['includes_media_start'] else 'no'}", 18)
    text((panels[2][0] + 18, panels[2][1] + 78),
         'State: beats=' + timeline['event_states']['beats'] + ' | onsets=' + timeline['event_states']['onsets'] + ' | starts=' + timeline['event_states']['cuts'], 14, muted)
    text((panels[2][0] + 18, panels[2][3] - 28), 'Cut source: shots.start; media start (0 s) remains in the legacy denominator when supplied.', 14, muted)

    alignment, box = payload['alignment'], plots[3]
    left, top, right, bottom = box
    offsets = alignment.get('offsets_seconds') or []
    if not timeline['beats'] or alignment.get('status') not in ('measured', 'complete') or not offsets:
        if not timeline['beats']:
            cjk_text((left + 28, top + 44), '未测得（beats not measured）',
                     'beats not measured / not_measured', 24, muted)
        else:
            text((left + 28, top + 44), 'Alignment not_measured', 24, muted)
        count_label = 'null' if alignment.get('aligned_cut_count') is None else str(alignment['aligned_cut_count'])
        text((left + 28, top + 88), 'Alignment count: ' + count_label + '    Alignment ratio: null', 18, muted)
        text((left + 28, top + 130), 'No histogram data is drawn for an unmeasured result.', 16, muted)
        text((panels[3][0] + 18, panels[3][1] + 52), 'State: not_measured  |  No synchrony conclusion', 18)
    else:
        tolerance = float(alignment.get('tolerance_seconds', .15))
        hist = payload.get('offset_histogram') or histogram(offsets, bins=31)
        extent, counts, hist_edges = hist['extent_seconds'], hist['counts'], hist['edges_seconds']
        hx = lambda value: left + (value + extent) / (2 * extent) * (right - left)
        draw.rectangle((hx(-tolerance), top, hx(tolerance), bottom), fill='#edf5e9')
        maximum = max(1, max(counts))
        ticks = sorted(set(round(maximum * i / min(4, maximum)) for i in range(min(4, maximum) + 1)))
        for count in ticks:
            y = bottom - count / maximum * (bottom - top)
            draw.line((left, y, right, y), fill=grid)
            text((left - 40, y - 8), str(count), 14, muted)
        for i, count in enumerate(counts):
            if count:
                x1, x2 = hx(hist_edges[i]) + 1, hx(hist_edges[i + 1]) - 1
                y = bottom - count / maximum * (bottom - top)
                draw.rectangle((x1, y, max(x1, x2), bottom), fill='#587dad')
        for value in (-tolerance, tolerance):
            draw.line((hx(value), top, hx(value), bottom), fill='#488438', width=2)
            text((hx(value) + 4, top + 7), f'{value:+.2f}', 14, '#488438')
        for i in range(5):
            value = -extent + 2 * extent * i / 4
            text((hx(value) - 21, bottom + 10), f'{value:+.2f}', 14, muted)
        text((right - 199, bottom + 34), 'Signed offset (s)', 14, muted)
        text((panels[3][0] + 18, panels[3][1] + 52),
             f"State: {alignment['status']}  |  Aligned {alignment['aligned_cut_count']}/{alignment['cut_count']}  "
             f"({_format(alignment['aligned_ratio'] * 100, 1)}%)", 18)
        text((panels[3][0] + 18, panels[3][1] + 78),
             f"Nearest beat; cut minus beat; inclusive tolerance +/-{tolerance:.2f} s (green)", 14, muted)
    text((panels[3][0] + 18, panels[3][3] - 28), 'Offsets are measurements; negative = cut before beat, positive = cut after beat.', 14, muted)
    text((36, 1153), 'BOUNDARY: Full mix measurements; no genre, emotion, author, instrument or BGM identity conclusions.', 16, muted)
    text((36, 1180), 'Onsets and beats are detector outputs. Cut/beat proximity does not prove editing intent. Units and methods are in the paired metrics JSON.', 16, muted)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format='PNG', optimize=False, compress_level=6)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'status': 'complete', 'path': IMAGE_PATH, 'metadata_path': METRICS_PATH,
            'backend': 'Pillow', 'method': 'four_panel_physical_audio_measurements',
            'params': payload['renderer'], 'sha256': digest, 'data_sha256': payload_sha256(payload),
            'scope': 'whole_audio_mix', 'metrics': payload['metrics'], 'boundary_disclaimer': DISCLAIMER}


def render_from_json(metadata_path, output_path):
    payload = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    return render_board(payload, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render audio board from bounded persisted JSON; no audio decoding')
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        import os
        try:
            from runtime_environment import prepare_runtime_environment
        except ImportError:
            prepare_runtime_environment = None
        if prepare_runtime_environment is not None:
            os.environ.update(prepare_runtime_environment(create=os.environ.get('REELPACK_READONLY_CHECK') != '1'))
        print(json.dumps(render_from_json(args.metrics, args.output), ensure_ascii=False, allow_nan=False, sort_keys=True))
    except Exception as exc:
        print(json.dumps({'status': 'not_run', 'reason': f'{type(exc).__name__}: {exc}',
                          'impact': 'Audio board unavailable; physical measurements remain usable.',
                          'remedy': 'Repair local Pillow/metrics and rerender the saved JSON.'}, sort_keys=True))
        sys.exit(0)
