"""Global frame board: evenly sampled stills for downstream content-type judgment.

This module only produces evidence. It never classifies the media and never
changes OCR decisions; callers may read the returned grid/frame timestamps and
join them with other analysis products.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path


DEFAULT_FRAMES = 16
MIN_FRAMES = 12
MAX_FRAMES = 24
TILE_WIDTH = 320
TILE_HEIGHT = 180
LABEL_HEIGHT = 18
BACKGROUND = '#151515'


def _failed(reason, *, media=None, out=None) -> dict:
    return {
        'status': 'failed',
        'reason': str(reason),
        'path': None,
        'media': str(media) if media is not None else None,
        'out': str(out) if out is not None else None,
        'frames': [],
        'grid': {},
    }


def _probe_duration(media_path: Path) -> float:
    completed = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(media_path)],
        capture_output=True, text=True, timeout=30,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or '').strip() or f'ffprobe exited {completed.returncode}'
        raise RuntimeError(f'无法读取媒体时长: {detail}')
    duration = float((completed.stdout or '').strip())
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f'媒体时长无效: {duration!r}')
    return duration


def _extract_frame(media_path: Path, timestamp: float, destination: Path) -> None:
    completed = subprocess.run(
        ['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{timestamp:.3f}',
         '-i', str(media_path), '-frames:v', '1',
         '-vf', 'scale=640:-2', '-q:v', '2', '-y', str(destination)],
        capture_output=True, text=True, timeout=60,
    )
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        detail = (completed.stderr or '').strip() or f'ffmpeg exited {completed.returncode}'
        raise RuntimeError(f'无法提取 {timestamp:.3f}s 的帧: {detail}')


def _even_timestamps(duration: float, count: int) -> list[float]:
    """Midpoint sampling avoids the first/last frame and covers the whole clip."""
    return [duration * (index + 0.5) / count for index in range(count)]


def _load_font(size: int = 12):
    from PIL import ImageFont
    for name in ('DejaVuSans.ttf', 'Arial.ttf', 'Helvetica.ttc'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def build_global_board(media_path, out_path, *, frames: int = DEFAULT_FRAMES,
                       grid_cols: int = 4) -> dict:
    """Sample 12–24 evenly spaced frames into one labelled JPEG board.

    Returns ``{"status", "path", "frames": [{"timestamp", "index"}], "grid": {...}}``.
    Any decode, dependency or filesystem failure is returned as
    ``status == "failed"`` with a ``reason``; this function never raises.
    """
    try:
        media = Path(media_path).expanduser()
        out = Path(out_path).expanduser().resolve()
    except Exception as exc:
        return _failed(exc, media=media_path, out=out_path)
    try:
        if not media.is_file():
            return _failed('媒体文件不存在或不是普通文件', media=media, out=out)
        try:
            count = int(frames)
            cols = int(grid_cols)
        except (TypeError, ValueError) as exc:
            return _failed(f'帧数或列数无效: {exc}', media=media, out=out)
        count = max(MIN_FRAMES, min(MAX_FRAMES, count))
        cols = max(1, min(cols, count))
        rows = math.ceil(count / cols)

        duration = _probe_duration(media)
        timestamps = _even_timestamps(duration, count)

        try:
            from PIL import Image, ImageDraw, ImageOps
        except ImportError as exc:
            return _failed(f'帧板需要 Pillow: {exc}', media=media, out=out)

        out.parent.mkdir(parents=True, exist_ok=True)
        tile_height = TILE_HEIGHT + LABEL_HEIGHT
        board = Image.new('RGB', (cols * TILE_WIDTH, rows * tile_height), BACKGROUND)
        draw = ImageDraw.Draw(board)
        font = _load_font()
        recorded = []

        with tempfile.TemporaryDirectory(prefix='.frame-board-', dir=str(out.parent)) as temp:
            temp_dir = Path(temp)
            for index, timestamp in enumerate(timestamps):
                source = temp_dir / f'frame_{index:03d}.jpg'
                _extract_frame(media, timestamp, source)
                with Image.open(source) as raw:
                    tile = ImageOps.contain(raw.convert('RGB'), (TILE_WIDTH, TILE_HEIGHT))
                x = (index % cols) * TILE_WIDTH
                y = (index // cols) * tile_height
                board.paste(tile, (x + (TILE_WIDTH - tile.width) // 2, y))
                draw.text((x + 6, y + TILE_HEIGHT + 3), f'{timestamp:.2f}s', fill='white', font=font)
                recorded.append({'timestamp': round(timestamp, 3), 'index': index})

        fd, temporary_name = tempfile.mkstemp(prefix=f'.{out.name}.', suffix='.jpg', dir=str(out.parent))
        os.close(fd)
        try:
            board.save(temporary_name, format='JPEG', quality=90)
            os.replace(temporary_name, out)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return {
            'status': 'complete',
            'path': str(out),
            'duration_seconds': round(duration, 3),
            'frames': recorded,
            'grid': {
                'cols': cols,
                'rows': rows,
                'count': count,
                'tile_width': TILE_WIDTH,
                'tile_height': TILE_HEIGHT,
                'label_height': LABEL_HEIGHT,
                'frame_width': cols * TILE_WIDTH,
                'frame_height': rows * tile_height,
            },
        }
    except Exception as exc:  # engine contract: surface failure, never raise
        return _failed(exc, media=locals().get('media'), out=locals().get('out'))
