"""On-demand boards from stored evidence; original media is never touched."""
from pathlib import Path
import subprocess
import uuid

from asset_store import AssetStoreError, _safe_relative_path


def _extract_detail_frame(media_path: Path, timestamp: float, destination: Path):
    """Decode one original-resolution frame without modifying the source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", str(max(0.0, timestamp)), "-i", str(media_path),
        "-frames:v", "1", "-q:v", "2", "-y", str(destination),
    ], capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        return None
    try:
        from PIL import Image
        with Image.open(destination) as image:
            width, height = image.size
    except Exception:
        destination.unlink(missing_ok=True)
        return None
    return width, height


def frame_board(store, asset_id, *, view='overview', shot=None, start=None, end=None,
                timestamp=None, grid=9, full_res=False):
    if shot:
        result = store.query_shot(asset_id, shot)
    elif start is not None and end is not None:
        result = store.query_time(asset_id, start, end)
    else:
        raise AssetStoreError('需要 --shot 或 --start/--end')
    package = store.packages_dir / asset_id
    frames = sorted(result['images'], key=lambda f: f['timestamp'])
    if not frames:
        raise AssetStoreError('范围内没有图片证据')
    if view == 'detail':
        candidates = frames if timestamp is not None else [f for f in frames if f.get('keyframe')] or frames
        target = timestamp if timestamp is not None else candidates[len(candidates)//2]['timestamp']
        evidence = min(candidates, key=lambda f: abs(f['timestamp'] - target))
        path = _safe_relative_path(package, evidence['relative_path'])
        if not path.is_file():
            raise AssetStoreError('图片证据文件缺失')
        source, width, height = 'proxy', None, None
        if full_res:
            media = store.get_asset(asset_id).get('media') or {}
            media_path = Path(media['local_path']) if media.get('local_path') else None
            if media_path and media_path.is_file():
                detail_path = package / 'boards' / f'detail_{target:.3f}_{uuid.uuid4().hex[:8]}.jpg'
                dimensions = _extract_detail_frame(media_path, target, detail_path)
                if dimensions:
                    path, (width, height), source = detail_path, dimensions, 'original'
        if width is None:
            try:
                from PIL import Image
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                width = height = None
        metadata = {'timestamp': target, 'relative_path': str(path.relative_to(package)),
                    'absolute_path': str(path.resolve()), 'width': width, 'height': height,
                    'resolution': f'{width}x{height}' if width and height else None, 'source': source}
        analysis = store.get_asset(asset_id).get('analysis')
        if analysis is not None:
            details = [item for item in analysis.get('detail_frames', []) if abs(item.get('timestamp', -1) - target) > 0.001]
            details.append(metadata)
            analysis['detail_frames'] = details
            store.save_analysis(asset_id, analysis)
        item = {**evidence, 'relative_path': str(path.relative_to(package)),
                'absolute_path': str(path.resolve()), 'width': width, 'height': height,
                'resolution': f'{width}x{height}' if width and height else None, 'source': source}
        return {'asset_id': asset_id, 'view': view, 'absolute_path': str(path.resolve()),
                'relative_path': str(path.relative_to(package)), 'frames': [item],
                'width': width, 'height': height, 'resolution': item['resolution'], 'source': source}
    try:
        from PIL import Image, ImageOps, ImageDraw
    except ImportError as exc:
        raise AssetStoreError('图片板需要 Pillow；请使用包含 Pillow 的 Python') from exc
    paths, all_used = [], []
    columns = 3 if grid == 9 else 4
    folder = package / 'boards'
    folder.mkdir(exist_ok=True)
    run = uuid.uuid4().hex[:12]
    for page in range(0, len(frames), grid):
        used = frames[page:page+grid]
        board = Image.new('RGB', (columns*384, 3*240), '#151515')
        draw = ImageDraw.Draw(board)
        for index, frame in enumerate(used):
            source_path = _safe_relative_path(package, frame['relative_path'])
            with Image.open(source_path) as image:
                tile = ImageOps.contain(image.convert('RGB'), (384, 216))
            x, y = index % columns * 384, index // columns * 240
            board.paste(tile, (x+(384-tile.width)//2, y))
            draw.text((x+6, y+219), f"{frame['timestamp']:.3f}s", fill='white')
        path = folder / f'{run}_{page//grid+1:03d}.jpg'
        board.save(path, quality=92)
        paths.append({'absolute_path': str(path.resolve()), 'relative_path': str(path.relative_to(package)), 'frames': used})
        all_used.extend(used)
    return {'asset_id': asset_id, 'view': view, 'grid': grid, 'pages': paths, 'frames': all_used}
