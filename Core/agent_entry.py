#!/usr/bin/env python3
"""Thin installed `reelpack` entry: native CLI forwarding and read-only evidence lookup.

This module never imports the analysis engine, downloads media, loads models,
performs video analysis. The explicit enrich command updates audio only.
Evidence commands (`frames`/`montage`, `review`, `preview`) reuse the stored
package APIs or decode only the explicit local video they were asked for.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

# Isolated Python (-I) omits the script directory. Only add this entry's
# immutable sibling Core directory; never borrow modules from the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))


LOCAL_FILE_GUIDANCE = (
    "Reelpack 只处理本地视频文件。请让你的 Agent（或你）先把视频取成本地文件，"
    "再传入本地路径。详见 SKILL.md。"
)

EVIDENCE_COMMANDS = ("frames", "montage", "review", "motion", "annotate", "supplements")
DEFAULT_LIBRARY_ROOT = "~/Downloads/Reelpack/library"
PREVIEW_BOARD_NAME = "preview-board.jpg"
PREVIEW_METADATA_NAME = "preview.json"


def option_value(arguments, name):
    """Read --name VALUE / --name=VALUE without reimplementing the real parser."""
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def existing_library_root(arguments):
    """Return the library root only if a packages directory already exists.

    The asset store constructor creates missing directories, so evidence
    commands reject an absent library before constructing it.
    """
    root = Path(option_value(arguments, "--library-root") or DEFAULT_LIBRARY_ROOT).expanduser().resolve()
    return root, (root / "packages").is_dir()


def evidence_main(arguments):
    """Route an evidence command to the real asset CLI without creating a library.

    The asset CLI shares the saved-package argument contract with development
    usage. Help exits inside argparse before any store is constructed.
    """
    command = arguments[0]
    if not any(token in arguments for token in ("--help", "-h")):
        root, present = existing_library_root(arguments)
        if not present:
            print(f"Reelpack {command}: 找不到已有资产库：{root}。"
                  "请先处理本地视频，或指定 --library-root。未创建新库。", file=sys.stderr)
            return 2
    from asset_cli import main as asset_cli_main
    return asset_cli_main(arguments)


def path_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def preview_library_roots():
    """Existing/default library locations preview output must never touch."""
    candidates = []
    explicit = os.environ.get("REELPACK_LIBRARY_ROOT")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    output_root = os.environ.get("REELPACK_OUTPUT_ROOT") or "~/Downloads/Reelpack"
    candidates.append(Path(output_root).expanduser() / "library")
    roots = []
    for candidate in candidates:
        try:
            roots.append(candidate.resolve())
        except OSError:
            continue
    return roots


def preview_conflict(source, output):
    if output == source:
        return "预览输出目录不能是源视频路径本身"
    if path_within(source, output):
        return "预览输出目录不能包含源视频"
    for root in preview_library_roots():
        if path_within(output, root) or path_within(root, output):
            return f"预览输出不能与资产库路径重叠：{root}"
    return None


def prepare_preview_output(output):
    """Return whether this call created the directory; refuse occupied output."""
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"预览输出已存在且不是目录：{output}")
        if any(output.iterdir()):
            raise ValueError(f"预览输出目录非空，拒绝覆盖：{output}")
        return False
    output.mkdir(parents=True)
    return True


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def preview_main(arguments):
    parser = argparse.ArgumentParser(
        prog="reelpack preview",
        description="不经 ASR/导入，为本地视频生成整片预览帧板与元数据；不分类、不选画像、不联网")
    parser.add_argument("source", help="本地视频文件路径")
    parser.add_argument("--output", required=True, help="库外的新目录；帧板和 preview.json 只写到这里")
    parser.add_argument("--frames", type=int, default=16,
                        help="预览帧数（现有帧板实现限制在 12..24，元数据记录实际值）")
    args = parser.parse_args(arguments)
    if args.frames < 1:
        parser.error("--frames 必须为正整数")

    source = Path(args.source).expanduser()
    output = Path(args.output).expanduser()
    if not source.is_file():
        print(f"Reelpack preview: 源视频不存在或不是普通文件：{source}", file=sys.stderr)
        return 2
    try:
        source = source.resolve(strict=True)
        output = output.resolve()
        source_size = source.stat().st_size
    except OSError as exc:
        print(f"Reelpack preview: {exc}", file=sys.stderr)
        return 2
    conflict = preview_conflict(source, output)
    if conflict:
        print(f"Reelpack preview: {conflict}", file=sys.stderr)
        return 2

    created = False
    try:
        created = prepare_preview_output(output)
    except (OSError, ValueError) as exc:
        print(f"Reelpack preview: {exc}", file=sys.stderr)
        return 2

    from global_board import build_global_board
    board = build_global_board(source, output / PREVIEW_BOARD_NAME,
                               frames=args.frames, grid_cols=4)
    if board.get("status") != "complete":
        if created:
            try:
                output.rmdir()  # Only an empty directory this call created.
            except OSError:
                pass
        print(f"Reelpack preview: 无法生成预览帧板：{board.get('reason')}", file=sys.stderr)
        return 2

    metadata_path = output / PREVIEW_METADATA_NAME
    metadata = {
        "command": "preview",
        "status": board["status"],
        "source": {"kind": "local_file", "path": str(source),
                   "display_name": source.name, "size_bytes": source_size},
        "output": str(output),
        "board": {"path": board["path"], "source": "original_media"},
        "duration_seconds": board.get("duration_seconds"),
        "frames": board["frames"],
        "grid": board["grid"],
        "requested_frames": args.frames,
        "metadata_path": str(metadata_path),
        "analysis_run": False,
        "network_used": False,
        "model_used": False,
        "content_classified": False,
        "profile_selected": False,
    }
    try:
        atomic_json(metadata_path, metadata)
    except OSError as exc:
        print(f"Reelpack preview: 无法写入元数据：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def read_object(path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"资产文件必须是 object: {path.name}")
    return value


def query_library(root, text, limit=20):
    """Literal text lookup over saved transcript/OCR/summary, without analysis."""
    root = Path(root).expanduser().resolve()
    packages = root / "packages"
    if not packages.is_dir():
        raise FileNotFoundError(f"找不到已有资产库：{root}。请先处理本地视频，或指定 --library-root。")
    needle = text.casefold().strip()
    matches, warnings = [], []
    for folder in sorted(packages.iterdir()):
        if not folder.is_dir() or folder.is_symlink() or not folder.name.startswith("asset_"):
            continue
        analysis_path = folder / "analysis.json"
        if not analysis_path.is_file() or analysis_path.is_symlink():
            continue
        try:
            analysis = read_object(analysis_path)
            asset_path = folder / "asset.json"
            asset = read_object(asset_path) if asset_path.is_file() and not asset_path.is_symlink() else {}
            title = (asset.get("source") or {}).get("display_name", folder.name)
            transcript = analysis.get("transcript") or {}
            ocr = analysis.get("ocr") or {}
            segments = [s for s in transcript.get("raw_segments", [])
                        if isinstance(s, dict) and needle in str(s.get("text", "")).casefold()]
            items = [s for s in ocr.get("items", [])
                     if isinstance(s, dict) and needle in str(s.get("text", "")).casefold()]
            other = [str(title), folder.name, str(transcript.get("raw_text", "")),
                     str(transcript.get("corrected_text") or ""),
                     str(ocr.get("reading_text", "")),
                     str((analysis.get("summary") or {}).get("text", ""))]
            if needle and not segments and not items and not any(needle in s.casefold() for s in other):
                continue
            matches.append({
                "asset_id": folder.name, "title": title, "status": analysis.get("status"),
                "analysis_path": str(analysis_path), "package_path": str(folder),
                "transcript_matches": segments[:50], "ocr_matches": items[:50],
                "matches_truncated": len(segments) > 50 or len(items) > 50,
                "shots_count": len((analysis.get("shots") or {}).get("items", [])),
            })
            if len(matches) >= limit:
                break
        except (OSError, ValueError, AttributeError, TypeError):
            warnings.append({"asset_id": folder.name, "reason": "资产 JSON 缺失、损坏或字段类型不匹配；未修改文件"})
    return {"query": text, "mode": "literal_saved_text", "library_root": str(root),
            "count": len(matches), "limit": limit, "results": matches,
            "warnings": warnings, "analysis_run": False, "network_used": False}


def query_main(arguments):
    parser = argparse.ArgumentParser(prog="reelpack query", description="只读查询已保存的资产包文字；不重新分析或联网")
    parser.add_argument("text", nargs="?", default="", help="文字关键词或 asset_id；省略时列出资产")
    parser.add_argument("--output-dir", default="~/Downloads/Reelpack", help="与处理命令相同的输出目录，其下 library 为资产库")
    parser.add_argument("--library-root", help="直接指定已有资产库根目录")
    parser.add_argument("--limit", type=int, default=20, help="最多返回的资产数（1..100）")
    args = parser.parse_args(arguments)
    if not 1 <= args.limit <= 100:
        parser.error("limit 必须在 1..100")
    root = args.library_root or Path(args.output_dir).expanduser() / "library"
    try:
        result = query_library(root, args.text, args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"Reelpack query: {exc}", file=sys.stderr)
        return 2


def enrich_main(arguments):
    parser = argparse.ArgumentParser(prog='reelpack enrich', description='只补已有资产音频，不重跑转录、镜头或证据帧')
    parser.add_argument('asset_id')
    parser.add_argument('--only', choices=['audio'], default='audio')
    parser.add_argument('--output-dir', default='~/Downloads/Reelpack')
    parser.add_argument('--library-root')
    parser.add_argument('--music-backend', choices=['librosa', 'panns'])
    parser.add_argument('--redraw-board', action='store_true', help='只根据已存 JSON 重绘，不解码媒体')
    args = parser.parse_args(arguments)
    from asset_store import AssetStore
    from asset_enrich import enrich, compact_audio_result
    library = Path(args.library_root).expanduser() if args.library_root else Path(args.output_dir).expanduser()/'library'
    if not (library/'packages'/args.asset_id/'analysis.json').is_file():
        print('Reelpack enrich: 找不到已有资产包，未创建新库。', file=sys.stderr)
        return 2
    result = enrich(AssetStore(library), args.asset_id, only='audio',
                    music_backend=args.music_backend, redraw_board=args.redraw_board)
    print(json.dumps(compact_audio_result(result), ensure_ascii=False, indent=2))
    return 0


def native_cli():
    core = Path(__file__).resolve().parent
    resources = core.parent
    if resources.name == "Resources" and resources.parent.name == "Contents":
        executable = resources.parent / "MacOS/reelpack-cli"
        if executable.is_file():
            return executable
        raise FileNotFoundError("安装包中缺少 reelpack-cli。请重新安装 Reelpack。")
    root = core.parent
    if (root / "release-info.json").is_file():
        executable = root / "bin/reelpack-cli"
        if executable.is_file():
            return executable
        raise FileNotFoundError("standalone 安装中缺少 bin/reelpack-cli。请重新运行 install-cli.sh。")
    # SwiftPM emits `.build/{release,debug}` aliases pointing into
    # `.build/out/Products/...`; older toolchains exposed the architecture
    # directory directly. Check both, keeping release ahead of debug.
    build = root / ".build"
    candidates = []
    for mode in ("release", "debug"):
        candidates.append(build / f"{platform.machine()}-apple-macosx" / mode / "reelpack-cli")
        candidates.append(build / mode / "reelpack-cli")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到 reelpack-cli。请先构建或运行 install-cli.sh。详见 SKILL.md。")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "query":
        return query_main(arguments[1:])
    from runtime_environment import prepare_pipeline_environment
    # Evidence retrieval and preview never need the analysis cache, so they
    # prepare read-only paths and cannot create a library or runtime cache.
    read_only = ('--version' in arguments or '--runtime-check' in arguments or '--help' in arguments or '-h' in arguments
                 or (arguments and arguments[0] in (*EVIDENCE_COMMANDS, "preview")))
    os.environ.update(prepare_pipeline_environment(create=not read_only))
    if arguments and arguments[0] in EVIDENCE_COMMANDS:
        return evidence_main(arguments)
    if arguments and arguments[0] == "preview":
        return preview_main(arguments[1:])
    if arguments and arguments[0] == 'enrich':
        return enrich_main(arguments[1:])
    if arguments and arguments[0] in ("--help", "-h"):
        print("Reelpack: 本地视频 → 资产包\n\nreelpack <本地文件> --output-dir <目录>\n"
              "reelpack preview <本地视频> --output <库外新目录> [--frames N]\n"
              "reelpack query --output-dir <目录> <关键词>\n"
              "reelpack frames <asset_id> --start <秒> --end <秒> --library-root <库>\n"
              "reelpack montage <asset_id> --shot <id> --library-root <库>\n"
              "reelpack review <asset_id> --start <秒> --end <秒> --output <库外新目录> --library-root <库>\n"
              "reelpack motion <asset_id> --start <秒> --end <秒> --library-root <库> [--save]\n"
              "reelpack annotate <asset_id> --input <注释JSON> --library-root <库>\n"
              "reelpack supplements <asset_id> --library-root <库> [--id <ID>]\n"
              "reelpack enrich <asset_id> --only audio --output-dir <目录>\n\n"
              "处理命令参数由 reelpack-cli 提供；Agent 使用说明见 SKILL.md。", flush=True)
    try:
        executable = native_cli()
        # Argument arrays and an absolute executable path avoid shell interpretation.
        return subprocess.call([str(executable), *arguments], env=dict(os.environ))
    except OSError as exc:
        print(f"Reelpack: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
