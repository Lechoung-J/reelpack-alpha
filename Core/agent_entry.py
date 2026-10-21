#!/usr/bin/env python3
"""Thin installed `reelpack` entry: native CLI forwarding and read-only text lookup.

This module never imports the analysis engine, downloads media, loads models,
performs video analysis. The explicit enrich command updates audio only.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


LOCAL_FILE_GUIDANCE = (
    "Reelpack 只处理本地视频文件。请让你的 Agent（或你）先把视频取成本地文件，"
    "再传入本地路径。详见 SKILL.md。"
)


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
    candidates = [root / ".build" / f"{platform.machine()}-apple-macosx" / mode / "reelpack-cli"
                  for mode in ("release", "debug")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到 reelpack-cli。请先构建或运行 install.sh。详见 SKILL.md。")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "query":
        return query_main(arguments[1:])
    from runtime_environment import prepare_pipeline_environment
    read_only = '--runtime-check' in arguments or '--help' in arguments or '-h' in arguments
    os.environ.update(prepare_pipeline_environment(create=not read_only))
    if arguments and arguments[0] == 'enrich':
        return enrich_main(arguments[1:])
    if arguments and arguments[0] in ("--help", "-h"):
        print("Reelpack: 本地视频 → 资产包\n\nreelpack <本地文件> --output-dir <目录>\n"
              "reelpack query --output-dir <目录> <关键词>\n"
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
