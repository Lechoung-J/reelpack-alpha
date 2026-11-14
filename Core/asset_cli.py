#!/usr/bin/env python3
"""CLI for inspecting and querying portable Reelpack Agent Video packages.

Examples:
  python3 Core/asset_cli.py import ./clip.mp4 --library-root ~/ReelpackLibrary
  python3 Core/asset_cli.py list --library-root ~/ReelpackLibrary
  python3 Core/asset_cli.py show asset_... --library-root ~/ReelpackLibrary
  python3 Core/asset_cli.py query asset_... --start 10 --end 20 --library-root ~/ReelpackLibrary
"""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path

# Allow `python Core/asset_cli.py ...` from any working directory.
CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from asset_contract import assert_valid_analysis
from asset_store import AssetNotFoundError, AssetStore, AssetStoreError


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def shot_evidence_view(analysis):
    evidence = {e['evidence_id']: e for e in analysis.get('evidence', [])}
    result = []
    for shot in analysis.get('shots', {}).get('items', []):
        frames = sorted([evidence[e] for e in shot.get('evidence_ids', [])
                         if e in evidence and evidence[e].get('kind') in ('image', 'frame')],
                        key=lambda e: e['timestamp'])
        anchors = {}
        if frames:
            midpoint = (shot['start'] + shot['end']) / 2
            for role, frame in [('head', frames[0]),
                                ('middle', min(frames, key=lambda e: abs(e['timestamp'] - midpoint))),
                                ('tail', frames[-1])]:
                anchors[role] = frame
        result.append(dict(shot_id=shot['shot_id'], start=shot['start'], end=shot['end'],
                           anchors=anchors, frames=frames,
                           content_description=dict(status='not_run', reason='由下游多模态 Agent 按证据描述人/车/场景/动作；不推测未见内容')))
    return dict(backend=analysis.get('shots', {}).get('backend'), items=result,
                path_base='asset package directory', time_unit='seconds')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reelpack Agent Video 资产包查询 CLI")
    parser.add_argument("--library-root", default="~/Downloads/Reelpack/library", help="资产库根目录")
    sub = parser.add_subparsers(dest="command", required=True)

    import_cmd = sub.add_parser("import", help="导入本地媒体并按完整文件哈希去重")
    import_cmd.add_argument("path", help="媒体文件路径")

    sub.add_parser("list", help="列出已导入资产")

    show_cmd = sub.add_parser("show", help="显示资产元数据、分析与媒体状态")
    show_cmd.add_argument("asset_id")

    shots_cmd = sub.add_parser("shots", help="只读逐镜关键帧证据（不调用视觉模型）")
    shots_cmd.add_argument("asset_id")

    query_cmd = sub.add_parser("query", help="按时间范围查询转录、OCR、镜头和证据")
    query_cmd.add_argument("asset_id")
    query_cmd.add_argument("--start", type=float)
    query_cmd.add_argument("--end", type=float)

    query_cmd.add_argument("--shot")
    frames_cmd = sub.add_parser("frames", aliases=["montage"], help="按需输出图片板或清晰单帧")
    frames_cmd.add_argument("asset_id")
    frames_cmd.add_argument("--shot")
    frames_cmd.add_argument("--start", type=float)
    frames_cmd.add_argument("--end", type=float)
    frames_cmd.add_argument("--view", choices=["overview", "detail"], default="overview")
    frames_cmd.add_argument("--timestamp", type=float)
    frames_cmd.add_argument("--full-res", action="store_true", help="detail 视图优先从归档原片按需抽取全分辨率帧")
    frames_cmd.add_argument("--grid", type=int, choices=[9, 12], default=9)
    summary_cmd = sub.add_parser("set-summary", help="下游 Agent 回写摘要（不调用模型）")
    summary_cmd.add_argument("asset_id")
    summary_cmd.add_argument("--text", required=True)

    enrich_cmd = sub.add_parser("enrich", aliases=["backfill"], help="仅离线补 OCR/摘要；不重转录")
    enrich_cmd.add_argument("asset_id")
    enrich_cmd.add_argument("--only", choices=["audio", "ocr", "summary"])
    enrich_cmd.add_argument("--music-backend", choices=["librosa", "panns"], default=None)
    enrich_cmd.add_argument("--redraw-board", action="store_true", help="根据已存 JSON 重绘音频图，不解码媒体")
    review_cmd = sub.add_parser("review", help="只读存疑段音频/前后帧/OCR 回查")
    review_cmd.add_argument("asset_id")
    review_cmd.add_argument("--start", type=float, required=True)
    review_cmd.add_argument("--end", type=float, required=True)
    review_cmd.add_argument("--output", required=True, help="库外新目录；内含 JSON 和人类可读文本")
    review_cmd.add_argument("--padding", type=float, default=1.5)
    review_cmd.add_argument("--interval", type=float, default=0.5)
    review_cmd.add_argument("--frames", type=int, help="前后各 N 个采样帧，覆盖 padding=N*interval")

    motion_cmd = sub.add_parser(
        "motion", help="只读测量显式时间范围的画面运动；--save 保存不可变引擎测量补充文档")
    motion_cmd.add_argument("asset_id")
    motion_cmd.add_argument("--start", type=float, required=True,
                            help="范围起点（秒，必须为有限数）")
    motion_cmd.add_argument("--end", type=float, required=True,
                            help="范围终点（秒，必须大于 start；不隐式测量整片）")
    motion_cmd.add_argument("--save", action="store_true",
                            help="保存可用测量到 supplements/；失败不覆盖旧文档")
    motion_cmd.add_argument("--sample-fps", type=float,
                            help="沿用 motion_analyze 默认 8.0；范围 (0,240]")
    motion_cmd.add_argument("--analysis-width", type=int,
                            help="沿用 motion_analyze 默认 480；范围 64..1920")
    motion_cmd.add_argument("--max-seconds", type=float,
                            help="沿用 motion_analyze 默认 120.0；累计分析时长预算")
    motion_cmd.add_argument("--smoothing-window", type=int,
                            help="沿用 motion_analyze 默认 5；奇数化后 1..100000")
    motion_cmd.add_argument("--head-tail", type=float,
                            help="沿用 motion_analyze 默认 1.0；逐镜首尾窗口秒数")

    annotate_cmd = sub.add_parser(
        "annotate", help="写入带来源的 Agent 身份解释补充文档（不运行分类器）")
    annotate_cmd.add_argument("asset_id")
    annotate_cmd.add_argument("--input", required=True, help="reelpack.agent-annotations JSON 文件")

    supplements_cmd = sub.add_parser(
        "supplements", help="只读列出/读取补充文档及 current/stale/unknown 关系")
    supplements_cmd.add_argument("asset_id")
    supplements_cmd.add_argument("--id", help="只读取指定 supplement ID")

    validate_cmd = sub.add_parser("validate", help="校验资产包的结构化结果")
    validate_cmd.add_argument("asset_id")
    # Accept --library-root both before and after the subcommand for shell ergonomics.
    for child in (import_cmd, sub.choices["list"], show_cmd, shots_cmd, query_cmd,
                  frames_cmd, summary_cmd, validate_cmd, enrich_cmd, review_cmd,
                  motion_cmd, annotate_cmd, supplements_cmd):
        child.add_argument("--library-root", dest="sub_library_root", default=None, help=argparse.SUPPRESS)
    return parser


def _root_from_args(args):
    return args.sub_library_root or args.library_root


def _supplement_command(args):
    """Dispatch the new commands before AssetStore can create any directory."""
    if args.command == "motion":
        from asset_motion import failure_result, run_motion
        try:
            result, code = run_motion(
                _root_from_args(args), args.asset_id,
                start=args.start, end=args.end,
                sample_fps=args.sample_fps,
                analysis_width=args.analysis_width,
                max_seconds=args.max_seconds,
                smoothing_window=args.smoothing_window,
                head_tail_seconds=args.head_tail,
                save=args.save)
        except Exception as exc:  # explicit structured failure, never a traceback
            from asset_motion import MotionCommandError
            error = MotionCommandError("motion_failed", str(exc), status="failed")
            result, code = failure_result(error), 4
        _json(result)
        if code:
            reason = result.get("reason")
            if not reason and isinstance(result.get("save"), dict):
                reason = result["save"].get("reason")
            if not reason and result.get("errors"):
                reason = result["errors"][0].get("message")
            print(f"Reelpack motion: {reason or 'measurement failed'}", file=sys.stderr)
        return code

    from asset_supplements import SupplementError, list_supplements, save_annotations
    try:
        if args.command == "annotate":
            input_path = Path(args.input).expanduser()
            if input_path.is_symlink() or not input_path.is_file():
                raise SupplementError(f"注释输入不是普通文件：{input_path}")
            try:
                payload = json.loads(input_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SupplementError(f"注释输入 JSON 无法读取：{input_path}") from exc
            result = save_annotations(_root_from_args(args), args.asset_id, payload)
        else:
            result = list_supplements(_root_from_args(args), args.asset_id, args.id)
    except (SupplementError, ValueError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    _json(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"motion", "annotate", "supplements"}:
        # These commands validate the existing library themselves.  Constructing
        # AssetStore here would create media/cache directories even for help-like
        # or read-only invocations before validation.
        return _supplement_command(args)
    store = AssetStore(args.sub_library_root or args.library_root)
    try:
        if args.command == "import":
            _json(store.import_file(args.path))
        elif args.command == "list":
            _json({"library_root": str(store.root), "assets": store.list_assets()})
        elif args.command == "show":
            _json(store.get_asset(args.asset_id))
        elif args.command == "shots":
            asset = store.get_asset(args.asset_id)
            if not asset.get("analysis"):
                raise ValueError("资产尚未有 analysis.json")
            _json(shot_evidence_view(asset["analysis"]))
        elif args.command == "query":
            if args.shot:
                if args.start is not None or args.end is not None:
                    raise ValueError("--shot 与时间范围不能同时指定")
                _json(store.query_shot(args.asset_id, args.shot))
            elif args.start is not None and args.end is not None:
                _json(store.query_time(args.asset_id, args.start, args.end))
            else:
                raise ValueError("需要 --shot 或 --start/--end")
        elif args.command in {"frames", "montage"}:
            from frame_board import frame_board
            if args.shot and (args.start is not None or args.end is not None):
                raise ValueError("--shot 与时间范围不能同时指定")
            _json(frame_board(store, args.asset_id, view=args.view, shot=args.shot,
                              start=args.start, end=args.end, timestamp=args.timestamp, grid=args.grid, full_res=args.full_res))
        elif args.command == "set-summary":
            _json(store.set_summary(args.asset_id, args.text))
        elif args.command in {"enrich", "backfill"}:
            from asset_enrich import enrich, compact_audio_result
            result = enrich(store, args.asset_id, args.only, music_backend=args.music_backend,
                            redraw_board=args.redraw_board)
            _json(compact_audio_result(result) if args.only == 'audio' else result)
        elif args.command == "review":
            from asset_enrich import review
            result = review(store, args.asset_id, args.start, args.end, args.output,
                            args.padding, args.interval, args.frames)
            _json(result)
            print((Path(args.output).expanduser() / "review.txt").read_text(encoding="utf-8"), file=sys.stderr)
        elif args.command == "validate":
            result = store.get_asset(args.asset_id)
            analysis = result.get("analysis")
            if analysis is None:
                raise AssetStoreError(f"资产尚未有 analysis.json: {args.asset_id}")
            assert_valid_analysis(analysis)
            _json({"asset_id": args.asset_id, "valid": True, "analysis_version": analysis.get("analysis_version")})
        return 0
    except (AssetStoreError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except (OSError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
