"""Shared human-readable sections for exported and package Markdown."""
import math


def screen_text_section(ocr):
    ocr = ocr or {}
    lines = ['## 画面文字', '']
    seen = set()
    for item in ocr.get('items') or []:
        text = str(item.get('text') or '').strip()
        key = ' '.join(text.split())
        if not key or key in seen:
            continue
        seen.add(key)
        start = item.get('timestamp', item.get('start'))
        label = f'[{start:.2f}s]  ' if isinstance(start, (int, float)) and math.isfinite(start) else ''
        lines.append(f'- {label}{text}')
    if not seen:
        lines.append('（无可见文字）' if ocr.get('status') == 'complete' else '（画面文字未运行或识别不可用）')
    if ocr.get('backend'):
        lines += ['', f"识别后端：`{ocr['backend']}`；状态：`{ocr.get('status', 'unknown')}`"]
    if ocr.get('status') in ('partial', 'failed', 'not_run') and ocr.get('reason'):
        lines += [str(ocr['reason'])]
    return '\n'.join(lines)


def rewrite_section(rewrite):
    value = rewrite or {'status': 'not_run', 'reason': '未整理'}
    lines = ['## 整理稿', '']
    if value.get('status') == 'complete':
        lines.append(str(value.get('text') or '（无转录内容）'))
        lines += ['', '整理稿为模型辅助编辑；原始转录保留，疑似术语应结合原片复核。']
    else:
        lines.append('（未整理／整理不可用；请以原始转录为准）')
        if value.get('reason'): lines.append(str(value['reason']))
    lines += ['', f"整理状态：`{value.get('status','not_run')}`；模式：`{value.get('params',{}).get('mode','off')}`"]
    edits = value.get('edits') or []
    if edits:
        lines += ['', '### 整理改动记录', '']
        for edit in edits:
            lines.append(f"- {edit.get('segment_id','')}：{edit.get('before','')} → {edit.get('after','')}；{edit.get('reason','语义整理')}")
    return '\n'.join(lines)
