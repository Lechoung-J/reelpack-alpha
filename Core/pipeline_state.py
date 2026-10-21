"""Additive run signatures, capability reuse and actionable result semantics."""
import hashlib
import json
import os
from pathlib import Path

REQUIRED_CAPABILITIES = ('transcription', 'shots', 'ocr')
SIGNATURE_OPTIONS = {
    'language': 'auto', 'model': 'large', 'ocr_backend': 'auto', 'ocr_max_width': 1920,
    'music_backend': 'librosa', 'summary_backend': 'none', 'correction_backend': 'none',
    'vision_model': 'internvl3.5:4b', 'genre_threshold': .05, 'genre_top_k': 3,
    'skip_summary': False, 'rewrite': 'off', 'rewrite_model': 'qwen2.5:7b', 'cloud': False,
}


def run_signature(content_sha256, options):
    values = {name: getattr(options, name, default) if not isinstance(options, dict)
              else options.get(name, default) for name, default in SIGNATURE_OPTIONS.items()}
    def file_identity(value):
        if not value: return None
        path = Path(value).expanduser()
        try:
            info = path.stat()
            return {'path': str(path.resolve()), 'size': info.st_size, 'mtime_ns': info.st_mtime_ns}
        except OSError:
            return {'path': str(path), 'missing': True}
    helper = os.getenv('REELPACK_OCR_BIN', str(Path(__file__).resolve().parent.parent / 'Tools/macos_ocr/macos_ocr'))
    runtime = os.getenv('REELPACK_RUNTIME_ROOT')
    ocr_python = os.getenv('REELPACK_OCR_PYTHON') or (str(Path(runtime) / '.venv-ocr/bin/python') if runtime else None)
    backend_context = {
        'ocr_strategy': 'rapid-probe-v2' if values['ocr_backend'] == 'auto' else values['ocr_backend'],
        'native_ocr': file_identity(helper) if values['ocr_backend'] in ('auto', 'macos') else None,
        'rapidocr_python': file_identity(ocr_python) if values['ocr_backend'] in ('auto', 'rapidocr') else None,
        'tesseract_language': os.getenv('REELPACK_TESSERACT_LANG', 'chi_sim+eng'),
        'rewrite_endpoint_digest': hashlib.sha256((os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1') if values['cloud'] else os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')).encode()).hexdigest(),
        'rewrite_effective_model': (os.getenv('REELPACK_CLOUD_MODEL') or os.getenv('OPENAI_MODEL') or 'gpt-4o') if values['cloud'] else values['rewrite_model'],
        'rewrite_cloud_key_present': bool(os.getenv('OPENAI_API_KEY')) if values['cloud'] else False,
    }
    payload = {'version': 2, 'pipeline_revision': '0.2.0', 'input_sha256': content_sha256, 'options': values,
               'backend_context': backend_context}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return {**payload, 'digest': hashlib.sha256(canonical.encode()).hexdigest()}


def reusable(analysis, signature):
    if not isinstance(analysis, dict) or not isinstance(signature, dict):
        return False
    previous = analysis.get('run_signature')
    if not isinstance(previous, dict) or previous != signature:
        return False
    if analysis.get('execution', {}).get('status', 'complete') != 'complete':
        return False
    capabilities = analysis.get('capabilities') or {}
    sections = {'transcription': 'transcript', 'shots': 'shots', 'ocr': 'ocr'}
    for name in REQUIRED_CAPABILITIES:
        capability = capabilities.get(name, {})
        section = analysis.get(sections[name]) or {}
        if capability.get('status') == section.get('status') == 'complete':
            continue
        if name == 'transcription' and capability.get('status') == section.get('status') == 'not_run':
            track = (analysis.get('video') or {}).get('audio_track') or {}
            if (track.get('status') == 'complete' and track.get('present') is False
                    and capability.get('outcome') == section.get('outcome') == 'no_audio_track'):
                continue
        return False
    return True


DEFAULT_HINTS = {
    'transcription': ('没有可用的带时间转录。', '确认媒体有语音；需要重新分析时使用 reelpack <本地文件> --force。'),
    'shots': ('没有完整镜头边界和代表帧。', '运行 reelpack --runtime-check 检查镜头环境；必要时重新分析本地文件。'),
    'ocr': ('画面文字索引不完整；请核对 evidence/*.jpg。', '运行 reelpack --runtime-check 检查 OCR；必要时重新分析本地文件或直接看证据帧。'),
    'vision': ('本包没有预生成镜头语义；画面判断需读取图片证据。', '直接阅读包内 evidence/*.jpg；按镜头查询可用 Core/asset_cli.py shots。'),
    'music': ('节拍、响度或音色测量不可用；不能据此判断音乐。', 'reelpack enrich <asset_id> --only audio --output-dir <原输出目录>'),
    'audio_board': ('没有音频证据图；已有音频数字仍可使用。', 'reelpack enrich <asset_id> --only audio --redraw-board --output-dir <原输出目录>'),
    'tagging': ('没有可用曲风/情绪候选；物理音频测量状态独立。', '先运行 reelpack --runtime-check 查看 BYO 模型；补算前告知下载成本，再用 --music-backend panns。'),
    'genre': ('曲风候选未运行或失败，不代表没有曲风。', '先核对 music.tagging 的模型状态；明确同意成本后启用 PANNs。'),
    'mood': ('情绪候选未运行或失败，不代表没有情绪。', '先依据已保存的测量和证据作有限解释；需要模型时先告知成本。'),
    'events': ('音频事件标签未运行或失败。', '核对 music.tagging 的模型与后端状态，不把缺标签解释为事件不存在。'),
    'summary': ('没有模型生成的摘要，转录与证据仍可阅读。', '请 Agent 读取转录、OCR 和证据帧后回答；可用 asset_cli set-summary 写回。'),
    'correction': ('转录未经过模型校验，原文误识仍可能存在。', '结合 OCR 和证据帧核对原文；需要模型时显式选择 --correction-backend。'),
    'cut_alignment': ('拍点/切点对齐尚未测得，不能解释为 0%。', '先成功计算拍点，再补跑音频；不要把空节拍数组当作无音乐。'),
}


def actionable(value, name):
    """Preserve backend results, adding hints only to actual state objects."""
    if not isinstance(value, dict):
        return value
    status = value.get('status')
    if status in ('failed', 'not_run', 'not_measured', 'unknown', 'media_offline'):
        impact, remedy = DEFAULT_HINTS.get(name, ('此项结果不可用或未确认。', '检查对应后端的原因与运行条件。'))
        value.setdefault('reason', value.get('error') or
                         ('按当前配置未运行此能力。' if status == 'not_run' else '该项没有有效测量或执行失败。'))
        value.setdefault('impact', impact)
        value.setdefault('remedy', remedy)
        if not any(token in value['remedy'] for token in ('reelpack ', 'python ', 'bash ', 'export ')):
            command = ('reelpack enrich <asset_id> --only audio --redraw-board --library-root <库>'
                       if name in ('audio_board', 'board_data') else
                       'reelpack enrich <asset_id> --only audio --library-root <库>'
                       if name in ('music', 'cut_alignment', 'beats', 'timbre', 'loudness') else
                       'reelpack --runtime-check')
            value['remedy'] += '\n命令模板（替换资产/库参数后执行）: ' + command
    for key, child in value.items():
        if isinstance(child, dict):
            actionable(child, key)
    return value


def annotate_analysis(analysis):
    for name, capability in (analysis.get('capabilities') or {}).items():
        actionable(capability, name)
    for name in ('transcript', 'ocr', 'shots', 'music', 'summary'):
        actionable(analysis.get(name), 'transcription' if name == 'transcript' else name)
    return analysis


def capability_summary(analysis):
    capabilities = analysis.get('capabilities') or {}
    items = []
    for name in ('transcription', 'shots', 'ocr', 'music', 'vision'):
        value = dict(capabilities.get(name) or {'status': 'unknown'})
        actionable(value, name)
        items.append({'name': name, **{key: value.get(key) for key in
                     ('status', 'backend', 'model', 'reason', 'impact', 'remedy', 'outcome', 'fallback_from')}})
    return {'complete': sum(item['status'] == 'complete' for item in items),
            'total': len(items), 'capabilities': items,
            'audio_board': (analysis.get('music') or {}).get('audio_board'),
            'tagging': (analysis.get('music') or {}).get('tagging'),
            'rewrite': {k:v for k,v in ((analysis.get('transcript') or {}).get('rewrite') or {}).items()
                        if k in ('status','backend','model','reason','params')}}


def summary_json(analysis):
    return json.dumps(capability_summary(analysis), ensure_ascii=False)
