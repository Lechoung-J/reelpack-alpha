#!/usr/bin/env python3
"""
Reelpack Core Pipeline — analyze local files, transcribe, vision-correct, and output.
Acts as a subprocess for the SwiftUI app and CLI. Communicates via NDJSON on stdout.

Protocol v1 — every output line is a JSON object with "protocol_version": 1.

Usage:
    python3 core.py --input <file> [--output-dir ~/Downloads/Reelpack]
                    [--model large] [--cloud] [--pdf]
                    [--vision-model internvl3.5:4b]

Vision model (via Ollama):
    - internvl3.5:4b — lightweight, ~6GB RAM, good OCR
    Setup InternVL3.5: see Core/internvl_setup.md
"""
from __future__ import annotations

import argparse
import base64
import copy
import gc
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Core is launched both as a bundled script and directly from the repository.
CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from runtime_environment import prepare_pipeline_environment
os.environ.update(prepare_pipeline_environment(
    create=os.environ.get('REELPACK_READONLY_CHECK') != '1'))

from asset_contract import empty_analysis
from asset_store import AssetStore, sha256_file
from pipeline_state import run_signature, reusable, annotate_analysis, summary_json
from ocr_sampling import merge_observations, sampling_candidates
from ocr_regions import REGION_SUPPORT, ensure_ocr_regions, image_size as ocr_image_size
from ocr_conflicts import find_conflicts
from subtitle_trigger import (DEFAULT_BAND, DEFAULT_CHANGED_PIXEL_COUNT,
                              DEFAULT_CHANGED_PIXEL_DELTA, DEFAULT_MEAN_ABS_DIFF_THRESHOLD,
                              DEFAULT_SIGNATURE_SIZE, band_signatures, count_signature_changes,
                              duplicate_text_timestamps, extract_band_frames, subtitle_band_times,
                              triggered_times_from_signatures)
from shot_detect import detect_shots
from audio_alignment import align_cuts_to_beats

try:
    import cv2
    import numpy as np
except ImportError:  # Imported lazily for media analysis only.
    cv2 = None
    np = None
import requests
try:
    from faster_whisper import WhisperModel
except ImportError:  # Storage/query commands do not need Whisper.
    WhisperModel = None
try:
    import speaker_campp
except ImportError:  # The experimental speaker switch reports failed, never crashes.
    speaker_campp = None

PROTOCOL_VERSION = 1

# ── Config ──
OLLAMA_BASE = "http://localhost:11434"
FRAME_INTERVAL = 5  # seconds between keyframes
MAX_FRAMES = 30
FRAME_WIDTH = 1920  # OCR samples come from original media; cap width without upscaling
# Per-profile coarse subtitle-band scan; ad is fast-paced, so it samples the
# shortest interval.  auto is intentionally absent and keeps the legacy pass.
PROFILE_COARSE_INTERVAL = {'talk': 30.0, 'podcast': 60.0, 'ad': 10.0}
SUBTITLE_BAND = DEFAULT_BAND
SUBTITLE_SIGNATURE_SIZE = DEFAULT_SIGNATURE_SIZE
SUBTITLE_DENSIFY_WINDOW = 1.0
SUBTITLE_DENSIFY_STEP = 0.5


def _subtitle_trigger_params() -> dict:
    """Tunable trigger thresholds, recorded in ``ocr.sampling.params``."""
    return {
        'trigger': 'subtitle_band_image',
        'signature_size': list(SUBTITLE_SIGNATURE_SIZE),
        'mean_abs_diff_threshold': DEFAULT_MEAN_ABS_DIFF_THRESHOLD,
        'changed_pixel_delta': DEFAULT_CHANGED_PIXEL_DELTA,
        'changed_pixel_count_threshold': DEFAULT_CHANGED_PIXEL_COUNT,
        'densify_window': SUBTITLE_DENSIFY_WINDOW,
        'densify_step': SUBTITLE_DENSIFY_STEP,
        'text_comparison': 'dedupe_only',
    }

# ── NDJSON Output ──

def emit(msg: dict):
    msg["protocol_version"] = PROTOCOL_VERSION
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def emit_progress(stage: str, percent: int, message: str):
    emit({"type": "progress", "stage": stage, "percent": percent, "message": message})


def emit_complete(output: dict):
    emit({"type": "complete", "output": output})


def emit_error(stage: str, code: str, message: str):
    emit({"type": "error", "stage": stage, "code": code, "message": message})


# ── Dependency Check ──

def check_dependencies():
    missing = []
    for cmd in ["ffmpeg", "ffprobe"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            missing.append(cmd)
    return missing


# ── Input Resolution ──

def resolve_input(input_path: str) -> str:
    """Validate a local video file without resolving or downloading URLs."""
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", input_path.lstrip()):
        raise ValueError("仅支持本地视频文件路径，请先将视频保存到本地")
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if not path.is_file():
        raise ValueError(f"输入必须是本地文件: {path}")
    if path.suffix.lower() not in [".mp4", ".mov", ".mkv"]:
        raise ValueError(f"不支持的格式: {path.suffix}")
    return str(path)


# ── Audio Transcription (faster-whisper) ──

def transcribe_audio(video_path: str, model_name: str = "base", language: str = "auto", *, return_info: bool = False):
    """Transcribe audio using faster-whisper; optionally return detected language metadata."""
    if WhisperModel is None:
        raise RuntimeError("未安装 faster-whisper；请先安装 Core/requirements.txt 中的依赖")
    emit_progress("transcribe", 70, f"加载 Whisper {model_name} 模型...")

    resolved_model = model_name
    if model_name in ('large', 'large-v3'):
        from runtime_environment import inspect_runtime_roots
        prepared_model = inspect_runtime_roots()['models']['whisper']
        if not prepared_model['ready']:
            emit_progress('model_preflight', 70,
                          'Whisper large-v3 尚未本地准备，权重约 3.1 GB；不会静默下载。请先告知用户成本并确认，再按安装指引预下载。')
            raise RuntimeError('Whisper large-v3 本地模型未就绪；没有自动下载。运行 reelpack --runtime-check 查看持久模型根，然后按 install.sh 提示显式准备模型。')
        resolved_model = prepared_model['path']
    # A prepared local snapshot can be used read-only without Hub network writes.
    model = WhisperModel(resolved_model, device="auto", compute_type="int8")

    emit_progress("transcribe", 72, "正在语音转录...")
    transcribe_options = {
        "vad_filter": True,
        "condition_on_previous_text": False,
    }
    warnings = []
    try:
        from faster_whisper.vad import VadOptions
        vad_parameters = {"min_silence_duration_ms": 500, "speech_pad_ms": 400}
        VadOptions(**vad_parameters)  # Validate against the installed Silero API.
        transcribe_options["vad_parameters"] = vad_parameters
    except (ImportError, TypeError) as exc:
        warnings.append(f"Silero custom parameters unavailable; using VAD defaults: {exc}")
        emit_progress("transcribe", 72, warnings[-1])
    if language != "auto":
        transcribe_options["language"] = language
    duration = (_probe_video_metadata(video_path) or {}).get("duration_seconds")
    if duration is None or not math.isfinite(duration) or duration <= 0:
        duration = None
        warnings.append("Media duration unavailable; timestamp boundary enforcement unavailable")
        emit_progress("transcribe", 72, warnings[-1])
    segments_iter, info = model.transcribe(video_path, **transcribe_options)

    segments = []
    dropped = clipped = 0
    for seg in segments_iter:
        start, end = float(seg.start), float(seg.end)
        if not math.isfinite(start) or not math.isfinite(end):
            dropped += 1
            continue
        if duration is not None and start >= duration:
            dropped += 1
            continue
        bounded_start = max(0.0, start)
        bounded_end = min(end, duration) if duration is not None else end
        if bounded_end <= bounded_start:
            dropped += 1
            continue
        clipped += int(bounded_start != start or bounded_end != end)
        segments.append({"start": bounded_start, "end": bounded_end, "text": seg.text.strip()})

    # Free Whisper model memory before loading vision model
    del model
    gc.collect()

    lang = info.language
    emit_progress("transcribe", 78, f"转录完成 ({lang}, {len(segments)} 片段)")

    metadata = {
        "language": lang, "language_probability": getattr(info, "language_probability", None),
        "params": {**transcribe_options, "device": "auto", "compute_type": "int8"},
        "timestamp_bounds": {"duration_seconds": duration, "dropped_segments": dropped,
                             "clipped_segments": clipped},
        "warnings": warnings,
    }
    return (segments, metadata) if return_info else segments


def probe_audio_track(path):
    """An empty successful streams list is absence; a probe error is not absence."""
    completed = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a',
                                '-show_entries', 'stream=index', '-of', 'json', str(path)],
                               capture_output=True, text=True, timeout=30, check=True)
    value = json.loads(completed.stdout)
    if not isinstance(value.get('streams'), list):
        raise ValueError('ffprobe audio stream response is invalid')
    return {'status': 'complete', 'present': bool(value['streams'])}


def check_transcription_model(model):
    """Metadata preflight before expensive work, never model loading/downloading."""
    if model in ('large', 'large-v3'):
        from runtime_environment import inspect_runtime_roots
        if not inspect_runtime_roots()['models']['whisper']['ready']:
            raise RuntimeError('Whisper large-v3 未本地准备（约 3.1 GB）；请显式准备模型，没有自动下载。')


def audio_track_absent(shots):
    from audio_alignment import align_cuts_to_beats
    reason = 'no_audio_track'
    result = {'status': 'not_run', 'outcome': reason, 'reason': reason,
              'backend': None, 'tempo_bpm': None, 'beat_count': 0, 'beat_times_seconds': [],
              'impact': '媒体无音轨，音频测量和语音识别不适用；画面能力仍可用。',
              'remedy': '若需要音频，请提供含音轨的本地媒体；无需重试无音轨识别。'}
    result['cut_alignment'] = align_cuts_to_beats([s['start'] for s in shots], [])
    for name in ('genre', 'mood', 'audio_board'):
        result[name] = {'status': 'not_run', 'outcome': reason, 'reason': reason,
                        'impact': result['impact'], 'remedy': result['remedy']}
    return result


# ── Experimental Speaker Separation (CAM++) ──

def _speaker_state(status, *, backend=None, model=None, reason=None):
    """Minimal, contract-valid speakers block for non-complete outcomes."""
    state = {"status": status, "backend": backend, "model": model,
             "experimental": True, "segments": [], "confidence": {}}
    if reason:
        state["reason"] = reason
    return state


def _speaker_segments(windows, segments, duration=None):
    """Label transcript segments by the majority-overlap CAM++ window cluster.

    Window and segment intervals are clamped to ``[0, duration]`` first; a
    segment without any positive overlap keeps ``speaker=None`` instead of
    guessing. Ties between clusters fall back to the lowest (earliest) label.
    """
    def bounds(item):
        try:
            start, end = float(item.get("start")), float(item.get("end"))
        except (TypeError, ValueError, AttributeError):
            return None
        if not math.isfinite(start) or not math.isfinite(end):
            return None
        if end < start:
            start, end = end, start
        if duration is not None:
            start = max(0.0, min(float(duration), start))
            end = max(0.0, min(float(duration), end))
        else:
            start = max(0.0, start)
        return (start, end) if end > start else None

    usable = []
    for window in windows or []:
        interval = bounds(window)
        try:
            cluster = int(window.get("cluster"))
        except (TypeError, ValueError, AttributeError):
            continue
        if interval is not None and cluster >= 0:
            usable.append((interval[0], interval[1], cluster))

    output = []
    for segment in segments or []:
        interval = bounds(segment)
        if interval is None:
            continue
        start, end = interval
        overlaps = {}
        for window_start, window_end, cluster in usable:
            overlap = min(end, window_end) - max(start, window_start)
            if overlap > 0:
                overlaps[cluster] = overlaps.get(cluster, 0.0) + overlap
        speaker = None
        if overlaps:
            cluster = max(overlaps, key=lambda value: (overlaps[value], -value))
            speaker = f"S{cluster + 1}"
        output.append({"start": round(start, 3), "end": round(end, 3),
                       "speaker": speaker, "text": str(segment.get("text") or "")})
    return output


def _analyze_speakers(video_path, segments, duration=None):
    """Run CAM++ on a temporary 16 kHz mono WAV and map labels to segments.

    Never raises and never downloads a model: a missing local model comes back
    as ``not_run`` / ``model_missing``; every other failure carries ``reason``.
    The temporary WAV is deleted when the block exits, before any result is used.
    """
    if speaker_campp is None:
        return _speaker_state("failed", reason="speaker_campp unavailable")
    try:
        with tempfile.TemporaryDirectory(prefix="reelpack-speakers-") as temp_dir:
            wav_path = Path(temp_dir) / "speaker-input.wav"
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path),
                "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", "-y", str(wav_path),
            ], check=True, capture_output=True, timeout=900)
            result = speaker_campp.cluster_speakers(str(wav_path))
    except Exception as exc:  # noqa: BLE001 - optional stage must not abort the package
        detail = getattr(exc, "stderr", None) or str(exc)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        return _speaker_state("failed", backend=speaker_campp.BACKEND,
                              reason=f"{type(exc).__name__}: {str(detail)[-800:]}")

    status = result.get("status", "failed")
    confidence = result.get("confidence") or {}
    state = _speaker_state(
        status,
        backend=result.get("backend") or speaker_campp.BACKEND,
        model=(result.get("params") or {}).get("embedding_model"),
        reason=result.get("reason"),
    )
    for key in ("selected_k", "silhouette"):
        if confidence.get(key) is not None:
            state["confidence"][key] = confidence[key]
    if status == "failed" and not state.get("reason"):
        state["reason"] = "speaker clustering failed without a reason"
    if status == "complete":
        state["segments"] = _speaker_segments(result.get("windows"), segments, duration)
    return state


# ── Frame Extraction ──

def has_video_track(file_path: str) -> bool:
    # OpenCV may fail to open some valid macOS/MP4 combinations; ffprobe is the
    # authoritative fallback for deciding whether the pipeline should run video stages.
    if cv2 is not None:
        cap = cv2.VideoCapture(file_path)
        has = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        cap.release()
        if has:
            return True
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                 "-show_entries", "stream=codec_type", "-of", "default=nw=1", file_path],
                                capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and "codec_type=video" in result.stdout
    except OSError:
        return False


def extract_frames(file_path: str, interval: int = FRAME_INTERVAL, max_frames: int = MAX_FRAMES,
                   max_width: int = FRAME_WIDTH) -> list[tuple[float, bytes]]:
    """Sample OCR frames from original media, independently of the shot proxy.

    max_width=0 retains source resolution. Never upscale an already small source.
    The FFmpeg fallback uses the same half-interval timestamps as OpenCV.
    """
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python；无法提取视频帧")
    if interval <= 0 or max_frames <= 0 or max_width < 0:
        raise ValueError("invalid OCR frame sampling parameters")
    cap = cv2.VideoCapture(file_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        readable = cap.isOpened() and fps > 0 and total_frames > 0
        duration = total_frames / fps if readable else float(
            _probe_video_metadata(file_path).get("duration_seconds") or 0)
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("无法读取原片时长；不能生成 OCR 采样帧")
        actual_interval = max(interval, duration / max_frames)
        frames, prev_hash = [], None
        t = actual_interval / 2
        while t < duration and len(frames) < max_frames:
            ok, frame = False, None
            if readable:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                ok, frame = cap.read()
            if not ok:
                decoded = subprocess.run([
                    "ffmpeg", "-nostdin", "-v", "error", "-ss", str(t),
                    "-i", file_path, "-frames:v", "1", "-f", "image2pipe",
                    "-c:v", "png", "pipe:1",
                ], capture_output=True, check=True, timeout=60)
                frame = cv2.imdecode(np.frombuffer(decoded.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"无法解码原片 OCR 帧: {t:.3f}s")
            h = _compute_phash(frame)
            if prev_hash is None or np.count_nonzero(h != prev_hash) >= 4:
                height, width = frame.shape[:2]
                if max_width and width > max_width:
                    frame = cv2.resize(frame, (max_width, max(1, round(height * max_width / width))),
                                       interpolation=cv2.INTER_AREA)
                encoded, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not encoded:
                    raise RuntimeError(f"无法编码 OCR 帧: {t:.3f}s")
                frames.append((t, buf.tobytes()))
                prev_hash = h
            t += actual_interval
        return frames
    finally:
        cap.release()


def _compute_phash(img: np.ndarray) -> np.ndarray:
    """Perceptual hash: 32x32 grayscale → DCT → 8x8 low-freq → threshold at mean."""
    small = cv2.resize(img, (32, 32))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))[:8, :8]
    return (dct > dct.mean()).flatten()


# ── Vision: Frame Description + OCR Extraction ──

FRAME_OCR_PROMPT = """请严格提取这张视频画面中实际存在的可见文字。

规则：
1. 只提取画面中真实出现的文字：PPT文字、字幕、标题条、图表标注、LOGO文字
2. 原样输出，不要翻译、不要解释、不要补充
3. 如果没有可见文字，请严格回复 "（无可见文字）"
4. 不要描述画面内容，不要猜测视频主题"""


def describe_frame(img_bytes: bytes, vision_model: str) -> str:
    """Call Ollama vision model to extract visible text from a frame."""
    img_b64 = base64.b64encode(img_bytes).decode()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": vision_model,
                "prompt": FRAME_OCR_PROMPT,
                "images": [img_b64],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"(帧 OCR 失败: {e})"


def describe_frames_macos(frames: list[tuple[float, bytes]], ocr_bin: str) -> list[tuple[float, str]]:
    """Run the macOS Vision OCR helper once for all extracted frames."""
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = []
        for index, (_, img_bytes) in enumerate(frames):
            path = Path(temp_dir) / f"frame_{index:03d}.jpg"
            path.write_bytes(img_bytes)
            paths.append(str(path))

        result = subprocess.run(
            [ocr_bin, "--json", *paths],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"退出码 {result.returncode}"
            raise RuntimeError(f"macOS Vision OCR 失败: {detail}")
        try:
            json_start = result.stdout.find("{")
            payload = json.loads(result.stdout[json_start:] if json_start >= 0 else result.stdout)
        except json.JSONDecodeError as exc:
            detail = result.stderr.strip()
            raise RuntimeError(f"macOS Vision OCR 返回无效 JSON{': ' + detail if detail else ''}") from exc

        ocr_frames = payload.get("frames")
        if not isinstance(ocr_frames, list) or len(ocr_frames) != len(frames):
            raise RuntimeError("macOS Vision OCR 返回的帧数量不匹配")

        output = []
        for (timestamp, _), item in zip(frames, ocr_frames):
            if not isinstance(item, dict):
                raise RuntimeError("macOS Vision OCR 返回了无效帧结果")
            if item.get("error"):
                raise RuntimeError(str(item["error"]))
            output.append((timestamp, str(item.get("text") or "")))
        return output


def optional_macos_ocr(frames, ocr_bin):
    """OCR failure never aborts transcription/music/package persistence."""
    try:
        return describe_frames_macos(frames, ocr_bin), None
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return [], f"{type(exc).__name__}: {exc}"


# ── Transcript Correction ──

CORRECT_PROMPT = """你是专业的多语言语音转录校对助手。请根据画面中实际可见文字（OCR 结果），校正转录中的明确识别错误。

OCR 提取的可见文字（画面中出现的专有名词、术语、文字）：
{ocr_terms}

原始语音转录稿：
{raw_transcript}

修正规则：
1. 如果语音转录的某个词明显与 OCR 文字不一致，修正为 OCR 中的正确写法（如 "微信是小号" → "微信视频号"）
2. 仅修正专有名词、技术术语、数字等 OCR 中明确出现的文字
3. 不要改变说话者的措辞、口语表达、语气词
4. 不确定的修正请在原文基础上用 [校正: 原词 → 建议词] 标记
5. 保持转录原文的语言、原始分段和标点；不得翻译、不得切换语言。OCR 若为另一种语言，只能用于确认相同的专有名词/数字，不得据此翻译或改写原文

输出格式：直接输出修正后的完整转录稿，不加前缀和后缀。"""

CLOUD_CORRECT_PROMPT = CORRECT_PROMPT.replace(
    "5. 保持转录原文的语言、原始分段和标点；不得翻译、不得切换语言。OCR 若为另一种语言，只能用于确认相同的专有名词/数字，不得据此翻译或改写原文",
    "5. 保持转录原文的语言、原始分段及每行时间戳标记（如有）；不得翻译或切换语言。"
    "OCR 若为另一种语言，只能用于确认相同专有名词/数字，不得翻译或改写原文。"
    "可补充适合原语言的标点，不得增删原文。",
)


def correct_transcript(raw_transcript: str, ocr_terms: str, model_name: str = "internvl3.5:4b", use_cloud: bool = False, language: str = "auto") -> str:
    """Use LLM (local Ollama or cloud) to correct transcription errors using OCR terms."""
    if not ocr_terms or ocr_terms.strip() in ["", "（无可见文字）"]:
        return raw_transcript

    emit_progress("transcribe", 85, "正在视觉校正...")

    template = CLOUD_CORRECT_PROMPT if use_cloud else CORRECT_PROMPT
    prompt = template.format(ocr_terms=ocr_terms[:3000], raw_transcript=raw_transcript[:8000])

    prompt += f"\n\n转录语言代码：{language}。输出必须保持该语言；如为 auto，则严格保持原稿语言。"
    result = _cloud_correct(prompt) if use_cloud else _ollama_generate(prompt, model_name)
    # Model/network failure markers must never be persisted as transcript text.
    if not result or result.startswith(("[", "(", "（")):
        return raw_transcript
    return result


def _ollama_generate(prompt: str, model: str) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[校正失败: {e}]"


def _cloud_correct(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "[云端校正不可用: 未设置 OPENAI_API_KEY]"
    try:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("REELPACK_CLOUD_MODEL", "gpt-4o")
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[云端校正失败: {e}]"


# ── Summary Generation ──

SUMMARY_PROMPT = """请用下方转录稿的原语言撰写简洁摘要（约150词以内）。不要翻译、不要切换语言，不要添加标题或前缀。

校正后的转录稿：
{transcript}"""


def generate_summary(transcript: str, model_name: str = "internvl3.5:4b", use_cloud: bool = False, language: str = "auto") -> str:
    emit_progress("output", 90, "正在生成摘要...")
    prompt = SUMMARY_PROMPT.format(transcript=transcript[:6000]) + f"\n\n转录语言代码：{language}；摘要必须保持转录稿语言。"

    if use_cloud:
        return _cloud_summary(prompt)
    else:
        text = _ollama_generate(prompt, model_name)
        return text if text and not text.startswith(("[", "(摘要", "（摘要")) else ""


def _cloud_summary(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return ""
    try:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("REELPACK_CLOUD_MODEL", "gpt-4o")
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.5},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return ""


# ── Output ──

_TIMED_TRANSCRIPT_RE = re.compile(
    r"^\s*\[(?P<start>\d+):(\d+(?:\.\d+)?)-(?P<end>\d+):(\d+(?:\.\d+)?)\]\s*(?P<text>.*)$"
)
_ANY_TRANSCRIPT_TIME_RE = re.compile(r"\[\d+:\d+(?:\.\d+)?-\d+:\d+(?:\.\d+)?\]")
_ORDINAL_START_RE = re.compile(r"^第[一二三四五六七八九十]个")
_CJK_RE = r"[\u3400-\u9fff]"


def _transcript_segments(transcript):
    """Normalize timestamped text or segment dictionaries to (start, end, text)."""
    if isinstance(transcript, (list, tuple)):
        output = []
        for item in transcript:
            if isinstance(item, dict):
                start = item.get("start")
                end = item.get("end")
                text = item.get("text", "")
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                start, end, text = item[0], item[1], item[2]
            else:
                start, end, text = None, None, str(item)
            output.append((float(start) if start is not None else None,
                           float(end) if end is not None else None,
                           str(text).strip()))
        return output

    output = []
    for line in str(transcript).splitlines():
        if not line.strip():
            continue
        match = _TIMED_TRANSCRIPT_RE.match(line)
        if match:
            start = int(match.group("start")) * 60 + float(match.group(2))
            end = int(match.group("end")) * 60 + float(match.group(4))
            output.append((start, end, match.group("text").strip()))
        else:
            output.append((None, None, _ANY_TRANSCRIPT_TIME_RE.sub("", line).strip()))
    return output


def _append_transcript_text(current, text):
    """Join adjacent spoken fragments and space CJK/alphanumeric boundaries."""
    joined = current + text
    joined = re.sub(rf"({_CJK_RE})(?=[A-Za-z0-9])", r"\1 ", joined)
    joined = re.sub(rf"([A-Za-z0-9])(?={_CJK_RE})", r"\1 ", joined)
    return joined


def format_transcript_paragraphs(transcript):
    """Render transcript fragments as readable paragraphs without timestamps.

    Accepts the timestamped Markdown text emitted by the pipeline, or a sequence
    of ``{start, end, text}`` dictionaries / ``(start, end, text)`` tuples.
    """
    segments = _transcript_segments(transcript)
    paragraphs = []
    current = ""

    for index, (start, end, text) in enumerate(segments):
        if not text:
            continue
        if current and _ORDINAL_START_RE.match(text):
            paragraphs.append(current)
            current = ""

        current = _append_transcript_text(current, text)
        next_start = segments[index + 1][0] if index + 1 < len(segments) else None
        gap = next_start - end if next_start is not None and end is not None else None
        ends_sentence = bool(re.search(r"[。！？!?…]$", text))
        if len(current) >= 260 or (
            len(current) >= 170 and (ends_sentence or gap is None or gap >= 0.6)
        ):
            paragraphs.append(current)
            current = ""

    if current:
        paragraphs.append(current)

    merged = []
    for paragraph in paragraphs:
        if len(paragraph) < 40 and merged:
            merged[-1] = _append_transcript_text(merged[-1], paragraph)
        else:
            merged.append(paragraph)
    return "\n\n".join(merged)

def output_markdown(summary: str, transcript: str, video_path: str, output_dir: str, has_audio: bool = True, *, original=None, rewrite=None, ocr=None) -> str:
    emit_progress("output", 95, "正在输出 Markdown...")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    md_path = out_dir / f"{video_name}.md"

    audio_section = ""
    if has_audio:
        transcript = format_transcript_paragraphs(transcript)
        audio_section = f"""
## 口播原文

{transcript}
"""

    content = f"""# {video_name}

## 视频摘要

{summary}
{audio_section}
"""
    if original is not None:
        from readable_sections import rewrite_section, screen_text_section
        content = (f'# {video_name}\n\n## 视频摘要\n\n{summary}\n\n## 原始转录\n\n'
                   + (str(original) if original else '（未检出语音／无音轨）') + '\n\n'
                   + rewrite_section(rewrite) + '\n\n' + screen_text_section(ocr) + '\n')
    else:
        from readable_sections import screen_text_section
        content += '\n' + screen_text_section(ocr) + '\n'
    md_path.write_text(content, encoding="utf-8")
    return str(md_path)


def output_pdf(md_path: str):
    """Convert Markdown to PDF via Pandoc."""
    pdf_path = md_path.replace(".md", ".pdf")
    subprocess.run(["pandoc", md_path, "-o", pdf_path, "--pdf-engine=xelatex",
                    "-V", "mainfont=Noto Sans SC", "-V", "geometry=margin=1in"],
                   capture_output=True)
    return pdf_path if os.path.exists(pdf_path) else None



def _probe_video_metadata(video_path: str) -> dict:
    """Best-effort metadata; unavailable fields remain null instead of failing the package."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,r_frame_rate,codec_name", "-of", "json", video_path],
            capture_output=True, text=True, timeout=30,
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else {}
        streams = payload.get("streams", [])
        video = next((item for item in streams if item.get("width") and item.get("height")), {})
        rate = video.get("r_frame_rate", "")
        fps = None
        if "/" in rate:
            numerator, denominator = rate.split("/", 1)
            if float(denominator):
                fps = float(numerator) / float(denominator)
        return {
            "duration_seconds": float(payload.get("format", {}).get("duration")) if payload.get("format", {}).get("duration") else None,
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": fps,
            "codec": video.get("codec_name"),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"duration_seconds": None, "width": None, "height": None, "fps": None, "codec": None}


def _transcript_coverage_seconds(segments, duration: float):
    """Union of transcribed intervals, clipped to the media duration."""
    intervals = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            start, end = float(segment.get("start")), float(segment.get("end"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        if end < start:
            start, end = end, start
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return None
    intervals.sort()
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total


def _ocr_text_entries(ocr_items) -> list[tuple[float, int, str]]:
    """Non-empty OCR texts ordered by timestamp, with whitespace normalized."""
    entries = []
    for index, item in enumerate(ocr_items or []):
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        timestamp = item.get("timestamp", item.get("start"))
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            timestamp = float(index)
        if not math.isfinite(timestamp):
            timestamp = float(index)
        entries.append((timestamp, index, " ".join(text.split())))
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    return entries


def compute_content_hints(shots=None, segments=None, ocr_items=None,
                          duration_seconds=None, *, frame_board=None) -> dict:
    """Evidence-only hints for downstream content-type judgment.

    Missing or unusable inputs stay null; a failed computation is never
    reported as a real zero. Only shots_per_min may legitimately be 0.0, and
    only when shot detection completed and reported an empty list.
    """
    try:
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration <= 0:
            duration = None
    except (TypeError, ValueError):
        duration = None

    shots_per_min = None
    if duration is not None and isinstance(shots, dict) and shots.get("status") == "complete":
        items = shots.get("items")
        if isinstance(items, list):
            shots_per_min = round(len(items) / (duration / 60.0), 4)

    speech_ratio = None
    if duration is not None and isinstance(segments, list) and segments:
        coverage = _transcript_coverage_seconds(segments, duration)
        if coverage is not None:
            speech_ratio = round(max(0.0, min(1.0, coverage / duration)), 4)

    ocr_change_rate = None
    entries = _ocr_text_entries(ocr_items) if isinstance(ocr_items, list) else []
    if entries:
        changes = sum(1 for previous, current in zip(entries, entries[1:])
                      if previous[2] != current[2])
        ocr_change_rate = round(changes / len(entries), 4)

    return {
        "shots_per_min": shots_per_min,
        "speech_ratio": speech_ratio,
        "ocr_change_rate": ocr_change_rate,
        "frame_board": frame_board if isinstance(frame_board, str) and frame_board else None,
        "duration_seconds": round(duration, 3) if duration is not None else None,
    }


def _existing_frame_board(archive) -> str | None:
    """Record the package board only when it actually exists on disk."""
    package_dir = (archive or {}).get("package_dir") if isinstance(archive, dict) else None
    if not package_dir:
        return None
    try:
        if (Path(package_dir) / "assets" / "frame-board.jpg").is_file():
            return "assets/frame-board.jpg"
    except OSError:
        return None
    return None


def _ocr_items_from_lines(ocr_terms: str) -> list[dict]:
    items = []
    for index, line in enumerate(ocr_terms.splitlines()):
        match = re.match(r"^\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$", line.strip())
        if not match or not match.group(3).strip():
            continue
        timestamp = int(match.group(1)) * 60 + float(match.group(2))
        items.append({
            "ocr_id": f"ocr_{index + 1}",
            "timestamp": timestamp,
            "start": timestamp,
            "end": timestamp,
            "text": match.group(3).strip(),
            "source": "frame_ocr",
            "evidence_ids": [],
            "uncertain": False,
        })
    return items


def _frame_size_map(frames) -> dict:
    """Map exact sampling timestamps to decoded image sizes, for region_norm."""
    sizes = {}
    for timestamp, image in frames or []:
        try:
            sizes[round(float(timestamp), 6)] = ocr_image_size(image)
        except (TypeError, ValueError):
            continue
    return sizes


def _write_ocr_frames(frames, package_dir, evidence, start_index=0) -> dict:
    """Persist OCR input frames as evidence and return timestamp -> evidence ids."""
    refs = {}
    for index, (timestamp, image) in enumerate(frames, start_index):
        evidence_id = f"ocr_frame_{index:05d}"
        relative = f"evidence/{evidence_id}.jpg"
        target = Path(package_dir) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(image)
        evidence.append({"evidence_id": evidence_id, "kind": "image", "timestamp": timestamp,
                         "relative_path": relative})
        refs[timestamp] = [evidence_id]
    return refs


def _recognize_ocr(args, frames, refs) -> dict:
    """Run the configured OCR backend once, normalizing every result shape."""
    backend = args.ocr_backend
    if backend in ("auto", "tesseract", "rapidocr"):
        if backend == "auto":
            from ocr_fallback import recognize_auto
            ocr_bin = os.getenv('REELPACK_OCR_BIN', str(CORE_DIR.parent / 'Tools/macos_ocr/macos_ocr'))
            def recognize(selected, references):
                return recognize_auto(selected, references, ocr_bin,
                                      ocr_model=args.vision_model, ollama_base=OLLAMA_BASE)
        elif backend == "rapidocr":
            from optional_backends import recognize_frames as recognize
        else:
            from ocr_tesseract import recognize_frames as recognize
        return recognize(frames, refs)
    if backend == "macos":
        ocr_bin = os.getenv(
            "REELPACK_OCR_BIN",
            str(Path(__file__).resolve().parent.parent / "Tools/macos_ocr/macos_ocr"),
        )
        if not Path(ocr_bin).exists():
            return {"status": "not_run", "backend": "macos", "model": "macOS Vision",
                    "params": {}, "items": [], "errors": [],
                    "reason": f"macOS OCR tool missing: {ocr_bin}", "attempted": False}
        results, error = optional_macos_ocr(frames, ocr_bin)
        items = [{"timestamp": timestamp, "start": timestamp, "end": timestamp, "text": text,
                  "source": "frame_ocr", "evidence_ids": refs.get(timestamp, []),
                  "uncertain": False}
                 for timestamp, text in results if text.strip()]
        return {"status": "failed" if error else "complete", "backend": "macos",
                "model": "macOS Vision", "params": {}, "items": items,
                "errors": [{"backend": "macos", "reason": error}] if error else [],
                "reason": error, "attempted": True}
    # Remaining backend: local Ollama vision model.
    items, errors = [], []
    for timestamp, image in frames:
        description = describe_frame(image, args.vision_model)
        if description and not description.startswith("(帧 OCR 失败"):
            if description != "（无可见文字）":
                items.append({"timestamp": timestamp, "start": timestamp, "end": timestamp,
                              "text": description, "source": "ollama",
                              "evidence_ids": refs.get(timestamp, []), "uncertain": False})
        else:
            errors.append({"backend": "ollama", "timestamp": timestamp, "reason": description})
    status = "failed" if errors and len(errors) == len(frames) else ("partial" if errors else "complete")
    return {"status": status, "backend": "ollama", "model": args.vision_model, "params": {},
            "items": items, "errors": errors,
            "reason": "; ".join(error["reason"] for error in errors) or None, "attempted": True}


def _merge_ocr_results(results) -> dict:
    """Combine coarse and densified OCR passes without losing provenance."""
    results = [result for result in results if isinstance(result, dict)]
    items, errors, attempts, probes = [], [], [], []
    for result in results:
        items.extend(result.get("items") or [])
        errors.extend(result.get("errors") or [])
        attempts.extend(result.get("attempts") or [])
        probes.extend(result.get("probes") or [])
    statuses = [result.get("status") for result in results]
    if statuses and all(status == "complete" for status in statuses):
        status = "complete"
    elif any(status == "complete" for status in statuses):
        status = "partial"
    elif statuses and all(status == "not_run" for status in statuses):
        status = "not_run"
    else:
        status = "failed"
    backends = [result.get("backend") for result in results if result.get("backend")]
    unique = list(dict.fromkeys(backends))
    backend = "mixed" if len(unique) > 1 else (unique[0] if unique else None)
    models = {result.get("backend"): result.get("model") for result in results if result.get("backend")}
    merged = {"status": status, "backend": backend,
              "model": models if len(unique) > 1 else models.get(backend),
              "params": {"strategy": "subtitle_trigger", "phases": len(results)},
              "items": items, "errors": errors, "attempted": True,
              "reason": "; ".join(dict.fromkeys(result["reason"] for result in results
                                                if result.get("reason"))) or None}
    if attempts:
        merged["attempts"] = attempts
    if probes:
        merged["probes"] = probes
    fallback = next((result.get("fallback_from") for result in results
                     if result.get("fallback_from")), None)
    if fallback:
        merged["fallback_from"] = fallback
    return merged


def _record_ocr_result(args, result) -> None:
    args.ocr_result = result
    args.ocr_attempted = bool(result.get("attempted", True))
    args.ocr_observations = result.get("items") or []
    args.ocr_error = result.get("reason") if result.get("status") != "complete" else None


def _texts_by_timestamp(items) -> dict:
    texts = {}
    for item in items or []:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        timestamp = item.get("timestamp", item.get("start"))
        try:
            timestamp = round(float(timestamp), 6)
        except (TypeError, ValueError):
            continue
        texts.setdefault(timestamp, []).append(item["text"])
    return {timestamp: "\n".join(parts) for timestamp, parts in texts.items()}


def _triggered_ocr(args, video_path, package_dir, evidence, duration):
    """Two-pass profile OCR (ad/talk/podcast): coarse band scan then densified windows.

    The coarse pass crops the bottom subtitle band and downscales each crop to
    a small grayscale signature.  Consecutive signatures that differ schedule a
    densified window; OCR text is compared only to de-duplicate identical
    sentences (``duplicate_text_timestamps``), never as the trigger.

    Every sampling point whose frame could not be decoded is listed in
    ``dropped[]``; ``total_ocr_frames == coarse_scans + densified_frames -
    dropped_count`` is the self-consistent coverage invariant (the two counts are
    planned sampling points, not decoded frames).
    """
    interval = PROFILE_COARSE_INTERVAL[args.profile]
    coarse_times = subtitle_band_times(video_path, interval=interval, band=SUBTITLE_BAND,
                                       duration=duration)
    dropped = []
    coarse_frames = extract_band_frames(video_path, coarse_times, band=SUBTITLE_BAND,
                                        max_width=args.ocr_max_width, dropped=dropped)
    signatures = band_signatures(coarse_frames, size=SUBTITLE_SIGNATURE_SIZE)
    coarse_refs = _write_ocr_frames(coarse_frames, package_dir, evidence)
    coarse_result = _recognize_ocr(args, coarse_frames, coarse_refs)
    texts = _texts_by_timestamp(coarse_result.get("items"))
    texts_by_timestamp = {round(float(timestamp), 6): texts.get(round(float(timestamp), 6), "")
                          for timestamp, _ in coarse_frames}
    triggered_changes = count_signature_changes(signatures)
    duplicates = duplicate_text_timestamps(signatures, texts_by_timestamp)
    densified = triggered_times_from_signatures(
        signatures, window=SUBTITLE_DENSIFY_WINDOW, step=SUBTITLE_DENSIFY_STEP,
        duration=duration, existing=[timestamp for timestamp, _ in coarse_frames],
        skip_timestamps=duplicates)
    dense_frames = []
    if densified:
        # A wholly failed densified pass must not discard the coarse OCR result;
        # its sampling points are reported in dropped[] instead of silently lost.
        try:
            dense_frames = extract_band_frames(video_path, densified, band=SUBTITLE_BAND,
                                               max_width=args.ocr_max_width, dropped=dropped)
        except RuntimeError:
            dense_frames = []
    dense_refs = _write_ocr_frames(dense_frames, package_dir, evidence,
                                   start_index=len(coarse_frames))
    dense_result = _recognize_ocr(args, dense_frames, dense_refs) if dense_frames else None
    result = _merge_ocr_results([coarse_result, dense_result] if dense_result else [coarse_result])
    frames = coarse_frames + dense_frames
    dropped.sort(key=lambda item: item["timestamp"])
    sampling = {
        "strategy": "subtitle_trigger",
        "profile": args.profile,
        "coarse_interval": int(interval) if float(interval).is_integer() else interval,
        "coarse_scans": len(coarse_times),
        "triggered_changes": triggered_changes,
        "text_deduplicated_changes": len(duplicates),
        "densified": len(densified),
        "densified_frames": len(densified),
        "total_ocr_frames": len(frames),
        "dropped": dropped,
        "dropped_count": len(dropped),
        "band": SUBTITLE_BAND,
        "params": _subtitle_trigger_params(),
        "coarse_timestamps": [timestamp for timestamp, _ in coarse_frames],
        "densified_timestamps": [timestamp for timestamp, _ in dense_frames],
        "frame_source": "original_media_subtitle_band",
        "max_width": args.ocr_max_width,
        "jpeg_quality": 95,
        "executed_frame_count": len(frames),
        "executed_timestamps": [timestamp for timestamp, _ in frames],
        "independent_sampling": False,
    }
    return result, sampling, frames


def _save_pipeline_analysis(store: AssetStore, archive: dict, input_ref: str, video_path: str,
                            segments: list[dict], raw_transcript: str, corrected: str,
                            ocr_terms: str, summary: str, args, *, shots: dict | None = None,
                            evidence: list[dict] | None = None, ocr_sampling_info: dict | None = None,
                            transcription_info: dict | None = None, music: dict | None = None) -> dict:
    video_metadata = _probe_video_metadata(video_path)
    analysis = empty_analysis(
        archive["asset_id"],
        {"kind": "local_file", "reference": None},
        video_metadata,
    )
    # Package-level completeness and reuse are independent: optional not_run/failed
    # statuses remain truthful, while a matching run can reuse completed core work.
    analysis["status"] = "partial"
    if getattr(args, "run_signature", None):
        analysis["run_signature"] = args.run_signature
    analysis["source"]["collected_at"] = analysis["source"].get("collected_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    analysis["capabilities"]["transcription"] = {
        "status": "complete" if segments else "unknown",
        "backend": "faster-whisper",
        "params": (transcription_info or {}).get("params", {}),
        "timestamp_bounds": (transcription_info or {}).get("timestamp_bounds", {}),
        "warnings": (transcription_info or {}).get("warnings", []),
        "model": args.model,
        "language": (transcription_info or {}).get("language"),
        "language_probability": (transcription_info or {}).get("language_probability"),
    }
    analysis["capabilities"]["ocr"] = {
        "status": "complete" if ocr_terms else "unknown",
        "backend": args.ocr_backend,
        "model": args.vision_model if args.ocr_backend == "ollama" else (os.getenv("REELPACK_TESSERACT_LANG", "chi_sim+eng") if args.ocr_backend == "tesseract" else "macOS Vision"),
        "sampling": ocr_sampling_info or {"strategy": "independent_interval", "interval_seconds": FRAME_INTERVAL, "independent_sampling": True},
    }
    analysis["capabilities"]["shots"] = {
        "status": (shots or {}).get("status", "not_run"),
        "backend": (shots or {}).get("backend"),
        "model": None,
    }
    analysis["capabilities"]["vision"] = {"status": "not_run", "backend": None, "model": None}
    analysis["capabilities"]["music"] = {"status": (music or {}).get("status", "not_run"), "backend": (music or {}).get("backend"), "model": (music or {}).get("model"), "experimental": True}
    analysis["transcript"] = {
        "status": "complete" if segments else "unknown",
        "raw_segments": [
            {"segment_id": f"seg_{index + 1}", "start": item["start"], "end": item["end"], "text": item["text"], "speaker": item.get("speaker")}
            for index, item in enumerate(segments)
        ],
        "language": (transcription_info or {}).get("language"),
        "language_probability": (transcription_info or {}).get("language_probability"),
        "raw_text": raw_transcript,
        "corrected_text": corrected,
        "corrected_language": (transcription_info or {}).get("language"),
        "correction": {
            "status": "complete" if corrected != raw_transcript else "not_run",
            "edits": [],
            "basis": "OCR terms supplied to correction prompt" if ocr_terms else None,
            "model": args.vision_model if getattr(args, "correction_backend", "none") != "none" else None,
            "backend": getattr(args, "correction_backend", "none"),
            "reason": "跳过，交由下游 Agent 生成/回写" if getattr(args, "correction_backend", "none") == "none" else "校正请求已运行；未变更不表示失败",
        },
    }
    if getattr(args, 'rewrite_result', None) is not None:
        analysis['transcript']['rewrite'] = copy.deepcopy(args.rewrite_result)
    # Absent unless --speakers auto ran, which keeps --speakers off byte-stable.
    if getattr(args, 'speakers_result', None) is not None:
        analysis['speakers'] = copy.deepcopy(args.speakers_result)
    raw_ocr_items = getattr(args, "ocr_observations", None)
    if raw_ocr_items is None:
        raw_ocr_items = _ocr_items_from_lines(ocr_terms)
    raw_ocr_items = ensure_ocr_regions(raw_ocr_items,
                                       getattr(args, "ocr_frame_sizes", None))
    merged_ocr = merge_observations(raw_ocr_items)["items"] if raw_ocr_items else []
    analysis["ocr"] = {
        "status": "complete" if ocr_terms else "unknown",
        # A null region means "this backend does not report coordinates", not a
        # failed implementation; see ocr_regions.REGION_SUPPORT.
        "region_support": dict(REGION_SUPPORT),
        "sampling": ocr_sampling_info or {"strategy": "independent_interval", "interval_seconds": FRAME_INTERVAL, "independent_sampling": True,
                                          "dropped": [], "dropped_count": 0},
        "items": merged_ocr,
        "reading_text": "\n".join(item["text"] for item in merged_ocr),
        "reading_reconstruction": merge_observations(raw_ocr_items)["reading_reconstruction"] if raw_ocr_items else {"status": "not_run"},
    }
    analysis["ocr_conflicts"] = find_conflicts(merged_ocr, segments)
    if getattr(args, "ocr_attempted", False) and not getattr(args, "ocr_error", None):
        analysis["ocr"]["status"] = "complete"
        analysis["capabilities"]["ocr"]["status"] = "complete"
    analysis["shots"] = shots or {"status": "not_run", "items": []}
    analysis["content_hints"] = compute_content_hints(
        shots=analysis["shots"],
        segments=segments,
        ocr_items=merged_ocr,
        duration_seconds=video_metadata.get("duration_seconds"),
        frame_board=_existing_frame_board(archive),
    )
    analysis["music"] = music or {"status":"not_run", "experimental":True}
    transcription_state = getattr(args, 'transcription_state', None)
    if transcription_state is not None:
        analysis['transcript'].update(transcription_state)
        analysis['capabilities']['transcription'].update(transcription_state)
    audio_track = getattr(args, 'audio_track', None)
    if audio_track is not None:
        analysis['video']['audio_track'] = audio_track
    execution = getattr(args, 'execution', None)
    if execution is not None:
        analysis['execution'] = dict(execution)
    if analysis['music'].get('outcome'):
        analysis['capabilities']['music']['outcome'] = analysis['music']['outcome']
        for key in ('reason', 'impact', 'remedy'):
            if analysis['music'].get(key): analysis['capabilities']['music'][key] = analysis['music'][key]
    analysis["evidence"] = evidence or []
    summary_skipped = getattr(args, "skip_summary", False) or getattr(args, "summary_backend", "none") == "none"
    analysis["summary"] = {"status": "complete" if summary else ("not_run" if summary_skipped else "failed"), "text": summary, "backend": getattr(args, "summary_backend", "none")}
    if not summary and summary_skipped:
        analysis["summary"]["reason"] = "跳过，交由下游 Agent 生成/回写（asset_cli set-summary）"
    if not summary and not summary_skipped:
        analysis["summary"]["reason"] = "Summary request returned no usable text; connectivity/model/generation not distinguished. Localhost failure inside sandbox does not establish host service unavailability."
    if getattr(args, "ocr_error", None):
        analysis["ocr"].update(status="failed", reason=args.ocr_error)
        analysis["capabilities"]["ocr"].update(status="failed", reason=args.ocr_error)
    if getattr(args, "ocr_result", None):
        for key in ("status", "backend", "model", "params", "errors"):
            analysis["ocr"][key] = args.ocr_result[key]
        analysis["capabilities"]["ocr"].update({key: args.ocr_result[key] for key in ("status", "backend", "model")})
        for key in ('fallback_from', 'reason', 'attempts', 'probes'):
            if key in args.ocr_result:
                analysis['ocr'][key] = args.ocr_result[key]
                if key not in ('attempts', 'probes'): analysis['capabilities']['ocr'][key] = args.ocr_result[key]
        analysis['ocr']['frame_provenance'] = args.ocr_result['items']
    analysis["limitations"] = [
        "镜头默认使用 TransNetV2，不可用时回退 OpenCV 高召回启发式；不等同于语义场景识别，边界置信度未校准。",
        "OCR 仅在实际后端运行并返回结果时标记 complete；未运行/缺工具时保持 not_run/unknown。",
        "音频与视觉复核需要对应工具，当前包不声称已回听或完整看图。",
    ]
    annotate_analysis(analysis)
    store.save_analysis(archive["asset_id"], analysis)
    return analysis


def analyze_audio_isolated(media_path: str, shot_items: list[dict], package_dir=None,
                           backend="librosa", genre_threshold=.05, genre_top_k=3) -> dict:
    """Shared audio-only service; optional errors never abort the main pipeline."""
    from audio_service import analyze_audio
    return analyze_audio(media_path, shot_items, package_dir, backend=backend,
                         genre_threshold=genre_threshold, genre_top_k=genre_top_k,
                         progress=emit_progress)

# ── Main Pipeline ──

def main(argv=None):
    parser = argparse.ArgumentParser(description="Reelpack Core Pipeline")
    parser.add_argument("--input", required=True, help="本地视频文件路径（.mp4/.mov/.mkv）")
    parser.add_argument("--output-dir", default="~/Downloads/Reelpack", help="输出目录")
    parser.add_argument("--model", default="large", choices=["small", "large"], help="Whisper 模型：small 或 large，默认 large（large-v3）")
    parser.add_argument("--language", default="auto", help="转录语言代码（默认 auto 自动识别，例如 ja/en/ko/zh）")
    parser.add_argument("--vision-model", default="internvl3.5:4b", help="视觉模型 (Ollama tag, 默认 internvl3.5:4b)")
    parser.add_argument("--music-backend", choices=["librosa", "panns"], default="librosa", help="可选音乐语义旁路；默认保留 librosa")
    parser.add_argument("--ocr-backend", choices=["auto", "macos", "ollama", "tesseract", "rapidocr"], default="auto", help="OCR 后端（默认 auto）")
    parser.add_argument("--ocr-max-width", type=int, default=FRAME_WIDTH, help="原片 OCR 帧宽度上限（默认 1920；0 保留原分辨率，不放大低分辨率源）")
    parser.add_argument("--profile", choices=["auto", "ad", "talk", "podcast"], default="auto",
                        help="OCR 采样配置：auto 保持旧行为；ad 10s 粗扫+字幕触发；talk 30s 粗扫+触发；podcast 60s 粗扫+触发")
    parser.add_argument("--speakers", choices=["off", "auto"], default="off",
                        help="实验性说话人分离：off（默认，保持旧结果）；auto 需本地 CAM++ 模型（REELPACK_SPEAKER_MODEL），不自动下载")
    parser.add_argument('--rewrite', choices=['short', 'film', 'off'], default='off', help='默认off；显式short或film开启语义整理，原文始终保留')
    parser.add_argument('--rewrite-model', default='qwen2.5:7b', help='本地Ollama整理模型，须已准备；不自动下载')
    parser.add_argument("--summary-backend", choices=["none", "ollama", "cloud"], default="none", help="默认交由下游 Agent；本地摘要需显式启用")
    parser.add_argument("--correction-backend", choices=["none", "ollama", "cloud"], default="none", help="可选 OCR 辅助转录校正")
    parser.add_argument("--genre-threshold", type=float, default=0.05, help="PANNs genre 候选分数下限 (0..1，未校准)")
    parser.add_argument("--genre-top-k", type=int, default=3, help="最多 genre 候选数 (1..10)")
    parser.add_argument("--skip-summary", action="store_true", help="跳过模型摘要，允许 Agent 后续回写")
    parser.add_argument("--force", action="store_true", help="即使运行签名相同也重新分析；覆盖同一资产的分析结果，保留原媒体")
    parser.add_argument("--cloud", action="store_true", help="使用云端 API (需要 OPENAI_API_KEY)")
    parser.add_argument("--pdf", action="store_true", help="同时输出 PDF")
    parser.add_argument("--library-root", default=None, help="资产库根目录（默认 output-dir/library）")
    args = parser.parse_args(argv)
    args.library_root = args.library_root or str(Path(args.output_dir).expanduser() / "library")
    if args.cloud:
        if args.summary_backend == "none": args.summary_backend = "cloud"
        if args.correction_backend == "none": args.correction_backend = "cloud"
    if args.skip_summary:
        args.summary_backend = "none"
    # Keep auto as the requested strategy in the signature. The saved OCR result
    # records actual per-frame backends, including any failure-only fallback.

    if not 0 <= args.genre_threshold <= 1 or not 1 <= args.genre_top_k <= 10:
        parser.error("genre-threshold 必须在 0..1；genre-top-k 必须在 1..10")
    if not 0 <= args.ocr_max_width <= 8192:
        parser.error("ocr-max-width 必须在 0..8192（0 保留原分辨率）")

    attempt = None
    checkpoint = None
    try:
        # 1. Check deps
        missing = check_dependencies()
        if missing:
            emit_error("startup", "MISSING_DEPS", f"缺少依赖: {', '.join(missing)}")
            sys.exit(1)

        if os.environ.get('REELPACK_CACHE_PREPARE_ERROR'):
            emit_progress('runtime_cache', 0,
                          '加速缓存不可写；保留主流程与已存包复用，音频会独立报告失败。原因: ' + os.environ['REELPACK_CACHE_PREPARE_ERROR'])
        # 2. Validate local input
        emit_progress("input", 0, "正在准备本地文件...")
        video_path = resolve_input(args.input)

        # Archive identity and request signature are checked before any costly stage.
        store = AssetStore(args.library_root)
        args.run_signature = run_signature(sha256_file(video_path), args)
        archive = store.import_file(video_path, run_signature=args.run_signature, force=args.force)
        cached = store.get_asset(archive["asset_id"])
        previous = cached.get("analysis")
        package_dir = Path(archive["package_dir"])
        if archive["analysis_complete"]:
            emit_progress("cache", 100, "运行签名一致，复用转录/镜头/OCR；可选缺项可只补音频")
            analysis = annotate_analysis(previous)
            emit_complete({
                "asset_id": archive["asset_id"],
                "markdown": str(package_dir / "analysis.md"),
                "analysis": str(package_dir / "analysis.json"),
                "package": str(package_dir),
                "video": cached["media"].get("local_path") or video_path,
                "cached": True, "reuse": "cached: true", "capability_summary": summary_json(analysis),
            })
            return
        if previous:
            reason = "要求 --force" if args.force else "签名不一致或必需能力缺失"
            emit_progress("cache", 5, f"{reason}：重新分析，覆盖同一资产的分析/文稿，归档原片保持不变")

        from asset_attempt import AssetAttempt
        attempt = AssetAttempt(store, archive)
        attempt.__enter__()
        package_dir = attempt.package_dir
        archive = {**archive, 'package_dir': str(package_dir)}
        segments, transcription_info, raw_transcript, corrected = [], {}, '', ''
        ocr_terms, summary = '', ''
        shots, evidence = {'status': 'not_run', 'items': []}, []
        music = {'status': 'not_run', 'reason': 'audio stage not yet run'}
        ocr_sampling_info = {}
        args.transcription_state = {'status': 'not_run', 'reason': 'transcription stage not yet run'}
        args.execution = {'status': 'in_progress', 'stage': 'input'}
        def checkpoint(stage, status='in_progress', reason=None):
            args.execution = {'status': status, 'stage': stage}
            if reason: args.execution['reason'] = reason
            return _save_pipeline_analysis(
                attempt, archive, args.input, video_path, segments, raw_transcript,
                corrected, ocr_terms, summary, args, shots=shots, evidence=evidence,
                transcription_info=transcription_info, ocr_sampling_info=ocr_sampling_info, music=music)
        checkpoint('input')
        try:
            args.audio_track = probe_audio_track(video_path)
        except Exception as exc:
            args.audio_track = {'status': 'failed', 'present': None, 'reason': str(exc)}
            raise RuntimeError('audio stream probe failed: ' + str(exc)) from exc
        if args.audio_track['present']:
            emit_progress('model_preflight', 2, '检查转录模型本地就绪状态，不下载模型')
            try:
                check_transcription_model(args.model)
            except Exception as exc:
                args.transcription_state = {'status': 'not_run', 'outcome': 'model_unavailable', 'reason': str(exc)}
                raise
        else:
            args.transcription_state = {'status': 'not_run', 'outcome': 'no_audio_track', 'reason': 'no_audio_track',
                'impact': '媒体无音轨，未运行语音识别；镜头和画面文字仍可分析。',
                'remedy': '需要转录时提供有音轨的本地文件，无需重试此音轨缺失。'}
        # 2c. Analyze video structure before OCR. This is independent of Whisper.
        is_video = has_video_track(video_path)
        if is_video:
            emit_progress("shots", 20, "正在检测镜头切点并生成代表帧...")
            shots, evidence = detect_shots(video_path, Path(archive["package_dir"]))
            shots, evidence = copy.deepcopy(shots), copy.deepcopy(evidence)
        checkpoint('shots')
        duration = (_probe_video_metadata(video_path) or {}).get("duration_seconds") or 0.0
        ocr_sampling_info = sampling_candidates(
            duration, segments=(), interval=float(FRAME_INTERVAL),
            change_timestamps=[item["start"] for item in shots.get("items", []) if item.get("start") is not None],
        ) if duration > 0 else {"strategy": "independent_interval", "interval_seconds": FRAME_INTERVAL, "independent_sampling": True, "candidates": []}

        # 2d. Audio descriptors run in isolated .venv-audio.
        if args.audio_track['present']:
            music = analyze_audio_isolated(
                video_path, shots.get("items", []), package_dir,
                backend=args.music_backend, genre_threshold=args.genre_threshold, genre_top_k=args.genre_top_k)
        else:
            music = audio_track_absent(shots.get('items', []))
            emit_progress('audio_complete', 50, 'no_audio_track：媒体无音轨，跳过音频与转录')
        checkpoint('audio')

        # 3. Empty successful ASR is a valid content outcome, not an error event.
        if args.audio_track['present']:
            try:
                segments, transcription_info = transcribe_audio(video_path, args.model, args.language, return_info=True)
            except Exception as exc:
                args.transcription_state = {'status': 'failed', 'outcome': 'decode_or_inference_failed', 'reason': str(exc)}
                raise
            args.transcription_state = {'status': 'complete',
                'outcome': 'speech_detected' if segments else 'speech_not_detected'}
            if not segments:
                emit_progress('transcribe', 78, '识别已完成，未检出语音；不代表没有声音或已识别音乐')
        raw_transcript = "\n".join(
            f"[{_fmt_ts(s['start'])}-{_fmt_ts(s['end'])}] {s['text']}" for s in segments)
        corrected = raw_transcript
        # Experimental and default-off: only 'auto' writes analysis.speakers.
        if args.speakers == 'auto':
            if args.audio_track['present']:
                emit_progress('speakers', 79, '实验性说话人分离（CAM++）；仅使用本地模型，不下载')
                args.speakers_result = _analyze_speakers(video_path, segments, duration)
                if args.speakers_result.get('status') == 'complete':
                    emit_progress('speakers', 79,
                                  f"说话人分离完成 ({args.speakers_result['confidence'].get('selected_k', '?')} 组)")
                else:
                    emit_progress('speakers', 79, '说话人分离未完成：' +
                                  str(args.speakers_result.get('reason', 'unknown')))
            else:
                args.speakers_result = _speaker_state('not_run', reason='no_audio_track')
        checkpoint('transcription')

        from frame_sampling import mark_speech_frames
        mark_speech_frames(evidence, segments)

        # 4. Extract frames + OCR
        ocr_terms_parts = []

        if is_video:
            profile = getattr(args, "profile", "auto") or "auto"
            frames = []
            if profile in PROFILE_COARSE_INTERVAL:
                emit_progress("transcribe", 80, "正在字幕带粗扫并按变化加密...")
                try:
                    ocr_result, ocr_sampling_info, frames = _triggered_ocr(
                        args, video_path, archive["package_dir"], evidence, duration)
                except Exception as exc:
                    emit_progress("transcribe", 80, "字幕带采样失败，未运行 OCR：" + str(exc))
                    ocr_result = {"status": "failed", "backend": args.ocr_backend,
                                  "model": None, "params": {}, "items": [], "errors": [],
                                  "reason": str(exc), "attempted": True}
                    interval = PROFILE_COARSE_INTERVAL[profile]
                    failed_dropped = [item for item in getattr(exc, "dropped", []) or []
                                      if isinstance(item, dict)]
                    # A failed pass still reports every sampling point that was
                    # attempted and dropped, so coverage stays auditable.
                    attempted = getattr(exc, "attempted", len(failed_dropped)) or 0
                    ocr_sampling_info = {
                        "strategy": "subtitle_trigger", "profile": profile,
                        "coarse_interval": int(interval) if float(interval).is_integer() else interval,
                        "coarse_scans": int(attempted),
                        "triggered_changes": 0, "text_deduplicated_changes": 0,
                        "densified": 0, "densified_frames": 0,
                        "total_ocr_frames": 0,
                        "dropped": failed_dropped, "dropped_count": len(failed_dropped),
                        "band": SUBTITLE_BAND, "params": _subtitle_trigger_params(),
                        "status": "failed",
                        "reason": str(exc), "executed_frame_count": 0,
                        "executed_timestamps": [], "independent_sampling": False,
                    }
                args.ocr_frame_sizes = _frame_size_map(frames)
                _record_ocr_result(args, ocr_result)
            else:
                emit_progress("transcribe", 80, "正在提取关键帧...")
                frames = extract_frames(video_path, max_width=args.ocr_max_width)
                ocr_sampling_info.update(
                    frame_source="original_media", max_width=args.ocr_max_width, jpeg_quality=95,
                    dropped=[], dropped_count=0,
                    executed_frame_count=len(frames), executed_timestamps=[ts for ts, _ in frames],
                )
                args.ocr_frame_sizes = _frame_size_map(frames)
                if frames:
                    if args.ocr_backend == "auto":
                        emit_progress('transcribe', 82, '正在 OCR 识别（auto：macOS Vision，异常时使用本地 RapidOCR）')
                    elif args.ocr_backend == "macos":
                        emit_progress("transcribe", 82, "正在 OCR 识别 (macOS Vision)...")
                    elif args.ocr_backend not in ("tesseract", "rapidocr"):
                        emit_progress("transcribe", 82, f"正在 OCR 识别 ({args.vision_model})...")
                    refs = _write_ocr_frames(frames, archive["package_dir"], evidence)
                    ocr_result = _recognize_ocr(args, frames, refs)
                    _record_ocr_result(args, ocr_result)
                    if args.ocr_backend == "macos" and not ocr_result.get("attempted"):
                        emit_progress("transcribe", 82, "macOS OCR 工具缺失，跳过 OCR")
                    elif ocr_result.get("fallback_from"):
                        emit_progress('transcribe', 84, 'OCR 已回退至 ' + str(ocr_result.get('backend')) + '：' + str(ocr_result.get('reason', '原生后端异常')))
                    elif ocr_result.get("status") == "failed" and ocr_result.get("reason"):
                        emit_progress("transcribe", 82, str(ocr_result["reason"]))

            for item in getattr(args, "ocr_observations", None) or []:
                if item.get("text") and item.get("timestamp") is not None:
                    ocr_terms_parts.append(f"[{_fmt_ts(item['timestamp'])}] {item['text']}")

            # Free frame data from memory
            del frames
            gc.collect()

        ocr_terms = "\n".join(ocr_terms_parts) if ocr_terms_parts else ""
        checkpoint('ocr')

        # 5. Correct transcript using OCR terms
        corrected = raw_transcript if args.correction_backend == "none" else correct_transcript(
            raw_transcript, ocr_terms,
            model_name=args.vision_model,
            use_cloud=args.correction_backend == "cloud", language=transcription_info.get("language") or args.language
        )

        # Semantic rewriting is optional; preserve original segments and never let
        # a missing local model/network failure prevent the evidence package.
        from transcript_rewrite import rewrite_transcript
        if segments and args.rewrite != 'off':
            emit_progress('rewrite', 86, '整理原始转录（' + args.rewrite + '）；原文保留，不下载模型；每批本地最多30秒/云端60秒')
        try:
            args.rewrite_result = rewrite_transcript(
                [{'segment_id': f'seg_{i + 1}', **item} for i, item in enumerate(segments)],
                ocr_text=ocr_terms, mode=args.rewrite, use_cloud=args.cloud,
                model=None if args.cloud else args.rewrite_model,
                no_speech_reason=getattr(args, 'transcription_state', {}).get('outcome'))
        except Exception as exc:
            args.rewrite_result = {'status': 'failed', 'text': raw_transcript, 'segments': copy.deepcopy(segments),
                                   'edits': [], 'params': {'mode': args.rewrite}, 'reason': str(exc)}
        if args.rewrite_result.get('status') == 'complete':
            corrected = args.rewrite_result['text']
        else:
            if args.rewrite != 'off': corrected = raw_transcript
            emit_progress('rewrite', 89, '未整理/整理不可用：' + str(args.rewrite_result.get('reason', 'unknown')))
        prepared_analysis = checkpoint('rewrite')
        # Free vision model memory
        gc.collect()

        # 6. Generate summary
        summary = "" if args.skip_summary or args.summary_backend == "none" else generate_summary(corrected, model_name=args.vision_model, use_cloud=args.summary_backend == "cloud", language=transcription_info.get("language") or args.language)

        # 7. Output MD (and optionally PDF)
        has_audio = bool(segments)
        ocr_display = prepared_analysis['ocr']
        md_path = output_markdown(summary, corrected, video_path, str(package_dir / 'readable'), has_audio,
                                  original=raw_transcript, rewrite=args.rewrite_result, ocr=ocr_display)
        readable_name = Path(md_path).name
        analysis = checkpoint('complete', 'complete')
        promoted = attempt.promote()
        package_dir = Path(promoted['package_dir'])
        archive['package_dir'] = str(package_dir)
        # Export the readable result only after the package was committed successfully.
        md_path = str(package_dir / 'readable' / readable_name)
        try:
            export_dir = Path(args.output_dir).expanduser()
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / readable_name
            temporary = export_dir / ('.' + readable_name + '.' + str(os.getpid()) + '.tmp')
            with temporary.open('xb') as handle:
                handle.write(Path(md_path).read_bytes())
            os.replace(temporary, export_path)
            md_path = str(export_path)
        except OSError as export_error:
            emit_progress('output', 96, '资产包已保存；外部文稿导出失败，使用包内文稿：' + str(export_error))
        output = {
            "markdown": md_path,
            "asset_id": archive["asset_id"],
            "analysis": str(Path(archive["package_dir"]) / "analysis.json"),
            "package": archive["package_dir"],
            "cached": False, "reuse": "cached: false", "capability_summary": summary_json(analysis),
            "execution_status": "complete", "transcript_outcome": analysis['transcript'].get('outcome', ''),
        }

        if args.pdf:
            emit_progress("output", 98, "正在生成 PDF...")
            pdf_path = output_pdf(md_path)
            if pdf_path:
                output["pdf"] = pdf_path

        # Always expose the original local video as a result artifact.
        output["video"] = video_path

        emit_complete(output)

    except Exception as e:
        detail = str(e)
        if checkpoint is not None and attempt is not None:
            try:
                checkpoint((getattr(args, 'execution', {}) or {}).get('stage', 'unknown'), 'failed', detail)
                detail += '\n部分结果检查点: ' + str(attempt.package_dir / 'analysis.json')
            except Exception as save_error:
                detail += '\n检查点保存失败: ' + str(save_error)
        emit_error("core", "PIPELINE_ERROR", detail)
        sys.exit(1)
    finally:
        if attempt is not None:
            attempt.__exit__(*sys.exc_info())


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    main()
