"""Tests for Core/motion_analyze.py.

Fixtures are real H.264 files encoded from analytically generated textured frames;
no mocks are used for measurement. The generator lives in this file so a reviewer can
reproduce every parameter. Fixtures and scratch files live in a temporary directory
that is removed at exit; nothing is written to the repository or source inputs.
"""
import atexit
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "Core"
sys.path.insert(0, str(CORE))

import numpy as np  # noqa: E402

import motion_analyze  # noqa: E402

try:
    import cv2  # noqa: E402
except ImportError:  # pragma: no cover - dependency gate
    cv2 = None

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
HAVE_MEDIA = bool(cv2 is not None and FFMPEG and FFPROBE)
PYTHON = sys.executable


# --------------------------------------------------------------------------------------
# Fixture generation
# --------------------------------------------------------------------------------------

def _texture(width, height, seed=0):
    """Dense multi-scale texture: survives H.264, yields hundreds of LK corners."""
    rng = np.random.default_rng(seed)
    coarse = rng.normal(0, 1, (height, width)).astype(np.float32)
    coarse = cv2.GaussianBlur(coarse, (0, 0), max(2.0, width / 180.0))
    fine = rng.normal(0, 1, (height, width)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), 1.0)
    coarse /= max(float(coarse.std()), 1e-6)
    fine /= max(float(fine.std()), 1e-6)
    yy, xx = np.mgrid[0:height, 0:width]
    checker = (((xx // 24) + (yy // 24)) % 2).astype(np.float32) * 46.0
    image = 118.0 + 55.0 * coarse + 22.0 * fine + checker
    for _ in range(120):
        x0 = int(rng.integers(0, width))
        y0 = int(rng.integers(0, height))
        size = int(rng.integers(4, max(5, min(width, height) // 18)))
        value = int(rng.integers(0, 256))
        if rng.random() < 0.5:
            cv2.rectangle(image, (x0, y0), (x0 + size, y0 + size), value, -1)
        else:
            cv2.circle(image, (x0, y0), size, value, -1)
    return np.clip(image, 0, 255).astype(np.uint8)


def _transform(texture, scale=1.0, angle_deg=0.0, tx=0.0, ty=0.0):
    """Forward similarity transform about the frame center: x' = c + sR(x-c) + t."""
    height, width = texture.shape
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radians = math.radians(angle_deg)
    a = scale * math.cos(radians)
    b = scale * math.sin(radians)
    matrix = np.array([
        [a, -b, cx - a * cx + b * cy + tx],
        [b, a, cy - b * cx - a * cy + ty],
    ], dtype=np.float32)
    return cv2.warpAffine(texture, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _encode(path, frames, fps):
    width, height = frames[0].shape[1], frames[0].shape[0]
    if width % 2 or height % 2:
        raise ValueError("fixture dimensions must be even for yuv420p")
    command = [
        FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "15",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"fixture encoding failed: {path}")


def _frame_count(seconds, fps):
    return max(1, int(round(seconds * fps)))


class Fixtures:
    _root = None

    @classmethod
    def root(cls):
        if cls._root is None:
            cls._root = Path(tempfile.mkdtemp(prefix="reelpack-motion-tests-"))
            atexit.register(shutil.rmtree, cls._root, ignore_errors=True)
        return cls._root

    @classmethod
    def path(cls, name):
        return cls.root() / name

    @classmethod
    def build(cls, name, builder):
        target = cls.path(name)
        if not target.exists():
            builder(target)
        return target


def _translation_fixture(width, height, fps, seconds=3.0, vx=0.15, vy=-0.06, seed=1):
    def build(target):
        texture = _texture(width, height, seed)
        frames = []
        for index in range(_frame_count(seconds, fps)):
            t = index / float(fps)
            frames.append(_transform(texture, tx=vx * width * t, ty=vy * height * t))
        _encode(target, frames, fps)
    return build


def _acceleration_fixture(width, height, fps, seconds=4.0, v0=(0.08, 0.05), accel=(0.10, 0.06)):
    def build(target):
        texture = _texture(width, height, seed=2)
        frames = []
        for index in range(_frame_count(seconds, fps)):
            t = index / float(fps)
            tx = width * (v0[0] * t + 0.5 * accel[0] * t * t)
            ty = height * (v0[1] * t + 0.5 * accel[1] * t * t)
            frames.append(_transform(texture, tx=tx, ty=ty))
        _encode(target, frames, fps)
    return build


def _zoom_fixture(width, height, fps, k=0.12, seconds=3.0):
    def build(target):
        texture = _texture(width, height, seed=3)
        frames = [_transform(texture, scale=math.exp(k * index / fps))
                  for index in range(_frame_count(seconds, fps))]
        _encode(target, frames, fps)
    return build


def _rotation_fixture(width, height, fps, omega=12.0, seconds=3.0):
    def build(target):
        texture = _texture(width, height, seed=4)
        frames = [_transform(texture, angle_deg=omega * index / fps)
                  for index in range(_frame_count(seconds, fps))]
        _encode(target, frames, fps)
    return build


def _two_shot_fixture(width, height, fps, seconds=3.0, speed=0.18, cut=1.5):
    def build(target):
        texture = _texture(width, height, seed=5)
        frames = []
        for index in range(_frame_count(seconds, fps)):
            t = index / fps
            if t < cut:
                tx = speed * width * t
            else:
                tx = speed * width * cut - speed * width * (t - cut)
            frames.append(_transform(texture, tx=tx))
        _encode(target, frames, fps)
    return build


def _black_fixture(width, height, fps, seconds=1.0):
    def build(target):
        frames = [np.zeros((height, width), np.uint8) for _ in range(_frame_count(seconds, fps))]
        _encode(target, frames, fps)
    return build


def _single_frame_fixture(width, height, fps):
    def build(target):
        _encode(target, [_texture(width, height, seed=6)], fps)
    return build


def _dropout_fixture(width, height, fps, seconds=3.0):
    def build(target):
        texture = _texture(width, height, seed=7)
        frames = []
        for index in range(_frame_count(seconds, fps)):
            t = index / fps
            if 1.0 <= t < 2.0:
                frames.append(np.zeros((height, width), np.uint8))
            else:
                frames.append(_transform(texture, tx=0.12 * width * t))
        _encode(target, frames, fps)
    return build


def _vfr_fixture(width, height, vx=0.15):
    """Variable frame rate: concat 1 s @24 fps and 1 s @12 fps with -c copy."""
    def build(target):
        texture = _texture(width, height, seed=9)
        parts = []
        for fps in (24, 12):
            part = Fixtures.path(f"vfr_source_{fps}.mp4")
            if not part.exists():
                frames = [_transform(texture, tx=vx * width * index / fps)
                          for index in range(_frame_count(1.0, fps))]
                _encode(part, frames, fps)
            parts.append(part)
        concat_list = Fixtures.path("vfr_concat.txt")
        concat_list.write_text("".join(f"file '{part.name}'\n" for part in parts), encoding="utf-8")
        subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                        "-i", str(concat_list), "-c", "copy", str(target)],
                       check=True, cwd=str(Fixtures.root()))
    return build


# --------------------------------------------------------------------------------------
# Assertion helpers
# --------------------------------------------------------------------------------------

def assert_finite_json(testcase, document):
    text = json.dumps(document, allow_nan=False)

    def walk(value, path="root"):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, float):
            testcase.assertTrue(math.isfinite(value), f"non-finite number at {path}: {value}")

    walk(document)
    return text


def _median(values):
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def _interior_pairs(result, start, end):
    return [pair for pair in result["pairs"]
            if pair["status"] == "valid" and pair["center_time"] is not None
            and start <= pair["center_time"] <= end]


def _run_cli(args, env_extra=None):
    environment = os.environ.copy()
    if env_extra:
        environment.update(env_extra)
    return subprocess.run([PYTHON, str(CORE / "motion_analyze.py")] + args,
                          capture_output=True, text=True, timeout=180, env=environment)


# --------------------------------------------------------------------------------------
# Translation, finite JSON, units/signs, resolution and frame-rate agreement
# --------------------------------------------------------------------------------------

@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class TranslationEvidenceTests(unittest.TestCase):
    CONFIGS = [
        ("t_640x360_24.mp4", 640, 360, 24),
        ("t_640x360_30.mp4", 640, 360, 30),
        ("t_1280x720_24.mp4", 1280, 720, 24),
        ("t_1280x720_30.mp4", 1280, 720, 30),
    ]
    VX = 0.15
    VY = -0.06

    @classmethod
    def setUpClass(cls):
        cls.paths = {}
        for name, width, height, fps in cls.CONFIGS:
            cls.paths[name] = Fixtures.build(
                name, _translation_fixture(width, height, fps, vx=cls.VX, vy=cls.VY))
        cls.results = {}
        for name, _, _, _ in cls.CONFIGS:
            cls.results[name] = motion_analyze.analyze_motion(
                cls.paths[name], start=0.0, end=3.0, sample_fps=8.0, analysis_width=480)

    def test_status_and_finite_json(self):
        for name, _, _, _ in self.CONFIGS:
            with self.subTest(name=name):
                result = self.results[name]
                assert_finite_json(self, result)
                self.assertEqual(result["status"], "complete", result["reason"])
                self.assertGreater(result["summary"]["valid_pair_count"], 0)
                self.assertEqual(result["schema"], motion_analyze.SCHEMA)
                self.assertEqual(result["backend"], motion_analyze.BACKEND)
                self.assertIn("vx_frame_width_s", result["units"])

    def test_signs_and_normalized_speed(self):
        for name, _, _, _ in self.CONFIGS:
            with self.subTest(name=name):
                summary = self.results[name]["summary"]["velocity"]
                vx = summary["median_vx_frame_width_s"]
                vy = summary["median_vy_frame_height_s"]
                self.assertGreater(vx, 0.0, "content moving right must have positive vx")
                self.assertLess(vy, 0.0, "content moving up must have negative vy")
                self.assertLessEqual(abs(vx - self.VX), max(0.15 * abs(self.VX), 0.01))
                self.assertLessEqual(abs(vy - self.VY), max(0.15 * abs(self.VY), 0.01))

    def test_resolution_and_framerate_agreement(self):
        speeds = [self.results[name]["summary"]["velocity"]["median_speed_normalized_frame_s"]
                  for name, _, _, _ in self.CONFIGS]
        center = _median(speeds)
        for speed in speeds:
            self.assertLessEqual(abs(speed - center), max(0.15 * center, 0.01))

    def test_timestamp_and_raw_smoothed_series(self):
        result = self.results["t_640x360_24.mp4"]
        times = result["timeline"]["frame_times_seconds"]
        self.assertEqual(len(times), len(result["pairs"]) + 1)
        for left, right in zip(times, times[1:]):
            self.assertAlmostEqual(right - left, 1.0 / 8.0, places=6)
        self.assertEqual(result["timeline"]["timestamp_source"],
                         "ffmpeg-fps-filter-cfr+showinfo-pts")
        for pair, entry in zip(result["pairs"], result["series"]):
            self.assertEqual(pair["status"], "valid")
            self.assertAlmostEqual(pair["delta_t"], 1.0 / 8.0, places=6)
            self.assertIsNotNone(entry["velocity_raw"])
            self.assertIsNotNone(entry["velocity_smoothed"])
            self.assertEqual(entry["smoothing"]["method"], "centered_moving_average")
            self.assertEqual(entry["smoothing"]["window_samples"], 5)
        interior = [entry for entry in result["series"]
                    if entry["acceleration_smoothed"] is not None]
        self.assertGreater(len(interior), 10)
        for entry in interior:
            self.assertLess(abs(entry["acceleration_smoothed"]["ax"]), 0.03)
            self.assertLess(abs(entry["acceleration_smoothed"]["ay"]), 0.03)

    def test_cli_stdout_json_and_read_only_source(self):
        path = self.paths["t_640x360_24.mp4"]
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        run = _run_cli([str(path), "--start", "0", "--end", "2",
                        "--sample-fps", "8", "--analysis-width", "480"])
        self.assertEqual(run.returncode, 0, run.stderr)
        document = json.loads(run.stdout)
        assert_finite_json(self, document)
        self.assertEqual(document["source"]["digest_sha256"], before)
        self.assertIn("status=", run.stderr)
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(before, after, "source video must remain unmodified")

    def test_max_seconds_truncation_reports_coverage(self):
        result = motion_analyze.analyze_motion(
            self.paths["t_640x360_24.mp4"], start=0.0, end=3.0,
            sample_fps=8.0, analysis_width=480, max_seconds=1.5)
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["coverage"]["fully_covered"])
        self.assertIn("max_seconds", result["coverage"]["truncation_reasons"])
        self.assertLessEqual(result["coverage"]["analyzed"]["duration"], 1.5 + 1e-6)


@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class VariableFrameRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Fixtures.build("vfr.mp4", _vfr_fixture(640, 360))
        cls.result = motion_analyze.analyze_motion(
            cls.path, start=0.0, end=1.9, sample_fps=8.0, analysis_width=480)

    def test_vfr_uses_resampled_pts_not_nominal_fps(self):
        result = self.result
        assert_finite_json(self, result)
        self.assertEqual(result["source"]["fps"]["variable"], True)
        self.assertNotEqual(result["source"]["fps"]["r_frame_rate"],
                            result["source"]["fps"]["avg_frame_rate"])
        validation = result["timeline"]["timestamp_validation"]
        self.assertIn("showinfo", result["timeline"]["timestamp_source"])
        self.assertEqual(validation["status"], "ok")
        deltas = {round(pair["delta_t"], 6) for pair in result["pairs"] if pair["delta_t"]}
        self.assertEqual(deltas, {0.125},
                         "sampling uses the documented 8 fps resampled timeline, never nominal 24 fps")
        times = result["timeline"]["frame_times_seconds"]
        for left, right in zip(times, times[1:]):
            self.assertAlmostEqual(right - left, 0.125, places=6)
        self.assertGreater(result["summary"]["valid_pair_count"], 0)
        self.assertGreater(result["summary"]["velocity"]["median_vx_frame_width_s"], 0.10)


# --------------------------------------------------------------------------------------
# Centered zoom and rotation, no spurious center translation
# --------------------------------------------------------------------------------------

@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class ZoomAndRotationTests(unittest.TestCase):
    ZOOM_K = 0.12
    OMEGA = 12.0

    @classmethod
    def setUpClass(cls):
        cls.zoom = Fixtures.build("zoom.mp4", _zoom_fixture(960, 540, 24, k=cls.ZOOM_K))
        cls.rotation = Fixtures.build("rotation.mp4", _rotation_fixture(960, 540, 24, omega=cls.OMEGA))
        cls.zoom_result = motion_analyze.analyze_motion(
            cls.zoom, start=0.0, end=3.0, sample_fps=8.0, analysis_width=480)
        cls.rotation_result = motion_analyze.analyze_motion(
            cls.rotation, start=0.0, end=3.0, sample_fps=8.0, analysis_width=480)

    def test_zoom_rate_and_no_center_translation(self):
        result = self.zoom_result
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "complete", result["reason"])
        middle = _interior_pairs(result, 0.75, 2.25)
        self.assertGreater(len(middle), 8)
        rates = [pair["velocity"]["zoom_log_scale_s"] for pair in middle]
        rate = _median(rates)
        self.assertGreater(rate, 0.5 * self.ZOOM_K)
        self.assertLessEqual(abs(rate - self.ZOOM_K), max(0.25 * self.ZOOM_K, 0.01))
        for pair in middle:
            self.assertLess(abs(pair["velocity"]["vx_frame_width_s"]), 0.02)
            self.assertLess(abs(pair["velocity"]["vy_frame_height_s"]), 0.02)

    def test_rotation_rate_and_no_center_translation(self):
        result = self.rotation_result
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "complete", result["reason"])
        middle = _interior_pairs(result, 0.75, 2.25)
        self.assertGreater(len(middle), 8)
        rates = [pair["velocity"]["rotation_deg_s"] for pair in middle]
        rate = _median(rates)
        self.assertGreater(rate, 0.5 * self.OMEGA)
        self.assertLessEqual(abs(rate - self.OMEGA), max(0.2 * self.OMEGA, 1.0))
        for pair in middle:
            self.assertLess(abs(pair["velocity"]["vx_frame_width_s"]), 0.02)
            self.assertLess(abs(pair["velocity"]["vy_frame_height_s"]), 0.02)
            self.assertLess(abs(pair["velocity"]["zoom_log_scale_s"]), 0.02)

    def test_rotation_metadata_respected(self):
        source = Fixtures.path("rotation_metadata_src.mp4")
        rotated = Fixtures.path("rotation_metadata_90.mp4")
        if not source.exists():
            _encode(source, [_texture(360, 640, 8) for _ in range(24)], 24)
        if not rotated.exists():
            subprocess.run([FFMPEG, "-y", "-v", "error", "-display_rotation", "90",
                            "-i", str(source), "-c", "copy", str(rotated)], check=True)
        result = motion_analyze.analyze_motion(rotated, start=0.0, end=0.5,
                                               sample_fps=8.0, analysis_width=480)
        assert_finite_json(self, result)
        self.assertEqual(result["source"]["rotation_degrees"], 90)
        self.assertEqual(result["source"]["display_width"], 640)
        self.assertEqual(result["source"]["display_height"], 360)
        self.assertEqual(result["parameters"]["analysis_height"], 270)
        self.assertIn(result["status"], {"complete", "insufficient_evidence"})


# --------------------------------------------------------------------------------------
# Acceleration from analytic motion
# --------------------------------------------------------------------------------------

@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class AccelerationTests(unittest.TestCase):
    V0 = (0.08, 0.05)
    ACCEL = (0.10, 0.06)

    @classmethod
    def setUpClass(cls):
        path = Fixtures.build("acceleration.mp4",
                              _acceleration_fixture(960, 540, 30, seconds=4.0,
                                                    v0=cls.V0, accel=cls.ACCEL))
        cls.result = motion_analyze.analyze_motion(
            path, start=0.0, end=4.0, sample_fps=16.0, analysis_width=480)

    def test_acceleration_direction_and_magnitude(self):
        result = self.result
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "complete", result["reason"])
        entries = [entry for entry in result["series"]
                   if entry["acceleration_smoothed"] is not None
                   and 1.0 <= entry["center_time"] <= 3.0
                   and entry["status"] == "valid"]
        self.assertGreater(len(entries), 20, "need sufficiently textured interior samples")
        ax = _median([entry["acceleration_smoothed"]["ax"] for entry in entries])
        ay = _median([entry["acceleration_smoothed"]["ay"] for entry in entries])
        magnitude = _median([entry["acceleration_smoothed"]["magnitude"] for entry in entries])
        expected_magnitude = math.hypot(*self.ACCEL)
        for measured, expected, label in ((ax, self.ACCEL[0], "ax"),
                                          (ay, self.ACCEL[1], "ay"),
                                          (magnitude, expected_magnitude, "magnitude")):
            self.assertGreater(measured, 0.0, f"{label} direction must be positive")
            self.assertLessEqual(abs(measured - expected), max(0.25 * expected, 0.03),
                                 f"{label}: measured {measured:.4f}, expected {expected:.4f}")

    def test_raw_and_smoothed_metadata(self):
        result = self.result
        self.assertEqual(result["parameters"]["smoothing"]["window_samples"], 5)
        self.assertEqual(result["parameters"]["smoothing"]["method"], "centered_moving_average")
        self.assertEqual(result["parameters"]["acceleration"]["method"], "central_difference")
        valid = [entry for entry in result["series"] if entry["status"] == "valid"]
        self.assertTrue(any(entry["acceleration_raw"] for entry in valid))
        self.assertTrue(any(entry["acceleration_smoothed"] for entry in valid))
        for entry in valid:
            if entry["acceleration_smoothed"] is not None:
                self.assertEqual(entry["differentiation"]["method"], "central_difference")
                self.assertIsNotNone(entry["smoothing"])

    def test_boundaries_do_not_get_acceleration(self):
        result = self.result
        for entry in result["series"]:
            if entry["status"] == "valid" and entry["acceleration_smoothed"] is None:
                self.assertEqual(entry["differentiation"]["reason"], "segment_boundary")


# --------------------------------------------------------------------------------------
# Negative cases; failures never become valid zero motion
# --------------------------------------------------------------------------------------

@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class NegativeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.black = Fixtures.build("black.mp4", _black_fixture(320, 180, 24, seconds=1.0))
        cls.single = Fixtures.build("single_frame.mp4", _single_frame_fixture(320, 180, 24))
        cls.dropout = Fixtures.build("dropout.mp4", _dropout_fixture(640, 360, 24, seconds=3.0))
        cls.corrupt = Fixtures.path("corrupt.mp4")
        if not cls.corrupt.exists():
            cls.corrupt.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(4096) + b"not a video")
        cls.truncated = Fixtures.path("truncated.mp4")
        if not cls.truncated.exists():
            source = Fixtures.build("truncate_source.mp4",
                                    _translation_fixture(640, 360, 24, seconds=3.0))
            data = source.read_bytes()
            cls.truncated.write_bytes(data[: int(len(data) * 0.55)])

    def test_flat_black_is_insufficient_not_static(self):
        result = motion_analyze.analyze_motion(self.black, start=0.0, end=1.0,
                                               sample_fps=8.0, analysis_width=320)
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "insufficient_evidence", result["reason"])
        self.assertEqual(result["summary"]["valid_pair_count"], 0)
        self.assertGreater(len(result["pairs"]), 0)
        for pair in result["pairs"]:
            self.assertEqual(pair["status"], "invalid")
            self.assertIsNotNone(pair["reason"])
            self.assertIsNone(pair["velocity"])
            self.assertIsNone(pair["transform"])
        self.assertIn("flat_or_low_texture_frame", result["summary"]["invalid_reasons"])

    def test_single_frame_is_insufficient(self):
        result = motion_analyze.analyze_motion(self.single, start=0.0, end=1.0,
                                               sample_fps=8.0, analysis_width=320)
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["pairs"], [])
        self.assertFalse(result["coverage"]["fully_covered"])

    def test_corrupt_media_reports_failure(self):
        result = motion_analyze.analyze_motion(self.corrupt, start=0.0, end=1.0)
        assert_finite_json(self, result)
        self.assertIn(result["status"], {"failed", "invalid_input"})
        self.assertTrue(result["errors"])
        self.assertIsNotNone(result["reason"])
        run = _run_cli([str(self.corrupt)])
        self.assertNotEqual(run.returncode, 0)
        json.loads(run.stdout)

    def test_truncated_decode_never_claims_full_coverage(self):
        result = motion_analyze.analyze_motion(self.truncated, start=0.0, end=3.0,
                                               sample_fps=8.0, analysis_width=480)
        assert_finite_json(self, result)
        self.assertNotEqual(result["status"], "complete")
        self.assertFalse(result["coverage"]["fully_covered"] or
                         (result["status"] == "complete"))
        for pair in result["pairs"]:
            if pair["status"] == "valid":
                self.assertGreater(pair["velocity"]["speed_normalized_frame_s"], 1e-6)

    def test_dropout_invalid_pairs_and_no_bridging(self):
        result = motion_analyze.analyze_motion(self.dropout, start=0.0, end=3.0,
                                               sample_fps=8.0, analysis_width=480)
        assert_finite_json(self, result)
        invalid = [pair for pair in result["pairs"] if pair["status"] == "invalid"]
        self.assertTrue(invalid, "black middle segment must not produce valid motion")
        for pair in invalid:
            self.assertIsNone(pair["velocity"])
        for pair in result["pairs"]:
            if pair["status"] == "valid":
                self.assertGreater(pair["velocity"]["speed_normalized_frame_s"], 1e-6)
        # Every segment stays inside one contiguous textured run.
        for segment in result["segments"]:
            members = [pair for pair in result["pairs"]
                       if pair["pair_index"] in range(segment["first_pair"], segment["last_pair"] + 1)]
            self.assertTrue(all(pair["status"] == "valid" for pair in members))
        # Acceleration is absent at segment boundaries that touch the black gap.
        for entry in result["series"]:
            if entry["status"] == "valid" and entry["acceleration_smoothed"] is None:
                self.assertEqual(entry["differentiation"]["reason"], "segment_boundary")

    def test_parameter_and_interval_validation(self):
        cases = [
            ({"start": 2.0, "end": 1.0}, "invalid_interval"),
            ({"start": -1.0}, "invalid_interval"),
            ({"start": 100.0}, "interval_outside_media"),
            ({"sample_fps": 0.0}, "invalid_parameters"),
            ({"analysis_width": 8}, "invalid_parameters"),
            ({"max_seconds": 0.0}, "invalid_parameters"),
            ({"smoothing_window": 0}, "invalid_parameters"),
        ]
        for kwargs, reason in cases:
            with self.subTest(kwargs=kwargs):
                result = motion_analyze.analyze_motion(self.dropout, **kwargs)
                assert_finite_json(self, result)
                self.assertEqual(result["status"], "invalid_input")
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["pairs"], [])
        missing = motion_analyze.analyze_motion("/definitely/not/a/file.mp4")
        self.assertEqual(missing["status"], "invalid_input")
        self.assertEqual(missing["reason"], "file_not_found")


# --------------------------------------------------------------------------------------
# Shots, cuts, gaps, head/tail windows
# --------------------------------------------------------------------------------------

@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class ShotEvidenceTests(unittest.TestCase):
    SPEED = 0.18
    CUT = 1.5

    @classmethod
    def setUpClass(cls):
        cls.two_shot = Fixtures.build(
            "two_shot.mp4", _two_shot_fixture(640, 360, 30, seconds=3.0,
                                              speed=cls.SPEED, cut=cls.CUT))
        cls.forward_shots = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]
        cls.result = motion_analyze.analyze_motion(
            cls.two_shot, start=0.0, end=3.0, shots=cls.forward_shots,
            sample_fps=8.0, analysis_width=480)
        cls.gap_shots = [{"start": 0.0, "end": 1.0}, {"start": 1.4, "end": 3.0}]
        cls.gap_result = motion_analyze.analyze_motion(
            cls.two_shot, start=0.0, end=3.0, shots=cls.gap_shots,
            sample_fps=8.0, analysis_width=480)

    def test_per_shot_signs_and_no_cross_cut_pairs(self):
        result = self.result
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "complete", result["reason"])
        self.assertEqual(len(result["shots"]), 2)
        first, second = result["shots"]
        self.assertEqual(first["status"], "measured")
        self.assertEqual(second["status"], "measured")
        first_vx = first["summary"]["velocity"]["median_vx_frame_width_s"]
        second_vx = second["summary"]["velocity"]["median_vx_frame_width_s"]
        self.assertGreater(first_vx, 0.10)
        self.assertLess(second_vx, -0.10)
        pairs = result["pairs"]
        self.assertTrue(pairs)
        for pair in pairs:
            self.assertGreaterEqual(pair["start_time"], -1e-6)
            self.assertLessEqual(pair["end_time"], 3.0 + 1e-6)
            self.assertFalse(pair["start_time"] < self.CUT - 1e-6
                             and pair["end_time"] > self.CUT + 1e-6,
                             "a pair must never cross the cut")
        first_indices = set(range(first["pair_range"]["first"], first["pair_range"]["last"] + 1))
        second_indices = set(range(second["pair_range"]["first"], second["pair_range"]["last"] + 1))
        self.assertFalse(first_indices & second_indices)
        for segment in result["segments"]:
            self.assertEqual(len({pairs[index]["shot_index"] for index in range(
                segment["first_pair"], segment["last_pair"] + 1)}), 1)

    def test_head_tail_windows_reference_real_pairs(self):
        for shot in self.result["shots"]:
            start = shot["analyzed"]["start"]
            end = shot["analyzed"]["end"]
            self.assertIsNotNone(start)
            self.assertIsNotNone(end)
            head = shot["head_window"]
            tail = shot["tail_window"]
            self.assertIsNotNone(head)
            self.assertIsNotNone(tail)
            for window, expected_start, expected_end in (
                    (head, start, min(start + 1.0, end)),
                    (tail, max(end - 1.0, start), end)):
                self.assertAlmostEqual(window["start"], expected_start, places=6)
                self.assertAlmostEqual(window["end"], expected_end, places=6)
                self.assertTrue(window["pair_indices"])
                for index in window["pair_indices"]:
                    pair = self.result["pairs"][index]
                    self.assertIsNotNone(pair)
                    self.assertGreaterEqual(pair["center_time"], expected_start - 1e-6)
                    self.assertLessEqual(pair["center_time"], expected_end + 1e-6)
                self.assertIn("summary", window)
                self.assertIsNotNone(window["summary"]["velocity"])

    def test_invalid_gap_not_bridged(self):
        result = self.gap_result
        assert_finite_json(self, result)
        pairs = result["pairs"]
        gap_start, gap_end = 1.0, 1.4
        for pair in pairs:
            center = pair["center_time"]
            self.assertFalse(gap_start + 1e-6 < center < gap_end - 1e-6,
                             f"pair at {center} lies inside the unmeasured gap")
        first_shot = [pair for pair in pairs if pair["shot_index"] == 0]
        second_shot = [pair for pair in pairs if pair["shot_index"] == 1]
        self.assertTrue(first_shot and second_shot)
        self.assertLessEqual(max(pair["end_time"] for pair in first_shot), gap_start + 1e-6)
        self.assertGreaterEqual(min(pair["start_time"] for pair in second_shot), gap_end - 1e-6)
        # Boundary samples cannot be differentiated across the gap.
        boundary = [entry for entry in result["series"]
                    if entry["status"] == "valid" and entry["acceleration_smoothed"] is None]
        self.assertTrue(boundary)
        self.assertTrue(all(entry["differentiation"]["reason"] == "segment_boundary"
                            for entry in boundary))

    def test_budget_truncation_reports_coverage(self):
        result = motion_analyze.analyze_motion(
            self.two_shot, start=0.0, end=3.0, shots=self.forward_shots,
            sample_fps=8.0, analysis_width=480, max_seconds=1.7)
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["coverage"]["fully_covered"])
        self.assertIn("max_seconds", result["coverage"]["truncation_reasons"])
        self.assertLessEqual(result["coverage"]["analyzed"]["duration"], 1.7 + 1e-6)
        self.assertTrue(result["shots"][1]["truncated"])
        self.assertLessEqual(result["shots"][1]["analyzed"]["duration"], 0.2 + 1e-6)

    def test_cli_shots_json(self):
        shots_file = Fixtures.path("shots.json")
        shots_file.write_text(json.dumps({"shots": self.forward_shots}), encoding="utf-8")
        run = _run_cli([str(self.two_shot), "--shots-json", str(shots_file),
                        "--sample-fps", "8", "--analysis-width", "480"])
        self.assertEqual(run.returncode, 0, run.stderr)
        document = json.loads(run.stdout)
        assert_finite_json(self, document)
        self.assertEqual(document["request"]["shots"], True)
        self.assertEqual(len(document["shots"]), 2)


# --------------------------------------------------------------------------------------
# Dependency isolation and report metadata
# --------------------------------------------------------------------------------------

@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class DependencyAndMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video = Fixtures.build(
            "cli_translation.mp4", _translation_fixture(640, 360, 24, seconds=2.0))

    def test_help_and_missing_dependency_without_cv2(self):
        stub_dir = Fixtures.path("cv2_stub")
        if not stub_dir.exists():
            stub_dir.mkdir()
            (stub_dir / "cv2.py").write_text("raise ImportError('cv2 blocked for test')\n",
                                             encoding="utf-8")
        environment = {"PYTHONPATH": str(stub_dir)}
        help_run = _run_cli(["--help"], environment)
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("usage:", help_run.stdout)
        run = _run_cli([str(self.video), "--start", "0", "--end", "1"], environment)
        self.assertEqual(run.returncode, 3, run.stderr)
        document = json.loads(run.stdout)
        self.assertEqual(document["status"], "unavailable")
        self.assertEqual(document["reason"], "dependencies_missing")

    def test_cli_output_file_and_invalid_exit_codes(self):
        output = Fixtures.path("cli_output.json")
        run = _run_cli([str(self.video), "--start", "0", "--end", "1",
                        "--output", str(output)])
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(output.exists())
        document = json.loads(output.read_text(encoding="utf-8"))
        assert_finite_json(self, document)
        self.assertEqual(document["status"], "complete")
        invalid = _run_cli([str(self.video), "--start", "2", "--end", "1"])
        self.assertEqual(invalid.returncode, 2, invalid.stderr)
        self.assertEqual(json.loads(invalid.stdout)["status"], "invalid_input")
        missing = _run_cli([str(Fixtures.path("does-not-exist.mp4"))])
        self.assertEqual(missing.returncode, 2, missing.stderr)

    def test_third_party_notices_present(self):
        base = Path(__file__).resolve().parents[1] / "third_party" / "rea-video"
        license_text = (base / "LICENSE").read_text(encoding="utf-8")
        notice_text = (base / "NOTICE").read_text(encoding="utf-8")
        upstream_text = (base / "UPSTREAM.md").read_text(encoding="utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn("Copyright 2026 JuneYaooo", notice_text)
        self.assertIn("3974260fb52476b453b5992f667e7dc1dd83c6eb", upstream_text)
        self.assertIn("features.py", upstream_text)


if __name__ == "__main__":
    unittest.main()
