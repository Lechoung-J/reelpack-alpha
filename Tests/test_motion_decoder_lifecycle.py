"""Motion decoder regressions: decoder lifecycle and honest coverage.

The decoder fault fixtures are real child processes selected by prepending a fake
``ffmpeg`` to PATH. ffprobe, LK tracking, RANSAC and aggregation remain the real code.
"""
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

CORE = Path(__file__).resolve().parents[1] / "Core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import cv2  # noqa: E402

import motion_analyze  # noqa: E402
from test_motion_analyze import (  # noqa: E402
    HAVE_MEDIA, Fixtures, _black_fixture, _encode, _frame_count, _texture,
    _translation_fixture, assert_finite_json,
)

FAKE_SOURCE = r'''#!/usr/bin/env python3
import os
import re
import signal
import sys
import time

args = sys.argv[1:]
mode = os.environ.get("FAKE_FFMPEG_MODE", "ok")
if os.environ.get("FAKE_FFMPEG_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
pidfile = os.environ.get("FAKE_FFMPEG_PIDFILE")
if pidfile:
    with open(pidfile, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()))

fps = 8.0
width = height = 64
for index, value in enumerate(args):
    if value == "-vf" and index + 1 < len(args):
        match = re.search(r"scale=(\d+):(\d+)", args[index + 1])
        if match:
            width, height = int(match.group(1)), int(match.group(2))
        match = re.search(r"fps=([0-9.]+)", args[index + 1])
        if match:
            fps = float(match.group(1))
frame_size = width * height
delay = float(os.environ.get("FAKE_FFMPEG_DELAY", "0"))


def frame(index):
    data = bytearray(frame_size)
    for y in range(height):
        row = y * width
        for x in range(width):
            data[row + x] = 236 if ((((x + 2 * index) // 4) + (y // 4)) % 2) else 18
    return bytes(data)


def emit(count):
    for index in range(count):
        sys.stdout.buffer.write(frame(index))
        sys.stdout.buffer.flush()
        sys.stderr.write(
            "[Parsed_showinfo_1 @ 0x0] n:%d pts:%d pts_time:%.6f\n"
            % (index, index, index / fps))
        sys.stderr.flush()
        if delay:
            time.sleep(delay)


if mode == "stall":
    time.sleep(60)
    sys.exit(0)
if mode == "stall_partial":
    sys.stdout.buffer.write(frame(0)[: frame_size // 2])
    sys.stdout.buffer.flush()
    time.sleep(60)
    sys.exit(0)
if mode == "zero_exit1":
    sys.exit(1)
if mode == "frames_exit1":
    emit(int(os.environ.get("FAKE_FFMPEG_FRAMES", "9")))
    sys.exit(1)
if mode == "partial_then_stall":
    emit(1)
    sys.stdout.buffer.write(frame(1)[: frame_size // 2])
    sys.stdout.buffer.flush()
    time.sleep(60)
    sys.exit(0)
emit(int(os.environ.get("FAKE_FFMPEG_FRAMES", "9")))
sys.exit(0)
'''


class FakeFFmpeg:
    def __init__(self, mode, frames=9, delay=0.0, ignore_term=False):
        self.directory = Path(tempfile.mkdtemp(prefix="reelpack-fake-ffmpeg-"))
        self.pidfile = self.directory / "decoder.pid"
        self._pid = None
        executable = self.directory / "ffmpeg"
        executable.write_text(FAKE_SOURCE, encoding="utf-8")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.environment = {
            "FAKE_FFMPEG_MODE": mode,
            "FAKE_FFMPEG_FRAMES": str(frames),
            "FAKE_FFMPEG_DELAY": str(delay),
            "FAKE_FFMPEG_IGNORE_TERM": "1" if ignore_term else "",
            "FAKE_FFMPEG_PIDFILE": str(self.pidfile),
        }

    def __enter__(self):
        self._path = mock.patch.dict(os.environ, self.environment)
        old_path = os.environ.get("PATH", "")
        self._path.start()
        os.environ["PATH"] = str(self.directory) + os.pathsep + old_path
        return self

    def __exit__(self, *exc):
        self._pid = self._read_pid()
        self._path.stop()
        shutil.rmtree(self.directory, ignore_errors=True)

    def _read_pid(self):
        if not self.pidfile.exists():
            return None
        return int(self.pidfile.read_text(encoding="ascii").strip())

    def pid(self):
        return self._pid if self._pid is not None else self._read_pid()

    def assert_no_survivor(self, testcase):
        pid = self.pid()
        testcase.assertIsNotNone(pid, "fake decoder never started")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        testcase.fail(f"fake decoder child {pid} survived cleanup")


def _media():
    return Fixtures.build("lifecycle_translation.mp4",
                          _translation_fixture(320, 180, 24, seconds=1.2, vx=0.05, vy=0.0,
                                               seed=21))


def _long_media():
    return Fixtures.build("coverage_twelve_seconds.mp4",
                          _translation_fixture(320, 180, 24, seconds=12.0, vx=0.01, vy=0.0,
                                               seed=22))


def _texture_black_fixture(width, height, fps, seconds, black_frames, seed):
    """Static texture with whole black frames at exact 1/fps sample indices."""
    def build(target):
        texture = _texture(width, height, seed)
        black = np.zeros_like(texture)
        frames = [black if index in black_frames else texture
                  for index in range(_frame_count(seconds, fps))]
        _encode(target, frames, fps)
    return build


def _coverage_texture_black_texture():
    # 3s @ 8fps: black frames 1.0-1.5 inclusive of 1.0, exclusive of 1.5.
    # Pairs touching black occupy 0.875-1.5 (5 invalid pairs).
    return Fixtures.build(
        "coverage_texture_black_texture.mp4",
        _texture_black_fixture(320, 180, 8, 3.0, {8, 9, 10, 11}, seed=31))


def _coverage_two_holes():
    # Single black frames at 0.5 and 1.5: valid first and last pairs, two holes.
    return Fixtures.build(
        "coverage_two_holes.mp4",
        _texture_black_fixture(320, 180, 8, 3.0, {4, 12}, seed=32))


@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class DecoderFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _media()

    def _analyze(self, fake):
        with fake:
            return motion_analyze.analyze_motion(
                self.source, start=0.0, end=1.0, sample_fps=8.0, analysis_width=64)

    def test_expected_frames_then_nonzero_exit_is_not_complete(self):
        fake = FakeFFmpeg("frames_exit1", frames=9)
        result = self._analyze(fake)
        assert_finite_json(self, result)
        self.assertNotEqual(result["status"], "complete")
        self.assertTrue(result["errors"], result["warnings"])
        codes = {error["code"] for error in result["errors"]}
        self.assertIn("decode_error", codes)
        self.assertFalse(result["coverage"]["fully_covered"])
        self.assertFalse(result["coverage"]["measured"]["complete"])
        self.assertGreater(result["summary"]["valid_pair_count"], 0)
        fake.assert_no_survivor(self)

    def test_nonzero_exit_after_partial_output_is_partial(self):
        fake = FakeFFmpeg("frames_exit1", frames=4)
        result = self._analyze(fake)
        assert_finite_json(self, result)
        self.assertIn(result["status"], {"partial", "insufficient_evidence"})
        self.assertTrue(result["errors"])
        self.assertFalse(result["coverage"]["fully_covered"])
        fake.assert_no_survivor(self)

    def test_nonzero_exit_with_zero_frames_fails(self):
        fake = FakeFFmpeg("zero_exit1")
        result = self._analyze(fake)
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "decode_no_frames")
        self.assertFalse(result["coverage"]["fully_covered"])
        fake.assert_no_survivor(self)


@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class DecoderDeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _media()

    def test_stalled_decoder_hits_wall_deadline_and_is_reaped(self):
        fake = FakeFFmpeg("stall")
        started = time.monotonic()
        with mock.patch.object(motion_analyze, "_decode_timeout_seconds", return_value=0.75):
            result = self._analyze(fake)
        elapsed = time.monotonic() - started
        assert_finite_json(self, result)
        self.assertLess(elapsed, 8.0, "wall deadline did not bound the blocked pipe read")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "decode_timeout")
        self.assertFalse(result["coverage"]["fully_covered"])
        fake.assert_no_survivor(self)

    def test_stalled_partial_frame_also_hits_deadline(self):
        fake = FakeFFmpeg("stall_partial")
        with mock.patch.object(motion_analyze, "_decode_timeout_seconds", return_value=0.75):
            result = self._analyze(fake)
        assert_finite_json(self, result)
        self.assertEqual(result["reason"], "decode_timeout")
        fake.assert_no_survivor(self)

    def test_private_decode_cleanup_reports_finished_readers(self):
        fake = FakeFFmpeg("stall")
        with fake, mock.patch.object(motion_analyze, "_decode_timeout_seconds", return_value=0.75):
            decoded = motion_analyze._decode_interval(
                str(self.source), 0.0, 1.0, 8.0, 64, 320, 180,
                cv2, np, max_frames=10)
        decode = decoded["decode"]
        self.assertTrue(decode["timed_out"])
        self.assertEqual(decode["frames_decoded"], 0)
        self.assertTrue(decode["cleanup"]["readers_finished"], decode["cleanup"])
        self.assertFalse(decode["cleanup"]["stdout_reader_alive"])
        self.assertFalse(decode["cleanup"]["stderr_reader_alive"])
        fake.assert_no_survivor(self)

    def test_decoder_ignoring_sigterm_is_killed(self):
        fake = FakeFFmpeg("stall", ignore_term=True)
        with fake, \
                mock.patch.object(motion_analyze, "_decode_timeout_seconds", return_value=0.5), \
                mock.patch.object(motion_analyze, "DECODE_CLEANUP_SECONDS", 0.2):
            decoded = motion_analyze._decode_interval(
                str(self.source), 0.0, 1.0, 8.0, 64, 320, 180, cv2, np, max_frames=10)
        decode = decoded["decode"]
        self.assertTrue(decode["timed_out"])
        self.assertTrue(decode["cleanup"]["killed"], decode["cleanup"])
        self.assertTrue(decode["cleanup"]["readers_finished"], decode["cleanup"])
        fake.assert_no_survivor(self)

    def test_partial_frame_stall_is_terminated(self):
        fake = FakeFFmpeg("partial_then_stall")
        with fake, mock.patch.object(motion_analyze, "_decode_timeout_seconds", return_value=0.75):
            decoded = motion_analyze._decode_interval(
                str(self.source), 0.0, 1.0, 8.0, 64, 320, 180,
                cv2, np, max_frames=10)
        self.assertTrue(decoded["decode"]["timed_out"])
        self.assertTrue(decoded["decode"]["cleanup"]["readers_finished"])
        fake.assert_no_survivor(self)

    def test_partial_frame_stall_api_reports_timeout(self):
        fake = FakeFFmpeg("partial_then_stall")
        with mock.patch.object(motion_analyze, "_decode_timeout_seconds", return_value=0.75):
            result = self._analyze(fake)
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "partial")
        self.assertIn("decode_timeout", result["reason"])
        self.assertIn("decode_timeout", {error["code"] for error in result["errors"]})
        self.assertFalse(result["coverage"]["fully_covered"])
        fake.assert_no_survivor(self)

    def test_exception_during_measurement_still_cleans_up_child(self):
        fake = FakeFFmpeg("ok", frames=9)
        with fake:
            with mock.patch.object(motion_analyze, "_measure_pair",
                                   side_effect=RuntimeError("injected measurement failure")):
                result = motion_analyze.analyze_motion(
                    self.source, start=0.0, end=1.0, sample_fps=8.0, analysis_width=64)
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "decode_failed")
        fake.assert_no_survivor(self)

    def test_module_cap_termination_is_not_called_decoder_failure(self):
        fake = FakeFFmpeg("ok", frames=12, delay=0.05)
        with fake, mock.patch.object(motion_analyze, "_decode_timeout_seconds", return_value=5.0):
            decoded = motion_analyze._decode_interval(
                str(self.source), 0.0, 1.2, 8.0, 64, 320, 180,
                cv2, np, max_frames=3)
        decode = decoded["decode"]
        self.assertTrue(decode["cap_hit"])
        self.assertTrue(decode["intentional_termination"], decode)
        self.assertFalse(decode["decoder_failed"], decode)
        self.assertTrue(decode["truncated"])
        fake.assert_no_survivor(self)

    def _analyze(self, fake):
        with fake:
            return motion_analyze.analyze_motion(
                self.source, start=0.0, end=1.0,
                sample_fps=8.0, analysis_width=64)


@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class NumericParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _media()

    def _assert_invalid(self, **kwargs):
        result = motion_analyze.analyze_motion(self.source, start=0.0, end=0.5, **kwargs)
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "invalid_input", result)
        self.assertTrue(result["errors"])
        json.dumps(result, allow_nan=False)

    def test_infinite_analysis_width(self):
        self._assert_invalid(analysis_width=float("inf"), sample_fps=8.0)
        self._assert_invalid(analysis_width=float("-inf"), sample_fps=8.0)

    def test_nan_analysis_width(self):
        self._assert_invalid(analysis_width=float("nan"), sample_fps=8.0)

    def test_infinite_smoothing_window(self):
        self._assert_invalid(smoothing_window=float("inf"), sample_fps=8.0)
        self._assert_invalid(smoothing_window=float("-inf"), sample_fps=8.0)

    def test_nan_smoothing_window(self):
        self._assert_invalid(smoothing_window=float("nan"), sample_fps=8.0)

    def test_extreme_numeric_values_are_structured(self):
        self._assert_invalid(analysis_width=10 ** 30, sample_fps=8.0)
        self._assert_invalid(smoothing_window=10 ** 30, sample_fps=8.0)
        self._assert_invalid(analysis_width=10 ** 5000, sample_fps=8.0)
        self._assert_invalid(smoothing_window=10 ** 5000, sample_fps=8.0)
        self._assert_invalid(analysis_width="not-a-number", sample_fps=8.0)
        self._assert_invalid(smoothing_window=float("1e309"), sample_fps=8.0)


@unittest.skipUnless(HAVE_MEDIA, "ffmpeg/ffprobe/cv2 required")
class CoverageSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _media()
        cls.long_source = _long_media()

    def test_single_sample_interval_has_no_measured_coverage(self):
        result = motion_analyze.analyze_motion(
            self.source, start=0.371, end=0.521, sample_fps=8.0, analysis_width=64)
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertLessEqual(result["coverage"]["decoded"]["sample_count"], 1)
        self.assertEqual(result["coverage"]["measured"]["pair_count"], 0)
        self.assertEqual(result["coverage"]["measured"]["valid_pair_count"], 0)
        self.assertEqual(result["coverage"]["measured"]["duration"], 0.0)
        self.assertEqual(result["coverage"]["measured"]["ranges"], [])
        self.assertFalse(result["coverage"]["measured"]["complete"])
        self.assertFalse(result["coverage"]["fully_covered"])
        self.assertEqual(result["coverage"]["decoded"]["start"],
                         result["coverage"]["decoded"]["end"])

    def test_twelve_second_request_reports_last_sample_not_requested_end(self):
        result = motion_analyze.analyze_motion(
            self.long_source, start=0.0, end=12.0, sample_fps=8.0, analysis_width=64)
        assert_finite_json(self, result)
        coverage = result["coverage"]
        self.assertAlmostEqual(coverage["decoded"]["end"], 11.875, places=6)
        self.assertAlmostEqual(coverage["measured"]["end"], 11.875, places=6)
        self.assertAlmostEqual(coverage["decoded"]["unobserved_tail_seconds"], 0.125, places=6)
        self.assertAlmostEqual(coverage["measured"]["unobserved_tail_seconds"], 0.125, places=6)
        self.assertNotAlmostEqual(coverage["measured"]["end"], 12.0, places=6)
        self.assertTrue(coverage["measured"]["complete"])
        self.assertTrue(coverage["fully_covered"])

    def test_empty_interval_has_no_measured_coverage(self):
        result = motion_analyze.analyze_motion(
            self.source, start=0.5, end=0.5, sample_fps=8.0, analysis_width=64)
        assert_finite_json(self, result)
        self.assertEqual(result["status"], "invalid_input")
        self.assertFalse(result["coverage"]["fully_covered"])
        self.assertFalse(result["coverage"]["decoded"]["complete"])
        self.assertFalse(result["coverage"]["measured"]["complete"])

    def test_multi_shot_zero_measurement_component_is_a_gap(self):
        result = motion_analyze.analyze_motion(
            self.source, start=0.0, end=1.0, sample_fps=8.0, analysis_width=64,
            shots=[[0.0, 0.6], [0.6, 0.7]])
        assert_finite_json(self, result)
        coverage = result["coverage"]
        self.assertFalse(coverage["fully_covered"])
        self.assertFalse(coverage["measured"]["complete"])
        self.assertTrue(coverage["measurement_gaps"], coverage)
        gap_shots = {gap["shot_index"] for gap in coverage["measurement_gaps"]}
        self.assertIn(1, gap_shots)
        component = next(item for item in coverage["components"] if item["shot_index"] == 1)
        self.assertLessEqual(component["sample_count"], 1)
        self.assertEqual(component["valid_pair_count"], 0)
        self.assertFalse(component["complete"])

    def test_internal_black_region_is_not_measured_coverage(self):
        result = motion_analyze.analyze_motion(
            _coverage_texture_black_texture(), start=0.0, end=3.0,
            sample_fps=8.0, analysis_width=160)
        assert_finite_json(self, result)
        coverage = result["coverage"]
        # Decoding sampled the whole 3s clip at 8fps, but LK could not measure the
        # black region. Sampling completeness and measured completeness differ.
        self.assertEqual(coverage["decoded"]["sample_count"], 24)
        self.assertTrue(coverage["decoded"]["complete"])
        self.assertEqual(coverage["measured"]["pair_count"], 23)
        self.assertEqual(coverage["measured"]["valid_pair_count"], 18)
        self.assertAlmostEqual(coverage["measured"]["duration"], 2.25, places=6)
        self.assertAlmostEqual(coverage["measured"]["envelope_duration"], 2.875, places=6)
        self.assertNotAlmostEqual(coverage["measured"]["duration"],
                                  coverage["measured"]["envelope_duration"], places=6)
        self.assertAlmostEqual(coverage["measured"]["start"], 0.0, places=6)
        self.assertAlmostEqual(coverage["measured"]["end"], 2.875, places=6)
        ranges = coverage["measured"]["ranges"]
        self.assertEqual(len(ranges), 2)
        self.assertAlmostEqual(ranges[0]["start"], 0.0, places=6)
        self.assertAlmostEqual(ranges[0]["end"], 0.875, places=6)
        self.assertAlmostEqual(ranges[1]["start"], 1.5, places=6)
        self.assertAlmostEqual(ranges[1]["end"], 2.875, places=6)
        self.assertFalse(coverage["measured"]["complete"])
        self.assertFalse(coverage["fully_covered"])
        self.assertEqual(result["status"], "partial")
        component = coverage["components"][0]
        self.assertEqual(component["valid_pair_count"], 18)
        self.assertAlmostEqual(component["measured_duration"], 2.25, places=6)
        self.assertAlmostEqual(component["measured_envelope_duration"], 2.875, places=6)
        self.assertFalse(component["complete"])
        gaps = coverage["measurement_gaps"]
        self.assertEqual(len(gaps), 1)
        gap = gaps[0]
        self.assertEqual(gap["kind"], "internal")
        self.assertAlmostEqual(gap["start"], 0.875, places=6)
        self.assertAlmostEqual(gap["end"], 1.5, places=6)
        self.assertAlmostEqual(gap["duration"], 0.625, places=6)
        self.assertEqual(gap["invalid_pair_count"], 5)
        self.assertIn("flat_or_low_texture_frame", gap["invalid_pair_reasons"])

    def test_multiple_holes_keep_valid_first_and_last_pairs(self):
        result = motion_analyze.analyze_motion(
            _coverage_two_holes(), start=0.0, end=3.0,
            sample_fps=8.0, analysis_width=160)
        assert_finite_json(self, result)
        coverage = result["coverage"]
        self.assertEqual(coverage["measured"]["pair_count"], 23)
        self.assertEqual(coverage["measured"]["valid_pair_count"], 19)
        self.assertAlmostEqual(coverage["measured"]["duration"], 2.375, places=6)
        self.assertAlmostEqual(coverage["measured"]["envelope_duration"], 2.875, places=6)
        ranges = coverage["measured"]["ranges"]
        self.assertEqual(len(ranges), 3)
        self.assertAlmostEqual(ranges[0]["start"], 0.0, places=6)
        self.assertAlmostEqual(ranges[0]["end"], 0.375, places=6)
        self.assertAlmostEqual(ranges[1]["start"], 0.625, places=6)
        self.assertAlmostEqual(ranges[1]["end"], 1.375, places=6)
        self.assertAlmostEqual(ranges[2]["start"], 1.625, places=6)
        self.assertAlmostEqual(ranges[2]["end"], 2.875, places=6)
        gaps = [(gap["start"], gap["end"], gap["kind"])
                for gap in coverage["measurement_gaps"]]
        self.assertEqual(len(gaps), 2)
        self.assertAlmostEqual(gaps[0][0], 0.375, places=6)
        self.assertAlmostEqual(gaps[0][1], 0.625, places=6)
        self.assertEqual(gaps[0][2], "internal")
        self.assertAlmostEqual(gaps[1][0], 1.375, places=6)
        self.assertAlmostEqual(gaps[1][1], 1.625, places=6)
        self.assertEqual(gaps[1][2], "internal")
        self.assertFalse(coverage["measured"]["complete"])
        self.assertEqual(result["status"], "partial")

    def test_zero_valid_pairs_has_zero_measured_duration(self):
        black = Fixtures.build("coverage_black_zero_pairs.mp4",
                               _black_fixture(320, 180, 8, seconds=1.0))
        result = motion_analyze.analyze_motion(
            black, start=0.0, end=1.0, sample_fps=8.0, analysis_width=160)
        assert_finite_json(self, result)
        coverage = result["coverage"]
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(coverage["measured"]["valid_pair_count"], 0)
        self.assertEqual(coverage["measured"]["duration"], 0.0)
        self.assertEqual(coverage["measured"]["ranges"], [])
        self.assertFalse(coverage["measured"]["complete"])
        self.assertFalse(coverage["fully_covered"])
        self.assertFalse(coverage["components"][0]["complete"])
        self.assertEqual(coverage["components"][0]["measured_duration"], 0.0)
        gap = coverage["measurement_gaps"][0]
        self.assertEqual(gap["kind"], "no_valid_pairs")
        self.assertAlmostEqual(gap["start"], 0.0, places=6)
        self.assertAlmostEqual(gap["end"], 1.0, places=6)

    def test_disjoint_shots_do_not_count_inter_shot_gap(self):
        result = motion_analyze.analyze_motion(
            self.long_source, start=0.0, end=3.0, sample_fps=8.0, analysis_width=64,
            shots=[[0.0, 1.0], [2.0, 3.0]])
        assert_finite_json(self, result)
        coverage = result["coverage"]
        self.assertAlmostEqual(coverage["requested"]["shots_total_duration"], 2.0, places=6)
        self.assertAlmostEqual(coverage["measured"]["duration"], 1.75, places=6)
        ranges = coverage["measured"]["ranges"]
        self.assertEqual(len(ranges), 2)
        self.assertAlmostEqual(ranges[0]["start"], 0.0, places=6)
        self.assertAlmostEqual(ranges[0]["end"], 0.875, places=6)
        self.assertAlmostEqual(ranges[1]["start"], 2.0, places=6)
        self.assertAlmostEqual(ranges[1]["end"], 2.875, places=6)
        for gap in coverage["measurement_gaps"]:
            self.assertFalse(gap["start"] <= 1.0 + 1e-6 and gap["end"] >= 2.0 - 1e-6,
                             f"inter-shot space recorded as a measurement gap: {gap}")
        self.assertTrue(coverage["decoded"]["complete"])
        self.assertTrue(coverage["measured"]["complete"])
        self.assertTrue(coverage["fully_covered"])
        self.assertEqual(result["status"], "complete")

    def test_selected_disjoint_shots_complete_without_unsupplied_tail(self):
        result = motion_analyze.analyze_motion(
            self.long_source, start=0.0, end=8.0, sample_fps=8.0, analysis_width=480,
            shots=[[1.0, 2.0], [4.0, 5.0]])
        assert_finite_json(self, result)
        coverage = result["coverage"]
        self.assertEqual(coverage["decoded"]["sample_count"], 16)
        self.assertEqual(coverage["measured"]["pair_count"], 14)
        self.assertEqual(coverage["measured"]["valid_pair_count"], 14)
        self.assertAlmostEqual(coverage["measured"]["duration"], 1.75, places=6)
        self.assertAlmostEqual(coverage["measured"]["envelope_duration"], 3.875, places=6)
        self.assertAlmostEqual(coverage["decoded"]["duration"], 1.75, places=6)
        self.assertAlmostEqual(coverage["decoded"]["envelope_duration"], 3.875, places=6)
        ranges = coverage["measured"]["ranges"]
        self.assertEqual(len(ranges), 2)
        self.assertAlmostEqual(ranges[0]["start"], 1.0, places=6)
        self.assertAlmostEqual(ranges[0]["end"], 1.875, places=6)
        self.assertAlmostEqual(ranges[1]["start"], 4.0, places=6)
        self.assertAlmostEqual(ranges[1]["end"], 4.875, places=6)
        self.assertEqual(coverage["measurement_gaps"], [])
        self.assertEqual(coverage["incomplete_reasons"], [])
        self.assertAlmostEqual(coverage["decoded"]["unobserved_tail_seconds"], 0.125, places=6)
        self.assertAlmostEqual(coverage["measured"]["unobserved_tail_seconds"], 0.125, places=6)
        self.assertTrue(coverage["decoded"]["complete"])
        self.assertTrue(coverage["measured"]["complete"])
        self.assertTrue(coverage["fully_covered"])
        self.assertEqual(result["status"], "complete")
        self.assertIsNone(result["reason"])
        # The outer request stays visible as outer metadata.
        self.assertAlmostEqual(coverage["requested"]["end"], 8.0, places=6)
        self.assertAlmostEqual(coverage["clip_to_media"]["end"], 8.0, places=6)
        for index, (selected_start, selected_end) in enumerate(
                ((1.0, 2.0), (4.0, 5.0))):
            component = coverage["components"][index]
            self.assertEqual(component["shot_index"], index)
            self.assertAlmostEqual(component["selected_start"], selected_start, places=6)
            self.assertAlmostEqual(component["selected_end"], selected_end, places=6)
            self.assertEqual(component["sample_count"], 8)
            self.assertEqual(component["valid_pair_count"], 7)
            self.assertAlmostEqual(component["measured_duration"], 0.875, places=6)
            self.assertAlmostEqual(component["decoded_unobserved_tail_seconds"],
                                   0.125, places=6)
            self.assertAlmostEqual(component["measured_unobserved_tail_seconds"],
                                   0.125, places=6)
            self.assertTrue(component["complete"])

    def test_end_none_selected_shots_ignore_media_tail(self):
        result = motion_analyze.analyze_motion(
            self.long_source, start=0.0, end=None, sample_fps=8.0, analysis_width=480,
            shots=[[1.0, 2.0], [4.0, 5.0]])
        assert_finite_json(self, result)
        coverage = result["coverage"]
        self.assertAlmostEqual(coverage["requested"]["end"], 12.0, places=6)
        self.assertAlmostEqual(coverage["measured"]["duration"], 1.75, places=6)
        self.assertAlmostEqual(coverage["measured"]["unobserved_tail_seconds"], 0.125,
                               places=6)
        self.assertTrue(coverage["fully_covered"])
        self.assertEqual(result["status"], "complete")

    def test_unsorted_and_overlapping_selected_shots(self):
        unsorted = motion_analyze.analyze_motion(
            self.long_source, start=0.0, end=8.0, sample_fps=8.0, analysis_width=480,
            shots=[[4.0, 5.0], [1.0, 2.0]])
        assert_finite_json(self, unsorted)
        coverage = unsorted["coverage"]
        self.assertEqual([component["selected_start"]
                          for component in coverage["components"]], [4.0, 1.0])
        self.assertAlmostEqual(coverage["measured"]["duration"], 1.75, places=6)
        self.assertTrue(coverage["fully_covered"])
        self.assertEqual(unsorted["status"], "complete")
        overlap = motion_analyze.analyze_motion(
            self.long_source, start=0.0, end=3.0, sample_fps=8.0, analysis_width=480,
            shots=[[0.0, 2.0], [1.0, 3.0]])
        assert_finite_json(self, overlap)
        coverage = overlap["coverage"]
        component_sum = sum(component["measured_duration"]
                            for component in coverage["components"])
        self.assertAlmostEqual(coverage["measured"]["duration"], 2.875, places=6)
        self.assertLess(coverage["measured"]["duration"], component_sum - 1e-6)
        self.assertAlmostEqual(coverage["decoded"]["unobserved_tail_seconds"], 0.125,
                               places=6)
        self.assertTrue(coverage["fully_covered"])
        self.assertEqual(overlap["status"], "complete")

    def test_overall_clipped_shot_uses_selected_bounds(self):
        result = motion_analyze.analyze_motion(
            self.long_source, start=1.0, end=3.0, sample_fps=8.0, analysis_width=480,
            shots=[[0.0, 4.0]])
        assert_finite_json(self, result)
        coverage = result["coverage"]
        component = coverage["components"][0]
        self.assertAlmostEqual(component["selected_start"], 1.0, places=6)
        self.assertAlmostEqual(component["selected_end"], 3.0, places=6)
        self.assertAlmostEqual(component["measured_duration"], 1.875, places=6)
        self.assertAlmostEqual(coverage["measured"]["duration"], 1.875, places=6)
        self.assertAlmostEqual(coverage["measured"]["unobserved_tail_seconds"], 0.125,
                               places=6)
        self.assertTrue(coverage["fully_covered"])
        self.assertEqual(result["status"], "complete")

    def test_selected_invalid_gap_with_later_complete_shot_is_partial(self):
        result = motion_analyze.analyze_motion(
            _coverage_texture_black_texture(), start=0.0, end=3.0,
            sample_fps=8.0, analysis_width=160, shots=[[1.0, 2.0], [2.0, 3.0]])
        assert_finite_json(self, result)
        coverage = result["coverage"]
        self.assertFalse(coverage["fully_covered"])
        self.assertFalse(coverage["measured"]["complete"])
        self.assertEqual(result["status"], "partial")
        first, second = coverage["components"]
        self.assertFalse(first["complete"])
        self.assertTrue(second["complete"])
        self.assertAlmostEqual(second["measured_end"], 2.875, places=6)
        gaps = [gap for gap in coverage["measurement_gaps"] if gap["shot_index"] == 0]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["kind"], "leading")
        self.assertAlmostEqual(gaps[0]["start"], 1.0, places=6)
        self.assertAlmostEqual(gaps[0]["end"], 1.5, places=6)

    def test_selected_budget_truncated_component_is_not_masked(self):
        result = motion_analyze.analyze_motion(
            self.long_source, start=0.0, end=8.0, sample_fps=8.0, analysis_width=64,
            shots=[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], max_seconds=2.2)
        assert_finite_json(self, result)
        coverage = result["coverage"]
        self.assertEqual(result["status"], "partial")
        self.assertFalse(coverage["decoded"]["complete"])
        self.assertFalse(coverage["measured"]["complete"])
        self.assertFalse(coverage["fully_covered"])
        self.assertTrue(coverage["components"][0]["complete"])
        self.assertTrue(coverage["components"][1]["complete"])
        self.assertFalse(coverage["components"][2]["complete"])
        self.assertGreater(coverage["measured"]["unobserved_tail_seconds"], 0.125)
        self.assertIn("truncated_decode", coverage["decoded"]["reasons"])

    def test_overlapping_shots_are_deduplicated(self):
        result = motion_analyze.analyze_motion(
            self.long_source, start=0.0, end=1.2, sample_fps=8.0, analysis_width=64,
            shots=[[0.0, 1.0], [0.6, 1.2]])
        assert_finite_json(self, result)
        coverage = result["coverage"]
        expected = motion_analyze._merge_interval_union(
            (item_start, item_end)
            for component in coverage["components"]
            for item_start, item_end in
            ((range_["start"], range_["end"]) for range_ in component["covered_ranges"]))
        self.assertAlmostEqual(motion_analyze._interval_union_duration(expected),
                               coverage["measured"]["duration"], places=6)
        component_sum = sum(component["measured_duration"]
                            for component in coverage["components"])
        self.assertLess(coverage["measured"]["duration"], component_sum - 1e-6)
        ranges = coverage["measured"]["ranges"]
        for previous, current in zip(ranges, ranges[1:]):
            self.assertGreaterEqual(current["start"], previous["end"] - 1e-6)


class IntervalCoveragePrimitiveTests(unittest.TestCase):
    def test_union_deduplicates_overlaps_and_drops_nonpositive(self):
        merged = motion_analyze._merge_interval_union(
            [(0.0, 1.0), (0.5, 1.5), (1.5, 2.0), (2.0, 2.0),
             (float("nan"), 3.0), (3.0, float("inf"))])
        self.assertEqual(merged, [(0.0, 2.0)])
        self.assertAlmostEqual(motion_analyze._interval_union_duration(merged), 2.0)

    def test_multiple_holes_with_valid_first_and_last_range(self):
        covered = motion_analyze._clip_interval_union(
            [(0.0, 0.3), (0.5, 0.7), (0.9, 1.2)], 0.0, 1.2)
        self.assertEqual(covered, [(0.0, 0.3), (0.5, 0.7), (0.9, 1.2)])
        gaps = motion_analyze._interval_gaps(covered, 0.0, 1.2)
        self.assertEqual(gaps, [(0.3, 0.5), (0.7, 0.9)])
        self.assertAlmostEqual(motion_analyze._interval_union_duration(covered), 0.8)

    def test_leading_and_trailing_gaps_use_actual_bounds(self):
        covered = motion_analyze._clip_interval_union(
            [(0.2, 0.4), (0.6, 0.8)], 0.0, 1.0)
        gaps = motion_analyze._interval_gaps(covered, 0.0, 1.0)
        self.assertEqual(gaps, [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)])

    def test_clip_excludes_outside_and_merges_adjacent(self):
        covered = motion_analyze._clip_interval_union(
            [(-1.0, 0.25), (0.25, 0.5), (0.9, 2.0)], 0.0, 1.0)
        self.assertEqual(covered, [(0.0, 0.5), (0.9, 1.0)])
        self.assertEqual(motion_analyze._interval_gaps(covered, 0.0, 1.0),
                         [(0.5, 0.9)])


if __name__ == "__main__":
    unittest.main()
