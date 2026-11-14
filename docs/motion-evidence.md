# Motion evidence (`Core/motion_analyze.py`)

Quantitative image-motion evidence for a video interval or supplied shot intervals.
The module measures the dominant global similarity transform (pan, zoom, rotation)
between sampled frames and reports velocities, accelerations, rates, validity counts,
fit quality and residual local motion. It is **evidence for a downstream multimodal
Agent**, not an interpretation: it makes no claim about physical camera speed, camera
travel direction, focal length, depth, actor identity or content category.

- Schema: `reelpack.motion-evidence` (version `1`)
- Backend: `opencv-lk-ransac-similarity-v1` (module version `1.0.0`)
- Dependencies: OpenCV, NumPy, Python standard library, `ffmpeg`/`ffprobe`
- Inputs are opened read-only; no decode cache or report file is written unless an
  explicit `--output` path is passed.

The global-motion fit is adapted from JuneYaooo/rea-video
(`rea_video/features.py`, commit `3974260fb52476b453b5992f667e7dc1dd83c6eb`,
Apache-2.0, Copyright 2026 JuneYaooo). The adaptation is not verbatim: backward
tracking order, timestamp handling, center displacement and validity semantics were
changed. See `third_party/rea-video/UPSTREAM.md`; license and notice are retained in
`third_party/rea-video/`.

## CLI

```sh
python3 Core/motion_analyze.py INPUT --start 0 --end 3 \
    --sample-fps 8 --analysis-width 480 --max-seconds 120
```

Writes one JSON document to stdout and diagnostics to stderr. `--help` works on an
interpreter without OpenCV; a measurement attempt without OpenCV/NumPy returns
`status: "unavailable"` as JSON with exit code 3.

| Option | Default | Meaning |
|---|---|---|
| `--start` | `0` | interval start, seconds (source presentation timeline) |
| `--end` | media end | interval end, seconds |
| `--sample-fps` | `8` | analysis sampling rate |
| `--analysis-width` | `480` | grayscale analysis frame width; height follows display aspect |
| `--max-seconds` | `120` | cumulative analyzed-duration budget |
| `--shots-json` | – | JSON file: list of `{start, end}` shots (or `{"shots": [...]}`) |
| `--head-tail` | `1.0` | per-shot head/tail window length, clipped to the shot |
| `--output` | – | optional path to also write the same JSON (stdout is always written) |

Exit codes: `0` measured (`complete`, `partial` or `insufficient_evidence`; the
measurement ran), `2` invalid input/arguments, `3` missing dependency, `4`
decode/analysis failure.

## Installed asset adapter (`reelpack motion`)

```bash
reelpack motion <asset_id> --start 10.0 --end 12.0 --library-root <library>
reelpack motion <asset_id> --start 10.0 --end 12.0 --save --library-root <library>
```

`Core/asset_motion.py` resolves an **existing** package, its `analysis.json` and its
archived original, then calls the API below. It adds no measurement mathematics.

- The range is mandatory and finite (`0 <= start < end`); it is never widened to the
  whole video. The media is clipped by `motion_analyze` and coverage is reported.
- Current saved shot intervals are read from `analysis.json`, clipped to the request
  and passed as `shots`. Frames never pair across a saved cut. If no saved shot
  overlaps, or the saved shot list is empty, the adapter reports
  `cuts.status: "unavailable"` with an explicit reason and measures the single range
  without inventing cut boundaries. The module does not run shot detection.
- Statuses and exit codes are unchanged: usable statuses exit 0,
  `invalid_input` 2, `unavailable` 3, `failed` 4. Offline archived media is
  `failed`/`media_offline`, never a zero-motion measurement.
- Read-only measurement creates no library, cache, lock, media or supplement file.
- `--save` writes one immutable `reelpack.engine-measurement` supplement beside the
  package (see `docs/agent-supplements.md`) for usable statuses only. Persisted
  `source.path`/`resolved_path` are removed in favour of
  `media_relative_path` + `path_base: "library_root"`; absolute runtime paths embedded
  in warning text are rewritten to library-relative form. Saving requires the same
  source-analysis SHA256 as at measurement start and the same archived content SHA256
  as `asset.json`; a changed analysis, offline media or digest mismatch refuses the
  save and leaves earlier supplement bytes untouched.

## Python API

```python
from motion_analyze import analyze_motion

result = analyze_motion(
    "/path/clip.mp4",
    start=0.0,
    end=3.0,
    shots=None,                 # or [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]
    sample_fps=8.0,
    analysis_width=480,
    max_seconds=120.0,
    smoothing_window=5,         # optional, odd centered moving-average window, 1-100000
    head_tail_seconds=1.0,      # optional, per-shot head/tail windows
)
```

`start`/`end` clip every supplied shot when `shots` is provided. Supplied shots are
decoded separately, so **no frame pair crosses a caller-supplied shot boundary or an
invalid gap represented in the sampled pairs**. A frontend requesting a single interval
passes `shots=None`.

This module does **not** perform automatic shot/cut detection. If an unsplit interval
contains a cut that the caller did not supply as a shot boundary, a pair may span that
cut and smoothing/differentiation may continue across it; that is a caller/input
limitation, not a continuity claim. Passing the known cuts as `shots` is required to
obtain the no-crossing guarantee.

## Units and signs

Image coordinates after applying source rotation metadata: **x right, y down**.
Translation, velocity and acceleration describe **image-content movement**, not camera
travel. Content moving right has positive `vx`; content moving down has positive `vy`.

| Field | Unit / convention |
|---|---|
| `start_time`, `end_time`, `center_time`, `delta_t` | seconds (source presentation timeline) |
| `vx_frame_width_s` | frame-widths/s: `dx_center_px / analysis_width / delta_t` |
| `vy_frame_height_s` | frame-heights/s: `dy_center_px / analysis_height / delta_t` |
| `speed_normalized_frame_s` | `sqrt(vx² + vy²)` in normalized frame units/s |
| `speed_px_s` | analysis-frame pixels/s |
| `zoom_log_scale_s` | `log(scale) / delta_t`; **positive = content expands (zoom in)** |
| `rotation_deg_s` | degrees/s; **positive = clockwise on screen** |
| `acceleration.ax`, `.ay`, `.magnitude` | frame-widths/s², frame-heights/s², `sqrt(ax²+ay²)` |
| `residual_*_px`, `flow_*_px` | pixels in the analysis frame |

The fitted transform is `prev -> next` as a similarity matrix
`[[a, -b, tx], [b, a, ty]]`. Raw affine translation `tx, ty` is measured at the image
origin and is contaminated by a centered zoom/rotation, so velocity uses
**center displacement**:

```
dx_center = (a-1)*cx - b*cy + tx
dy_center =  b*cx + (a-1)*cy + ty        # c = frame center
```

Thus a perfectly centered zoom or rotation yields ~zero lateral velocity.

## Output schema

Top-level keys (all numeric values are finite JSON numbers or `null`; NaN/Infinity are
never emitted):

- `schema`, `schema_version`, `backend`, `backend_version`
- `status`, `reason` — see statuses below
- `source` — path, resolved path, `digest_sha256`, size, duration, start time,
  coded/display width/height, rotation, codec, pixel format, format name, and
  `fps: {r_frame_rate, avg_frame_rate, nominal, variable}`
- `request` — caller arguments
- `parameters` — effective analysis parameters (feature/LK/RANSAC thresholds,
  smoothing window, frame budget, analysis frame size)
- `units`, `sign_convention`, `limitations`
- `coverage` — `requested`, `clip_to_media`, `budget`, `analyzed`, `decoded`,
  `measured`, `measurement_gaps`, `components`, `incomplete_reasons`,
  `fully_covered`, `truncation_reasons`
- `timeline` — timestamp source, sampling fps, analysis frame size, absolute sample
  times, timestamp validation
- `pairs` — per sampled frame pair (below)
- `series` — raw/smoothed velocity and acceleration per pair
- `segments` — contiguous valid runs (no gap crossing)
- `shots` — per-shot reports when `shots` was supplied, otherwise `null`
- `summary` — aggregate valid-pair statistics
- `warnings`, `errors` — structured `{code, message}` entries

### Pair record

```json
{
  "pair_index": 12, "shot_index": null, "segment_index": 3,
  "from_frame": 12, "to_frame": 13,
  "start_time": 1.5, "end_time": 1.625, "center_time": 1.5625, "delta_t": 0.125,
  "status": "valid", "reason": null,
  "transform": {"model": "similarity", "scale": 1.0004, "rotation_deg": 0.12,
                 "matrix_2x3": [[...]], "translation_origin_px": {"x": 0.7, "y": -0.2},
                 "center_px": {"x": 240.0, "y": 135.0}},
  "center_displacement_px": {"x": 0.9, "y": -0.1},
  "velocity": {"vx_frame_width_s": 0.015, "vy_frame_height_s": -0.003,
                "speed_normalized_frame_s": 0.0153, "speed_px_s": 7.2,
                "zoom_log_scale_s": 0.003, "rotation_deg_s": 0.96},
  "tracking": {"detected_corners": 300, "forward_valid": 291, "forward_backward_valid": 286,
                "used_points": 286, "inlier_count": 284, "inlier_fraction": 0.993,
                "residual_median_px": 0.04, "residual_p95_px": 0.31,
                "flow_median_px": 7.2, "flow_p90_px": 7.4},
  "quality": {"inlier_fraction": 0.993, "residual_median_px": 0.04,
               "local_motion_ratio": 0.0, "photometric_mad": 0.6,
               "photometric_block_p98": 2.1},
  "frames": {"from": {"mean": ..., "std": ..., "gradient_mean": ...}, "to": {...}},
  "local_motion_note": "..."
}
```

`status` is `valid` or `invalid`; invalid pairs carry a machine-readable `reason`
(`flat_or_low_texture_frame`, `too_few_features`, `too_few_forward_tracks`,
`forward_backward_inconsistent`, `ransac_failed`, `degenerate_transform`,
`low_inlier_support`, `implausible_transform`, `nonpositive_delta_t`,
`missing_timestamp`, …) and `null` transform/velocity. **A failed measurement is never
reported as static zero motion.**

Residual/local-motion values are evidence relative to the fitted dominant transform
(the field is explicitly labelled as such). They are not a foreground/background
segmentation and do not identify objects.

### Series record

Every pair has an entry in `series`:

- `velocity_raw` — the pair velocity above
- `velocity_smoothed` — centered moving average (`parameters.smoothing`,
  default window 5 samples) over **valid contiguous samples in the same segment only**
- `smoothing` — method, window, `used_samples`, `quality` (`full`, `reduced`,
  `not_available`)
- `acceleration_raw` / `acceleration_smoothed` — central difference of raw/smoothed
  velocity, using the real `center_time` span; computed only for interior samples of a
  valid contiguous segment
- `differentiation` — method and time span, or a reason such as `segment_boundary` /
  `invalid_pair`

Invalid pairs, supplied shot boundaries and invalid gaps break segments. Smoothing and
differentiation never bridge a gap and never fill zeros.

### Shot report

When `shots` is supplied, each entry has:

- `requested` / `analyzed` ranges, `truncated`, `status`, `reason`
- `timeline` — that shot's own sample times and validation
- `pair_range` — first/last/count/valid count of its pairs
- `head_window` / `tail_window` — default 1 s clipped to the measured shot, with
  `start`, `end`, `pair_indices` (real entries in the top-level `pairs` array),
  `pair_count`, `valid_pair_count` and a `summary`
- `summary` — median velocity (vx, vy, normalized speed, p90), zoom rate, rotation
  rate, acceleration, and fit quality (inlier fraction, residual, local motion),
  sufficient for later edit-continuity matching

No pairing or differentiation crosses a supplied shot boundary. A shot with fewer than
two decoded samples has no pair and is reported as a measurement gap rather than as
measured coverage.

## Statuses

| `status` | Meaning |
|---|---|
| `complete` | decoded coverage complete, at least one valid pair, and the valid-pair interval union completely covers every selected component within its selected scope |
| `partial` | usable evidence was retained, but decoding was truncated/failed, a selected component had no valid pair or stopped before its selected end, a valid-interval gap exists, or coverage is otherwise incomplete |
| `insufficient_evidence` | measurement ran; no valid pair (flat/black, low texture, single frame, all tracks lost). Not an error and not "static" |
| `invalid_input` | bad path, interval or parameter |
| `unavailable` | OpenCV/NumPy (or ffprobe) missing |
| `failed` | probe/decode failure, corrupt media |

`insufficient_evidence` exits the CLI with 0: the measurement succeeded, but the
picture did not support a valid similarity fit. Consumers should treat it as "no
motion evidence", never as zero motion.

## Timestamps and variable frame rate

Frames are decoded by ffmpeg with the `fps` filter and scaled to the analysis width;
each output frame's PTS is read from `showinfo`. `timeline.timestamp_source` records
the mechanism, and `timeline.timestamp_validation` records the number of PTS samples,
frames decoded and the maximum deviation between observed PTS and the nominal
resampled grid (with a warning/fallback status if the PTS stream is unavailable).

For variable-frame-rate sources the analysis timeline is ffmpeg's documented
`fps=<sample_fps>` resampled timeline. The recorded `frame_times_seconds` and
`delta_t` are the actual PTS of the resampled frames mapped back onto the source
presentation timeline; values are never multiplied by the nominal source FPS. For a
CFR source, `delta_t` is exactly `1 / sample_fps`. The source's
`fps.r_frame_rate`/`avg_frame_rate` and a `variable` flag are reported.

Intervals use ffmpeg `-ss` seconds on the source presentation timeline. A non-zero
container `start_time` is reported and warned about.

## Rotation and display size

Source rotation from the display matrix (or `rotate` tag) is applied. Decoded frames
are display-oriented; `source.display_width/height` swap the coded dimensions for
90°/270° rotation, and the analysis frame preserves that displayed aspect ratio.

## Budget, bounds and coverage

- `max_seconds` caps the **cumulative** analyzed duration across shots; the frame cap
  is 4000 analysis frames (`parameters.max_frames`, `max_pairs` = 3999).
- Frames are streamed through a bounded queue; only the previous frame is retained for
  pair measurement. The whole video is never loaded into memory.
- Each ffmpeg decode runs under one monotonic wall deadline that starts before process
  creation and covers startup, frame-pipe reads and process exit. On deadline, read
  error, module frame cap or an exception, the decoder is terminated, killed if it does
  not exit, reaped, and both pipe-reader threads are joined within a bounded cleanup
  allowance. A cap reached by this module is reported separately from a decoder
  failure.
- A decoder that exits nonzero is an error even when it emitted all expected frames.
  Partial pairs are retained as `partial` evidence with a structured `decode_error` or
  `decode_timeout`; it is never reported as `complete` or fully covered.
- Coverage uses two scopes. In single-interval mode the scope is the requested interval
  clipped to the media. In shot mode the selected scope is the union of the
  caller-supplied shot intervals after the overall `start`/`end` and media intersection:
  `coverage.requested` and `coverage.clip_to_media` remain the outer request metadata,
  while unsupplied media before, between or after selected shots is outside the
  coverage scope. It is never counted as measured, decoded or missing selected
  evidence.
  - `coverage.decoded` — source spans actually sampled, with sample count, completeness
    and the unobserved tail. `start`/`end` are the outer envelope of sampled frames;
    `duration` is the sampled-span union length across selected components and
    `envelope_duration` is `end - start` (equal in single-interval mode). Decoded
    completeness requires a positive sampled span, every selected component reaching
    the last sampling quantum before its own selected end, and no decode/budget
    truncation.
  - `coverage.measured` — the union of usable valid-pair intervals, clipped to the
    selected sampled scope. `duration` is that union's length, never an envelope:
    `envelope_duration` (`end - start`) is reported separately and is larger whenever
    internal invalid pairs split the measurement. `start`/`end` are the outermost union
    bounds; `ranges` lists each covered `{start, end, duration}`. Overlapping valid
    intervals are deduplicated. No valid pair means `duration` 0 and empty `ranges`.
  - `coverage.components` reports, per component, `selected_start`/`selected_end`
    (effective selected bounds after intersection), the sampled span, pair counts,
    union `measured_duration`, `measured_envelope_duration`, `covered_ranges`, `gaps`,
    `decoded_unobserved_tail_seconds`, `measured_unobserved_tail_seconds` and
    `complete`. Completeness requires the valid union to cover the whole sampled span
    with no leading, internal or trailing hole; a valid pair at each end does not make
    an internal invalid region measured. A component whose sampling stops more than one
    quantum before its selected end, or whose decoding is truncated/failed/budget
    skipped, is incomplete.
  - `coverage.measurement_gaps` records every uncovered range with actual `start`,
    `end`, `duration`, `kind` (`leading`, `internal`, `trailing` or
    `no_valid_pairs`), `reason`, pair counts and the reasons of invalid pairs
    overlapping the gap. Inter-shot space between caller-supplied disjoint shots is
    outside every selected component; it is not counted as measured and is not
    recorded as a measurement gap.
  - `coverage.fully_covered` requires complete decoding **and** complete measured
    coverage for every selected component, with each component's unobserved tails no
    larger than one sampling quantum. A larger tail is reported as incomplete rather
    than relabelled as measured. Decoded coverage may therefore be complete while
    measured coverage is partial.
- In shot mode, `coverage.decoded.unobserved_tail_seconds` and
  `coverage.measured.unobserved_tail_seconds` are the largest per-component tail
  between that component's last sample/measurement and its selected end. Unsupplied
  space between or after shots is not included. With overlapping components, decoded
  and measured spans are deduplicated while each component still reports its own tail.
- The final sampling quantum before the scope end is not an observed frame. A 12 s
  single-interval request at 8 fps ends at the last sample 11.875 s and reports a
  `0.125` s unobserved tail; `measured.end` is not rounded up to 12 s. The same
  one-quantum convention applies independently to each selected shot end.
- If a request is clipped or decoding stops early, `coverage.requested`,
  `coverage.clip_to_media`, `coverage.analyzed`, `coverage.incomplete_reasons`,
  `coverage.truncation_reasons` and `coverage.fully_covered` report it explicitly.
  Full coverage is never claimed silently.

## Interpretation limits

The module outputs evidence only. It does not claim physical camera speed, travel
direction, focal length, depth, actor identity, object categories or calibrated
probability. Dominant-transform residuals and local-motion ratios indicate that part
of the picture moved differently from the fitted dominant transform; they do not
separate foreground from background and should be interpreted by the downstream
Agent with the rest of the evidence.

## Tests

```sh
PYTHONPATH=Core python -m unittest discover -s Tests -p 'test_motion*.py' -v
PYTHONPATH=Core python -m unittest Tests.test_asset_motion Tests.test_agent_supplements -v
```

Fixtures are real H.264 files encoded from analytically generated textured frames
(translation at two resolutions and two frame rates, centered zoom, rotation,
acceleration, two-shot direction reversal, gap, black/flat, single frame, dropout,
truncated decode, corrupt bytes and a concatenated VFR clip). Generation parameters
are in `Tests/test_motion_analyze.py`.

`Tests/test_motion_decoder_lifecycle.py` adds real-subprocess decoder-failure fixtures
(exit 1 after all/partial/zero frames, stalled output, partial-frame stall, exception
cleanup and module-cap termination), non-finite numeric-parameter regressions, and
coverage-semantics checks for a single-sample interval, a 12 s/8 fps tail, an empty
interval, a multi-shot zero-pair component, encoded texture/black/texture clips with
internal and multiple measurement holes, disjoint and overlapping supplied shots,
selected-shot scope with unsupplied leading/trailing media, `end=None` shot selection,
overall-clipped shots, invalid-gap and budget-truncated selected components, and the
interval-union primitives. The fake decoder is selected by PATH; ffprobe, LK, RANSAC
and aggregation remain the real implementation.
