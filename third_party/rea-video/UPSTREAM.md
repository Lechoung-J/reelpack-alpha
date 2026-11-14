# rea-video upstream notice

Source: JuneYaooo/rea-video
Pinned commit: `3974260fb52476b453b5992f667e7dc1dd83c6eb`
URL: https://github.com/JuneYaooo/rea-video/blob/3974260fb52476b453b5992f667e7dc1dd83c6eb/rea_video/features.py
License: Apache License 2.0 (see `LICENSE`), NOTICE Copyright 2026 JuneYaooo (see `NOTICE`).

## What was reused

`Core/motion_analyze.py` adapts the small measured global-motion algorithm from
`rea_video/features.py`:

- `global_motion()` — corner detection with `goodFeaturesToTrack`, pyramidal
  Lucas–Kanade forward tracking, forward/backward consistency filtering, and a
  RANSAC similarity fit (`estimateAffinePartial2D`), including the compensated
  photometric residual used as local-motion evidence.
- The general parameter choices (300 corners, quality 0.01, min distance 7,
  21 px LK window, 3 pyramid levels, 1.0 px RANSAC threshold) follow the upstream
  implementation.
- `motion_summary()` inspired the idea of summarizing per-pair motion, but its
  rates are not reused (see below).

No REA production labels, classifiers, model files or dependencies were copied.

## Changes made (not a verbatim copy)

`Core/motion_analyze.py` was written for timestamped quantitative evidence, and
deliberately corrects upstream behavior:

1. **Backward tracking order.** Upstream back-tracks `nxt_pts` before checking
   `st`/`nxt_pts`; lost or non-finite tracks therefore contaminate the
   forward/backward error. The adaptation filters forward-valid, finite tracks
   *before* the backward pass and reports the distinct forward-valid and
   forward/backward-valid counts.
2. **Timestamps instead of nominal FPS.** Upstream multiplies per-pair values by
   the declared source FPS, which is wrong for variable frame rate and loses the
   per-pair interval. The adaptation records ffmpeg output PTS for every sampled
   frame (`showinfo`), derives `delta_t` from real sample timestamps, and
   differentiates with actual time spans. VFR handling is documented in
   `docs/motion-evidence.md`.
3. **Center displacement.** Upstream reports the raw affine translation (`tx`,
   `ty`), which is measured at the image origin and is contaminated by a centered
   zoom or rotation. The adaptation evaluates the transform at the frame center,
   so centered zoom/rotation has ~zero lateral velocity.
4. **Validity is explicit and never zero-filled.** A pair is either `valid` with
   a full fitted transform and sufficient inlier support, or `invalid` with a
   machine-readable reason and `null` motion fields. Flat/black frames, lost
   tracks, RANSAC failure and low inlier support cannot become "static camera"
   measurements.
5. **No camera classification.** Upstream's Ken Burns score / camera labels are
   not reused; this module emits evidence only (velocity, acceleration, zoom and
   rotation rates, fit quality, residual local motion) for downstream
   interpretation.
6. **Shot isolation.** Each supplied shot is decoded separately, so no frame pair
   ever spans a cut; smoothing and acceleration operate inside contiguous valid
   segments only.
7. **Bounded decode/coverage.** Streaming raw decode with an explicit frame cap
   and duration budget, with requested vs analyzed coverage reported.

## License compliance

The Apache-2.0 license text is retained verbatim in `LICENSE`; the upstream
copyright notice is retained in `NOTICE`. The adaptation is marked at the top of
`Core/motion_analyze.py` and in `docs/motion-evidence.md`.
