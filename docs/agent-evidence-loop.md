# Installed agent evidence loop

This document describes the installed `reelpack` entry points an Agent uses to
inspect a local video, choose a documented OCR sampling profile, request more
evidence after analysis, measure an explicit motion interval, and persist
attributed identity/role annotations. It adds no new VLM call or automatic person
recognition; motion mathematics remains the `motion_analyze` module.

## Contract summary

| Command | Purpose | Writes | Analysis/model/network |
|---|---|---|---|
| `reelpack <video> --output-dir <dir> --profile <p>` | Full analysis into an asset package | Output/library | Yes (explicit user request) |
| `reelpack preview <video> --output <new dir> [--frames N]` | Whole-video board before ASR/import | Only `<new dir>` | No |
| `reelpack frames <asset_id> ...` | Stored evidence boards / one detail frame | Rebuilt boards (or a detail frame) under the package `boards/` | No |
| `reelpack montage <asset_id> ...` | Alias of `frames` | Same as `frames` | No |
| `reelpack review <asset_id> ...` | Range re-check: frames, OCR candidates, audio segment | Only the requested new output directory | Local decode/OCR for the requested range |
| `reelpack motion <asset_id> --start S --end E ...` | Image-motion measurement for one explicit finite interval | `supplements/` only with `--save` | Local decode + OpenCV/NumPy |
| `reelpack annotate <asset_id> --input notes.json` | Persist attributed Agent voice/person/name/role assertions | One immutable supplement file | No |
| `reelpack supplements <asset_id> [--id ID]` | Read supplement history and current/stale/unknown relation | None | No |
| `reelpack query ...` | Literal lookup in saved text | None | No |
| `reelpack enrich ...` | Offline audio backfill | Package | No ASR/OCR rerun |

`frames`, `montage`, `review`, `motion`, `annotate` and `supplements` are dispatched
by the installed Python entry (`Core/agent_entry.py`) before any native CLI
forwarding. `frames`, `montage`, `review`, `annotate` and `supplements` do not import
the ASR/analysis engine (`core.py`, faster-whisper, torch); `motion` imports only
`motion_analyze` and its OpenCV/NumPy dependencies. They preserve the existing
`Core/asset_cli.py` argument contract, default library layout
(`<output-dir>/library`) and returned source/status metadata.

Evidence commands prepare runtime paths read-only. If the library or
`packages/` directory does not exist, they exit 2 with an explicit message and
do not create a library. Missing assets, missing media, invalid ranges and
occupied review output directories also exit nonzero.

## Preview

```bash
reelpack preview "/本地/视频 路径.mp4" --output "/tmp/reelpack-preview" --frames 16
```

- Uses `global_board.build_global_board`, sampling 12–24 evenly spaced frames
  (the requested count is clamped by the board builder and both the request and
  the effective grid count are reported).
- Writes `preview-board.jpg` and `preview.json` only under the explicitly
  requested output directory. The directory must not exist or must be empty.
- Refuses an output that equals the source file, contains the source, lies in
  the asset library (including `REELPACK_LIBRARY_ROOT` and
  `<REELPACK_OUTPUT_ROOT>/library`), or is an ancestor of that library.
- Works with paths containing spaces; all process launches use argument arrays,
  never shell interpolation.
- Does not classify content, call a model, download anything, choose a profile,
  import the media, or create a library.
- JSON metadata includes `status`, `source` (resolved path, display name, size),
  `board.path`, `duration_seconds`, `frames[]` (`timestamp`, `index`) and
  `grid` (`cols`, `rows`, `count`, tile/frame dimensions). Explicit flags record
  `analysis_run`, `network_used`, `model_used`, `content_classified` and
  `profile_selected` as false.
- Missing or corrupt media exits 2 and leaves no output directory. A nonempty
  output directory is never overwritten.

## Profile selection

`reelpack --profile auto|ad|talk|podcast` (default `auto`) is validated by the
Swift CLI and forwarded to `core.py`. Profiles change only OCR coarse-scan
density:

- `auto`: default sampling (cache reuse within the same pipeline revision).
- `ad`: 10 s coarse scan plus subtitle-band change trigger.
- `talk`: 30 s coarse scan plus trigger.
- `podcast`: 60 s coarse scan plus trigger.

The Agent chooses a profile after looking at the preview board and timestamps.
The engine does not infer content type from the board and does not pretend a
threshold is semantic classification. A non-default profile joins the run
signature, so an analysis cached under another profile cannot be reused; `auto`
omits the profile field. Pipeline revision 0.2.2 still invalidates 0.2.0 results.

## Motion evidence

```bash
reelpack motion <asset_id> --start 10.0 --end 12.0 \
    --sample-fps 8 --library-root "<资产库>"             # read-only
reelpack motion <asset_id> --start 10.0 --end 12.0 \
    --save --library-root "<资产库>"                     # immutable supplement
```

- `--start` and `--end` are mandatory finite numbers with `0 <= start < end`; there
  is no implicit whole-video measurement and no invented cut list.
- Saved shot intervals are read from the current `analysis.json`, clipped to the
  request and passed to `motion_analyze` as shot boundaries. Frames never pair across
  a saved cut. If no saved shot overlaps the request, the result says
  `cuts.status: "unavailable"` with an explicit reason and measures the single range
  without claiming cut guarantees.
- The adapter does not implement or alter motion mathematics. It delegates to the
  `motion_analyze.analyze_motion` API and keeps its statuses
  (`complete`, `partial`, `insufficient_evidence`, `invalid_input`, `unavailable`,
  `failed`).
- Reads the archived original by its library-relative path. Offline media is a
  structured `failed`/`media_offline` result, not a zero-motion result.
- `--save` publishes only usable measurements, and only when the current analysis
  digest and archived source digest still match. Prior immutable documents are never
  overwritten by a failed save. Persisted paths are library-relative; absolute runtime
  paths are removed.
- Read-only measurement creates no lock, cache, media directory or supplement file.

## Agent annotations

```bash
reelpack annotate <asset_id> --input annotations.json --library-root "<资产库>"
reelpack supplements <asset_id> --library-root "<资产库>"
reelpack supplements <asset_id> --id sup_an_... --library-root "<资产库>"
```

Annotations are versioned `reelpack.agent-annotations` records with three kinds:
`voice_person`, `person_identity`, `person_role`. Every assertion carries a status
(`supported`, `tentative`, `ambiguous`, `unknown`), reasoning, stated uncertainty,
optional `[0,1]` confidence and at least one evidence reference
(`package_evidence`, `media_range`, `source_text`). `source_text` is stored as
attributed text and is never opened or fetched. `speaker` is allowed only on
`voice_person`, and must already exist in the current analysis.

Each supplement records the exact source analysis SHA256. Readback reports
`current`, `stale` or `unknown`. A stale document keeps its bytes and original
references; they are not re-resolved against the newer analysis. Reanalysis
preserves supplement bytes through whole-package promotion, so history survives but
is no longer presented as current.

See `docs/agent-supplements.md` for the schema, locks, provenance and readback shape.

## Agent loop

1. `preview` the local video and actually read the board plus `preview.json`.
2. Decide whether an explicit profile is warranted, then run the analysis.
3. Read the saved package first: frame board, shot evidence, timeline, OCR,
   transcript, speakers block.
4. Name the remaining evidence gap and choose the range, timestamps, interval
   and resolution yourself.
5. Use `frames`/`montage` for stored images and `review` for a small requested
   range. Open the returned image paths and read the per-item status.
6. Use `motion` when continuity or camera/content movement is the open question.
   State the exact interval; do not extrapolate one interval to the whole video or
   read image-plane velocity as physical camera speed. Use `--save` when the
   measurement is worth keeping as engine evidence.
7. When an identity/role inference is warranted, write it with `annotate` using
   explicit evidence, reasoning and uncertainty. Re-read with `supplements`; treat
   `stale`/`unknown` as unusable for the current analysis rather than silently
   re-resolving old references.
8. Infer, or report the specific uncertainty. Do not turn a failed OCR/frame
   item into success, and do not treat sparse stills as proof of mouth/voice
   synchronisation.

There is no fixed three-frame quota, no visibility percentage requirement, and
no mandatory two-shot/close-up or static-mouth proof. Evidence density is an
Agent decision driven by the gap.

## Identity layers

Keep these distinct:

1. Anonymous voice clusters (`S1`, `S2`) from optional diarization.
2. Voice → on-screen person (Agent inference from time-aligned frames).
3. On-screen person → name or host/guest role (Agent naming, sourced).
4. The final label actually used in the answer.

The engine supplies only layer 1 when explicitly enabled with `--speakers auto`.
Layers 2–3 remain Agent-level inference. The CLI can persist those inferences
as attributed, provenance-carrying supplements, but it does not make them engine
capabilities or ground truth. Never rename `S1`/`S2` into names without
source support, and never claim the engine automatically recognises hosts or guests.
Annotations are working records, not a global person database.

## Non-goals

- Evidence lookup reuses the motion and shot modules; it includes no identity classifier.
- No new analysis-engine, model, network or download path for evidence lookup.
- No automatic mutation of the analysis, transcript, speakers block or summary from
  an annotation.
- No change to default speakers, summary, rewrite or user-facing scope.
