# Agent annotations and supplement documents

An **asset package** contains the engine's structured evidence. It does not
contain the downstream Agent's identity/role interpretation. The CLI stores these
interpretations in an immutable, versioned supplement layer beside the analysis:

```
packages/<asset_id>/
  asset.json
  analysis.json
  analysis.md
  evidence/ ...
  supplements/
    sup_an_<timestamp>_<hex>.json    # Agent annotations (interpretation)
    sup_em_<timestamp>_<hex>.json    # engine measurement (e.g. motion)
```

There is **no mutable supplement index**. Publication creates a complete file with a
same-directory temporary file and an exclusive hard link. A published document is
never overwritten. Listing validates each file, sorts it deterministically and reports
corrupt entries as warnings instead of silently repairing or resolving them.

## Commands

```bash
# Write one attributed Agent annotation record (no classifier, no model, no network)
reelpack annotate <asset_id> --input notes.json --library-root <library>

# Read every supplement plus its current/stale/unknown relation
reelpack supplements <asset_id> --library-root <library>

# Read one document verbatim
reelpack supplements <asset_id> --id sup_an_... --library-root <library>
```

`annotate` refuses a missing, non-file or symlinked input. It validates the complete
payload before creating a lock directory, so invalid input leaves the library byte
identical. Annotation input is treated as attributed Agent text; `source_text.locator`
is **never opened or fetched**.

## Versioned annotation input

```json
{
  "schema": "reelpack.agent-annotations",
  "schema_version": 1,
  "asset_id": "asset_...",
  "author": "downstream-agent-name",
  "supersedes": "sup_an_...",
  "assertions": [
    {
      "assertion_id": "a_voice",
      "kind": "voice_person",
      "status": "tentative",
      "person_id": "P1",
      "speaker": "S1",
      "value": null,
      "reasoning": "S1 overlaps the left-seated figure in frame_...",
      "uncertainty": "No lip-sync measurement; the cut changes framing.",
      "confidence": 0.6,
      "evidence_refs": [
        {"kind": "package_evidence", "evidence_id": "frame_..."},
        {"kind": "media_range", "start": 0.0, "end": 2.0}
      ]
    },
    {
      "assertion_id": "a_name",
      "kind": "person_identity",
      "status": "supported",
      "person_id": "P1",
      "value": "Attributed Name",
      "reasoning": "Name presented by the Agent-supplied source label.",
      "uncertainty": "Source attribution only; not independently verified.",
      "confidence": 0.8,
      "evidence_refs": [
        {"kind": "source_text", "source_label": "attributed title",
         "locator": "video description", "excerpt": "Attributed Name"}
      ]
    },
    {
      "assertion_id": "a_role",
      "kind": "person_role",
      "status": "unknown",
      "person_id": null,
      "value": null,
      "reasoning": "The analysed interval does not establish host or guest.",
      "uncertainty": "Role remains unknown.",
      "confidence": null,
      "evidence_refs": [
        {"kind": "source_text", "source_label": "attributed description",
         "locator": "source page", "excerpt": "conversation"}
      ]
    }
  ]
}
```

Rules enforced by the writer:

- Exact keys only; `author`, `assertions`, `reasoning` and `uncertainty` are nonempty.
- `kind` ∈ `voice_person` | `person_identity` | `person_role`.
- `status` ∈ `supported` | `tentative` | `ambiguous` | `unknown`.
- `speaker` is allowed only on `voice_person`, must match `S<number>` and must occur in
  the current analysis. An unknown label is rejected; it is never invented.
- `voice_person.value` must be `null` (the binding is expressed by `person_id` and
  `speaker`). A supported/tentative identity or role must carry a nonempty `value`;
  ambiguous/unknown may use `null`.
- `confidence` is finite in `[0,1]` or `null`; it is not calibrated probability.
- Every assertion has at least one evidence reference and a unique `assertion_id`.
- `package_evidence.evidence_id` must exist in the current analysis.
- `media_range` requires finite `0 <= start < end <= available duration`.
- `source_text` requires `source_label`, `locator`, `excerpt`; it is stored as text only.
- `supersedes` must name an existing `agent_annotations` document of the same asset.
  The earlier document stays byte-identical; superseding links history, it does not
  edit it.

`agent_annotations` documents always record `"source": "downstream_agent"` and the
caller's `author`. The engine never writes this kind.

## Provenance and relationship

Every document records portable provenance:

```json
{
  "source_content_sha256": "...",
  "source_analysis_sha256": "...",
  "analysis_present": true,
  "analysis_relative_path": "packages/asset_.../analysis.json",
  "media_relative_path": "media/originals/asset_.../clip.mp4",
  "path_base": "library_root",
  "available_video_duration_seconds": 12.5,
  "source_evidence_ids": ["frame_..."],
  "known_speaker_labels": ["S1", "S2"]
}
```

The digest is over the exact `analysis.json` bytes at save time. Readback compares it
with the current bytes:

| Relation | Meaning | Evidence handling |
|---|---|---|
| `current` | recorded digest equals current analysis | recorded references belong to the current analysis |
| `stale` | analysis changed since the document was saved | refs are returned verbatim but **not** re-resolved; warning `stale_source_analysis` |
| `unknown` | current analysis is absent | refs cannot be resolved |

A stale or unknown document is not deleted, rewritten, upgraded or partially
reinterpreted. Reanalysis preserves its bytes and its original digest so it becomes
`stale` rather than falsely `current`.

## Immutability and locks

- Writes take the same per-asset `.locks/<asset_id>.lock` non-blocking exclusive lock
  as `AssetAttempt`. A concurrent reanalysis, annotation write or measurement save
  fails explicitly with a contention message instead of waiting or interleaving.
- Publication uses a private temporary file, `fsync`, then `os.link` to the final name.
  The destination must not already exist; an existing target or symlink is rejected.
- `packages/<asset_id>/supplements` itself must be a real directory, not a symlink.
- `AssetAttempt` copies the supplement directory verbatim into its staged package.
  Whole-package promotion therefore retains the supplement history while replacing
  the analysis, evidence and boards. Hidden temporary residue (`.…tmp`) is not
  promoted. Old transcript/evidence is deliberately **not** carried forward.
- Failed attempts never modify the canonical supplement bytes.

Read-only `supplements`/`motion` commands create no library, media directory, cache,
lock file or supplement directory.

## Engine measurement documents

`reelpack motion ... --save` writes a separate `engine_measurement` document only for
usable statuses (`complete`, `partial`, `insufficient_evidence`). It records the
backend, request, parameters, full portable measurement and the same provenance. The
runtime `source.path`/`resolved_path` are removed; persisted paths use
`media_relative_path` with `path_base: "library_root"`, and absolute paths embedded in
warning text are rewritten to the same relative form. Saving is refused if the
analysis changed during measurement, the archived media is offline, or the measured
source digest no longer matches `asset.json`. A failed save leaves any earlier
document untouched.

Engine measurements and Agent annotations are never merged: `source` distinguishes
`"engine"` from `"downstream_agent"`, and `kind` distinguishes
`engine_measurement` from `agent_annotations`.

## Readback shape

`reelpack supplements` prints:

```json
{
  "schema": "reelpack.supplement-list",
  "schema_version": 1,
  "asset_id": "asset_...",
  "media_status": "available",
  "count": 1,
  "supplements": [
    {"supplement_id": "sup_an_...", "kind": "agent_annotations",
     "source": "downstream_agent", "author": "...", "created_at": "...",
     "supersedes": null, "assertion_count": 3, "assertions": [ ... ],
     "recorded_source_analysis_sha256": "...",
     "relationship": {"status": "current", "evidence_resolution": "...",
                       "recorded_source_analysis_sha256": "...",
                       "current_source_analysis_sha256": "..."}}
  ],
  "warnings": [],
  "analysis_present": true,
  "path_base": "library_root"
}
```

`--id` returns the validated document verbatim under `supplement`, plus `relationship`
and (for stale documents) a warning. Offline media does not prevent readback; it is
reported as `media_status: "offline"`.

## Boundaries

- Annotations are Agent interpretations, not engine facts. `person_id` is a
  recording-local working identifier, not a face template or global identity.
- The engine does not automatically recognise people, roles or names.
- `confidence` is supplied by the Agent and is not calibrated.
- Stale references are not silently mapped to new evidence IDs.
- This layer does not edit transcripts, summaries, speakers or any analysis field.

## Tests

```bash
PYTHONPATH=Core python -m unittest \
  Tests.test_asset_supplements \
  Tests.test_asset_attempt_supplements \
  Tests.test_agent_supplements -v
```
