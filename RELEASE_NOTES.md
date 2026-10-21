# Reelpack 0.2.0 — Alpha

**Status:** Alpha release, product 0.2.0 / Release tag v0.2.0. Assets are publicly downloadable.

## Changes in 0.2.0

- Auto OCR runs a real built-in text probe, choosing RapidOCR, then macOS Vision, then an already-prepared local Ollama vision model. A probe must return nonempty text; actual material may legitimately contain no text. Error/timeout/per-frame failure triggers bounded fallback while preserving successful frames and backend provenance.
- New `--rewrite short|film|off` defaults to off. Only an explicit `--rewrite short` or `--rewrite film` enables optional semantic editing after transcription and before summary. `--rewrite-model` selects a prepared local text model (default qwen2.5:7b); `--cloud` uses the configured cloud backend. No model is downloaded automatically.
- Exported and package Markdown retain original transcription next to the rewrite, with per-segment before/after audit. Missing models, invalid responses and timeouts keep the original and label rewriting unavailable. Model output is an editable draft, not a factual guarantee.
- Markdown now includes deduplicated on-screen text with timestamps when available, or an honest empty/unavailable state.
- The 0.1.1 empty-speech/no-track outcomes, transactional checkpoints, managed installation upgrades, and clean-source build provenance are retained.

## Compatibility

- Environment variables now use the `REELPACK_` prefix (`REELPACK_OCR_BIN`, `REELPACK_OCR_PYTHON`, `REELPACK_AUDIO_PYTHON`, `REELPACK_TRANSNET_PYTHON`, `REELPACK_CLOUD_MODEL`, `REELPACK_TESSERACT_LANG`, `REELPACK_PANNS_*`). The previous prefix is no longer read; update scripts that set it explicitly.

## Platform

- Apple Silicon / **arm64** only. This archive contains arm64 executables; Intel / x86_64 is not built or validated.
- **Minimum deployment target: macOS 14.** macOS 14 hardware validation has not yet been performed.
- Native **Python 3.12** and **FFmpeg / ffprobe** are required for the runtime. Five Python environments remain separate.

## Install

Download matching assets from the [official versioned Release](https://github.com/Lechoung-J/Reelpack-release/releases/tag/v0.2.0); older releases remain downloadable. Keep `install.sh`, both checksum files, the App ZIP and `release-manifest.json` together. Verify the checksums, save and inspect the script before running it. Full public-download and local-archive commands are in the README.

```sh
bash install.sh --archive Reelpack-0.2.0-arm64.zip \
  --manifest release-manifest.json --no-launch
```

Default destinations:

- App: `~/Applications/Reelpack.app`.
- Environments: `~/Library/Application Support/Reelpack/runtime/arm64/`.
- Skill: provided in `Reelpack.app/Contents/Resources/SKILL.md`, with a runtime copy and Agent integration instructions.
- Command: the installer prints the exact `reelpack` launcher location and PATH guidance. The native executable remains `reelpack-cli`.

The installer verifies the archive hash and contents, creates or checks native environments, and checks FFmpeg. Existing App/environment files are preserved unless an explicit replacement or repair option is supplied. Use `bash install.sh --help` for those options. It does not use sudo or silently install system FFmpeg.

## Local file engine and Agent workflow

```sh
reelpack "/path/to/video.mp4" --output-dir ~/Reelpack
reelpack query --output-dir ~/Reelpack "keyword"
```

Reelpack takes a **local video file** and produces an asset package. It does not download media, scrape webpages, resolve social/platform links, or include a video-channel parser. URL input is rejected with guidance to have the user or their Agent obtain a local file first.

The supplied [SKILL.md](SKILL.md) explains reference-film analysis, advertisement breakdown, transcription and reading the package. The installer provides instructions for Claude Code, Codex and Cursor and registers the skill only where a supported Agent directory is present and existing custom content can be preserved. It does not create an MCP server or add browsing/downloading capabilities.

`reelpack query` performs a literal read-only lookup over existing transcript, OCR and summary text. It does not run analysis, a model or a network request; detailed shot/time/evidence queries remain available through `Core/asset_cli.py`.

## Archive contents

- Reelpack Swift App and `reelpack-cli`.
- Python asset pipeline and pinned environment requirements.
- Native macOS Vision OCR helper and v3 icon.
- README files, Skill, these notes, license and installer.
- A ZIP without `__MACOSX/`, AppleDouble files or Finder metadata; matching SHA-256 file and external installation manifest.

Python environments, model weights, FFmpeg and user videos are not embedded in the App.

## Upgrade and uninstall

Use `--replace-app` to back up and replace an existing App. Unmodified installer-owned Skill/command copies can be backed up and updated; customized content is preserved with migration guidance. `--no-agent-register` still provides the runtime command path. The default installation prepares all five Python environments; optional model weights remain BYO. Read-only `--runtime-check` verifies metadata/start conditions, not model inference.

For uninstall, quit tasks, move the installed App and its selected runtime to a recovery directory, and inspect `skill-integration.json` before moving only installer-managed command/skill entries. Keep original media, output libraries, settings/history and shared model/package caches. The README includes the preservation checklist; no broad automatic data removal is performed.

## Known limitations

- **Not notarized.** This build uses local ad-hoc signing, without Developer ID distribution or Apple notarization. **Downloaded copies may be blocked by macOS Gatekeeper; clean-download Gatekeeper acceptance has not been validated.**
- **Models are separate.** large-v3 must be explicitly prepared; when not ready, the tool states the approximate ~3.1 GB cost and does not silently download. Optional PANNs checkpoints and its class-label file are BYO and have independent licenses; missing semantic weights remain `not_run` while successful physical audio analysis stays `complete`.
- **Local files only.** Supported pipeline input formats are MP4, MOV and MKV. Obtaining a video from a link belongs to the user or their Agent.
- OCR uses original-media sampling up to 1920px wide; recognition errors and missing fast-changing subtitles remain possible.
- OpenCV/PyAV duplicate AVFoundation class warnings remain observable. Existing tests and real-video runs completed without a crash, but the warning is not resolved.

## License

[FSL-1.1-MIT](LICENSE). Dependencies, system tools and model weights retain their own licenses.
