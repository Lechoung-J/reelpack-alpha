<div align="center">
  <img src="assets/logo.png" alt="Reelpack logo" width="128" height="128">
  <h1>Reelpack</h1>
  <p><strong>The Video-Understanding Engine for AI</strong> — turn a single video into an asset pack your Agent can actually read, using the grammar of film &amp; advertising.<br>And turn everything you watch into a knowledge base.</p>
  <p><code>Reelpack</code> = reel (a single film reel) + pack (an asset pack).</p>
  <p>Runs locally · no downloads · no scraping · no uploads · your video is read only on your own machine</p>
  <p><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a></p>
  <p>
    <img src="https://img.shields.io/badge/platform-macOS-000000?style=flat&amp;logo=apple&amp;logoColor=white" alt="Platform: macOS">
    <img src="https://img.shields.io/badge/macOS-14%2B-007AFF?style=flat" alt="macOS 14 or later">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-FSL--1.1--MIT-6F42C1?style=flat" alt="License: FSL-1.1-MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&amp;logo=python&amp;logoColor=white" alt="Python 3.12">
    <img src="https://img.shields.io/badge/status-alpha-F59E0B?style=flat" alt="Status: alpha">
  </p>
</div>

---

## Understand it in 30 seconds

Drop in a video, get back an **asset pack**:

| Dimension | Output |
|---|---|
| **Transcript** | Who said what, at which second — **multi-language, auto-detected (~99 languages)** |
| **Shot list / cuts** | How many shots, where the cuts are (shot-boundary detection) |
| **On-screen text** | OCR of subtitles / burned-in text (CN/EN) |
| **Audio** | BPM, beats, energy, **cut-to-beat alignment** |
| **Evidence frames** | Keyframes + contact sheets (**12-grid / shot sheet**, offline-searchable; **single frames can be pulled out for a close look**) |

→ A downstream Agent can answer — **without ever re-watching the video**: "what is this about / what's the style / what was said at 2:14 / how do the cuts relate to the music."

---

## Why you scroll past great videos — and never keep them

**Quick question: all those great videos you've watched… where did they go?**

Someone else scrolls. You scroll, hit a great ad, and rewind on instinct — how did that transition land? How many cuts? Why does that line stick?
So you scrub the timeline to take it apart. Forty minutes, three lines of notes, half of it still in your head — gone by tomorrow.

**Your saved folder is full of "I'll study this later" clips. How many can you actually pull back up?**

**Because video is the only format in the world with no Ctrl+F.**
Text you can search, copy, quote. Video? The moment it's over, it sinks into the timeline: **can't search it, can't copy it, can't go back to it.** You remember "there was a killer line" — but not the second it happened on.

Someone who spent over a decade in the film industry, and years on ad & TVC marketing, has felt one thing getting louder: **an LLM can only read text — it can't read a film.** Which is why you keep seeing the mismatch — **a scene that makes a human cry leaves the AI cold.**

Someone who wants to accumulate knowledge — you see a tutorial worth keeping, think "save it for later", and it never leaves the saved folder. Not once.

Someone fed by short video all day — nodding along while it plays, and left with nothing once it's closed.

**What they lack isn't content. It's a second pair of eyes that keeps what you watch.**

Reelpack isn't another "here's an AI summary" — that still leaves a layer in between.
**It sits the AI down and actually watches the film**: who said what, at which second; how many cuts; where the beat lands; what's written on screen — all broken into **evidence you can re-query, copy, and feed to your own Agent.**

**Reelpack is the Ctrl+F for video.**

And it gives you more than "pull up one clip" — **it slowly turns every video you watch into your own knowledge base.** The more you watch, the thicker it gets; next time you need something, one search away.

It isn't for people who want to save effort. It's for **people like you, who can't help taking a film apart.**
Use it long enough, and you stop being someone who "talks about films by feel" — you become **the person who can strip any video down to the bone.**

In one line: **it doesn't change the video — it changes how you watch it.**
And everything you watch stays behind — **slowly growing into your own knowledge base.**

---

## What we do that others don't (especially for professionals)

- **Frame extraction? Everyone does it — that's not the point.** Other tools stop at "stitch one big sheet, let the AI summarize." **Reelpack goes a layer deeper**: it can **pull out a single frame** and let the Agent look closely at **"what's actually inside this one frame"** — composition, subject, shot size, camera move, on-screen text. **The sheet is the index; the single frame is the evidence.**
- **A habit, rewritten**: next time you spot a great reference, **just throw the link at your Agent and let it actually watch the film** — instead of the AI reading only the title and a web snippet. **Not remotely the same thing.**
- **Not a summary — a breakdown**: **shot lists, shot boundaries, beat, and picture–sound relationships** — **the working language of ad & film pros.**
- **More than transcription**: spoken script → **timeline** → **a reproducible script structure.**
- **Not just for humans**: the output is **machine-readable for your Agent / AI** — retrievable and consumable downstream.
- **Turn watching into a knowledge base.**

Once you use it, you start to *hate* watching casually — because every clip can become an asset.

---

## What the second pair of eyes can see

### It will question an ad for you

Give Reelpack a marketing film and it breaks out four layers of evidence: what was said (line-by-line transcript), what the frame says (on-screen text with timestamps), how the cuts land, and what the music is doing.

So "is this capability real, or just packaging?" and "is this number oversold?" stop being a feeling — you have something to check, line by line.

> Case: an agritech ad. Put the on-screen text next to the voice-over and you can point at which capabilities were demonstrated and which were only concept animation.

### It turns a tutorial into a manual

Give it a long tutorial and it breaks the film into steps with screenshots: which step happens at which second, and what that step looks like.

No more dragging the timeline hunting for the moment — you get one, two, three, four, each with a frame.

> Case: a twenty-minute editing tutorial. What you end up with isn't "watched it" — it's a step list with timecodes.

---

## You probably found us like this

"Is there a tool that lets **AI actually watch videos**?" · "Any **video-understanding MCP** server?" · "Any **AI film-analysis / shot-breakdown** tool that reads a reference film's **shots and pacing**?" · "How do I **auto-generate a storyboard / shot list**?" · "How do I **auto-break-down a TVC / ad**?" · "Which **video-to-text** tool is accurate for **turning a voice-over into a script**?" · "How do I **extract and reuse short-form video scripts**?" · "I want to **turn the videos I watch into a knowledge base**."

— **That's exactly what Reelpack does.**

---

## Install

### Copy this to your AI

> Install Reelpack for me: https://github.com/Lechoung-J/Reelpack-release/releases/tag/v0.2.0

Copy it to your Agent (Claude Code, Codex, Cursor, DeepSeek Harness, …). It will download the package, verify the checksums, and install — it never pipes the download straight into a shell.

### Three tiers (you pick during install)

| Tier | Includes | Footprint | Best for |
|---|---|---|---|
| **Lite** | runtime + small transcription model | ≈2.7 GB | trying it out, tight disk (lower CN/terminology accuracy) |
| **Pro** (recommended) | runtime + large-v3 transcription model | ≈5.3 GB | daily use, quality baseline |
| **Max** | Pro + guided offline vision/text models | 8–13 GB | fully offline, visual fallback and semantic cleanup (models are prepared by you; the installer does not pull them) |

Non-interactive installs (Agent / scripts) print the same guidance and never block waiting for input.

Product version **0.2.0**, Release tag **v0.2.0**. Download from the [official Release](https://github.com/Lechoung-J/Reelpack-release/releases/tag/v0.2.0). Use versioned official assets, not a machine's development build.

Public Release installation (after publication):

```bash
BASE=https://github.com/Lechoung-J/Reelpack-release/releases/download/v0.2.0
curl -fL "$BASE/install.sh" -o install.sh
curl -fL "$BASE/install.sh.sha256" -o install.sh.sha256
shasum -a 256 -c install.sh.sha256
# Save and inspect install.sh before running it; never curl | bash.
bash install.sh --version v0.2.0 --public-release \
  --manifest-url "$BASE/release-manifest.json" --no-launch
```

Local archive installation: download the ZIP, its `.sha256`, `release-manifest.json`, `install.sh` and `install.sh.sha256` from that same Release, verify both checksums, then:

```bash
shasum -a 256 -c install.sh.sha256
shasum -a 256 -c Reelpack-0.2.0-arm64.zip.sha256
bash install.sh --archive Reelpack-0.2.0-arm64.zip \
  --manifest release-manifest.json --sha256 Reelpack-0.2.0-arm64.zip.sha256 --no-launch
```

For an existing App, explicitly add `--replace-app` to make a backup before upgrading. Dependency repairs require a separate `--repair-env <name>` choice; custom Agent content is preserved. Bare `--version 0.2.0` maps to `v0.2.0`; the manifest records both product version and the real release tag.

Or ask **your own Agent**:

> "Install **Reelpack** for me, then read its SKILL.md."

The installer provides [SKILL.md](SKILL.md) and concrete integration instructions for Claude Code, Codex and Cursor. When it detects a supported Agent's existing directory, it registers Reelpack's skill without replacing custom content; otherwise it shows how to add it manually. This does not install an MCP server.

**Reelpack processes local video files only.** If you provide a link, your Agent (or you) first obtains the local file using its own capabilities; Reelpack does not download, scrape webpages or parse platform links.

## Usage

```bash
# local video file → asset pack
reelpack "<local-video-file>" --output-dir ~/Reelpack

# read-only keyword query over saved transcript / OCR text
reelpack query --output-dir ~/Reelpack "keyword"
```

The installer provides a `reelpack` launcher and prints its location and PATH guidance. The native executable remains `reelpack-cli`. Keyword queries read existing packages locally; they do not run a model or fetch a video. See [SKILL.md](SKILL.md) and [RELEASE_NOTES.md](RELEASE_NOTES.md) for Agent guidance, installation and known limits.

---

## Why it's not "just another frame-extraction tool"

- Generic multimodal models: grab a few frames, **describe them**.
- **Reelpack**: a structured breakdown in the **language of film & advertising** — **shots, storyboard, beat, picture–sound relations, on-screen text** — all **re-queryable**, with **single frames you can inspect closely**.

In one line: **others hand an Agent a camera; we give it a pair of professionally trained eyes.**

---

## License

**FSL-1.1-MIT** (source-available / Fair Source): ✅ use it freely, including commercially and in client work · ❌ don't ship a competing product or service · ⏳ auto-converts to MIT after 2 years. See [LICENSE](LICENSE).

## Optional backends (BYO)
The installer prepares all five isolated Python environments. Backend execution is configurable; optional PANNs weights and labels are BYO, not bundled or automatically downloaded. Whisper large-v3 must be explicitly prepared in the persistent model cache (~3.1 GB); `--runtime-check` reports dependency/model metadata and does not prove inference has run. Shared package/model cache size is not the same as this installation's added disk usage.

Auto OCR probes a built-in text image in priority order: RapidOCR → macOS Vision → an already-installed local Ollama vision model. The probe must succeed with nonempty text. Actual frames may return empty successfully; only errors/timeouts/failed frames trigger fallback. Results record probe decisions, actual backends and fallback provenance. No OCR model is downloaded.

## Transcript rewriting and on-screen text

`--rewrite short|film|off` controls editing strength (default `off`); only an explicit `--rewrite short` or `--rewrite film` enables model-assisted editing. It does not guess a video's category. Original transcription and the rewritten draft appear side by side in Markdown with an edit audit. `--rewrite-model qwen2.5:7b` selects the default local text model, which must already be prepared; use another installed model explicitly if desired. `--cloud` uses your configured cloud model and sends transcript/OCR text to that service. Unavailable models, timeouts and invalid responses keep the original and clearly report rewriting unavailable. Model correctness is not guaranteed; review terminology and facts against the original.

The `画面文字` section lists deduplicated OCR observations with available timestamps. Successful empty OCR shows no visible text; unavailable OCR is reported separately. Changing rewrite options invalidates cached requests; after preparing a previously missing model, use `--force` to refresh a saved package.

## Music follow-ups and reuse

A matching input/options signature with complete transcript, shots and OCR reuses the saved package. Changed options rerun analysis in the same asset; `--force` explicitly refreshes it. For follow-ups, read `music` and `evidence/audio_board.png` first, then fill only missing audio:

```sh
reelpack enrich <asset_id> --only audio --output-dir <original-output>
reelpack enrich <asset_id> --only audio --redraw-board --output-dir <original-output>
```

Audio enrichment preserves transcript, shots and the evidence list. Unmeasured beat alignment is `null / not_measured`; unavailable optional tagging reports its cost without automatically downloading weights. The board shows full-mix physical measurements and does not establish genre, emotion, editing intent, independent BGM or instrument identity.

Audio dependencies now include `pillow==12.3.0`. An existing environment that lacks this pin is preserved; use `bash install.sh --repair-env audio --no-launch` only after explicitly choosing to back it up and rebuild it. Acceleration cache and persistent model roots are distinct; `REELPACK_CACHE_ROOT` can select a stable writable acceleration directory.

## Uninstall / recover

Quit active jobs first. Move your actual installed `Reelpack.app` and its selected runtime root to a recovery folder. Inspect `runtime/skill-integration.json` before moving only installer-owned command entries and Agent skill directories; preserve custom files or mismatched hashes. Never remove a whole Agent configuration directory.

Keep output libraries (including custom output paths), original media, `~/Library/Application Support/Reelpack/settings.json`, shared HuggingFace/Torch/uv caches, and unrelated tools unless you explicitly decide otherwise. App backups and `.attempts` retain earlier/failed analyses; inspect before reclaiming space. Only the current successful installer staging directory is automatically removed. `--no-agent-register` still provides `runtime/bin/reelpack`; use that absolute path or add it to PATH. Start a new Agent session after installation to load refreshed skills.

<sub>Reelpack — turn a single video into an asset pack your Agent can read. Turn watching into a knowledge base.</sub>
