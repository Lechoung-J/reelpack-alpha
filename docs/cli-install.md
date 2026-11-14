# Standalone CLI 0.2.2

Install the standalone CLI for preview, evidence review, optional speaker clustering, explicit motion queries and attributed Agent annotations. The downloadable App remains **0.2.0**. This installer prepares a separate CLI runtime and does not replace the App, rewrite libraries or download media. Apple Silicon, macOS 14+, native Python 3.12 and FFmpeg/ffprobe are required. Executables are ad-hoc signed, not notarized. Gatekeeper may require local review of downloaded executables.

## Fresh installation

Download from the explicit [v0.2.2 CLI Release](https://github.com/Lechoung-J/Reelpack-release/releases/tag/v0.2.2). Save and inspect the installer before running it. Do not pipe a download into a shell.

```bash
BASE=https://github.com/Lechoung-J/Reelpack-release/releases/download/v0.2.2
curl -fL "$BASE/install-cli.sh" -o install-cli.sh
curl -fL "$BASE/install-cli.sh.sha256" -o install-cli.sh.sha256
curl -fL "$BASE/Reelpack-CLI-0.2.2-arm64.zip" -o Reelpack-CLI-0.2.2-arm64.zip
curl -fL "$BASE/Reelpack-CLI-0.2.2-arm64.zip.sha256" -o Reelpack-CLI-0.2.2-arm64.zip.sha256
shasum -a 256 -c install-cli.sh.sha256
shasum -a 256 -c Reelpack-CLI-0.2.2-arm64.zip.sha256
# Read install-cli.sh, then run the verified local copy:
bash install-cli.sh --archive Reelpack-CLI-0.2.2-arm64.zip \
  --sha256 Reelpack-CLI-0.2.2-arm64.zip.sha256
"$HOME/.local/share/reelpack-cli/bin/reelpack" --version
"$HOME/.local/share/reelpack-cli/bin/reelpack" --runtime-check
```

Alternatively, after verifying the saved installer, `bash install-cli.sh --public-release --version v0.2.2` downloads the same explicit version and its external checksum over HTTPS. The ZIP contains its own per-file SHA256 manifest. The installer validates paths, rejects links/special files and verifies every file before activation.

Default code destination: `~/.local/share/reelpack-cli`. Each complete package lives under `versions/`. An atomic `current` symlink selects the active package. Default Python runtime: `~/Library/Application Support/Reelpack/cli-runtime/arm64`. Five environments use the existing pinned requirements for main, audio, TransNet, OCR and music. If native Python 3.12 is absent, an existing local `uv` can prepare Python 3.12.4. This installer uses no sudo or downloaded executable bootstrap. It does not install FFmpeg. Dependency packages may include their own upstream resources; no separate model weights are fetched. Whisper, CAM++ and optional PANNs weights require explicit local preparation. Model caches may be shared with the App.

Paths with spaces are supported:

```bash
bash install-cli.sh --archive Reelpack-CLI-0.2.2-arm64.zip \
  --sha256 Reelpack-CLI-0.2.2-arm64.zip.sha256 \
  --prefix "$HOME/Tools/Reelpack CLI" \
  --runtime-root "$HOME/Tools/Reelpack CLI Runtime" --no-agent-register
"$HOME/Tools/Reelpack CLI/bin/reelpack" --help
```

`--python /absolute/path/python3.12` selects a native interpreter. `--wheelhouse /absolute/wheels` uses only local wheels and disables index fallback. A managed runtime is reused after dependency checks. If preparation fails, rerun the same command to resume that installer-owned preparation. You can also select a new `--runtime-root` while preserving the failed directory for diagnosis. An arbitrary preexisting runtime is never repaired automatically.

## Upgrade from the 0.2.0 App CLI

Run the same local-archive command above. Keep the App installed. The standalone CLI uses its own runtime by default. A hash-verified managed `~/.local/bin/reelpack` and registered skills can be backed up and upgraded. Custom launchers and skills stay in place. The installer always prints an actionable absolute command; use that command when a custom `reelpack` still shadows it.

Existing DSH, Claude Code, Codex and Cursor directories are detected without reading their settings or credentials. `--no-agent-register` skips their skills and `~/.local/bin/reelpack`; the prefix's `bin/reelpack` remains available. Start a new Agent session after registration.

To reuse an already prepared runtime **read-only**, explicitly select it:

```bash
bash install-cli.sh --archive Reelpack-CLI-0.2.2-arm64.zip \
  --sha256 Reelpack-CLI-0.2.2-arm64.zip.sha256 \
  --existing-runtime "$HOME/Library/Application Support/Reelpack/runtime/arm64"
```

This mode checks native Python 3.12, required imports in all five environments and FFmpeg tools. It installs no dependencies and writes no files into that runtime. Analysis still writes its requested outputs and disposable caches outside the runtime. `--runtime-check` checks metadata/start conditions; it does not verify inference or identity accuracy.

## Rollback

Keep the printed `Active CLI` directory name for each installation. List retained entries:

```bash
ls "$HOME/.local/share/reelpack-cli/versions"
# Replace the value below with the exact previous directory basename printed by install:
PREVIOUS='0.2.2-<64-character-manifest-sha256>'
bash install-cli.sh --prefix "$HOME/.local/share/reelpack-cli" --rollback "$PREVIOUS"
```

Rollback verifies the preserved package, its ownership record and its recorded runtime before switching `current`. It retains all other versions. To return to the old App's command, use a managed launcher's printed `.backup-...` path:

```bash
# Use the exact backup filename printed during upgrade; this preserves the current CLI command too.
BACKUP="$HOME/.local/bin/reelpack.backup-<printed-id>"
mv "$HOME/.local/bin/reelpack" "$HOME/.local/bin/reelpack.cli-0.2.2"
cp "$BACKUP" "$HOME/.local/bin/reelpack"
```

The App and its runtime remain available throughout. Restore an Agent skill from its matching printed backups only after reviewing any later custom edits. Rollback changes code selection; it does not downgrade or rewrite output packages.

## Output compatibility and optional work

New analyses record `engine.version`, portable source revision when known, source-tree digest when available and pipeline revision **0.2.2**. A source checkout with no build metadata reports an unknown revision. Old successful 0.2.0 cache signatures do not count as recomputed 0.2.2 output. Reading a package never stamps a new version on it. A shared output library can contain several engine versions; their bytes and identity accuracy are not guaranteed equal.

Defaults remain `--profile auto` and `--speakers off`. `--speakers auto` needs prepared local CAM++; no download occurs. Motion requires an explicit finite range and persists only with `--save`. Identity, person and name associations remain Agent interpretations; only explicit `annotate` saves them. Reanalysis retains supplement history and reports its relationship to the current analysis. See [SKILL.md](../SKILL.md) for commands and evidence limits.

中文：新功能请安装 **0.2.2 CLI**，现有 App 保持 **0.2.0**。先保存脚本并核验两个校验文件。默认安装会准备独立环境。复用旧环境时，请显式加 `--existing-runtime`。自定义命令保留后，请使用安装器打印的绝对入口。运动测量加 `--save` 才保存。人物关联由 Agent 判断，需显式 `annotate` 才写入。回滚时，使用上方命令及安装器打印的版本目录名。

## Build from source

```bash
python3 Build/build_cli.py --output-dir /tmp/reelpack-cli-release
```

Use a clean committed checkout on native arm64 macOS. The builder compiles only `reelpack-cli` and the Vision OCR helper. It bundles Core, pinned requirements, canonical SKILL, licenses/NOTICE, version and portable source/per-file manifests. It remaps build paths, strips debug symbols and rejects executable bytes containing local build roots. It never builds the Reelpack App. `--allow-dirty` is for local testing only; its provenance explicitly marks the source as dirty. Published builds use a clean checkout and a new output directory.
