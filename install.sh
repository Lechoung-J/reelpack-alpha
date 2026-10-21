#!/bin/bash
# Reelpack macOS installer. Compatible with the system Bash 3.2.
# Save this file and inspect it before running it; never pipe a download into bash.
# Release/mirror publication is a separate operation. No mirror is assumed to exist.
set -Eeuo pipefail
umask 077

PROGRAM_NAME=${0##*/}
REPOSITORY=${REELPACK_REPOSITORY:-Lechoung-J/Reelpack-release}
APP_DIR=${REELPACK_APP_DIR:-"$HOME/Applications"}
RUNTIME_ROOT=${REELPACK_RUNTIME_ROOT:-}
ARCHIVE=
SHA256_INPUT=
VERSION=
PRODUCT_VERSION="0.2.0"   # 仅用于横幅展示；发布时与仓库 VERSION 文件保持一致
ASSET_NAME=
MANIFEST_FILE=
MANIFEST_URL=
GITHUB_BASE=${REELPACK_GITHUB_BASE:-}
GH_PROXY_BASE=${REELPACK_GH_PROXY_BASE:-}
JSDELIVR_MANIFEST_URL=${REELPACK_JSDELIVR_MANIFEST_URL:-}
GITEE_MIRROR_BASE=${REELPACK_GITEE_MIRROR_BASE:-}
PYTHON_REQUEST=${REELPACK_PYTHON:-}
PIP_BACKEND=${REELPACK_PIP_BACKEND:-auto}
WHEELHOUSE=
PUBLIC_RELEASE=0
CHECK_ONLY=0
NO_LAUNCH=0
TIER=pro
QUIET=0
NON_INTERACTIVE=0
DRY_RUN=0
NO_PYTHON_DOWNLOAD=0
NO_AGENT_REGISTER=0
REPLACE_APP=0
PYTHON=
UV=
GH=
ARCH=
WORK_DIR=
LOCK_DIR=
APP_TARGET=
APP_STAGE=
CORE_DIR=
EXPECTED_SHA256=
MANIFEST_ASSET_URL=
MANIFEST_ASSET_CDN_URL=
MANIFEST_ASSET_SIZE=
MANIFEST_VERSION=
MANIFEST_RELEASE_TAG=
ALLOW_HOSTS=(github.com api.github.com raw.githubusercontent.com objects.githubusercontent.com release-assets.githubusercontent.com cdn.jsdelivr.net)
REPAIR_ENVS=()

log() { printf '[Reelpack] %s\n' "$*"; }
warn() { printf '[Reelpack] WARNING: %s\n' "$*" >&2; }
die() { local code=$1; shift; printf '[Reelpack] ERROR: %s\n' "$*" >&2; exit "$code"; }

print_banner() {
    local columns=${COLUMNS:-80}
    [ "${REELPACK_NO_BANNER:-0}" != 1 ] || return 0
    [ "$QUIET" -eq 0 ] || return 0
    case "$columns" in ''|*[!0-9]*) columns=80 ;; esac
    if [ "$columns" -lt 60 ]; then
        printf '%s\n' "Reelpack ${PRODUCT_VERSION} - the second eye for your scroll"
    else
        cat <<'REELPACK_BANNER'
  ____            _ _                 _
 |  _ \ ___  ___| | |_ __   __ _  ___| | __
 | |_) / _ \/ _ \ | | '_ \ / _` |/ __| |/ /
 |  _ <  __/  __/ | | |_) | (_| | (__|   <
 |_| \_\___|\___|_|_| .__/ \__,_|\___|_|\_\
REELPACK_BANNER
        printf '                    |_|   the second eye for your scroll · v%s\n' "$PRODUCT_VERSION"

    fi
}

print_tier_plan() {
    log "Install tier: $TIER"
    log 'Runtime environments: main, audio, transnet, ocr, music (five isolated Python environments)'
    case "$TIER" in
        lite) log 'ASR model: Whisper small (about 0.45 GB); lower Chinese/terminology accuracy than large-v3'; log 'Select it when running the CLI with --model small; CLI default remains large.' ;;
        pro) log 'ASR model: Whisper large-v3 (about 3.1 GB); recommended quality baseline' ;;
        max)
            log 'ASR model: Whisper large-v3 (about 3.1 GB); recommended quality baseline'
            log 'Max adds Ollama setup guidance only; no Ollama model is pulled or downloaded'
            ;;
    esac
}

print_ollama_guide() {
    [ "$TIER" = max ] || return 0
    log 'Ollama (optional): install and start Ollama yourself, then explicitly choose an Ollama backend/model in Reelpack.'
    log 'This installer does not install Ollama, pull a model, or download Ollama weights.'
}

print_model_notice() {
    local model=$1 command_python=$2 mode=${3:-missing}
    if [ "$model" = small ]; then
        if [ "$mode" = preview ]; then
            log 'Missing-model guidance preview: Whisper small is about 0.45 GB. It is smaller and faster, but Chinese and terminology accuracy is noticeably lower than large-v3 (about 3.1 GB).'
        else
            log 'Whisper small is not locally prepared (about 0.45 GB). It is smaller and faster, but Chinese and terminology accuracy is noticeably lower than large-v3 (about 3.1 GB).'
        fi
        log 'No model is downloaded automatically. To prepare it explicitly, run:'
        printf '  HF_HOME=%q HF_HUB_CACHE=%q %q -I -B -c %q\n' "$MODEL_HF_HOME" "$MODEL_HUB_CACHE" "$command_python" 'from faster_whisper import WhisperModel; WhisperModel("small", device="cpu", compute_type="int8")'
    else
        if [ "$mode" = preview ]; then
            log 'Missing-model guidance preview: Whisper large-v3 needs about 3.1 GB (preparation time depends on network speed).'
        else
            log 'Whisper large-v3 is not locally prepared (about 3.1 GB; preparation time depends on network speed).'
        fi
        log 'It is the recommended transcription quality baseline; small is faster and smaller, but Chinese and terminology accuracy is noticeably lower.'
        log 'No model is downloaded automatically. To prepare it explicitly, run:'
        printf '  HF_HOME=%q HF_HUB_CACHE=%q %q -I -B -c %q\n' "$MODEL_HF_HOME" "$MODEL_HUB_CACHE" "$command_python" 'from faster_whisper import WhisperModel; WhisperModel("large-v3", device="cpu", compute_type="int8")'
    fi
    log 'Without a prepared Whisper model, transcription remains not_run; visual analysis and audio measurements remain available.'
}

print_dry_run() {
    local model
    if [ "$TIER" = lite ]; then model=small; else model=large-v3; fi
    MODEL_HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
    MODEL_HUB_CACHE=${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-$MODEL_HF_HOME/hub}}
    print_tier_plan
    print_ollama_guide
    log "Dry run: would verify or prepare Whisper $model; model readiness is not checked and no download will run."
    print_model_notice "$model" 'python3' preview
    log 'No files, environments, applications, or system settings were changed.'
}

usage() {
    printf '%s\n' "Usage: /bin/bash $PROGRAM_NAME [options]

Local, verified installation (no release required):
  --archive FILE             Local .zip containing Reelpack.app
  --sha256 HEX_OR_FILE        Expected SHA256, or a checksum list with the asset name
  --manifest FILE            Matching build's release-manifest.json (fresh ZIP hash, schema_version: 1)
  --wheelhouse DIR            Install only from local wheels; no index/network fallback
  --no-python-download       Do not install a managed Python via an existing uv

Release installation:
  --version VERSION          Semver 0.2.0 maps to v0.2.0; custom tags remain unchanged
  --manifest-url HTTPS_URL   Release manifest; requires --public-release
  --public-release           Explicitly authorize unauthenticated public downloads
  --repo OWNER/NAME          Default: Lechoung-J/Reelpack-release
  --asset-name NAME.zip      Override Reelpack-<version>-<arch>.zip
  --github-base HTTPS_URL    Override the releases/download base
  --gh-proxy-base HTTPS_URL  Optional public-only proxy base (prefixes a GitHub URL)
  --jsdelivr-manifest-url U  Optional published small manifest; never a binary ZIP
  --gitee-mirror-base URL    Optional published mirror directory
  --allow-host HOST          Allow a manifest asset hostname (repeatable)

Install profile:
  --tier lite|pro|max          Default: pro; lite uses small, pro uses large-v3, max adds Ollama guidance only
  --quiet                      Suppress the ASCII banner
  --non-interactive            Never prompt; print model preparation guidance instead
  --dry-run                    Preview tier actions without changing the system

Runtime and destination:
  --app-dir DIR              Default: ~/Applications (destination: Reelpack.app)
  --runtime-root DIR         Default: ~/Library/Application Support/Reelpack/runtime/<arch>
  --python FILE              Existing native macOS Python 3.12
  --pip-backend auto|pip|uv  auto uses an existing native uv cache when available
  --repair-env NAME          Back up and rebuild main/audio/transnet/ocr/music (repeatable)
  --replace-app              Back up an existing App before installing a verified replacement
  --check                    Read-only checks of the installed App and all five environments
  --no-launch                Run runtime checks without launching the GUI
  --no-agent-register        Skip agent skills and ~/.local/bin entry; keep runtime CLI/skill
  -h, --help                 Show this help

Official v0.2.0 (save and inspect install.sh first; copy either command):
  /bin/bash ./install.sh --public-release --version v0.2.0 --manifest-url https://github.com/Lechoung-J/Reelpack-release/releases/download/v0.2.0/release-manifest.json --no-launch
  /bin/bash ./install.sh --public-release --version v0.2.0 --no-launch
The second command verifies the matching release's .zip.sha256 sidecar.

Private repositories use an authenticated gh command directly, never a credentialed
proxy. Public fallback: GitHub -> configured gh-proxy -> manifest cdn_url -> configured Gitee.
ZIP CDN fallback requires an explicit manifest cdn_url for an actually published
repository file; jsDelivr is not a default proxy for GitHub Release binaries.
Unset mirrors are skipped. This script does not publish a release or create a mirror.
Pip tries PyPI, Tsinghua, then Aliyun, with exactly one index per attempt.
Python is not downloaded by --check. Model weights/labels are never downloaded here.

Exit codes: 0 ready; 2 usage/platform/Python; 3 App/download/integrity;
            4 environment/dependency; 5 App runtime check; 6 ffmpeg/ffprobe unavailable."
}

need_value() { [ "$#" -ge 2 ] && [ -n "$2" ] || die 2 "Missing value for $1"; }
while [ "$#" -gt 0 ]; do
    case "$1" in
        --tier) need_value "$@"; TIER=$2; shift 2 ;;
        --quiet) QUIET=1; shift ;;
        --non-interactive) NON_INTERACTIVE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --app-dir) need_value "$@"; APP_DIR=$2; shift 2 ;;
        --runtime-root) need_value "$@"; RUNTIME_ROOT=$2; shift 2 ;;
        --archive) need_value "$@"; ARCHIVE=$2; shift 2 ;;
        --sha256) need_value "$@"; SHA256_INPUT=$2; shift 2 ;;
        --version) need_value "$@"; VERSION=$2; shift 2 ;;
        --asset-name) need_value "$@"; ASSET_NAME=$2; shift 2 ;;
        --manifest) need_value "$@"; MANIFEST_FILE=$2; shift 2 ;;
        --manifest-url) need_value "$@"; MANIFEST_URL=$2; shift 2 ;;
        --repo) need_value "$@"; REPOSITORY=$2; shift 2 ;;
        --github-base) need_value "$@"; GITHUB_BASE=$2; shift 2 ;;
        --gh-proxy-base) need_value "$@"; GH_PROXY_BASE=$2; shift 2 ;;
        --jsdelivr-manifest-url) need_value "$@"; JSDELIVR_MANIFEST_URL=$2; shift 2 ;;
        --gitee-mirror-base) need_value "$@"; GITEE_MIRROR_BASE=$2; shift 2 ;;
        --allow-host) need_value "$@"; ALLOW_HOSTS+=("$2"); shift 2 ;;
        --python) need_value "$@"; PYTHON_REQUEST=$2; shift 2 ;;
        --pip-backend) need_value "$@"; PIP_BACKEND=$2; shift 2 ;;
        --wheelhouse) need_value "$@"; WHEELHOUSE=$2; shift 2 ;;
        --repair-env) need_value "$@"; REPAIR_ENVS+=("$2"); shift 2 ;;
        --public-release) PUBLIC_RELEASE=1; shift ;;
        --replace-app) REPLACE_APP=1; shift ;;
        --no-python-download) NO_PYTHON_DOWNLOAD=1; shift ;;
        --check) CHECK_ONLY=1; NO_LAUNCH=1; NO_PYTHON_DOWNLOAD=1; shift ;;
        --no-launch) NO_LAUNCH=1; shift ;;
        --no-agent-register) NO_AGENT_REGISTER=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die 2 "Unknown option: $1 (use --help)" ;;
    esac
done
case "$TIER" in lite|pro|max) ;; *) die 2 "--tier must be lite, pro or max" ;; esac
print_banner
if [ "$DRY_RUN" -eq 1 ]; then
    print_dry_run
    exit 0
fi
case "$PIP_BACKEND" in auto|pip|uv) ;; *) die 2 "--pip-backend must be auto, pip or uv" ;; esac
[[ "$REPOSITORY" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die 2 "Invalid repository name"
if [ "$CHECK_ONLY" -eq 1 ]; then
    [ -z "$ARCHIVE$SHA256_INPUT$MANIFEST_FILE$MANIFEST_URL$VERSION$WHEELHOUSE" ] &&
        [ "$REPLACE_APP" -eq 0 ] && [ "${#REPAIR_ENVS[@]}" -eq 0 ] ||
        die 2 "--check examines an existing installation; do not combine it with installation inputs"
fi
[ -z "$MANIFEST_FILE" ] || [ -z "$MANIFEST_URL" ] || die 2 "Choose --manifest or --manifest-url"
# Pure argument checks precede Python discovery/download and all runtime writes.
if [ -n "$VERSION" ]; then
    [[ "$VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]{0,79}$ ]] || die 2 "Invalid version/tag"
    if [[ "$VERSION" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[A-Za-z0-9.-]+)?(\+[A-Za-z0-9.-]+)?$ ]]; then VERSION="v$VERSION"; fi
fi
[ -z "$ASSET_NAME" ] || [[ "$ASSET_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,159}\.zip$ ]] || die 2 "Invalid ZIP asset filename"
if [ -n "$MANIFEST_URL" ]; then
    [ "$PUBLIC_RELEASE" -eq 1 ] || die 2 "--manifest-url requires --public-release; private metadata must use a local --manifest"
    [[ "${MANIFEST_URL##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*\.json$ ]] || die 2 "Manifest URL needs a plain .json basename"
fi
for configured_url in "$MANIFEST_URL" "$GITHUB_BASE" "$GH_PROXY_BASE" "$JSDELIVR_MANIFEST_URL" "$GITEE_MIRROR_BASE"; do
    [ -z "$configured_url" ] || [[ "$configured_url" =~ ^https://[A-Za-z0-9.-]+(:443)?(/[^[:space:]\?\#\\]*)?$ ]] || die 2 "Invalid configured HTTPS download location"
done
[ -z "$ARCHIVE" ] || { [ -f "$ARCHIVE" ] && [ ! -L "$ARCHIVE" ]; } || die 2 "Local archive must be a regular, non-symlink file"
[ -z "$ARCHIVE" ] || [ -n "$SHA256_INPUT$MANIFEST_FILE$MANIFEST_URL" ] || die 2 "Local archives require --sha256 or --manifest"
[ -z "$MANIFEST_FILE" ] || [ -f "$MANIFEST_FILE" ] || die 2 "Manifest file does not exist"
[ -z "$WHEELHOUSE" ] || [ -d "$WHEELHOUSE" ] || die 2 "Wheelhouse directory does not exist"
[ -z "$SHA256_INPUT" ] || [[ "$SHA256_INPUT" =~ ^[a-fA-F0-9]{64}$ ]] || [ -f "$SHA256_INPUT" ] || die 2 "Invalid --sha256 value or missing checksum file"
for host in "${ALLOW_HOSTS[@]}"; do
    [[ "$host" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ ]] || die 2 "Invalid allowed hostname"
done
for name in ${REPAIR_ENVS[@]+"${REPAIR_ENVS[@]}"}; do
    case "$name" in main|audio|transnet|ocr|music) ;; *) die 2 "Unknown environment to repair: $name" ;; esac
done

[ "$(/usr/bin/uname -s)" = Darwin ] || die 2 "This installer requires macOS 14 or later"
OS_VERSION=$(/usr/bin/sw_vers -productVersion)
[[ "$OS_VERSION" =~ ^([0-9]+)\. ]] || die 2 "Cannot determine macOS version"
[ "${BASH_REMATCH[1]}" -ge 14 ] || die 2 "macOS $OS_VERSION is unsupported; macOS 14+ is required"
UNAME_ARCH=$(/usr/bin/uname -m)
TRANSLATED=$(/usr/sbin/sysctl -n sysctl.proc_translated 2>/dev/null || printf '0')
ARM_CAPABLE=$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || printf '0')
case "$UNAME_ARCH" in arm64) ARCH=arm64 ;; x86_64) ARCH=x86_64 ;; *) die 2 "Unsupported uname architecture: $UNAME_ARCH" ;; esac
[ "$ARM_CAPABLE" != 1 ] || ARCH=arm64
log "macOS=$OS_VERSION hardware=$ARCH uname=$UNAME_ARCH translated=$TRANSLATED"
[ "$TRANSLATED" != 1 ] || die 2 "Rosetta shell detected. Re-run from a native Terminal: /usr/bin/arch -arm64 /bin/bash <saved-install.sh> <options>"
[ "${EUID:-$(id -u)}" -ne 0 ] || die 2 "Run as your regular user; sudo is not required"
if [ "$ARCH" = arm64 ]; then
    NATIVE_BREW=/opt/homebrew
else
    NATIVE_BREW=/usr/local
fi
if [ -z "$RUNTIME_ROOT" ]; then
    RUNTIME_ROOT="$HOME/Library/Application Support/Reelpack/runtime/$ARCH"
fi
export PATH="$RUNTIME_ROOT/bin:$NATIVE_BREW/bin:$NATIVE_BREW/sbin:$PATH:/usr/bin:/bin:/usr/sbin:/sbin"

native_python() {
    [ -x "$1" ] || return 1
    "$1" -I -B -c 'import platform,struct,sys; sys.exit(0 if sys.platform == "darwin" and sys.version_info[:2] == (3,12) and platform.machine() == sys.argv[1] and struct.calcsize("P") == 8 else 1)' "$ARCH" >/dev/null 2>&1
}
native_binary() {
    local info
    [ -f "$1" ] && [ -x "$1" ] || return 1
    info=$(/usr/bin/file -L -b "$1" 2>/dev/null) || return 1
    case "$info" in *Mach-O*"$ARCH"*) return 0 ;; *) return 1 ;; esac
}
UV=$(command -v uv 2>/dev/null || true)
if [ -n "$UV" ] && ! native_binary "$UV"; then
    warn "Ignoring uv that is not a native $ARCH Mach-O executable"
    UV=
fi
if [ -n "$PYTHON_REQUEST" ]; then
    native_python "$PYTHON_REQUEST" || die 2 "--python must be a native $ARCH macOS Python 3.12 executable"
    PYTHON=$PYTHON_REQUEST
else
    for candidate in "$(command -v python3.12 2>/dev/null || true)" \
        "$NATIVE_BREW/opt/python@3.12/bin/python3.12" \
        "$RUNTIME_ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && native_python "$candidate"; then PYTHON=$candidate; break; fi
    done
    if [ -z "$PYTHON" ] && [ -n "$UV" ] && [ "$CHECK_ONLY" -eq 0 ]; then
        candidate=$(UV_NO_CONFIG=1 "$UV" --no-config python find --offline 3.12 2>/dev/null || true)
        if [ -n "$candidate" ] && native_python "$candidate"; then PYTHON=$candidate; fi
        if [ -z "$PYTHON" ] && [ "$NO_PYTHON_DOWNLOAD" -eq 0 ]; then
            log "Using your existing native uv to install managed Python 3.12.4"
            UV_NO_CONFIG=1 "$UV" --no-config python install 3.12.4 || die 2 "uv Python installation failed; install native Python 3.12 and pass --python"
            candidate=$(UV_NO_CONFIG=1 "$UV" --no-config python find --offline 3.12.4 2>/dev/null || true)
            if [ -n "$candidate" ] && native_python "$candidate"; then PYTHON=$candidate; fi
        fi
    fi
fi
[ -n "$PYTHON" ] || die 2 "Native Python 3.12 not found. With an already installed native uv, run: uv python install 3.12.4; then rerun. No remote executable installer is fetched."
# Keep the bootstrap executable outside a venv that may be explicitly backed up.
PYTHON_BASE=$("$PYTHON" -I -B -c 'import os,sys; print(os.path.realpath(getattr(sys,"_base_executable",sys.executable)))') || die 2 "Cannot resolve native Python base executable"
native_python "$PYTHON_BASE" || die 2 "Native Python base executable is unavailable"
PYTHON=$PYTHON_BASE
log "Native Python: $PYTHON"
[ "$PIP_BACKEND" != uv ] || [ -n "$UV" ] || die 2 "--pip-backend uv requires an existing native uv"
if [ "$PIP_BACKEND" = auto ]; then
    if [ -n "$UV" ]; then PIP_BACKEND=uv; else PIP_BACKEND=pip; fi
fi

# All filesystem/ZIP/JSON operations use the selected native Python, not shell eval.
# A manifest can only supply data; it cannot select commands or dependency files.
py_task() {
    "$PYTHON" -I -B - "$@" <<'PY'
import csv
import hashlib
import hmac
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import sys
import unicodedata
from urllib.parse import urlsplit
import zipfile

command, *args = sys.argv[1:]
def fail(message):
    raise ValueError(message)
def plain(value):
    if not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        fail("empty value or control characters")
    return value
def path(value):
    return Path(os.path.abspath(os.path.expanduser(plain(value))))
def canonical(value):
    return Path(os.path.realpath(path(value)))
def version(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,79}", value):
        fail("invalid version/tag")
    return value

def release_tag(value):
    value = version(value)
    return "v" + value if re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[A-Za-z0-9.-]+)?(?:\+[A-Za-z0-9.-]+)?", value) else value
def filename(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\.zip", value):
        fail("invalid ZIP asset filename")
    return value
def url(value, hosts=None):
    plain(value)
    u = urlsplit(value)
    if u.scheme != "https" or not u.hostname or u.username or u.password or u.fragment or u.query:
        fail("URLs must use HTTPS without credentials, queries or fragments")
    if u.port not in (None, 443) or "\\" in value:
        fail("invalid HTTPS port or path")
    hostname = u.hostname.lower()
    if hosts is not None and hostname not in [h.lower() for h in hosts]:
        fail("asset hostname is not allowed; configure --allow-host explicitly")
    return hostname

def skill_safe_path(value):
    p = Path(plain(str(value)))
    if not p.is_absolute() or p != Path(os.path.abspath(p)):
        fail("skill/launcher path must be a normalized absolute path")
    for component in (*reversed(p.parents), p):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode): fail(f"symlink preserved: {component}")
        if component != p and not stat.S_ISDIR(mode): fail(f"parent is not a directory: {component}")
    return p

def skill_file_status(destination, content):
    p = skill_safe_path(destination)
    if not os.path.lexists(p): return "missing"
    if not p.is_file(): fail(f"non-file preserved: {p}")
    if p.stat().st_size != len(content): return "conflict"
    return "reused" if p.read_bytes() == content else "conflict"

def skill_digest(p):
    p = skill_safe_path(p)
    if not p.is_file() or p.stat().st_nlink != 1 or p.stat().st_size > 2 * 1024 * 1024:
        fail(f"not a small single-link managed file: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()

def skill_record(p):
    p = skill_safe_path(p)
    skill_digest(p)
    record = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("managed_by") != "reelpack-install" or record.get("schema_version") != 1:
        fail("unmanaged integration report preserved")
    return record

def skill_upgrade(directory, files, states, owned, check_only):
    # A marker is never ownership proof: every existing member must match its old hash.
    for name in files:
        p = directory / name
        if states[name] == "missing" or owned.get(str(p)) != skill_digest(p):
            return {"status": "conflict", "reason": "existing content differs from verified ownership; use manual integration", "files": []}
    if check_only:
        return {"status": "update_available", "reason": "verified managed files can be backed up and upgraded", "files": []}
    backup = Path(tempfile.mkdtemp(prefix=directory.name + ".backup-", dir=directory.parent))
    staged = Path(tempfile.mkdtemp(prefix=".reelpack-update-", dir=directory.parent))
    changed = []
    try:
        for name, (content, mode) in files.items():
            source = skill_safe_path(directory / name)
            if owned[str(source)] != skill_digest(source): fail("managed file changed during upgrade")
            shutil.copy2(source, backup / name, follow_symlinks=False)
            with (staged / name).open("xb") as out: out.write(content)
            (staged / name).chmod(mode)
        for name in files:
            destination = skill_safe_path(directory / name)
            if owned[str(destination)] != skill_digest(destination): fail("managed file changed during upgrade")
            replacement = skill_safe_path(staged / name)
            os.replace(replacement, destination)
            changed.append(name)
    except BaseException:
        for name in reversed(changed):
            source, destination = skill_safe_path(backup / name), skill_safe_path(directory / name)
            os.replace(source, destination)
        raise
    finally:
        # Only this transaction's files/empty staging directory; never glob cleanup.
        for name in files:
            p = skill_safe_path(staged / name)
            if p.is_file(): p.unlink()
        skill_safe_path(staged).rmdir()
    return {"status": "upgraded", "backup": str(backup), "files": [
        {"path": str(directory / name), "sha256": hashlib.sha256(content).hexdigest(), "status": "upgraded", "previous_sha256": owned[str(directory / name)], "backup": str(backup / name)}
        for name, (content, _) in files.items()]}

def skill_materialize(directory, files, check_only=False, require_empty=False, owned=None):
    directory = skill_safe_path(directory)
    if directory.exists() and not directory.is_dir(): fail(f"non-directory preserved: {directory}")
    states = {name: skill_file_status(directory / name, content) for name, (content, _) in files.items()}
    if "conflict" in states.values():
        if owned:
            return skill_upgrade(directory, files, states, owned, check_only)
        return {"status": "conflict", "reason": "existing content differs; use manual integration", "files": []}
    for name, (_, mode) in files.items():
        if mode & 0o111 and states[name] == "reused" and not os.access(directory / name, os.X_OK):
            return {"status": "conflict", "reason": "existing command entry is not executable; set permissions manually", "files": []}
    if require_empty and states.get("SKILL.md") == "missing" and directory.exists() and any(directory.iterdir()):
        return {"status": "conflict", "reason": "occupied skill directory preserved; use manual integration", "files": []}
    records = []
    if not check_only:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        skill_safe_path(directory)
    for name, (content, mode) in files.items():
        destination = directory / name
        state = states[name]
        if state == "missing" and not check_only:
            skill_safe_path(destination)
            # Exclusive creation also refuses a file/symlink appearing after the checks.
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
            with os.fdopen(fd, "wb") as out:
                out.write(content)
                os.fchmod(out.fileno(), mode)
            state = "installed"
        records.append({"path": str(destination), "sha256": hashlib.sha256(content).hexdigest(), "status": state})
    status = "missing" if check_only and "missing" in states.values() else "ready" if check_only else "installed" if "missing" in states.values() else "reused"
    return {"status": status, "files": records}

def integrate_skills(app, runtime, register=True, check_only=False, native_brew="/opt/homebrew"):
    app, runtime, home = skill_safe_path(app), skill_safe_path(runtime), Path.home()
    if runtime in (Path("/"), home): fail("unsafe runtime root")
    source = skill_safe_path(app / "Contents/Resources/SKILL.md")
    if not source.is_file() or not 0 < source.stat().st_size <= 2 * 1024 * 1024:
        fail("canonical App SKILL.md is missing, empty or oversized")
    canonical = source.read_bytes()
    canonical.decode("utf-8")
    entry = runtime / "bin/reelpack"
    adapter = skill_safe_path(app / "Contents/Resources/Core/agent_entry.py")
    if not adapter.is_file(): fail("App agent_entry.py is missing")
    frontmatter = ("---\nname: reelpack\ndescription: Analyze local videos with Reelpack into transcripts, shots, OCR, audio timing, and evidence frames; use the asset pack to answer questions.\n---\n\n")
    direct_command = (f"env REELPACK_RUNTIME_ROOT={shlex.quote(str(runtime))} "
        f"{shlex.quote(str(runtime / '.venv/bin/python'))} -B {shlex.quote(str(adapter))}")
    guidance = f"## Installed command\n\nCall this absolute entry if `reelpack` is not on PATH:\n\n```bash\n{shlex.quote(str(entry))} \"<local video file>\" --output-dir ~/Reelpack\n```\n\nIf the installer preserved an existing custom entry, use this direct adapter instead:\n\n```bash\n{direct_command} \"<local video file>\" --output-dir ~/Reelpack\n```\n\n"
    guidance += "If the entry and adapter are both missing, this is a leftover skill from a removed installation. Restore the verified official App/runtime with the saved installer and matching release manifest or SHA256; use --replace-app only when replacing an existing App, then start a new agent session. Do not substitute a development build for a verified release.\n\n"
    native = (frontmatter + guidance).encode("utf-8") + canonical
    metadata = {"schema_version": 1, "managed_by": "reelpack-install", "name": "reelpack", "canonical_sha256": hashlib.sha256(canonical).hexdigest(), "skill_sha256": hashlib.sha256(native).hexdigest()}
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    launcher = ("#!/bin/sh\n# Generated by the Reelpack installer.\n"
        f"export REELPACK_RUNTIME_ROOT={shlex.quote(str(runtime))}\n"
        f"export PATH={shlex.quote(str(runtime / 'bin') + ':' + native_brew + '/bin:' + native_brew + '/sbin')}:\"${{PATH:-/usr/bin:/bin:/usr/sbin:/sbin}}\"\n"
        'reelpack_home="${HOME:-}"\n'
        f'[ -n "$reelpack_home" ] || reelpack_home={shlex.quote(str(home))}\n'
        'export REELPACK_CACHE_ROOT="${REELPACK_CACHE_ROOT:-$reelpack_home/Library/Caches/Reelpack}"\n'
        'export NUMBA_CACHE_DIR="$REELPACK_CACHE_ROOT/numba"\n'
        'export MPLCONFIGDIR="$REELPACK_CACHE_ROOT/mpl"\n'
        'export PYTHONPYCACHEPREFIX="$REELPACK_CACHE_ROOT/pycache"\n'
        'export PYTHONDONTWRITEBYTECODE=1\n'
        'reelpack_model_cache="${XDG_CACHE_HOME:-$reelpack_home/.cache}"\n'
        'export HF_HOME="${HF_HOME:-$reelpack_model_cache/huggingface}"\n'
        'export TORCH_HOME="${TORCH_HOME:-$reelpack_model_cache/torch}"\n'
        # agent_entry prepares the real cache before importing Core or workers.
        f"exec {shlex.quote(str(runtime / '.venv/bin/python'))} -B {shlex.quote(str(adapter))} \"$@\"\n").encode("utf-8")
    report = {"schema_version": 1, "managed_by": "reelpack-install", "app_skill": str(source), "canonical_sha256": metadata["canonical_sha256"], "skill_sha256": metadata["skill_sha256"], "launchers": [], "agents": [], "files": []}
    report_path = runtime / "skill-integration.json"
    previous = {}
    try:
        if os.path.lexists(report_path): previous = skill_record(report_path)
    except (OSError, ValueError):
        pass  # Unknown reports never authorize overwriting a canonical skill or launcher.
    owned = {item.get("path"): item.get("sha256") for item in previous.get("files", []) if isinstance(item, dict)} if isinstance(previous.get("files", []), list) else {}
    report["upgrades"] = previous.get("upgrades", []) if isinstance(previous.get("upgrades", []), list) else []
    def materialize(directory, files, require_empty=False):
        try:
            ownership = owned
            if require_empty:
                ownership = {}
                metadata_path = directory / ".metadata.json"
                if os.path.lexists(metadata_path):
                    try:
                        old = skill_record(metadata_path)
                        ownership = {str(directory / "SKILL.md"): old.get("skill_sha256"), str(metadata_path): skill_digest(metadata_path)}
                    except (OSError, ValueError):
                        pass
            if not require_empty and "reelpack" in files and str(directory / "reelpack") not in ownership:
                # Runtime relocation: parse the old launcher assignment as data, never source it.
                destination = directory / "reelpack"
                if os.path.lexists(destination):
                    try:
                        skill_digest(destination)
                        assignments = [line for line in destination.read_text(encoding="utf-8").splitlines() if line.startswith("export REELPACK_RUNTIME_ROOT=")]
                        if len(assignments) == 1:
                            words = shlex.split(assignments[0])
                            if len(words) == 2:
                                old_runtime = skill_safe_path(words[1].split("=", 1)[1])
                                old_report = skill_record(old_runtime / "skill-integration.json")
                                for item in old_report.get("files", []):
                                    if isinstance(item, dict) and item.get("path") == str(destination):
                                        ownership = {**ownership, str(destination): item.get("sha256")}
                    except (OSError, ValueError, TypeError):
                        pass
            outcome = skill_materialize(directory, files, check_only, require_empty, ownership)
        except (OSError, ValueError) as exc:
            outcome = {"status": "unsafe", "reason": str(exc), "files": []}
        report["files"].extend(outcome["files"])
        if outcome["status"] == "upgraded":
            report["upgrades"].append({"directory": str(directory), "backup": outcome["backup"], "files": outcome["files"]})
            print(f"[Reelpack] Managed integration upgraded; backup: {outcome['backup']}")
        return {key: value for key, value in outcome.items() if key != "files"}
    canonical_path = runtime / "share/reelpack/SKILL.md"
    report["canonical_skill"] = {"path": str(canonical_path), **materialize(canonical_path.parent, {"SKILL.md": (canonical, 0o600)})}
    print(f"[Reelpack] Canonical skill: {canonical_path} ({report['canonical_skill']['status']}); App source: {source}")
    if report["canonical_skill"].get("reason"): print(f"[Reelpack] WARNING: {report['canonical_skill']['reason']}")
    for destination, enabled in ((entry, True), (home / ".local/bin/reelpack", register)):
        outcome = materialize(destination.parent, {"reelpack": (launcher, 0o700)}) if enabled else {"status": "disabled"}
        report["launchers"].append({"path": str(destination), **outcome})
        print(f"[Reelpack] Command entry: {destination} ({outcome['status']})")
        if outcome.get("reason"): print(f"[Reelpack] WARNING: {outcome['reason']}")
    ready_entries = [item for item in report["launchers"] if item["status"] in ("installed", "upgraded", "reused", "ready")]
    command_entry = shlex.quote(ready_entries[0]["path"]) if ready_entries else direct_command
    print(f"[Reelpack] Command to use: {command_entry}")
    if ready_entries:
        print(f"[Reelpack] To use reelpack by name, add its command directory to PATH:")
        print(f"  export PATH={shlex.quote(str(Path(ready_entries[0]['path']).parent))}:\"$PATH\"")
    for agent, detectors, relative in (
        ("Claude Code", (".claude",), ".claude/skills/reelpack"),
        ("Codex", (".agents", ".codex"), ".agents/skills/reelpack"),
        ("Cursor", (".cursor",), ".cursor/skills/reelpack"),
    ):
        destination = home / relative
        # Existence only: no settings, credentials or agent configuration files are read.
        detected = [home / name for name in detectors if (home / name).is_dir()]
        outcome = {"status": "disabled" if not register else "not_detected"}
        if register and detected:
            try:
                for config_directory in detected: skill_safe_path(config_directory)
                outcome = materialize(destination, {"SKILL.md": (native, 0o600), ".metadata.json": (metadata_bytes, 0o600)}, require_empty=True)
            except (OSError, ValueError) as exc:
                outcome = {"status": "unsafe", "reason": str(exc)}
        report["agents"].append({"agent": agent, "detected": bool(detected), "path": str(destination / "SKILL.md"), **outcome})
        print(f"[Reelpack] {agent}: {outcome['status']} at {destination / 'SKILL.md'}")
        if outcome.get("reason"): print(f"[Reelpack] WARNING: {outcome['reason']}")
        if outcome["status"] in ("installed", "upgraded", "reused", "ready"):
            print(f"[Reelpack] {agent}: start a new session and ask it to use the reelpack skill.")
        else:
            print(f"[Reelpack] {agent}: manual integration: ask the agent to read {source}, then use: {command_entry}. To enable native discovery, create its config directory and rerun without --no-agent-register; existing custom content must be moved manually first.")
    # Keep old ownership evidence for disabled/conflicting entries; hashes never authorize a changed file.
    recorded = {item["path"] for item in report["files"]}
    for item in previous.get("files", []) if isinstance(previous.get("files", []), list) else []:
        if isinstance(item, dict) and item.get("path") not in recorded:
            report["files"].append(item)
    report_path = runtime / "skill-integration.json"
    if check_only:
        print(f"[Reelpack] Read-only integration check; no skills, command entries or report written ({report_path}).")
    else:
        try:
            skill_safe_path(report_path)
            content = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            if os.path.lexists(report_path):
                if not report_path.is_file() or report_path.stat().st_size > 1024 * 1024 or report_path.stat().st_nlink != 1:
                    fail("integration report target must be a regular small file with one link")
                previous = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(previous, dict) or previous.get("managed_by") != "reelpack-install" or previous.get("schema_version") != 1:
                    fail("unmanaged integration report preserved")
                # Record verified upgrades and retained rollback backups.
                fd = os.open(report_path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW)
            else:
                fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as out: out.write(content)
            print(f"[Reelpack] Integration result: {report_path}")
        except (OSError, ValueError) as exc:
            print(f"[Reelpack] WARNING: integration report preserved/not written: {exc}")
    return report

try:
    if command == "canonical":
        original = path(args[0])
        if original.is_symlink(): fail("directory itself must not be a symlink")
        resolved = canonical(args[0])
        forbidden = {Path("/"), Path("/System"), Path("/usr"), Path("/Library"), Path.home().resolve()}
        if resolved in forbidden: fail("unsafe destination directory")
        if resolved.exists() and not resolved.is_dir(): fail("destination is not a directory")
        print(resolved)
    elif command == "directory":
        p = canonical(args[0])
        if not p.is_dir(): fail("directory does not exist")
        print(p)
    elif command == "host":
        print(url(args[0]))
    elif command == "url":
        url(args[0], args[1:])
    elif command == "names":
        if args[0]: version(args[0])
        if args[1]: filename(args[1])
        for host in args[2:]:
            if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", host): fail("invalid allowed hostname")
    elif command == "copy-archive":
        source, destination = path(args[0]), path(args[1])
        if source.is_symlink() or not source.is_file(): fail("local archive must be a regular, non-symlink file")
        with source.open("rb") as inp, destination.open("xb") as out:
            shutil.copyfileobj(inp, out, 1024 * 1024)
    elif command == "checksum":
        value, asset_name = args
        if re.fullmatch(r"[a-fA-F0-9]{64}", value):
            print(value.lower())
        else:
            p = path(value)
            if not p.is_file() or p.stat().st_size > 2 * 1024 * 1024: fail("invalid checksum list")
            matches = []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"): continue
                if re.fullmatch(r"[a-fA-F0-9]{64}", line):
                    matches.append(line.lower()); continue
                m = re.fullmatch(r"([a-fA-F0-9]{64})\s+\*?(.+)", line)
                if m and m[2] == asset_name: matches.append(m[1].lower()); continue
                m = re.fullmatch(r"SHA256 \((.+)\) = ([a-fA-F0-9]{64})", line)
                if m and m[1] == asset_name: matches.append(m[2].lower())
            if len(matches) != 1: fail("checksum list needs exactly one matching asset or bare SHA256")
            print(matches[0])
    elif command == "manifest":
        p, arch, repo, requested = path(args[0]), args[1], args[2], args[3]
        if not p.is_file() or p.stat().st_size > 1024 * 1024: fail("manifest exceeds 1 MiB or is missing")
        data = json.loads(p.read_text(encoding="utf-8"))
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            fail("unsupported manifest schema_version (expected integer 1)")
        ver = version(data["version"])
        if re.match(r"^v[0-9]+\.[0-9]+\.[0-9]+", ver): fail("manifest version must omit the v prefix")
        tag = version(data["release_tag"]) if "release_tag" in data else release_tag(ver)
        if data.get("repository", repo) != repo: fail("manifest repository mismatch")
        if requested and release_tag(requested) != tag: fail("manifest release_tag/version mismatch")
        if re.fullmatch(r"v?[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?(?:\+[A-Za-z0-9.-]+)?", tag) and tag.removeprefix("v") != ver:
            fail("manifest version and release_tag mismatch")
        asset = data["assets"][arch]
        name = filename(asset["filename"])
        digest = asset["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest): fail("invalid manifest SHA256")
        asset_url = asset.get("url", "")
        if asset_url:
            url(asset_url)
            location = urlsplit(asset_url)
            if location.path.rsplit("/", 1)[-1] != name: fail("manifest URL filename mismatch")
            if "/releases/download/" in location.path:
                expected_path = f"/{repo}/releases/download/{tag}/{name}"
                if location.path != expected_path: fail("manifest URL repository/tag/filename mismatch")
        asset_cdn_url = asset.get("cdn_url", "")
        if asset_cdn_url: url(asset_cdn_url)
        size = asset.get("size", "")
        if size != "" and (type(size) is not int or not 0 < size <= 4 * 1024**3): fail("invalid manifest asset size")
        for key, value in (("version", ver), ("release_tag", tag), ("filename", name), ("sha256", digest.lower()), ("url", asset_url), ("cdn_url", asset_cdn_url), ("size", size)):
            print(f"{key}\t{value}")
    elif command == "hash":
        p, expected, size = path(args[0]), args[1], args[2]
        if not p.is_file() or p.is_symlink() or p.stat().st_size > 4 * 1024**3: fail("invalid/oversized archive")
        if size and p.stat().st_size != int(size): fail("archive size differs from manifest")
        digest = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""): digest.update(block)
        actual = digest.hexdigest()
        if not hmac.compare_digest(actual, expected): fail(f"SHA256 mismatch: got {actual}")
        print(f"SHA256 verified: {actual}")
    elif command == "extract":
        archive, destination = path(args[0]), canonical(args[1])
        if destination.exists(): fail("extraction directory already exists")
        # Validate the ENTIRE central directory before creating any output.
        with zipfile.ZipFile(archive) as z:
            members, seen, files, total = [], set(), set(), 0
            if len(z.infolist()) > 100000: fail("too many ZIP entries")
            for item in z.infolist():
                name = item.orig_filename
                plain(name)
                if "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name): fail("absolute/foreign ZIP path")
                trimmed = name[:-1] if name.endswith("/") else name
                parts = trimmed.split("/")
                if any(p in ("", ".", "..") for p in parts) or len(name) > 1024: fail("ZIP path traversal/invalid component")
                if parts[0] not in ("Reelpack.app", "__MACOSX"): fail("ZIP must contain only Reelpack.app (plus optional AppleDouble metadata)")
                mode = item.external_attr >> 16
                kind = stat.S_IFMT(mode)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR): fail("ZIP symlink or special file rejected")
                if kind == stat.S_IFDIR and not item.is_dir(): fail("inconsistent ZIP directory type")
                if item.flag_bits & 1 or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED): fail("encrypted/unsupported ZIP entry")
                key = "/".join(unicodedata.normalize("NFC", p).casefold() for p in parts)
                if key in seen: fail("duplicate/case-colliding ZIP path")
                seen.add(key)
                if not item.is_dir(): files.add(key)
                total += item.file_size
                if total > 4 * 1024**3 or item.file_size > 1024**3: fail("ZIP uncompressed size limit exceeded")
                if item.file_size > 1024**2 and item.file_size > max(1, item.compress_size) * 1000: fail("ZIP compression ratio limit exceeded")
                if item.is_dir() and item.file_size: fail("nonempty ZIP directory")
                members.append((item, parts, mode, key))
            for _, parts, _, key in members:
                components = key.split("/")
                if any("/".join(components[:i]) in files for i in range(1, len(components))): fail("ZIP file/directory parent collision")
            if not any(parts[0] == "Reelpack.app" for _, parts, _, _ in members): fail("missing Reelpack.app")
            destination.mkdir(mode=0o700)
            for item, parts, mode, _ in members:
                if parts[0] == "__MACOSX": continue  # Never restore AppleDouble/xattrs.
                target = destination.joinpath(*parts)
                if os.path.commonpath((str(destination), os.path.realpath(target))) != str(destination): fail("extraction escaped staging directory")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o755)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                    with z.open(item) as inp, target.open("xb") as out:
                        shutil.copyfileobj(inp, out, 1024 * 1024)  # zipfile validates CRC.
                    target.chmod((mode & 0o777) or 0o644)
        print(destination / "Reelpack.app")
    elif command == "app":
        app, os_version = path(args[0]), args[1]
        if app.is_symlink() or not app.is_dir(): fail("App is missing or a symlink")
        for root, directories, filenames in os.walk(app, followlinks=False):
            for name in directories + filenames:
                p = Path(root) / name
                if p.is_symlink(): fail("App contains a symlink")
                if not (p.is_dir() or p.is_file()): fail("App contains a special file")
        with (app / "Contents/Info.plist").open("rb") as f: info = plistlib.load(f)
        if info.get("CFBundleExecutable") != "Reelpack": fail("CFBundleExecutable must be Reelpack")
        if info.get("CFBundleIdentifier") != "com.reelpack.app": fail("unexpected Reelpack bundle identifier")
        if info.get("CFBundleIconFile") != "AppIcon": fail("CFBundleIconFile must be AppIcon")
        icon = app / "Contents/Resources/AppIcon.icns"
        if not icon.is_file(): fail("App resource missing: AppIcon.icns")
        with icon.open("rb") as f:
            if f.read(4) != b"icns": fail("invalid AppIcon.icns")
        minimum = info.get("LSMinimumSystemVersion", "14.0")
        def numbers(v): return tuple(int(x) for x in v.split("."))
        if numbers(minimum) > numbers(os_version): fail("App needs a newer macOS")
        for name in ("SKILL.md", "RELEASE_NOTES.md"):
            p = app / "Contents/Resources" / name
            if not p.is_file() or not 0 < p.stat().st_size <= 2 * 1024 * 1024:
                fail(f"App resource missing/invalid: {name}")
        core = app / "Contents/Resources/Core"
        required = ["agent_entry.py", "runtime_environment.py", "core.py", "audio_analyze.py", "shot_detect_transnet.py", "optional_backend_worker.py", "requirements-arm64.txt", "requirements-audio.txt", "requirements-transnet.txt", "constraints-transnet.txt", "requirements-ocr.txt", "requirements-music.txt"]
        for name in required:
            if not (core / name).is_file(): fail(f"App resource missing: Core/{name}")
        for rel in ("Contents/MacOS/Reelpack", "Contents/Resources/Tools/macos_ocr/macos_ocr"):
            p = app / rel
            if not p.is_file() or not os.access(p, os.X_OK): fail(f"App executable missing: {rel}")
        print(f"App layout verified: {app}")
    elif command == "skills":
        integrate_skills(path(args[0]), path(args[1]), register=args[2] == "0", check_only=args[3] == "1", native_brew=args[4])
    elif command == "move":
        source, destination, root = path(args[0]), path(args[1]), canonical(args[2])
        # Check both resolved absolute paths BEFORE any rename; never overwrite.
        if canonical(str(source)) != source or canonical(str(destination)) != destination: fail("rename path contains a symlink")
        if source == root or destination == root: fail("cannot rename destination root")
        for p in (source, destination):
            if os.path.commonpath((str(root), str(p))) != str(root): fail("rename path outside expected root")
        if not source.exists() or os.path.lexists(destination): fail("rename source missing or destination occupied")
        os.rename(source, destination)
    elif command == "clean-app-stage":
        stage, root, target = path(args[0]), canonical(args[1]), path(args[2])
        if stage.parent != root or not stage.name.startswith(".reelpack-stage.") or canonical(str(stage)) != stage:
            fail("unsafe App stage cleanup path")
        if target != root / "Reelpack.app" or not target.is_dir() or target.is_symlink():
            fail("promoted App is missing")
        skill_safe_path(stage).rmdir()  # Empty only, exact stage passed by this invocation.
    elif command == "lock":
        root, owner = canonical(args[0]), args[1]
        lock = root / ".reelpack-install.lock"
        lock.mkdir(mode=0o700)  # Atomic; an existing/stale lock is never removed automatically.
        (lock / "owner").write_text(owner, encoding="ascii")
        print(lock)
    elif command == "unlock":
        root, lock, owner = canonical(args[0]), path(args[1]), args[2]
        expected = root / ".reelpack-install.lock"
        if lock != expected or canonical(str(lock)) != expected or lock.is_symlink(): fail("unsafe lock cleanup path")
        marker = lock / "owner"
        if marker.is_symlink() or marker.read_text(encoding="ascii") != owner: fail("lock ownership changed")
        marker.unlink()
        lock.rmdir()  # Only our verified, otherwise-empty lock directory.
    elif command == "gh":
        executable, repo, tag, asset, destination, timeout = args
        env = os.environ.copy()
        env.pop("GH_DEBUG", None)
        env["GH_HOST"] = "github.com"
        env["GH_PROMPT_DISABLED"] = "1"
        env["GH_PAGER"] = "cat"
        if asset:
            cmd = [executable, "release", "download", tag, "--repo", repo, "--pattern", asset, "--dir", destination]
        else:
            cmd = [executable, "release", "view", "--repo", repo, "--json", "tagName", "--jq", ".tagName"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=int(timeout))
        if proc.returncode: fail("authenticated gh release operation failed; confirm access and that the release asset exists")
        if not asset: print(version(proc.stdout.strip()))
    else:
        fail("unknown helper command")
except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
    print(f"[Reelpack] {command}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

APP_DIR=$(py_task canonical "$APP_DIR") || die 2 "Unsafe --app-dir"
RUNTIME_ROOT=$(py_task canonical "$RUNTIME_ROOT") || die 2 "Unsafe --runtime-root"
APP_TARGET="$APP_DIR/Reelpack.app"
CORE_DIR="$APP_TARGET/Contents/Resources/Core"
export PATH="$RUNTIME_ROOT/bin:$NATIVE_BREW/bin:$NATIVE_BREW/sbin:$PATH"
export REELPACK_RUNTIME_ROOT="$RUNTIME_ROOT"
if [ -n "$WHEELHOUSE" ]; then
    WHEELHOUSE=$(py_task directory "$WHEELHOUSE") || die 2 "--wheelhouse must be an existing local directory"
fi
py_task names "$VERSION" "$ASSET_NAME" "${ALLOW_HOSTS[@]}" || die 2 "Invalid version, asset or hostname"
if [ -z "$GITHUB_BASE" ]; then GITHUB_BASE="https://github.com/$REPOSITORY/releases/download"; fi
for configured_url in "$GITHUB_BASE" "$GH_PROXY_BASE" "$JSDELIVR_MANIFEST_URL" "$GITEE_MIRROR_BASE"; do
    if [ -n "$configured_url" ]; then
        configured_host=$(py_task host "$configured_url") || die 2 "Invalid configured HTTPS download location"
        ALLOW_HOSTS+=("$configured_host")
    fi
done

LOCK_OWNER="$$-$(/bin/date +%s)"
cleanup() {
    local code=$?
    trap - EXIT
    if [ -n "$LOCK_DIR" ]; then
        py_task unlock "$RUNTIME_ROOT" "$LOCK_DIR" "$LOCK_OWNER" || warn "Could not release our installation lock: $LOCK_DIR"
    fi
    if [ "$code" -ne 0 ] && [ -n "$WORK_DIR" ]; then
        warn "Staging and dependency logs preserved at: $WORK_DIR"
    fi
    if [ "$code" -ne 0 ] && [ -n "$APP_STAGE" ]; then
        warn "App staging preserved for diagnosis: $APP_STAGE"
    fi
    exit "$code"
}
trap cleanup EXIT

validate_app() {
    py_task app "$1" "$OS_VERSION" || die 3 "Invalid Reelpack App bundle"
    native_binary "$1/Contents/MacOS/Reelpack" || die 3 "App executable has no native $ARCH slice"
    native_binary "$1/Contents/Resources/Tools/macos_ocr/macos_ocr" || die 3 "macOS OCR helper has no native $ARCH slice"
}

# curl never loads ~/.curlrc, .netrc, auth headers or gh tokens. Configured mirror
# requests are always unauthenticated. Signed GitHub redirects are handled by curl.
curl_download() {
    local source=$1 destination=$2 kind=$3 limit max_size effective
    py_task url "$source" "${ALLOW_HOSTS[@]}" || return 1
    if [ "$kind" = archive ]; then limit=300; max_size=4294967296; else limit=45; max_size=1048576; fi
    if ! effective=$(/usr/bin/curl --disable --fail --silent --show-error --location \
        --proto '=https' --proto-redir '=https' --connect-timeout 10 \
        --max-time "$limit" --retry 2 --retry-delay 2 --retry-max-time "$limit" \
        --max-filesize "$max_size" --output "$destination" --write-out '%{url_effective}' "$source"); then
        return 1
    fi
    # Redirect URLs can be short-lived signed URLs. Check the final host separately,
    # without treating their query as an installer-configurable credential.
    final_host=${effective#https://}; final_host=${final_host%%/*}; final_host=${final_host%%:*}
    local allowed=0 host
    for host in "${ALLOW_HOSTS[@]}"; do [ "$final_host" != "$host" ] || allowed=1; done
    [ "$allowed" -eq 1 ] || { warn "Download redirected to a hostname outside the allowlist"; return 1; }
    [ -s "$destination" ] || return 1
}

public_download() {
    local original=$1 basename=$2 destination=$3 kind=$4 source attempt
    [ "$PUBLIC_RELEASE" -eq 1 ] || die 3 "Public/proxy download is disabled for private releases"
    attempt=1
    log "Download attempt $attempt: direct GitHub/origin ($kind)"
    if curl_download "$original" "$destination.direct" "$kind"; then
        py_task move "$destination.direct" "$destination" "$WORK_DIR"; return
    fi
    if [ -n "$GH_PROXY_BASE" ] && [[ "$original" = https://github.com/* || "$original" = https://raw.githubusercontent.com/* ]]; then
        attempt=$((attempt + 1)); source="${GH_PROXY_BASE%/}/$original"
        log "Download attempt $attempt: configured public gh-proxy ($kind)"
        if curl_download "$source" "$destination.proxy" "$kind"; then
            py_task move "$destination.proxy" "$destination" "$WORK_DIR"; return
        fi
    else
        log "Skip gh-proxy: unconfigured or origin is not a public GitHub URL"
    fi
    if [ "$kind" = manifest ] && [ -n "$JSDELIVR_MANIFEST_URL" ]; then
        attempt=$((attempt + 1)); log "Download attempt $attempt: configured jsDelivr manifest"
        if curl_download "$JSDELIVR_MANIFEST_URL" "$destination.jsdelivr" manifest; then
            py_task move "$destination.jsdelivr" "$destination" "$WORK_DIR"; return
        fi
    elif [ "$kind" = archive ] && [ -n "$MANIFEST_ASSET_CDN_URL" ]; then
        attempt=$((attempt + 1)); log "Download attempt $attempt: explicit manifest CDN repository file"
        if curl_download "$MANIFEST_ASSET_CDN_URL" "$destination.cdn" archive; then
            py_task move "$destination.cdn" "$destination" "$WORK_DIR"; return
        fi
    elif [ "$kind" = archive ]; then
        log "Skip repository-file CDN: no explicit manifest cdn_url configured"
    else
        log "Skip jsDelivr: no published manifest URL configured"
    fi
    if [ -n "$GITEE_MIRROR_BASE" ]; then
        attempt=$((attempt + 1)); source="${GITEE_MIRROR_BASE%/}/$basename"
        log "Download attempt $attempt: configured Gitee mirror ($kind)"
        if curl_download "$source" "$destination.gitee" "$kind"; then
            py_task move "$destination.gitee" "$destination" "$WORK_DIR"; return
        fi
    else
        log "Skip Gitee: no published mirror base configured"
    fi
    die 3 "All configured $kind download sources failed; use a local --archive with a trusted --sha256"
}

parse_manifest() {
    local fields key value
    fields=$(py_task manifest "$1" "$ARCH" "$REPOSITORY" "$VERSION") || die 3 "Invalid release manifest"
    while IFS=$'\t' read -r key value; do
        case "$key" in
            version) MANIFEST_VERSION=$value ;;
            release_tag) MANIFEST_RELEASE_TAG=$value ;;
            filename) [ -z "$ASSET_NAME" ] || [ "$ASSET_NAME" = "$value" ] || die 3 "Manifest asset name mismatch"; ASSET_NAME=$value ;;
            sha256) EXPECTED_SHA256=$value ;;
            url) MANIFEST_ASSET_URL=$value ;;
            cdn_url) MANIFEST_ASSET_CDN_URL=$value ;;
            size) MANIFEST_ASSET_SIZE=$value ;;
        esac
    done <<< "$fields"
    [ -n "$VERSION" ] || VERSION=$MANIFEST_RELEASE_TAG
    [ -z "$MANIFEST_ASSET_URL" ] || py_task url "$MANIFEST_ASSET_URL" "${ALLOW_HOSTS[@]}" || die 3 "Manifest archive domain is not allowed"
    [ -z "$MANIFEST_ASSET_CDN_URL" ] || py_task url "$MANIFEST_ASSET_CDN_URL" "${ALLOW_HOSTS[@]}" || die 3 "Manifest CDN URL/domain is not allowed"
}

if [ "$CHECK_ONLY" -eq 1 ]; then
    [ -d "$APP_TARGET" ] || die 3 "Installed App not found at $APP_TARGET"
    validate_app "$APP_TARGET"
elif [ -e "$APP_TARGET" ] || [ -L "$APP_TARGET" ]; then
    [ ! -L "$APP_TARGET" ] || die 3 "Existing App target is a symlink"
    if [ "$REPLACE_APP" -eq 0 ]; then
        [ -z "$ARCHIVE$VERSION$MANIFEST_FILE$MANIFEST_URL$SHA256_INPUT" ] || die 3 "Existing App target preserved. Use --check, or explicitly pass --replace-app to create a backup before installing"
        validate_app "$APP_TARGET"
        log "Reusing the existing App; no release download requested"
    fi
fi

# Validate local manifest data before acquiring locks or creating runtime/staging.
if [ -n "$MANIFEST_FILE" ]; then parse_manifest "$MANIFEST_FILE"; fi
if [ "$CHECK_ONLY" -eq 0 ] && [ "$PUBLIC_RELEASE" -eq 1 ] && [ -z "$ARCHIVE$VERSION$MANIFEST_URL" ] && { [ ! -e "$APP_TARGET" ] || [ "$REPLACE_APP" -eq 1 ]; }; then
    die 2 "Public downloads require --version or a versioned --manifest-url"
fi
if [ "$CHECK_ONLY" -eq 0 ]; then
    /bin/mkdir -p "$RUNTIME_ROOT" || die 4 "Cannot create runtime root without sudo: $RUNTIME_ROOT"
    LOCK_DIR=$(py_task lock "$RUNTIME_ROOT" "$LOCK_OWNER") || die 4 "Runtime is locked by another/stale installer. Inspect .reelpack-install.lock manually; it was not removed."
    WORK_DIR=$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/reelpack-install.XXXXXXXX") || die 3 "Cannot create private staging directory"
    WORK_DIR=$(py_task directory "$WORK_DIR") || die 3 "Invalid staging directory"
    log "Staging/logs: $WORK_DIR"
fi

if [ "$CHECK_ONLY" -eq 0 ] && { [ ! -e "$APP_TARGET" ] || [ "$REPLACE_APP" -eq 1 ]; }; then
    if [ -n "$MANIFEST_URL" ]; then
        [ "$PUBLIC_RELEASE" -eq 1 ] || die 3 "--manifest-url requires --public-release; private metadata must use authenticated gh or a local --manifest"
        manifest_basename=${MANIFEST_URL##*/}
        [[ "$manifest_basename" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*\.json$ ]] || die 2 "Manifest URL needs a plain .json basename"
        public_download "$MANIFEST_URL" "$manifest_basename" "$WORK_DIR/install-manifest.json" manifest
        MANIFEST_FILE="$WORK_DIR/install-manifest.json"
    fi
    if [ -n "$MANIFEST_URL" ]; then parse_manifest "$MANIFEST_FILE"; fi
    if [ -n "$ARCHIVE" ]; then
        [ -n "$ASSET_NAME" ] || ASSET_NAME=${ARCHIVE##*/}
        # A local filename need not follow release naming; it is never a remote URL.
        py_task copy-archive "$ARCHIVE" "$WORK_DIR/application.zip" || die 3 "Cannot stage local archive"
    else
        if [ "$PUBLIC_RELEASE" -eq 0 ]; then
            GH=$(command -v gh 2>/dev/null || true)
            [ -n "$GH" ] || die 3 "Private release download requires an authenticated gh CLI, or --archive/--sha256. No public Release/mirror is assumed."
            if [ -z "$VERSION" ]; then VERSION=$(py_task gh "$GH" "$REPOSITORY" "" "" "" 60) || die 3 "No accessible private release found"; fi
        elif [ -z "$VERSION" ]; then
            die 2 "Public downloads require --version or a versioned --manifest-url; no release is invented"
        fi
        py_task names "$VERSION" "$ASSET_NAME" || die 2 "Invalid release version/asset"
        [ -n "$ASSET_NAME" ] || ASSET_NAME="Reelpack-${VERSION#v}-$ARCH.zip"
        if [ "$PUBLIC_RELEASE" -eq 0 ]; then
            /bin/mkdir "$WORK_DIR/gh-archive" || die 3 "Cannot create gh staging directory"
            log "Authenticated private GitHub download: $ASSET_NAME (direct only)"
            py_task gh "$GH" "$REPOSITORY" "$VERSION" "$ASSET_NAME" "$WORK_DIR/gh-archive" 300 || die 3 "Private release ZIP unavailable; no proxy was contacted"
            py_task move "$WORK_DIR/gh-archive/$ASSET_NAME" "$WORK_DIR/application.zip" "$WORK_DIR" || die 3 "Cannot stage gh archive"
        else
            asset_url=${MANIFEST_ASSET_URL:-"${GITHUB_BASE%/}/$VERSION/$ASSET_NAME"}
            public_download "$asset_url" "$ASSET_NAME" "$WORK_DIR/application.zip" archive
        fi
    fi
    if [ -n "$SHA256_INPUT" ]; then
        supplied_digest=$(py_task checksum "$SHA256_INPUT" "$ASSET_NAME") || die 3 "Invalid --sha256 value/list"
        [ -z "$EXPECTED_SHA256" ] || [ "$EXPECTED_SHA256" = "$supplied_digest" ] || die 3 "--sha256 conflicts with the manifest"
        EXPECTED_SHA256=$supplied_digest
    fi
    if [ -z "$EXPECTED_SHA256" ]; then
        [ -z "$ARCHIVE" ] || die 3 "Local archives require --sha256 or --manifest; unverified packages are never unpacked"
        checksum_name="$ASSET_NAME.sha256"
        if [ "$PUBLIC_RELEASE" -eq 0 ]; then
            /bin/mkdir "$WORK_DIR/gh-checksum" || die 3 "Cannot create checksum staging directory"
            py_task gh "$GH" "$REPOSITORY" "$VERSION" "$checksum_name" "$WORK_DIR/gh-checksum" 90 || die 3 "Private release SHA256 asset unavailable"
            checksum_path="$WORK_DIR/gh-checksum/$checksum_name"
        else
            checksum_path="$WORK_DIR/asset.sha256"
            public_download "${GITHUB_BASE%/}/$VERSION/$checksum_name" "$checksum_name" "$checksum_path" checksum
        fi
        EXPECTED_SHA256=$(py_task checksum "$checksum_path" "$ASSET_NAME") || die 3 "Release checksum list is invalid"
    fi
    py_task hash "$WORK_DIR/application.zip" "$EXPECTED_SHA256" "$MANIFEST_ASSET_SIZE" || die 3 "Archive integrity check failed; nothing unpacked"
    extracted_app=$(py_task extract "$WORK_DIR/application.zip" "$WORK_DIR/extracted") || die 3 "Unsafe or corrupt ZIP rejected"
    validate_app "$extracted_app"
    /bin/mkdir -p "$APP_DIR" || die 3 "Cannot create App directory without sudo: $APP_DIR"
    [ -w "$APP_DIR" ] || die 3 "App directory is not writable by your user: $APP_DIR"
    APP_STAGE=$(/usr/bin/mktemp -d "$APP_DIR/.reelpack-stage.XXXXXXXX") || die 3 "Cannot stage App in destination volume"
    /usr/bin/ditto "$extracted_app" "$APP_STAGE/Reelpack.app" || die 3 "ditto could not copy App; existing target preserved"
    validate_app "$APP_STAGE/Reelpack.app"
    if [ -e "$APP_TARGET" ]; then
        backup_app="$APP_TARGET.backup-$(/bin/date +%Y%m%d-%H%M%S)-$$"
        py_task move "$APP_TARGET" "$backup_app" "$APP_DIR" || die 3 "Cannot safely back up existing App"
        log "Existing App backed up: $backup_app"
    fi
    py_task move "$APP_STAGE/Reelpack.app" "$APP_TARGET" "$APP_DIR" || die 3 "Could not promote staged App. Any existing backup remains intact."
    py_task clean-app-stage "$APP_STAGE" "$APP_DIR" "$APP_TARGET" || die 3 "App promoted but empty stage cleanup failed: $APP_STAGE"
    APP_STAGE=
    log "Installed App: $APP_TARGET"
fi

# A requirements snapshot is intentionally kept per environment. In particular,
# requirements-arm64.txt is also resolved natively on Intel: unavailable pins fail
# explicitly; there is no Rosetta or cross-architecture fallback.
verify_environment() {
    local name=$1 environment=$2 requirements=$3 constraints=$4 python="$2/bin/python"
    [ ! -L "$environment" ] && [ -d "$environment" ] || { warn "$name: missing environment or symlink directory"; return 1; }
    native_python "$python" || { warn "$name: Python is not native $ARCH / 3.12"; return 1; }
    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        "$python" -I -B - "$name" "$ARCH" "$requirements" "$constraints" "$environment" "$CHECK_ONLY" <<'PY'
import importlib
import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path
import platform
import re
import sys
name, arch, requirements, constraints, environment, check_only = sys.argv[1:]
read_only = check_only == "1"
sys.dont_write_bytecode = True
if read_only:
    os.environ["REELPACK_READONLY_CHECK"] = "1"
try:
    sys.path.insert(0, str(Path(requirements).parent))
    from runtime_environment import prepare_runtime_environment
    os.environ.update(prepare_runtime_environment(create=not read_only))
    if platform.machine() != arch or sys.version_info[:2] != (3, 12): raise RuntimeError("native Python mismatch")
    if sys.prefix == sys.base_prefix: raise RuntimeError("interpreter is not isolated in a venv")
    if Path(sys.prefix).resolve() != Path(environment).resolve(): raise RuntimeError("interpreter belongs to a different environment")
    for filename, required in ((requirements, True), (constraints, False)):
        if not filename: continue
        for line in Path(filename).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line: continue
            m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
            if not m: raise RuntimeError("unsupported requirement syntax; exact pins expected")
            try: actual = metadata.version(m[1])
            except metadata.PackageNotFoundError:
                if required: raise RuntimeError(f"missing distribution {m[1]}")
                continue
            if actual != m[2]: raise RuntimeError(f"{m[1]}: installed {actual}, expected {m[2]}")
    imports = {
        "main": ["numpy", "av", "cv2", "faster_whisper", "ctranslate2", "onnxruntime", "PIL", "yaml", "requests"],
        "audio": ["numpy", "librosa", "soundfile", "scipy", "pyloudnorm", "PIL"],
        "transnet": ["numpy", "torch", "cv2", "transnetv2_pytorch"],
        "ocr": ["numpy", "onnxruntime", "cv2", "rapidocr_onnxruntime"],
        "music": ["numpy", "torch", "librosa", "soundfile", "torchlibrosa"],
    }[name]
    for module in imports:
        if read_only:
            if importlib.util.find_spec(module) is None: raise RuntimeError(f"module missing: {module}")
        else:
            importlib.import_module(module)
    if name == "music":
        # Importing panns_inference can download labels. Metadata/find_spec only.
        metadata.version("panns-inference")
        if importlib.util.find_spec("panns_inference") is None: raise RuntimeError("panns_inference module missing")
    checks = "pins/module metadata=ok; no imports or cache writes" if read_only else "pins/imports=ok"
    print(f"{name}: native={platform.machine()} Python={platform.python_version()} {checks}; JIT not_run")
except Exception as exc:
    print(f"{name}: validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
    local status=$?
    [ "$status" -eq 0 ] || return "$status"
    if [ "$CHECK_ONLY" -eq 1 ]; then
        log "$name: read-only exact pins/module metadata checked; dependency solver and JIT not_run"
        return 0
    fi
    if "$python" -I -B -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("pip") else 1)' >/dev/null 2>&1; then
        PYTHONDONTWRITEBYTECODE=1 PIP_CONFIG_FILE=/dev/null "$python" -I -B -m pip --disable-pip-version-check check || return 1
    elif [ -n "$UV" ]; then
        UV_NO_CONFIG=1 "$UV" --no-config --offline pip check --python "$python" || return 1
    else
        warn "$name: pip check unavailable (venv has no pip and native uv was not found)"; return 1
    fi
}

pip_attempt() {
    local python=$1 requirements=$2 constraints=$3 source=$4
    local arguments=(--python "$python" -r "$requirements")
    [ -z "$constraints" ] || arguments+=(-c "$constraints")
    # Ignore user/project pip/uv index configuration, including extra indexes.
    # Each attempt has one index only, never a simultaneous mixed index resolver.
    if [ "$PIP_BACKEND" = uv ]; then
        if [ -n "$WHEELHOUSE" ]; then
            env -u UV_INDEX -u UV_DEFAULT_INDEX -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL \
                -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_FIND_LINKS -u PIP_NO_INDEX -u PIP_TRUSTED_HOST \
                UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null UV_HTTP_TIMEOUT=45 UV_HTTP_RETRIES=2 \
                "$UV" --no-config --offline pip install "${arguments[@]}" --no-index --find-links "$WHEELHOUSE"
        else
            env -u UV_INDEX -u UV_DEFAULT_INDEX -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL \
                -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_FIND_LINKS -u PIP_NO_INDEX -u PIP_TRUSTED_HOST \
                UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null UV_HTTP_TIMEOUT=45 UV_HTTP_RETRIES=2 \
                "$UV" --no-config pip install "${arguments[@]}" --index-url "$source"
        fi
    else
        local pip_arguments=(-r "$requirements" --disable-pip-version-check --no-input --timeout 45 --retries 2)
        [ -z "$constraints" ] || pip_arguments+=(-c "$constraints")
        if [ -n "$WHEELHOUSE" ]; then pip_arguments+=(--no-index --find-links "$WHEELHOUSE"); else pip_arguments+=(--index-url "$source"); fi
        env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_FIND_LINKS -u PIP_NO_INDEX -u PIP_TRUSTED_HOST \
            PIP_CONFIG_FILE=/dev/null "$python" -I -B -m pip install "${pip_arguments[@]}"
    fi
}

setup_environment() {
    local name=$1 dirname=$2 requirements=$3 constraints=$4 environment="$RUNTIME_ROOT/$2"
    local repair=0 requested source attempt=0 logfile
    for requested in ${REPAIR_ENVS[@]+"${REPAIR_ENVS[@]}"}; do [ "$requested" != "$name" ] || repair=1; done
    if [ -e "$environment" ] || [ -L "$environment" ]; then
        [ ! -L "$environment" ] || die 4 "$name environment is a symlink; preserved and refused"
        if [ "$repair" -eq 0 ]; then
            log "Verify existing $name: $environment"
            if ! verify_environment "$name" "$environment" "$requirements" "$constraints"; then
                if [ "$name" = audio ]; then
                    die 4 "音频环境需要更新，请运行 --repair-env audio；会先备份。现有音频环境已保留，未自动重建。"
                fi
                die 4 "Existing $name environment preserved. To explicitly back it up and rebuild, rerun with --repair-env $name (plus --no-launch if desired)."
            fi
            return
        fi
        backup_env="$environment.backup-$(/bin/date +%Y%m%d-%H%M%S)-$$"
        py_task move "$environment" "$backup_env" "$RUNTIME_ROOT" || die 4 "Cannot safely back up $name environment"
        log "$name environment backed up: $backup_env"
    fi
    [ "$CHECK_ONLY" -eq 0 ] || die 4 "$name environment is missing: $environment"
    log "Create isolated native $name environment: $environment"
    "$PYTHON" -I -B -m venv "$environment" || die 4 "Cannot create $name venv; partial environment preserved"
    if [ -n "$WHEELHOUSE" ]; then
        logfile="$WORK_DIR/pip-$name-offline.log"
        log "$name: local wheelhouse only ($PIP_BACKEND); log=$logfile"
        pip_attempt "$environment/bin/python" "$requirements" "$constraints" "" >"$logfile" 2>&1 ||
            die 4 "$name wheelhouse install failed; no network was attempted. Inspect $logfile; use --repair-env $name to rebuild a partial environment."
    else
        local installed=0
        for source in https://pypi.org/simple https://pypi.tuna.tsinghua.edu.cn/simple https://mirrors.aliyun.com/pypi/simple; do
            attempt=$((attempt + 1)); logfile="$WORK_DIR/pip-$name-$attempt.log"
            log "$name: dependency attempt $attempt, single index=$source backend=$PIP_BACKEND"
            if pip_attempt "$environment/bin/python" "$requirements" "$constraints" "$source" >"$logfile" 2>&1; then installed=1; break; fi
            warn "$name: index attempt failed; log=$logfile"
        done
        [ "$installed" -eq 1 ] || die 4 "$name dependencies unavailable for native $ARCH / Python 3.12. No architecture/pin substitution. Logs: $WORK_DIR; explicit repair: --repair-env $name"
    fi
    verify_environment "$name" "$environment" "$requirements" "$constraints" ||
        die 4 "$name verification failed; environment preserved for diagnosis. Explicit rebuild: --repair-env $name"
}

setup_environment main .venv "$CORE_DIR/requirements-arm64.txt" ""
setup_environment audio .venv-audio "$CORE_DIR/requirements-audio.txt" ""
setup_environment transnet .venv-transnet "$CORE_DIR/requirements-transnet.txt" "$CORE_DIR/constraints-transnet.txt"
setup_environment ocr .venv-ocr "$CORE_DIR/requirements-ocr.txt" ""
setup_environment music .venv-music "$CORE_DIR/requirements-music.txt" ""

log "App: $APP_TARGET"
log "Runtime root: $RUNTIME_ROOT"
log "Environments: .venv / .venv-audio / .venv-transnet / .venv-ocr / .venv-music"
print_tier_plan
log "Acceleration cache: REELPACK_CACHE_ROOT or ~/Library/Caches/Reelpack; actual write denial falls back to stable TMPDIR/reelpack-cache-uid."
MODEL_HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
MODEL_HUB_CACHE=${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-$MODEL_HF_HOME/hub}}
case "$TIER" in lite) MODEL_NAME=small ;; *) MODEL_NAME=large-v3 ;; esac
model_cache_ready() {
    local repository=$1 snapshot required
    for snapshot in "$MODEL_HUB_CACHE/models--Systran--faster-whisper-$repository"/snapshots/*; do
        [ -d "$snapshot" ] || continue
        local ready=1
        for required in model.bin config.json tokenizer.json; do
            [ -s "$snapshot/$required" ] || ready=0
        done
        if [ -s "$snapshot/vocabulary.json" ] || [ -s "$snapshot/vocabulary.txt" ]; then :; else ready=0; fi
        [ "$ready" -eq 0 ] || return 0
    done
    return 1
}
if [ "$CHECK_ONLY" -eq 1 ]; then
    if model_cache_ready "$MODEL_NAME"; then
        log "Read-only model check: Whisper $MODEL_NAME files are present (filename/size presence only; inference not tested)."
    else
        log "Read-only model check: Whisper $MODEL_NAME is not locally prepared."
        print_model_notice "$MODEL_NAME" "$RUNTIME_ROOT/.venv/bin/python"
    fi
elif ! model_cache_ready "$MODEL_NAME"; then
    print_model_notice "$MODEL_NAME" "$RUNTIME_ROOT/.venv/bin/python"
    if [ "$NON_INTERACTIVE" -eq 0 ] && [ -t 0 ]; then
        printf 'Prepare Whisper %s now? This explicitly accesses the model source and may use the network. [y/N] ' "$MODEL_NAME"
        read -r model_answer || model_answer=n
        case "$model_answer" in y|Y|yes|YES)
            log "Preparing Whisper $MODEL_NAME after explicit confirmation; this may download model weights."
            HF_HOME="$MODEL_HF_HOME" HF_HUB_CACHE="$MODEL_HUB_CACHE" \
                "$RUNTIME_ROOT/.venv/bin/python" -I -B -c \
                "from faster_whisper import WhisperModel; WhisperModel(\"$MODEL_NAME\", device=\"cpu\", compute_type=\"int8\")" || warn "Whisper $MODEL_NAME preparation did not complete; no further download was attempted."
            ;;
            *) log "Whisper $MODEL_NAME was not prepared; no download attempted." ;;
        esac
    else
        log 'Non-interactive mode: no prompt and no model download was attempted.'
    fi
else
    log "Whisper $MODEL_NAME files are present in the local model cache (filename/size presence only; inference not tested)."
fi
log "ASR cache: preserves HF_HOME; persistent default is XDG_CACHE_HOME/huggingface or ~/.cache/huggingface. TORCH_HOME likewise stays persistent. Model checks never download weights."
print_ollama_guide
"$PYTHON" -I -B - <<'PY'
import csv
import os
from pathlib import Path
home = Path.home()
labels = Path(os.environ.get("REELPACK_PANNS_LABELS", str(home / "panns_data/class_labels_indices.csv"))).expanduser()
weights = Path(os.environ.get("REELPACK_PANNS_CHECKPOINT", str(home / "panns_data/Cnn14_mAP=0.431.pth"))).expanduser()
sed = Path(os.environ.get("REELPACK_PANNS_SED_CHECKPOINT", str(home / "panns_data/Cnn14_DecisionLevelMax_mAP=0.385.pth"))).expanduser()
try:
    valid_labels = labels.is_file() and labels.stat().st_size < 1024 * 1024
    if valid_labels:
        with labels.open(encoding="utf-8") as f: rows = list(csv.DictReader(f))
        valid_labels = len(rows) == 527 and all(row.get("display_name") for row in rows)
    valid_weights = weights.is_file() and weights.stat().st_size >= 300000000
    print("[Reelpack] PANNs: " + ("files present; inference not run" if valid_labels and valid_weights else "not_run (bring your own checkpoint and 527-label CSV; no download attempted)"))
    print(f"[Reelpack] PANNs checkpoint: {weights}")
    print(f"[Reelpack] PANNs labels: {labels}")
    print("[Reelpack] Optional PANNs SED: " + ("file present; inference not run" if sed.is_file() and sed.stat().st_size >= 300000000 else "not_run (optional weights absent/incomplete)"))
except (OSError, ValueError, csv.Error) as exc:
    print(f"[Reelpack] PANNs: not_run ({type(exc).__name__}: invalid/unreadable model files; no download attempted)")
PY

missing_tools=0
for tool in ffmpeg ffprobe; do
    executable=$(command -v "$tool" 2>/dev/null || true)
    if [ -z "$executable" ]; then
        warn "$tool missing. Install a native build (with existing native Homebrew: brew install ffmpeg), or place it in $RUNTIME_ROOT/bin."
        missing_tools=1
    elif ! native_binary "$executable"; then
        warn "$tool is not a native $ARCH executable: $executable. Supply a native build in $RUNTIME_ROOT/bin or $NATIVE_BREW/bin."
        missing_tools=1
    elif ! "$executable" -version >/dev/null 2>&1; then
        warn "$tool exists but cannot execute (check its native libraries): $executable"
        missing_tools=1
    else
        log "$tool: $executable"
    fi
done
[ "$missing_tools" -eq 0 ] || die 6 "App/environment checks completed, but a full App runtime check cannot run without native ffmpeg and ffprobe. Fix PATH/tools, then rerun with --check."

log "Run App runtime check with REELPACK_RUNTIME_ROOT and native PATH"
if ! env REELPACK_RUNTIME_ROOT="$RUNTIME_ROOT" PATH="$PATH" REELPACK_READONLY_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$APP_TARGET/Contents/MacOS/Reelpack" --runtime-check; then
    die 5 "App runtime check failed; App and environments preserved. Review the checker output, then rerun with --check."
fi
log "Runtime checks passed. Missing optional PANNs weights remain not_run."
py_task skills "$APP_TARGET" "$RUNTIME_ROOT" "$NO_AGENT_REGISTER" "$CHECK_ONLY" "$NATIVE_BREW" ||
    die 3 "Cannot validate Reelpack skill/CLI integration resources"
if [ "$NO_LAUNCH" -eq 0 ]; then
    launch_arguments=(--env "REELPACK_RUNTIME_ROOT=$RUNTIME_ROOT" --env "PATH=$PATH")
    for variable in REELPACK_CACHE_ROOT NUMBA_CACHE_DIR MPLCONFIGDIR PYTHONPYCACHEPREFIX HF_ENDPOINT HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TORCH_HOME XDG_CACHE_HOME REELPACK_PANNS_CHECKPOINT REELPACK_PANNS_SED_CHECKPOINT REELPACK_PANNS_LABELS; do
        value=${!variable:-}
        [ -z "$value" ] || launch_arguments+=(--env "$variable=$value")
    done
    /usr/bin/open "${launch_arguments[@]}" "$APP_TARGET" || die 5 "Checks passed, but macOS could not launch the App; open it manually"
    log "Launched Reelpack"
else
    log "GUI launch skipped. Open $APP_TARGET when ready; --check repeats read-only runtime checks."
fi
[ -z "$WORK_DIR" ] || log "Staging/logs retained for review: $WORK_DIR"

# Small manifest example (publish only when these assets actually exist):
# {"schema_version":1,"version":"0.2.0","release_tag":"v0.2.0","repository":"Lechoung-J/Reelpack-release",
#  "assets":{"arm64":{"filename":"Reelpack-0.2.0-arm64.zip",
#   "url":"https://github.com/Lechoung-J/Reelpack-release/releases/download/v0.2.0/Reelpack-0.2.0-arm64.zip",
#   "cdn_url":"https://cdn.jsdelivr.net/gh/Lechoung-J/Reelpack-release@v0.2.0/Reelpack-0.2.0-arm64.zip",
#   "sha256":"<64 lowercase hex digits>","size":12345}}}
# A cdn_url is optional and must refer to an actually published repository file,
# not an invented GitHub Release proxy. The identical ZIP still needs SHA256.
# For domestic script delivery, publish this same inspectable install.sh plus its
# SHA256 at a reachable origin or configured small-file mirror. CDN/Gitee asset
# URLs are never guessed when the manifest/configuration does not supply them.
# Do not pass GitHub tokens in mirror URLs or curl headers; private assets use gh.
exit 0
