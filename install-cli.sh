#!/bin/bash
# Save and inspect this installer before running it. Never curl | bash.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec "${REELPACK_INSTALL_PYTHON:-python3}" -I -B - "$@" <<'PY'
"""Install a verified standalone Reelpack CLI without changing an App."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import uuid
import zipfile

VERSION = '0.2.2'
ENVIRONMENTS = {
    '.venv': ('requirements-arm64.txt', ('faster_whisper', 'cv2', 'numpy', 'PIL', 'requests')),
    '.venv-audio': ('requirements-audio.txt', ('librosa', 'soundfile', 'pyloudnorm')),
    '.venv-transnet': ('requirements-transnet.txt', ('transnetv2_pytorch', 'torch', 'cv2')),
    '.venv-ocr': ('requirements-ocr.txt', ('rapidocr_onnxruntime',)),
    '.venv-music': ('requirements-music.txt', ('panns_inference',)),
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def safe_path(value):
    p = Path(os.path.abspath(os.path.expanduser(str(value))))
    # macOS owns these root aliases. Normalize them before rejecting user links.
    if len(p.parts) > 1 and p.parts[1] in ('tmp', 'var'):
        alias = Path('/') / p.parts[1]
        if alias.is_symlink() and alias.resolve() == Path('/private') / p.parts[1]:
            p = alias.resolve().joinpath(*p.parts[2:])
    for component in [*reversed(p.parents), p]:
        if component.is_symlink():
            raise ValueError('symlink path preserved: ' + str(component))
        if component.exists() and component != p and not component.is_dir():
            raise ValueError('parent is not a directory: ' + str(component))
    return p


def regular(path):
    return path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1


def read_json(path):
    if not regular(safe_path(path)):
        raise ValueError('expected a single-link regular file: ' + str(path))
    return json.loads(path.read_text())


def write_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    atomic_file(path, data, 0o600)


def atomic_file(path, data, mode):
    safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.reelpack-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checksum(value, filename):
    if re.fullmatch('[a-fA-F0-9]{64}', value):
        return value.lower()
    rows = []
    for line in safe_path(value).read_text().splitlines():
        match = re.fullmatch(r'([a-fA-F0-9]{64})\s+\*?(.+)', line)
        if match and match[2] == filename:
            rows.append(match[1].lower())
    if len(rows) != 1:
        raise ValueError('checksum file must contain exactly one matching archive filename')
    return rows[0]


def unpack_verified(archive, expected, stage):
    if not regular(archive) or digest(archive) != expected:
        raise ValueError('archive SHA256 mismatch or nonregular archive')
    with zipfile.ZipFile(archive) as zipped:
        entries, folded, roots, total = [], set(), set(), 0
        for member in zipped.infolist():
            name = member.filename
            parts = PurePosixPath(name).parts
            if (not name or '\\' in name or name.startswith('/') or
                any(ord(c) < 32 for c in name) or any(p in ('..', '.') for p in name.rstrip('/').split('/')) or
                any(not p or ':' in p for p in name.rstrip('/').split('/'))):
                raise ValueError('unsafe archive path: ' + repr(name))
            key = name.rstrip('/').casefold()
            if key in folded:
                raise ValueError('duplicate archive member: ' + name)
            folded.add(key)
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or (member.is_dir() and kind == stat.S_IFREG):
                raise ValueError('archive links and special files are forbidden: ' + name)
            if member.flag_bits & 1:
                raise ValueError('encrypted archive member')
            roots.add(parts[0])
            total += member.file_size
            if total > 256 * 1024 * 1024 or len(entries) > 4096:
                raise ValueError('archive exceeds CLI resource limits')
            entries.append(member)
        if roots != {f'Reelpack-CLI-{VERSION}-arm64'}:
            raise ValueError('unexpected standalone package root')
        # Only extract regular files ourselves. ZIP metadata never creates a link.
        for member in entries:
            target = stage.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as handle:
                    handle.write(zipped.read(member))
                target.chmod((member.external_attr >> 16) & 0o777 & ~0o022)
    package = stage / f'Reelpack-CLI-{VERSION}-arm64'
    verify_package(package)
    return package


def verify_package(package):
    manifest = read_json(package / 'manifest.json')
    if (manifest.get('schema_version') != 1 or manifest.get('product') != 'Reelpack CLI' or
        manifest.get('version') != VERSION or manifest.get('architecture') != 'arm64'):
        raise ValueError('unsupported CLI manifest')
    files = manifest.get('files')
    required = {'bin/reelpack-cli', 'Tools/macos_ocr/macos_ocr', 'Core/core.py', 'Core/agent_entry.py',
                'Core/pipeline_state.py', 'VERSION', 'SKILL.md', 'LICENSE', 'release-info.json',
                'third_party/rea-video/LICENSE', 'third_party/rea-video/NOTICE'}
    required.update('Core/' + spec[0] for spec in ENVIRONMENTS.values())
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError('incomplete CLI manifest')
    actual = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    if actual != set(files) | {'manifest.json'}:
        raise ValueError('archive members differ from manifest')
    for name, record in files.items():
        parts = PurePosixPath(name).parts
        if not parts or name.startswith('/') or any(p in ('..', '.') for p in parts) or '\\' in name:
            raise ValueError('unsafe manifest path')
        path = package.joinpath(*parts)
        if (not regular(path) or digest(path) != record.get('sha256') or
            path.stat().st_size != record.get('size') or
            path.stat().st_mode & 0o777 != record.get('mode')):
            raise ValueError('per-file integrity failure: ' + name)
    for name in ('bin/reelpack-cli', 'Tools/macos_ocr/macos_ocr'):
        if (package / name).stat().st_mode & 0o111 != 0o111:
            raise ValueError('missing executable permissions: ' + name)
    info = read_json(package / 'release-info.json')
    for key, value in info.items():
        if manifest.get(key) != value:
            raise ValueError('release info and manifest disagree: ' + key)
    if (package / 'VERSION').read_text().strip() != VERSION:
        raise ValueError('VERSION and manifest disagree')
    return manifest


def native_python(python):
    try:
        subprocess.run([str(python), '-I', '-B', '-c',
            'import sys,platform; assert sys.version_info[:2]==(3,12); assert sys.platform=="darwin"; assert platform.machine()=="arm64"'],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def check_runtime(root):
    for name, (_, modules) in ENVIRONMENTS.items():
        python = root / name / 'bin/python'
        if not native_python(python):
            raise ValueError('runtime needs native Python 3.12: ' + str(python))
        code = 'import importlib.util; missing=[n for n in ' + repr(modules) + ' if importlib.util.find_spec(n) is None]; assert not missing, missing'
        subprocess.run([str(python), '-I', '-B', '-c', code], check=True)
    for name in ('ffmpeg', 'ffprobe'):
        if not shutil.which(name, path=str(root / 'bin') + ':' + os.environ.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin'):
            raise ValueError(name + ' missing; install it separately and rerun')


def prepare_runtime(root, core, requested_python, wheelhouse):
    if root.exists():
        if not regular(root / '.reelpack-cli-runtime.json'):
            raise ValueError('nonmanaged runtime preserved; choose --runtime-root with a new absolute path or --existing-runtime')
        marker = read_json(root / '.reelpack-cli-runtime.json')
        if marker.get('managed_by') != 'reelpack-cli-install':
            raise ValueError('nonmanaged runtime preserved')
        if marker.get('status') == 'ready':
            check_runtime(root)
            return
        if marker.get('status') != 'preparing':
            raise ValueError('unrecognized managed runtime state')
    candidates = [requested_python] if requested_python else [shutil.which('python3.12'), '/opt/homebrew/opt/python@3.12/bin/python3.12']
    uv = shutil.which('uv')
    if not any(p and native_python(p) for p in candidates) and uv:
        # uv is an existing local executable; no shell bootstrap is downloaded.
        subprocess.run([uv, '--no-config', 'python', 'install', '3.12.4'], check=True)
        candidates.append(subprocess.check_output([uv, '--no-config', 'python', 'find', '--offline', '3.12.4'], text=True).strip())
    python = next((p for p in candidates if p and native_python(p)), None)
    if not python:
        raise ValueError('install native Python 3.12 or existing uv; then pass --python /absolute/python3.12')
    root.mkdir(parents=True, exist_ok=True)
    # Environments are built at their final paths: venv scripts contain absolute
    # shebangs and must not be renamed after pip has prepared them.
    write_json(root / '.reelpack-cli-runtime.json', {'managed_by': 'reelpack-cli-install', 'status': 'preparing', 'python': '3.12'})
    for name, (requirements, _) in ENVIRONMENTS.items():
        environment = root / name
        subprocess.run([str(python), '-I', '-B', '-m', 'venv', str(environment)], check=True)
        if uv:
            command = [uv, '--no-config', 'pip', 'install', '--python', str(environment / 'bin/python'), '--only-binary=:all:']
        else:
            command = [str(environment / 'bin/python'), '-I', '-B', '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check', '--only-binary=:all:']
        if wheelhouse:
            command += ['--no-index', '--find-links', str(wheelhouse)]
        else:
            command += ['--index-url', 'https://pypi.org/simple']
        if name == '.venv-transnet':
            command += ['-c', 'constraints-transnet.txt']
        command += ['-r', requirements]
        subprocess.run(command, check=True, cwd=core)
    check_runtime(root)
    write_json(root / '.reelpack-cli-runtime.json', {'managed_by': 'reelpack-cli-install', 'status': 'ready', 'python': '3.12'})


def old_ownership(destination):
    """Prove an old App-managed launcher's hash without executing its contents."""
    try:
        if not regular(destination):
            return {}
        assignments = [line for line in destination.read_text().splitlines() if line.startswith('export REELPACK_RUNTIME_ROOT=')]
        if len(assignments) != 1:
            return {}
        tokens = shlex.split(assignments[0])
        if len(tokens) != 2:
            return {}
        runtime = safe_path(tokens[1].split('=', 1)[1])
        report = read_json(runtime / 'skill-integration.json')
        if report.get('managed_by') != 'reelpack-install' or report.get('schema_version') != 1:
            return {}
        return {item['path']: item['sha256'] for item in report.get('files', [])}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def materialize(destination, data, mode, owned):
    destination = safe_path(destination)
    if destination.exists():
        if not regular(destination):
            print('Custom entry preserved: ' + str(destination))
            return None
        before = digest(destination)
        expected = hashlib.sha256(data).hexdigest()
        if before == expected:
            return {'path': str(destination), 'sha256': before}
        if owned.get(str(destination)) != before:
            print('Custom entry preserved: ' + str(destination))
            return None
        if destination.name in ('SKILL.md', '.metadata.json'):
            directory = destination.parent.with_name(destination.parent.name + '.backup-' + uuid.uuid4().hex)
            safe_path(directory).mkdir()
            backup = directory / destination.name
        else:
            backup = destination.with_name(destination.name + '.backup-' + uuid.uuid4().hex)
        shutil.copy2(destination, backup)
        print('Managed entry backup: ' + str(backup))
    atomic_file(destination, data, mode)
    return {'path': str(destination), 'sha256': digest(destination)}


def register(home, prefix, runtime, no_register):
    launcher = prefix / 'bin/reelpack'
    content = ('#!/bin/sh\n# Generated by the standalone Reelpack CLI installer.\n'
        + 'export REELPACK_RUNTIME_ROOT=' + shlex.quote(str(runtime)) + '\n'
        + 'export PATH=' + shlex.quote(str(runtime / 'bin') + ':/opt/homebrew/bin:/usr/local/bin') + ':"${PATH:-/usr/bin:/bin}"\n'
        + 'export PYTHONDONTWRITEBYTECODE=1\n'
        + 'exec ' + shlex.quote(str(runtime / '.venv/bin/python')) + ' -I -B '
        + shlex.quote(str(prefix / 'current/Core/agent_entry.py')) + ' "$@"\n').encode()
    report_path = prefix / 'integration.json'
    previous = read_json(report_path) if report_path.exists() else {}
    owned = {r['path']: r['sha256'] for r in previous.get('files', [])}
    files = []
    record = materialize(launcher, content, 0o755, owned)
    if not record:
        raise ValueError('nonmanaged internal launcher; choose another --prefix')
    files.append(record)
    if not no_register:
        destination = home / '.local/bin/reelpack'
        proof = {**old_ownership(destination), **owned}
        try:
            record = materialize(destination, content, 0o755, proof)
            if record:
                files.append(record)
        except (OSError, ValueError) as exc:
            print('Custom command preserved: ' + str(exc))
        canonical = (prefix / 'current/SKILL.md').read_bytes()
        front = ('---\nname: reelpack\ndescription: Analyze local videos into transcripts, OCR, shots, audio timing and evidence frames.\n---\n\n'
                 + '## Installed standalone CLI 0.2.2\n\nUse this absolute command:\n\n```bash\n'
                 + shlex.quote(str(launcher)) + ' "<local video file>" --output-dir ~/Reelpack\n```\n\n').encode()
        native = front + canonical
        for detectors, relative in [(('.dsh',), '.dsh/skills/reelpack'), (('.claude',), '.claude/skills/reelpack'),
                                    (('.agents', '.codex'), '.agents/skills/reelpack'), (('.cursor',), '.cursor/skills/reelpack')]:
            if not any((home / name).is_dir() for name in detectors):
                continue
            directory = home / relative
            try:
                safe_path(directory)
                skill, metadata = directory / 'SKILL.md', directory / '.metadata.json'
                local_owned = dict(owned)
                if metadata.exists():
                    old = read_json(metadata)
                    if old.get('managed_by') in ('reelpack-install', 'reelpack-cli-install') and old.get('schema_version') == 1:
                        local_owned[str(skill)] = old.get('skill_sha256')
                        local_owned[str(metadata)] = digest(metadata)
                # Preserve an entire custom skill, including extra files.
                if directory.exists() and (set(p.name for p in directory.iterdir()) - {'SKILL.md', '.metadata.json'} or
                    not regular(skill) or local_owned.get(str(skill)) != digest(skill)):
                    print('Custom skill preserved: ' + str(directory) + '; read ' + str(prefix / 'current/SKILL.md'))
                    continue
                record = materialize(skill, native, 0o644, local_owned)
                if record:
                    files.append(record)
                    meta = {'schema_version': 1, 'managed_by': 'reelpack-cli-install', 'skill_sha256': record['sha256']}
                    result = materialize(metadata, (json.dumps(meta, sort_keys=True) + '\n').encode(), 0o600, local_owned)
                    if result:
                        files.append(result)
            except (OSError, ValueError) as exc:
                print('Custom skill preserved: ' + str(exc))
    # Retain ownership evidence for disabled entries so a later explicit run can upgrade them.
    paths = {r['path'] for r in files}
    files += [r for r in previous.get('files', []) if r['path'] not in paths]
    write_json(report_path, {'managed_by': 'reelpack-cli-install', 'schema_version': 1, 'files': files})
    print('Command to use (also when custom entries were preserved): ' + shlex.quote(str(launcher)))
    print('Runtime: ' + str(runtime))


@contextmanager
def install_lock(prefix):
    prefix.mkdir(parents=True, exist_ok=True)
    lock_path = safe_path(prefix / '.install.lock')
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if os.fstat(fd).st_nlink != 1:
            raise ValueError('nonregular install lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Local Reelpack-CLI-0.2.2-arm64.zip')
    parser.add_argument('--sha256', help='Trusted external SHA256 or matching checksum file')
    parser.add_argument('--public-release', action='store_true', help='Download official public 0.2.2 CLI ZIP and checksum')
    parser.add_argument('--version', choices=['0.2.2', 'v0.2.2'], default='0.2.2')
    parser.add_argument('--prefix', type=Path, default=Path.home() / '.local/share/reelpack-cli')
    parser.add_argument('--runtime-root', type=Path, help='Separate managed CLI runtime; never the App runtime')
    parser.add_argument('--existing-runtime', type=Path, help='Read-only Python 3.12 runtime; check all five environments, install no dependencies')
    parser.add_argument('--python', type=Path, help='Native Python 3.12 for preparing a new managed runtime')
    parser.add_argument('--wheelhouse', type=Path, help='Offline wheels only, without network fallback')
    parser.add_argument('--no-agent-register', action='store_true', help='Skip ~/.local/bin and Agent registrations; prefix/bin/reelpack remains available')
    parser.add_argument('--rollback', metavar='VERSION_DIRECTORY', help='Activate a preserved versions/ entry, after verifying it and its recorded runtime')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or os.geteuid() == 0:
        parser.error('run as your regular user on native arm64 macOS 14+')
    if int(platform.mac_ver()[0].split('.')[0]) < 14:
        parser.error('macOS 14+ required')
    if args.runtime_root and args.existing_runtime:
        parser.error('choose --runtime-root or --existing-runtime')
    if args.public_release and args.archive:
        parser.error('choose --public-release or --archive')
    if args.rollback and (args.archive or args.public_release or args.sha256 or args.runtime_root or args.existing_runtime):
        parser.error('--rollback uses the recorded install; omit archive/runtime options')
    if not args.rollback and not args.public_release and not (args.archive and args.sha256):
        parser.error('local installation requires --archive and trusted --sha256')
    prefix, home = safe_path(args.prefix), safe_path(Path.home())
    default_runtime = home / 'Library/Application Support/Reelpack/cli-runtime/arm64'
    runtime = safe_path(args.existing_runtime or args.runtime_root or default_runtime)
    old_default = home / 'Library/Application Support/Reelpack/runtime'
    if not args.existing_runtime and (runtime == old_default or old_default in runtime.parents):
        parser.error('App runtime is preserved; choose a separate --runtime-root or read-only --existing-runtime')
    with tempfile.TemporaryDirectory(prefix='reelpack-cli-download-') as downloads:
        archive = safe_path(args.archive) if args.archive else None
        expected = args.sha256
        if args.public_release:
            filename = f'Reelpack-CLI-{VERSION}-arm64.zip'
            base = f'https://github.com/Lechoung-J/Reelpack-release/releases/download/v{VERSION}/'
            archive = Path(downloads) / filename
            sidecar = Path(downloads) / (filename + '.sha256')
            urllib.request.urlretrieve(base + filename, archive)
            urllib.request.urlretrieve(base + sidecar.name, sidecar)
            expected = str(sidecar)
        with install_lock(prefix):
            current = prefix / 'current'
            # current is the only intentional managed symlink. It must point to
            # a verified version directly under this prefix, never outside it.
            if os.path.lexists(current):
                if not current.is_symlink() or current.readlink().parts[:1] != ('versions',) or len(current.readlink().parts) != 2:
                    raise ValueError('nonmanaged current entry preserved; choose a new --prefix')
                verify_package(current.resolve())
                active_record = read_json(prefix / 'installs' / (current.readlink().name + '.json'))
                if active_record.get('manifest_sha256') != digest(current.resolve() / 'manifest.json'):
                    raise ValueError('active installation ownership mismatch')
            launcher = prefix / 'bin/reelpack'
            previous = read_json(prefix / 'integration.json') if (prefix / 'integration.json').exists() else {}
            owned = {r['path']: r['sha256'] for r in previous.get('files', [])}
            if os.path.lexists(launcher) and (not regular(safe_path(launcher)) or owned.get(str(launcher)) != digest(launcher)):
                raise ValueError('nonmanaged internal launcher preserved; choose a new --prefix')
            versions = safe_path(prefix / 'versions')
            versions.mkdir(exist_ok=True)
            if args.rollback:
                if not re.fullmatch(r'0\.2\.2-[0-9a-f]{64}', args.rollback):
                    raise ValueError('use the exact preserved versions/ directory basename')
                package = safe_path(versions / args.rollback)
                verify_package(package)
                record = read_json(prefix / 'installs' / (args.rollback + '.json'))
                if record.get('manifest_sha256') != digest(package / 'manifest.json'):
                    raise ValueError('rollback ownership mismatch')
                runtime = safe_path(record['runtime'])
                check_runtime(runtime)
            else:
                expected = checksum(expected, archive.name)
                # Stage on the destination filesystem, then rename a complete,
                # verified immutable package into versions/ before activation.
                with tempfile.TemporaryDirectory(prefix='.stage-', dir=prefix) as temporary:
                    staged = unpack_verified(archive, expected, Path(temporary))
                    manifest_hash = digest(staged / 'manifest.json')
                    package = safe_path(versions / (VERSION + '-' + manifest_hash))
                    record_path = prefix / 'installs' / (package.name + '.json')
                    if package.exists():
                        verify_package(package)
                        if not record_path.exists() or read_json(record_path).get('manifest_sha256') != manifest_hash:
                            raise ValueError('nonmanaged version directory preserved')
                    if args.existing_runtime:
                        check_runtime(runtime)
                    else:
                        prepare_runtime(runtime, staged / 'Core', args.python, args.wheelhouse)
                    if not package.exists():
                        os.rename(staged, package)
                    write_json(record_path, {'managed_by': 'reelpack-cli-install', 'manifest_sha256': manifest_hash,
                                           'archive_sha256': expected, 'runtime': str(runtime)})
            temporary_link = prefix / ('.current-' + uuid.uuid4().hex)
            temporary_link.symlink_to(Path('versions') / package.name)
            old_target = current.readlink() if current.is_symlink() else None
            try:
                os.replace(temporary_link, current)
                register(home, prefix, runtime, args.no_agent_register)
            except BaseException:
                if old_target is not None:
                    if temporary_link.is_symlink():
                        temporary_link.unlink()
                    temporary_link.symlink_to(old_target)
                    os.replace(temporary_link, current)
                elif current.is_symlink():
                    current.unlink()
                raise
            finally:
                if temporary_link.is_symlink():
                    temporary_link.unlink()
            print('Active CLI: ' + package.name)
            print('Rollback: bash install-cli.sh --prefix ' + shlex.quote(str(prefix)) + ' --rollback ' + shlex.quote(old_target.name if old_target else package.name))


try:
    main()
except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
    print('Reelpack CLI installation failed: ' + str(exc), file=sys.stderr)
    sys.exit(1)
PY
