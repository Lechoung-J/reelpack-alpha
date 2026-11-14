#!/usr/bin/env python3
"""Build the arm64 standalone CLI, OCR helper and portable verified ZIP.

Release builds require a clean checkout. --allow-dirty is for local testing;
its manifest explicitly records that the source differs from source_commit.
Python runtimes, FFmpeg, models and user data are never bundled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def run(command, **kwargs):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), cwd=ROOT, check=True, **kwargs)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): {'sha256': sha256(p), 'size': p.stat().st_size,
            'mode': p.stat().st_mode & 0o777}
            for p in sorted(root.rglob('*')) if p.is_file()}


def source_identity():
    commit = run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip())
    # Hash current source bytes as well as the public parent revision. A dirty
    # development build must never claim that its parent fully identifies it.
    paths = run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                capture_output=True).stdout.decode().split('\0')
    files = {name: sha256(ROOT / name) for name in sorted(set(paths)) if name and (ROOT / name).is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'source_commit': commit, 'source_dirty': dirty, 'source_tree_sha256': digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=['arm64'], default='arm64')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory outside the source checkout')
    parser.add_argument('--allow-dirty', action='store_true', help='Local development only; recorded in manifest')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('build on a native Apple Silicon macOS host')
    version = (ROOT / 'VERSION').read_text().strip()
    if version != '0.2.2':
        parser.error('VERSION must be 0.2.2')
    folder = args.output_dir.expanduser().resolve()
    if folder == ROOT or ROOT in folder.parents or folder.exists():
        parser.error('choose a new output directory outside the source checkout')
    identity = source_identity()
    if identity['source_dirty'] and not args.allow_dirty:
        parser.error('release builds require clean committed source; use --allow-dirty only for local tests')
    # Swift embeds source locations in assertions and debug records. Remap the
    # whole build tree and strip local symbols before signing either executable.
    mapping = str(ROOT) + '=.'
    flags = ['-Xswiftc', '-file-prefix-map', '-Xswiftc', mapping,
             '-Xswiftc', '-debug-prefix-map', '-Xswiftc', mapping,
             '-Xcc', '-ffile-prefix-map=' + mapping]
    run(['swift', 'build', '-c', 'release', '--arch', args.arch, '--product', 'reelpack-cli', *flags])
    output = run(['swift', 'build', '-c', 'release', '--arch', args.arch, '--show-bin-path', *flags],
                 capture_output=True, text=True).stdout.strip()
    folder.mkdir(parents=True)
    package_name = f'Reelpack-CLI-{version}-{args.arch}'
    package = folder / package_name
    package.mkdir()
    (package / 'bin').mkdir()
    shutil.copy2(Path(output) / 'reelpack-cli', package / 'bin/reelpack-cli')
    (package / 'Tools/macos_ocr').mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='reelpack-ocr-') as temporary:
        run(['swiftc', '-O', '-target', 'arm64-apple-macosx14.0', '-module-cache-path',
             Path(temporary) / 'modules', '-file-prefix-map', mapping, '-debug-prefix-map', mapping,
             'Tools/macos_ocr/ocr_main.swift', '-o', package / 'Tools/macos_ocr/macos_ocr',
             '-framework', 'Vision', '-framework', 'AppKit'])
    for binary in [package / 'bin/reelpack-cli', package / 'Tools/macos_ocr/macos_ocr']:
        run(['/usr/bin/strip', '-S', binary])
        run(['/usr/bin/codesign', '--force', '--sign', '-', binary])
        run(['/usr/bin/codesign', '--verify', '--strict', binary])
        archs = run(['/usr/bin/lipo', '-archs', binary], capture_output=True, text=True).stdout.strip()
        if archs != 'arm64':
            raise RuntimeError('unexpected executable architecture: ' + archs)
        contents = binary.read_bytes()
        for forbidden in (str(ROOT).encode(), b'/Users/', b'/private/tmp/'):
            if forbidden in contents:
                raise RuntimeError('nonportable build location embedded in executable: ' + forbidden.decode())
    (package / 'Core').mkdir()
    for pattern in ['*.py', 'requirements*.txt', 'constraints*.txt']:
        for source in sorted((ROOT / 'Core').glob(pattern)):
            shutil.copy2(source, package / 'Core' / source.name)
    for name in ['VERSION', 'SKILL.md', 'LICENSE', 'RELEASE_NOTES.md']:
        shutil.copy2(ROOT / name, package / name)
    (package / 'docs').mkdir()
    for name in ['agent-evidence-loop.md', 'agent-supplements.md', 'motion-evidence.md', 'cli-install.md']:
        shutil.copy2(ROOT / 'docs' / name, package / 'docs' / name)
    shutil.copytree(ROOT / 'third_party', package / 'third_party', ignore=shutil.ignore_patterns('__pycache__', '.DS_Store'))
    info = {'schema_version': 1, 'product': 'Reelpack CLI', 'version': version,
            'architecture': args.arch, 'minimum_macos': '14.0', 'python': '3.12',
            'pipeline_revision': '0.2.2', **identity,
            'includes_python_runtime': False, 'includes_model_weights': False,
            'signing': 'ad-hoc', 'notarized': False}
    (package / 'release-info.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n')
    manifest = {**info, 'files': inventory(package)}
    (package / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    built_version = run([package / 'bin/reelpack-cli', '--version'], capture_output=True, text=True).stdout.strip()
    if built_version != version:
        raise RuntimeError('built executable version differs from VERSION')
    archive = folder / (package_name + '.zip')
    # Stable metadata/order makes the archive deterministic for identical signed bytes.
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for path in sorted(package.rglob('*')):
            if not path.is_file():
                continue
            member = zipfile.ZipInfo(package_name + '/' + path.relative_to(package).as_posix(), (2026, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = (0o100000 | (path.stat().st_mode & 0o777)) << 16
            member.compress_type = zipfile.ZIP_DEFLATED
            zipped.writestr(member, path.read_bytes())
    digest = sha256(archive)
    (folder / (archive.name + '.sha256')).write_text(digest + '  ' + archive.name + '\n')
    shutil.copy2(ROOT / 'install-cli.sh', folder / 'install-cli.sh')
    (folder / 'install-cli.sh.sha256').write_text(sha256(folder / 'install-cli.sh') + '  install-cli.sh\n')
    (folder / 'release-manifest.json').write_text(json.dumps({**info, 'filename': archive.name,
          'sha256': digest, 'size': archive.stat().st_size}, indent=2, sort_keys=True) + '\n')
    if source_identity() != identity:
        raise RuntimeError('source changed during build; discard this candidate')
    print(json.dumps({'archive': archive.name, 'sha256': digest, 'files': len(manifest['files']), **info}, indent=2))


if __name__ == '__main__':
    main()
