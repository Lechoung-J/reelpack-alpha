#!/usr/bin/env python3
"""Build an architecture-specific Reelpack app and a checksum-verified archive.

Python environments, model weights and FFmpeg are installed separately by install.sh.
This builder never deletes or overwrites an existing app bundle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def run(command, **kwargs):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), cwd=ROOT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=['arm64', 'x86_64'], default=platform.machine())
    parser.add_argument('--version', default=(ROOT / 'VERSION').read_text().strip())
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--sign-identity', default='-')
    args = parser.parse_args()
    if not all(part.isdigit() for part in args.version.split('.')) or len(args.version.split('.')) != 3:
        parser.error('version must have three numeric components')
    canonical_version = (ROOT / 'VERSION').read_text().strip()
    if args.version != canonical_version:
        parser.error('--version must match the committed VERSION file')
    cli_source = (ROOT / 'CLI/main.swift').read_text()
    if f'version: "{args.version}"' not in cli_source or f'Reelpack CLI v{args.version}' not in cli_source:
        parser.error('CLI version/banner must match VERSION')
    for name in ['README.md', 'README.zh-CN.md', 'RELEASE_NOTES.md']:
        if args.version not in (ROOT / name).read_text():
            parser.error(name + ' must describe this product version')
    source_commit = run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    source_dirty = bool(run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip())
    if source_dirty:
        parser.error('release builds require a clean committed source tree')
    source_time = run(['git', 'show', '-s', '--format=%cI', 'HEAD'], capture_output=True, text=True).stdout.strip()
    swift_version = run(['swift', '--version'], capture_output=True, text=True).stdout.strip()
    release_tag = 'v' + args.version
    folder = (args.output_dir or ROOT/'dist'/('Reelpack-' + args.version + '-' + args.arch)).resolve()
    bundle = folder/'Reelpack.app'
    archive = folder/('Reelpack-' + args.version + '-' + args.arch + '.zip')
    if bundle.exists() or archive.exists():
        parser.error('output already exists; select a new output directory to preserve previous artifacts')
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for product in ['Reelpack', 'reelpack-cli']:
        run(['swift', 'build', '-c', 'release', '--arch', args.arch, '--product', product])
    output = run(['swift', 'build', '-c', 'release', '--arch', args.arch, '--show-bin-path'],
                 capture_output=True, text=True).stdout.strip()
    bin_dir = Path(output)
    if run([bin_dir/'reelpack-cli', '--version'], capture_output=True, text=True).stdout.strip() != args.version:
        raise RuntimeError('built CLI version differs from the requested version')
    run(['sh', 'Tools/macos_ocr/build.sh'], env={**__import__('os').environ, 'ARCH': args.arch})
    resources = bundle/'Contents/Resources'
    macos = bundle/'Contents/MacOS'
    macos.mkdir(parents=True)
    (resources/'Core').mkdir(parents=True)
    (resources/'Tools/macos_ocr').mkdir(parents=True)
    shutil.copy2(bin_dir/'Reelpack', macos/'Reelpack')
    shutil.copy2(bin_dir/'reelpack-cli', macos/'reelpack-cli')
    for pattern in ['*.py', 'requirements*.txt', 'constraints*.txt']:
        for path in sorted((ROOT/'Core').glob(pattern)):
            shutil.copy2(path, resources/'Core'/path.name)
    shutil.copy2(ROOT/'Tools/macos_ocr/macos_ocr', resources/'Tools/macos_ocr/macos_ocr')
    icon = ROOT/'Build/AppIcon.icns'
    if not icon.is_file() or icon.read_bytes()[:4] != b'icns':
        raise RuntimeError('required v3 App icon missing or invalid: Build/AppIcon.icns')
    shutil.copy2(icon, resources/'AppIcon.icns')
    (resources/'assets').mkdir()
    for name in ['logo.png']:
        shutil.copy2(ROOT/'assets'/name, resources/'assets'/name)
    for name in ['install.sh', 'LICENSE', 'README.md', 'README.zh-CN.md', 'SKILL.md', 'RELEASE_NOTES.md']:
        if not (ROOT/name).is_file():
            raise RuntimeError('required release source missing: ' + name)
        shutil.copy2(ROOT/name, resources/name)
    info = {
        'CFBundleExecutable': 'Reelpack', 'CFBundleIdentifier': 'com.reelpack.app',
        'CFBundleName': 'Reelpack', 'CFBundleDisplayName': 'Reelpack',
        'CFBundleIconFile': 'AppIcon',
        'CFBundlePackageType': 'APPL', 'CFBundleShortVersionString': args.version,
        'CFBundleVersion': args.version, 'LSMinimumSystemVersion': '14.0',
        'NSHighResolutionCapable': True, 'NSPrincipalClass': 'NSApplication',
    }
    with (bundle/'Contents/Info.plist').open('wb') as handle:
        plistlib.dump(info, handle)
    manifest = {'name': 'Reelpack', 'version': args.version, 'release_tag': release_tag,
                'source_commit': source_commit, 'source_dirty': source_dirty,
                'source_commit_time': source_time, 'toolchain': swift_version,
                'architecture': args.arch,
                'minimum_macos': '14.0', 'python': '3.12.4',
                'runtime_layout': '.venv / .venv-audio / .venv-transnet / .venv-ocr / .venv-music',
                'includes_python_runtime': False, 'includes_model_weights': False,
                'includes_ffmpeg': False}
    (resources/'release-info.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for executable in [macos/'Reelpack', macos/'reelpack-cli', resources/'Tools/macos_ocr/macos_ocr']:
        architectures = run(['/usr/bin/lipo', '-archs', executable], capture_output=True, text=True).stdout.strip().split()
        if architectures != [args.arch]:
            raise RuntimeError('unexpected architecture slices: ' + str(executable) + ': ' + repr(architectures))
        run(['/usr/bin/codesign', '--force', '--sign', args.sign_identity, executable])
    run(['/usr/bin/codesign', '--force', '--options', 'runtime', '--entitlements',
         'Build/entitlements.plist', '--sign', args.sign_identity, bundle])
    run(['/usr/bin/codesign', '--verify', '--deep', '--strict', '--verbose=2', bundle])
    run(['/usr/bin/ditto', '-c', '-k', '--norsrc', '--keepParent', bundle, archive],
        env={**os.environ, 'COPYFILE_DISABLE': '1'})
    with zipfile.ZipFile(archive) as zipped:
        metadata = [name for name in zipped.namelist()
                    if '__MACOSX' in Path(name).parts or Path(name).name.startswith('._')
                    or Path(name).name == '.DS_Store']
        if metadata or zipped.testzip() is not None:
            raise RuntimeError('ZIP metadata or CRC validation failed: ' + repr(metadata))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = folder/(archive.name+'.sha256')
    checksum.write_text(digest+'  '+archive.name+'\n')
    files = sorted(path for path in bundle.rglob('*') if path.is_file())
    categories = {
        'swift_executables': sum(path.stat().st_size for path in macos.iterdir() if path.is_file()),
        'python_source_and_requirements': sum(path.stat().st_size for path in (resources/'Core').iterdir() if path.is_file()),
        'vision_ocr': (resources/'Tools/macos_ocr/macos_ocr').stat().st_size,
        'app_icon': (resources/'AppIcon.icns').stat().st_size,
        'readme_images': sum(path.stat().st_size for path in (resources/'assets').iterdir()),
        'other_resources_and_metadata': 0,
    }
    total = sum(path.stat().st_size for path in files)
    categories['other_resources_and_metadata'] = total - sum(categories.values())
    record = {**manifest, 'bundle': str(bundle), 'archive': str(archive),
              'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
              'bundle_bytes': total, 'bundle_files': len(files), 'components_bytes': categories,
              'build_seconds': time.perf_counter()-started,
              'signing': 'ad-hoc' if args.sign_identity == '-' else 'provided signing identity',
              'notarized': False, 'zip_metadata_entries': metadata,
              'skill_sha256': hashlib.sha256((resources/'SKILL.md').read_bytes()).hexdigest()}
    (folder/'build-manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    install_manifest = {
        'schema_version': 1, 'version': args.version, 'release_tag': release_tag,
        'source_commit': source_commit, 'source_dirty': source_dirty,
        'repository': 'Lechoung-J/Reelpack-release',
        'minimum_macos': '14.0', 'distribution_status': 'preview',
        'assets': {args.arch: {'filename': archive.name, 'sha256': digest,
                              'size': archive.stat().st_size,
                               'url': f'https://github.com/Lechoung-J/Reelpack-release/releases/download/{release_tag}/{archive.name}'}},
    }
    # External to the archive: no self-referential checksum in the bundled installer.
    (folder/'release-manifest.json').write_text(json.dumps(install_manifest, indent=2)+'\n')
    for name in ['install.sh', 'LICENSE', 'RELEASE_NOTES.md']:
        shutil.copy2(ROOT / name, folder / name)
    installer_digest = hashlib.sha256((folder / 'install.sh').read_bytes()).hexdigest()
    (folder / 'install.sh.sha256').write_text(installer_digest + '  install.sh\n')
    if run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() != source_commit or run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip():
        raise RuntimeError('source changed during build; candidate is not releasable')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
