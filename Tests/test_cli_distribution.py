"""Standalone archive integrity, activation, ownership and provenance contracts."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Core'))
from pipeline_state import engine_provenance, run_signature, reusable
from asset_contract import empty_analysis, migrate_speakers


def installer_module():
    source = (ROOT / 'install-cli.sh').read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    source = source.split('\ntry:\n    main()')[0]
    module = types.ModuleType('standalone_installer')
    exec(compile(source, 'install-cli.sh', 'exec'), module.__dict__)
    return module


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='reelpack CLI test ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.home = self.root / 'home space'
        self.home.mkdir()
        self.prefix = self.home / 'tool space'
        self.runtime = self.home / 'existing runtime'
        self.runtime.mkdir()
        (self.runtime / 'sentinel').write_text('unchanged')
        self.module = installer_module()
        self.counter = 0

    def archive(self, changes=None):
        self.counter += 1
        files = {name: b'resource\n' for name in [
            'bin/reelpack-cli', 'Tools/macos_ocr/macos_ocr', 'Core/core.py', 'Core/agent_entry.py',
            'Core/pipeline_state.py', 'SKILL.md', 'LICENSE', 'third_party/rea-video/LICENSE',
            'third_party/rea-video/NOTICE', 'Core/constraints-transnet.txt']}
        files.update({'Core/' + spec[0]: b'package==1\n' for spec in self.module.ENVIRONMENTS.values()})
        files['VERSION'] = b'0.2.2\n'
        files['SKILL.md'] = f'# CLI skill {self.counter}\n'.encode()
        info = {'schema_version': 1, 'product': 'Reelpack CLI', 'version': '0.2.2', 'architecture': 'arm64',
                'source_commit': 'a' * 40, 'source_dirty': False}
        files['release-info.json'] = json.dumps(info).encode()
        records = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data),
                         'mode': 0o755 if name.startswith(('bin/', 'Tools/')) else 0o644}
                   for name, data in files.items()}
        files['manifest.json'] = json.dumps({**info, 'files': records}).encode()
        if changes:
            changes(files)
        archive = self.root / f'fixture-{self.counter}.zip'
        with zipfile.ZipFile(archive, 'w') as zipped:
            for name, data in files.items():
                member = zipfile.ZipInfo('Reelpack-CLI-0.2.2-arm64/' + name)
                member.create_system = 3
                member.external_attr = (stat.S_IFREG | (0o755 if name.startswith(('bin/', 'Tools/')) else 0o644)) << 16
                zipped.writestr(member, data)
        return archive, hashlib.sha256(archive.read_bytes()).hexdigest()

    def install(self, archive=None, checksum=None, extra=(), runtime_error=None):
        arguments = ['install-cli.sh', '--prefix', str(self.prefix)]
        if archive:
            arguments += ['--archive', str(archive), '--sha256', checksum, '--existing-runtime', str(self.runtime)]
        arguments += list(extra)
        with patch.object(sys, 'argv', arguments), patch.object(Path, 'home', return_value=self.home), \
             patch.object(self.module.platform, 'system', return_value='Darwin'), \
             patch.object(self.module.platform, 'machine', return_value='arm64'), \
             patch.object(self.module.platform, 'mac_ver', return_value=('14.0', (), '')), \
             patch.object(self.module.os, 'geteuid', return_value=501), \
             patch.object(self.module, 'check_runtime', side_effect=runtime_error) as check, patch('sys.stdout', new=io.StringIO()):
            self.module.main()
        return check

    def test_fresh_repeat_upgrade_and_rollback_with_spaces(self):
        archive, checksum = self.archive()
        before = (self.runtime / 'sentinel').read_bytes()
        self.install(archive, checksum)
        previous = (self.prefix / 'current').readlink().name
        self.install(archive, checksum)
        self.assertEqual(len(list((self.prefix / 'versions').iterdir())), 1)
        second, digest = self.archive()
        self.install(second, digest)
        self.assertEqual(len(list((self.prefix / 'versions').iterdir())), 2)
        self.assertNotEqual((self.prefix / 'current').readlink().name, previous)
        check = self.install(extra=['--rollback', previous])
        check.assert_called_once_with(self.runtime)
        self.assertEqual((self.prefix / 'current').readlink().name, previous)
        self.assertEqual(before, (self.runtime / 'sentinel').read_bytes())

    def test_custom_launcher_and_skill_preserved_with_absolute_alternate(self):
        entry = self.home / '.local/bin/reelpack'
        entry.parent.mkdir(parents=True)
        entry.write_text('#!/bin/sh\necho custom\n')
        skill = self.home / '.dsh/skills/reelpack/SKILL.md'
        skill.parent.mkdir(parents=True)
        skill.write_text('custom skill')
        archive, checksum = self.archive()
        self.install(archive, checksum)
        self.assertEqual(entry.read_text(), '#!/bin/sh\necho custom\n')
        self.assertEqual(skill.read_text(), 'custom skill')
        self.assertTrue((self.prefix / 'bin/reelpack').is_file())

    def test_managed_app_launcher_upgraded_only_with_matching_ownership(self):
        entry = self.home / '.local/bin/reelpack'
        entry.parent.mkdir(parents=True)
        old = self.home / 'old App runtime'
        old.mkdir()
        import shlex
        entry.write_text('#!/bin/sh\nexport REELPACK_RUNTIME_ROOT=' + shlex.quote(str(old)) + '\n')
        original = entry.read_bytes()
        report = old / 'skill-integration.json'
        report.write_text(json.dumps({'managed_by': 'reelpack-install', 'schema_version': 1,
                          'files': [{'path': str(entry), 'sha256': hashlib.sha256(original).hexdigest()}]}))
        report_bytes = report.read_bytes()
        archive, checksum = self.archive()
        self.install(archive, checksum)
        self.assertNotEqual(entry.read_bytes(), original)
        self.assertEqual(next(entry.parent.glob('reelpack.backup-*')).read_bytes(), original)
        self.assertEqual(report.read_bytes(), report_bytes)

    def test_agent_directories_registered_and_opt_out(self):
        for directory in ('.dsh', '.claude', '.codex', '.cursor'):
            (self.home / directory).mkdir()
        archive, checksum = self.archive()
        self.install(archive, checksum, extra=['--no-agent-register'])
        self.assertFalse((self.home / '.local/bin/reelpack').exists())
        self.install(archive, checksum)
        for directory in ('.dsh', '.claude', '.agents', '.cursor'):
            self.assertIn('0.2.2', (self.home / directory / 'skills/reelpack/SKILL.md').read_text())
        # A second release preserves backups and updates managed skills.
        second, second_hash = self.archive()
        self.install(second, second_hash)
        self.assertTrue(list((self.home / '.dsh/skills').glob('reelpack.backup-*/SKILL.md')))

    def test_corrupt_archive_and_per_file_mutation_rejected_before_activation(self):
        archive, checksum = self.archive()
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.install(archive, '0' * 64)
        self.assertFalse((self.prefix / 'current').exists())
        broken, digest = self.archive(lambda files: files.update({'Core/core.py': b'tampered'}))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.install(broken, digest)
        self.assertFalse((self.prefix / 'current').exists())

    def test_paths_links_and_duplicate_members_rejected(self):
        archive, _ = self.archive()
        attacks = [('../escape', stat.S_IFREG), ('/absolute', stat.S_IFREG),
                   ('Reelpack-CLI-0.2.2-arm64/../../escape', stat.S_IFREG),
                   ('Reelpack-CLI-0.2.2-arm64/link', stat.S_IFLNK),
                   ('Reelpack-CLI-0.2.2-arm64/fifo', stat.S_IFIFO),
                   ('Reelpack-CLI-0.2.2-arm64/CORE/core.py', stat.S_IFREG)]
        for index, (name, kind) in enumerate(attacks):
            with self.subTest(name=name):
                bad = self.root / f'bad-{index}.zip'
                bad.write_bytes(archive.read_bytes())
                with zipfile.ZipFile(bad, 'a') as zipped:
                    member = zipfile.ZipInfo(name)
                    member.create_system = 3
                    member.external_attr = (kind | 0o644) << 16
                    zipped.writestr(member, '../../outside')
                with self.assertRaises(ValueError):
                    self.install(bad, hashlib.sha256(bad.read_bytes()).hexdigest())
                self.assertFalse((self.prefix / 'current').exists())
        self.assertFalse((self.root / 'escape').exists())

    def test_hardlinked_local_archive_rejected(self):
        archive, checksum = self.archive()
        os.link(archive, self.root / 'second-link.zip')
        with self.assertRaisesRegex(ValueError, 'nonregular'):
            self.install(archive, checksum)

    def test_corrupt_retained_package_and_conflicting_prefix_rejected(self):
        archive, checksum = self.archive()
        self.install(archive, checksum)
        active = (self.prefix / 'current').readlink()
        (self.prefix / 'current/Core/core.py').write_text('modified')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.install(archive, checksum)
        self.assertEqual((self.prefix / 'current').readlink(), active)

    def test_runtime_validation_failure_never_activates(self):
        archive, checksum = self.archive()
        with self.assertRaisesRegex(ValueError, 'broken runtime fixture'):
            self.install(archive, checksum, runtime_error=ValueError('broken runtime fixture'))
        self.assertFalse((self.prefix / 'current').exists())


    def test_nonmanaged_internal_launcher_and_symlink_prefix_preserved(self):
        self.prefix.mkdir()
        (self.prefix / 'bin').mkdir()
        entry = self.prefix / 'bin/reelpack'
        entry.write_text('custom')
        archive, checksum = self.archive()
        with self.assertRaisesRegex(ValueError, 'nonmanaged internal launcher'):
            self.install(archive, checksum)
        self.assertEqual(entry.read_text(), 'custom')
        self.assertFalse((self.prefix / 'current').exists())
        alias = self.home / 'alias'
        alias.symlink_to(self.prefix, target_is_directory=True)
        self.prefix = alias
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.install(archive, checksum)

    def test_managed_runtime_preparation_resumes_and_uses_relative_constraints(self):
        runtime = self.root / 'new runtime with spaces'
        core = self.root / 'Core with spaces'
        core.mkdir()
        (core / 'constraints-transnet.txt').write_text('numpy==1.26.4')
        calls = []
        def process(argv, **kwargs):
            calls.append((argv, kwargs))
        with patch.object(self.module, 'native_python', return_value=True), \
             patch.object(self.module.shutil, 'which', return_value='/usr/bin/uv'), \
             patch.object(self.module.subprocess, 'run', side_effect=process), \
             patch.object(self.module, 'check_runtime'):
            self.module.prepare_runtime(runtime, core, Path('/native/python3.12'), None)
            marker = self.module.read_json(runtime / '.reelpack-cli-runtime.json')
            self.assertEqual(marker['status'], 'ready')
            installs = [(argv, kwargs) for argv, kwargs in calls if 'install' in argv]
            self.assertEqual(len(installs), 5)
            transnet = next((argv, kwargs) for argv, kwargs in installs if 'constraints-transnet.txt' in argv)
            self.assertEqual(transnet[1]['cwd'], core)
            self.assertNotIn(str(core / 'constraints-transnet.txt'), transnet[0])
            calls.clear()
            marker['status'] = 'preparing'
            self.module.write_json(runtime / '.reelpack-cli-runtime.json', marker)
            self.module.prepare_runtime(runtime, core, Path('/native/python3.12'), None)
            self.assertEqual(len([argv for argv, _ in calls if 'install' in argv]), 5)
        custom = self.root / 'custom runtime'
        custom.mkdir()
        with self.assertRaisesRegex(ValueError, 'nonmanaged runtime'):
            self.module.prepare_runtime(custom, core, Path('/native/python3.12'), None)


class BuilderTests(unittest.TestCase):
    def test_dirty_source_release_build_rejected_before_compilation(self):
        spec = importlib.util.spec_from_file_location('cli_builder', ROOT / 'Build/build_cli.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(sys, 'argv', ['build_cli.py', '--output-dir', str(Path(temporary) / 'new')]), \
             patch.object(module.platform, 'system', return_value='Darwin'), \
             patch.object(module.platform, 'machine', return_value='arm64'), \
             patch.object(module, 'source_identity', return_value={'source_dirty': True}), \
             patch.object(module, 'run') as run, patch('sys.stderr', new=io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                module.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


class ProvenanceTests(unittest.TestCase):
    def test_old_pipeline_invalidated_without_default_opt_in_changes(self):
        signature = run_signature('a' * 64, {})
        self.assertEqual(signature['pipeline_revision'], '0.2.2')
        self.assertNotIn('profile', signature['options'])
        self.assertNotIn('speakers', signature['options'])
        old = copy.deepcopy(signature)
        old['pipeline_revision'] = '0.2.0'
        analysis = {'run_signature': old, 'execution': {'status': 'complete'}}
        self.assertFalse(reusable(analysis, signature))
        self.assertNotEqual(signature, run_signature('a' * 64, {'speakers': 'auto'}))
        self.assertNotEqual(signature, run_signature('a' * 64, {'profile': 'talk'}))

    def test_new_analysis_provenance_and_read_time_preservation(self):
        analysis = empty_analysis('asset_fixture', {})
        self.assertEqual(analysis['engine']['version'], '0.2.2')
        self.assertEqual(analysis['engine']['source_revision'], 'unknown')
        self.assertNotIn('/Users/', json.dumps(analysis['engine']))
        old = {'asset_id': 'asset_old', 'engine': {'version': '0.2.0'}}
        migrate_speakers(old)
        self.assertEqual(old['engine'], {'version': '0.2.0'})

    def test_packaged_revision_is_honest_and_portable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'Core').mkdir()
            info = {'source_commit': 'a' * 40, 'source_dirty': True, 'source_tree_sha256': 'b' * 64}
            (root / 'release-info.json').write_text(json.dumps(info))
            import pipeline_state
            with patch.object(pipeline_state, '__file__', str(root / 'Core/pipeline_state.py')):
                result = engine_provenance()
            self.assertEqual(result['source_revision'], 'a' * 40)
            self.assertTrue(result['source_dirty'])
            self.assertEqual(result['source_tree_sha256'], 'b' * 64)

    def test_standalone_native_resolution_does_not_fall_back(self):
        import agent_entry
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'Core').mkdir()
            (root / 'release-info.json').write_text('{}')
            (root / '.build/release').mkdir(parents=True)
            (root / '.build/release/reelpack-cli').write_text('old')
            with patch.object(agent_entry, '__file__', str(root / 'Core/agent_entry.py')):
                with self.assertRaises(FileNotFoundError):
                    agent_entry.native_cli()
                (root / 'bin').mkdir()
                (root / 'bin/reelpack-cli').write_text('new')
                self.assertEqual(agent_entry.native_cli(), root / 'bin/reelpack-cli')


if __name__ == '__main__':
    unittest.main()
