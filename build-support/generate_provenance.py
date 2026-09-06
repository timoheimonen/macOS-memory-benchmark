#!/usr/bin/env python3
"""Freeze source/build inputs at make time; update the header only on change."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def command(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    commit = command(['git', 'rev-parse', '--verify', 'HEAD'])
    status = command(['git', 'status', '--porcelain', '--untracked-files=normal']) if commit else None
    compiler = shlex.split(os.environ['PROVENANCE_CXX'])
    manifest = dict(manifest_version=1, status='available', reason_code='valid', binary_sha256=None,
                    git_commit=commit, git_dirty=bool(status) if status is not None else None,
                    compiler=command(compiler + ['--version']),
                    compile_flags={key: os.environ['PROVENANCE_' + key.upper()]
                                   for key in ('cxxflags', 'test_cxxflags', 'asflags')},
                    link_flags=os.environ['PROVENANCE_LDFLAGS'],
                    target_arch=command(compiler + ['-arch', 'arm64', '-dumpmachine']),
                    sdk=command(['xcrun', '--sdk', 'macosx', '--show-sdk-version']),
                    min_os=os.environ.get('MACOSX_DEPLOYMENT_TARGET'))
    if any(manifest[key] is None for key in ('git_commit', 'git_dirty', 'compiler', 'target_arch', 'sdk', 'min_os')):
        manifest.update(status='partial', reason_code='build-fields-unavailable')
    payload = json.dumps(manifest, sort_keys=True)
    if len(payload) > 16384:
        raise SystemExit('build provenance exceeds 16 KiB budget')
    header = '// Generated build inputs; do not edit.\n#define LLM_BUILD_MANIFEST_JSON ' + json.dumps(payload) + '\n'
    path = Path(sys.argv[1])
    if not path.exists() or path.read_text() != header:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(header)
        temporary.replace(path)


if __name__ == '__main__':
    main()
