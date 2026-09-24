"""Run frozen category ablations and audit after an already-live refit audit."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    import json
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--dependency', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, dependency, out = args.experiment.resolve(), args.dependency.resolve(), args.output.resolve()
    launch = read(dependency/'launch.json')
    dependency_pid = launch['pid']
    observed = subprocess.check_output(['ps', '-p', str(dependency_pid), '-o', 'command='], text=True).strip()
    if 'finalize_recent_refit.py' not in observed or dependency.name not in observed:
        raise ValueError('Dependency does not match observed audit process')
    out.mkdir()
    (out/'launcher.py').write_bytes(Path(__file__).read_bytes())
    write_json(out/'launch.json', dict(pid=os.getpid(), started_at=utc_now(), experiment=str(root),
        dependency=str(dependency), dependency_pid=dependency_pid, observed_command=observed,
        frozen_manifest_sha256=file_hash(root/'frozen-manifest.json'),
        launcher_sha256=file_hash(out/'launcher.py')))
    try:
        while read(dependency/'status.json')['status'] != 'completed':
            if read(dependency/'status.json')['status'] == 'failed':
                raise RuntimeError('Dependency audit failed; no training launched')
            status = subprocess.run(['ps', '-p', str(dependency_pid), '-o', 'command='], capture_output=True, text=True)
            if status.returncode or status.stdout.strip() != observed:
                if read(dependency/'status.json')['status'] == 'completed':
                    break
                raise RuntimeError('Dependency process disappeared')
            write_json(out/'status.json', dict(status='waiting_for_refit_audit', updated_at=utc_now()))
            time.sleep(10)
        write_json(out/'dependency-receipt.json', dict(status_sha256=file_hash(dependency/'status.json'),
            verification_sha256=read(dependency/'status.json')['verification_sha256']))
        for stage, script, arguments in [
            ('training', 'category_ablation_run.py', ['run']),
            ('verifying', 'verify_category_ablation.py', [])]:
            write_json(out/'status.json', dict(status=stage, updated_at=utc_now()))
            with (out/f'{stage}.log').open('x') as log:
                subprocess.run([sys.executable, str(root/'code/scripts'/script), *arguments, '--output', str(root)],
                    env={**os.environ, 'PYTHONPATH': str(root/'code/src')}, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not read(root/'verification.json')['passed']:
            raise ValueError('Category audit failed')
        write_json(out/'status.json', dict(status='completed', updated_at=utc_now(),
            verification_sha256=file_hash(root/'verification.json')))
    except BaseException as exc:
        write_json(out/'status.json', dict(status='failed', updated_at=utc_now(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
