"""Run a frozen full audit after the observed recent-refit process completes."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    import json
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.experiment.resolve(), args.output.resolve()
    state = read(root/'run-status.json')
    if state['status'] not in ['running', 'completed']:
        raise ValueError('Refit is not live or complete')
    pid, command = state.get('pid'), None
    if state['status'] == 'running':
        command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True).strip()
        if 'plan_recent_refit.py run' not in command or root.name not in command:
            raise ValueError('Observed refit process does not match')
    out.mkdir()
    base = Path(__file__).resolve().parents[1]
    shutil.copytree(base/'src/quant_research', out/'code/src/quant_research', ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['verify_plan_recent_refit.py', 'verify_recent_refit_fit.py', 'verify_industry_decisions.py',
                 'verify_plan_prequential.py', 'verify_plan_asof_fit.py', 'verify_plan_risk_decision.py']:
        shutil.copy2(base/'scripts'/name, out/'code/scripts'/name)
    shutil.copy2(base/'uv.lock', out/'code/uv.lock')
    hashes = {str(p.relative_to(out)): file_hash(p) for p in (out/'code').rglob('*') if p.is_file()}
    write_json(out/'launch.json', dict(pid=os.getpid(), training_pid=pid, observed_command=command,
        experiment=str(root), started_at=utc_now(), verification_sources=hashes))
    marker = out/'status.json'
    try:
        while read(root/'run-status.json')['status'] == 'running':
            observed = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
            if observed.returncode or observed.stdout.strip() != command:
                if read(root/'run-status.json')['status'] == 'completed':
                    break
                raise RuntimeError('Observed refit disappeared without completion')
            write_json(marker, dict(status='waiting_for_live_refit', observed_pid=pid, updated_at=utc_now()))
            time.sleep(10)
        if read(root/'run-status.json')['status'] != 'completed':
            raise RuntimeError('Refit failed; no automatic restart')
        for name, expected in hashes.items():
            if file_hash(out/name) != expected:
                raise ValueError('Frozen verification source changed')
        write_json(marker, dict(status='verifying', updated_at=utc_now()))
        with (out/'verification.log').open('x') as log:
            subprocess.run([sys.executable, str(out/'code/scripts/verify_plan_recent_refit.py'), '--output', str(root)],
                env={**os.environ, 'PYTHONPATH': str(out/'code/src')}, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not read(root/'verification.json')['passed']:
            raise ValueError('Full refit verification did not pass')
        write_json(marker, dict(status='completed', updated_at=utc_now(), verification_sha256=file_hash(root/'verification.json')))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


if __name__ == '__main__':
    main()
