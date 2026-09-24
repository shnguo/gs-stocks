"""Finish historical industry capture and its frozen model contrast after the live collector."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['industry', 'prior', 'operator', 'protocol', 'experiment', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    industry, out = args.industry.resolve(), args.output.resolve()
    prior, experiment = args.prior.resolve(), args.experiment.resolve()
    operator = args.operator.resolve()
    if not read(operator.parent/'smoke-verification.json')['passed']:
        raise ValueError('New industry operator smoke capture has not passed')
    build = read(operator.parent/'build.json')
    if file_hash(operator) != build['files'][operator.name]:
        raise ValueError('Operator differs from verified build')
    old = read(industry/'recovery-supervisor.json')
    pid, expected = old.get('pid'), None
    if old['status'] == 'running':
        expected = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True).strip()
        if 'industry_history_recover.py' not in expected or industry.name not in expected:
            raise ValueError('Observed collector supervisor identity mismatch')
    out.mkdir()
    base = Path(__file__).resolve().parent
    names = ['industry_history_recover.py', 'industry_ablation_run.py', 'verify_industry_ablation.py']
    sources = {name: file_hash(base/name) for name in names}
    write_json(out/'launch.json', dict(pid=os.getpid(), old_collector_pid=pid,
        expected_command=expected, source_hashes=sources, operator=str(operator),
        operator_sha256=file_hash(operator), started_at=utc_now()))
    marker = out/'status.json'
    try:
        while read(industry/'recovery-supervisor.json')['status'] == 'running':
            current = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], text=True, capture_output=True)
            if current.returncode or current.stdout.strip() != expected:
                # New recovery checks every batch PID and rejects any surviving child.
                break
            write_json(marker, dict(status='waiting_for_live_collector', observed_pid=pid, updated_at=utc_now()))
            time.sleep(10)
        for name, expected_hash in sources.items():
            if file_hash(base/name) != expected_hash:
                raise ValueError('Pipeline source changed after launch')
        dates = read(industry/'training-dates.json')
        completed = read(industry/'verified-snapshots.json')
        if set(completed) != set(dates):
            write_json(marker, dict(status='recovering_with_longer_deadlines', updated_at=utc_now()))
            with (out/'recovery.log').open('x') as log:
                subprocess.run([sys.executable, str(base/'industry_history_recover.py'), '--root', str(industry),
                    '--binary', str(operator), '--run-name', 'recovery-supervisor-v2', '--max-attempts', '12'],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(marker, dict(status='freezing_industry_ablation', updated_at=utc_now()))
        with (out/'freeze.log').open('x') as log:
            subprocess.run([sys.executable, str(base/'industry_ablation_run.py'), 'freeze',
                '--prior', str(prior), '--industry', str(industry), '--protocol', str(args.protocol.resolve()),
                '--output', str(experiment)], stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(marker, dict(status='training_industry_ablation', updated_at=utc_now()))
        with (out/'training.log').open('x') as log:
            subprocess.run([sys.executable, str(experiment/'code/scripts/industry_ablation_run.py'),
                'run', '--output', str(experiment)], stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(marker, dict(status='verifying_industry_ablation', updated_at=utc_now()))
        with (out/'verification.log').open('x') as log:
            subprocess.run([sys.executable, str(base/'verify_industry_ablation.py'), '--output', str(experiment)],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(marker, dict(status='completed', updated_at=utc_now()))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


if __name__ == '__main__':
    main()
