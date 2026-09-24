"""Finalize the observed live Kronos inference without restarting any model."""
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
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--analysis-output', type=Path, required=True)
    args = p.parse_args()
    root, out = args.experiment.resolve(), args.output.resolve()
    state = read(root/'run-status.json')
    if state['status'] not in ['running', 'completed']:
        raise ValueError('Inference did not complete')
    pid = state.get('pid')
    expected = None
    if pid:
        expected = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True).strip()
        if 'plan_kronos_run.py run' not in expected or str(root) not in expected:
            raise ValueError('Observed process is not the registered inference job')
    out.mkdir()
    base = Path(__file__).resolve().parent
    sources = {name: file_hash(base/name) for name in ['verify_plan_kronos.py', 'analyze_plan_kronos.py']}
    write_json(out/'launch.json', dict(pid=os.getpid(), inference_pid=pid, expected_command=expected,
        experiment=str(root), source_hashes=sources, started_at=utc_now()))
    marker = out/'status.json'
    try:
        while read(root/'run-status.json')['status'] == 'running':
            status = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], text=True, capture_output=True)
            if status.returncode or status.stdout.strip() != expected:
                if read(root/'run-status.json')['status'] == 'completed':
                    break
                raise RuntimeError('Observed inference process disappeared without completion')
            write_json(marker, dict(status='waiting_for_live_inference', observed_pid=pid, updated_at=utc_now()))
            time.sleep(10)
        if read(root/'run-status.json')['status'] != 'completed':
            raise RuntimeError('Inference failed; preserve completed chunks, do not restart')
        for name, expected_hash in sources.items():
            if file_hash(base/name) != expected_hash:
                raise ValueError('Finalization source changed after launch')
        write_json(marker, dict(status='verifying', updated_at=utc_now()))
        with (out/'verification.log').open('x') as log:
            subprocess.run([sys.executable, str(base/'verify_plan_kronos.py'), '--output', str(root)],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(marker, dict(status='analyzing', updated_at=utc_now()))
        with (out/'analysis.log').open('x') as log:
            subprocess.run([sys.executable, str(base/'analyze_plan_kronos.py'), '--experiment', str(root),
                '--output', str(args.analysis_output.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(marker, dict(status='completed', updated_at=utc_now()))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


if __name__ == '__main__':
    main()
