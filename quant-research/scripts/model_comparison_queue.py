"""Wait for the observed live training process, verify it, then run Kronos."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['transformer', 'kronos', 'kronos-python', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    tf, kr, out = args.transformer.resolve(), args.kronos.resolve(), args.output.resolve()
    state = read(tf/'run-status.json')
    if state['status'] not in ['running', 'completed']:
        raise ValueError('Training is terminal without completion')
    pid = state.get('pid')
    expected = None
    if state['status'] == 'running':
        expected = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True).strip()
        if 'plan_transformer_run.py run' not in expected or str(tf.name) not in expected:
            raise ValueError('Training PID identity does not match')
    out.mkdir()
    write_json(out/'launch.json', dict(pid=os.getpid(), transformer_pid=pid, transformer_command=expected,
        transformer=str(tf), kronos=str(kr), started_at=utc_now()))
    marker = out/'status.json'
    try:
        while read(tf/'run-status.json')['status'] == 'running':
            process = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], text=True, capture_output=True)
            if process.returncode or process.stdout.strip() != expected:
                # Re-read final status once after authoritative process disappearance.
                if read(tf/'run-status.json')['status'] == 'completed':
                    break
                raise RuntimeError('Observed training process exited without completed evidence')
            write_json(marker, dict(status='waiting_for_live_training', observed_pid=pid, updated_at=utc_now()))
            time.sleep(10)
        if read(tf/'run-status.json')['status'] != 'completed':
            raise RuntimeError('Training failed; no automatic retraining')
        write_json(marker, dict(status='verifying_transformer', updated_at=utc_now()))
        base = Path(__file__).resolve().parents[1]
        verifier = base/'scripts/verify_plan_transformer.py'
        write_json(out/'verification-source.json', dict(path=str(verifier), sha256=file_hash(verifier)))
        with (out/'transformer-verification.log').open('x') as log:
            subprocess.run([sys.executable, str(verifier), '--output', str(tf)], stdout=log,
                stderr=subprocess.STDOUT, check=True)
        write_json(marker, dict(status='running_kronos', updated_at=utc_now()))
        with (out/'kronos.log').open('x') as log:
            subprocess.run([str(args.kronos_python.absolute()), str(kr/'code/scripts/plan_kronos_run.py'),
                'run', '--output', str(kr)], stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(marker, dict(status='completed_inference_pending_kronos_audit', updated_at=utc_now()))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


if __name__ == '__main__':
    main()
