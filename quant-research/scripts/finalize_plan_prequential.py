"""Audit the observed rolling run on completion, without restarting training."""
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root,out=a.experiment.resolve(),a.output.resolve()
    state=read(root/'run-status.json')
    if state['status'] not in ['running','completed']:
        raise ValueError('Training is not live or completed')
    pid=state.get('pid')
    expected=None
    if state['status']=='running':
        expected=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True).strip()
        if 'plan_prequential_run.py run' not in expected or str(root) not in expected:
            raise ValueError('Registered training process does not match observation')
    out.mkdir()
    base=Path(__file__).resolve().parents[1]
    shutil.copytree(base/'src/quant_research',out/'code/src/quant_research',ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['verify_plan_prequential.py','verify_plan_asof_fit.py']:
        shutil.copy2(base/'scripts'/name,out/'code/scripts'/name)
    snapshot={str(p.relative_to(out)):file_hash(p) for p in (out/'code').rglob('*') if p.is_file()}
    write_json(out/'launch.json',dict(pid=os.getpid(),training_pid=pid,observed_command=expected,
        experiment=str(root),started_at=utc_now(),verification_sources=snapshot))
    marker=out/'status.json'
    try:
        while read(root/'run-status.json')['status']=='running':
            observed=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
            if observed.returncode or observed.stdout.strip()!=expected:
                if read(root/'run-status.json')['status']=='completed':
                    break
                raise RuntimeError('Observed training disappeared without a completed state')
            write_json(marker,dict(status='waiting_for_live_training',observed_pid=pid,updated_at=utc_now()))
            time.sleep(10)
        if read(root/'run-status.json')['status']!='completed':
            raise RuntimeError('Training failed; completed evidence preserved, no automatic restart')
        for name,h in snapshot.items():
            if file_hash(out/name)!=h:
                raise ValueError('Frozen verification implementation changed')
        write_json(marker,dict(status='verifying',updated_at=utc_now()))
        with (out/'verification.log').open('x') as log:
            subprocess.run([sys.executable,str(out/'code/scripts/verify_plan_prequential.py'),'--output',str(root)],
                env={**os.environ,'PYTHONPATH':str(out/'code/src')},stdout=log,stderr=subprocess.STDOUT,check=True)
        if not read(root/'verification.json')['passed']:
            raise ValueError('Full audit did not pass')
        write_json(marker,dict(status='completed',updated_at=utc_now(),verification_sha256=file_hash(root/'verification.json')))
    except BaseException as exc:
        write_json(marker,dict(status='failed',error=repr(exc),updated_at=utc_now()))
        raise


if __name__=='__main__':
    main()
