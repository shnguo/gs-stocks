"""Run the preregistered opportunity diagnostic after the live rolling audit."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['experiment','finalizer','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    root,finalizer,out=[x.resolve() for x in [a.experiment,a.finalizer,a.output]]
    rolling=Path(read(root/'parents.json')['rolling']['root'])
    launch=read(finalizer/'launch.json')
    if Path(launch['experiment'])!=rolling:
        raise ValueError('Finalizer belongs to another rolling experiment')
    status=read(finalizer/'status.json')['status']
    if status=='failed':
        raise ValueError('Rolling verification has failed')
    pid=launch['pid']
    command=None
    if status!='completed':
        command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True).strip()
        if 'finalize_plan_prequential.py' not in command or rolling.name not in command:
            raise ValueError('Observed rolling finalizer differs')
    out.mkdir()
    verifier=Path(__file__).resolve().parent/'verify_plan_opportunity.py'
    shutil.copy2(verifier,out/'verify_plan_opportunity.py')
    verification_hash=file_hash(out/'verify_plan_opportunity.py')
    frozen_hash=file_hash(root/'frozen-manifest.json')
    write_json(out/'launch.json',dict(pid=os.getpid(),observed_finalizer_pid=pid,observed_command=command,
        experiment=str(root),finalizer=str(finalizer),registered_at=utc_now(),
        frozen_diagnostic_sha256=frozen_hash,verifier_sha256=verification_hash))
    marker=out/'status.json'
    try:
        while read(finalizer/'status.json')['status']!='completed':
            if read(finalizer/'status.json')['status']=='failed':
                raise RuntimeError('Rolling audit failed; no diagnostic or retraining started')
            observed=subprocess.run(['ps','-p',str(pid),'-o','command='],text=True,capture_output=True)
            if observed.returncode or observed.stdout.strip()!=command:
                if read(finalizer/'status.json')['status']=='completed':
                    break
                raise RuntimeError('Observed rolling finalizer disappeared')
            write_json(marker,dict(status='waiting_for_live_rolling_audit',observed_pid=pid,updated_at=utc_now()))
            time.sleep(10)
        if file_hash(root/'frozen-manifest.json')!=frozen_hash or file_hash(out/'verify_plan_opportunity.py')!=verification_hash:
            raise ValueError('Registered diagnostic or verifier changed')
        write_json(marker,dict(status='running_diagnostic',updated_at=utc_now()))
        with (out/'diagnostic.log').open('x') as log:
            subprocess.run([sys.executable,str(root/'implementation.py'),'run','--output',str(root)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(marker,dict(status='verifying_diagnostic',updated_at=utc_now()))
        with (out/'verification.log').open('x') as log:
            subprocess.run([sys.executable,str(out/'verify_plan_opportunity.py'),'--output',str(root)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        if not read(root/'verification.json')['passed']:
            raise ValueError('Diagnostic verification did not pass')
        write_json(marker,dict(status='completed',updated_at=utc_now(),verification_sha256=file_hash(root/'verification.json')))
    except BaseException as exc:
        write_json(marker,dict(status='failed',error=repr(exc),updated_at=utc_now()))
        raise


if __name__=='__main__':
    main()
