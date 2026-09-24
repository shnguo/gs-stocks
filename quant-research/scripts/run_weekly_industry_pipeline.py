"""Bounded weekly source capture followed by the full frozen six-window contrast."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from quant_research.storage import file_hash, utc_now, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['industry', 'binary', 'prior', 'experiment', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    out, base = a.output.resolve(), Path(__file__).resolve().parent
    out.mkdir()
    sources = {n: file_hash(base/n) for n in ['industry_history_recover.py', 'industry_ablation_run.py',
        'verify_industry_ablation.py', 'plan_value_run.py', 'price_pilot.py', 'model_quality_run.py']}
    module_root = base.parent/'src/quant_research'
    modules = {str(f.relative_to(module_root)): file_hash(f) for f in module_root.rglob('*.py')}
    industry = a.industry.resolve()
    protocol_hash = file_hash(industry/'protocol.json')
    write_json(out/'launch.json', dict(pid=os.getpid(), started_at=utc_now(), source_hashes=sources,
        module_hashes=modules, industry=str(industry), experiment=str(a.experiment.resolve()),
        binary_sha256=file_hash(a.binary), protocol_sha256=protocol_hash))
    steps = [
        ('capture', [sys.executable, str(base/'industry_history_recover.py'), '--root', str(industry),
            '--binary', str(a.binary.resolve()), '--seed-index', str(industry/'seed-index.json'), '--max-attempts', '12']),
        ('freeze', [sys.executable, str(base/'industry_ablation_run.py'), 'freeze', '--prior', str(a.prior.resolve()),
            '--industry', str(industry), '--protocol', str(industry/'protocol.json'), '--output', str(a.experiment.resolve())]),
        ('train', [sys.executable, str(base/'industry_ablation_run.py'), 'run', '--output', str(a.experiment.resolve())]),
        ('verify', [sys.executable, str(base/'verify_industry_ablation.py'), '--output', str(a.experiment.resolve())]),
    ]
    try:
        for name, command in steps:
            for source, expected in sources.items():
                if file_hash(base/source) != expected:
                    raise ValueError('Pipeline source changed after registration')
            for source, expected in modules.items():
                if file_hash(module_root/source) != expected:
                    raise ValueError('Model modules changed after registration')
            if file_hash(industry/'protocol.json') != protocol_hash:
                raise ValueError('Industry protocol changed')
            write_json(out/'status.json', dict(status=name, pid=os.getpid(), updated_at=utc_now()))
            with (out/f'{name}.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        verification = json.loads((a.experiment/'verification.json').read_text())
        if not verification['passed']:
            raise ValueError('Industry verification did not pass')
        write_json(out/'status.json', dict(status='completed', updated_at=utc_now()))
    except BaseException as exc:
        write_json(out/'status.json', dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


if __name__ == '__main__':
    main()
