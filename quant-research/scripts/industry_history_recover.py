"""Research recovery supervisor; all financial acquisition stays in Rust."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from quant_research.industry_context import read_snapshot
from quant_research.storage import file_hash, utc_now, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--max-attempts', type=int, default=12)
    p.add_argument('--run-name', default='recovery-supervisor')
    p.add_argument('--seed-index', type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    if not args.run_name.startswith('recovery-supervisor') or '/' in args.run_name or '.' in args.run_name:
        raise ValueError('Recovery run name must be a simple recovery-supervisor suffix')
    lock = (root/'recovery.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for other in root.glob('recovery-supervisor*.json'):
        state = json.loads(other.read_text())
        if state.get('status') == 'running' and state.get('pid'):
            try:
                os.kill(state['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Another recovery supervisor is live')
    marker = root/f'{args.run_name}.json'
    if marker.exists():
        raise FileExistsError('Preserve previous recovery supervisor attempts')
    dates = json.loads((root/'training-dates.json').read_text())
    complete = {}
    if args.seed_index:
        seed = json.loads(args.seed_index.read_text())
        for day, directory in seed.items():
            if day not in dates:
                raise ValueError('Seed date is outside the registered capture plan')
            read_snapshot(directory, day)
            complete[day] = str(Path(directory).resolve())
    # A saved running marker is only evidence to inspect the PID, never to restart.
    for batch in sorted(root.glob('batch-v*')):
        if not batch.is_dir():
            continue
        state = json.loads((batch/'progress.json').read_text())
        if state['status'] == 'running':
            try:
                os.kill(state['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('An industry collector is still live')
        for date in dates:
            directory = batch/date
            if not (directory/'manifest.json').exists():
                continue
            manifest = json.loads((directory/'manifest.json').read_text())
            if manifest['status'] == 'completed':
                read_snapshot(directory, date)
                complete[date] = str(directory)
    launch = dict(status='running', pid=os.getpid(), started_at=utc_now(),
        binary=str(args.binary.resolve()), binary_sha256=file_hash(args.binary),
        supervisor_source_sha256=file_hash(Path(__file__)))
    write_json(root/f'{args.run_name}-launch.json', {k: v for k, v in launch.items() if k != 'status'})
    write_json(marker, launch)
    zero_progress = 0
    for _ in range(args.max_attempts):
        remaining = [d for d in dates if d not in complete]
        if not remaining:
            break
        version = 1
        while (root/f'batch-v{version}').exists():
            version += 1
        plan, dest = root/f'remaining-v{version}.json', root/f'batch-v{version}'
        write_json(plan, remaining)
        write_json(root/'verified-snapshots.json', complete)
        print(f'Starting {dest.name}: {len(remaining)} remaining; {len(complete)} verified', flush=True)
        with (root/f'batch-v{version}.log').open('x') as log:
            result = subprocess.run([str(args.binary.resolve()), 'research-capture-baostock-industry-batch',
                str(plan), str(dest)], stdout=log, stderr=subprocess.STDOUT)
        # wait returned a terminal exit code; only now inspect and recover missing dates.
        before = len(complete)
        state = json.loads((dest/'progress.json').read_text())
        for date in remaining:
            directory = dest/date
            if (directory/'manifest.json').exists():
                manifest = json.loads((directory/'manifest.json').read_text())
                if manifest['status'] == 'completed':
                    read_snapshot(directory, date)
                    complete[date] = str(directory)
        write_json(root/'verified-snapshots.json', complete)
        zero_progress = zero_progress+1 if len(complete) == before else 0
        write_json(marker, dict(status='running', pid=os.getpid(), completed=len(complete), total=len(dates),
            last_batch=str(dest), child_exit_code=result.returncode, last_status=state, updated_at=utc_now()))
        if zero_progress >= 3:
            break
        if result.returncode:
            time.sleep(5)
    done = len(complete) == len(dates)
    write_json(root/'verified-snapshots.json', complete)
    write_json(marker, dict(status='completed' if done else 'incomplete', completed=len(complete),
        total=len(dates), consecutive_no_progress=zero_progress, updated_at=utc_now()))
    if not done:
        raise SystemExit('Bounded recovery stopped; preserved captures and remaining scope')


if __name__ == '__main__':
    main()
