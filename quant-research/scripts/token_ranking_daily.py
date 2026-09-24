"""Manual paired candidate continuation with immutable lineage and mature-outcome review."""
import argparse
import fcntl
from pathlib import Path

import torch
from token_ranking_incremental import ARMS, BASE, forecast, prepare, report, train
from token_ranking_shadow import review_shadow
from tokenizer_reconstruction_run import load_decoder
from verify_token_ranking_incremental import audit

from quant_research.daily_loop import digest, read, verify
from quant_research.storage import file_hash, utc_now, write_json


def pending_daily_runs(target, parent_daily, main_store):
    """Follow actual verified update ancestry; never silently jump over an update."""
    parent_daily = Path(parent_daily).resolve()
    run, pending, visited = Path(target).resolve(), [], set()
    while run != parent_daily:
        if run in visited or run.parent != (Path(main_store)/'runs').resolve():
            raise ValueError('Invalid or cyclic daily run ancestry')
        visited.add(run)
        verify(run)
        info, binding = read(run/'run.json'), read(run/'binding.json')
        previous = binding.get('previous_state')
        if not previous or previous['signal_date'] >= info['signal_date']:
            raise ValueError('Target does not descend from the comparison state')
        prior = Path(previous['run']).resolve()
        if file_hash(prior/'manifest.json') != previous['manifest_sha256']:
            raise ValueError('Changed previous daily manifest')
        verify(prior)
        actual = read(prior/'run.json')
        if previous['signal_date'] != actual['signal_date'] or previous['checkpoints'] != actual['checkpoints']:
            raise ValueError('Previous daily state differs from its immutable run')
        if binding['checkpoints'] != actual['checkpoints']:
            raise ValueError('Daily update checkpoint lineage differs')
        pending.append(run)
        run = prior
    verify(parent_daily)
    return list(reversed(pending))


def compatible_protocol(bootstrap, cfg):
    for key in ['seeds', 'device', 'samples', 'forecast_batch', 'minimum_legal_paths', 'cost',
                'return_ranking_loss', 'price_input', 'midpoint_loss_weight', 'experiment',
                'historical_replay', 'learning_rate', 'training_batch']:
        if cfg[key] != bootstrap[key]:
            raise ValueError('Continuation protocol differs from matched bootstrap: '+key)


def ensure_audit(root):
    receipt_path = root.with_name(root.name+'-verification.json')
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get('passed') and receipt['run_manifest_sha256'] == file_hash(root/'manifest.json'):
            return
    audit(root)


def publish_pointer(store, root, cfg):
    verify(root)
    receipt = read(root.with_name(root.name+'-verification.json'))
    if not receipt['passed'] or receipt['run_manifest_sha256'] != file_hash(root/'manifest.json'):
        raise ValueError('Independent audit required before advancing comparison state')
    info = read(root/'prepared/input.json')
    pointer = dict(signal_date=info['signal_date'], root=str(root.resolve()),
        trained_signal_through=info['new_signal_through'], manifest_sha256=file_hash(root/'manifest.json'),
        config_sha256=digest(cfg), report=str((root/'report.md').resolve()))
    state = store/'current.json'
    if state.exists() and read(state)['signal_date'] > pointer['signal_date']:
        raise ValueError('Cannot regress the comparison pointer')
    write_json(state, pointer)


def run_daily_comparison(config, target=None):
    cfg = read(config)
    store, main_store = Path(cfg['store']), Path(cfg['main_store'])
    bootstrap = Path(cfg['bootstrap'])
    store.mkdir(parents=True, exist_ok=True)
    with (store/'cycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(bootstrap)
        if file_hash(bootstrap/'manifest.json') != cfg['bootstrap_sha256']:
            raise ValueError('Changed bootstrap')
        initial = read(bootstrap/'prepared/binding.json')['config']
        compatible_protocol(initial, cfg)
        target = Path(target or read(main_store/'current.json')['run']).resolve()
        verify(target)
        parent = bootstrap
        if (store/'current.json').exists():
            state = read(store/'current.json')
            if state['config_sha256'] != digest(cfg):
                raise ValueError('Changed comparison configuration; use a new store')
            parent = Path(state['root'])
            verify(parent)
            if file_hash(parent/'manifest.json') != state['manifest_sha256']:
                raise ValueError('Changed comparison parent')
        parent_cfg = read(parent/'prepared/binding.json')['config']
        parent_daily = BASE/parent_cfg['daily_run']
        # Reusing an older completed date must not move the current pointer backwards.
        old = store/'runs'/read(target/'run.json')['signal_date']
        if (old/'manifest.json').exists():
            verify(old)
            old_binding = read(old/'prepared/binding.json')
            if old_binding['daily_manifest'] != file_hash(target/'manifest.json') or old_binding['config']['chain_config_sha256'] != digest(cfg):
                raise ValueError('Completed comparison binding differs')
            ensure_audit(old)
            if read(old/'prepared/input.json')['signal_date'] >= read(parent/'prepared/input.json')['signal_date']:
                publish_pointer(store, old, cfg)
            review_source = read(parent_daily/'inputs/input.json')['source'] if parent_daily.name > target.name else read(target/'inputs/input.json')['source']
            review_shadow(store, Path(review_source))
            return old/'report.md'
        pending = pending_daily_runs(target, parent_daily, main_store)
        if not pending:
            review_shadow(store, Path(read(target/'inputs/input.json')['source']))
            return parent/'report.md'
        torch.set_num_threads(4)
        spec = read(BASE/cfg['experiment']/'profiles.json')['decoder']
        if file_hash(Path(spec['path'])) != spec['sha256']:
            raise ValueError('Changed decoder')
        decoder = load_decoder(spec['path'], cfg['device'])
        for daily in pending:
            signal = read(daily/'run.json')['signal_date']
            root = store/'runs'/signal
            input_info = read(daily/'inputs/input.json')
            previous = read(parent/'prepared/input.json')
            source_previous = read(daily/'binding.json')['previous_state']
            if source_previous['trained_signal_through'] != previous['new_signal_through']:
                raise ValueError('Incremental input cutoff differs from matched parent')
            protocol = {k: v for k, v in cfg.items() if k not in ['store', 'main_store', 'bootstrap', 'bootstrap_sha256']}
            protocol.update(parent_root=str(parent.resolve()), daily_run=str(daily), output=str(root), chain_config_sha256=digest(cfg))
            if not (root/'manifest.json').exists():
                prepare(daily, root, protocol)
                for seed in cfg['seeds']:
                    for arm in ARMS:
                        train(daily, root, protocol, seed, arm, decoder)
                        forecast(daily, root, protocol, seed, arm, decoder)
                report(root, daily, protocol)
            else:
                verify(root)
                existing = read(root/'prepared/binding.json')
                if existing['config'] != protocol or existing['daily_manifest'] != file_hash(daily/'manifest.json'):
                    raise ValueError('Completed intermediate run binding differs')
            ensure_audit(root)
            publish_pointer(store, root, cfg)
            review_shadow(store, Path(input_info['source']))
            parent = root
        write_json(store/'last-status.json', dict(status='completed', at=utc_now(), run=str(parent)))
        return parent/'report.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'configs/token-ranking-daily-v1.json')
    parser.add_argument('--run', type=Path, help='Existing completed daily run; defaults to the current daily pointer')
    args = parser.parse_args()
    print(run_daily_comparison(args.config, args.run), flush=True)
