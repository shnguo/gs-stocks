"""Independent prices/ranks, exposure, frozen normalization and forecast replay audit."""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_ranking_incremental import optimizer_steps, parent_checkpoint
from tokenizer_reconstruction_run import load_decoder
from verify_token_ranking import independent_stats

from quant_research.daily_loop import digest, read, verify
from quant_research.daily_token import load_arrays
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_cached_inference import decode_cached, generate_cached
from quant_research.token_transformer import restore_model

BASE = Path(__file__).resolve().parents[1]


def audit(root):
    verify(root)
    binding = read(root/'prepared/binding.json')
    cfg = binding['config']
    source = BASE/cfg['daily_run']
    verify(source)
    assert file_hash(source/'manifest.json') == binding['daily_manifest']
    for path, expected in read(root/'prepared/live-state.json').items():
        assert file_hash(Path(path)) == expected, path
    result = read(root/'result.json')
    info = read(root/'prepared/input.json')
    timely = bool(cfg.get('parent_root') and pd.Timestamp(result['completed_at']) < pd.Timestamp(info['horizon_dates'][0]+'T09:15:00', tz='Asia/Shanghai'))
    assert result['prospective'] == timely
    rows = pd.read_parquet(root/'prepared/forecast-rows.parquet')
    training = pd.read_parquet(root/'prepared/training-rows.parquet')
    pd.testing.assert_frame_equal(training, pd.read_parquet(source/'inputs/training-rows.parquet'))
    assert training.label_end.max() <= result['signal']
    info = read(root/'prepared/input.json')
    n, checked = len(rows), 0
    batches = {}
    arrays = load_arrays(source/'inputs', 'forecast')
    ids = np.load(root/'prepared/forecast-ids.npy')
    source_rows = pd.read_parquet(source/'inputs/forecast-rows.parquet')
    pd.testing.assert_frame_equal(rows, source_rows.iloc[ids].reset_index(drop=True))
    size = min(cfg['forecast_batch'], n)
    ix = ids[:size]
    torch.set_num_threads(4)
    spec = read(BASE/cfg['experiment']/'profiles.json')['decoder']
    assert file_hash(Path(spec['path'])) == spec['sha256']
    decoder = load_decoder(spec['path'], cfg['device'])
    features = np.load(root/'prepared/forecast-features.npy')
    a, b = [torch.tensor(np.array(arrays[k][ix]), device=cfg['device'], dtype=torch.long) for k in ['s1', 's2']]
    stamps = torch.tensor(np.array(arrays['stamps']), device=cfg['device'], dtype=torch.long)[None].expand(size, -1, -1)
    for arm in ['baseline', 'indicators_ranking']:
        models = []
        for seed in cfg['seeds']:
            dest = root/f'seed{seed}'/arm
            log = read(dest/'training/training.json')
            batches.setdefault(seed, log['batches_sha256'])
            assert batches[seed] == log['batches_sha256']
            assert log['rows'] == len(training) and log['epochs'] == 1 and log['reload_exact']
            parent = parent_checkpoint(cfg, seed, arm)
            assert log['parent_sha256'] == file_hash(parent)
            original, parent_saved = restore_model(parent, 'cpu')
            model, saved = restore_model(dest/'training/model.pt', cfg['device'])
            assert saved['trained_labels_through'] == info['observed_labels_through']
            assert saved['trained_signal_through'] == info['new_signal_through']
            assert saved['incremental_protocol_sha256'] == digest(cfg)
            assert saved['parent_sha256'] == file_hash(parent)
            if cfg.get('parent_root'):
                before = optimizer_steps(parent_saved['daily_optimizer'])
                after = optimizer_steps(saved['daily_optimizer'])
                assert before == log['optimizer_steps_before']
                assert after == log['optimizer_steps_after']
                assert set(before) == set(after)
                assert all(after[k] == before[k]+log['steps'] for k in before)
                assert len(before) > 0
            changed = 0
            for name, value in original.state_dict().items():
                actual = model.state_dict()[name].cpu()
                if 'auxiliary_center' in name or 'auxiliary_scale' in name:
                    torch.testing.assert_close(actual, value, atol=0, rtol=0)
                changed += int(not torch.equal(actual, value))
            assert changed > 0
            paths = np.load(dest/'forecast/paths.npy', mmap_mode='r')
            assert paths.shape == (n, cfg['samples'], 5, 6)
            aux = torch.tensor(features[:size], device=cfg['device']) if arm != 'baseline' else None
            pairs = generate_cached(model, a, b, stamps[:, :60], stamps[:, 60:], samples=cfg['samples'], seed=17,
                temperature=1., top_p=1., top_k=0, history_auxiliary=aux)
            replay, _ = decode_cached(decoder, pairs, arrays['mean'][ix], arrays['scale'][ix], 5)
            np.testing.assert_array_equal(replay, paths[:size])
            models.append(paths)
            del original, model
            gc.collect()
            if cfg['device'] == 'mps':
                torch.mps.empty_cache()
        stats = [independent_stats(p, cfg['minimum_legal_paths']) for p in models]
        okay = np.logical_and.reduce([s[0] for s in stats])
        day = np.mean([s[1] for s in stats], axis=0).argmax(1)+1
        buy = np.mean([s[2] for s in stats], axis=0)
        sell = np.mean([s[3] for s in stats], axis=0)[np.arange(n), day-1]
        expected = pd.DataFrame(dict(instrument_id=rows.instrument_id[okay], expected_net_return=(sell/buy-1-cfg['cost'])[okay],
            buy_reference_price=buy[okay], sell_reference_price=sell[okay],
            sell_reference_date=np.asarray(info['horizon_dates'])[day[okay]]))
        expected = expected.sort_values(['expected_net_return', 'instrument_id'], ascending=[False, True])
        actual = pd.read_csv(root/f'{arm}-ranking.csv')
        np.testing.assert_array_equal(actual.instrument_id, expected.instrument_id)
        np.testing.assert_array_equal(actual['rank'], np.arange(1, len(actual)+1))
        np.testing.assert_array_equal(actual.sell_reference_date, expected.sell_reference_date)
        assert actual.buy_date.eq(info['horizon_dates'][0]).all()
        for key in ['expected_net_return', 'buy_reference_price', 'sell_reference_price']:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
        checked += len(actual)
    receipt = dict(passed=True, rankings_checked=checked, replayed_inputs=size*len(cfg['seeds'])*2,
        training_rows_per_fit=len(training), fits=2*len(cfg['seeds']), late_excluded=not timely,
        main_unchanged=True, run_manifest_sha256=file_hash(root/'manifest.json'),
        code_sha256=file_hash(Path(__file__)), at=utc_now())
    write_json(root.with_name(root.name+'-verification.json'), receipt)
    print(receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'configs/token-ranking-incremental-v1.json')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only:
        audit(BASE/read(args.config)['output'])
