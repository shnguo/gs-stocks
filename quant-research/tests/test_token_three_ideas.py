import json

import numpy as np
import pandas as pd
import pytest
from finalize_token_three_ideas import compare, finalize, path_rows
from token_features_run import feature_schema
from token_three_ideas import cohort_rows

from quant_research.storage import file_hash, write_json
from quant_research.token_market_context import FEATURES


def test_context_schema_is_explicit_and_preserves_indicator_prefix():
    from quant_research.token_indicators import FEATURES as indicators
    names, groups = feature_schema({'feature_family': 'market_context'})
    assert names == FEATURES and names[:len(indicators)] == indicators
    assert groups['context_ranking'] == FEATURES
    assert len(names) == 24


def test_runner_accepts_decoder_object_and_runs_only_new_context_fits(tmp_path, monkeypatch):
    import finalize_token_three_ideas as report_module
    import token_three_ideas as run_module
    import torch
    cfg = dict(output=str(tmp_path), device='cpu', seeds=[17, 29, 43])
    write_json(tmp_path/'sources.json', {})
    write_json(tmp_path/'profiles.json', dict(decoder=dict(path='decoder.pt')))
    calls = []
    monkeypatch.setattr(run_module, 'prepare', lambda *a: None)
    monkeypatch.setattr(run_module, 'load_decoder', lambda *a: torch.nn.Linear(1, 1))
    monkeypatch.setattr(run_module, 'Dataset', lambda: None)
    monkeypatch.setattr(run_module, 'EvaluationDataset', lambda *a: None)
    monkeypatch.setattr(run_module, 'train', lambda r, c, d, seed, arm, dec: calls.append(('train', seed, arm)))
    monkeypatch.setattr(run_module, 'forecast', lambda r, c, d, seed, arm, dec: calls.append(('forecast', seed, arm)))
    monkeypatch.setattr(report_module, 'finalize', lambda *a: calls.append(('finalize',)))
    run_module.run(cfg)
    assert calls[:3] == [('train', s, 'context_ranking') for s in cfg['seeds']]
    assert len([c for c in calls if c[0] == 'forecast']) == 12
    assert calls[-1] == ('finalize',)


def test_full_cohort_expands_dates_without_labels_or_sample_membership():
    class Source:
        rows = pd.DataFrame(dict(date_index=[60, 61], valid=[False, True]))
    dates = pd.bdate_range('2020-01-01', periods=70).strftime('%Y-%m-%d').tolist()
    symbols = ['cn.xshe.000001', 'cn.xshg.600001', 'cn.xshe.000003']
    coords = np.array([[2, 60], [0, 61], [0, 60], [1, 60]])
    result = cohort_rows(coords, Source(), [0], dates, symbols)
    assert len(result) == 3 and set(result.stock_index) == {0, 1, 2}
    assert list(result.instrument_id) == sorted(symbols)
    with pytest.raises(ValueError, match='Duplicate'):
        cohort_rows(np.vstack([coords, coords[:1]]), Source(), [0], dates, symbols)


def test_effect_frequency_does_not_discard_positive_majority_with_wide_interval():
    rows = []
    for seed in ['17', 'ensemble']:
        for i, change in enumerate([1, 1, 1, -2, -2]):
            for arm in ['a', 'b']:
                value = change if arm == 'a' else 0
                rows.append(dict(seed=seed, arm=arm, date=str(i), return_mae_pp=10-value,
                    top_extrema_scenario_pct=value, top_lift_pp=value, top_q10_pct=value,
                    rank_ic=value/10, top_adverse_excursion_pct=value, top_prediction_bias_pp=10-value))
    result = compare(pd.DataFrame(rows), dict(seeds=[17], comparisons=[['a', 'b']],
        bootstrap_replicates=1000, bootstrap_block_dates=2))
    mae = next(r for r in result if r['seed'] == 'ensemble' and r['metric'] == 'return_mae_pp')
    assert mae['dates_improved'] == 3 and mae['date_win_fraction'] == .6
    assert mae['improvement'] == pytest.approx(-.2)
    assert mae['bootstrap_95'][0] < 0 < mae['bootstrap_95'][1]


def test_chunked_full_report_freezes_unknown_top_stock_and_detects_corruption(tmp_path):
    cfg = dict(seeds=[17, 29], variants=['baseline', 'indicators', 'indicators_ranking', 'context_ranking'],
        samples=16, chunk_rows=16, minimum_legal_paths=16, top_n=20, cost=.0025,
        comparisons=[['indicators_ranking', 'indicators'], ['context_ranking', 'indicators_ranking']],
        bootstrap_replicates=20, bootstrap_block_dates=2, limitations='Synthetic test')
    write_json(tmp_path/'protocol.json', cfg)
    write_json(tmp_path/'prepared.json', dict(files={}))
    write_json(tmp_path/'live-state.json', {})
    write_json(tmp_path/'input-parity.json', dict(passed=True))
    dataset = tmp_path/'evaluation-inputs'
    dataset.mkdir()
    rows = pd.DataFrame(dict(row_id=np.arange(44), date=np.repeat(['2024-07-02', '2025-01-02'], 22),
        instrument_id=[f'cn.xshe.{i:06}' for i in range(22)]*2))
    rows.to_parquet(dataset/'rows.parquet', index=False)
    future = np.zeros((44, 5, 6))
    future[..., :4] = [10., 12., 9., 10.]
    future[..., 4:] = [100., 1000.]
    future[:, 1, 1] = 11.  # Frozen forecast picks day 1, not actual best day 2+.
    valid = np.ones((44, 5), bool)
    valid[[21, 43]] = False
    for name, value in dict(future=future, valid=valid, volatility=np.tile(np.arange(22), 2)).items():
        np.save(dataset/f'{name}.npy', value)
    for seed in cfg['seeds']:
        for arm in cfg['variants']:
            directory = tmp_path/'forecasts'/f'seed{seed}'/arm
            directory.mkdir(parents=True)
            manifests = {}
            for start in range(0, 44, 16):
                stop = min(start+16, 44)
                paths = np.broadcast_to(future[start:stop, None], (stop-start, 16, 5, 6)).copy()
                paths[:, :, 1, 1] = 13+np.arange(start, stop)[:, None]%22/10
                p = directory/f'{start:07d}.npy'
                np.save(p, paths)
                marker = p.with_suffix('.json')
                write_json(marker, dict(start=start, end=stop, sha256=file_hash(p)))
                manifests[marker.name] = file_hash(marker)
            write_json(directory/'completed.json', dict(files=manifests,
                identity=dict(prepared_sha256=file_hash(tmp_path/'prepared.json'))))
    reordered = path_rows(tmp_path, cfg, 17, 'baseline', np.array([43, 0, 20, 16]))
    np.testing.assert_allclose(reordered[:, 0, 1, 1], [15.1, 13., 15., 14.6])
    finalize(tmp_path)
    report = pd.read_csv(tmp_path/'validation/paired-daily.csv')
    assert report.top_known.eq(19).all() and report.top_unknown.eq(1).all()
    np.testing.assert_allclose(report.top_extrema_scenario_pct, (11/9-1-.0025)*100)
    ranks = pd.read_parquet(tmp_path/'validation/paired/2024-07-02.parquet')
    assert ranks[ranks['rank'] == 1].local_row.eq(21).all()
    assert ranks[ranks['rank'] == 1].actual_extrema_scenario.isna().all()
    assert json.loads((tmp_path/'completed.json').read_text())['all_three_ideas_validated']
    p = tmp_path/'forecasts/seed17/baseline/0000000.npy'
    with p.open('ab') as stream:
        stream.write(b'corrupted')
    with pytest.raises(ValueError, match='paths changed'):
        finalize(tmp_path)
