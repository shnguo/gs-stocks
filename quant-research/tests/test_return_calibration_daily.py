import json
from pathlib import Path

import pandas as pd
import pytest
from token_return_calibration_daily import mature_publications

from quant_research.daily_loop import write_manifest
from quant_research.storage import write_json


def publication(store, day, dates, *, prospective=True, cost=.0025):
    root = store/'runs'/day
    write_json(root/'publication.json', dict(signal_date=day, horizon_dates=dates,
        prospective=prospective, cost=cost))
    pd.DataFrame([dict(instrument_id='cn.xshg.600000', expected_net_return=.15,
        volatility_pp=2., buy_date=dates[0], sell_reference_date=dates[1])]).to_csv(root/'raw-ranking.csv', index=False)
    write_manifest(root)
    return root


def test_daily_supervision_excludes_late_pending_and_unknown_and_uses_selected_exit(tmp_path):
    dates = pd.bdate_range('2026-01-01', periods=6).strftime('%Y-%m-%d').tolist()
    root = publication(tmp_path, dates[0], dates[1:])
    bars = pd.DataFrame(dict(instrument_id=['cn.xshg.600000']*6, date=dates,
        open=[10.]*6, high=[11., 11., 12., 13., 14., 15.], low=[9.]*6, close=[10.]*6,
        volume=[100.]*6, amount=[1000.]*6, factor=[1.]*6,
        sequence_id=['one']*6, label_sequence_id=['one']*6, source_trade_status=[1]*6))
    added, lineage = mature_publications(tmp_path, bars, '2026-02-01', .0025)
    assert len(added) == 1 and len(lineage) == 1
    assert added.iloc[0].actual_extrema_scenario == pytest.approx(12/9-1-.0025)
    assert added.iloc[0].predicted == .15
    assert mature_publications(tmp_path, bars, dates[-1], .0025)[0].empty
    suspended = bars.copy()
    suspended.loc[2, 'source_trade_status'] = 0
    assert mature_publications(tmp_path, suspended, '2026-02-01', .0025)[0].empty
    info = json.loads((root/'publication.json').read_text())
    write_json(root/'publication.json', dict(info, prospective=False))
    write_manifest(root)
    assert mature_publications(tmp_path, bars, '2026-02-01', .0025)[0].empty
    write_json(root/'publication.json', dict(info, cost=.01))
    write_manifest(root)
    with pytest.raises(ValueError, match='cost differs'):
        mature_publications(tmp_path, bars, '2026-02-01', .0025)


def test_manual_calibration_flag_requires_matched_model():
    import os
    import subprocess
    import sys
    base = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(base/'scripts/daily_token_cycle.py'),
                             '--calibration-config', 'does-not-exist.json'],
                            cwd=base, env=dict(os.environ, PYTHONPATH='src:scripts'),
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert '--calibration-config requires --matched-config' in result.stderr


def test_publication_reuses_identical_artifact_and_rejects_changed_head(tmp_path, monkeypatch):
    """Exercise daily publishing end to end with tiny but hash-verified artifacts."""
    import token_return_calibration_daily as module

    from quant_research.daily_loop import read
    from quant_research.return_calibration import FEATURES
    from quant_research.storage import file_hash

    monkeypatch.setattr(module, 'BASE', tmp_path)
    cfg = dict(output='fitted', source='source', daily_store='daily', arm='indicators_ranking',
               cost=.0025, top_n=20, minimum_training_dates=20, ridge=.01)
    config = tmp_path/'config.json'
    write_json(config, cfg)
    fitted, matched = tmp_path/'fitted', tmp_path/'matched'
    protocol = dict(prior='experiment', samples=64, seeds=[17, 29, 43],
                    minimum_legal_paths=16, cost=.0025, return_ranking_loss={})
    write_json(tmp_path/'source/protocol.json', protocol)
    write_json(fitted/'binding.json', dict(config=cfg))
    write_json(fitted/'models/ensemble.json', dict(version='return-calibration-v1',
        features=list(FEATURES), center=[0.]*3, scale=[1.]*3, coefficients=[-1., 0., 0., 0.],
        as_of='2025-01-01', trained_labels_through='2024-12-31'))
    write_manifest(fitted/'models')
    write_json(fitted/'verification.json', dict(passed=True))
    write_json(fitted/'completed.json', dict(passed=True,
        verification_sha256=file_hash(fitted/'verification.json'), model_sha256=file_hash(fitted/'models/manifest.json')))
    signal = '2026-01-30'
    dates = pd.bdate_range(end=signal, periods=21).strftime('%Y-%m-%d').tolist()
    horizons = ['2026-02-02', '2026-02-03', '2026-02-04', '2026-02-05', '2026-02-06']
    data = tmp_path/'data'
    write_json(data/'snapshot/manifest.json', dict(files={}))
    write_json(matched/'prepared/input.json', dict(source=str(data), signal_date=signal,
        snapshot_sha256=file_hash(data/'snapshot/manifest.json'), horizon_dates=horizons))
    write_json(matched/'prepared/binding.json', dict(config=dict(protocol, experiment='experiment')))
    pd.DataFrame([dict(instrument_id='cn.xshg.600000', name='Example', rank=1,
        expected_net_return=.1975, buy_reference_price=10., sell_reference_price=12.,
        buy_date=horizons[0], sell_reference_date=horizons[1])]).to_csv(matched/'indicators_ranking-ranking.csv', index=False)
    write_manifest(matched)
    write_json(matched.with_name('matched-verification.json'), dict(passed=True, run_manifest_sha256=file_hash(matched/'manifest.json')))
    bars = pd.DataFrame(dict(date=dates, instrument_id=['cn.xshg.600000']*21, close=[10.]*21, factor=[1.]*21))
    monkeypatch.setattr(module, 'panel', lambda _: (dict(price_data_through=signal, dates=dates), bars, None))
    monkeypatch.setattr('token_ranking_shadow.review_shadow', lambda *args: [])
    # The binding snapshots this actual module plus the head implementation.
    (tmp_path/'src/quant_research').mkdir(parents=True)
    (tmp_path/'src/quant_research/return_calibration.py').write_text('fixture implementation')
    report = module.publish(matched, config)
    frame = pd.read_csv(report.parent/'calibrated-ranking.csv')
    assert frame.expected_net_return.iloc[0] == pytest.approx(.1875)
    assert frame.reference_price_net_return.iloc[0] == .1975
    assert frame.sell_reference_price.iloc[0] == 12.
    assert '历史补发' in report.read_text()
    assert module.publish(matched, config) == report
    write_json(fitted/'models/ensemble.json', dict(read(fitted/'models/ensemble.json'), coefficients=[-2., 0., 0., 0.]))
    with pytest.raises(ValueError, match='hash mismatch'):
        module.publish(matched, config)
