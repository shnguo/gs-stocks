from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from token_ranking_daily import compatible_protocol, pending_daily_runs, publish_pointer
from token_ranking_incremental import optimizer_steps, validate_rows

from quant_research.daily_loop import digest, write_manifest
from quant_research.storage import file_hash, write_json


def continuation_rows():
    rows = pd.DataFrame(dict(instrument_id=['a', 'b', 'c'], date=['2026-09-09', '2026-09-08', '2023-12-01'],
        label_end=['2026-09-16', '2026-09-15', '2023-12-08'], role=['newly_mature', 'recent_replay', 'historical_replay']))
    arrays = dict(s1=np.zeros((3, 65)), valid=np.ones((3, 5), bool))
    info = dict(signal_date='2026-09-16', observed_labels_through='2026-09-16', new_signal_through='2026-09-09')
    prev = dict(signal_date='2026-09-15', new_signal_through='2026-09-08')
    return rows, arrays, info, prev


def test_continuation_rejects_repeated_or_mislabeled_training():
    rows, arrays, info, prev = continuation_rows()
    validate_rows(rows, arrays, info, prev)
    with pytest.raises(ValueError, match='Repeated'):
        validate_rows(rows, arrays, info, dict(prev, signal_date=info['signal_date']))
    rows.loc[1, 'role'] = 'newly_mature'
    with pytest.raises(ValueError, match='Previously consumed'):
        validate_rows(rows, arrays, info, prev)
    rows, arrays, info, prev = continuation_rows()
    rows.loc[1, 'date'] = '2026-09-09'
    with pytest.raises(ValueError, match='mislabeled replay'):
        validate_rows(rows, arrays, info, prev)


def daily(tmp, day, previous=None):
    root = tmp/'runs'/day
    root.mkdir(parents=True)
    checkpoints = {'17': {'path': str(root/'model.pt'), 'sha256': day}}
    write_json(root/'run.json', dict(signal_date=day, checkpoints=checkpoints))
    state = None
    if previous:
        from quant_research.daily_loop import read
        info = read(previous/'run.json')
        state = dict(signal_date=previous.name, run=str(previous), checkpoints=info['checkpoints'], manifest_sha256=file_hash(previous/'manifest.json'))
    write_json(root/'binding.json', dict(previous_state=state, checkpoints=state['checkpoints'] if state else {}))
    write_manifest(root)
    return root


def test_walks_every_verified_parent_and_rejects_unrelated_or_tampered_runs(tmp_path):
    a = daily(tmp_path, '2026-09-15')
    b = daily(tmp_path, '2026-09-16', a)
    c = daily(tmp_path, '2026-09-17', b)
    assert pending_daily_runs(c, a, tmp_path) == [b, c]
    assert pending_daily_runs(c, c, tmp_path) == []
    with pytest.raises(ValueError, match='does not descend'):
        pending_daily_runs(a, c, tmp_path)
    (b/'run.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        pending_daily_runs(c, a, tmp_path)


def test_optimizer_steps_can_prove_continuation():
    import torch
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([p], lr=1e-5)
    (p*p).backward()
    optimizer.step()
    saved = optimizer.state_dict()
    other = torch.optim.AdamW([p], lr=1e-5)
    other.load_state_dict(saved)
    assert optimizer_steps(other.state_dict()) == {'0': 1}
    other.zero_grad()
    (p*p).backward()
    other.step()
    assert optimizer_steps(other.state_dict()) == {'0': 2}


def test_protocol_cannot_change_training_budget():
    from quant_research.daily_loop import read
    cfg = read(Path(__file__).parents[1]/'configs/token-ranking-incremental-v1.json')
    compatible_protocol(cfg, dict(cfg, forecast_inputs=0))
    with pytest.raises(ValueError, match='learning_rate'):
        compatible_protocol(cfg, dict(cfg, learning_rate=0.001))


def test_pointer_requires_audit_and_does_not_regress(tmp_path):
    root = tmp_path/'runs/2026-09-16'
    write_json(root/'prepared/input.json', dict(signal_date='2026-09-16', new_signal_through='2026-09-09'))
    write_manifest(root)
    cfg = {'version': 'test'}
    receipt = root.with_name(root.name+'-verification.json')
    write_json(receipt, dict(passed=False, run_manifest_sha256=file_hash(root/'manifest.json')))
    with pytest.raises(ValueError, match='audit required'):
        publish_pointer(tmp_path, root, cfg)
    write_json(receipt, dict(passed=True, run_manifest_sha256=file_hash(root/'manifest.json')))
    publish_pointer(tmp_path, root, cfg)
    from quant_research.daily_loop import read
    assert read(tmp_path/'current.json')['config_sha256'] == digest(cfg)
    write_json(tmp_path/'current.json', {'signal_date': '2026-09-17'})
    with pytest.raises(ValueError, match='regress'):
        publish_pointer(tmp_path, root, cfg)


def test_manual_reuse_and_pointer_recovery_do_not_train_again(tmp_path, monkeypatch):
    import token_ranking_daily as daily_module

    from quant_research.daily_loop import read

    main = tmp_path/'main'
    first = daily(main, '2026-09-15')
    target = daily(main, '2026-09-16', first)
    # This fixture exercises completed-run recovery, with real hash manifests.
    cfg = read(Path(__file__).parents[1]/'configs/token-ranking-daily-v1.json')
    boot = tmp_path/'bootstrap'
    boot_cfg = dict(cfg, daily_run=str(first))
    write_json(boot/'prepared/binding.json', {'config': boot_cfg})
    write_json(boot/'prepared/input.json', dict(signal_date=first.name, new_signal_through='2026-09-08'))
    write_manifest(boot)
    store = tmp_path/'comparison'
    cfg.update(bootstrap=str(boot), bootstrap_sha256=file_hash(boot/'manifest.json'), main_store=str(main), store=str(store))
    config = tmp_path/'config.json'
    write_json(config, cfg)
    root = store/'runs'/target.name
    write_json(root/'prepared/binding.json', dict(daily_manifest=file_hash(target/'manifest.json'),
        config=dict(cfg, daily_run=str(target), chain_config_sha256=digest(cfg))))
    write_json(root/'prepared/input.json', dict(signal_date=target.name, new_signal_through='2026-09-09'))
    write_manifest(root)
    write_json(root.with_name(root.name+'-verification.json'), dict(passed=True, run_manifest_sha256=file_hash(root/'manifest.json')))
    # Add only the source pointer consumed by review; re-seal target and refresh its binding.
    write_json(target/'inputs/input.json', {'source': str(tmp_path/'source')})
    write_manifest(target)
    binding = read(root/'prepared/binding.json')
    binding['daily_manifest'] = file_hash(target/'manifest.json')
    write_json(root/'prepared/binding.json', binding)
    write_manifest(root)
    write_json(root.with_name(root.name+'-verification.json'), dict(passed=True, run_manifest_sha256=file_hash(root/'manifest.json')))
    def unexpected(*args, **kwargs):
        pytest.fail('Completed date was trained or forecast again')
    for name in ['train', 'forecast', 'prepare', 'load_decoder', 'audit']:
        monkeypatch.setattr(daily_module, name, unexpected)
    monkeypatch.setattr(daily_module, 'review_shadow', lambda *args: [])
    expected_hash = file_hash(root/'manifest.json')
    assert daily_module.run_daily_comparison(config, target) == root/'report.md'
    assert read(store/'current.json')['manifest_sha256'] == expected_hash
    assert daily_module.run_daily_comparison(config, target) == root/'report.md'
    assert file_hash(root/'manifest.json') == expected_hash


@pytest.mark.parametrize('published,eligible', [('2026-09-17T01:14:59Z', True), ('2026-09-17T01:15:00Z', False)])
def test_daily_publication_freezes_eligibility_at_opening_auction(tmp_path, monkeypatch, published, eligible):
    import token_ranking_incremental as incremental

    from quant_research.daily_loop import read

    cfg = dict(seeds=[17, 29, 43], parent_root='unused', forecast_inputs=0, samples=16,
        cost=.0025, minimum_legal_paths=16, learning_rate=1e-5)
    root = tmp_path/'result'
    write_json(root/'prepared/input.json', dict(signal_date='2026-09-16', observed_labels_through='2026-09-16',
        training_rows=3, horizon_dates=['2026-09-17', '2026-09-18', '2026-09-21', '2026-09-22', '2026-09-23']))
    write_json(root/'prepared/live-state.json', {})
    pd.DataFrame(dict(instrument_id=['cn.xshe.000001'], name=['A'])).to_parquet(root/'prepared/forecast-rows.parquet')
    for seed in cfg['seeds']:
        for arm in incremental.ARMS:
            dest = root/f'seed{seed}'/arm
            write_json(dest/'training/training.json', dict(batches_sha256='same', rows=3))
            (dest/'forecast').mkdir()
            np.save(dest/'forecast/paths.npy', np.ones((1, 16, 5, 6), np.float32))
    monkeypatch.setattr(incremental, 'utc_now', lambda: published)
    monkeypatch.setattr(incremental, 'live_state', lambda run: {})
    incremental.report(root, tmp_path/'source', cfg)
    assert read(root/'result.json')['prospective'] is eligible
    publication = read(root/'publication.json')
    assert publication['prospective'] is eligible
    assert publication['ranking_files']['baseline'] == 'baseline-ranking.csv'
    ranking = pd.read_csv(root/'indicators_ranking-ranking.csv')
    assert ranking.iloc[0].sell_reference_date == '2026-09-18'
    assert ranking.iloc[0].expected_net_return == pytest.approx(-.0025)


def test_verified_intermediate_audit_can_be_reused_during_catchup(tmp_path, monkeypatch):
    import token_ranking_daily as module
    root = tmp_path/'2026-09-16'
    write_json(root/'result.json', {'sealed': True})
    write_manifest(root)
    receipt = root.with_name(root.name+'-verification.json')
    write_json(receipt, {'passed': True, 'run_manifest_sha256': file_hash(root/'manifest.json')})
    calls = []
    monkeypatch.setattr(module, 'audit', lambda r: calls.append(r))
    module.ensure_audit(root)
    assert calls == []
    write_json(receipt, {'passed': False, 'run_manifest_sha256': file_hash(root/'manifest.json')})
    module.ensure_audit(root)
    assert calls == [root]
