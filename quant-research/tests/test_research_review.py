import pandas as pd
import pytest

from quant_research.backtest import simulate
from quant_research.daily_loop import read, write_manifest
from quant_research.fixtures import create_fixture
from quant_research.publication import OperationJournal, publication_status
from quant_research.research_review import (
    freeze_decision,
    match_candidates,
    refresh_research,
    review_decision,
)
from quant_research.storage import file_hash, write_json
from quant_research.universe import universe_on


def protocol():
    from pathlib import Path
    return read(Path(__file__).parents[1] / "configs/research-review-v1.json")


def market(tmp_path):
    snapshot = create_fixture(tmp_path / "raw", tmp_path / "source/snapshot", stocks=6, days=80)
    dates = snapshot.dates
    signal, horizons = dates[-6], dates[-5:]
    store, run = tmp_path / "store", tmp_path / "store/runs" / signal
    write_json(run / "run.json", {"signal_date": signal, "horizon_dates": horizons,
        "prospective": True, "checkpoints": {}, "published_at": signal + "T09:00:00Z"})
    write_json(run / "binding.json", {"source": str(tmp_path / "source"), "config": {
        "ranking_method": "t_low_to_future_high", "round_trip_cost_scenario": .0025}})
    rows = pd.DataFrame({"instrument_id": snapshot.tables["instruments"].instrument_id,
                         "rank": range(1, 7), "name": ["A", "B", "C", "D", "E", "F"],
                         "expected_net_return": [.15, .12, .1, .08, .06, .04],
                         "buy_date": horizons[0], "sell_reference_date": horizons[-1],
                         "return_q10": -.03, "model_disagreement": .02})
    run.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run / "ranking.csv", index=False)
    write_manifest(run)
    report = store / "reports" / signal
    report.mkdir(parents=True)
    rows.to_csv(report / "ranking.csv", index=False)
    write_json(report / "delivery.json", {"run_manifest_sha256": file_hash(run / "manifest.json"),
        "publication_at": signal + "T09:00:00Z", "prospective": True})
    write_manifest(report)
    write_json(store / "reference-publications" / (signal + ".json"), {
        "report": str(report), "manifest_sha256": file_hash(report / "manifest.json")})
    return snapshot, store, run


def test_deadline_is_strict_shanghai_time_and_rejects_backdating():
    before = publication_status("2026-09-28", ["2026-09-29"], "2026-09-29T01:14:59Z")
    assert before["prospective"]
    after = publication_status("2026-09-28", ["2026-09-29"], "2026-09-29T01:15:00Z")
    assert not after["prospective"] and after["minutes_late"] == 0
    with pytest.raises(ValueError, match="timezone"):
        publication_status("2026-09-28", ["2026-09-29"], "2026-09-29")
    with pytest.raises(ValueError, match="precedes"):
        publication_status("2026-09-28", ["2026-09-29"], "2026-09-28T01:00:00Z")


def test_stage_failure_remains_visible_without_corrupting_frozen_runs(tmp_path):
    journal = OperationJournal(tmp_path)
    with pytest.raises(RuntimeError), journal.stage("forecast_seed_17"):
        raise RuntimeError("expected")
    journal.finish("failed")
    saved = read(journal.path)
    assert saved["status"] == "failed" and saved["stages"][0]["status"] == "failed"
    assert saved["stages"][0]["seconds"] >= 0


def test_compact_evidence_survives_model_cleanup_and_does_not_read_future(tmp_path):
    snapshot, store, run = market(tmp_path)
    cfg = protocol()
    original = file_hash(run / "manifest.json")
    bundle = freeze_decision(run, cfg)
    history = pd.read_parquet(bundle / "history.parquet")
    assert history.date.max() == run.name
    assert len(pd.read_csv(bundle / "cohort.csv")) == 6
    assert freeze_decision(run, cfg) == bundle
    assert file_hash(run / "manifest.json") == original
    # Review only requires the compact decision and the new outcome snapshot.
    import shutil
    shutil.rmtree(run)
    review = review_decision(bundle, snapshot.path.parent, cfg)
    assert read(review / "review.json")["cards"][0]["execution"]["status"] == "blocked"


def test_incremental_unknown_prices_remain_unknown_and_cannot_authorize_simulation(tmp_path):
    snapshot, store, run = market(tmp_path)
    manifest = read(snapshot.path / "manifest.json")
    manifest.pop("schema_version")
    manifest["declaration"]["purpose"] = "training"
    bars = pd.read_parquet(snapshot.path / "bars.parquet")
    bars.loc[bars.date.eq(snapshot.dates[-1]), "close"] = float("nan")
    bars.to_parquet(snapshot.path / "bars.parquet", index=False)
    manifest["files"]["bars.parquet"] = file_hash(snapshot.path / "bars.parquet")
    write_json(snapshot.path / "manifest.json", manifest)
    result = refresh_research(store, snapshot.path.parent, protocol())
    summary = read(result["summary"])
    assert "training_snapshot_has_no_execution_contract" in summary["latest"]["arms"][0]["execution"]["blockers"]


def test_refresh_retains_history_after_large_model_outputs_are_removed(tmp_path):
    import shutil
    snapshot, store, run = market(tmp_path)
    cfg = protocol()
    first = refresh_research(store, snapshot.path.parent, cfg)
    shutil.rmtree(run)
    second = refresh_research(store, snapshot.path.parent, cfg)
    assert read(first["summary"])["latest"]["signal_date"] == read(second["summary"])["latest"]["signal_date"]
    assert len(read(second["summary"])["history"]) == 1


def test_execution_has_frozen_terminal_close_and_cost_stress(tmp_path):
    snapshot, store, run = market(tmp_path)
    cfg = protocol()
    cfg["allow_synthetic_for_tests"] = True
    cfg["portfolio"]["max_weight"] = .15
    cfg["portfolio"]["industry_deviation"] = .5
    cfg["portfolio"]["adv_participation"] = .1
    result = refresh_research(store, snapshot.path.parent, cfg)
    summary = read(result["summary"])
    execution = summary["latest"]["arms"][0]["execution"]
    assert execution["status"] == "simulated"
    assert len(execution["scenarios"]) == 3
    assert all(s["ending_positions"] == 0 for s in execution["scenarios"])
    assert execution["scenarios"][0]["total_return"] > execution["scenarios"][-1]["total_return"]
    folder = next((store / "research-reviews" / run.name).iterdir())
    fills = pd.read_csv(folder / "control-cost-1.0-fills.csv")
    exits = fills[fills.side.eq("sell")]
    assert exits.date.eq(snapshot.dates[-1]).all() and exits.price_field.eq("close").all()
    assert summary["prospective_mature_dates"] == 0 and summary['engineering_only']


def test_late_publications_are_diagnosed_but_never_forward_scored(tmp_path):
    snapshot, store, run = market(tmp_path)
    info = read(run / "run.json")
    info["prospective"] = False
    write_json(run / "run.json", info)
    write_manifest(run)
    report = store / "reports" / run.name
    write_json(report / "delivery.json", {"run_manifest_sha256": file_hash(run / "manifest.json"),
        "publication_at": snapshot.dates[-5] + "T02:00:00Z", "prospective": False})
    write_manifest(report)
    write_json(store / "reference-publications" / (run.name + ".json"), {
        "report": str(report), "manifest_sha256": file_hash(report / "manifest.json")})
    result = refresh_research(store, snapshot.path.parent, protocol())
    assert read(result["summary"])["prospective_mature_dates"] == 0
    assert read(result["summary"])["latest"]["arms"][0]["publication"]["minutes_late"] == 45


def test_matching_is_outcome_blind_and_enforces_board_and_calipers():
    cohort = pd.DataFrame([
        {"arm": "control", "instrument_id": "a", "rank": 1, "board": "main", "predicted_rank": .5, "volatility_rank": .5, "liquidity_rank": .5},
        {"arm": "control", "instrument_id": "b", "rank": 2, "board": "main", "predicted_rank": .52, "volatility_rank": .51, "liquidity_rank": .5},
        {"arm": "control", "instrument_id": "c", "rank": 3, "board": "other", "predicted_rank": .5, "volatility_rank": .5, "liquidity_rank": .5},
        {"arm": "start_stage", "instrument_id": "a", "rank": 1, "board": "main", "predicted_rank": .5, "volatility_rank": .5, "liquidity_rank": .5},
    ])
    pairs = match_candidates(cohort, protocol())
    assert pairs.control.tolist() == ["b"]
    cohort["future_return"] = [100, -100, 200, -200]
    pd.testing.assert_frame_equal(match_candidates(cohort, protocol()), pairs)


def test_close_at_lower_limit_retains_unliquidated_holdings(tmp_path):
    snapshot, _, run = market(tmp_path)
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    terminal = snapshot.dates[-1]
    bars = snapshot.tables["bars"]
    mask = bars.instrument_id.eq(symbol) & bars.date.eq(terminal)
    bars.loc[mask, "lower_limit"] = bars.loc[mask, "close"]
    preds = pd.DataFrame({"date": [run.name], "instrument_id": [symbol], "score": [1],
                          "planned_exit_date": [terminal]})
    cfg = protocol()["portfolio"]
    result = simulate(snapshot, preds, 5, cfg)
    assert result["summary"]["terminal_positions"] == 1
    assert "close_at_price_limit" in result["orders"].block_reason.values


def test_signal_universe_does_not_use_risk_revisions_after_publication(tmp_path):
    snapshot, _, run = market(tmp_path)
    symbol = snapshot.tables['instruments'].instrument_id.iloc[0]
    snapshot.tables['risk_events'] = pd.DataFrame([{
        'instrument_id': symbol, 'start_date': run.name, 'end_date': '', 'status': '*ST',
        'known_at': pd.Timestamp(run.name+'T10:00:00Z'),
    }])
    early = universe_on(snapshot, run.name, availability_at=run.name+'T09:00:00Z')
    later = universe_on(snapshot, run.name)
    assert early.set_index('instrument_id').loc[symbol, 'eligible']
    assert not later.set_index('instrument_id').loc[symbol, 'eligible']


def test_missed_deadline_does_not_start_training_or_fetch_daily_prices(tmp_path, monkeypatch):
    import daily_token_cycle as operator
    cfg = {
        'store': str(tmp_path/'store'), 'calendar_evidence': str(tmp_path/'calendar.json'),
        'rust_repo': str(tmp_path), 'env_file': str(tmp_path/'unused.env'),
        'run_after_shanghai': '16:30',
    }
    config = tmp_path/'config.json'
    write_json(config, cfg)
    write_json(tmp_path/'calendar.json', {'valid_from': '2026-09-28', 'valid_through': '2026-10-12'})
    calendar = pd.DataFrame({'date': pd.bdate_range('2026-09-28', periods=8).strftime('%Y-%m-%d'),
                             'is_open': True})
    calls = []
    monkeypatch.setattr(operator, 'utc_now', lambda: '2026-09-29T01:45:00Z')
    monkeypatch.setattr(operator, 'capture', lambda *args: calls.append(args[3]) or tmp_path)
    monkeypatch.setattr(operator, 'normalized', lambda _: (calendar, {}))
    monkeypatch.setattr(operator, 'cycle', lambda *args, **kwargs: pytest.fail('Training must not start'))
    result = operator.run(config)
    assert result['status'] == 'missed_publication_deadline'
    assert calls == ['trade_cal']
    assert read(result['operation_journal'])['status'] == 'missed_publication_deadline'


def test_retention_reuses_core_identity_when_only_compact_evidence_changes(tmp_path, monkeypatch):
    from daily_token_cycle import retain_daily
    snapshot, store, run = market(tmp_path)
    config = tmp_path/'config.json'
    write_json(config, {})
    operation = store/'operations/test.json'
    write_json(operation, {'seconds': 1})
    compact = freeze_decision(run, protocol())
    calls = []
    def archive(config, identity, sources, output, cli, cloud):
        calls.append((identity, sources))
        return output/'retention.json'
    monkeypatch.setattr('archive_research_artifact.archive', archive)
    result = {'run': str(run), 'signal': run.name, 'research_evidence': [str(compact)],
              'operation_journal': str(operation)}
    retain_daily(result, config, config, tmp_path/'cli', tmp_path/'cloud')
    write_json(operation, {'seconds': 2})
    retain_daily(result, config, config, tmp_path/'cli', tmp_path/'cloud')
    assert calls[0][0] == calls[2][0]
    assert calls[1][0] != calls[3][0]
    assert compact not in calls[0][1] and compact in calls[1][1]
    assert run not in calls[1][1]


def test_explicit_event_bundle_persists_across_later_manual_reviews(tmp_path):
    snapshot, store, run = market(tmp_path)
    overlay = tmp_path/'event-overlay'
    overlay.mkdir()
    control = pd.read_csv(run/'ranking.csv')
    control.to_csv(overlay/'control-ranking.csv', index=False)
    event = control.head(1).assign(event_overlay_rank=1, event_type='periodic_report', event_score=80)
    event.to_csv(overlay/'event-overlay-ranking.csv', index=False)
    event.head(0).assign(confirmed_event_rank=pd.Series(dtype=int)).to_csv(
        overlay/'confirmed-event-ranking.csv', index=False)
    write_json(overlay/'metadata.json', {'signal_date': run.name, 'decision_at': run.name+'T09:00:00Z'})
    write_manifest(overlay)
    first = refresh_research(store, snapshot.path.parent, protocol(), [overlay])
    assert 'event_overlay' in [a['arm'] for a in read(first['summary'])['latest']['arms']]
    second = refresh_research(store, snapshot.path.parent, protocol())
    assert 'event_overlay' in [a['arm'] for a in read(second['summary'])['latest']['arms']]
    assert read(store/'event-overlay-publications.json')[run.name]['manifest_sha256'] == file_hash(overlay/'manifest.json')


def test_matching_timestamp_cannot_turn_historical_reconstruction_into_forward_evidence(tmp_path, monkeypatch):
    _, _, run = market(tmp_path)
    cfg = protocol()
    monkeypatch.setattr('quant_research.research_review.utc_now', lambda: run.name+'T09:00:00Z')
    timely = freeze_decision(run, cfg)
    assert read(timely/'decision.json')['matching_prospective']
    cfg['version'] = 'historical-matching-review'
    monkeypatch.setattr('quant_research.research_review.utc_now', lambda: read(run/'run.json')['horizon_dates'][0]+'T02:00:00Z')
    reconstructed = freeze_decision(run, cfg)
    assert not read(reconstructed/'decision.json')['matching_prospective']
