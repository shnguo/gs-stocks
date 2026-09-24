import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research.daily_loop import (
    POLICY,
    generate,
    read,
    review,
    verify,
    write_manifest,
)
from quant_research.daily_models import ADAPTERS, EmpiricalAdapter, adapter
from quant_research.price_strategy import ordered_quantiles
from quant_research.storage import file_hash, write_json


@pytest.fixture
def case(tmp_path, monkeypatch):
    root, package, store = [tmp_path / n for n in ["source", "package", "store"]]
    panel, snap = root / "panel-h5", root / "snapshot"
    for p in [panel, snap, package]:
        p.mkdir(parents=True)
    dates = pd.bdate_range("2024-01-01", periods=75).strftime("%Y-%m-%d").tolist()
    symbols = ["a", "b"]
    pd.DataFrame(
        {"instrument_id": symbols, "listed_at": "2020-01-01", "delisted_at": ""}
    ).to_parquet(snap / "instruments.parquet")
    bars = pd.DataFrame(
        [
            {
                "instrument_id": symbol,
                "date": day,
                "open": 10.0,
                "high": 10.1,
                "low": 9.8,
                "close": 10.0,
                "factor": 1.0,
                "volume": 100.0,
                "amount": 1000.0,
                "upper_limit": 11.0,
                "lower_limit": 9.0,
                "sequence_id": symbol,
                "label_sequence_id": symbol,
            }
            for symbol in symbols
            for day in dates
        ]
    )
    bars.to_parquet(snap / "bars.parquet", index=False)
    pd.DataFrame(
        {"instrument_id": pd.Series([], dtype=str), "ex_date": pd.Series([], dtype=str)}
    ).to_parquet(snap / "actions.parquet")
    np.save(panel / "values.npy", np.zeros((2, len(dates), 2)))
    for p in [panel, snap]:
        write_manifest(p)
        m = read(p / "manifest.json")
        m.update(dataset_id="fixture", created_at="2024-01-01T00:00:00+00:00")
        if p == panel:
            m.update(dates=dates, instruments=symbols, feature_names=["x", "y"])
        write_json(p / "manifest.json", m)
    write_json(root / "schedule.json", {"sealed_holdout_start": dates[70]})
    write_json(
        package / "model.json",
        {
            "adapter": "fixture",
            "model_id": "fixture-v1",
            "feature_names": ["x", "y"],
            "lookback": 60,
            "horizon": 5,
            "trained_labels_through": "2023-12-31",
            "quality_passed": False,
            "registered_at": "2024-01-01T00:00:00+00:00",
        },
    )
    q = np.log(np.tile([0.95, 1.05, 1.10], (1, 5, 4, 1)))
    q[:, :, 2] = np.log([0.95, 0.99, 1.0])
    np.save(package / "naive.npy", ordered_quantiles(q)[0])
    write_manifest(package)
    monkeypatch.setitem(ADAPTERS, "fixture", EmpiricalAdapter)
    return root, package, store, dates


def draft(case, **kwargs):
    root, package, store, dates = case
    return generate(
        root,
        package,
        store,
        dates[60],
        dates[60] + "T08:50:00+08:00",
        mode=kwargs.pop("mode", "replay"),
        **kwargs,
    )


def update_snapshot(root, transform):
    p = root / "snapshot/bars.parquet"
    transform(pd.read_parquet(p)).to_parquet(p, index=False)
    m = read(root / "snapshot/manifest.json")
    m["files"]["bars.parquet"] = file_hash(p)
    write_json(root / "snapshot/manifest.json", m)


def test_idempotent_immutable_and_policy_version(case):
    run = draft(case)
    before = file_hash(run / "manifest.json")
    assert draft(case) == run
    newer = draft(case, policy={**POLICY, "min_reward_risk": 2.0})
    assert newer != run
    assert read(newer / "run.json")["series_id"] != read(run / "run.json")["series_id"]
    assert file_hash(run / "manifest.json") == before
    (run / "report.md").write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        draft(case)


def test_future_prices_do_not_select_inputs_or_change_predictions(case):
    run = draft(case)
    root, _, _, dates = case
    update_snapshot(root, lambda b: b.loc[~((b.instrument_id == "b") & (b.date > dates[59]))])
    x = np.load(root / "panel-h5/values.npy")
    x[:, 60:] = np.nan
    np.save(root / "panel-h5/values.npy", x)
    m = read(root / "panel-h5/manifest.json")
    m["files"]["values.npy"] = file_hash(root / "panel-h5/values.npy")
    write_json(root / "panel-h5/manifest.json", m)
    other = draft(case)
    assert other != run
    pd.testing.assert_frame_equal(
        pd.read_parquet(run / "rows.parquet"), pd.read_parquet(other / "rows.parquet")
    )
    np.testing.assert_array_equal(
        np.load(run / "fixture-forecast.npy"), np.load(other / "fixture-forecast.npy")
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(run / "plans.parquet"), pd.read_parquet(other / "plans.parquet")
    )


def test_pending_unknown_and_mature_results(case):
    root, _, store, dates = case
    run = draft(case)
    original = file_hash(run / "plans.parquet")
    partial = review(run, root, store, dates[61])
    p = pd.read_parquet(partial / "outcomes.parquet")
    assert p.state.eq("pending").all()
    assert p.hypothetical_net_return.isna().all()
    assert all(
        m["known_rows"] == 0
        for m in read(partial / "review.json")["metrics"]
        if m["day_offset"] > 1
    )
    mature = review(run, root, store, dates[64])
    assert review(run, root, store, dates[64]) == mature
    assert pd.read_parquet(mature / "outcomes.parquet").state.eq("closed").all()
    update_snapshot(root, lambda b: b.loc[~((b.instrument_id == "b") & (b.date == dates[62]))])
    missing = review(run, root, store, dates[64])
    p = pd.read_parquet(missing / "outcomes.parquet")
    assert p.loc[p.instrument_id == "b"].state.eq("unknown").all()
    assert p.loc[p.instrument_id == "b"].hypothetical_net_return.isna().all()
    assert file_hash(run / "plans.parquet") == original
    assert len(read(missing / "review.json")["comparisons"]) == 3


def test_opening_cancellation_and_reference_revision(case):
    root, _, store, dates = case
    run = draft(case)

    def gap(b):
        b.loc[b.date == dates[60], ["open", "high", "low", "close"]] = [9.0, 9.3, 8.9, 9.1]
        return b

    update_snapshot(root, gap)
    r = review(run, root, store, dates[60])
    assert pd.read_parquet(r / "outcomes.parquet").state.eq("cancelled_at_open").all()

    def correction(b):
        b.loc[b.date == dates[59], "factor"] = 2.0
        return b

    update_snapshot(root, correction)
    r = review(run, root, store, dates[64])
    assert read(r / "review.json")["reference_revisions"] == 2
    assert pd.read_parquet(r / "outcomes.parquet").state.eq("unknown").all()


def test_prospective_window_stale_source_and_holdout(case):
    root, package, store, dates = case
    now = dates[60] + "T08:55:00+08:00"
    run = draft(case, mode="prospective", now=now)
    assert read(run / "run.json")["status"] == "research_draft"
    with pytest.raises(ValueError, match="not complete"):
        review(run, root, store, dates[60], now=now)
    late = draft(case, mode="prospective", now=dates[60] + "T09:20:00+08:00")
    assert read(late / "run.json")["status"] == "blocked"
    assert not (late / "plans.parquet").exists()
    for date in [dates[69], "2026-09-14"]:
        blocked = generate(root, package, store, date, date + "T08:50:00+08:00", "replay")
        assert read(blocked / "run.json")["status"] == "blocked"
    with pytest.raises(ValueError, match="Timezone"):
        generate(root, package, store, dates[60], dates[60] + "T08:50:00", "replay")


def test_no_valid_input_retains_coverage(case):
    root, _, _, dates = case
    update_snapshot(root, lambda b: b.loc[b.date != dates[59]])
    run = draft(case)
    assert read(run / "run.json")["status"] == "blocked"
    assert len(pd.read_parquet(run / "coverage.parquet")) == 2


def test_model_fallback_forbidden_and_cycle_repeat(case):
    root, package, store, dates = case
    with pytest.raises(ValueError, match="no silent fallback"):
        adapter("kronos", package)
    spec = importlib.util.spec_from_file_location(
        "daily_script", Path(__file__).parents[1] / "scripts/daily_research.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    draft(case)
    args = (root, package, store, dates[61], dates[61] + "T08:50:00+08:00", "replay", dates[60])
    first = mod.cycle(*args)
    assert first == mod.cycle(*args)
    assert len(first["reviews"]) == 1 and not first["review_errors"]
    verify(Path(first["generated"]))


def test_intraday_publication_is_explicit_and_never_backfills_fills(case):
    root, package, store, dates = case
    cutoff = dates[60] + "T10:30:00+08:00"
    run = generate(root, package, store, dates[60], cutoff, "intraday_research", now=cutoff)
    assert read(run / "run.json")["status"] == "research_draft"
    assert "盘中补发" in (run / "report.md").read_text()
    result = review(run, root, store, dates[64], now=dates[65] + "T16:00:00+08:00")
    outcomes = pd.read_parquet(result / "outcomes.parquet")
    candidate = outcomes.loc[outcomes.signal == "conditional_candidate"]
    assert candidate.state.eq("unknown").all()
    assert candidate.hypothetical_net_return.isna().all()
