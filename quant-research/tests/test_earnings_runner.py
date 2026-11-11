import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research.daily_loop import verify
from quant_research.earnings_fixture import create_earnings_bundle, seal_artificial_bundle
from quant_research.earnings_runner import run_bundle, run_earnings_research
from quant_research.storage import file_hash

PROTOCOL = Path(__file__).parents[1] / "configs/earnings-drift-validation-20261012-v1.json"


@pytest.fixture(scope="module")
def research(tmp_path_factory):
    root = tmp_path_factory.mktemp("earnings-runner")
    inputs = create_earnings_bundle(root / "input", PROTOCOL)
    return root, inputs, run_earnings_research(**inputs)


def test_unified_scenarios_reconcile_cash_days_and_keep_terminal_candidates(research):
    _, inputs, result = research
    assert sum(len(arms) for scenarios in result["accounts"].values() for arms in scenarios.values()) == 36
    assert result["freeze_rows"] == 31
    for horizon, scenarios in result["accounts"].items():
        for scenario, accounts in scenarios.items():
            for arm, account in accounts.items():
                assert account["nav"].date.tolist() == result["observed_dates"]
                assert account["nav"].iloc[0].nav == 200_000.
                assert not account["summary"]["formal_profit_qualified"]
            comparison = result["comparisons"][horizon, scenario]
            assert comparison["internal_arithmetic_reconciled"]
            assert len(comparison["daily_accounts"]) == 31
    pending = result["pending"][20]["base"]["E1"]
    assert len(pending) == 1 and pending.date.iloc[0] == inputs["end"]
    assert not set(pending.trade_id) & set(result["accounts"][20]["base"]["E1"]["fills"].trade_id)
    assert len(result["matches"][20]["pairs"]) == 3
    planned = result["accounts"][20]["base"]["E1"]["pending_entries"]
    assert planned.trade_id.tolist() == pending.trade_id.tolist()
    assert planned.budget.tolist() == [30_000.]
    assert result["primary_diagnostics"]["formal_profit_qualified"] is False
    assert result["formal_return_experiment_run"] is False


def test_costs_horizons_delay_and_fixed_pair_identities_are_executed(research):
    _, inputs, result = research
    pairs = result["matches"][20]["pairs"]
    ids = set(pairs.treated_trade_id)
    for scenario, accounts in result["accounts"][20].items():
        assert set(accounts["E1_matched"]["selections"].trade_id) == ids
    main = result["accounts"][20]
    base, stress = main["base"]["E1"]["fills"], main["stress"]["E1"]["fills"]
    buys = base.loc[base.side.eq("buy")]
    stress_buys = stress.loc[stress.side.eq("buy")]
    assert np.allclose(stress_buys.commission, stress_buys.notional.mul(.0005).clip(lower=5).mul(2).round(2), atol=.01)
    assert np.allclose(stress_buys.transfer_fee, stress_buys.notional.mul(.00001).round(2), atol=.01)
    assert stress_buys.stamp_duty.eq(0).all()
    date_index = {d: i for i, d in enumerate(inputs["snapshot"].dates)}
    delayed = main["delayed"]["E1"]["fills"]
    assert date_index[delayed.loc[delayed.side.eq("buy"), "date"].iloc[0]] == date_index[buys.date.iloc[0]] + 1
    for horizon in [5, 20]:
        fills = result["accounts"][horizon]["base"]["E1"]["fills"]
        for _, group in fills.groupby("trade_id"):
            assert date_index[group.loc[group.side.eq("sell"), "date"].iloc[0]] - date_index[group.loc[group.side.eq("buy"), "date"].iloc[0]] == horizon
    assert main["slippage_5"]["E1"]["summary"]["total_return"] > main["slippage_20"]["E1"]["summary"]["total_return"]


def test_v2_runner_uses_504_day_protocol_without_changing_early_accounts(research):
    _, inputs, old = research
    current = {**inputs, "protocol": json.loads(PROTOCOL.with_name("earnings-drift-validation-20261012-v2.json").read_text())}
    result = run_earnings_research(**current)
    assert result["protocol_version"].endswith("-v2")
    assert result["single_formal_look_sessions"] == 504
    assert result["primary_diagnostics"]["primary_interval"] is None
    assert result["primary_diagnostics"]["stability"]["minimum_positive_blocks"] == 6
    for horizon, scenarios in old["accounts"].items():
        for scenario, arms in scenarios.items():
            for arm, account in arms.items():
                pd.testing.assert_frame_equal(account["nav"], result["accounts"][horizon][scenario][arm]["nav"])
                pd.testing.assert_frame_equal(account["fills"], result["accounts"][horizon][scenario][arm]["fills"])
    assert not result["formal_return_experiment_run"]


def test_future_price_and_event_changes_do_not_change_earlier_matches(research):
    _, inputs, result = research
    altered = copy.deepcopy(inputs)
    future = altered["snapshot"].dates[170]
    bars = altered["snapshot"].tables["bars"]
    mask = bars.date.ge(future)
    for col in ["open", "high", "low", "close", "upper_limit", "lower_limit"]:
        bars.loc[mask, col] *= .8
    later = altered["events"].iloc[[0]].copy()
    later["announcement_id"] = "artificial-later-information"
    later["instrument_id"] = altered["snapshot"].tables["instruments"].instrument_id.iloc[12]
    later["available_at"] = future + "T10:00:00Z"
    altered["events"] = pd.concat([altered["events"], later], ignore_index=True)
    changed = run_earnings_research(**altered)
    def before(frame):
        return frame.loc[frame.date < future].reset_index(drop=True)

    pd.testing.assert_frame_equal(before(changed["matches"][20]["pairs"]), before(result["matches"][20]["pairs"]))
    for arm in ["P", "M", "E0", "E1"]:
        pd.testing.assert_frame_equal(before(changed["candidates"][arm]), before(result["candidates"][arm]))
    assert changed["accounts"][20]["base"]["E1"]["summary"]["total_return"] != result["accounts"][20]["base"]["E1"]["summary"]["total_return"]


def test_complete_no_event_input_retains_cash_accounts(research):
    _, inputs, _ = research
    altered = copy.deepcopy(inputs)
    altered["events"] = altered["events"].iloc[:0]
    result = run_earnings_research(**altered)
    assert result["matches"][20]["pairs"].empty
    for account in result["accounts"][20]["base"].values():
        assert len(account["nav"]) == 31
    for arm in ["E0", "E1", "E1_matched", "P_matched_exposure"]:
        account = result["accounts"][20]["base"][arm]
        assert account["fills"].empty and account["nav"].nav.eq(200_000).all()


@pytest.mark.parametrize("change,reason", [("training", "Training-only"), ("real", "Real-input"),
    ("protocol", "implemented v1/v2"), ("freeze", "every observed close"), ("fee", "Base fee"),
    ("model_missing", "M ranks"), ("model_late", "M ranks")])
def test_unregistered_sources_rules_and_incomplete_controls_are_rejected(research, change, reason):
    _, inputs, _ = research
    altered = copy.deepcopy(inputs)
    if change == "training":
        altered["snapshot"].manifest["declaration"]["purpose"] = "training"
    elif change == "real":
        altered["snapshot"].manifest["declaration"]["synthetic"] = False
    elif change == "protocol":
        altered["protocol"]["event"]["minimum_net_profit_yoy_lower_bound"] = .1
    elif change == "freeze":
        altered["freezes"] = altered["freezes"].iloc[1:]
    elif change == "fee":
        altered["fees"]["commission_rate"] = .0003
    elif change == "model_missing":
        altered["model_ranks"] = altered["model_ranks"].iloc[1:]
    else:
        altered["model_ranks"].loc[0, "available_at"] = altered["end"] + "T11:00:00Z"
    with pytest.raises(ValueError, match=reason):
        run_earnings_research(**altered)


def test_file_bundle_preserves_input_and_binds_every_output(research):
    root, _, _ = research
    source, output = root / "input", root / "output"
    before = file_hash(source / "manifest.json")
    receipt = run_bundle(source, output)
    assert receipt["status"] == "synthetic_offline_research_complete_not_formal_acceptance"
    assert file_hash(source / "manifest.json") == before
    assert len(list(output.glob("h*/**/summary.json"))) == 36
    assert all(file_hash(output / name) == expected for name, expected in receipt["output_files"].items())
    assert not receipt["complete_environment_registered"]
    with pytest.raises(FileExistsError):
        run_bundle(source, output)
    verify(source)


def test_failed_bundle_preserved_and_changed_input_hash_rejected(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    create_earnings_bundle(source, PROTOCOL)
    spec = json.loads((source / "inputs.json").read_text())
    spec["tables"]["fees"] = "../outside.parquet"
    (source / "inputs.json").write_text(json.dumps(spec))
    seal_artificial_bundle(source)
    with pytest.raises(ValueError, match="inside and hashed"):
        run_bundle(source, output)
    assert not output.exists()
    spec["tables"]["fees"] = "fees.parquet"
    spec["start"] = "2099-01-01"
    (source / "inputs.json").write_text(json.dumps(spec))
    seal_artificial_bundle(source)
    with pytest.raises(ValueError, match="actually observed"):
        run_bundle(source, output)
    receipt = json.loads((output / "run-receipt.json").read_text())
    assert receipt["status"] == "rejected_or_failed_research_attempt"
    with pytest.raises(FileExistsError):
        run_bundle(source, output)
    (source / "fees.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_bundle(source, tmp_path / "another-output")
