"""Calendar and bank-exclusion diagnostics; no signal fitting or sealed prices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.storage import file_hash, utc_now


def daily_returns(nav, initial=200000.0):
    values = np.r_[initial, nav.equity.to_numpy()]
    return values[1:] / values[:-1] - 1


def calendar_metrics(nav):
    frame = nav.assign(ret=daily_returns(nav), year=nav.date.str[:4])
    out = []
    for year, group in frame.groupby("year", sort=True):
        wealth = np.r_[1.0, np.cumprod(1 + group.ret.to_numpy())]
        out.append(
            dict(
                year=year,
                full_year=year in ["2022", "2023", "2024"],
                sessions=len(group),
                net_return=float(wealth[-1] - 1),
                within_year_drawdown=float(-(wealth / np.maximum.accumulate(wealth) - 1).min()),
            )
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    result = json.loads((root / "results/comparison.json").read_text())
    phases = ["base", "phase5", "phase10", "phase15"]
    annual, rolling, removal = [], [], []
    daily = {}
    for item in result["summaries"]:
        if item["scenario"] not in phases:
            continue
        nav = pd.read_parquet(
            root / "results" / item["scenario"] / item["strategy"] / "nav.parquet"
        )
        r = daily_returns(nav)
        key = (item["variant"], item["scenario"])
        daily[key] = (nav.date.to_numpy(), r)
        for row in calendar_metrics(nav):
            annual.append(dict(variant=item["variant"], scenario=item["scenario"], **row))
        eq = np.r_[200000.0, nav.equity.to_numpy()]
        values = eq[252:] / eq[:-252] - 1
        assert len(values) == 700
        rolling.append(
            dict(
                variant=item["variant"],
                scenario=item["scenario"],
                windows=len(values),
                positive_fraction=float(np.mean(values > 0)),
                min_return=float(values.min()),
                median_return=float(np.median(values)),
                max_return=float(values.max()),
                worst_window_end=str(nav.date.iloc[251 + np.argmin(values)]),
                overlapping_not_independent=True,
            )
        )
        for year in ["2022", "2023", "2024"]:
            mask = np.array([not day.startswith(year) for day in nav.date])
            value = float(np.expm1(np.log1p(r[mask]).sum()))
            removal.append(
                dict(
                    variant=item["variant"],
                    scenario=item["scenario"],
                    removed_year=year,
                    remaining_observed_return=value,
                    not_a_tradeable_strategy=True,
                )
            )
    annual_frame = pd.DataFrame(annual)
    year_summary = (
        annual_frame.groupby(["variant", "year", "full_year"])
        .agg(
            median_net_return=("net_return", "median"),
            min_net_return=("net_return", "min"),
            max_net_return=("net_return", "max"),
            positive_phase_count=("net_return", lambda x: int((x > 0).sum())),
            max_within_year_drawdown=("within_year_drawdown", "max"),
        )
        .reset_index()
    )
    gates = []
    for variant in ["original", "nonbank"]:
        s = [x for x in result["summaries"] if x["variant"] == variant]
        p = [x for x in s if x["scenario"] in phases]
        ys = year_summary[(year_summary.variant == variant) & year_summary.full_year]
        best = str(ys.sort_values("median_net_return", ascending=False).year.iloc[0])
        removed = [x for x in removal if x["variant"] == variant and x["removed_year"] == best]
        positives = int(sum(x["net_return"] > 0 for x in p))
        positive_years = int((ys.median_net_return > 0).sum())
        remaining_positive = int(sum(x["remaining_observed_return"] > 0 for x in removed))
        double = next(x for x in s if x["scenario"] == "double_cost")["net_return"]
        gates.append(
            dict(
                variant=variant,
                positive_total_return_phases=positives,
                full_years_positive_median=positive_years,
                best_full_year=best,
                positive_phases_after_removing_best_year=remaining_positive,
                double_cost_return=double,
                gate_passed=bool(
                    positives >= 3
                    and positive_years >= 2
                    and remaining_positive >= 3
                    and double > 0
                ),
            )
        )
    final = all(x["gate_passed"] for x in gates)
    decision = dict(
        computed_at=utc_now(),
        decision="continue" if final else "eliminate_current_standalone_ten_stock_lowvol_version",
        continue_direction=final,
        gates=gates,
        full_year_sample_count=3,
        new_blind_evidence=False,
        sealed_returns_used=False,
        parameter_search=False,
        reason="Net profitability did not satisfy the frozen cross-year and bank-exclusion robustness gate"
        if not final
        else "Research continuation gates passed subject to source limitations",
        scope="Current 60-day lowvol / MA5 / ten-stock / 20-session / 90% rule; not all lowvol investing",
        formal_profit_qualified=False,
    )
    for name, value in [
        ("annual.json", annual),
        ("rolling-252.json", rolling),
        ("year-removal.json", removal),
        ("decision.json", decision),
    ]:
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    year_summary.to_csv(root / "annual-summary.csv", index=False)
    pd.DataFrame(annual).to_csv(root / "annual-detail.csv", index=False)
    pd.DataFrame(rolling).to_csv(root / "rolling-252.csv", index=False)
    pd.DataFrame(removal).to_csv(root / "year-removal.csv", index=False)
    (root / "analysis-bindings.json").write_text(
        json.dumps(
            {
                str(root / "results/comparison.json"): file_hash(root / "results/comparison.json"),
                str(Path(__file__)): file_hash(Path(__file__)),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(year_summary.to_string(index=False))
    print(pd.DataFrame(removal).query("removed_year=='2024'").to_string(index=False))
    print(pd.DataFrame(rolling).to_string(index=False))


if __name__ == "__main__":
    main()
