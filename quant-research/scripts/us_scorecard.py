"""Score every frozen US screen run forward and publish the record to the dashboard.

Usage:
  uv run python scripts/us_scorecard.py configs/us-scorecard-v1.json \
      --credentials-file /path/.env [--dashboard ../stock-indicator-dashboard]

For each run in artifacts/us-screen/ (files checked against the run's manifest): buy at the
open after the signal date, sell at the open 21 sessions later, compare the baskets with the
equal-weight pool and SPY. A run whose exit open exists is scored once and frozen under
artifacts/us-scorecard/outcomes/<signal date>/; later invocations only read it back. Unfinished
runs get an interim mark that is recomputed every time and never enters the record. The rules
are in the config and were fixed before any run had an outcome. Research record, not advice.
"""
import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd

from quant_research.factors import price_panel
from quant_research.scorecard import last_complete_day, score_run, summarize, timing, verified_files
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.us_sources import Http, alpaca_bars, alpaca_credentials, gap_segments

BASE = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "benchmark"


def run_panel(http, creds, ranking, as_of, calendar, benchmark, holding, max_gap):
    """Bars from the signal date to the exit session (or the latest session), by ranking symbol.

    Each symbol is resolved as of the signal date, so a later rename keeps its history.
    """
    t = calendar.index(as_of)
    dates = calendar[t:t + holding + 2]
    stocks = alpaca_bars(http, creds, sorted(ranking.symbol), as_of, as_of, dates[-1])
    bench = alpaca_bars(http, creds, [benchmark], None, as_of, dates[-1]).assign(symbol=BENCHMARK_ID)
    bars = pd.concat([stocks, bench], ignore_index=True)
    bars = bars.loc[bars.date.isin(set(dates))].assign(
        instrument_id=lambda d: d.symbol, factor=1.0, source_trade_status=1.0, source_is_st=0.0)
    segment = bars.groupby("instrument_id", group_keys=False).date.apply(
        lambda d: gap_segments(d, dates, max_gap))
    bars["sequence_id"] = bars.instrument_id + "#" + segment.astype(str)
    return price_panel(bars)


def backtest_reference(spec: dict) -> dict | None:
    """What the backtest of the same basket averaged per month, to set expectations."""
    if "backtest" not in spec:
        return None
    result = json.loads((BASE / spec["backtest"]["results"]).read_text())["baskets"][spec["backtest"]["basket"]]
    net = result["excess_net"]
    return {"months": result["months"], "mean_monthly_excess": round(net["mean_monthly"], 6),
            "monthly_sd": round(net["annual_volatility"] / math.sqrt(12), 6),
            "hit_rate": round(net["hit_rate"], 4)}


def frozen_outcome(folder: Path, cfg: dict, screen_sha: str) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    verified_files(folder, manifest["files"])
    if manifest["version"] != cfg["version"]:
        raise ValueError(f"{folder} was scored under {manifest['version']}; use a new output folder")
    if manifest["screen_manifest_sha256"] != screen_sha:
        raise ValueError(f"The screen run behind {folder} changed after it was scored")
    return json.loads((folder / "outcome.json").read_text())


def freeze(folder: Path, outcome: dict, stocks: pd.DataFrame, cfg: dict, config_sha: str) -> None:
    folder.mkdir(parents=True)
    stocks.to_csv(folder / "returns.csv", index=False, float_format="%.6g")
    write_json(folder / "outcome.json", outcome)
    write_json(folder / "manifest.json", {
        "version": cfg["version"], "scored_at": utc_now(), "as_of": outcome["as_of"],
        "config_sha256": config_sha, "screen_manifest_sha256": outcome["screen_manifest_sha256"],
        "files": {name: file_hash(folder / name) for name in ["outcome.json", "returns.csv"]}})


def show(value) -> str:
    return "     —" if value is None else f"{value * 100:+6.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path, help="also write public/us-screen/scorecard.json there")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    config_sha = file_hash(args.config)
    screen_cfg = json.loads((BASE / cfg["screen_config"]).read_text())
    data_cfg = json.loads((BASE / screen_cfg["data_config"]).read_text())
    started = time.time()
    root, output = BASE / screen_cfg["output_root"], BASE / cfg["output"]

    runs, pending = [], []
    for folder in sorted(p for p in root.iterdir() if (p / "manifest.json").exists()):
        manifest = json.loads((folder / "manifest.json").read_text())
        verified_files(folder, manifest["files"])
        run = {"as_of": manifest["as_of"], "folder": folder, "manifest": manifest,
               "sha": file_hash(folder / "manifest.json"), "outcome": None}
        if (output / "outcomes" / run["as_of"]).exists():
            run["outcome"] = frozen_outcome(output / "outcomes" / run["as_of"], cfg, run["sha"])
        else:
            pending.append(run)
        runs.append(run)
    if not runs:
        raise SystemExit(f"No screen runs in {root}")

    http = Http(BASE / data_cfg["cache"], data_cfg["proxy"], data_cfg["user_agent"],
                data_cfg["min_interval_seconds"])
    creds = alpaca_credentials(args.credentials_file)
    first = min(r["as_of"] for r in pending) if pending else runs[-1]["as_of"]
    calendar = sorted(alpaca_bars(http, creds, [data_cfg["benchmark"]], None, first, last_complete_day()).date)

    for run in pending:
        as_of = run["as_of"]
        if as_of not in calendar:
            raise ValueError(f"Signal date {as_of} is not a session in the calendar")
        ranking = pd.read_csv(run["folder"] / "ranking.csv")
        panel = run_panel(http, creds, ranking, as_of, calendar, data_cfg["benchmark"],
                          cfg["holding_sessions"], screen_cfg["max_gap_sessions"])
        outcome, stocks = score_run(ranking, panel, BENCHMARK_ID, as_of, cfg)
        outcome.update({
            "run_created_at": run["manifest"]["created_at"], "screen_version": run["manifest"]["version"],
            "screen_manifest_sha256": run["sha"],
            "timing": timing(run["manifest"]["created_at"], outcome["entry_date"]) if outcome["entry_date"] else None,
        })
        if outcome["status"] == "mature":
            freeze(output / "outcomes" / as_of, outcome, stocks, cfg, config_sha)
        run["outcome"] = outcome

    outcomes = [{**r["outcome"], "frozen": r["outcome"]["status"] == "mature"} for r in runs]
    payload = {
        "schema_version": 1, "generated_at": utc_now(), "version": cfg["version"],
        "last_session": calendar[-1], "holding_sessions": cfg["holding_sessions"],
        "cost_round_trip": 2 * cfg["cost_one_way"], "benchmark": data_cfg["benchmark"],
        "min_independent_runs": cfg["min_independent_runs"],
        "baskets": [{"key": s["key"], "label": s["label"], "note": s["note"], "primary": bool(s.get("primary")),
                     "backtest": backtest_reference(s)} for s in cfg["baskets"]],
        "runs": outcomes,
        "summary": summarize(outcomes, [s["key"] for s in cfg["baskets"]], cfg["min_independent_runs"]),
        "source": {"config": str(args.config), "config_sha256": config_sha,
                   "outcomes": f"quant-research/{cfg['output']}/outcomes"},
    }
    write_json(output / "latest.json", payload)
    if args.dashboard:
        target = args.dashboard / "public/us-screen/scorecard.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        print(f"wrote {target}")

    primary = next(s["key"] for s in cfg["baskets"] if s.get("primary"))
    print(f"last session {calendar[-1]}, {http.fetched} requests, {time.time() - started:.0f}s")
    print("signal      timing       status        held  top-fifth net   pool     SPY   vs pool  vs SPY  rank IC")
    for o in outcomes:
        b = o["baskets"].get(primary, {})
        ic = "     —" if o["rank_ic"] is None else f"{o['rank_ic']:+.3f}"
        print(f"{o['as_of']}  {o['timing'] or '—':11s}  {o['status']:8s}  {o['sessions_held']:>2}/{cfg['holding_sessions']}"
              f"   {show(b.get('net'))}%       {show(o['universe_return'])}% {show(o['benchmark_return'])}%"
              f"  {show(b.get('excess_universe'))}% {show(b.get('excess_benchmark'))}%  {ic}"
              + (f"  ({o['price_mismatches']} price mismatches)" if o["price_mismatches"] else ""))
    record = payload["summary"]["prospective"]
    print(f"prospective record: {record['runs']} matured runs, {record['independent_runs']} independent "
          f"(no conclusion before {cfg['min_independent_runs']})")


if __name__ == "__main__":
    main()
