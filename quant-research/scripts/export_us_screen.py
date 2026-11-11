"""Export a US screen run to the dashboard as public/us-screen/latest.json.

Usage:
  uv run python scripts/export_us_screen.py configs/us-screen-v2.json \
      --dashboard ../stock-indicator-dashboard [--as-of 2026-11-04]

Reads artifacts/us-screen/<as-of>/ (default: the latest run) plus the backtest results the
screen cites, and writes one JSON file. The dashboard only reads that file.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.storage import file_hash, utc_now

BASE = Path(__file__).resolve().parents[1]
EXCHANGES = {"NASDAQ": "xnas", "NYSE": "xnys", "ARCA": "xarc"}
SHAPES = {"rising_now": "现在稳步上涨", "turning_up": "见底回升", "long_uptrend": "长期上涨且没有破位"}
SHAPE_NOTES = {
    "rising_now": "最近这段时间（含最近 1 个月）的漂移排在前五分之一",
    "turning_up": "之前约 9 个月漂移为负，最近这段时间漂移为正",
    "long_uptrend": "12-1 漂移在前五分之一，且最近这段时间漂移仍为正",
}
PERIODS = {"1m": "1 个月", "3m": "3 个月", "6m": "6 个月", "12m": "12 个月"}
PULLBACK_LABELS = {
    "trend_all": "上升趋势全部（对照，主排名最强五分之一）",
    "trend_above_ma50": "没有回调：在 50 日均线上方",
    "pullback_ma50": "回调：在 50 日均线下方",
    "depth_0_10": "浅回调：近 1 个月跌 0–10%",
    "depth_10_20": "中等回调：近 1 个月跌 10–20%",
    "depth_20_plus": "深回调：近 1 个月跌 20% 以上",
    "pullback_gap": "回调中有跳空大跌日",
    "pullback_gentle": "缓慢回调（没有跳空大跌日）",
    "pullback_stabilized": "回调后近 5 日已上涨",
    "pullback_still_falling": "回调中近 5 日仍在跌",
}


def clean(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(value) else round(float(value), 6)
    if value is None or (isinstance(value, str) and value == ""):
        return None
    if pd.isna(value):
        return None
    return value


def instrument_id(symbol, exchange):
    code = EXCHANGES.get(str(exchange))
    return f"us.{code}.{str(symbol).lower()}" if code else None


def stock_rows(frame: pd.DataFrame) -> list[dict]:
    simple = {"drift_log_return": "drift", "jump_log_return": "jump_return", "prior_drift_9m": "prior_drift_9m",
              "recent_drift_3m": "recent_drift_3m"}
    out = []
    for row in frame.to_dict("records"):
        item = {simple.get(k, k): clean(v) for k, v in row.items()}
        for source, target in simple.items():
            if item.get(target) is not None:
                item[target] = round(math.expm1(item[target]), 6)
        item["instrument_id"] = instrument_id(row.get("symbol"), row.get("exchange"))
        out.append(item)
    return out


def perf(summary: dict) -> dict:
    return {"annual_excess": clean(summary["annual_return"]), "volatility": clean(summary["annual_volatility"]),
            "max_drawdown": clean(summary["max_drawdown"]), "hit_rate": clean(summary["hit_rate"])}


FUNDAMENTAL_LABELS = {
    "beaten_all": "股价深跌全部（对照：离一年高点跌幅最深三分之一）",
    "profit_turn": "扭亏为盈：一年前亏损、现在盈利",
    "loss_improving": "亏损收窄：一年前亏损、ROA 改善",
    "beaten_improving": "深跌 + ROA 改善 + 收入增长",
    "beaten_improving_cash": "深跌 + ROA 改善 + 收入增长 + 经营现金流为正（名单用这个）",
    "beaten_high_f": "深跌 + 检查表得分 ≥ 0.7",
    "beaten_low_f": "深跌 + 检查表得分 ≤ 0.3（基本面恶化）",
    "cheap_high_f": "低估值 + 检查表得分 ≥ 0.7（经典 Piotroski）",
    "cheap_low_f": "低估值 + 检查表得分 ≤ 0.3",
}


def basket_rows(results: dict, labels: dict) -> list[dict]:
    rows = []
    for key, label in labels.items():
        ev = results[key]
        rows.append({"key": key, "label": label, **perf(ev["excess_net"]),
                     "monthly_ci": [clean(x) for x in (ev["monthly_ci"] or [None, None])],
                     "losing_years": ev["losing_years"], "years": ev["years"],
                     "median_stocks": clean(ev["median_stocks"]), "months": ev["months"]})
    return rows


def evidence(cfg: dict) -> dict:
    robust = json.loads((BASE / "artifacts/physics-robustness-20261106-v2/results.json").read_text())["results"]["us"]
    uptrend = json.loads((BASE / "artifacts/uptrend-check-20261106-v1/results.json").read_text())["variants"]
    shapes = json.loads((BASE / cfg["shapes"]["results"]).read_text())["results"]
    main = robust["sigma_3.0"]["drift"]
    sector = robust["drift_in_sector"]
    rows = [
        {"label": "普通 12-1 动量（对照）", "ic": robust["momentum_12_1"]["rank_ic_mean"],
         "top_excess": robust["momentum_12_1"]["top_excess_net_annual"],
         "long_short": robust["momentum_12_1"]["long_short_net_annual"]},
        {"label": "漂移（主排名）", "ic": main["rank_ic_mean"], "top_excess": main["top_excess_net_annual"],
         "long_short": main["long_short_net_annual"], "ic_ci": main["rank_ic_ci"]},
        {"label": "漂移，行业内排名", "ic": sector["rank_ic_mean"], "top_excess": sector["top_excess_net_annual"],
         "long_short": sector["long_short_net_annual"], "ic_ci": sector["rank_ic_ci"]},
        {"label": "漂移，控制动量、波动率、成交额", "ic": robust["drift_ex_all"]["rank_ic_mean"],
         "top_excess": robust["drift_ex_all"]["top_excess_net_annual"],
         "long_short": robust["drift_ex_all"]["long_short_net_annual"], "ic_ci": robust["drift_ex_all"]["rank_ic_ci"]},
    ]
    trend_labels = {"drift_12_0": "漂移包含最近 1 个月", "drift_12_1_above_ma200": "只在高于 200 日均线的股票里排名",
                    "drift_12_1_above_ma50": "只在高于 50 日均线的股票里排名"}
    trend = [{"label": label, "top_excess": uptrend[key]["top_excess_net"]["annual_return"]}
             for key, label in trend_labels.items()]
    shape_rows = []
    for shape, shape_label in SHAPES.items():
        for period, period_label in PERIODS.items():
            ev = shapes[f"{shape}_{period}"]
            shape_rows.append({"key": f"{shape}_{period}", "shape": shape, "shape_label": shape_label,
                               "period": period, "period_label": period_label, "note": SHAPE_NOTES[shape],
                               **perf(ev["excess_net"]), "losing_years": ev["losing_years"], "years": ev["years"],
                               "monthly_ci": [clean(x) for x in ev["monthly_ci"]],
                               "mean_stocks": clean(ev["mean_stocks"])})
    pullback = json.loads((BASE / cfg["pullback"]["results"]).read_text())["baskets"]
    fundamental = json.loads((BASE / cfg["fundamental_turnaround"]["results"]).read_text())["baskets"]
    extra = {"pullback": basket_rows(pullback, PULLBACK_LABELS),
             "fundamental": basket_rows(fundamental, FUNDAMENTAL_LABELS),
             "fundamental_warning": cfg["fundamental_turnaround"]["warning"]}
    return {**extra, "period": "2017-01 至 2026-09，美股指数成分股，每月调仓、等权、单边成本 0.05%",
            "factors": [{k: clean(v) if not isinstance(v, list) else [clean(x) for x in v] for k, v in r.items()}
                        for r in rows],
            "trend_filters": [{k: clean(v) for k, v in r.items()} for r in trend], "shapes": shape_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = BASE / cfg["output_root"]
    runs = sorted(p.name for p in root.iterdir() if (p / "manifest.json").exists())
    as_of = args.as_of or runs[-1]
    run = root / as_of
    manifest = json.loads((run / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Screen file changed after the run: {name}")
    ranking = pd.read_csv(run / "ranking.csv")
    shapes = pd.read_csv(run / "shapes.csv")
    stocks = stock_rows(ranking)
    known = {s["symbol"] for s in stocks}
    extra = shapes.loc[~shapes.symbol.isin(known), ["symbol", "name", "exchange", "sector", "return_1m"]]
    stocks += stock_rows(extra.assign(extra_only=True))
    matches = {}
    for column in [c for c in shapes.columns if c.endswith("_match")]:
        key = column[:-len("_match")]
        hit = shapes.loc[shapes[column].astype(bool)].sort_values(f"{key}_score", ascending=False)
        matches[key] = [{"symbol": r.symbol, "score": clean(getattr(r, f"{key}_score"))} for r in hit.itertuples()]
    top = ranking.loc[ranking.drift_quintile == 5]
    mix = pd.DataFrame({"all": ranking.sector.fillna("未知").value_counts(),
                        "top": top.sector.fillna("未知").value_counts()}).fillna(0).sort_values("top", ascending=False)
    payload = {
        "schema_version": 1, "generated_at": utc_now(), "as_of": as_of, "runs": runs,
        "membership_date": manifest["membership_date"], "window": manifest["window"],
        "scored": manifest["scored"], "unscored": manifest["unscored"],
        "jump_sigma": cfg["jump_sigma"],
        "sector_mix": [{"sector": k, "all": int(v["all"]), "top": int(v["top"])} for k, v in mix.iterrows()],
        "top_count": int(len(top)), "stocks": stocks, "shape_matches": matches,
        "evidence": evidence(cfg),
        "source": {"run": f"quant-research/{cfg['output_root']}/{as_of}", "manifest_sha256": file_hash(run / "manifest.json")},
    }
    target = args.dashboard / "public/us-screen/latest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {target}: {len(stocks)} stocks, {target.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
