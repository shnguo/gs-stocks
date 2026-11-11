"""Rank current US index members by drift momentum (12-1 move with jump days removed).

Usage:
  uv run python scripts/us_screen.py configs/us-screen-v2.json --credentials-file /path/.env [--as-of 2026-11-04]

Universe: the S&P 500 / Nasdaq-100 / rebuilt Russell 1000 union on the latest month end in
the membership file. Fresh Alpaca daily bars are fetched for those members up to the as-of
session (default: the last complete session before today). The factor is computed exactly
as in the backtest (physics_momentum, 3-sigma jumps). Research output, not a trade list.
"""
import argparse
import importlib.util
import json
import time
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.factors import (
    composite,
    group_percentile,
    percentile,
    physics_momentum,
    price_panel,
    residualize,
)
from quant_research.shapes import shape_scores
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.us_fundamentals import FF12_NAMES
from quant_research.us_sources import (
    Http,
    alpaca_assets,
    alpaca_bars,
    alpaca_credentials,
    alpaca_symbol,
    gap_segments,
)

BASE = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fundamental_turnaround(panel, t, as_of, members, mask, f_cfg):
    """Point-in-time SEC inputs now and a year ago for every member, plus the backtested
    baskets. Share change is not available here (no unadjusted prices), so the checklist
    score uses up to 8 of the 9 signals."""
    tf = load_script("turnaround_fundamental")
    t_cfg = json.loads((BASE / f_cfg["config"]).read_text())
    prev = (Date.fromisoformat(as_of) - timedelta(days=f_cfg["lag_days"])).isoformat()
    facts = pd.concat([pd.read_parquet(BASE / path) for path in t_cfg["facts"]], ignore_index=True)
    info = members.drop_duplicates("entity").set_index("entity").reindex(panel.ids)
    ciks = info.cik.dropna().astype(int)
    facts = facts.loc[facts.cik.isin(set(ciks))]
    rows = []
    for cik, group in facts.groupby("cik"):
        rows += tf.company_inputs((cik, group, [as_of, prev]))
    fund = {d: g.set_index("cik") for d, g in pd.DataFrame(rows).groupby("date")}
    cik = info.cik.fillna(-1).astype(int).to_numpy()
    now = {f: fund[as_of][f].reindex(cik).to_numpy(float) for f in tf.FIELDS}
    before = {f: fund[prev][f].reindex(cik).to_numpy(float) for f in tf.FIELDS}
    x = tf.checklist(now, before, np.full(len(cik), np.nan), t_cfg["issuance_threshold"])
    x.update({"drawdown": tf.drawdown(panel, t, t_cfg["drawdown_days"], t_cfg["drawdown_min_share"]),
              "book_to_price": np.full(len(cik), np.nan), "net_income": now["net_income"],
              "net_income_prev": before["net_income"], "operating_cash": now["operating_cash"]})
    with np.errstate(invalid="ignore", divide="ignore"):
        x["fscore"] = np.where(x["signals_available"] >= t_cfg["fscore_min_signals"],
                               x["signals_passed"] / x["signals_available"], np.nan)
    picks, beaten = tf.turnaround_picks(x, mask, t_cfg)
    return {"drawdown_12m": x["drawdown"], "beaten_down": beaten, "delta_roa": x["delta_roa"],
            "revenue_growth": x["revenue_growth"], "net_income_ttm": now["net_income"],
            "net_income_ttm_prev": before["net_income"], "operating_cash_ttm": now["operating_cash"],
            "fscore": x["fscore"], "fscore_signals": x["signals_available"],
            "fund_turnaround": picks[f_cfg["list"]], "fund_profit_turn": picks["profit_turn"],
            "fund_deteriorating": picks["beaten_low_f"]}


def members_on_latest(path: Path) -> tuple[str, pd.DataFrame]:
    members = pd.read_parquet(path)
    for flag in ["in_sp500", "in_ndx", "in_r1000"]:
        members[flag] = members[flag].astype("boolean").fillna(False).astype(bool)
    latest = members.date.max()
    m = members.loc[(members.date == latest) & (members.in_sp500 | members.in_ndx | members.in_r1000)].copy()
    m["asof"] = m.entity.str.split("@").str[1]
    m["symbol"] = m.ticker.map(alpaca_symbol)
    return latest, m


def fetch(http, creds, members, start, end):
    """Bars keyed by entity; each symbol resolved as of the date its membership was recorded."""
    frames = []
    for asof, group in members.groupby("asof"):
        bars = alpaca_bars(http, creds, sorted(set(group.symbol)), asof, start, end)
        key = group.drop_duplicates("symbol").set_index("symbol").entity
        frames.append(bars.assign(instrument_id=bars.symbol.map(key)))
    return pd.concat(frames, ignore_index=True).dropna(subset=["instrument_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--as-of", help="signal session (YYYY-MM-DD); default: last complete session")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    data_cfg = json.loads((BASE / cfg["data_config"]).read_text())
    started = time.time()
    http = Http(BASE / data_cfg["cache"], data_cfg["proxy"], data_cfg["user_agent"],
                data_cfg["min_interval_seconds"])
    creds = alpaca_credentials(args.credentials_file)

    end = args.as_of or (Date.today() - timedelta(days=1)).isoformat()
    start = (Date.fromisoformat(end) - timedelta(days=cfg["history_calendar_days"])).isoformat()
    calendar = sorted(alpaca_bars(http, creds, [data_cfg["benchmark"]], None, start, end).date)
    as_of = calendar[-1]
    if args.as_of and as_of != args.as_of:
        raise SystemExit(f"{args.as_of} is not a trading session (last session: {as_of})")
    output = BASE / cfg["output_root"] / as_of
    if output.exists():
        raise FileExistsError(f"Screen already exists: {output}")

    membership_date, members = members_on_latest(BASE / cfg["membership"])
    bars = fetch(http, creds, members, start, as_of)
    bars = bars.loc[bars.date.isin(set(calendar))].assign(
        factor=1.0, source_trade_status=1.0, source_is_st=0.0)
    segment = bars.groupby("instrument_id", group_keys=False).date.apply(
        lambda d: gap_segments(d, calendar, cfg["max_gap_sessions"]))
    bars["sequence_id"] = bars.instrument_id + "#" + segment.astype(str)
    panel = price_panel(bars)
    print(f"{len(members)} members as of {membership_date}, {len(panel.ids)} priced, "
          f"{http.fetched} requests, {time.time() - started:.0f}s", flush=True)

    m = cfg["momentum"]
    phys = physics_momentum(panel, as_of, m["lookback"], m["skip"], m["min_traded"], cfg["jump_sigma"])
    mask = panel.traded[panel.index(as_of)] & np.isfinite(phys["drift"])
    momentum = np.expm1(phys["total"])
    rank = percentile(phys["drift"], mask)
    controlled = residualize(phys["drift"], [momentum, phys["volatility"], phys["traded_value"]], mask)
    # Watchlist only: "fell for months, turning up over the last 3" (backtested negative, see config).
    t_cfg = cfg["turnaround"]
    prior = physics_momentum(panel, as_of, **t_cfg["prior_window"], jump_sigma=cfg["jump_sigma"])["drift"]
    recent = physics_momentum(panel, as_of, **t_cfg["recent_window"], jump_sigma=cfg["jump_sigma"])["drift"]
    turn_ok = mask & np.isfinite(prior) & np.isfinite(recent)
    turnaround = np.where(turn_ok, composite([-prior, recent], turn_ok, 2), np.nan)
    turned_up = turn_ok & (prior < 0) & (recent > 0)
    t = panel.index(as_of)
    with np.errstate(divide="ignore", invalid="ignore"):
        return_1m = panel.close[t] / panel.close[t - 21] - 1

    # Pullback within the uptrend: same rules as scripts/pullback_check.py.
    pull_cfg = json.loads((BASE / cfg["pullback"]["config"]).read_text())
    pull_picks, pull = load_script("pullback_check").pullback_picks(
        panel, t, np.where(mask, phys["drift"], np.nan), mask, pull_cfg)
    with np.errstate(divide="ignore", invalid="ignore"):
        ma50_gap = panel.close[t] / pull["ma50"] - 1

    # Fundamental turnaround (困境反转): same inputs as scripts/turnaround_fundamental.py.
    ft = fundamental_turnaround(panel, t, as_of, members, mask, cfg["fundamental_turnaround"])

    asset_table = alpaca_assets(http, creds).drop_duplicates("symbol").set_index("symbol")
    assets, exchanges = asset_table.name, asset_table.exchange
    info = members.drop_duplicates("entity").set_index("entity").reindex(panel.ids)
    industry_path = BASE / data_cfg["sec"]["industry"]
    industry = pd.read_parquet(industry_path).dropna(subset=["ff12"]).set_index("cik")
    ff12 = info.cik.map(industry.ff12).to_numpy(dtype=object)

    # 20-stock shortlists: same function as scripts/shortlist_check.py, only the rules that passed.
    short = {}
    if "shortlist" in cfg:
        s_cfg = json.loads((BASE / cfg["shortlist"]["config"]).read_text())
        size = cfg["shortlist"]["size"]
        s_picks = load_script("shortlist_check").shortlist_picks(
            np.where(mask, phys["drift"], np.nan), controlled, phys["volatility"], ff12, pull_picks, mask, s_cfg)
        short = {"shortlist_mild": s_picks[f"mild_drift_{size}"], "shortlist_top": s_picks[f"top_drift_{size}"]}
    table = pd.DataFrame({
        "symbol": info.symbol.to_numpy(), "name": info.symbol.map(assets).to_numpy(),
        "exchange": info.symbol.map(exchanges).to_numpy(),
        "sector": pd.Series(ff12).map(FF12_NAMES).to_numpy(), "ff12": ff12,
        "sic_description": info.cik.map(industry.sic_description).to_numpy(),
        "in_sp500": info.in_sp500.to_numpy(), "in_ndx": info.in_ndx.to_numpy(),
        "in_r1000": info.in_r1000.to_numpy(),
        "drift_percentile": rank, "drift_quintile": np.clip(np.ceil(rank * 5), 1, 5),
        "drift_log_return": phys["drift"], "momentum_12_1": momentum,
        "jump_log_return": phys["impulse"], "jump_days": phys["jump_days"],
        "daily_volatility": phys["volatility"],
        "avg_traded_value_musd": np.exp(phys["traded_value"]) / 1e6,
        "momentum_percentile": percentile(momentum, mask),
        "controlled_drift_percentile": percentile(controlled, mask),
        "drift_sector_percentile": group_percentile(phys["drift"], ff12, mask),
        "return_1m": np.where(panel.sequence[t] == panel.sequence[t - 21], return_1m, np.nan),
        "prior_drift_9m": prior, "recent_drift_3m": recent,
        "turned_up": turned_up, "turnaround_score": turnaround,
        "ma50_gap": ma50_gap, "return_5d": pull["return_5d"], "gap_down_1m": pull["gap_down_1m"],
        "pullback": pull_picks["pullback_ma50"], "pullback_mild": pull_picks["depth_0_10"],
        **short, **ft,
        "last_close": panel.close[panel.index(as_of)],
    })
    unscored = table.loc[~mask].assign(reason=np.where(
        panel.traded[panel.index(as_of)][~mask], "under 200 sessions of history in the window",
        "no trade on the as-of session (acquired, delisted or halted)"))
    unscored = unscored.drop_duplicates("symbol")
    table = table.loc[mask].sort_values("drift_percentile", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    output.mkdir(parents=True)
    table.to_csv(output / "ranking.csv", index=False, float_format="%.6g")
    watch = table.loc[table.turned_up].sort_values("turnaround_score", ascending=False)
    watch.to_csv(output / "turnaround.csv", index=False, float_format="%.6g")
    shape_cfg = json.loads((BASE / cfg["shapes"]["config"]).read_text())
    shape_evidence = json.loads((BASE / cfg["shapes"]["results"]).read_text())["results"]
    traded = panel.traded[t]
    shapes = shape_scores(panel, as_of, traded, np.where(traded, phys["drift"], np.nan), shape_cfg)
    shape_table = pd.DataFrame({"symbol": info.symbol.to_numpy(), "name": info.symbol.map(assets).to_numpy(),
                                "exchange": info.symbol.map(exchanges).to_numpy(),
                                "sector": pd.Series(ff12).map(FF12_NAMES).to_numpy(),
                                "return_1m": np.where(panel.sequence[t] == panel.sequence[t - 21], return_1m, np.nan)})
    for key, (score, basket) in shapes.items():
        shape_table[f"{key}_score"] = score
        shape_table[f"{key}_match"] = basket
    shape_table = shape_table.loc[traded].drop_duplicates("symbol")
    shape_table.to_csv(output / "shapes.csv", index=False, float_format="%.6g")
    unscored[["symbol", "name", "reason"]].to_csv(output / "unscored.csv", index=False)
    window = f"{panel.dates[panel.index(as_of) - m['lookback']]} to {panel.dates[panel.index(as_of) - m['skip']]}"
    write_summary(output / "summary.md", table, watch, unscored, cfg, as_of, membership_date, window,
                  shape_section(shape_table, list(shapes), shape_evidence, cfg["shapes"]["top_n"]))
    write_json(output / "manifest.json", {
        "version": cfg["version"], "created_at": utc_now(), "as_of": as_of,
        "membership_date": membership_date, "window": window, "config": cfg,
        "membership_sha256": file_hash(BASE / cfg["membership"]),
        "industry_sha256": file_hash(industry_path),
        "members": len(members), "scored": len(table), "unscored": len(unscored), "turned_up": len(watch),
        "pullback": int(table.pullback.sum()), "pullback_mild": int(table.pullback_mild.sum()),
        "fund_turnaround": int(table.fund_turnaround.sum()),
        **({"shortlist_mild": int(table.shortlist_mild.sum()), "shortlist_top": int(table.shortlist_top.sum()),
            "shortlist_results_sha256": file_hash(BASE / cfg["shortlist"]["results"])} if short else {}),
        "shape_matches": {k: int(shape_table[f"{k}_match"].sum()) for k in shapes},
        "shape_results_sha256": file_hash(BASE / cfg["shapes"]["results"]),
        "files": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
        "elapsed_seconds": round(time.time() - started, 1)})
    print(f"wrote {output}: {len(table)} scored, {len(unscored)} unscored")


SHAPE_NAMES = {"rising_now": "现在稳步上涨", "turning_up": "见底回升", "long_uptrend": "长期上涨且没有破位"}
PERIOD_NAMES = {"1m": "1 个月", "3m": "3 个月", "6m": "6 个月", "12m": "12 个月"}


def shape_section(shape_table, keys, evidence, top_n):
    lines = ["| 形态 | 时间段 | 回测年化超额（扣费后） | 10 年左右中亏损年数 | 今天符合的股票数 | 得分最高的股票（括号内为近 1 个月涨跌） |",
             "|---|---|---|---|---|---|"]
    order = {name: i for i, name in enumerate(SHAPE_NAMES)}
    for key in sorted(keys, key=lambda k: order[k.rsplit("_", 1)[0]]):
        shape, period = key.rsplit("_", 1)
        ev = evidence[key]
        match = shape_table.loc[shape_table[f"{key}_match"]].sort_values(f"{key}_score", ascending=False)
        picks = "、".join(f"{r.symbol}（{r.return_1m * 100:+.0f}%）" if np.isfinite(r.return_1m) else r.symbol
                         for r in match.head(top_n).itertuples())
        lines.append(f"| {SHAPE_NAMES[shape]} | {PERIOD_NAMES[period]} | {ev['excess_net']['annual_return'] * 100:+.1f}% | "
                     f"{ev['losing_years']}/{ev['years']} | {len(match)} | {picks} |")
    return "\n".join(lines)


def pullback_section(table, cfg):
    ev = json.loads((BASE / cfg["pullback"]["results"]).read_text())["baskets"]

    def ann(key):
        return f"{ev[key]['excess_net']['annual_return'] * 100:+.1f}%"

    pull = table.loc[table.pullback.astype(bool)]
    mild = table.loc[table.pullback_mild.astype(bool)]  # exactly the backtested depth_0_10 basket
    lines = ["| # | 代码 | 名称 | 行业 | 漂移百分位 | 近 1 个月 | 离 50 日均线 | 近 5 日 | 跳空大跌日 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(mild.itertuples(), 1):
        lines.append(f"| {i} | {r.symbol} | {(r.name if isinstance(r.name, str) else '—')[:28]} | "
                     f"{r.sector if isinstance(r.sector, str) else '未知'} | {r.drift_percentile:.2f} | "
                     f"{r.return_1m * 100:+.1f}% | {r.ma50_gap * 100:+.1f}% | {r.return_5d * 100:+.1f}% | "
                     f"{'有' if r.gap_down_1m else '无'} |")
    return f"""## 上升趋势中的回调

条件：漂移最强五分之一里，收盘在 50 日均线下方的"回调"今天 {len(pull)} 只；近 1 个月跌 0–10% 的"浅回调" {len(mild)} 只（下表，大多也在均线下方）。

回测（`docs/pullback-turnaround-20261108.md`）：回调篮子扣费后年化超额 {ann('pullback_ma50')}，浅回调 {ann('depth_0_10')}，整个上升趋势组 {ann('trend_all')}，没回调的 {ann('trend_above_ma50')}。回调不比没回调差，但比整个上升趋势组多出的部分统计上分辨不出来；跌 10% 以上没有证据支持，等"止跌"也没有帮助。

{chr(10).join(lines)}"""


def shortlist_section(table, cfg):
    """The two 20-stock rules that passed scripts/shortlist_check.py, with what they cost in risk."""
    if "shortlist" not in cfg:
        return ""
    n = cfg["shortlist"]["size"]
    ev = json.loads((BASE / cfg["shortlist"]["results"]).read_text())["baskets"]
    lists = [("浅回调里漂移最强的", f"mild_drift_{n}", table.loc[table.shortlist_mild.astype(bool)]),
             ("全池漂移最强的", f"top_drift_{n}", table.loc[table.shortlist_top.astype(bool)])]
    evidence = ["| 名单 | 扣费后年化超额 | 95% 区间 | 超额波动 | 最差单月 | 最差年份 |", "|---|---|---|---|---|---|"]
    for label, key in [("漂移最强五分之一（对照，约 200 只）", "trend_all"), ("浅回调（对照，约 66 只）", "depth_0_10"),
                       *[(f"{label} {n} 只", key) for label, key, _ in lists]]:
        r = ev[key]
        low, high = (x * 12 for x in r["monthly_ci"])
        evidence.append(f"| {label} | {r['annual_mean_net'] * 100:+.1f}% | [{low * 100:+.1f}%, {high * 100:+.1f}%] | "
                        f"{r['excess_net']['annual_volatility'] * 100:.1f}% | {r['worst_month'] * 100:+.1f}% | "
                        f"{min(r['by_year'].values()) * 100:+.1f}% |")
    parts = []
    for label, _, frame in lists:
        deep = int((frame.return_1m <= -0.2).sum())
        rows = ["| # | 代码 | 名称 | 行业 | 漂移百分位 | 近 1 个月 | 近 5 日 |", "|---|---|---|---|---|---|---|"]
        for i, r in enumerate(frame.itertuples(), 1):
            rows.append(f"| {i} | {r.symbol} | {(r.name if isinstance(r.name, str) else '—')[:28]} | "
                        f"{r.sector if isinstance(r.sector, str) else '未知'} | {r.drift_percentile:.2f} | "
                        f"{r.return_1m * 100:+.1f}% | {r.return_5d * 100:+.1f}% |")
        warning = (f"\n\n> **注意：** 其中 {deep} 只近 1 个月跌超 20%。这种深回调在回测里样本很少、没有证据支持"
                   "（`docs/pullback-turnaround-20261108.md`），这份名单的历史成绩对今天的参考意义有限。") if deep else ""
        parts.append(f"### {label} {len(frame)} 只{warning}\n\n" + "\n".join(rows))
    return f"""## {n} 只的短名单

回测（`{cfg['evidence_shortlist']}`，2017–2026，月末信号、等权持有一个月）：缩到 {n} 只是把同一个信号放大，不是选得更准。平均超额更高，波动同比例变大，年化超额 ÷ 超额波动和 200 只时几乎一样。只有下面两种选法通过了事先定的标准；按低波动选、加行业上限、在浅回调里只选 10 只都没有通过。回测只检验了月末的名单，月中的名单没有检验过。

{chr(10).join(evidence)}

{(chr(10) * 2).join(parts)}"""


def fundamental_section(table, cfg):
    f_cfg = cfg["fundamental_turnaround"]
    pick = table.loc[table.fund_turnaround.astype(bool)].sort_values("drawdown_12m")
    lines = ["| # | 代码 | 名称 | 行业 | 离一年高点 | ROA 变化 | 收入增长 | 净利润（近 12 个月 / 一年前，百万美元） | 检查表得分 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(pick.head(f_cfg["top_n"]).itertuples(), 1):
        fs = f"{r.fscore:.2f}（{int(r.fscore_signals)} 项）" if np.isfinite(r.fscore) else "—"
        lines.append(f"| {i} | {r.symbol} | {(r.name if isinstance(r.name, str) else '—')[:28]} | "
                     f"{r.sector if isinstance(r.sector, str) else '未知'} | {r.drawdown_12m * 100:+.0f}% | "
                     f"{r.delta_roa * 100:+.1f} 点 | {r.revenue_growth * 100:+.0f}% | "
                     f"{r.net_income_ttm / 1e6:,.0f} / {r.net_income_ttm_prev / 1e6:,.0f} | {fs} |")
    worse = table.loc[table.fund_deteriorating.astype(bool)]
    return f"""## 困境反转（基本面）观察名单

> **警告：回测没有通过。** {f_cfg['warning']}

条件：股价离一年高点的跌幅在全池最深的三分之一，近 12 个月利润率（ROA）比一年前改善、收入比一年前增长、经营现金流为正。财务数据只用信号日前已提交给 SEC 的 10-K/10-Q。共 {len(pick)} 只，按跌幅从深到浅列出前 {f_cfg['top_n']} 只。检查表得分是 Piotroski 9 项里通过的比例（这里缺股本变化一项）。

股价深跌且检查表得分 ≤ 0.3（基本面继续恶化）的有 {len(worse)} 只：{"、".join(worse.symbol.head(20)) or "无"}。这一组回测每年低 11.6%，但样本很少。

{chr(10).join(lines)}"""


def write_summary(path, table, watch, unscored, cfg, as_of, membership_date, window, shapes_md):
    def pct(x):
        return f"{x * 100:+.1f}%"

    def rows(frame):
        out = ["| # | 代码 | 名称 | 行业 | 漂移百分位 | 行业内百分位 | 漂移 | 12-1 收益 | 跳动部分 | 跳动天数 | 日波动 "
               "| 控制后百分位 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in frame.itertuples():
            sector = r.sector if isinstance(r.sector, str) else "未知"
            detail = f"（{str(r.sic_description)[:24]}）" if isinstance(r.sic_description, str) else ""
            out.append(f"| {r.rank} | {r.symbol} | {(r.name if isinstance(r.name, str) else '—')[:28]} | {sector}{detail} | "
                       f"{r.drift_percentile:.2f} | {r.drift_sector_percentile:.2f} | "
                       f"{pct(np.expm1(r.drift_log_return))} | {pct(r.momentum_12_1)} | "
                       f"{pct(np.expm1(r.jump_log_return))} | {int(r.jump_days)} | "
                       f"{r.daily_volatility * 100:.1f}% | {r.controlled_drift_percentile:.2f} |")
        return "\n".join(out)

    n = cfg["top_n"]
    pull_md, fund_md = pullback_section(table, cfg), fundamental_section(table, cfg)
    short_md = shortlist_section(table, cfg)
    if short_md:
        pull_md = f"{pull_md}\n\n{short_md}"
    t_cfg = cfg["turnaround"]
    turn_rows = ["| # | 代码 | 名称 | 行业 | 前 9 个月漂移 | 近 3 个月漂移 | 近 1 个月涨跌 | 日波动 | 得分 |",
                 "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(watch.head(t_cfg["top_n"]).itertuples(), 1):
        sector = r.sector if isinstance(r.sector, str) else "未知"
        turn_rows.append(f"| {i} | {r.symbol} | {(r.name if isinstance(r.name, str) else '—')[:28]} | {sector} | "
                         f"{pct(np.expm1(r.prior_drift_9m))} | {pct(np.expm1(r.recent_drift_3m))} | "
                         f"{pct(r.return_1m)} | {r.daily_volatility * 100:.1f}% | {r.turnaround_score:.2f} |")
    turn_table = "\n".join(turn_rows)
    top = table.loc[table.drift_quintile == 5]
    mix = pd.DataFrame({"all": table.sector.fillna("未知").value_counts(),
                        "top": top.sector.fillna("未知").value_counts()}).fillna(0)
    mix = mix.sort_values("top", ascending=False)
    lines = ["| 行业 | 全部股票 | 占比 | 最强五分之一 | 占比 | 超配倍数 |", "|---|---|---|---|---|---|"]
    for sector, r in mix.iterrows():
        share_all, share_top = r["all"] / len(table), r["top"] / max(1, len(top))
        lines.append(f"| {sector} | {int(r['all'])} | {share_all * 100:.0f}% | {int(r['top'])} | "
                     f"{share_top * 100:.0f}% | {share_top / share_all:.1f} |")
    sector_mix = "\n".join(lines)
    per_sector = []
    for sector in mix.index:
        group = table.loc[table.sector == sector].sort_values("drift_sector_percentile", ascending=False)
        if sector == "未知" or group.empty:
            continue
        picks = "、".join(f"{r.symbol}（{np.expm1(r.drift_log_return) * 100:+.0f}%）"
                         for r in group.head(cfg["per_sector"]).itertuples())
        per_sector.append(f"| {sector} | {len(group)} | {picks} |")
    per_sector = "\n".join(["| 行业 | 股票数 | 行业内漂移最强（括号内为漂移涨幅） |", "|---|---|---|", *per_sector])
    text = f"""# 美股漂移动量筛选（{as_of}）

研究用排名，不是买卖建议。

- 信号日：{as_of}；计算窗口：{window}（过去 12 个月，去掉最近 1 个月）
- 股票池：{membership_date} 的标普 500、纳斯达克 100、重建罗素 1000 成分股并集；之后的成分股变动没有计入
- 打分 {len(table)} 只，未打分 {len(unscored)} 只（见 `unscored.csv`）
- 漂移 = 12-1 涨幅去掉"跳动日"（日收益偏离中位数超过 {cfg['jump_sigma']:g} 倍稳健标准差）后的部分。跳动部分在回测里倾向回吐
- 控制后百分位：去掉普通动量、波动率和成交额影响后的漂移排名，稳健性检验里这个版本仍然有效
- 行业：SEC 登记的 SIC 行业代码（当前值），归入 Fama-French 12 个大类。行业内百分位 = 在同一大类里按漂移的排名
- 历史证据：{cfg['evidence']}

## 行业分布

回测验证的是持有最强的五分之一（{len(top)} 只）。超配倍数 = 该行业在这一组里的占比 ÷ 在全部股票里的占比。

{sector_mix}

## 各行业内漂移最强的 {cfg['per_sector']} 只

按行业内百分位排序。回测里"行业内排名"的版本扣费后多空年化约 5.7%，比不分行业的 7.5% 低，但不押注单一行业。

{per_sector}

{pull_md}

{fund_md}

## 三种形态 × 四个时间段

按要求把三种走势形态在 1、3、6、12 个月上全部列出。回测：美股指数成分股 2017 到 2026 年，每月买入符合条件的股票、等权持有一个月，扣费后相对股票池等权的年化超额（`docs/us-screen-20261106.md` 有完整说明）。作为对照，主排名（漂移最强五分之一）是 +4.4%。12 个组合是同时检验的，单个结果的显著性要打折扣。

- 现在稳步上涨：最近这段时间的漂移（去掉跳动日）排在前五分之一
- 见底回升：之前 9 个月漂移为负，最近这段时间漂移为正
- 长期上涨且没有破位：12-1 漂移在前五分之一，且最近这段时间漂移仍为正

{shapes_md}

全部得分和是否符合见 `shapes.csv`。

## 见底回升观察名单

> **警告：这个形态在回测里明显跑输。** {t_cfg['warning']}

条件：前 9 个月（{t_cfg['prior_window']['lookback']} 到 {t_cfg['prior_window']['skip']} 个交易日前）漂移为负，近 3 个月（最近 {t_cfg['recent_window']['lookback']} 个交易日，含最近 1 个月）漂移为正。共 {len(watch)} 只，按得分（前期跌得越多、近期涨得越稳越高）列出前 {t_cfg['top_n']} 只，全部见 `turnaround.csv`。只适合作为自己看图的观察名单。

{turn_table}

## 漂移最强的 {n} 只

{rows(table.head(n))}

## 漂移最弱的 {n} 只

{rows(table.tail(n).iloc[::-1])}

完整排名见 `ranking.csv`。
"""
    path.write_text(text)


if __name__ == "__main__":
    main()
