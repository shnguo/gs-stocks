"""Date-level checks and fixed-order cost stress for an immutable price pilot."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .price_strategy import Plan, TradeAssumptions, trade_diagnostic
from .storage import file_hash, utc_now, write_json


def doubled_costs(a: TradeAssumptions) -> TradeAssumptions:
    return replace(a, commission=a.commission * 2, minimum_fee=a.minimum_fee * 2,
                   sell_tax=a.sell_tax * 2, slippage_bps=a.slippage_bps * 2)


def date_metrics(rows, targets, forecasts, prior_dates=()):
    """Pair models on exactly the same known outcome rows before aggregation."""
    common = np.logical_and.reduce([np.isfinite(p).all(axis=(1, 2, 3))
                                    for p in forecasts.values()])
    records = []
    y = targets[:, 4, 3]
    for model, q in forecasts.items():
        for scope, mask in (("full", np.ones(len(rows), bool)), ("common", common)):
            finite = mask & np.isfinite(y) & np.isfinite(q[:, 4, 3]).all(axis=1)
            for day in sorted(rows.date.unique()):
                take = finite & rows.date.eq(day).to_numpy()
                e = y[take, None] - q[take, 4, 3]
                losses = np.maximum(np.array([.1, .5, .9]) * e,
                                    np.array([-.9, -.5, -.1]) * e)
                records.append({"model": model, "scope": scope, "date": day,
                    "previous_price_pilot_date": day in prior_dates, "rows": int(take.sum()),
                    "pinball": float(losses.mean()) if take.any() else np.nan,
                    "coverage80": float(((e[:, 0] >= 0) & (e[:, 2] <= 0)).mean())
                        if take.any() else np.nan,
                    "median_log_mae": float(abs(e[:, 1]).mean()) if take.any() else np.nan})
    return pd.DataFrame(records)


def aggregate_dates(daily):
    results = []
    for (scope, model), group in daily.groupby(["scope", "model"]):
        for subset in ("all_dates", "additional_price_dates"):
            g = group if subset == "all_dates" else group.loc[~group.previous_price_pilot_date]
            g = g.loc[g.pinball.notna()]
            baseline = daily.loc[(daily.model == "naive") & (daily.scope == scope),
                                 ["date", "pinball"]].rename(columns={"pinball": "baseline_loss"})
            paired = g.merge(baseline, on="date", validate="one_to_one").dropna(subset=["baseline_loss"])
            # Full Kronos is a subset: only the common scope supports paired comparisons.
            comparable = model != "kronos" or scope == "common"
            base = paired.baseline_loss.mean()
            results.append({"scope": scope, "model": model, "subset": subset,
                "dates": len(g), "labelled_rows": int(g.rows.sum()),
                "mean_daily_pinball": float(g.pinball.mean()) if len(g) else None,
                "mean_daily_coverage80": float(g.coverage80.mean()) if len(g) else None,
                "paired_days_better_than_naive": int((paired.pinball < paired.baseline_loss).sum())
                    if comparable else None,
                "paired_dates": len(paired) if comparable else 0,
                "paired_relative_loss_reduction": float(1 - paired.pinball.mean() / base)
                    if comparable and len(paired) and base > 0 else None,
                "confidence_interval": None,
                "uncertainty": "few development dates; no stock-IID significance claim"})
    return results


def stress_selected_orders(plans, rows, labels, assumptions):
    index = {(r.date, r.instrument_id): i for i, r in enumerate(rows.itertuples())}
    out = plans.copy()
    for col, default in (("stress_net_return", 0.), ("stress_status", "no_order"),
                         ("stress_filled", False), ("stress_exit_day", np.nan)):
        out[col] = default
    out["stress_filled"] = out["stress_filled"].astype("boolean")
    costs = doubled_costs(assumptions)
    for i, row in plans.loc[plans.signal == "research_candidate"].iterrows():
        k = index[(row.date, row.instrument_id)]
        # Freeze original order prices. No new calibration, re-optimization or selection.
        result = trade_diagnostic(labels["future"][k], labels["valid"][k],
            Plan(row.buy, row.take_profit, row.stop), costs, labels["upper"][k], labels["lower"][k])
        out.loc[i, "stress_net_return"] = result["net_return"]
        out.loc[i, "stress_status"] = result["status"]
        out.loc[i, "stress_filled"] = result["filled"]
        out.loc[i, "stress_exit_day"] = result["exit_day"]
    return out


def trade_metrics(plans):
    records = []
    for model, g in plans.groupby("model"):
        for scope in ("full", "common"):
            f = g if scope == "full" else g.loc[g.common_comparison]
            orders = f.loc[f.signal == "research_candidate"]
            for stress in (False, True):
                net = "stress_net_return" if stress else "net_return"
                filled = "stress_filled" if stress else "filled"
                status = "stress_status" if stress else "status"
                resolved = orders.loc[orders[net].notna()]
                closed = orders.loc[orders[status] == "closed"]
                records.append({"model": model, "scope": scope, "cost_multiplier": 2 if stress else 1,
                    "signals": len(f), "candidate_orders": len(orders),
                    "resolved_orders": len(resolved), "unresolved_orders": int(orders[net].isna().sum()),
                    "filled": int(orders[filled].eq(True).sum()), "closed": len(closed),
                    "mean_net_per_resolved_order": float(resolved[net].mean()) if len(resolved) else None,
                    "mean_net_per_closed_trade": float(closed[net].mean()) if len(closed) else None,
                    "mean_daily_net_per_resolved_signal": float(f.groupby("date")[net].mean().mean())
                        if f[net].notna().any() else None,
                    "not_portfolio_return": True})
    return records


def verify_inputs(out):
    for name in ("prepared-manifest.json", "tree-status.json", "kronos-status.json", "report-status.json"):
        info = json.loads((out / name).read_text())
        if name != "prepared-manifest.json":
            if info["status"] != "completed":
                raise ValueError("Incomplete stage")
            info = info["files"]
        for relative, sha in info.items():
            p = (out / relative).resolve()
            if not p.is_relative_to(out.resolve()) or file_hash(p) != sha:
                raise ValueError("Input artifact hash mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New stability output directory; never overwritten")
    args = parser.parse_args()
    out = args.pilot
    verify_inputs(out)
    result_dir = args.output or out / "stability"
    result_dir.mkdir()
    config = json.loads((out / "config.json").read_text())
    rows = pd.read_parquet(out / "evaluation-rows.parquet")
    with np.load(out / "evaluation.npz") as z:
        labels = {k: z[k] for k in z.files}
    forecasts = {m: np.load(out / f"{m}-evaluation.npy") for m in ("naive", "lightgbm", "kronos")}
    daily = date_metrics(rows, labels["targets"], forecasts,
                         config.get("prior_price_evaluation_dates", []))
    daily.to_csv(result_dir / "daily-price-metrics.csv", index=False)
    plans = pd.read_csv(out / "trade-plans.csv", low_memory=False)
    stressed = stress_selected_orders(plans, rows, labels, TradeAssumptions(**config["assumptions"]))
    stressed.to_parquet(result_dir / "fixed-order-cost-stress.parquet", index=False)
    price = aggregate_dates(daily)
    trades = trade_metrics(stressed)
    conclusion = {"created_at": utc_now(), "price": price, "trades": trades,
        "matched_calibration": config.get("matched_calibration", False),
        "decision": "development_diagnostic_only; no automatic fine-tuning or live orders",
        "formal_ready": False, "executable": False}
    write_json(result_dir / "summary.json", conclusion)
    lines = ["# 5 日模型：扩大日期验证与成本压力测试", "",
        "以下是历史开发实验，不是当前买卖建议。所有订单仍不可执行，未生成组合净值。",
        "按信号日期等权统计，避免把同日股票数误当成独立市场样本。",
        "新增日期指首轮价格实验未覆盖的日期；本季度曾用于排名模型开发，并非独立最终测试。", "",
        "| 比较范围 | 模型 | 日期数 | 可评分行数 | 每日平均分位损失 | 优于朴素基线的日期 | 80% 区间覆盖 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in price:
        if r["subset"] != "additional_price_dates" or (r["scope"] == "full" and r["model"] == "kronos"):
            continue
        loss = f"{r['mean_daily_pinball']:.6f}" if r['mean_daily_pinball'] is not None else "无数据"
        cov = f"{r['mean_daily_coverage80']:.1%}" if r['mean_daily_coverage80'] is not None else "无数据"
        wins = f"{r['paired_days_better_than_naive']}/{r['paired_dates']}"
        lines.append(f"| {r['scope']} | {r['model']} | {r['dates']} | {r['labelled_rows']} | {loss} | {wins} | {cov} |")
    lines += ["", "full 为该模型的全量预测；common 为所有模型共同预测样本。交易校准也使用相同的共同样本。",
        "开发日期较少且股票受共同市场因素影响，暂不报告显著性或独立样本置信区间。", "",
        f"以下成本统计覆盖全部 {rows.date.nunique()} 个评估日期；上方价格主表仅覆盖新增价格评估日期。",
        "两倍成本使用相同已选订单价格，同时将佣金、最低费用、卖出税和滑点翻倍；不会重新挑选交易。", "",
        "| 共同样本 | 成本倍数 | 候选订单 | 结果未知 | 已关闭交易 | 每张已知结果订单平均净收益 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in trades:
        if r["scope"] != "common":
            continue
        net = f"{r['mean_net_per_resolved_order']:.3%}" if r['mean_net_per_resolved_order'] is not None else "无订单"
        lines.append(f"| {r['model']} | {r['cost_multiplier']} | {r['candidate_orders']} | {r['unresolved_orders']} | {r['closed']} | {net} |")
    lines += ["", "未成交为零收益，无法判断的结果保留未知；已知结果均值存在删失偏差。",
        "逐笔收益不代表组合收益，未模拟资金容量、手数及真实排队成交。",
        "Kronos 原始路径中不合法或不足以形成预测的样本保留在覆盖账本，其预训练历史重叠仍未排除。",
        "", "daily-price-metrics.csv 保留逐日原始统计；fixed-order-cost-stress.parquet 保留逐单基础与压力结果。"]
    (result_dir / "report.md").write_text("\n".join(lines) + "\n")
    write_json(result_dir / "manifest.json", {"status": "completed", "files": {
        f.name: file_hash(f) for f in result_dir.iterdir() if f.is_file()}})


if __name__ == "__main__":
    main()
