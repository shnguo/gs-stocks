"""Validate causal early-stage reranking against frozen full-universe forecasts."""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from token_features_run import read, verify_manifest
from token_ranking_run import seal

from quant_research.early_stage import history_features, path_execution_statistics, rerank
from quant_research.storage import file_hash, utc_now, write_json

BASE = Path(__file__).resolve().parents[1]
ARMS = ("control", "overheat_penalty", "early_stage", "combined")


def load_paths(source, cfg, seed, indices, verified):
    directory = source/"forecasts"/f"seed{seed}"/cfg["source_arm"]
    out = np.empty((len(indices), cfg["samples"], 5, 6), np.float32)
    chunk = indices//cfg["chunk_rows"]*cfg["chunk_rows"]
    for start in np.unique(chunk):
        marker = directory/f"{start:07d}.json"
        path = marker.with_suffix(".npy")
        key = (seed, int(start))
        if key not in verified:
            record = read(marker)
            if file_hash(path) != record["sha256"]:
                raise ValueError("Changed forecast chunk: "+str(path))
            verified.add(key)
        mask = chunk == start
        values = np.load(path, mmap_mode="r")
        out[mask] = values[indices[mask]-start]
    return out


def actual_outcomes(future, known, indices, offsets, cost):
    values = np.asarray(future[indices], dtype=float)
    known = np.asarray(known[indices], dtype=bool).all(axis=1)
    offsets = np.asarray(offsets, dtype=int)
    row = np.arange(len(indices))
    entry_open = values[:, 0, 0]
    t_low = values[:, 0, 2]
    selected_high = values[row, offsets, 1]
    extrema = selected_high/t_low-1-cost
    execution = selected_high/entry_open-1-cost
    ideal = np.max(values[:, 1:, 1], axis=1)/entry_open-1-cost
    adverse = np.array([values[i, :offsets[i]+1, 2].min()/entry_open[i]-1
                        for i in range(len(indices))])
    output = pd.DataFrame({
        "local_row": indices,
        "label_known": known,
        "actual_extrema_scenario": np.where(known, extrema, np.nan),
        "actual_execution_return": np.where(known, execution, np.nan),
        "actual_ideal_execution_return": np.where(known, ideal, np.nan),
        "actual_adverse": np.where(known, adverse, np.nan),
    })
    return output


def daily_metrics(ranked, top_n):
    records = []
    for (arm, date), group in ranked.groupby(["arm", "date"], sort=True):
        top = group.sort_values("rank").head(top_n)
        visible = top[top.label_known]
        pool = group[group.label_known]
        records.append(dict(
            arm=arm, date=date, rows=len(group), top_rows=len(top), top_known=len(visible),
            top_unknown=top_n-len(visible),
            current_extrema_scenario_pct=visible.actual_extrema_scenario.mean()*100,
            selected_execution_pct=visible.actual_execution_return.mean()*100,
            ideal_execution_pct=visible.actual_ideal_execution_return.mean()*100,
            execution_q10_pct=visible.actual_execution_return.quantile(.1)*100,
            execution_positive_fraction=(visible.actual_execution_return > 0).mean(),
            adverse_excursion_pct=visible.actual_adverse.mean()*100,
            execution_lift_pp=(visible.actual_execution_return.mean()-pool.actual_execution_return.mean())*100,
            overextended_fraction=top.overextended.mean(),
            right_side_fraction=top.right_side.mean(),
            mean_early_stage_score=top.early_stage_score.mean(),
            mean_prior_return_10_pct=top.return_10.mean()*100,
            predicted_reference_pct=top.predicted.mean()*100,
            predicted_execution_pct=top.predicted_execution_return.mean()*100,
        ))
    return pd.DataFrame(records)


def comparisons(daily, cfg):
    metrics = ["current_extrema_scenario_pct", "selected_execution_pct", "ideal_execution_pct",
               "execution_q10_pct", "execution_positive_fraction", "adverse_excursion_pct",
               "execution_lift_pp", "overextended_fraction", "right_side_fraction",
               "mean_early_stage_score", "mean_prior_return_10_pct"]
    output = []
    control = daily[daily.arm.eq("control")].set_index("date").sort_index()
    for arm in ARMS[1:]:
        candidate = daily[daily.arm.eq(arm)].set_index("date").sort_index()
        if not candidate.index.equals(control.index):
            raise ValueError("Unpaired comparison dates")
        for metric in metrics:
            delta = candidate[metric]-control[metric]
            if metric in ("overextended_fraction", "mean_prior_return_10_pct"):
                delta = -delta
            values = delta.dropna().to_numpy()
            n = len(values)
            block = min(cfg["bootstrap_block_dates"], n)
            rng = np.random.default_rng(271828)
            starts = rng.integers(0, n, (cfg["bootstrap_replicates"], math.ceil(n/block)))
            ix = ((starts[:, :, None]+np.arange(block)) % n).reshape(len(starts), -1)[:, :n]
            boot = values[ix].mean(axis=1)
            output.append(dict(candidate=arm, control="control", metric=metric,
                improvement=float(values.mean()), median_improvement=float(np.median(values)),
                dates=n, dates_improved=int((values > 1e-10).sum()),
                date_win_fraction=float((values > 1e-10).mean()),
                bootstrap_95=np.quantile(boot, [.025, .975]).tolist()))
    return output


def run(config):
    cfg = read(config)
    root = BASE/cfg["output"]
    source = BASE/cfg["source"]
    if (root/"completed.json").exists():
        verify_manifest(root, "completed.json")
        return root/"report.md"
    if root.exists():
        raise ValueError("Partial output retained; inspect it or choose a new output")
    verify_manifest(source/"validation", "completed.json")
    for seed in cfg["seeds"]:
        marker = read(source/"forecasts"/f"seed{seed}"/cfg["source_arm"]/"completed.json")
        if marker["identity"]["samples"] != cfg["samples"] or marker["inputs"] != 378079:
            raise ValueError("Forecast identity differs from frozen protocol")
    rows = pd.read_parquet(source/"evaluation-inputs/rows.parquet")
    future = np.load(source/"evaluation-inputs/future.npy", mmap_mode="r")
    known = np.load(source/"evaluation-inputs/valid.npy", mmap_mode="r")
    raw_path = BASE/cfg["price_input"]/"values.npy"
    raw = np.load(raw_path, mmap_mode="r")
    root.mkdir(parents=True)
    write_json(root/"protocol.json", cfg)
    live = BASE/"artifacts/daily-token-live-v1"
    protected = [BASE/"configs/daily-token-v1.json", live/"current.json", live/"latest.json"]
    write_json(root/"live-state.json", {str(p): file_hash(p) for p in protected})
    sources = [config, source/"completed.json", source/"validation/completed.json",
               source/"evaluation-inputs/rows.parquet", source/"evaluation-inputs/future.npy",
               source/"evaluation-inputs/valid.npy", raw_path]
    write_json(root/"sources.json", {str(p.resolve()): file_hash(p) for p in sources})
    shutil.copy2(BASE/"src/quant_research/early_stage.py", root/"early_stage.py")
    shutil.copy2(Path(__file__), root/"token_early_stage_rerank.py")

    output = root/"rankings"
    output.mkdir()
    all_daily = []
    verified = set()
    causal_rows = 0
    for day, group in rows.groupby("date", sort=True):
        source_file = source/"validation"/cfg["source_mode"]/f"{day}.parquet"
        published = pd.read_parquet(source_file)
        base = published[(published.arm == cfg["source_arm"]) &
                         (published.seed.astype(str) == "ensemble")].copy()
        base = base.sort_values("local_row").reset_index(drop=True)
        indices = base.local_row.to_numpy(int)
        inherited = rows.iloc[indices]
        if not np.array_equal(inherited.row_id.to_numpy(), indices):
            raise ValueError("Source row identity changed")
        ss = inherited.stock_index.to_numpy(int)
        tt = inherited.date_index.to_numpy(int)
        history = raw[ss[:, None], tt[:, None]+np.arange(-59, 1)].copy()
        features = history_features(history)
        features.insert(0, "local_row", indices)
        models = [load_paths(source, cfg, seed, indices, verified) for seed in cfg["seeds"]]
        execution = path_execution_statistics(models, base.sell_offset.to_numpy(int), cfg["cost"],
                                              cfg["minimum_legal_paths"])
        if not execution.execution_forecast_eligible.all():
            raise ValueError("Paired source contains an ineligible execution forecast")
        candidate = base[["local_row", "instrument_id", "date", "sell_offset", "predicted"]].copy()
        candidate = pd.concat([candidate.reset_index(drop=True),
                               features.drop(columns="local_row").reset_index(drop=True),
                               execution.reset_index(drop=True)], axis=1)
        frozen = rerank(candidate, cfg["weights"])
        keys = frozen[["arm", "local_row", "rank", "sell_offset"]].copy()
        outcomes = actual_outcomes(future, known, indices, base.sell_offset.to_numpy(int), cfg["cost"])
        ranked = frozen.merge(outcomes, on="local_row", validate="many_to_one")
        pd.testing.assert_frame_equal(keys.reset_index(drop=True),
            ranked[["arm", "local_row", "rank", "sell_offset"]].reset_index(drop=True))
        ranked.to_parquet(output/f"{day}.parquet", index=False)
        all_daily.append(daily_metrics(ranked, cfg["top_n"]))
        causal_rows += len(candidate)
        write_json(root/"progress.json", dict(stage="reranking", date=day,
            dates_completed=len(all_daily), rows=causal_rows, at=utc_now()))
        print("Reranked", day, len(candidate), flush=True)

    daily = pd.concat(all_daily, ignore_index=True)
    daily.to_csv(root/"daily.csv", index=False)
    summary = daily.groupby("arm").mean(numeric_only=True).reset_index()
    summary.to_csv(root/"summary.csv", index=False)
    comparison = comparisons(daily, cfg)
    write_json(root/"comparisons.json", comparison)
    checks = []
    lookup = {(x["candidate"], x["metric"]): x for x in comparison}
    summaries = summary.set_index("arm")
    for arm in ARMS[1:]:
        ret = lookup[(arm, "selected_execution_pct")]
        tail = lookup[(arm, "execution_q10_pct")]
        adverse = lookup[(arm, "adverse_excursion_pct")]
        ext = lookup[(arm, "overextended_fraction")]
        right = lookup[(arm, "right_side_fraction")]
        primary = ret["improvement"] > 0 and ret["date_win_fraction"] > .5 and ext["improvement"] > 0
        stage_useful = (summaries.loc[arm, "selected_execution_pct"] > 0 and
            tail["improvement"] > 0 and tail["date_win_fraction"] > .5 and
            adverse["improvement"] > 0 and adverse["date_win_fraction"] > .5 and
            ext["improvement"] > 0 and right["improvement"] > 0)
        checks.append(dict(arm=arm, executable_improvement_pp=ret["improvement"],
            executable_date_win_fraction=ret["date_win_fraction"],
            overextension_reduction=ext["improvement"],
            q10_improvement_pp=tail["improvement"], q10_date_win_fraction=tail["date_win_fraction"],
            adverse_improvement_pp=adverse["improvement"], adverse_date_win_fraction=adverse["date_win_fraction"],
            right_side_improvement=right["improvement"], primary_replacement=primary,
            useful_as_entry_stage_watchlist=stage_useful))
    eligible = [x for x in checks if x["useful_as_entry_stage_watchlist"]]
    selected = max(eligible, key=lambda x: summaries.loc[x["arm"], "selected_execution_pct"])["arm"] if eligible else None
    write_json(root/"decision.json", dict(rule=cfg["decision_rule"], arms=checks,
        live_promotion=False, selected_for_shadow=selected,
        interpretation="A risk-oriented entry-stage watchlist may be useful without replacing the return-maximizing primary ranking."))

    lines = ["# 上涨初期重排验证", "",
        "本实验复用冻结的 71 个历史日期、全量输入候选和三随机种子预测路径。所有上涨阶段特征只读取信号日及以前 60 个交易日；排名在读取未来结果前冻结。", "",
        "四组分别为原始收益排名、过热惩罚、上涨初期评分，以及上涨初期与首日开盘可执行收益和预测下行风险的联合排名。", "",
        "## 汇总结果", "",
        "| 方案 | 原极值情景 | 首日开盘可执行收益 | 可执行收益10%分位 | 不利波动 | 过热占比 | 近10日涨幅 |", 
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary.itertuples():
        lines.append(f"| {r.arm} | {r.current_extrema_scenario_pct:.2f}% | {r.selected_execution_pct:.2f}% | {r.execution_q10_pct:.2f}% | {r.adverse_excursion_pct:.2f}% | {r.overextended_fraction:.1%} | {r.mean_prior_return_10_pct:.2f}% |")
    lines += ["", "## 相对原始排名", "",
        "| 方案 | 指标 | 平均改善 | 改善日期 | 日期区块95%区间 |",
        "| --- | --- | ---: | ---: | --- |"]
    names = {"selected_execution_pct": "首日开盘可执行收益",
             "execution_q10_pct": "可执行收益10%分位",
             "adverse_excursion_pct": "不利波动",
             "overextended_fraction": "过热占比下降",
             "mean_prior_return_10_pct": "入选前10日涨幅下降"}
    for item in comparison:
        if item["metric"] not in names:
            continue
        lo, hi = item["bootstrap_95"]
        unit = "" if item["metric"] == "overextended_fraction" else " pp"
        lines.append(f"| {item['candidate']} | {names[item['metric']]} | {item['improvement']:+.3f}{unit} | {item['dates_improved']}/{item['dates']} ({item['date_win_fraction']:.1%}) | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "## 决策", ""]
    for item in checks:
        primary = "可以替换主排名" if item["primary_replacement"] else "不替换主排名"
        stage = "可作为上涨初期观察榜" if item["useful_as_entry_stage_watchlist"] else "不进入上涨初期观察榜"
        lines.append(f"- {item['arm']}：{primary}，{stage}；可执行收益改善 {item['executable_improvement_pp']:+.3f} 个百分点，改善日期占比 {item['executable_date_win_fraction']:.1%}，差日期收益改善 {item['q10_improvement_pp']:+.3f} 个百分点，不利波动改善 {item['adverse_improvement_pp']:+.3f} 个百分点，过热占比改善 {item['overextension_reduction']:.1%}。")
    if selected:
        lines += ["", f"选择 {selected} 作为独立的上涨初期观察榜。它不会替换原始收益排名，因为平均收益没有胜出；它在多数日期改善差日期收益和不利波动，并显著减少追高暴露。"]
    lines += ["", "正值代表候选方案更好。不利波动按数值更接近零计为改善；过热占比和入选前涨幅已反向计算。区间跨零不会自动否定多数日期上的改善。", "",
        "原极值情景仍是 T 日最低价到模型选择日期最高价。可执行收益改为首个预测交易日开盘价买入、模型选择日期最高价卖出并扣除 0.25% 成本；它仍未模拟滑点、涨跌停和日内先后顺序。", "",
        "## 限制", "", cfg["limitations"], "", "本次没有修改模型检查点、线上排名指针或定时任务。", ""]
    (root/"report.md").write_text("\n".join(lines))
    for path, expected in read(root/"live-state.json").items():
        if file_hash(Path(path)) != expected:
            raise ValueError("Protected live state changed: "+path)
    write_json(root/"verification.json", dict(passed=True, at=utc_now(), causal_history_rows=causal_rows,
        evaluation_dates=int(daily.date.nunique()), arms=list(ARMS), forecast_chunks_verified=len(verified),
        unknown_outcomes_never_replace_top20=True, live_promoted=False))
    seal(root, "completed.json", [p for p in root.rglob("*") if p.is_file() and p.name not in ("progress.json", "completed.json")],
         passed=True, report="report.md", live_promoted=False)
    return root/"report.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE/"configs/token-early-stage-v1.json")
    print(run(parser.parse_args().config))
