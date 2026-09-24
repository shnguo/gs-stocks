"""Publish a supplemental early-stage watchlist from an immutable daily run."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from token_features_run import read, verify_manifest

from quant_research.daily_loop import verify, write_manifest
from quant_research.early_stage import (
    history_features,
    market_segment,
    matching_history_controls,
    path_execution_statistics,
    rerank,
)
from quant_research.storage import file_hash, utc_now, write_json


def histories(bars, ranks, signal):
    fields = ["open", "high", "low", "close", "volume", "amount", "factor"]
    bars = bars[bars.date.le(signal) & bars.instrument_id.isin(ranks.instrument_id)].copy()
    bars = bars.sort_values(["instrument_id", "date"])
    groups = {name: value.tail(60) for name, value in bars.groupby("instrument_id", sort=False)}
    raw = np.full((len(ranks), 60, len(fields)), np.nan, np.float64)
    for i, symbol in enumerate(ranks.instrument_id):
        group = groups.get(symbol)
        if group is not None and len(group) == 60:
            raw[i] = group[fields].to_numpy(float)
    return raw


def publish(run, validation):
    run, validation = Path(run), Path(validation)
    verify(run)
    verify_manifest(validation, "completed.json")
    audit = read(validation/"independent-verification.json")
    if (not audit["passed"] or audit["artifact_sha256"] != file_hash(validation/"completed.json")
            or audit["selected_for_shadow"] != "early_stage"):
        raise ValueError("Early-stage validation has not passed independent audit")
    protocol = read(validation/"protocol.json")
    decision = read(validation/"decision.json")
    if decision["selected_for_shadow"] != "early_stage" or decision["live_promotion"]:
        raise ValueError("Validation decision does not authorize a supplemental watchlist")
    info = read(run/"run.json")
    store = run.parent.parent
    pointer = read(store/"reference-publications"/(run.name+".json"))
    primary = Path(pointer["report"])
    verify(primary)
    if pointer["manifest_sha256"] != file_hash(primary/"manifest.json"):
        raise ValueError("Primary publication pointer changed")
    ranks = pd.read_csv(primary/"ranking.csv")
    forecast_rows = pd.read_parquet(run/"inputs/forecast-rows.parquet")
    positions = pd.Series(np.arange(len(forecast_rows)), index=forecast_rows.instrument_id)
    if not ranks.instrument_id.isin(positions.index).all():
        raise ValueError("Published stock is absent from forecast rows")
    indices = positions.loc[ranks.instrument_id].to_numpy(int)
    offsets = ranks.sell_reference_date.map({date: i for i, date in enumerate(info["horizon_dates"])}).to_numpy(int)
    if np.any((offsets < 1) | (offsets > 4)):
        raise ValueError("Primary sell date is outside T+1 through T+4")
    models = [np.asarray(np.load(run/f"forecasts/seed{seed}/paths.npy", mmap_mode="r")[indices])
              for seed in info["checkpoints"]]
    execution = path_execution_statistics(models, offsets, info["cost_scenario"], 16)
    source = Path(read(run/"binding.json")["source"])
    bars = pd.read_parquet(source/"snapshot/bars.parquet")
    raw_history = histories(bars, ranks, info["signal_date"])
    features = history_features(raw_history)
    controls = matching_history_controls(raw_history)
    candidate = pd.DataFrame({
        "instrument_id": ranks.instrument_id,
        "name": ranks.name,
        "primary_rank": ranks["rank"],
        "predicted": ranks.reference_price_net_return,
        "buy_date": ranks.buy_date,
        "sell_reference_date": ranks.sell_reference_date,
    })
    candidate = pd.concat([candidate, features, controls, execution], axis=1)
    ranked = rerank(candidate, protocol["weights"])
    ranked = ranked.merge(ranks[["instrument_id", "buy_reference_price", "sell_reference_price"]],
                          on="instrument_id", validate="many_to_one")
    selected = ranked[ranked.arm.eq("early_stage")].sort_values("rank")
    destination = store/"reports"/(run.name+"-early-stage-v2")
    if (destination/"manifest.json").exists():
        verify(destination)
        delivery = read(destination/"delivery.json")
        if (delivery["run_manifest_sha256"] != file_hash(run/"manifest.json") or
                delivery["validation_sha256"] != file_hash(validation/"completed.json")):
            raise ValueError("Cached early-stage publication source changed")
        return destination/"report.md"
    destination.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(destination/"all-rankings.csv", index=False)
    selected.to_csv(destination/"ranking.csv", index=False)
    cohort = ranked[ranked.arm.eq("control")].copy()
    cohort["date"] = info["signal_date"]
    cohort["local_row"] = (int(info["signal_date"].replace("-", ""))*100000+
                           np.arange(len(cohort), dtype=np.int64))
    cohort["entry_date"] = info["horizon_dates"][0]
    cohort["horizon_end"] = info["horizon_dates"][-1]
    cohort["market_segment"] = cohort.instrument_id.map(market_segment)
    cohort["predicted_rank"] = cohort.predicted.rank(method="average", pct=True)
    cohort["volatility_rank"] = cohort.historical_volatility_20.rank(method="average", pct=True)
    cohort["liquidity_rank"] = cohort.log_amount_20.rank(method="average", pct=True)
    cohort["early_stage_rank"] = cohort.early_stage_score.rank(method="average", pct=True)
    cohort[[
        "local_row", "date", "instrument_id", "market_segment", "entry_date", "sell_reference_date",
        "horizon_end", "predicted", "predicted_execution_return", "predicted_downside",
        "predicted_rank", "historical_volatility_20", "volatility_rank", "log_amount_20",
        "liquidity_rank", "early_stage_score", "early_stage_rank", "right_side",
        "overextended", "extension_risk",
    ]].to_csv(destination/"matching-cohort.csv", index=False)
    lines = ["# 上涨初期观察榜", "",
        f"行情截至：{info['signal_date']} 收盘。预测交易日："+"、".join(info["horizon_dates"])+"。", "",
        "本榜单是原收益排名的补充，不替换原排名。历史验证显示，它降低了追高和下行风险，但平均收益低于原收益榜。", "",
        "排序同时考虑原预测收益和上涨初期评分。上涨初期评分只使用截至信号日收盘的均线位置、短期涨幅、ATR、成交量和近期回踩信息。", "",
        "预测可执行收益以首个预测交易日开盘参考价买入、原模型选择日期的最高价卖出并扣除往返成本；日线模型不能保证成交。", "",
        "| 排名 | 原收益排名 | 股票 | 原参考收益 | 预测可执行收益 | 开盘买入参考价 | 卖出参考价 | 近10日涨幅 | 距MA20 | 阶段评分 | 过热项 | 参考买入日 | 参考卖出日 |", 
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in selected.head(info["top_n"]).itertuples():
        lines.append(f"| {row.rank} | {row.primary_rank} | {row.name}（{row.instrument_id}） | {row.predicted:.2%} | {row.predicted_execution_return:.2%} | {row.execution_entry:.4f} | {row.execution_sell:.4f} | {row.return_10:.2%} | {row.ma20_distance:.2%} | {row.early_stage_score:.3f} | {row.extension_flags} | {row.buy_date} | {row.sell_reference_date} |")
    validation_summary = pd.read_csv(validation/"summary.csv").set_index("arm").loc["early_stage"]
    lines += ["", "## 历史验证摘要", "",
        f"- 71 个历史日期的平均首日开盘可执行收益为 {validation_summary.selected_execution_pct:.2f}%。",
        f"- 可执行收益 10% 分位为 {validation_summary.execution_q10_pct:.2f}%，平均不利波动为 {validation_summary.adverse_excursion_pct:.2f}%。",
        f"- Top20 过热占比为 {validation_summary.overextended_fraction:.1%}，原收益榜为 {pd.read_csv(validation/'summary.csv').set_index('arm').loc['control'].overextended_fraction:.1%}。",
        "- 它没有达到替换原收益榜的标准，因此两份榜单同时保留。", "",
        "阶段评分不等于买入指令。正式入场仍需确认价格站上已完成复权 MA5、支撑或突破回踩、成交量和风险收益比。", ""]
    (destination/"report.md").write_text("\n".join(lines))
    write_json(destination/"delivery.json", dict(at=utc_now(), source_run=str(run.resolve()),
        run_manifest_sha256=file_hash(run/"manifest.json"), primary_publication_sha256=pointer["manifest_sha256"],
        validation=str(validation.resolve()), validation_sha256=file_hash(validation/"completed.json"),
        independent_verification_sha256=file_hash(validation/"independent-verification.json"),
        selected_arm="early_stage", primary_replaced=False, layout_version="early-stage-v2",
        top_n=info["top_n"], signal_date=info["signal_date"],
        horizon_dates=info["horizon_dates"], cost_scenario=info["cost_scenario"],
        matching_cohort_rows=len(cohort)))
    write_manifest(destination)
    write_json(store/"early-stage-publications"/(run.name+".json"), dict(
        report=str(destination.resolve()), manifest_sha256=file_hash(destination/"manifest.json"),
        signal_date=info["signal_date"], primary_replaced=False))
    latest = store/"latest-early-stage.json"
    if not latest.exists() or read(latest)["signal_date"] <= info["signal_date"]:
        write_json(latest, dict(signal_date=info["signal_date"], report=str((destination/"report.md").resolve()),
            ranking=str((destination/"ranking.csv").resolve()), primary_replaced=False))
    return destination/"report.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    print(publish(args.run, args.validation))
