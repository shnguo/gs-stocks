"""Publish a strict, descriptive watchlist for stocks in an early right-side move."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from publish_daily_early_stage import histories

from quant_research.daily_loop import read, verify, write_manifest
from quant_research.early_stage import history_features, select_start_stage
from quant_research.storage import file_hash, utc_now, write_json


def publish(run, config):
    run, config = Path(run), Path(config)
    verify(run)
    criteria = read(config)
    info = read(run / "run.json")
    store = run.parent.parent
    pointer = read(store / "reference-publications" / (run.name + ".json"))
    primary = Path(pointer["report"])
    verify(primary)
    if pointer["manifest_sha256"] != file_hash(primary / "manifest.json"):
        raise ValueError("Primary publication pointer changed")

    destination = store / "reports" / (run.name + "-start-stage-v1")
    if (destination / "manifest.json").exists():
        verify(destination)
        delivery = read(destination / "delivery.json")
        if (
            delivery["run_manifest_sha256"] != file_hash(run / "manifest.json")
            or delivery["config_sha256"] != file_hash(config)
        ):
            raise ValueError("Cached start-stage publication source changed")
        return destination / "report.md"

    ranks = pd.read_csv(primary / "ranking.csv")
    ranks = ranks.rename(columns={"rank": "primary_rank"})
    source = Path(read(run / "binding.json")["source"])
    bars = pd.read_parquet(source / "snapshot/bars.parquet")
    features = history_features(histories(bars, ranks, info["signal_date"]))
    candidates = pd.concat([ranks.reset_index(drop=True), features], axis=1)
    selected = select_start_stage(candidates, criteria)
    top_n = min(criteria["top_n"], len(selected))

    destination.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination / "ranking.csv", index=False)
    candidates.to_parquet(destination / "all-candidates.parquet", index=False)
    lines = [
        "# 上涨初期候选榜",
        "",
        f"行情截至：{info['signal_date']} 收盘。预测交易日：" + "、".join(info["horizon_dates"]) + "。",
        "",
        "本榜单先排除近期涨幅过大和明显过热的股票，再按模型预期净收益率排序。它与收益最大化主榜同时保留，不替换主榜。",
        "",
        "筛选条件：价格位于右侧趋势；过热项为零；近5日涨幅 0% 至 10%；近10日涨幅 0% 至 8%；近20日涨幅 -2% 至 12%；距MA20为 0% 至 5%；20日量比 0.9 至 2.5；距离近5日高点为 -6% 至 0.5%；阶段评分不低于 0.50。",
        "",
        "历史实验没有证明本筛选能超过无约束收益主榜，因此它用于寻找相对低位的观察对象，不是收益更高的承诺。",
        "",
        f"全市场收益排名共 {len(ranks)} 只，本次有 {len(selected)} 只通过上涨初期筛选。",
        "",
        "| 初期排名 | 原收益排名 | 股票 | 预期净收益率 | 近5日涨幅 | 近10日涨幅 | 近20日涨幅 | 距MA20 | 20日量比 | 距近5日高点 | 阶段评分 | 买入日期 | 买入参考价 | 卖出参考日期 | 卖出参考价 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for row in selected.head(top_n).itertuples():
        lines.append(
            f"| {row.start_stage_rank} | {row.primary_rank} | {row.name}（{row.instrument_id}） | "
            f"{row.reference_price_net_return:.2%} | {row.return_5:.2%} | {row.return_10:.2%} | "
            f"{row.return_20:.2%} | {row.ma20_distance:.2%} | {row.volume_ratio_20:.2f} | "
            f"{row.pullback_5:.2%} | {row.early_stage_score:.3f} | {row.buy_date} | "
            f"{row.buy_reference_price:.4f} | {row.sell_reference_date} | {row.sell_reference_price:.4f} |"
        )
    lines += [
        "",
        "预期收益来自模型参考价，已经扣除报告设定的往返成本；极值价格不保证成交。正式入场还需要检查复权MA5、支撑或突破回踩、量价结构、公告和政策风险。",
        "",
        "[完整上涨初期排名](ranking.csv)",
        "",
    ]
    (destination / "report.md").write_text("\n".join(lines))
    write_json(
        destination / "delivery.json",
        {
            "at": utc_now(),
            "signal_date": info["signal_date"],
            "source_run": str(run.resolve()),
            "run_manifest_sha256": file_hash(run / "manifest.json"),
            "primary_publication_sha256": pointer["manifest_sha256"],
            "config": str(config.resolve()),
            "config_sha256": file_hash(config),
            "layout_version": "start-stage-v1",
            "status": criteria["status"],
            "eligible": len(selected),
            "top_n": top_n,
            "primary_replaced": False,
        },
    )
    write_manifest(destination)
    write_json(
        store / "start-stage-publications" / (run.name + ".json"),
        {
            "signal_date": info["signal_date"],
            "report": str((destination / "report.md").resolve()),
            "ranking": str((destination / "ranking.csv").resolve()),
            "manifest_sha256": file_hash(destination / "manifest.json"),
            "primary_replaced": False,
        },
    )
    write_json(
        store / "latest-start-stage.json",
        {
            "signal_date": info["signal_date"],
            "report": str((destination / "report.md").resolve()),
            "ranking": str((destination / "ranking.csv").resolve()),
            "primary_replaced": False,
        },
    )
    latest = read(store / "latest.json")
    if latest["signal_date"] == info["signal_date"]:
        latest["start_stage_report"] = str((destination / "report.md").resolve())
        latest["start_stage_ranking"] = str((destination / "ranking.csv").resolve())
        write_json(store / "latest.json", latest)
    return destination / "report.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(publish(args.run, args.config))
