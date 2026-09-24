"""Compare fresh dense token forecasts with causal early-stage reranking."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.early_stage import history_features, path_execution_statistics, rerank
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import valid_bars

BASE = Path(__file__).resolve().parents[1]
ARMS = ("control", "overheat_penalty", "early_stage", "combined")


def read(path: Path):
    return json.loads(path.read_text())


def verify(folder: Path, marker: str = "completed.json"):
    meta = read(folder / marker)
    if not meta.get("passed"):
        raise ValueError(f"Incomplete source: {folder}")
    for name, expected in meta["files"].items():
        if file_hash(folder / name) != expected:
            raise ValueError(f"Changed source: {folder / name}")


def prediction(paths: np.ndarray, cost: float, minimum_paths: int):
    legal = valid_bars(paths).all(axis=-1)
    n = len(paths)
    offset = np.zeros(n, np.int8)
    predicted = np.full(n, np.nan)
    eligible = np.zeros(n, bool)
    for i in range(n):
        samples = paths[i, legal[i]]
        if len(samples) < minimum_paths:
            continue
        returns = np.median(
            samples[:, 1:5, 1] / samples[:, None, 0, 2] - 1 - cost,
            axis=0,
        )
        offset[i] = int(np.argmax(returns)) + 1
        predicted[i] = float(returns[offset[i] - 1])
        eligible[i] = True
    return offset, predicted, eligible


def outcomes(future: np.ndarray, known: np.ndarray, offsets: np.ndarray, cost: float):
    complete = known.all(axis=1)
    row = np.arange(len(future))
    entry_open = future[:, 0, 0]
    entry_low = future[:, 0, 2]
    selected_high = future[row, offsets, 1]
    adverse = np.array(
        [future[i, : offsets[i] + 1, 2].min() / entry_open[i] - 1 for i in row]
    )
    return pd.DataFrame(
        {
            "label_known": complete,
            "actual_extrema_scenario": np.where(
                complete, selected_high / entry_low - 1 - cost, np.nan
            ),
            "actual_execution_return": np.where(
                complete, selected_high / entry_open - 1 - cost, np.nan
            ),
            "actual_adverse": np.where(complete, adverse, np.nan),
        }
    )


def daily_metrics(frame: pd.DataFrame, top_n: int):
    records = []
    for (fold, arm, date), group in frame.groupby(["fold", "arm", "date"], sort=True):
        top = group.sort_values("rank").head(top_n)
        visible = top[top.label_known]
        records.append(
            {
                "fold": fold,
                "arm": arm,
                "date": date,
                "pool_rows": len(group),
                "top_rows": len(top),
                "known_rows": len(visible),
                "extrema_scenario_pct": visible.actual_extrema_scenario.mean() * 100,
                "execution_return_pct": visible.actual_execution_return.mean() * 100,
                "execution_q10_pct": visible.actual_execution_return.quantile(0.1) * 100,
                "positive_fraction": (visible.actual_execution_return > 0).mean(),
                "adverse_excursion_pct": visible.actual_adverse.mean() * 100,
                "overextended_fraction": top.overextended.mean(),
                "right_side_fraction": top.right_side.mean(),
                "mean_prior_return_10_pct": top.return_10.mean() * 100,
                "predicted_return_pct": top.predicted.mean() * 100,
            }
        )
    return pd.DataFrame(records)


def compare(daily: pd.DataFrame, cfg: dict):
    metrics = [
        "extrema_scenario_pct",
        "execution_return_pct",
        "execution_q10_pct",
        "positive_fraction",
        "adverse_excursion_pct",
        "overextended_fraction",
        "right_side_fraction",
        "mean_prior_return_10_pct",
    ]
    output = []
    keys = ["fold", "date"]
    control = daily[daily.arm.eq("control")].set_index(keys).sort_index()
    for arm in ARMS[1:]:
        candidate = daily[daily.arm.eq(arm)].set_index(keys).sort_index()
        if not candidate.index.equals(control.index):
            raise ValueError("Unpaired comparison dates")
        for metric in metrics:
            delta = candidate[metric] - control[metric]
            if metric in ("overextended_fraction", "mean_prior_return_10_pct"):
                delta = -delta
            values = delta.dropna().to_numpy()
            n = len(values)
            block = min(cfg["bootstrap_block_dates"], n)
            rng = np.random.default_rng(271828)
            starts = rng.integers(
                0, n, (cfg["bootstrap_replicates"], math.ceil(n / block))
            )
            indices = (
                (starts[:, :, None] + np.arange(block)) % n
            ).reshape(len(starts), -1)[:, :n]
            boot = values[indices].mean(axis=1)
            output.append(
                {
                    "candidate": arm,
                    "control": "control",
                    "metric": metric,
                    "improvement": float(values.mean()),
                    "dates": n,
                    "dates_improved": int((values > 1e-10).sum()),
                    "date_win_fraction": float((values > 1e-10).mean()),
                    "bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
                }
            )
    return output


def run(config_path: Path):
    cfg = read(config_path)
    root = BASE / cfg["output"]
    experiment = BASE / cfg["experiment"]
    data_root = BASE / cfg["data"]
    price_root = BASE / cfg["price_input"]
    if (root / "completed.json").exists():
        verify(root)
        return root / "report.md"
    if root.exists():
        raise ValueError("Output already exists; retain it or choose a new version")
    verify(data_root)
    rows = pd.read_parquet(data_root / "rows.parquet")
    future = np.load(data_root / "future.npy", mmap_mode="r")
    known = np.load(data_root / "valid.npy", mmap_mode="r")
    raw = np.load(price_root / "values.npy", mmap_mode="r")
    root.mkdir(parents=True)
    write_json(root / "protocol.json", cfg)
    source_hashes = {
        str(config_path.resolve()): file_hash(config_path),
        str((data_root / "completed.json").resolve()): file_hash(data_root / "completed.json"),
        str((price_root / "values.npy").resolve()): file_hash(price_root / "values.npy"),
    }
    ranked_days = []
    causal_rows = 0
    for fold in cfg["folds"]:
        forecast_root = experiment / fold / cfg["model"] / "forecast"
        verify(forecast_root)
        source_hashes[str((forecast_root / "completed.json").resolve())] = file_hash(
            forecast_root / "completed.json"
        )
        ids = np.load(forecast_root / "row-ids.npy")
        paths = np.load(forecast_root / "paths.npy", mmap_mode="r")
        fold_rows = rows.iloc[ids].reset_index(drop=True)
        for date, positions in fold_rows.groupby("date", sort=True).groups.items():
            local = np.asarray(list(positions), dtype=int)
            day_ids = ids[local]
            day_paths = np.asarray(paths[local])
            sell_offset, predicted, eligible = prediction(
                day_paths, cfg["cost"], cfg["minimum_legal_paths"]
            )
            if eligible.sum() < cfg["top_n"]:
                raise ValueError(f"Too few eligible forecasts in {fold} {date}")
            local = local[eligible]
            day_ids = day_ids[eligible]
            day_paths = day_paths[eligible]
            sell_offset = sell_offset[eligible]
            predicted = predicted[eligible]
            selected = fold_rows.iloc[local].reset_index(drop=True)
            stock = selected.stock_index.to_numpy(int)
            time = selected.date_index.to_numpy(int)
            history = raw[stock[:, None], time[:, None] + np.arange(-59, 1)].copy()
            features = history_features(history).reset_index(drop=True)
            execution = path_execution_statistics(
                [day_paths], sell_offset, cfg["cost"], cfg["minimum_legal_paths"]
            )
            candidate = pd.DataFrame(
                {
                    "row_id": day_ids,
                    "instrument_id": selected.instrument_id.to_numpy(),
                    "date": date,
                    "fold": fold,
                    "sell_offset": sell_offset,
                    "predicted": predicted,
                }
            )
            candidate = pd.concat([candidate, features, execution], axis=1)
            frozen = rerank(candidate, cfg["weights"])
            frozen_keys = frozen[["arm", "row_id", "rank", "sell_offset"]].copy()
            truth = outcomes(
                np.asarray(future[day_ids]),
                np.asarray(known[day_ids]),
                sell_offset,
                cfg["cost"],
            )
            truth.insert(0, "row_id", day_ids)
            ranked = frozen.merge(truth, on="row_id", validate="many_to_one")
            pd.testing.assert_frame_equal(
                frozen_keys.reset_index(drop=True),
                ranked[["arm", "row_id", "rank", "sell_offset"]].reset_index(drop=True),
            )
            ranked_days.append(ranked)
            causal_rows += len(candidate)
    ranked = pd.concat(ranked_days, ignore_index=True)
    ranked.to_parquet(root / "ranked-rows.parquet", index=False)
    daily = daily_metrics(ranked, cfg["top_n"])
    daily.to_csv(root / "daily.csv", index=False)
    summary = daily.groupby("arm").mean(numeric_only=True).reset_index()
    summary.to_csv(root / "summary.csv", index=False)
    comparisons = compare(daily, cfg)
    write_json(root / "comparisons.json", comparisons)
    lookup = {(x["candidate"], x["metric"]): x for x in comparisons}
    decisions = []
    for arm in ARMS[1:]:
        ret = lookup[(arm, "execution_return_pct")]
        ext = lookup[(arm, "overextended_fraction")]
        useful = (
            ret["improvement"] > 0
            and ret["date_win_fraction"] > 0.5
            and ext["improvement"] >= 0
        )
        decisions.append(
            {
                "arm": arm,
                "useful": useful,
                "execution_improvement_pp": ret["improvement"],
                "execution_date_win_fraction": ret["date_win_fraction"],
                "overextension_reduction": ext["improvement"],
            }
        )
    selected = max(
        (x for x in decisions if x["useful"]),
        key=lambda x: x["execution_improvement_pp"],
        default=None,
    )
    write_json(
        root / "decision.json",
        {
            "rule": cfg["decision_rule"],
            "arms": decisions,
            "selected_for_research": selected["arm"] if selected else None,
            "live_promotion": False,
        },
    )
    table = summary.set_index("arm")
    lines = [
        "# 新训练基线与上涨初期重排对照",
        "",
        "本次重新训练冻结的 372 万参数 token Transformer，并在完全相同的股票、日期、预测路径和未来标签上比较四种排序。上涨初期方案只是因果重排，不修改模型预测。",
        "",
        "## 汇总",
        "",
        "| 排序 | T+1 开盘到选定日最高价收益 | T+1 最低价到选定日最高价情景 | 10%分位收益 | 不利波动 | 过热占比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        value = table.loc[arm]
        lines.append(
            f"| {arm} | {value.execution_return_pct:.2f}% | "
            f"{value.extrema_scenario_pct:.2f}% | {value.execution_q10_pct:.2f}% | "
            f"{value.adverse_excursion_pct:.2f}% | {value.overextended_fraction:.1%} |"
        )
    lines += ["", "## 相对原始收益排序", ""]
    for item in decisions:
        lines.append(
            f"- {item['arm']}：平均可执行收益变化 "
            f"{item['execution_improvement_pp']:+.3f} 个百分点，改善日期占比 "
            f"{item['execution_date_win_fraction']:.1%}，过热占比改善 "
            f"{item['overextension_reduction']:.1%}，结论为 "
            f"{'保留研究价值' if item['useful'] else '未达到收益规则'}。"
        )
    lines += [
        "",
        "## 边界",
        "",
        cfg["limitations"],
        "",
        "本实验没有更新线上模型、排名指针或定时任务。",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    write_json(root / "sources.json", source_hashes)
    code = root / "code"
    code.mkdir()
    shutil.copy2(Path(__file__), code / Path(__file__).name)
    shutil.copy2(config_path, code / config_path.name)
    shutil.copy2(BASE / "src/quant_research/early_stage.py", code / "early_stage.py")
    write_json(
        root / "verification.json",
        {
            "passed": True,
            "created_at": utc_now(),
            "evaluation_dates": int(daily[["fold", "date"]].drop_duplicates().shape[0]),
            "input_rows": int(ranked.row_id.nunique()),
            "causal_history_rows": causal_rows,
            "same_forecasts_and_labels_across_arms": True,
            "rank_frozen_before_outcome_join": True,
            "live_promoted": False,
        },
    )
    write_json(
        root / "completed.json",
        {
            "passed": True,
            "files": {
                str(path.relative_to(root)): file_hash(path)
                for path in root.rglob("*")
                if path.is_file() and path.name != "completed.json"
            },
        },
    )
    return root / "report.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=BASE / "configs/retrained-early-stage-v1.json"
    )
    print(run(parser.parse_args().config.resolve()))
