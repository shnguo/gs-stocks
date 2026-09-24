"""Matched test of whether the causal early-stage score adds return information."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from token_features_run import read, verify_manifest
from token_ranking_run import seal

from quant_research.early_stage import market_segment
from quant_research.storage import file_hash, utc_now, write_json

BASE = Path(__file__).resolve().parents[1]
OUTCOMES = ("actual_execution_return", "actual_extrema_scenario",
            "actual_ideal_execution_return", "actual_adverse")
PAIR_COLUMNS = ("date", "market_segment", "treated_row", "control_row",
                "treated_instrument", "control_instrument", "predicted_rank_gap",
                "volatility_rank_gap", "liquidity_rank_gap", "early_stage_score_gap",
                "distance")


def source_availability(cfg):
    """Return the exact inputs needed to execute the frozen matched test."""
    if "cohort_glob" in cfg:
        files = sorted(BASE.glob(cfg["cohort_glob"]))
        rows = [{
            "path": str(path.resolve()),
            "exists": True,
            "expected_sha256": None,
            "actual_sha256": file_hash(path),
            "hash_matches": True,
        } for path in files]
        return {
            "passed": bool(files),
            "required_files": len(rows) if rows else 1,
            "available_files": len(rows),
            "missing_files": 0 if rows else 1,
            "mismatched_files": 0,
            "missing": [] if rows else [{"path": str((BASE/cfg["cohort_glob"]).resolve()),
                                           "exists": False, "expected_sha256": None,
                                           "actual_sha256": None, "hash_matches": True}],
            "mismatched": [],
            "files": rows,
        }
    stage = BASE/cfg["source"]
    forecast = BASE/cfg["forecast_source"]
    required = {
        stage/"completed.json": None,
        stage/"independent-verification.json": None,
        forecast/"completed.json": None,
        forecast/"evaluation-inputs/rows.parquet": None,
        BASE/cfg["price_input"]/"values.npy": None,
    }
    completed = stage/"completed.json"
    if completed.exists():
        marker = read(completed)
        for name, expected in marker.get("files", {}).items():
            if name.startswith("rankings/") and name.endswith(".parquet"):
                required[stage/name] = expected
                day = Path(name).stem
                required[forecast/"validation"/cfg["source_mode"]/(day+".parquet")] = None
    rows = []
    for path, expected in required.items():
        exists = path.exists()
        actual = file_hash(path) if exists else None
        rows.append({
            "path": str(path.resolve()),
            "exists": exists,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "hash_matches": expected is None or actual == expected,
        })
    missing = [x for x in rows if not x["exists"]]
    mismatched = [x for x in rows if x["exists"] and not x["hash_matches"]]
    return {
        "passed": not missing and not mismatched,
        "required_files": len(rows),
        "available_files": sum(x["exists"] for x in rows),
        "missing_files": len(missing),
        "mismatched_files": len(mismatched),
        "missing": missing,
        "mismatched": mismatched,
        "files": rows,
    }


def segment(symbol):
    return market_segment(symbol)


def build_cohort(cfg):
    if "cohort_glob" in cfg:
        files = sorted(BASE.glob(cfg["cohort_glob"]))
        if not files:
            raise FileNotFoundError("No matured prospective matching cohorts exist")
        frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        required = {"local_row", "date", "instrument_id", "market_segment", "predicted_rank",
                    "volatility_rank", "liquidity_rank", "early_stage_rank", "early_stage_score",
                    "right_side", "label_known", *OUTCOMES}
        missing = required-set(frame)
        if missing:
            raise ValueError("Prospective cohort is missing columns: "+str(sorted(missing)))
        if frame.local_row.duplicated().any():
            raise ValueError("Prospective cohort row identities are not unique")
        return frame
    stage = BASE/cfg["source"]
    forecast = BASE/cfg["forecast_source"]
    rows = pd.read_parquet(forecast/"evaluation-inputs/rows.parquet")
    raw = np.load(BASE/cfg["price_input"]/"values.npy", mmap_mode="r")
    parts = []
    for path in sorted((stage/"rankings").glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame = frame[frame.arm.eq("control")].copy().sort_values("local_row")
        source = pd.read_parquet(forecast/"validation"/cfg["source_mode"]/path.name)
        source = source[(source.arm == cfg["source_arm"]) &
                        (source.seed.astype(str) == "ensemble")][["local_row", "volatility_pp"]]
        frame = frame.merge(source, on="local_row", validate="one_to_one")
        inherited = rows.iloc[frame.local_row.to_numpy(int)]
        ss = inherited.stock_index.to_numpy(int)
        tt = inherited.date_index.to_numpy(int)
        history = raw[ss[:, None], tt[:, None]+np.arange(-19, 1)]
        amount = history[..., 5].astype(float)
        good = np.isfinite(amount).all(axis=1) & (amount > 0).all(axis=1)
        log_amount = np.full(len(frame), np.nan)
        log_amount[good] = np.log(amount[good].mean(axis=1))
        frame["log_amount_20"] = log_amount
        frame["market_segment"] = frame.instrument_id.map(segment)
        frame["predicted_rank"] = frame.predicted.rank(method="average", pct=True)
        frame["volatility_rank"] = frame.volatility_pp.rank(method="average", pct=True)
        frame["liquidity_rank"] = frame.log_amount_20.rank(method="average", pct=True)
        frame["early_stage_rank"] = frame.early_stage_score.rank(method="average", pct=True)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def match(frame, specification):
    """Greedy one-to-one matching using forecast-only covariates."""
    pairs = []
    for (date, market), group in frame.groupby(["date", "market_segment"], sort=True):
        usable = group.dropna(subset=["predicted_rank", "volatility_rank", "liquidity_rank"])
        treated = usable[(usable.early_stage_rank >= specification["treated_quantile"]) &
                         usable.right_side & usable.early_stage_score.gt(0)].copy()
        controls = usable[usable.early_stage_rank <= specification["control_quantile"]].copy()
        available = set(controls.index)
        treated = treated.sort_values(["early_stage_score", "predicted_rank", "instrument_id"],
                                      ascending=[False, False, True])
        for row in treated.itertuples():
            if not available:
                break
            candidate = controls.loc[list(available)]
            dp = (candidate.predicted_rank-row.predicted_rank).abs()
            dv = (candidate.volatility_rank-row.volatility_rank).abs()
            dl = (candidate.liquidity_rank-row.liquidity_rank).abs()
            okay = ((dp <= specification["predicted_rank_caliper"]) &
                    (dv <= specification["volatility_rank_caliper"]) &
                    (dl <= specification["liquidity_rank_caliper"]))
            if not okay.any():
                continue
            candidate = candidate[okay].copy()
            candidate["distance"] = ((dp[okay]/specification["predicted_rank_caliper"])**2+
                (dv[okay]/specification["volatility_rank_caliper"])**2+
                (dl[okay]/specification["liquidity_rank_caliper"])**2)
            chosen = candidate.sort_values(["distance", "instrument_id"]).iloc[0]
            available.remove(chosen.name)
            pairs.append(dict(date=date, market_segment=market,
                treated_row=int(row.local_row), control_row=int(chosen.local_row),
                treated_instrument=row.instrument_id, control_instrument=chosen.instrument_id,
                predicted_rank_gap=float(row.predicted_rank-chosen.predicted_rank),
                volatility_rank_gap=float(row.volatility_rank-chosen.volatility_rank),
                liquidity_rank_gap=float(row.liquidity_rank-chosen.liquidity_rank),
                early_stage_score_gap=float(row.early_stage_score-chosen.early_stage_score),
                distance=float(chosen.distance)))
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def attach_outcomes(pairs, cohort):
    columns = ["local_row", "label_known", *OUTCOMES]
    values = cohort[columns].drop_duplicates("local_row").set_index("local_row")
    out = pairs.copy()
    treated = values.loc[out.treated_row].reset_index(drop=True)
    control = values.loc[out.control_row].reset_index(drop=True)
    out["pair_known"] = treated.label_known.to_numpy() & control.label_known.to_numpy()
    for name in OUTCOMES:
        out["treated_"+name] = treated[name].to_numpy()
        out["control_"+name] = control[name].to_numpy()
        out["difference_"+name] = out["treated_"+name]-out["control_"+name]
    out["difference_positive"] = ((out.treated_actual_execution_return > 0).astype(float)-
                                  (out.control_actual_execution_return > 0).astype(float))
    return out


def summarize(name, pairs, cfg):
    known = pairs[pairs.pair_known].copy()
    if known.empty:
        raise ValueError("The matching specification produced no known pairs")
    daily = known.groupby("date").agg(
        pairs=("treated_row", "size"),
        execution_difference=("difference_actual_execution_return", "mean"),
        extrema_difference=("difference_actual_extrema_scenario", "mean"),
        ideal_difference=("difference_actual_ideal_execution_return", "mean"),
        adverse_difference=("difference_actual_adverse", "mean"),
        positive_difference=("difference_positive", "mean"),
        treated_execution=("treated_actual_execution_return", "mean"),
        control_execution=("control_actual_execution_return", "mean"),
    ).reset_index()
    q10 = known.groupby("date").apply(lambda x: pd.Series({
        "treated_q10": x.treated_actual_execution_return.quantile(.1),
        "control_q10": x.control_actual_execution_return.quantile(.1),
    }), include_groups=False).reset_index()
    daily = daily.merge(q10, on="date", validate="one_to_one")
    daily["q10_difference"] = daily.treated_q10-daily.control_q10
    metrics = ["execution_difference", "extrema_difference", "ideal_difference",
               "adverse_difference", "positive_difference", "q10_difference"]
    result = dict(specification=name, pairs=len(pairs), known_pairs=len(known),
        dates=int(daily.date.nunique()), treated_execution_pct=float(known.treated_actual_execution_return.mean()*100),
        control_execution_pct=float(known.control_actual_execution_return.mean()*100),
        treated_q10_pct=float(known.treated_actual_execution_return.quantile(.1)*100),
        control_q10_pct=float(known.control_actual_execution_return.quantile(.1)*100),
        mean_abs_predicted_rank_gap=float(pairs.predicted_rank_gap.abs().mean()),
        mean_abs_volatility_rank_gap=float(pairs.volatility_rank_gap.abs().mean()),
        mean_abs_liquidity_rank_gap=float(pairs.liquidity_rank_gap.abs().mean()),
        mean_early_stage_score_gap=float(pairs.early_stage_score_gap.mean()))
    rng = np.random.default_rng(141421)
    n = len(daily)
    block = min(cfg["bootstrap_block_dates"], n)
    starts = rng.integers(0, n, (cfg["bootstrap_replicates"], math.ceil(n/block)))
    ix = ((starts[:, :, None]+np.arange(block)) % n).reshape(len(starts), -1)[:, :n]
    for metric in metrics:
        values = daily[metric].to_numpy()
        boot = values[ix].mean(axis=1)
        result[metric] = dict(mean=float(values.mean()), median=float(np.median(values)),
            dates_improved=int((values > 1e-12).sum()), date_win_fraction=float((values > 1e-12).mean()),
            bootstrap_95=np.quantile(boot, [.025, .975]).tolist())
    return result, daily


def run(config):
    cfg = read(config)
    root = BASE/cfg["output"]
    source = BASE/cfg["source"] if "source" in cfg else None
    if (root/"completed.json").exists():
        verify_manifest(root, "completed.json")
        return root/"report.md"
    if root.exists():
        raise ValueError("Partial output retained; inspect it or choose another output")
    if source is not None:
        verify_manifest(source, "completed.json")
        source_audit = read(source/"independent-verification.json")
        if (not source_audit["passed"] or
                source_audit["artifact_sha256"] != file_hash(source/"completed.json")):
            raise ValueError("Source early-stage artifact is not independently verified")
    root.mkdir(parents=True)
    write_json(root/"protocol.json", cfg)
    protected = [BASE/"configs/daily-token-v1.json", BASE/"artifacts/daily-token-live-v1/current.json",
                 BASE/"artifacts/daily-token-live-v1/latest.json"]
    write_json(root/"live-state.json", {str(p): file_hash(p) for p in protected})
    if source is not None:
        sources = [config, source/"completed.json", source/"independent-verification.json",
                   BASE/cfg["forecast_source"]/"completed.json",
                   BASE/cfg["price_input"]/"values.npy"]
    else:
        sources = [config, *sorted(BASE.glob(cfg["cohort_glob"]))]
    write_json(root/"sources.json", {str(p.resolve()): file_hash(p) for p in sources})
    shutil.copy2(Path(__file__), root/"token_early_stage_matched.py")
    cohort = build_cohort(cfg)
    if cohort.date.nunique() < cfg["minimum_matched_dates"]:
        raise ValueError(
            f"Only {cohort.date.nunique()} mature dates are available; "
            f"{cfg['minimum_matched_dates']} are required by the frozen protocol"
        )
    cohort.to_parquet(root/"cohort.parquet", index=False)
    summaries = []
    for name, specification in cfg["specifications"].items():
        frozen = match(cohort.drop(columns=["label_known", *OUTCOMES]), specification)
        pairs = attach_outcomes(frozen, cohort)
        keys = pairs[["treated_row", "control_row"]].copy()
        pairs.to_parquet(root/(name+"-pairs.parquet"), index=False)
        summary, daily = summarize(name, pairs, cfg)
        daily.to_csv(root/(name+"-daily.csv"), index=False)
        pd.testing.assert_frame_equal(keys, pairs[["treated_row", "control_row"]])
        summaries.append(summary)
        print("Matched", name, summary["pairs"], "pairs", flush=True)
    write_json(root/"summaries.json", summaries)
    primary = next(x for x in summaries if x["specification"] == cfg["primary_specification"])
    ret, adverse = primary["execution_difference"], primary["adverse_difference"]
    balance = max(primary["mean_abs_predicted_rank_gap"], primary["mean_abs_volatility_rank_gap"],
                  primary["mean_abs_liquidity_rank_gap"])
    enough = (primary["dates"] >= cfg["minimum_matched_dates"] and
              primary["known_pairs"] >= cfg["minimum_known_pairs"])
    incremental = (enough and ret["mean"] > 0 and ret["date_win_fraction"] > .5 and
                   adverse["mean"] >= 0 and balance <= cfg["maximum_mean_rank_gap"])
    risk_value = (primary["q10_difference"]["mean"] > 0 and
                  primary["q10_difference"]["date_win_fraction"] > .5 and adverse["mean"] > 0 and
                  enough and balance <= cfg["maximum_mean_rank_gap"])
    decision = dict(rule=cfg["decision_rule"], primary_specification=cfg["primary_specification"],
        sufficient_sample=enough, matched_dates=primary["dates"], known_pairs=primary["known_pairs"],
        maximum_observed_mean_rank_gap=balance,
        incremental_return_value=incremental, incremental_risk_value=risk_value,
        train_auxiliary_head=incremental, retain_as_risk_filter=risk_value,
        live_promotion=False)
    write_json(root/"decision.json", decision)

    lines = ["# 上涨初期评分的匹配增量检验", "",
        "本实验只比较同一信号日、同一市场板块，并且预测收益排名、历史波动率和成交额流动性相近的股票。匹配过程不读取未来结果。", "",
        "## 结果", "",
        "| 规格 | 配对数 | 上涨初期组收益 | 对照组收益 | 平均差异 | 改善日期 | 不利波动差异 | 差日期收益差异 |", 
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in summaries:
        r, a, q = item["execution_difference"], item["adverse_difference"], item["q10_difference"]
        lines.append(f"| {item['specification']} | {item['known_pairs']:,} | {item['treated_execution_pct']:.2f}% | {item['control_execution_pct']:.2f}% | {r['mean']*100:+.3f} pp | {r['dates_improved']}/{item['dates']} ({r['date_win_fraction']:.1%}) | {a['mean']*100:+.3f} pp | {q['mean']*100:+.3f} pp |")
    lines += ["", "## 主规格匹配质量", "",
        f"- 预测排名平均绝对差：{primary['mean_abs_predicted_rank_gap']:.4f}。",
        f"- 波动率排名平均绝对差：{primary['mean_abs_volatility_rank_gap']:.4f}。",
        f"- 流动性排名平均绝对差：{primary['mean_abs_liquidity_rank_gap']:.4f}。",
        f"- 上涨初期评分平均差：{primary['mean_early_stage_score_gap']:.3f}。", "",
        "## 决策", "",
        f"- 独立收益信息：{'通过' if incremental else '未通过'}。",
        f"- 独立风险筛选价值：{'通过' if risk_value else '未通过'}。",
        f"- 下一步训练上涨初期辅助头：{'是' if decision['train_auxiliary_head'] else '否'}。", "",
        "只有独立收益信息通过时，才把上涨初期分数加入模型训练目标。若仅风险价值通过，它继续作为独立观察榜和风险过滤信息。", "",
        "## 限制", "", cfg["limitations"], ""]
    (root/"report.md").write_text("\n".join(lines))
    for path, expected in read(root/"live-state.json").items():
        if file_hash(Path(path)) != expected:
            raise ValueError("Protected live state changed: "+path)
    write_json(root/"verification.json", dict(passed=True, at=utc_now(), cohort_rows=len(cohort),
        dates=int(cohort.date.nunique()), specifications=list(cfg["specifications"]),
        matching_uses_outcomes=False, live_promoted=False))
    seal(root, "completed.json", [p for p in root.rglob("*") if p.is_file() and p.name not in ("completed.json",)],
         passed=True, report="report.md", live_promoted=False)
    return root/"report.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE/"configs/token-early-stage-matched-v1.json")
    parser.add_argument("--preflight", action="store_true",
        help="Check that every frozen stock-level input still exists without running the analysis")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if args.preflight:
        cfg = read(args.config)
        result = source_availability(cfg)
        result["checked_at"] = utc_now()
        result["config"] = str(args.config.resolve())
        target = args.preflight_output or (BASE/cfg["output"]).with_name(
            Path(cfg["output"]).name+"-preflight.json")
        write_json(target, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        cfg = read(args.config)
        availability = source_availability(cfg)
        if not availability["passed"]:
            raise FileNotFoundError(
                "Frozen matched-test inputs are unavailable: "
                f"{availability['missing_files']} missing and "
                f"{availability['mismatched_files']} hash mismatches. Run with --preflight for details."
            )
        print(run(args.config))
