"""Independent audit for the causal early-stage reranking experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from token_features_run import read, verify_manifest

from quant_research.early_stage import history_features, rerank
from quant_research.storage import file_hash, utc_now, write_json

BASE = Path(__file__).resolve().parents[1]


def verify(root):
    root = Path(root)
    verify_manifest(root, "completed.json")
    cfg = read(root/"protocol.json")
    source = BASE/cfg["source"]
    for path, expected in read(root/"sources.json").items():
        if file_hash(Path(path)) != expected:
            raise ValueError("Source changed: "+path)
    for path, expected in read(root/"live-state.json").items():
        if file_hash(Path(path)) != expected:
            raise ValueError("Protected live state changed: "+path)
    rows = pd.read_parquet(source/"evaluation-inputs/rows.parquet")
    future = np.load(source/"evaluation-inputs/future.npy", mmap_mode="r")
    valid = np.load(source/"evaluation-inputs/valid.npy", mmap_mode="r")
    raw = np.load(BASE/cfg["price_input"]/"values.npy", mmap_mode="r")
    saved_daily = pd.read_csv(root/"daily.csv").set_index(["arm", "date"]).sort_index()
    checked_ranks = checked_features = checked_outcomes = 0
    recomputed = []
    rng = np.random.default_rng(161803)
    for path in sorted((root/"rankings").glob("*.parquet")):
        frame = pd.read_parquet(path)
        base = frame[frame.arm.eq("control")].sort_values("local_row").reset_index(drop=True)
        hidden = base.drop(columns=[c for c in base if c.startswith("actual_") or c == "label_known"])
        again = rerank(hidden, cfg["weights"])
        expected = frame.sort_values(["arm", "rank"])[["arm", "local_row", "rank"]].reset_index(drop=True)
        found = again.sort_values(["arm", "rank"])[["arm", "local_row", "rank"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, found)
        checked_ranks += len(frame)

        pick = rng.choice(len(base), min(8, len(base)), replace=False)
        local = base.iloc[pick]
        inherited = rows.iloc[local.local_row.to_numpy(int)]
        history = raw[inherited.stock_index.to_numpy(int)[:, None],
                      inherited.date_index.to_numpy(int)[:, None]+np.arange(-59, 1)].copy()
        features = history_features(history)
        for name in features:
            if features[name].dtype == bool:
                np.testing.assert_array_equal(features[name], local[name].to_numpy())
            else:
                np.testing.assert_allclose(features[name], local[name].to_numpy(), equal_nan=True,
                                           rtol=1e-6, atol=1e-7)
        checked_features += len(local)

        for r in local.itertuples():
            values = np.asarray(future[r.local_row], float)
            known = bool(np.asarray(valid[r.local_row]).all())
            offset = int(r.sell_offset)
            extrema = values[offset, 1]/values[0, 2]-1-cfg["cost"]
            execution = values[offset, 1]/values[0, 0]-1-cfg["cost"]
            ideal = values[1:, 1].max()/values[0, 0]-1-cfg["cost"]
            adverse = values[:offset+1, 2].min()/values[0, 0]-1
            assert bool(r.label_known) == known
            for actual, expected_value in [(r.actual_extrema_scenario, extrema),
                                           (r.actual_execution_return, execution),
                                           (r.actual_ideal_execution_return, ideal),
                                           (r.actual_adverse, adverse)]:
                if known:
                    np.testing.assert_allclose(actual, expected_value, rtol=1e-12, atol=1e-12)
                else:
                    assert np.isnan(actual)
            checked_outcomes += 1

        for arm, group in frame.groupby("arm"):
            top = group.nsmallest(cfg["top_n"], "rank")
            visible = top[top.label_known]
            pool = group[group.label_known]
            recomputed.append(dict(arm=arm, date=path.stem,
                rows=len(group), top_rows=len(top), top_known=len(visible), top_unknown=cfg["top_n"]-len(visible),
                current_extrema_scenario_pct=visible.actual_extrema_scenario.mean()*100,
                selected_execution_pct=visible.actual_execution_return.mean()*100,
                ideal_execution_pct=visible.actual_ideal_execution_return.mean()*100,
                execution_q10_pct=visible.actual_execution_return.quantile(.1)*100,
                execution_positive_fraction=(visible.actual_execution_return > 0).mean(),
                adverse_excursion_pct=visible.actual_adverse.mean()*100,
                execution_lift_pp=(visible.actual_execution_return.mean()-pool.actual_execution_return.mean())*100,
                overextended_fraction=top.overextended.mean(), right_side_fraction=top.right_side.mean(),
                mean_early_stage_score=top.early_stage_score.mean(),
                mean_prior_return_10_pct=top.return_10.mean()*100,
                predicted_reference_pct=top.predicted.mean()*100,
                predicted_execution_pct=top.predicted_execution_return.mean()*100))
    independent = pd.DataFrame(recomputed).set_index(["arm", "date"]).sort_index()
    pd.testing.assert_frame_equal(saved_daily[independent.columns], independent,
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    decision = read(root/"decision.json")
    assert decision["live_promotion"] is False
    assert decision["selected_for_shadow"] == "early_stage"
    result = dict(passed=True, at=utc_now(), artifact_sha256=file_hash(root/"completed.json"),
        ranks_recomputed_without_outcomes=checked_ranks, causal_feature_rows_recomputed=checked_features,
        raw_future_outcomes_recomputed=checked_outcomes, daily_metrics_recomputed=len(independent),
        selected_for_shadow=decision["selected_for_shadow"], live_promoted=False)
    write_json(root/"independent-verification.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root)
