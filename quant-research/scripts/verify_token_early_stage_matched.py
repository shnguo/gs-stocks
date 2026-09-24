"""Independently replay matching and metrics for the early-stage increment test."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from token_early_stage_matched import OUTCOMES, attach_outcomes, match, summarize
from token_features_run import read, verify_manifest

from quant_research.storage import file_hash, utc_now, write_json

BASE = Path(__file__).resolve().parents[1]


def verify(config):
    cfg = read(config)
    root = BASE/cfg["output"]
    verify_manifest(root, "completed.json")
    cohort = pd.read_parquet(root/"cohort.parquet")
    saved = read(root/"summaries.json")
    by_name = {item["specification"]: item for item in saved}
    checked_pairs = 0
    checked_daily_rows = 0
    for name, specification in cfg["specifications"].items():
        inputs = cohort.drop(columns=["label_known", *OUTCOMES])
        frozen = match(inputs, specification)
        recorded = pd.read_parquet(root/(name+"-pairs.parquet"))
        columns = ["treated_row", "control_row", "predicted_rank_gap",
                   "volatility_rank_gap", "liquidity_rank_gap", "distance"]
        pd.testing.assert_frame_equal(
            frozen[columns].reset_index(drop=True),
            recorded[columns].reset_index(drop=True),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        rebuilt = attach_outcomes(frozen, cohort)
        for column in ["pair_known", *["difference_"+x for x in OUTCOMES]]:
            if column == "pair_known":
                np.testing.assert_array_equal(rebuilt[column], recorded[column])
            else:
                np.testing.assert_allclose(rebuilt[column], recorded[column], equal_nan=True)
        summary, daily = summarize(name, rebuilt, cfg)
        expected = by_name[name]
        if summary.keys() != expected.keys():
            raise AssertionError("Summary schema changed")
        for key, value in summary.items():
            if isinstance(value, dict):
                for metric, number in value.items():
                    np.testing.assert_allclose(number, expected[key][metric], rtol=1e-12, atol=1e-12)
            elif isinstance(value, float):
                np.testing.assert_allclose(value, expected[key], rtol=1e-12, atol=1e-12)
            else:
                assert value == expected[key]
        recorded_daily = pd.read_csv(root/(name+"-daily.csv"))
        pd.testing.assert_frame_equal(daily, recorded_daily, check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-12)
        checked_pairs += len(frozen)
        checked_daily_rows += len(daily)

    for path, expected in read(root/"live-state.json").items():
        if file_hash(Path(path)) != expected:
            raise ValueError("Protected live state changed: "+path)
    result = {
        "passed": True,
        "at": utc_now(),
        "artifact_sha256": file_hash(root/"completed.json"),
        "cohort_rows": len(cohort),
        "dates": int(cohort.date.nunique()),
        "pairs_recomputed_without_outcomes": checked_pairs,
        "daily_metric_rows_recomputed": checked_daily_rows,
        "live_state_unchanged": True,
    }
    write_json(root/"independent-verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=BASE/"configs/token-early-stage-matched-v1.json")
    print(verify(parser.parse_args().config))
