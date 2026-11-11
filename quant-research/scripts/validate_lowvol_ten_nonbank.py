"""Fixed ten-stock low-volatility annual and non-bank research validation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from compare_lowvol_position_counts_precise import simulate_balanced_target
from compare_strategy_directions import features, load_inputs, save

from quant_research.storage import file_hash, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    cfg = json.loads((root / "protocol.json").read_text())
    prior = Path("artifacts/lowvol-position-count-20261013-v3")
    old = json.loads(Path("artifacts/strategy-screen-20261013-v1/protocol.json").read_text())
    for path, expected in json.loads((root / "implementation-freeze.json").read_text())[
        "files"
    ].items():
        assert file_hash(Path(path)) == expected, path
    previous_freeze = json.loads((prior / "implementation-freeze.json").read_text())
    for path, expected in previous_freeze["files"].items():
        assert file_hash(Path(path)) == expected, path
    bankfile = Path(cfg["bank_identities"])
    assert file_hash(bankfile) == cfg["bank_identities_sha256"]
    bank_ids = set(pd.read_csv(bankfile).instrument_id)
    assert len(bank_ids) == 42
    output = root / "results"
    output.mkdir()
    save(
        output / "run-start.json",
        dict(
            started_at=utc_now(),
            protocol_sha256=file_hash(root / "protocol.json"),
            new_nonbank_results_read_before_freeze=False,
        ),
    )
    dates, symbols, instruments, arrays, actions = load_inputs(old, output)
    scores, adv = features(arrays, old)
    original_score = scores["lowvol60"]
    nonbank = original_score.copy()
    mask = pd.Index(symbols).isin(bank_ids)
    assert mask.sum() == 42
    nonbank[:, mask] = -np.inf
    assert np.array_equal(nonbank[:, ~mask], original_score[:, ~mask])
    assert not np.isfinite(nonbank[:, mask]).any()
    save(
        output / "signal-integrity.json",
        dict(
            excluded_bank_identities=42,
            nonbank_signal_values_identical=True,
            only_change_is_bank_eligibility=True,
            source_feature_implementation_sha256=file_hash(
                Path("scripts/compare_strategy_directions.py")
            ),
        ),
    )
    shutil.copyfile(bankfile, root / "bank-identities.csv")
    p = dict(old, max_positions=10, position_budget_fraction=0.09, retention_buffer_rank=20)
    summaries, reuse_hashes = [], {}
    for scenario in cfg["scenarios"]:
        sc = scenario["name"]
        src = prior / "results" / sc / "lowvol_10"
        dest = output / sc / "original_10"
        shutil.copytree(src, dest)
        item = json.loads((src / "summary.json").read_text())
        item.update(
            strategy="original_10", variant="original", position_count=10, reused_prior_account=True
        )
        # Values and ledgers are copied byte-for-byte; alias metadata is separate.
        summaries.append(item)
        for path in src.iterdir():
            reuse_hashes[str(path)] = file_hash(path)
        result, _ = simulate_balanced_target(
            dates,
            symbols,
            instruments,
            arrays,
            actions,
            nonbank,
            adv,
            p,
            scenario,
            "nonbank_10",
            output / sc / "nonbank_10",
        )
        result.update(variant="nonbank", position_count=10, reused_prior_account=False)
        held = pd.read_parquet(output / sc / "nonbank_10/holdings.parquet")
        assert not held.instrument_id.isin(bank_ids).any()
        summaries.append(result)
        print(
            sc,
            {
                k: result[k]
                for k in [
                    "net_return",
                    "max_drawdown",
                    "mean_exposure",
                    "unresolved_barrier_events",
                ]
            },
            flush=True,
        )
    save(output / "original-account-bindings.json", reuse_hashes)
    save(
        output / "comparison.json",
        dict(
            completed_at=utc_now(),
            summaries=summaries,
            formal_profit_qualified=False,
            sealed_returns_used=False,
        ),
    )
    pd.DataFrame(
        [{k: v for k, v in x.items() if not isinstance(v, (dict, list))} for x in summaries]
    ).to_csv(output / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
