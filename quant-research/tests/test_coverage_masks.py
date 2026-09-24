import json

import numpy as np
import pandas as pd
import pytest

from quant_research.coverage import load_coverage
from quant_research.features import build_panel
from quant_research.storage import file_hash


def evidence(snapshot, root, symbol, dates):
    root.mkdir()
    ledger = root / "gaps.jsonl"
    ledger.write_text("\n".join(json.dumps({"instrument_id": symbol, "date": d,
        "classification": "source_verified_suspension",
        "resolution": {"classification": "source_verified_suspension"}}) for d in dates))
    obs = root / "observation-exclusions.csv"
    obs.write_text("instrument_id,date,reason\n")
    manifest = root / "export-manifest.json"
    manifest.write_text(json.dumps({"sources": {"gap_ledger": {"path": str(ledger),
        "sha256": file_hash(ledger)}}, "files": {obs.name: {"sha256": file_hash(obs)}}}))
    snapshot.manifest["declaration"]["canonical_export_manifest_sha256"] = file_hash(manifest)
    return {"export_manifest": str(manifest)}


def test_verified_halt_is_masked_without_fabricating_execution_or_raw_prices(snapshot, tmp_path):
    symbol = "synthetic.xshe.000001"
    halted = snapshot.dates[200:270]
    descriptor = evidence(snapshot, tmp_path / "proof", symbol, halted)
    before = snapshot.tables["bars"].copy()
    snapshot.tables["bars"] = before.loc[~((before.instrument_id == symbol) & before.date.isin(halted))]
    original = snapshot.tables["bars"].copy()
    panel = build_panel(snapshot, 5, risk_policy="include", masked=True, coverage_evidence=descriptor)
    row = panel.samples.loc[(panel.samples.instrument_id == symbol) &
                            (panel.samples.date == halted[20])].iloc[0]
    assert row.label_status == "missing_execution_endpoint"
    assert row.entry_status == "suspended_or_no_turnover"
    assert np.isnan(row.forward_return)
    token = panel.values[int(row.stock_index), int(row.date_index)]
    assert np.isfinite(token).all()
    assert token[panel.feature_names.index("unpriced_suspension")] == 1
    assert token[panel.feature_names.index("suspended")] == 1
    pd.testing.assert_frame_equal(snapshot.tables["bars"], original)


def test_coverage_ledger_tampering_is_rejected(snapshot, tmp_path):
    descriptor = evidence(snapshot, tmp_path / "proof", "synthetic.xshe.000001", snapshot.dates[200:202])
    assert len(load_coverage(snapshot, descriptor)) == 2
    (tmp_path / "proof/gaps.jsonl").write_text("")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_coverage(snapshot, descriptor)


def test_unverified_missing_day_is_not_masked_away(snapshot):
    bars = snapshot.tables["bars"]
    symbol = "synthetic.xshe.000001"
    snapshot.tables["bars"] = bars.loc[~((bars.instrument_id == symbol) & (bars.date == snapshot.dates[200]))]
    panel = build_panel(snapshot, 5, masked=True, risk_policy="include")
    assert not ((panel.samples.instrument_id == symbol) &
                (panel.samples.date == snapshot.dates[220])).any()
