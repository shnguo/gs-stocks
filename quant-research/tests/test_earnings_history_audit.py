import json

import pandas as pd
import pytest
from audit_earnings_history import bound_snapshot, observed_calendar, profile_prices

from quant_research.fixtures import create_fixture
from quant_research.storage import file_hash


def test_streamed_profile_excludes_sealed_rows_and_detects_cross_batch_duplicates(tmp_path):
    root = tmp_path / "history"
    snapshot = create_fixture(root / "tables", root / "snapshot", stocks=3, days=180)
    bars_path = root / "snapshot/bars.parquet"
    bars = pd.read_parquet(bars_path)
    sealed_start, sealed_end = snapshot.dates[50], snapshot.dates[100]
    allowed_date = snapshot.dates[110]
    bars.loc[bars.date.eq(allowed_date), "factor"] = float("nan")
    duplicate = bars.loc[bars.date.eq(allowed_date)].iloc[[0]]
    bars = pd.concat([bars, duplicate], ignore_index=True)
    bars.to_parquet(bars_path, index=False)
    current = profile_prices(root, sealed_start, sealed_end, False, batch_size=7)
    assert current["inspected_rows"] == (50 + 79) * 3 + 1
    assert current["nonfinite_factor_rows"] == 4
    assert current["missing_factor_instruments"] == 3
    assert current["duplicate_business_key_excess"] == 1
    assert not any(sealed_start <= d <= sealed_end for d in current["observed_dates"])
    history = profile_prices(root, sealed_start, sealed_end, True, batch_size=7)
    assert history["inspected_rows"] == 150
    assert history["nonfinite_factor_rows"] == 0 and history["duplicate_business_key_excess"] == 0
    assert history["inspected_last_date"] == snapshot.dates[49]
    assert not history["returns_or_targets_read"] and not history["formal_profit_qualified"]


def test_source_binding_rejects_table_tampering_and_coordinated_manifest_identity_change(tmp_path):
    root = tmp_path / "history"
    create_fixture(root / "tables", root / "snapshot", stocks=3, days=40)
    bound_snapshot(root)
    path = root / "snapshot/bars.parquet"
    bars = pd.read_parquet(path)
    bars.loc[0, "close"] *= 1.01
    bars.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        bound_snapshot(root)
    manifest_path = root / "snapshot/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["bars.parquet"] = file_hash(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        bound_snapshot(root)


def test_missing_whole_market_day_and_outside_calendar_prices_are_not_compressed():
    calendar = pd.DataFrame(dict(date=["2025-01-02", "2025-01-03", "2025-01-06"], is_open=True))
    profile = dict(inspected_first_date="2025-01-02", inspected_last_date="2025-01-06",
                   observed_dates=calendar.date.tolist())
    assert observed_calendar(calendar, profile) == calendar.date.tolist()
    for dates in [["2025-01-02", "2025-01-06"], ["2025-01-02", "2025-01-03", "2025-01-04", "2025-01-06"]]:
        with pytest.raises(ValueError, match="do not compress"):
            observed_calendar(calendar, {**profile, "observed_dates": dates})
