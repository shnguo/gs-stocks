import json
from pathlib import Path

import pandas as pd
import pytest

from quant_research.config import load_config
from quant_research.pool import load_pool, pool_coverage, require_full_pool
from quant_research.storage import file_hash


def catalog(snapshot, root):
    root.mkdir()
    members = []
    for row in snapshot.tables["instruments"].itertuples():
        members.append({"instrument_id": row.instrument_id, "provider_code": row.instrument_id,
                        "exchange": row.exchange, "listed_at": row.listed_at,
                        "delisted_at": row.delisted_at or None, "in_selection_pool": True})
    pool = {"scope": "all_a_shares", "membership_source": "ashare_universe_snapshots",
            "source_member_count": len(members), "selection_members": members,
            "historical_candidates": members, "historical_identity_verified": False,
            "as_of": snapshot.dates[-1]}
    (root / "universe.json").write_text(json.dumps(pool))
    (root / "acquisition-plan.json").write_text("[]")
    for name in ["selection-members.csv", "historical-candidates.csv"]:
        pd.DataFrame(members).to_csv(root / name, index=False)
    files = {p.name: file_hash(p) for p in root.iterdir()}
    manifest = {"scope": "all_a_shares", "membership_source": "ashare_universe_snapshots",
                "not_a_personal_watchlist": True, "files": files,
                "universe_id": files["universe.json"], "counts": {"selection_members": len(members)}}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_dropped_stock_remains_in_full_pool_and_blocks_the_normal_runner(snapshot, tmp_path):
    path = catalog(snapshot, tmp_path / "catalog")
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    snapshot.tables["instruments"] = snapshot.tables["instruments"].iloc[1:]
    snapshot.tables["bars"] = snapshot.tables["bars"].loc[
        snapshot.tables["bars"].instrument_id != symbol]
    result = pool_coverage(snapshot, path)
    assert result["selection_pool_members"] == 6
    assert result["snapshot_instruments"] == 5
    assert result["missing_instruments"] == [symbol]
    assert not result["full_membership_present"]
    assert next(r for r in result["coverage"] if r["instrument_id"] == symbol)["retained_in_expected_pool"]
    with pytest.raises(ValueError, match="Full-market coverage is incomplete"):
        require_full_pool(snapshot, {"universe_catalog": str(path)})


def test_missing_prices_do_not_remove_membership_or_count_as_complete(snapshot, tmp_path):
    path = catalog(snapshot, tmp_path / "catalog")
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    snapshot.tables["bars"] = snapshot.tables["bars"].loc[
        snapshot.tables["bars"].instrument_id != symbol]
    result = pool_coverage(snapshot, path)
    assert result["full_membership_present"]
    assert not result["full_calendar_day_coverage"]
    assert result["selection_pool_members"] == 6
    assert result["instruments_with_data_gaps"] == [symbol]


def test_catalog_and_all_three_markets_are_required_and_hash_checked(snapshot, tmp_path):
    path = catalog(snapshot, tmp_path / "catalog")
    assert require_full_pool(snapshot, {"universe_catalog": str(path)})["full_calendar_day_coverage"]
    with (path.parent / "selection-members.csv").open("a") as handle:
        handle.write("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pool(path)
    with pytest.raises(ValueError, match="catalog is required"):
        require_full_pool(snapshot, {})


def test_personal_list_scope_is_not_a_supported_research_configuration(tmp_path):
    source = (Path(__file__).parents[1] / "configs/research.toml").read_text()
    path = tmp_path / "favorites.toml"
    path.write_text(source.replace('universe_scope = "all_a_shares"', 'universe_scope = "favorites"'))
    with pytest.raises(ValueError, match="personal stock lists are unsupported"):
        load_config(path)


def test_small_sample_needs_explicit_fixture_mode_before_any_training(snapshot, config, tmp_path):
    from quant_research.experiment import run_experiment
    from quant_research.splits import Fold

    config.pop("universe_catalog", None)
    output = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="catalog is required"):
        run_experiment(snapshot, config, Fold("2023-01-01", "2023-06-01", "2024-01-01", "2024-02-01"),
                       5, 17, output, engineering=True, training_only=True)
    assert not output.exists()
