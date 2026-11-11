"""Artificial research bundle for reproducible engineering checks, never market evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .fixtures import create_fixture
from .storage import file_hash, freeze, write_json


def seal_artificial_bundle(root):
    """Bind nested snapshot manifests as well as tables in the outer fixture envelope."""
    root = Path(root)
    write_json(root / "manifest.json", {"files": {str(p.relative_to(root)): file_hash(p)
        for p in sorted(root.rglob("*")) if p.is_file() and p != root / "manifest.json"}})


def create_earnings_bundle(root: Path, protocol_path: Path):
    root = Path(root)
    if root.exists():
        raise FileExistsError("Artificial input bundle must be new")
    root.mkdir(parents=True)
    initial = create_fixture(root / "seed-tables", root / "seed-snapshot", stocks=24, days=190)
    mapping = {"XSHG": "shanghai_main", "XSHE": "shenzhen_main", "XBSE": "beijing"}
    meta, bars = initial.tables["instruments"], initial.tables["bars"]
    meta["board"] = meta.board.map(mapping)
    initial.tables["rules"]["board"] = initial.tables["rules"].board.map(mapping)
    offsets = bars.date.map({d: i for i, d in enumerate(initial.dates)})
    bars["open"], bars["close"] = 10. + offsets * .001, 10. + offsets * .001
    bars["high"], bars["low"] = bars.close + .1, bars.close - .1
    bars["upper_limit"], bars["lower_limit"] = bars.close * 1.1, bars.close * .9
    bars["factor"], bars["volume"], bars["amount"] = 1., 4_000_000., 40_000_000.
    tables = root / "canonical-tables"
    tables.mkdir()
    for name, frame in initial.tables.items():
        frame.to_parquet(tables / (name + ".parquet"), index=False)
    snapshot = freeze(tables, root / "snapshot", initial.manifest["declaration"])
    dates = snapshot.dates
    industries = meta[["instrument_id", "industry"]].assign(
        start_date=dates[0], end_date="", known_at="2020-01-01T00:00:00Z")
    caps = industries.drop(columns="industry").assign(float_market_cap=1_000_000_000.)
    freezes = pd.DataFrame({"date": dates[150:181], "decision_at": [d + "T10:00:00Z" for d in dates[150:181]]})
    events = pd.DataFrame([dict(announcement_id=f"artificial-{index}-{stock}",
        instrument_id=meta.instrument_id.iloc[stock], fiscal_period=period,
        available_at=dates[index - 1] + "T10:00:00Z", document_kind="earnings_forecast",
        profit_metric="attributable_to_parent_same_reporting_period", net_profit_lower_cny=140.,
        net_profit_upper_cny=160., prior_same_period_profit_cny=100., disclosed_yoy_lower=.4,
        disclosed_yoy_upper=.6, financial_fields_verified=True, first_numeric_version_verified=True,
        historical_pit_verified=False, availability_basis="local_observation", pdf_sha256="a" * 64)
        for index, period, stock in [(150, "2023-06-30", 0), (150, "2023-06-30", 1),
                                      (180, "2023-09-30", 0)]])
    model = pd.DataFrame([dict(date=day, instrument_id=symbol, model_rank=i,
        available_at=day + "T09:00:00Z", model_manifest_sha256="b" * 64)
        for day in freezes.date for i, symbol in enumerate(meta.instrument_id)])
    coverage = pd.DataFrame([dict(exchange=exchange, start_date=dates[100], end_date=dates[180],
        known_at=dates[100] + "T09:00:00Z", complete=True, source_sha256="c" * 64)
        for exchange in ["XSHG", "XSHE"]])
    fees = pd.DataFrame([dict(board=board, start_date=dates[0], end_date="", commission_rate=.0005,
        minimum_commission=5., transfer_rate=.00001, stamp_duty=.0005, slippage_bps=10.)
        for board in mapping.values()])
    inputs = dict(freezes=freezes, events=events, industries=industries, capitalisations=caps,
                  model_ranks=model, event_coverage=coverage, fees=fees)
    paths = {}
    for name, frame in inputs.items():
        paths[name] = name + ".parquet"
        frame.to_parquet(root / paths[name], index=False)
    (root / "protocol.json").write_bytes(Path(protocol_path).read_bytes())
    spec = dict(schema_version=1, synthetic=True, evidence_role="engineering_only",
        snapshot="snapshot", protocol="protocol.json", tables=paths, start=dates[150], end=dates[180])
    (root / "inputs.json").write_text(json.dumps(spec, indent=2) + "\n")
    seal_artificial_bundle(root)
    return dict(snapshot=snapshot, **inputs, protocol=json.loads(Path(protocol_path).read_text()),
                start=spec["start"], end=spec["end"])
