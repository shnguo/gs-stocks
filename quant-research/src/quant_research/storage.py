from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import digest

TABLES = {
    "instruments": {"instrument_id", "exchange", "board", "listed_at", "delisted_at", "industry"},
    "calendar": {"date", "exchange", "is_open"},
    "bars": {"instrument_id", "date", "open", "high", "low", "close", "volume", "amount",
             "factor", "upper_limit", "lower_limit"},
    "risk_events": {"instrument_id", "start_date", "end_date", "status", "known_at"},
    "risk_coverage": {"date", "exchange", "complete", "known_at"},
    "rules": {"board", "start_date", "end_date", "min_buy", "buy_step", "sell_step", "t_plus",
              "buy_fee", "sell_fee", "stamp_duty", "minimum_fee", "slippage_bps"},
    "actions": {"instrument_id", "ex_date", "pay_date", "split_ratio", "cash_per_share"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_table(root: Path, name: str) -> pd.DataFrame:
    path = root / f"{name}.parquet"
    if path.exists():
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(root / f"{name}.csv", keep_default_na=False)
    missing = TABLES[name] - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    for col in ["date", "listed_at", "delisted_at", "start_date", "end_date", "ex_date", "pay_date"]:
        if col in frame:
            frame[col] = frame[col].fillna("").astype(str)
    if "known_at" in frame:
        frame["known_at"] = pd.to_datetime(frame.known_at, utc=True, errors="raise")
        if frame.known_at.isna().any():
            raise ValueError(f"{name}: missing information availability timestamp")
    for col in ["source_trade_status", "source_is_st"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col].replace("", float("nan")), errors="raise")
    return frame


def freeze(source: Path, destination: Path, declaration: dict) -> "Snapshot":
    """Import canonical research tables without claiming they are point-in-time verified."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {destination}")
    required = {"vintage", "source", "st_history_verified", "actions_verified", "rules_verified",
                "historical_universe_verified", "industry_history_verified", "synthetic"}
    if required - declaration.keys():
        raise ValueError(f"Missing dataset declarations: {sorted(required - declaration.keys())}")
    if declaration["vintage"] not in {"verified_pit", "reconstructed", "synthetic"}:
        raise ValueError("Invalid vintage declaration")
    for key in required - {"vintage", "source"}:
        if not isinstance(declaration[key], bool):
            raise ValueError(f"Dataset declaration {key} must be an explicit boolean")
    if declaration["synthetic"] != (declaration["vintage"] == "synthetic"):
        raise ValueError("Contradictory synthetic/vintage declarations")
    if declaration.get("purpose", "full_experiment") not in {"training", "full_experiment"}:
        raise ValueError("Invalid snapshot purpose")
    tables = {name: read_table(source, name) for name in TABLES}
    validate_tables(tables, require_execution_rules=declaration.get("purpose") != "training")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=destination.parent))
    try:
        hashes = {}
        for name, frame in tables.items():
            path = staging / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            hashes[path.name] = file_hash(path)
        identity = {"schema_version": 1, "declaration": declaration, "files": hashes}
        manifest = {**identity, "dataset_id": digest(identity), "created_at": utc_now()}
        write_json(staging / "manifest.json", manifest)
        os.rename(staging, destination)
    except BaseException:
        import shutil
        shutil.rmtree(staging)
        raise
    return Snapshot(destination)


def validate_tables(tables: dict[str, pd.DataFrame], *, require_execution_rules: bool = True) -> None:
    for name, frame in tables.items():
        for col in ["date", "listed_at", "delisted_at", "start_date", "end_date", "ex_date", "pay_date"]:
            if col in frame:
                nonempty = frame.loc[frame[col] != "", col]
                if not nonempty.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
                    raise ValueError(f"{name}: {col} must be ISO calendar dates")
                pd.to_datetime(nonempty, format="%Y-%m-%d", errors="raise")
    for name, keys in {"instruments": ["instrument_id"], "calendar": ["date", "exchange"],
                       "bars": ["instrument_id", "date"],
                       "risk_coverage": ["date", "exchange"]}.items():
        if tables[name].duplicated(keys).any():
            raise ValueError(f"{name}: duplicate business key")
    instruments = tables["instruments"]
    if instruments.empty or (instruments.instrument_id.astype(str).str.strip() == "").any():
        raise ValueError("Missing historical instrument identities")
    if (instruments.listed_at == "").any():
        raise ValueError("Missing listing date")
    if not set(instruments.exchange) <= {"XSHG", "XSHE", "XBSE"}:
        raise ValueError("Non-A-share exchange in research snapshot")
    ids = set(instruments.instrument_id)
    for name in ["bars", "risk_events", "actions"]:
        if not set(tables[name].instrument_id) <= ids:
            raise ValueError(f"{name}: unknown instrument")
    bars = tables["bars"]
    for col in ["sequence_id", "label_sequence_id"]:
        if col in bars and (bars[col].isna().any() or
                            bars[col].astype(str).str.strip().eq("").any()):
            raise ValueError("bars: missing sequence identity")
    for status in ["source_trade_status", "source_is_st"]:
        if status in bars and not set(bars[status].dropna()) <= {0, 1}:
            raise ValueError(f"bars: {status} must be 0/1 or explicitly unknown")
    numeric = ["open", "high", "low", "close", "volume", "amount", "factor"]
    for col in numeric:
        bars[col] = pd.to_numeric(bars[col], errors="raise")
    if not bars.empty:
        import numpy as np
        if not np.isfinite(bars[numeric].to_numpy()).all():
            raise ValueError("bars: non-finite required value")
        if (bars[["open", "high", "low", "close", "factor"]] <= 0).any().any():
            raise ValueError("bars: non-positive price/factor")
        if (bars[["volume", "amount"]] < 0).any().any():
            raise ValueError("bars: negative volume/amount")
        if ((bars.high < bars[["open", "close", "low"]].max(axis=1)) |
                (bars.low > bars[["open", "close", "high"]].min(axis=1))).any():
            raise ValueError("bars: invalid OHLC ordering")
    if not set(tables["risk_events"].status) <= {"normal", "ST", "*ST", "unknown"}:
        raise ValueError("risk_events: invalid risk status")
    for name in ["risk_events", "rules"]:
        f = tables[name]
        if ((f.end_date != "") & (f.end_date < f.start_date)).any():
            raise ValueError(f"{name}: reversed effective interval")
    for name, col in [("calendar", "is_open"), ("risk_coverage", "complete")]:
        if tables[name][col].isna().any() or not set(tables[name][col]) <= {True, False, 0, 1}:
            raise ValueError(f"{name}: {col} must be a boolean or 0/1")
    import numpy as np
    rules = tables["rules"]
    for col in ["min_buy", "buy_step", "sell_step", "t_plus", "buy_fee", "sell_fee",
                "stamp_duty", "minimum_fee", "slippage_bps"]:
        rules[col] = pd.to_numeric(rules[col], errors="raise")
        if not np.isfinite(rules[col]).all() or (rules[col] < 0).any():
            raise ValueError(f"rules: invalid {col}")
    for col in ["min_buy", "buy_step", "sell_step", "t_plus"]:
        if (rules[col] < 1).any() or (rules[col] % 1 != 0).any():
            raise ValueError(f"rules: {col} must be a positive integer")
    if require_execution_rules and not set(instruments.board) <= set(rules.board):
        raise ValueError("Missing board execution rules")
    for _, group in rules.groupby("board"):
        group = group.sort_values("start_date")
        for previous, following in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            if previous.end_date == "" or previous.end_date >= following.start_date:
                raise ValueError("Overlapping board execution rules")
    actions = tables["actions"]
    for col in ["split_ratio", "cash_per_share"]:
        actions[col] = pd.to_numeric(actions[col], errors="raise")
        if not np.isfinite(actions[col]).all():
            raise ValueError(f"actions: invalid {col}")
    if (actions.split_ratio <= 0).any() or (actions.cash_per_share < 0).any():
        raise ValueError("actions: invalid split or cash entitlement")
    if ((actions.ex_date == "") |
            ((actions.pay_date == "") & require_execution_rules) |
            ((actions.pay_date != "") & (actions.pay_date < actions.ex_date))).any():
        raise ValueError("actions: missing or reversed entitlement/pay dates")
    if actions.duplicated(["instrument_id", "ex_date"]).any():
        raise ValueError("actions: aggregate same-day actions before snapshot creation")
    calendar = tables["calendar"]
    if not set(calendar.exchange) <= {"XSHG", "XSHE", "XBSE"}:
        raise ValueError("calendar: unknown exchange")
    open_keys = set(zip(calendar.loc[calendar.is_open.astype(bool), "date"],
                        calendar.loc[calendar.is_open.astype(bool), "exchange"]))
    exchanges = dict(zip(instruments.instrument_id, instruments.exchange))
    if any((r.date, exchanges[r.instrument_id]) not in open_keys for r in bars.itertuples()):
        raise ValueError("bars: row outside its exchange trading calendar")


class Snapshot:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if set(self.manifest["files"]) != {f"{name}.parquet" for name in TABLES}:
            raise ValueError("Snapshot manifest does not hash every required table")
        for name, expected in self.manifest["files"].items():
            if Path(name).name != name or file_hash(self.path / name) != expected:
                raise ValueError(f"Snapshot hash mismatch: {name}")
        identity = {k: self.manifest[k] for k in ["schema_version", "declaration", "files"]}
        if digest(identity) != self.manifest["dataset_id"]:
            raise ValueError("Snapshot manifest identity mismatch")
        self.tables = {name: read_table(self.path, name) for name in TABLES}
        purpose = self.manifest["declaration"].get("purpose", "full_experiment")
        if purpose not in {"training", "full_experiment"}:
            raise ValueError("Invalid snapshot purpose")
        validate_tables(self.tables, require_execution_rules=purpose != "training")

    @property
    def dataset_id(self) -> str:
        return self.manifest["dataset_id"]

    @property
    def dates(self) -> list[str]:
        calendar = self.tables["calendar"]
        return sorted(calendar.loc[calendar.is_open.astype(bool), "date"].unique())

    def formal_blockers(self, min_years: int = 10, *, require_st_history: bool = True,
                        require_portfolio_data: bool = True) -> list[str]:
        d = self.manifest["declaration"]
        reasons = []
        if d["synthetic"]:
            reasons.append("synthetic_data_is_engineering_validation_only")
        if d["vintage"] != "verified_pit":
            reasons.append("historical_information_vintage_unverified")
        for key in ["st_history_verified", "actions_verified", "rules_verified",
                    "historical_universe_verified", "industry_history_verified"]:
            if key == "st_history_verified" and not require_st_history:
                continue
            if key in {"rules_verified", "industry_history_verified"} and not require_portfolio_data:
                continue
            if not d[key]:
                reasons.append(key + "_missing")
        if require_portfolio_data and d.get("purpose") == "training":
            reasons.append("training_snapshot_has_no_execution_contract")
        dates = self.dates
        if not dates or (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days < min_years * 365:
            reasons.append("insufficient_multiyear_coverage")
        return reasons
