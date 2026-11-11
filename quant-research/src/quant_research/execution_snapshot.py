"""Evidence-gated execution imports and stock-level gaps; never upgrade training flags."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .daily_loop import digest, read, verify
from .storage import TABLES, file_hash, freeze, read_table, utc_now, validate_tables, write_json
from .universe import cutoff, risk_status

COMPONENTS = {
    "universe": "股票身份与上市状态", "industry": "发布时的行业资料",
    "calendar": "交易日历", "risk": "风险状态", "rules": "成交规则与费用",
    "actions": "公司行动及无事件覆盖", "bars": "原始行情与复权因子",
    "limits": "涨跌停价格",
}
REASONS = {
    "evidence_missing": "缺少可核验的来源与覆盖证明",
    "available_too_late": "资料在所需判断时间之后才可见",
    "not_verified": "来源仍为重建或未知资料",
    "row_missing": "缺少对应股票或交易日记录",
    "invalid_value": "数值未知或不符合成交契约",
    "risk_unknown": "当时的风险状态未知",
    "history_changed": "发布前历史与冻结输入不一致",
    "outcome_pending": "该交易日尚未完整收盘",
}


def aware(value):
    result = pd.Timestamp(value)
    if result.tzinfo is None or pd.isna(result):
        raise ValueError("Execution evidence timestamps must include a timezone")
    return result


def member(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Evidence files must be inside their bundle")
    return path


class ExecutionEvidence:
    """Trusted canonical producer assertions are bound to exact archived bytes.

    Hashes prove integrity, not source semantics. Producers must verify derivation,
    exhaustive no-event coverage and historical publication evidence themselves.
    """

    def __init__(self, root, now=None):
        self.root = Path(root).resolve()
        self.spec = read(self.root / "execution-evidence.json")
        if self.spec.get("schema_version") != 1 or type(self.spec.get("synthetic")) is not bool:
            raise ValueError("Unsupported execution evidence schema")
        self.now = aware(now or utc_now())
        self.paths, self.sources = {}, {}
        for source_id, source in self.spec["sources"].items():
            path = member(self.root, source["path"])
            if file_hash(path) != source["sha256"]:
                raise ValueError("Execution raw source hash mismatch")
            observed = aware(source["observed_at"])
            available = aware(source["available_at"])
            if observed > self.now or available > observed:
                raise ValueError("Invalid execution source observation time")
            basis = source["availability_basis"]
            if basis not in {"local_observation", "verified_publication_archive"}:
                raise ValueError("Unsupported execution availability basis")
            if basis == "local_observation" and available != observed:
                raise ValueError("Local observations cannot be backdated")
            if basis == "verified_publication_archive":
                archive = member(self.root, source["publication_proof_path"])
                if file_hash(archive) != source["publication_proof_sha256"]:
                    raise ValueError("Publication archive hash mismatch")
                self.paths["publication:" + source_id] = archive
            if not self.spec["synthetic"] and urlparse(source["source_url"]).scheme not in {"http", "https"}:
                raise ValueError("Real execution evidence requires a source URL")
            self.sources[source_id] = {**source, "available": available, "observed": observed}
            self.paths["raw:" + source_id] = path
        self.tables = {}
        self.table_sources = {}
        for name, spec in self.spec["tables"].items():
            if name not in TABLES:
                raise ValueError("Unknown execution canonical table")
            path = member(self.root, spec["path"])
            if file_hash(path) != spec["sha256"]:
                raise ValueError("Execution canonical table hash mismatch")
            if not spec["source_ids"] or not set(spec["source_ids"]) <= self.sources.keys():
                raise ValueError("Execution table has no archived source")
            producer = member(self.root, spec["normalizer_path"])
            if file_hash(producer) != spec["normalizer_sha256"]:
                raise ValueError("Execution normalizer hash mismatch")
            if path.name != name + ".parquet":
                raise ValueError("Canonical evidence must use named Parquet tables")
            self.tables[name] = read_table(path.parent, name)
            self.table_sources[name] = set(spec["source_ids"])
            self.paths["table:" + name] = path
            self.paths["normalizer:" + name] = producer
        claims = self.spec["coverage"]
        path = member(self.root, claims["path"])
        if file_hash(path) != claims["sha256"]:
            raise ValueError("Execution coverage hash mismatch")
        coverage = pd.read_csv(path, keep_default_na=False)
        required = {"component", "instrument_id", "start_date", "end_date", "known_at", "source_id", "status"}
        if not required <= set(coverage):
            raise ValueError("Incomplete execution coverage schema")
        if not set(coverage.component) <= COMPONENTS.keys() or not set(coverage.status) <= {"verified_pit", "reconstructed", "unknown"}:
            raise ValueError("Invalid execution coverage component/status")
        if (coverage.end_date < coverage.start_date).any():
            raise ValueError("Reversed execution coverage interval")
        for col in ("start_date", "end_date"):
            pd.to_datetime(coverage[col], format="%Y-%m-%d", errors="raise")
        for row in coverage.itertuples():
            if row.source_id not in self.sources:
                raise ValueError("Coverage assertion lacks its archived source")
            source = self.sources[row.source_id]
            at = aware(row.known_at)
            if at < source["available"] or at > source["observed"]:
                raise ValueError("Coverage claim contradicts source availability")
            if (not self.spec["synthetic"] and row.status == "verified_pit" and
                    row.component == "bars" and at < cutoff(row.end_date, "15:00")):
                raise ValueError("Complete-price coverage precedes its final close")
            names = {"universe": ["instruments"], "industry": ["instruments"],
                     "calendar": ["calendar"], "risk": ["risk_events", "risk_coverage"],
                     "rules": ["rules"], "actions": ["actions"],
                     "bars": ["bars"], "limits": ["bars"]}[row.component]
            bound = set().union(*(self.table_sources.get(name, set()) for name in names))
            if row.source_id not in bound:
                raise ValueError("Coverage source is not bound to its canonical table")
        coverage["known_at"] = coverage.known_at.map(aware)
        self.coverage = {(component, symbol): group for (component, symbol), group
                         in coverage.groupby(["component", "instrument_id"])}
        self.paths["coverage"] = path
        self.paths["spec"] = self.root / "execution-evidence.json"
        self.identity = {key: file_hash(path) for key, path in self.paths.items()}

    def check(self, component, symbol, date, at):
        groups = [self.coverage[key] for key in ((component, symbol), (component, "*"))
                  if key in self.coverage]
        if not groups:
            return "evidence_missing"
        rows = pd.concat(groups)
        rows = rows[rows.start_date.le(date) & rows.end_date.ge(date)]
        if rows.empty:
            return "evidence_missing"
        available = rows[rows.known_at.le(at)]
        if available.empty:
            return "available_too_late"
        latest = available[available.known_at.eq(available.known_at.max())]
        if set(latest.status) != {"verified_pit"}:
            return "not_verified"
        return None


def build_execution_snapshot(bundle, source_snapshot, evidence=None):
    """Generate an immutable import attempt, with a formal snapshot only on success."""
    bundle = Path(bundle)
    verify(bundle)
    decision = read(bundle / "decision.json")
    evidence = (ExecutionEvidence(evidence) if evidence is not None and
                not isinstance(evidence, ExecutionEvidence) else evidence)
    tables = {name: frame.copy() for name, frame in source_snapshot.tables.items()}
    if evidence:
        tables.update({name: frame.copy() for name, frame in evidence.tables.items()})
    now = aware(utc_now())
    signal, horizons = decision["signal_date"], decision["horizon_dates"]
    cohort = pd.read_csv(bundle / "cohort.csv")
    top = cohort.sort_values("rank").groupby("arm", sort=False).head(decision["protocol"]["top_n"])
    identity = {"decision_sha256": file_hash(bundle / "manifest.json"),
                "outcome_dataset_id": source_snapshot.dataset_id,
                "evidence": evidence.identity if evidence else None,
                "code_sha256": {name: file_hash(Path(__file__).with_name(name))
                                for name in ("execution_snapshot.py", "storage.py", "universe.py")},
                "completed_sessions": [date for date in horizons if now >= cutoff(date, "15:00")],
                "observed_through": min(str(tables["bars"].date.max()), now.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d"))}
    dest = bundle.parents[2] / "execution-data" / signal / digest(identity)[:16]
    if dest.exists():
        verify(dest)
        return dest
    master = tables["instruments"].set_index("instrument_id")
    bars = tables["bars"].set_index(["instrument_id", "date"])
    if master.index.has_duplicates or bars.index.has_duplicates:
        raise ValueError("Duplicate execution instrument/price keys")
    frozen_master = pd.read_parquet(bundle / "instruments.parquet")
    publication = min(aware(arm["publication"]["published_at"]) for arm in decision["arms"])
    global_checks = []
    gaps = []
    global_reasons = {}

    def proof(component, symbol, date, at):
        if evidence is None:
            return "evidence_missing"
        return evidence.check(component, symbol, date, at)

    # Sector weights require the full frozen universe, not only successful Top 20 stocks.
    for component in ("universe", "industry", "risk"):
        missing = []
        states = risk_status(type("Tables", (), {"tables": tables})(), signal, publication)
        for row in frozen_master.itertuples():
            reason = proof(component, row.instrument_id, signal, publication)
            if row.instrument_id not in master.index:
                reason = "row_missing"
            elif component == "industry" and (pd.isna(master.loc[row.instrument_id, "industry"]) or
                    str(master.loc[row.instrument_id, "industry"]).strip() in {"", "unknown"}):
                reason = "invalid_value"
            elif component == "risk" and states.get(row.instrument_id, "unknown") == "unknown":
                reason = "risk_unknown" if reason is None else reason
            if reason:
                global_reasons[(component, row.instrument_id)] = reason
                missing.append(row.instrument_id)
                gaps.append({"arm": "universe", "instrument_id": row.instrument_id,
                             "component": component, "date": signal, "reason": reason})
        global_checks.append({"component": component, "label": COMPONENTS[component],
                              "required": len(frozen_master), "missing": len(missing)})
    history = pd.read_parquet(bundle / "history.parquet")
    risk_by_date = {date: risk_status(type("Tables", (), {"tables": tables})(), date, cutoff(date, "09:25"))
                    for date in horizons}
    candidates = []
    for row in top.itertuples():
        symbol = row.instrument_id
        own = []
        def add(component, date, reason):
            if reason:
                item = {"component": component, "date": date, "reason": reason}
                own.append(item)
                gaps.append({"arm": row.arm, "instrument_id": symbol, **item})
        for component in ("universe", "industry", "risk"):
            add(component, signal, global_reasons.get((component, symbol)) or
                proof(component, symbol, signal, aware(next(a["publication"]["published_at"]
                    for a in decision["arms"] if a["arm"] == row.arm))))
        lookback = history[history.instrument_id.eq(symbol)].sort_values("date").tail(20)
        if len(lookback) < 20:
            add("bars", signal, "row_missing")
        for previous in lookback.itertuples():
            reason = proof("bars", symbol, previous.date, publication)
            key = (symbol, previous.date)
            if key not in bars.index:
                reason = "row_missing"
            elif not np.isfinite(bars.loc[key, ["close", "amount", "factor"]].to_numpy(float)).all():
                reason = "invalid_value"
            elif not np.allclose(bars.loc[key, ["close", "amount", "factor"]].to_numpy(float),
                                 [previous.close, previous.amount, previous.factor], equal_nan=False):
                reason = "history_changed"
            add("bars", previous.date, reason)
        for date in horizons:
            for component in ("calendar", "risk", "rules", "actions", "limits", "bars"):
                at = now if component == "bars" else cutoff(date, "09:25")
                reason = proof(component, symbol, date, at)
                if component == "bars" and now < cutoff(date, "15:00"):
                    reason = "outcome_pending"
                if component == "calendar":
                    cal = tables["calendar"]
                    exchange = master.loc[symbol, "exchange"] if symbol in master.index else ""
                    if not ((cal.date == date) & (cal.exchange == exchange) & cal.is_open.astype(bool)).any():
                        reason = "row_missing"
                elif component == "risk" and risk_by_date[date].get(symbol, "unknown") == "unknown":
                    reason = reason or "risk_unknown"
                elif component == "rules":
                    rules = tables["rules"]
                    board = master.loc[symbol, "board"] if symbol in master.index else ""
                    legal = rules[(rules.board == board) & rules.start_date.le(date) &
                                  (rules.end_date.eq("") | rules.end_date.ge(date))]
                    if len(legal) != 1:
                        reason = "row_missing"
                elif component in {"bars", "limits"} and reason != "outcome_pending":
                    key = (symbol, date)
                    fields = ["upper_limit", "lower_limit"] if component == "limits" else [
                        "open", "high", "low", "close", "factor", "volume", "amount"]
                    if key not in bars.index:
                        reason = "row_missing"
                    elif not np.isfinite(bars.loc[key, fields].to_numpy(float)).all():
                        reason = "invalid_value"
                    elif component == "limits" and (bars.loc[key, fields] <= 0).any():
                        reason = "invalid_value"
                    elif component == "limits" and bars.loc[key, "lower_limit"] > bars.loc[key, "upper_limit"]:
                        reason = "invalid_value"
                add(component, date, reason)
        candidates.append({"arm": row.arm, "instrument_id": symbol,
                           "name": getattr(row, "name", symbol), "rank": int(row.rank),
                           "ready": not own, "gaps": own,
                           "missing_components": sorted({item["component"] for item in own})})
    # Extra future rows may contain revisions; retain only this frozen experiment window.
    tables["instruments"] = tables["instruments"][tables["instruments"].instrument_id.isin(frozen_master.instrument_id)]
    ids = set(tables["instruments"].instrument_id)
    history_start = str(history.date.min())
    tables["bars"] = tables["bars"][tables["bars"].instrument_id.isin(ids) &
                                    tables["bars"].date.between(history_start, horizons[-1])]
    tables["calendar"] = tables["calendar"][tables["calendar"].date.between(history_start, horizons[-1])]
    for name in ("actions", "risk_events"):
        tables[name] = tables[name][tables[name].instrument_id.isin(ids)]
    ready = not gaps and evidence is not None and set(evidence.tables) == set(TABLES)
    if evidence is not None and set(evidence.tables) != set(TABLES):
        global_checks.append({"component": "canonical_tables", "label": "正式来源表",
                              "required": len(TABLES), "missing": len(set(TABLES) - set(evidence.tables))})
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".execution-", dir=dest.parent))
    try:
        if evidence:
            # Keep proof bytes and normalizers independently of operator source directories.
            for path in set(evidence.paths.values()):
                target = staging / "evidence" / path.relative_to(evidence.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        synthetic = bool(source_snapshot.manifest["declaration"]["synthetic"] or
                         (evidence and evidence.spec["synthetic"]))
        if ready:
            try:
                validate_tables(tables)
            except ValueError as error:
                ready = False
                global_checks.append({"component": "table_validation", "label": "正式表校验",
                                      "required": 1, "missing": 1, "reason": str(error)})
        if ready:
            raw = staging / "canonical"
            raw.mkdir()
            for name, frame in tables.items():
                frame.to_parquet(raw / (name + ".parquet"), index=False)
            freeze(raw, staging / "snapshot", dict(vintage="synthetic" if synthetic else "verified_pit",
                source="hashed_execution_evidence_import", synthetic=synthetic,
                st_history_verified=True, actions_verified=True, rules_verified=True,
                historical_universe_verified=True, industry_history_verified=True,
                execution_scope={"signal_date": signal, "horizon_dates": horizons,
                                 "instruments": sorted(set(top.instrument_id))}))
            shutil.rmtree(raw)
        report = {"schema_version": 1, **identity, "generated_at": utc_now(),
                  "signal_date": signal, "horizon_dates": horizons,
                  "status": "ready" if ready else "blocked", "synthetic": synthetic,
                  "evidence_supplied": evidence is not None, "snapshot_generated": ready,
                  "component_labels": COMPONENTS, "reason_labels": REASONS,
                  "global_checks": global_checks, "candidates": candidates,
                  "acceptance": {"timely_publication": all(a["publication"]["prospective"] for a in decision["arms"]),
                                 "matching_frozen_before_deadline": decision["matching_prospective"],
                                 "outcomes_mature": now >= cutoff(horizons[-1], "15:00"),
                                 "execution_contract_ready": ready}}
        write_json(staging / "gap-report.json", report)
        pd.DataFrame(gaps, columns=["arm", "instrument_id", "component", "date", "reason"]).to_csv(
            staging / "gaps.csv", index=False)
        write_json(staging / "manifest.json", {"files": {
            str(path.relative_to(staging)): file_hash(path) for path in staging.rglob("*")
            if path.is_file() and path != staging / "manifest.json"}})
        staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return dest
