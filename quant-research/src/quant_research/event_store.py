from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

EVENT_TYPES = {
    "earnings_report",
    "earnings_forecast",
    "shareholder_meeting",
    "unlock",
    "buyback",
    "index_rebalance",
    "dividend",
    "convertible_bond",
    "restructuring",
    "approval",
    "major_contract",
    "macro_policy",
}
EVENT_STATUSES = {"scheduled", "confirmed", "postponed", "cancelled", "completed"}
SOURCE_TIERS = {
    "primary_exchange",
    "designated_disclosure",
    "issuer",
    "regulator",
    "index_provider",
    "official_statistics",
}
AVAILABILITY_BASES = {"verified_source_timestamp", "local_observation"}
SCOPES = {"instrument", "market", "industry"}

SCORE_RANGES = {
    "certainty": (0.0, 25.0),
    "transmission": (0.0, 25.0),
    "expectation_gap": (0.0, 20.0),
    "flow_impact": (0.0, 15.0),
    "price_confirmation": (0.0, 15.0),
    "crowding_penalty": (0.0, 20.0),
    "gap_penalty": (0.0, 20.0),
    "liquidity_penalty": (0.0, 20.0),
}

REQUIRED_FIELDS = {
    "event_id",
    "event_type",
    "scope",
    "instrument_id",
    "title",
    "scheduled_date",
    "published_at",
    "observed_at",
    "availability_basis",
    "source_name",
    "source_url",
    "source_tier",
    "status",
}

OPTIONAL_FIELDS = {
    "actual_date",
    "expected_window_end",
    "certainty",
    "transmission",
    "expectation_gap",
    "flow_impact",
    "price_confirmation",
    "crowding_penalty",
    "gap_penalty",
    "liquidity_penalty",
    "support_price",
    "target_price",
    "failure_price",
    "expected_metric",
    "expected_value",
    "actual_value",
    "unit",
    "notes",
    "source_document_sha256",
    "chain_id",
    "chain_stage",
    "next_validation_date",
    "financial_metric",
    "failure_condition",
    "bull_case",
    "base_case",
    "bear_case",
    "price_basis",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _parse_timestamp(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_date(value: object, field: str, *, required: bool = False) -> str:
    if value in {None, ""}:
        if required:
            raise ValueError(f"{field} must be an ISO date")
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc


def _optional_float(value: object, field: str) -> float | None:
    if value in {None, ""}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric or empty") from exc
    if result != result or result in {float("inf"), -float("inf")}:
        raise ValueError(f"{field} must be finite")
    return result


def normalize_event(record: Mapping[str, object]) -> dict[str, object]:
    unknown = set(record) - REQUIRED_FIELDS - OPTIONAL_FIELDS
    missing = REQUIRED_FIELDS - set(record)
    if unknown:
        raise ValueError(f"unknown event fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing event fields: {sorted(missing)}")

    result = {key: ("" if record.get(key) is None else record.get(key))
              for key in REQUIRED_FIELDS | OPTIONAL_FIELDS}
    for field in ["event_id", "event_type", "scope", "title", "availability_basis",
                  "source_name", "source_url", "source_tier", "status"]:
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
        result[field] = result[field].strip()
    instrument_id = result["instrument_id"]
    if not isinstance(instrument_id, str):
        raise ValueError("instrument_id must be a string")
    result["instrument_id"] = instrument_id.strip()

    if result["event_type"] not in EVENT_TYPES:
        raise ValueError("unsupported event_type")
    if result["scope"] not in SCOPES:
        raise ValueError("unsupported scope")
    if result["scope"] == "instrument" and not result["instrument_id"]:
        raise ValueError("instrument events require instrument_id")
    if result["scope"] != "instrument" and result["instrument_id"]:
        raise ValueError("market and industry events must not claim one instrument_id")
    if result["scope"] == "instrument":
        result["instrument_id"] = result["instrument_id"].lower()
        if not re.fullmatch(r"cn\.(xshg|xshe|xbse)\.\d{6}", result["instrument_id"]):
            raise ValueError("instrument_id must use the canonical A-share identifier")
    if result["status"] not in EVENT_STATUSES:
        raise ValueError("unsupported event status")
    if result["source_tier"] not in SOURCE_TIERS:
        raise ValueError("unsupported source tier")
    if result["availability_basis"] not in AVAILABILITY_BASES:
        raise ValueError("unsupported availability basis")
    if not result["source_url"].startswith(("https://", "http://")):
        raise ValueError("source_url must be an HTTP or HTTPS URL")

    result["scheduled_date"] = _parse_date(result["scheduled_date"], "scheduled_date",
                                             required=True)
    result["expected_window_end"] = _parse_date(result["expected_window_end"],
                                                  "expected_window_end")
    result["actual_date"] = _parse_date(result["actual_date"], "actual_date")
    result["next_validation_date"] = _parse_date(result["next_validation_date"],
                                                   "next_validation_date")
    if (result["expected_window_end"] and
            result["expected_window_end"] < result["scheduled_date"]):
        raise ValueError("expected window ends before scheduled date")
    if result["status"] == "completed" and not result["actual_date"]:
        raise ValueError("completed events require actual_date")

    result["published_at"] = _parse_timestamp(result["published_at"], "published_at")
    result["observed_at"] = _parse_timestamp(result["observed_at"], "observed_at")
    if result["observed_at"] < result["published_at"]:
        raise ValueError("observed_at cannot precede published_at")
    result["available_at"] = (result["published_at"]
                              if result["availability_basis"] == "verified_source_timestamp"
                              else result["observed_at"])

    for field, (minimum, maximum) in SCORE_RANGES.items():
        result[field] = _optional_float(result[field], field)
        if result[field] is not None and not minimum <= result[field] <= maximum:
            raise ValueError(f"{field} must be between {minimum:g} and {maximum:g}")
    for field in ["support_price", "target_price", "failure_price", "expected_value",
                  "actual_value"]:
        result[field] = _optional_float(result[field], field)
    for field in ["support_price", "target_price", "failure_price"]:
        if result[field] is not None and result[field] <= 0:
            raise ValueError(f"{field} must be positive")
    for field in ["expected_metric", "unit", "notes", "source_document_sha256", "chain_id",
                  "chain_stage", "financial_metric", "failure_condition", "bull_case",
                  "base_case", "bear_case", "price_basis"]:
        if not isinstance(result[field], str):
            raise ValueError(f"{field} must be a string")
        result[field] = result[field].strip()
    digest = result["source_document_sha256"]
    if digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower())):
        raise ValueError("source_document_sha256 must be a hexadecimal SHA-256")
    result["source_document_sha256"] = digest.lower()
    price_fields_present = any(result[field] is not None
                               for field in ["support_price", "target_price", "failure_price"])
    if price_fields_present and result["price_basis"] != "qfq":
        raise ValueError("event price levels require price_basis=qfq")
    if result["price_basis"] not in {"", "qfq"}:
        raise ValueError("unsupported event price basis")
    return result


def load_event_records(path: Path) -> list[dict[str, object]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("JSON event input must be an array")
        return value
    raise ValueError("event input must be CSV, JSON, or JSONL")


class EventStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ingest_batches (
                    batch_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    inserted_count INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS event_revisions (
                    revision_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    scheduled_date TEXT NOT NULL,
                    actual_date TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_tier TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    batch_id TEXT NOT NULL REFERENCES ingest_batches(batch_id)
                );

                CREATE INDEX IF NOT EXISTS event_revisions_asof
                ON event_revisions(event_id, available_at, published_at, observed_at);
                CREATE INDEX IF NOT EXISTS event_revisions_date
                ON event_revisions(scheduled_date, event_type, instrument_id);
                """
            )

    def ingest(self, records: Iterable[Mapping[str, object]], *, source_path: str,
               source_sha256: str, ingested_at: str | None = None) -> dict[str, object]:
        self.initialize()
        normalized = [normalize_event(row) for row in records]
        ingested = _parse_timestamp(
            ingested_at or datetime.now(timezone.utc).isoformat(), "ingested_at"
        )
        batch_identity = {
            "source_path": source_path,
            "source_sha256": source_sha256,
            "revisions": [_sha256_text(_canonical_json(row)) for row in normalized],
        }
        batch_id = _sha256_text(_canonical_json(batch_identity))
        inserted = 0
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT inserted_count, row_count FROM ingest_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if existing:
                return {"batch_id": batch_id, "rows": existing["row_count"],
                        "inserted": 0, "idempotent": True}
            connection.execute(
                "INSERT INTO ingest_batches VALUES (?, ?, ?, ?, ?, ?)",
                (batch_id, source_path, source_sha256, ingested, len(normalized), 0),
            )
            for row in normalized:
                payload = _canonical_json(row)
                revision_id = _sha256_text(payload)
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO event_revisions (
                        revision_id, event_id, event_type, scope, instrument_id,
                        scheduled_date, actual_date, published_at, observed_at,
                        available_at, status, source_tier, payload_json, batch_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (revision_id, row["event_id"], row["event_type"], row["scope"],
                     row["instrument_id"], row["scheduled_date"], row["actual_date"],
                     row["published_at"], row["observed_at"], row["available_at"],
                     row["status"], row["source_tier"], payload, batch_id),
                )
                inserted += cursor.rowcount
            connection.execute(
                "UPDATE ingest_batches SET inserted_count = ? WHERE batch_id = ?",
                (inserted, batch_id),
            )
        return {"batch_id": batch_id, "rows": len(normalized), "inserted": inserted,
                "idempotent": False}

    def ingest_file(self, path: Path, *, ingested_at: str | None = None) -> dict[str, object]:
        path = Path(path)
        source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return self.ingest(load_event_records(path), source_path=str(path.resolve()),
                           source_sha256=source_sha256, ingested_at=ingested_at)

    def as_of(self, timestamp: str, *, start_date: str | None = None,
              end_date: str | None = None) -> list[dict[str, object]]:
        self.initialize()
        as_of = _parse_timestamp(timestamp, "as_of")
        parameters: list[object] = [as_of]
        filters = []
        if start_date is not None:
            filters.append("scheduled_date >= ?")
            parameters.append(_parse_date(start_date, "start_date", required=True))
        if end_date is not None:
            filters.append("scheduled_date <= ?")
            parameters.append(_parse_date(end_date, "end_date", required=True))
        where = " AND " + " AND ".join(filters) if filters else ""
        query = f"""
            WITH visible AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY event_id
                    ORDER BY available_at DESC, published_at DESC,
                             observed_at DESC, revision_id DESC
                ) AS version_rank
                FROM event_revisions
                WHERE available_at <= ? {where}
            )
            SELECT payload_json, revision_id, batch_id
            FROM visible
            WHERE version_rank = 1
            ORDER BY scheduled_date, event_id
        """
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["revision_id"] = row["revision_id"]
            payload["batch_id"] = row["batch_id"]
            result.append(payload)
        return result

    def history(self, event_id: str) -> list[dict[str, object]]:
        self.initialize()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json, revision_id, batch_id
                FROM event_revisions
                WHERE event_id = ?
                ORDER BY available_at, published_at, observed_at, revision_id
                """,
                (event_id,),
            ).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["revision_id"] = row["revision_id"]
            payload["batch_id"] = row["batch_id"]
            result.append(payload)
        return result

    def audit(self, as_of: str) -> dict[str, object]:
        self.initialize()
        current = self.as_of(as_of)
        with self.connect() as connection:
            revision_count = int(connection.execute(
                "SELECT COUNT(*) FROM event_revisions"
            ).fetchone()[0])
            batch_rows = connection.execute(
                "SELECT * FROM ingest_batches ORDER BY ingested_at, batch_id"
            ).fetchall()
            revised_events = int(connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT event_id FROM event_revisions GROUP BY event_id HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0])
        source_integrity = []
        for row in batch_rows:
            path = Path(row["source_path"])
            if not path.exists() or not path.is_file():
                status = "missing"
                actual_hash = None
            else:
                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                status = "verified" if actual_hash == row["source_sha256"] else "hash_mismatch"
            source_integrity.append({"batch_id": row["batch_id"],
                                     "source_path": row["source_path"],
                                     "expected_sha256": row["source_sha256"],
                                     "actual_sha256": actual_hash, "status": status})
        missing_scores = {field: sum(event.get(field) is None for event in current)
                          for field in SCORE_RANGES}
        return {
            "as_of": _parse_timestamp(as_of, "as_of"),
            "events": len(current),
            "revisions": revision_count,
            "events_with_revisions": revised_events,
            "batches": len(batch_rows),
            "availability_basis": _counts(current, "availability_basis"),
            "source_tiers": _counts(current, "source_tier"),
            "event_types": _counts(current, "event_type"),
            "statuses": _counts(current, "status"),
            "missing_scores": missing_scores,
            "source_integrity": source_integrity,
            "source_integrity_passed": all(item["status"] == "verified"
                                           for item in source_integrity),
        }


def _counts(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))
