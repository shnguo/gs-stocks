from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .event_store import EventStore
from .storage import write_json

SSE_PERIODIC_PAGE = "https://www.sse.com.cn/disclosure/listedinfo/periodic/"
SSE_QUERY_ROOT = "https://query.sse.com.cn/"
SSE_TYPES_SQL_ID = "SSE_PL_SSGSXX_DQBGYYQK_LX_L"
SSE_PERIODIC_SQL_ID = "SSE_SZSGG_DQBGYYQK_CAST_NEW"
SSE_REPORT_NAMES = {
    "L011": "annual report",
    "L012": "half-year report",
    "L013": "first-quarter report",
    "L014": "third-quarter report",
}
CNINFO_PERIODIC_PAGE = "https://www.cninfo.com.cn/new/commonUrl?url=data/yypl"
CNINFO_ROOT = "https://www.cninfo.com.cn/new/information/"
CNINFO_REPORT_NAMES = {
    "03-31": "first-quarter report",
    "06-30": "half-year report",
    "09-30": "third-quarter report",
    "12-31": "annual report",
}


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def _fetch_json(url: str, params: dict[str, object], *, timeout: float = 20) -> dict:
    query = urlencode(params)
    request = Request(
        f"{url}?{query}",
        headers={"User-Agent": "Mozilla/5.0", "Referer": SSE_PERIODIC_PAGE,
                 "Accept": "application/json,text/javascript,*/*;q=0.8"},
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    if raw.lstrip().startswith("{"):
        return json.loads(raw)
    start, end = raw.find("("), raw.rfind(")")
    if start < 0 or end <= start:
        raise ValueError("SSE response is neither JSON nor JSONP")
    return json.loads(raw[start + 1:end])


def _post_json(url: str, params: dict[str, object], *, timeout: float = 20) -> object:
    request = Request(
        url,
        data=urlencode(params).encode(),
        headers={"User-Agent": "Mozilla/5.0", "Referer": CNINFO_PERIODIC_PAGE,
                 "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                 "Accept": "application/json,text/javascript,*/*;q=0.8"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_sse_report_types(fetch: Callable[..., dict] = _fetch_json) -> list[dict[str, str]]:
    payload = fetch(
        SSE_QUERY_ROOT + "commonQuery.do",
        {"sqlId": SSE_TYPES_SQL_ID, "jsonCallBack": "eventRadarCallback"},
    )
    if payload.get("sqlId") != SSE_TYPES_SQL_ID or not isinstance(payload.get("result"), list):
        raise ValueError("unexpected SSE report-type response")
    return [{"publish_year": str(row["PUBLISH_YEAR"]),
             "bulletin_type": str(row["BULLETIN_TYPE"]), "label": str(row["TYPE"])}
            for row in payload["result"]]


def fetch_sse_periodic_rows(publish_year: str, bulletin_type: str, *, page_size: int = 500,
                            fetch: Callable[..., dict] = _fetch_json) -> dict[str, object]:
    if not publish_year.isdigit() or len(publish_year) != 4:
        raise ValueError("publish_year must have four digits")
    if bulletin_type not in SSE_REPORT_NAMES:
        raise ValueError("unsupported SSE periodic report type")
    if page_size < 1 or page_size > 1000:
        raise ValueError("page_size must be between 1 and 1000")
    pages = []
    rows = []
    page_no = 1
    while True:
        params = {
            "sqlId": SSE_PERIODIC_SQL_ID,
            "isPagination": "true",
            "pageHelp.pageSize": page_size,
            "pageHelp.pageNo": page_no,
            "pageHelp.beginPage": page_no,
            "pageHelp.cacheSize": 1,
            "pageHelp.endPage": page_no,
            "bulletintype": bulletin_type,
            "publishYear": publish_year,
            "companyCode": "",
            "startTime": "",
            "order": "companyCode|asc",
        }
        payload = fetch(SSE_QUERY_ROOT + "commonSoaQuery.do", params)
        if payload.get("sqlId") != SSE_PERIODIC_SQL_ID:
            raise ValueError("unexpected SSE periodic response")
        result = payload.get("result")
        page_help = payload.get("pageHelp")
        if not isinstance(result, list) or not isinstance(page_help, dict):
            raise ValueError("SSE periodic response is missing rows or pagination")
        pages.append(payload)
        rows.extend(result)
        page_count = int(page_help.get("pageCount") or 0)
        if page_no >= page_count:
            break
        page_no += 1
        if page_no > 1000:
            raise ValueError("SSE periodic pagination exceeded safety limit")
    keys = [(str(row.get("companyCode", "")), str(row.get("bulletinType", "")),
             str(row.get("publishYear", ""))) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("SSE periodic response contains duplicate business keys")
    return {"publish_year": publish_year, "bulletin_type": bulletin_type,
            "pages": pages, "rows": rows}


def sse_periodic_event(row: dict[str, object], *, observed_at: str,
                       source_document_sha256: str) -> dict[str, object]:
    code = str(row.get("companyCode", "")).strip()
    year = str(row.get("publishYear", "")).strip()
    bulletin_type = str(row.get("bulletinType", "")).strip()
    if len(code) != 6 or not code.isdigit():
        raise ValueError("SSE periodic row has invalid company code")
    if len(year) != 4 or not year.isdigit() or bulletin_type not in SSE_REPORT_NAMES:
        raise ValueError("SSE periodic row has invalid report identity")
    appointment_history = [str(row.get(f"publishDate{index}", "")).strip()
                           for index in range(4)]
    scheduled = next((value for value in reversed(appointment_history) if value), "")
    actual = str(row.get("actualDate", "")).strip()
    if not scheduled:
        raise ValueError("SSE periodic row has no appointment date")
    name = str(row.get("companyAbbr", "")).strip()
    return {
        "event_id": f"sse:periodic:{code}:{year}:{bulletin_type}",
        "event_type": "earnings_report",
        "scope": "instrument",
        "instrument_id": f"cn.xshg.{code}",
        "title": f"{name} {year} {SSE_REPORT_NAMES[bulletin_type]}",
        "scheduled_date": scheduled,
        "actual_date": actual,
        "expected_window_end": "",
        "published_at": observed_at,
        "observed_at": observed_at,
        "availability_basis": "local_observation",
        "source_name": "Shanghai Stock Exchange",
        "source_url": SSE_PERIODIC_PAGE,
        "source_tier": "primary_exchange",
        "status": "completed" if actual else "scheduled",
        "certainty": "",
        "transmission": "",
        "expectation_gap": "",
        "flow_impact": "",
        "price_confirmation": "",
        "crowding_penalty": "",
        "gap_penalty": "",
        "liquidity_penalty": "",
        "support_price": "",
        "target_price": "",
        "failure_price": "",
        "expected_metric": "",
        "expected_value": "",
        "actual_value": "",
        "unit": "",
        "notes": "appointment_history=" + ",".join(appointment_history),
        "source_document_sha256": source_document_sha256,
    }


def collect_sse_periodic(store: EventStore, capture_directory: Path,
                         reports: list[tuple[str, str]] | None = None, *,
                         observed_at: str | None = None,
                         fetch: Callable[..., dict] = _fetch_json) -> dict[str, object]:
    observed = observed_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if reports is None:
        reports = [(item["publish_year"], item["bulletin_type"])
                   for item in fetch_sse_report_types(fetch)]
    capture_directory = Path(capture_directory)
    if capture_directory.exists():
        raise FileExistsError("SSE captures use new directories")
    capture_directory.mkdir(parents=True)
    raw = {
        "source": SSE_PERIODIC_PAGE,
        "observed_at": observed,
        "reports": [fetch_sse_periodic_rows(year, report, fetch=fetch)
                    for year, report in reports],
    }
    raw_bytes = _canonical_bytes(raw)
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    raw_path = capture_directory / "raw.json"
    raw_path.write_bytes(raw_bytes)
    records = [sse_periodic_event(row, observed_at=observed,
                                  source_document_sha256=raw_hash)
               for report in raw["reports"] for row in report["rows"]]
    normalized_path = capture_directory / "normalized.json"
    write_json(normalized_path, records)
    result = store.ingest(records, source_path=str(raw_path.resolve()), source_sha256=raw_hash,
                          ingested_at=observed)
    manifest = {
        "source": SSE_PERIODIC_PAGE,
        "observed_at": observed,
        "raw_sha256": raw_hash,
        "reports": [{"publish_year": year, "bulletin_type": report}
                    for year, report in reports],
        "normalized_records": len(records),
        "ingest": result,
        "availability_basis": "local_observation",
        "warning": "Historical source publication times are not exposed by this endpoint.",
    }
    write_json(capture_directory / "manifest.json", manifest)
    return manifest


def fetch_cninfo_sections(fetch: Callable[..., object] = _post_json) -> list[dict[str, str]]:
    payload = fetch(CNINFO_ROOT + "getSelectData", {"rows": 4})
    if not isinstance(payload, list):
        raise ValueError("unexpected CNInfo section response")
    result = []
    for row in payload:
        section = str(row.get("value0", ""))
        if len(section) != 10 or section[4] != "-" or section[7] != "-":
            raise ValueError("CNInfo returned an invalid report section")
        result.append({"section_time": section, "label": str(row.get("value1", ""))})
    return result


def fetch_cninfo_periodic_rows(section_time: str, *, market: str = "sz",
                               page_size: int = 500,
                               fetch: Callable[..., object] = _post_json) -> dict[str, object]:
    if len(section_time) != 10 or section_time[5:] not in CNINFO_REPORT_NAMES:
        raise ValueError("unsupported CNInfo report section")
    if market not in {"sz", "sh", "bj"}:
        raise ValueError("CNInfo market must be sz, sh, or bj")
    if page_size < 1 or page_size > 1000:
        raise ValueError("page_size must be between 1 and 1000")
    rows = []
    pages = []
    page_no = 1
    while True:
        payload = fetch(CNINFO_ROOT + "getPrbookInfo", {
            "sectionTime": section_time,
            "firstTime": "",
            "lastTime": "",
            "market": market,
            "stockCode": "",
            "orderClos": "",
            "isDesc": "",
            "pagesize": page_size,
            "pagenum": page_no,
        })
        if not isinstance(payload, dict) or not isinstance(payload.get("prbookinfos"), list):
            raise ValueError("unexpected CNInfo periodic response")
        pages.append(payload)
        rows.extend(payload["prbookinfos"])
        total_pages = int(payload.get("totalPages") or 0)
        if page_no >= total_pages:
            break
        page_no += 1
        if page_no > 1000:
            raise ValueError("CNInfo pagination exceeded safety limit")
    keys = [(str(row.get("seccode", "")), str(row.get("f001d_0102", ""))) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("CNInfo periodic response contains duplicate business keys")
    return {"section_time": section_time, "market": market, "pages": pages, "rows": rows}


def cninfo_periodic_event(row: dict[str, object], *, market: str, observed_at: str,
                          source_document_sha256: str) -> dict[str, object]:
    code = str(row.get("seccode", "")).strip()
    section = str(row.get("f001d_0102", "")).strip()
    if len(code) != 6 or not code.isdigit() or market not in {"sz", "sh", "bj"}:
        raise ValueError("CNInfo periodic row has an invalid security identity")
    if len(section) != 10 or section[5:] not in CNINFO_REPORT_NAMES:
        raise ValueError("CNInfo periodic row has an invalid report section")
    appointment_history = [str(row.get(field, "")).strip()
                           for field in ["f002d_0102", "f003d_0102", "f004d_0102",
                                         "f005d_0102"]]
    scheduled = next((value for value in reversed(appointment_history) if value), "")
    actual = str(row.get("f006d_0102", "")).strip()
    if not scheduled:
        raise ValueError("CNInfo periodic row has no appointment date")
    exchange = {"sz": "xshe", "sh": "xshg", "bj": "xbse"}[market]
    return {
        "event_id": f"cninfo:periodic:{market}:{code}:{section}",
        "event_type": "earnings_report",
        "scope": "instrument",
        "instrument_id": f"cn.{exchange}.{code}",
        "title": (f"{str(row.get('secname', '')).strip()} {section[:4]} "
                  f"{CNINFO_REPORT_NAMES[section[5:]]}"),
        "scheduled_date": scheduled,
        "actual_date": actual,
        "expected_window_end": "",
        "published_at": observed_at,
        "observed_at": observed_at,
        "availability_basis": "local_observation",
        "source_name": "CNInfo",
        "source_url": CNINFO_PERIODIC_PAGE,
        "source_tier": "designated_disclosure",
        "status": "completed" if actual else "scheduled",
        "certainty": "",
        "transmission": "",
        "expectation_gap": "",
        "flow_impact": "",
        "price_confirmation": "",
        "crowding_penalty": "",
        "gap_penalty": "",
        "liquidity_penalty": "",
        "support_price": "",
        "target_price": "",
        "failure_price": "",
        "expected_metric": "",
        "expected_value": "",
        "actual_value": "",
        "unit": "",
        "notes": "appointment_history=" + ",".join(appointment_history),
        "source_document_sha256": source_document_sha256,
    }


def collect_cninfo_periodic(store: EventStore, capture_directory: Path,
                            sections: list[str] | None = None, *, market: str = "sz",
                            observed_at: str | None = None,
                            fetch: Callable[..., object] = _post_json) -> dict[str, object]:
    observed = observed_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sections = sections or [item["section_time"] for item in fetch_cninfo_sections(fetch)]
    capture_directory = Path(capture_directory)
    if capture_directory.exists():
        raise FileExistsError("CNInfo captures use new directories")
    capture_directory.mkdir(parents=True)
    raw = {
        "source": CNINFO_PERIODIC_PAGE,
        "observed_at": observed,
        "market": market,
        "sections": [fetch_cninfo_periodic_rows(section, market=market, fetch=fetch)
                     for section in sections],
    }
    raw_bytes = _canonical_bytes(raw)
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    raw_path = capture_directory / "raw.json"
    raw_path.write_bytes(raw_bytes)
    records = [cninfo_periodic_event(row, market=market, observed_at=observed,
                                     source_document_sha256=raw_hash)
               for section in raw["sections"] for row in section["rows"]]
    write_json(capture_directory / "normalized.json", records)
    result = store.ingest(records, source_path=str(raw_path.resolve()), source_sha256=raw_hash,
                          ingested_at=observed)
    manifest = {
        "source": CNINFO_PERIODIC_PAGE,
        "observed_at": observed,
        "market": market,
        "raw_sha256": raw_hash,
        "sections": sections,
        "normalized_records": len(records),
        "ingest": result,
        "availability_basis": "local_observation",
        "warning": "Historical appointment-change publication times are not exposed.",
    }
    write_json(capture_directory / "manifest.json", manifest)
    return manifest
