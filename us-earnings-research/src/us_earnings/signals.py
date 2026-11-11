"""Source-bound guidance baseline and signals with an explicit information boundary."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

VERB = r"\b(?:rais(?:e[sd]?|ing)|increas(?:e[sd]?|ing)|lift(?:s|ed|ing)?|boost(?:s|ed|ing)?)\b"
ANNUAL = r"(?:full[ -]year|annual|fiscal\s+(?:year\s+)?20\d{2}|\bFY\s*20\d{2})"
OUTLOOK = r"(?:guidance|outlook|forecast)"
MODIFIER = (r"(?:its|our|the|their|company's|consolidated|adjusted|reported|non-GAAP|GAAP|"
            r"revenue|sales|earnings|per|share|EPS|diluted|net|income|profit|financial|and|"
            r"full-year|full|annual|fiscal|year|FY|for|20\d{2})")
RAISE_PATTERNS = [
    re.compile(VERB + r"\s+(?:" + MODIFIER + r"\s+){0,16}" + OUTLOOK, re.I),
    re.compile(ANNUAL + r"\s+(?:" + MODIFIER + r"\s+){0,8}" + OUTLOOK +
               r"\s+(?:(?:is|was|has been|have been)\s+)?" + VERB, re.I),
]


def guidance_label(text: str, version: str = "lexical-v1.1") -> dict:
    """Detect explicit language, not an earnings surprise or an LLM prediction.

    Conservative lexical coverage: numeric-only revisions are left unknown.
    The retained evidence makes false positives auditable rather than hidden.
    """
    normalized = re.sub(r"[\u2010-\u2015]", "-", " ".join(text.split()))
    annual, patterns = ANNUAL, RAISE_PATTERNS
    if version == "lexical-v1.2":
        annual = r"(?:full[ -]year|annual|fiscal\s+(?:year\s+)?20\d{2}|\bFY\s*(?:20)?\d{2})"
        modifier = (r"(?:its|our|the|their|company's|consolidated|adjusted|reported|operational|"
                    r"non-GAAP|GAAP|revenue|sales|earnings|per|share|EPS|diluted|net|income|profit|"
                    r"financial|and|full-year|full|annual|fiscal|year|FY(?:20)?\d{2}|FY|for|"
                    r"20\d{2}|low|end|of|midpoint|\d{1,2}(?:,\d{1,2})*)")
        patterns = [re.compile(VERB + r"\s+(?:" + modifier + r"\s+){0,20}" + OUTLOOK, re.I),
                    re.compile(annual + r"\s+(?:" + modifier + r"\s+){0,8}" + OUTLOOK +
                               r"\s+(?:(?:is|was|has been|have been)\s+)?" + VERB, re.I)]
    elif version != "lexical-v1.1":
        raise ValueError("Unknown parser version")
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            before = normalized[max(0, match.start() - 70):match.start()]
            matched = match.group()
            context = normalized[max(0, match.start() - 150):min(len(normalized), match.end() + 100)]
            # Require an annual scope near the direct verb/object phrase. A bare
            # year is accepted only inside that phrase, never from an old table.
            if not re.search(annual, context, re.I) and not re.search(r"\b20\d{2}\b", matched):
                continue
            # Avoid a dividend increase being attached to a separate outlook clause.
            if re.search(r"dividend|share repurchase|buyback|capital expenditure", matched, re.I):
                continue
            if re.search(r"(?:not|never|no longer|previously|last quarter|prior quarter)\W*$", before, re.I):
                continue
            if re.search(r"(?:previously|last quarter|prior quarter)\s+(?:\w+\s+){0,3}" + VERB,
                         context, re.I):
                continue
            if re.search(r"(?:does not|did not|will not|not|never)\s+(?:\w+\s+){0,2}" + VERB,
                         context, re.I):
                continue
            return {"guidance_state": "explicit_annual_raise", "raised_guidance": True,
                    "evidence": context, "parser_version": version}
    return {"guidance_state": "unknown_or_no_explicit_annual_raise", "raised_guidance": False,
            "evidence": "", "parser_version": version}


def prepare_events(filings: pd.DataFrame, data_dir: Path, parser_version: str = "lexical-v1.1") -> pd.DataFrame:
    rows = []
    for filing in filings.to_dict("records"):
        row = dict(filing)
        if filing["status"] == "ok":
            row.update(guidance_label((data_dir / filing["text_file"]).read_text(), parser_version))
        else:
            row.update(guidance_state="missing_source", raised_guidance=False,
                       evidence="", parser_version=parser_version)
        rows.append(row)
    return pd.DataFrame(rows)


def price_panel(prices: pd.DataFrame, config: dict) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices.date)
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date prices")
    for col in ["open", "high", "low", "close"]:
        if not np.isfinite(prices[col]).all() or (prices[col] <= 0).any():
            raise ValueError(f"Invalid {col}")
    if ((prices.high + 0.011 < prices[["open", "close", "low"]].max(axis=1)) |
            (prices.low - 0.011 > prices[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC")
    panel = {ticker: group.set_index("date").sort_index() for ticker, group in prices.groupby("ticker")}
    calendar = panel[config["benchmark"]].index
    for ticker, frame in panel.items():
        if not frame.index.equals(calendar):
            raise ValueError(f"{ticker}: missing or extra sessions relative to benchmark; repair source first")
        previous = frame.close.shift()
        tr = pd.concat([frame.high - frame.low, (frame.high - previous).abs(),
                        (frame.low - previous).abs()], axis=1).max(axis=1)
        frame["atr"] = tr.rolling(config["atr_sessions"]).mean()
        frame["ma"] = frame.close.rolling(config["ma_sessions"]).mean()
        frame["return"] = frame.close.pct_change(fill_method=None)
    return panel, calendar


def build_signals(panel: dict, calendar: pd.DatetimeIndex, events: pd.DataFrame, config: dict):
    benchmark = panel[config["benchmark"]]
    signal_rows, event_rows = [], []
    mapping = {}
    for event in events.to_dict("records"):
        filing_day = pd.Timestamp(event["filingDate"])
        index = calendar.searchsorted(filing_day, side="right")
        event = dict(event, signal_date=calendar[index] if index < len(calendar) else pd.NaT)
        event_rows.append(event)
        # Missing-source events remain in the audit, but do not enter a strategy.
        if event["status"] == "ok" and index < len(calendar):
            mapping.setdefault((event["ticker"], calendar[index]), []).append(event)
    for ticker in sorted(set(panel) - {config["benchmark"]}):
        frame = panel[ticker]
        relative = frame["return"] - benchmark["return"]
        valid = (frame.close > frame.ma) & (relative > 0) & (frame.atr > 0)
        for index in np.flatnonzero(valid.to_numpy()):
            day = calendar[index]
            if day < pd.Timestamp(config["start"]) or index + 1 == len(calendar):
                continue
            if day > pd.Timestamp(config["end"]):
                continue
            matching = mapping.get((ticker, day), [])
            raised = [event for event in matching if event["raised_guidance"]]
            signal_rows.append({"ticker": ticker, "signal_date": day,
                                "entry_date": calendar[index + 1], "atr": float(frame.atr.iloc[index]),
                                "rank": float(relative.iloc[index]), "earnings": bool(matching),
                                "guidance": bool(raised), "accession": "|".join(
                                    event["accessionNumber"] for event in matching),
                                "filing_date": matching[0]["filingDate"] if matching else ""})
    columns = ["ticker", "signal_date", "entry_date", "atr", "rank", "earnings", "guidance",
               "accession", "filing_date"]
    return pd.DataFrame(signal_rows, columns=columns), pd.DataFrame(event_rows)
