"""Research panel refresh from Rust-normalized, archived provider records.

This deliberately excludes new adjustment boundaries instead of inventing
corporate-action factors or holder entitlements from the reference close.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .daily_loop import digest, read, verify
from .features import stock_features
from .storage import file_hash, utc_now, write_json


def normalized(path):
    path = Path(path)
    manifest = verify(path)
    result = read(path / "normalized.json")
    if any("possible_row_limit" in reason for reason in result["blockers"]):
        raise ValueError("Truncated provider data")
    frame = pd.DataFrame(result["records"])
    if "provider_code" in frame:
        frame["instrument_id"] = (
            "cn." + frame.exchange.str.lower() + "." + frame.provider_code.str.split(".").str[0]
        )
    return frame, {
        "path": str(path.resolve()),
        "manifest_sha256": file_hash(path / "manifest.json"),
        "raw_sha256": manifest["raw_sha256"],
        "fetched_at": manifest["fetched_at"],
    }


def reference_chain(anchor, overlap, updates, sessions):
    """Only carry an existing factor where every official reference is unchanged."""
    if anchor is None or overlap is None:
        return False, "missing_cross_provider_anchor"
    fields = ["open", "high", "low", "close"]
    if not np.isclose(
        anchor[fields].to_numpy(float), overlap[fields].to_numpy(float), rtol=0, atol=0.005
    ).all():
        return False, "anchor_quote_mismatch"
    if not np.isfinite(anchor.factor) or anchor.factor <= 0:
        return False, "missing_anchor_factor"
    previous = anchor.close
    for day in sessions:
        if day not in updates.index:
            return False, "missing_session_no_suspension_assumption"
        row = updates.loc[day]
        if (
            pd.isna(row.pre_close)
            or not np.isfinite(row.pre_close)
            or not np.isclose(row.pre_close, previous, rtol=0, atol=1e-8)
        ):
            return False, "reference_change_requires_action_evidence"
        if (
            not np.isfinite(
                row[["open", "high", "low", "close", "volume", "amount"]].to_numpy(float)
            ).all()
            or row.volume <= 0
            or row.amount <= 0
        ):
            return False, "unpriced_or_suspended_session"
        previous = row.close
    return True, "source_reference_continuity_only"


def refresh(base, captures, output, trade_date, calendar_evidence):
    base, captures, output = map(lambda p: Path(p).resolve(), [base, captures, output])
    if output.exists():
        raise FileExistsError("New immutable input directory required")
    panel = read(base / "panel-h5/manifest.json")
    snap = read(base / "snapshot/manifest.json")
    # Verify both actual panel bytes and the raw-history source used for extension.
    verify(base / "panel-h5")
    verify(base / "snapshot")
    calendar, cal_source = normalized(captures / "calendar")
    if not read(captures / "calendar/normalized.json")["coverage"]["calendar_range_complete"]:
        raise ValueError("Incomplete provider calendar")
    evidence = read(calendar_evidence)
    if not (evidence["valid_from"] <= trade_date <= evidence["valid_through"]):
        raise ValueError("Exchange calendar cross-check expired")
    opened = calendar.loc[calendar.is_open, "date"].tolist()
    i = opened.index(trade_date)
    horizon = opened[i : i + 5]
    if len(horizon) != 5 or horizon[-1] > evidence["valid_through"]:
        raise ValueError("Incomplete forward calendar")
    signal = opened[i - 1]
    anchor_day = panel.get("price_data_through", panel["dates"][-1])
    anchor_index = panel["dates"].index(anchor_day)
    if not anchor_day < signal:
        raise ValueError("Refresh must extend the frozen source")
    sessions = [d for d in opened if anchor_day < d <= signal]
    master, master_source = normalized(captures / "current-master")
    lineage = [cal_source, master_source]
    current = master.set_index("instrument_id")
    daily = []
    for date in [anchor_day] + sessions:
        frame, source = normalized(captures / f"daily-{date.replace('-', '')}-reference")
        if (
            frame.date.nunique() != 1
            or frame.date.iloc[0] != date
            or frame.instrument_id.duplicated().any()
        ):
            raise ValueError("Daily source scope mismatch")
        daily.append(frame)
        lineage.append(source)
    provider = pd.concat(daily, ignore_index=True)
    old_dates = panel["dates"][max(0, anchor_index - 149) : anchor_index + 1]
    dates = old_dates + [d for d in opened if d > anchor_day]
    metadata = pd.read_parquet(base / "snapshot/instruments.parquet").set_index("instrument_id")
    symbols = sorted(set(panel["instruments"]) | set(current.index))
    metadata = metadata.reindex(symbols)
    for symbol, row in current.iterrows():
        metadata.loc[symbol, "name"] = row["name"]
        metadata.loc[symbol, "listed_at"] = row.listed_at
        metadata.loc[symbol, "delisted_at"] = ""
        metadata.loc[symbol, "exchange"] = row.exchange
        if pd.isna(metadata.loc[symbol, "board"]):
            metadata.loc[symbol, "board"] = row.market
    metadata["current_member"] = metadata.index.isin(current.index)
    metadata["industry"] = metadata.industry.fillna("unknown")
    metadata["listed_at"] = metadata.listed_at.fillna("9999-12-31")
    metadata["delisted_at"] = metadata.delisted_at.fillna("")
    raw = pd.read_parquet(
        base / "snapshot/bars.parquet",
        filters=[("date", ">=", old_dates[0]), ("date", "<=", anchor_day)],
    )
    anchors = raw.loc[raw.date.eq(anchor_day)].set_index("instrument_id")
    overlap = provider.loc[provider.date.eq(anchor_day)].set_index("instrument_id")
    incoming = provider.loc[provider.date.gt(anchor_day)].copy()
    groups = {s: g.set_index("date") for s, g in incoming.groupby("instrument_id")}
    raw_groups = {s: g.set_index("date") for s, g in raw.groupby("instrument_id")}
    old_values = np.load(base / "panel-h5/values.npy", mmap_mode="r")
    old_index = {s: i for i, s in enumerate(panel["instruments"])}
    values = np.full(
        (len(symbols), len(dates), len(panel["feature_names"])), np.nan, dtype=np.float32
    )
    ledger = []
    new_rows = []
    empty = incoming.iloc[:0].set_index("date")
    for j, symbol in enumerate(symbols):
        if symbol in old_index:
            values[j, : len(old_dates)] = old_values[
                old_index[symbol], anchor_index - len(old_dates) + 1 : anchor_index + 1
            ]
        anchor = anchors.loc[symbol] if symbol in anchors.index else None
        overlap_row = overlap.loc[symbol] if symbol in overlap.index else None
        update = groups.get(symbol, empty)
        good, reason = reference_chain(anchor, overlap_row, update, sessions)
        ledger.append(
            {
                "instrument_id": symbol,
                "current_member": symbol in current.index,
                "continuity_passed": good,
                "input_reason": reason,
            }
        )
        if update.empty:
            continue
        update = update.copy()
        update["factor"] = anchor.factor if good else np.nan
        for field in ["sequence_id", "label_sequence_id"]:
            update[field] = anchor.get(field, "") if good else "new_unverified_boundary"
        update["upper_limit"] = np.nan
        update["lower_limit"] = np.nan
        update["factor_evidence"] = "unchanged_source_reference" if good else reason
        new_rows.append(update.reset_index())
        if not good:
            continue
        history = raw_groups[symbol].reindex(dates[: len(old_dates) + len(sessions)]).copy()
        for col in history.columns.intersection(update.columns):
            history.loc[update.index, col] = update[col].astype(history[col].dtype).to_numpy()
        feat = stock_features(history)
        feat["suspended"] = history.source_trade_status.eq(0).astype(float)
        feat["unpriced_suspension"] = 0.0
        if list(feat.columns) != panel["feature_names"]:
            raise ValueError("Feature contract mismatch")
        values[j, len(old_dates) : len(old_dates) + len(sessions)] = feat.loc[sessions].to_numpy(
            np.float32
        )
    output.mkdir(parents=True)
    snapshot = output / "snapshot"
    snapshot.mkdir()
    new_panel = output / "panel-h5"
    new_panel.mkdir()
    metadata.reset_index(names="instrument_id").to_parquet(
        snapshot / "instruments.parquet", index=False
    )
    frames = [raw, *new_rows]
    columns = list(dict.fromkeys(c for frame in frames for c in frame.columns))
    combined = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in frames], ignore_index=True
    ).reindex(columns=columns)
    assert (
        combined.date.max() == signal and not combined.duplicated(["instrument_id", "date"]).any()
    )
    combined.to_parquet(snapshot / "bars.parquet", index=False)
    pd.DataFrame(
        [
            {"date": d, "exchange": e, "is_open": True}
            for d in dates
            for e in ["XSHG", "XSHE", "XBSE"]
        ]
    ).to_parquet(snapshot / "calendar.parquet", index=False)
    for name in ["actions", "risk_events", "risk_coverage", "rules"]:
        table = pd.read_parquet(base / f"snapshot/{name}.parquet")
        if name == "actions":
            table = table.loc[table.ex_date <= signal]
        table.to_parquet(snapshot / f"{name}.parquet", index=False)
    np.save(new_panel / "values.npy", values)
    pd.DataFrame(ledger).to_parquet(output / "refresh-coverage.parquet", index=False)
    specification = {
        "trade_date": trade_date,
        "signal_date": signal,
        "parent_dataset_id": panel["dataset_id"],
        "captures": lineage,
        "calendar_evidence": evidence,
        "implementation_sha256": file_hash(Path(__file__)),
        "method": "carry factor only across unchanged source reference; reject all new reference boundaries",
        "limitations": [
            "provider_reference_is_not_complete_action_evidence",
            "no_shareholder_entitlement_inference",
            "missing_new_sessions_not_imputed",
        ],
    }
    dataset_id = digest(specification)
    declaration = {
        **snap["declaration"],
        "source": "incremental Tushare Rust-normalized research feed",
        "price_data_through": signal,
        "calendar_evidence": evidence,
        "factor_update_method": specification["method"],
    }
    write_json(
        snapshot / "manifest.json",
        {
            "created_at": utc_now(),
            "dataset_id": dataset_id,
            "declaration": declaration,
            "files": {p.name: file_hash(p) for p in snapshot.glob("*.parquet")},
        },
    )
    write_json(
        new_panel / "manifest.json",
        {
            "created_at": utc_now(),
            "dataset_id": dataset_id,
            "dates": dates,
            "instruments": symbols,
            "feature_names": panel["feature_names"],
            "lookback": 60,
            "horizon": 5,
            "price_data_through": signal,
            "files": {"values.npy": file_hash(new_panel / "values.npy")},
        },
    )
    shutil.copy2(base / "schedule.json", output / "schedule.json")
    write_json(output / "lineage.json", specification)
    (output / "code").mkdir()
    for name in ["daily_input.py", "features.py"]:
        shutil.copy2(Path(__file__).with_name(name), output / "code" / name)
    usable = np.isfinite(values[:, dates.index(signal) - 59 : dates.index(signal) + 1]).all((1, 2))
    write_json(
        output / "refresh-summary.json",
        {
            "created_at": utc_now(),
            "dataset_id": dataset_id,
            "signal_date": signal,
            "horizon_dates": horizon,
            "current_members": len(current),
            "source_prices_on_signal": int(provider.date.eq(signal).sum()),
            "feature_complete": int(usable.sum()),
            "continuity_reasons": pd.DataFrame(ledger).groupby("input_reason").size().to_dict(),
            "formal_ready": False,
            "source_purpose": "daily_research",
        },
    )
    return output
