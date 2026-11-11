"""Run the registered comparison without selecting the best backtest variant."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import cluster_interval, event_returns, metrics, passive, simulate
from .signals import build_signals, prepare_events, price_panel
from .text_model import add_model_signals, export_model_audit, fit_text_model

ARM_LABELS = {"spy": "SPY buy and hold", "universe": "Sample buy and hold",
              "price": "Price only", "earnings": "Earnings + price",
              "guidance": "Raised guidance + price", "ml": "ML guidance + price"}


def run_research(root: Path, config_path: Path | None = None) -> dict:
    config_path = config_path or root / "configs/pilot-v1.json"
    config = json.loads(config_path.read_text())
    data_dir, out = root / "data", root / "artifacts" / config.get("artifact_subdir", "")
    out.mkdir(parents=True, exist_ok=True)
    protocol_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    registered = out / "protocol.sha256"
    if registered.exists() and registered.read_text().strip() != protocol_hash:
        raise ValueError("Registered protocol changed; use a new version and output directory")
    if not registered.exists():
        registered.write_text(protocol_hash)
    prices = pd.read_csv(data_dir / "prices.csv")
    filings = pd.read_csv(data_dir / "filings.csv", keep_default_na=False)
    expected = {c["ticker"] for c in config["companies"]}
    if set(filings.ticker) != expected:
        raise ValueError("Filing collection incomplete; finish every configured company before comparing returns")
    events = prepare_events(filings, data_dir, config.get("parser_version", "lexical-v1.1"))
    panel, calendar = price_panel(prices, config)
    signals, events = build_signals(panel, calendar, events, config)
    model_path = root / config.get("text_model_config", "configs/text-model-v1.json")
    model_settings = json.loads(model_path.read_text())
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model_lock = out / "text-model-protocol.sha256"
    if model_lock.exists() and model_lock.read_text().strip() != model_hash:
        raise ValueError("Text model protocol changed; create a new version")
    model_lock.write_text(model_hash)
    predictions, model_metadata, model = fit_text_model(events, data_dir, model_settings)
    export_model_audit(predictions, model_metadata, model, out)
    signals = add_model_signals(signals, predictions)
    events.to_csv(out / "events.csv", index=False)
    signals.to_csv(out / "signals.csv", index=False)
    curves, ledgers, metrics_rows = {}, {}, []
    masks = {"price": pd.Series(True, index=signals.index),
             "earnings": signals.earnings, "guidance": signals.guidance}
    for arm, mask in masks.items():
        curve, ledger, rejected = simulate(panel, calendar, signals.loc[mask], config)
        curves[arm], ledgers[arm] = curve, ledger
        ledger.to_csv(out / f"trades-{arm}.csv", index=False)
        rejected.to_csv(out / f"skipped-{arm}.csv", index=False)
    ml_config = dict(config, start=model_settings["evaluation_start"])
    ml_signals = signals.loc[signals.ml_guidance]
    curves["ml"], ledgers["ml"], ml_rejected = simulate(panel, calendar, ml_signals, ml_config)
    ledgers["ml"].to_csv(out / "trades-ml.csv", index=False)
    ml_rejected.to_csv(out / "skipped-ml.csv", index=False)
    curves["spy"] = passive(panel, calendar, [config["benchmark"]], config)
    curves["universe"] = passive(panel, calendar, sorted(expected), config)
    periods = [("2019-2022 development", config["start"], config["development_end"]),
               ("2023 validation", "2023-01-01", config["validation_end"]),
               ("2024-2025 later evaluation", "2024-01-01", config["end"]),
               ("2019-2025 full sample", config["start"], config["end"])]
    annual_rows = []
    for arm, curve in curves.items():
        curve.to_csv(out / f"equity-{arm}.csv")
        ledger = ledgers.get(arm)
        for period, start, end in periods:
            sliced = curve.loc[start:end]
            if sliced.empty:
                continue
            if arm == "ml" and start < model_settings["evaluation_start"]:
                # Never present an in-sample ML backtest or a misleading full-period result.
                continue
            prior = curve.loc[curve.index < pd.Timestamp(start), "nav"]
            prior_nav = float(prior.iloc[-1]) if len(prior) else 1.0
            closed = (ledger.loc[ledger.exit_date.between(start, end)] if ledger is not None else None)
            metrics_rows.append({"arm": arm, "label": ARM_LABELS[arm], "period": period,
                                 **metrics(sliced, closed, prior_nav)})
        for year, sliced in curve.groupby(curve.index.year):
            prior = curve.loc[curve.index < sliced.index[0], "nav"]
            annual_rows.append({"arm": arm, "year": int(year),
                                "return": float(sliced.nav.iloc[-1] / (prior.iloc[-1] if len(prior) else 1) - 1)})
    summary = pd.DataFrame(metrics_rows)
    summary.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(out / "annual-returns.csv", index=False)
    event_frame = event_returns(panel, calendar, signals, config)
    if event_frame.empty:
        raise ValueError("No mature earnings signals; inspect source/timing coverage")
    event_frame.to_csv(out / "event-returns.csv", index=False)
    intervals = []
    for period, start, end in periods:
        for filter_name, mask in [("earnings", event_frame.earnings), ("guidance", event_frame.guidance),
                                  ("ml", event_frame.ml_guidance)]:
            if filter_name == "ml" and start < model_settings["evaluation_start"]:
                continue
            selected = event_frame.loc[mask & event_frame.signal_date.between(start, end)]
            intervals.append({"period": period, "filter": filter_name,
                              **cluster_interval(selected, config["seed"])})
    pd.DataFrame(intervals).to_csv(out / "event-intervals.csv", index=False)
    stress_rows = []
    for cost in config["stress_cost_bps_per_side"] + [config["cost_bps_per_side"]]:
        for arm in ["earnings", "guidance"]:
            curve, ledger, _ = simulate(panel, calendar, signals.loc[masks[arm]], config, cost_bps=cost)
            stress_rows.append({"arm": arm, "cost_bps_per_side": cost, **metrics(curve, ledger)})
        curve, ledger, _ = simulate(panel, calendar, ml_signals, ml_config, cost_bps=cost)
        stress_rows.append({"arm": "ml", "cost_bps_per_side": cost, **metrics(curve, ledger)})
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(out / "cost-sensitivity.csv", index=False)
    horizons = []
    for holding in [5, 20, 40]:
        for arm in ["earnings", "guidance"]:
            curve, ledger, _ = simulate(panel, calendar, signals.loc[masks[arm]], config,
                                         holding_sessions=holding)
            horizons.append({"arm": arm, "holding_sessions": holding, **metrics(curve, ledger)})
    pd.DataFrame(horizons).to_csv(out / "horizon-sensitivity.csv", index=False)
    counts = events.groupby("ticker").agg(filings=("status", "size"),
                                            retrieved=("status", lambda x: (x == "ok").sum()),
                                            unclassified=("status", lambda x: (x == "unclassified_exhibit").sum()),
                                            explicit_raises=("raised_guidance", "sum"))
    counts["missing"] = counts.filings - counts.retrieved - counts.unclassified
    counts.to_csv(out / "coverage.csv")
    # All source failures stay inspectable, including companies with no detected raises.
    events.loc[events.status != "ok"].to_csv(out / "source-failures.csv", index=False)
    positives = events.loc[events.raised_guidance]
    positives.sample(min(20, len(positives)), random_state=config["seed"]).to_csv(
        out / "guidance-audit-sample.csv", index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
        "config": config, "ai_used": True, "ai_kind": "weakly supervised TF-IDF logistic text classifier",
        "text_model": model_metadata, "text_model_protocol_sha256": model_hash,
        "data_files": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                       for name in ["prices.csv", "filings.csv"]},
        "source_code": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (root / "src").rglob("*.py")},
        "packages": {name: importlib.metadata.version(name) for name in
                     ["pandas", "numpy", "beautifulsoup4", "matplotlib", "nbformat"]},
        "prices": {"provider": "Alpaca", "feed": "sip", "adjustment": "all", "rows": len(prices),
                   "first": str(calendar[0].date()), "last": str(calendar[-1].date()),
                   "interpretation": "fractional adjusted total-return units, not historical broker shares"},
        "events": {"catalog_filings": len(events), "retrieved": int((events.status == "ok").sum()),
                   "unclassified_exhibits": int((events.status == "unclassified_exhibit").sum()),
                   "missing_sources": int((events.status == "missing").sum()),
                   "explicit_raises": int(events.raised_guidance.sum())},
        "limitations": [
            "Fixed surviving-company convenience sample; survivorship and selection bias remain.",
            "SEC filing availability may lag the original earnings release; no claim of earliest news execution.",
            "Lexical annual-guidance filter is incomplete and is not a numeric comparable-period revision parser.",
            "ML targets are lexical weak labels, not human gold labels; scores are not calibrated probabilities.",
            "The local text model is not a generative LLM, and its trading results do not validate extraction accuracy.",
            "Adjusted OHLC stops approximate total-return risk rules; real cash-share fills need corporate-action replay.",
            "Cash earns zero; no taxes, financing, spread dynamics, slippage distribution or market impact model.",
            "2024-2025 is a later historical evaluation, not a live forward test or an untouched universe holdout.",
            "Event bootstrap clusters by month but cannot eliminate all overlapping-event or cross-company dependence.",
            "No capital allocation or order placement is supported by this exploratory pilot alone.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return {"config": config, "summary": summary, "curves": curves, "ledgers": ledgers,
            "events": events, "signals": signals, "event_returns": event_frame,
            "intervals": pd.DataFrame(intervals), "stress": stress, "coverage": counts,
            "manifest": manifest}
