from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .fixtures import create_fixture
from .splits import Fold, rolling_folds
from .storage import Snapshot, freeze, write_json
from .universe import audit_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share ranking with separate training/trading pools")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] /
                        "configs/research.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze", help="freeze canonical tables with a source declaration")
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("declaration", type=Path)
    p = sub.add_parser("audit", help="audit a frozen snapshot")
    p.add_argument("snapshot", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("folds", help="generate quarterly development folds and seal last 12 months")
    p.add_argument("snapshot", type=Path)
    p.add_argument("--horizon", type=int, choices=[5, 20], default=5)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run", help="train and compare one explicitly specified development fold")
    p.add_argument("snapshot", type=Path)
    p.add_argument("fold", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizon", type=int, choices=[5, 20], default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--engineering", action="store_true")
    p.add_argument("--training-only", action="store_true",
                   help="train and evaluate rankings without portfolio simulation or execution inputs")
    p.add_argument("--equal-information", action="store_true")
    p.add_argument("--test-fixture", action="store_true",
                   help="explicitly permit partial engineering fixtures; never a quantitative selection pool")
    p = sub.add_parser("pool", help="show the full market catalog, independent of personal lists")
    p = sub.add_parser("pool-audit", help="compare every expected market member with a snapshot")
    p.add_argument("snapshot", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("demo", help="run both horizons on clearly marked artificial data")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--equal-information", action="store_true")
    p = sub.add_parser("report", help="render an integrity-checked local HTML report")
    p.add_argument("run", type=Path)
    p = sub.add_parser("event-store-init", help="initialize an append-only event revision store")
    p.add_argument("store", type=Path)
    p = sub.add_parser("event-ingest", help="ingest normalized event revisions")
    p.add_argument("store", type=Path)
    p.add_argument("input", type=Path)
    p.add_argument("--ingested-at")
    p = sub.add_parser("event-snapshot", help="export the latest visible event revision as of a time")
    p.add_argument("store", type=Path)
    p.add_argument("--as-of", required=True)
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("event-sse-periodic", help="capture and ingest SSE report appointments")
    p.add_argument("store", type=Path)
    p.add_argument("capture_directory", type=Path)
    p.add_argument("--report", action="append", default=[], metavar="YEAR:TYPE",
                   help="repeatable report selector such as 2026:L012; default discovers all")
    p.add_argument("--observed-at")
    p = sub.add_parser("event-cninfo-periodic",
                       help="capture and ingest designated periodic-report appointments")
    p.add_argument("store", type=Path)
    p.add_argument("capture_directory", type=Path)
    p.add_argument("--section", action="append", default=[], metavar="YYYY-MM-DD",
                   help="repeatable report period end; default discovers the current four")
    p.add_argument("--market", choices=["sz", "sh", "bj"], default="sz")
    p.add_argument("--observed-at")
    p = sub.add_parser("event-radar", help="build a 60-day event radar from completed daily bars")
    p.add_argument("store", type=Path)
    p.add_argument("bars", type=Path)
    p.add_argument("--as-of", required=True)
    p.add_argument("--completed-through", required=True)
    p.add_argument("--past-days", type=int, default=5)
    p.add_argument("--future-days", type=int, default=60)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("event-study", help="run a conservative adjusted-open event study")
    p.add_argument("store", type=Path)
    p.add_argument("bars", type=Path)
    p.add_argument("--as-of", required=True)
    p.add_argument("--mode", choices=["pre", "post"], required=True)
    p.add_argument("--horizon-sessions", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("event-audit", help="audit event lineage, revision and source integrity")
    p.add_argument("store", type=Path)
    p.add_argument("--as-of", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("event-digest", help="render the four daily event-radar lists")
    p.add_argument("radar", type=Path)
    p.add_argument("--radar-date", required=True)
    p.add_argument("--completed-through")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "freeze":
        result = freeze(args.source, args.destination, json.loads(args.declaration.read_text()))
        print(json.dumps({"dataset_id": result.dataset_id, "path": str(result.path)}))
    elif args.command == "pool":
        from .pool import load_pool
        manifest, pool = load_pool(config["universe_catalog"])
        print(json.dumps({"universe_id": manifest["universe_id"], "scope": pool["scope"],
                          "as_of": pool["as_of"], "counts": pool["counts"],
                          "membership_source": pool["membership_source"], "data_ready": False}))
    elif args.command == "pool-audit":
        from .pool import coverage_frame, pool_coverage
        if args.output.exists() or args.output.with_suffix(".csv").exists():
            raise FileExistsError("Pool audits use new output paths")
        result = pool_coverage(Snapshot(args.snapshot), config["universe_catalog"])
        write_json(args.output, result)
        coverage_frame(result).to_csv(args.output.with_suffix(".csv"), index=False)
        print(json.dumps({k: result[k] for k in ["selection_pool_members",
                         "expected_historical_members", "snapshot_instruments", "full_membership_present",
                         "full_calendar_day_coverage"]}))
    elif args.command == "audit":
        result = audit_snapshot(Snapshot(args.snapshot), config["signal_time"],
                                config["min_history_years"],
                                training_risk_policy=config["training_risk_policy"])
        write_json(args.output, result)
        print(json.dumps({"formal_ready": result["formal_ready"],
                          "training_data_ready": result["training_data_ready"],
                          "training_data_blockers": result["training_data_blockers"],
                          "backtest_data_blockers": result["backtest_data_blockers"],
                          "formal_blockers": result["formal_blockers"]}))
    elif args.command == "folds":
        dates = Snapshot(args.snapshot).dates[:-(args.horizon + 1)]
        v = config["validation"]
        result = rolling_folds(dates, v["train_years"], v["validation_years"],
                               v["holdout_months"], v["refit_months"])
        write_json(args.output, result)
        print(json.dumps({"development_folds": len(result["development"]),
                          "sealed_holdout_start": result["sealed_holdout_start"]}))
    elif args.command == "report":
        from .report import render_report
        print(render_report(args.run).resolve())
    elif args.command == "event-store-init":
        from .event_store import EventStore
        EventStore(args.store).initialize()
        print(json.dumps({"store": str(args.store.resolve()), "initialized": True}))
    elif args.command == "event-ingest":
        from .event_store import EventStore
        result = EventStore(args.store).ingest_file(args.input, ingested_at=args.ingested_at)
        print(json.dumps(result))
    elif args.command == "event-snapshot":
        from .event_store import EventStore
        if args.output.exists():
            raise FileExistsError("Event snapshots use new output paths")
        events = EventStore(args.store).as_of(args.as_of, start_date=args.start_date,
                                              end_date=args.end_date)
        write_json(args.output, {"as_of": args.as_of, "events": events})
        print(json.dumps({"events": len(events), "output": str(args.output.resolve())}))
    elif args.command == "event-sse-periodic":
        from .event_sources import collect_sse_periodic
        from .event_store import EventStore
        reports = [_parse_report_selector(value) for value in args.report] or None
        result = collect_sse_periodic(EventStore(args.store), args.capture_directory,
                                      reports, observed_at=args.observed_at)
        print(json.dumps({"events": result["normalized_records"],
                          "inserted": result["ingest"]["inserted"],
                          "capture": str(args.capture_directory.resolve())}))
    elif args.command == "event-cninfo-periodic":
        from .event_sources import collect_cninfo_periodic
        from .event_store import EventStore
        result = collect_cninfo_periodic(EventStore(args.store), args.capture_directory,
                                         args.section or None, market=args.market,
                                         observed_at=args.observed_at)
        print(json.dumps({"events": result["normalized_records"],
                          "inserted": result["ingest"]["inserted"],
                          "market": result["market"],
                          "capture": str(args.capture_directory.resolve())}))
    elif args.command == "event-radar":
        from .event_radar import build_radar
        from .event_store import EventStore
        _require_new_outputs(args.output)
        bars = _read_frame(args.bars)
        events = EventStore(args.store).as_of(args.as_of)
        radar_date = pd.Timestamp(args.as_of).tz_convert("Asia/Shanghai").date().isoformat()
        radar = build_radar(events, bars, radar_date, completed_through=args.completed_through,
                            past_days=args.past_days, future_days=args.future_days)
        _write_frame_and_metadata(args.output, radar, {
            "as_of": args.as_of,
            "radar_date": radar_date,
            "completed_through": args.completed_through,
            "past_days": args.past_days,
            "future_days": args.future_days,
            "events": len(radar),
            "states": radar.radar_state.value_counts().to_dict() if len(radar) else {},
            "scope": "research_only_no_trade_instruction",
        })
        print(json.dumps({"events": len(radar), "output": str(args.output.resolve())}))
    elif args.command == "event-study":
        from .event_store import EventStore
        from .event_study import event_study
        _require_new_outputs(args.output)
        events = EventStore(args.store).as_of(args.as_of)
        rows, summary = event_study(events, _read_frame(args.bars), mode=args.mode,
                                    horizon_sessions=args.horizon_sessions)
        _write_frame_and_metadata(args.output, rows, {"as_of": args.as_of, **summary})
        print(json.dumps({"events": len(rows), "output": str(args.output.resolve())}))
    elif args.command == "event-audit":
        from .event_store import EventStore
        if args.output.exists():
            raise FileExistsError("Event audits use new output paths")
        result = EventStore(args.store).audit(args.as_of)
        write_json(args.output, result)
        print(json.dumps({"events": result["events"], "revisions": result["revisions"],
                          "source_integrity_passed": result["source_integrity_passed"],
                          "output": str(args.output.resolve())}))
    elif args.command == "event-digest":
        from .event_digest import render_daily_digest
        if args.output.exists():
            raise FileExistsError("Event digests use new output paths")
        digest = render_daily_digest(_read_frame(args.radar), args.radar_date,
                                     completed_through=args.completed_through)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(digest, encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve())}))
    elif args.command == "run":
        from .experiment import run_experiment
        from .report import render_report
        run_experiment(Snapshot(args.snapshot), config, Fold(**json.loads(args.fold.read_text())),
                       args.horizon, args.seed, args.output, args.engineering, args.equal_information,
                       args.training_only, args.test_fixture)
        print(render_report(args.output).resolve())
    elif args.command == "demo":
        from .experiment import run_experiment
        from .report import render_report
        if args.epochs < 1:
            parser.error("epochs must be positive")
        snapshot = create_fixture(args.output / "source", args.output / "snapshot")
        config = copy.deepcopy(config)
        config["model"]["max_epochs"] = args.epochs
        # Exercise score-dependent selection with a fixture smaller than the production pool.
        config["portfolio"]["max_positions"] = 6
        config["portfolio"]["parameters_status"] = "synthetic_demo_six_positions"
        dates = snapshot.dates
        fold = Fold(dates[120], dates[240], dates[310], dates[375])
        for horizon in config["horizons"]:
            output = args.output / f"h{horizon}-seed17"
            print(f"Starting engineering verification: horizon={horizon}, seed=17", flush=True)
            run_experiment(snapshot, config, fold, horizon, 17, output, True, args.equal_information,
                           allow_partial_universe=True)
            print(render_report(output).resolve(), flush=True)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    raise ValueError("tabular input must be CSV or Parquet")


def _require_new_outputs(path: Path) -> None:
    metadata = path.with_suffix(path.suffix + ".json")
    if path.exists() or metadata.exists():
        raise FileExistsError("Research outputs use new output paths")


def _write_frame_and_metadata(path: Path, frame: pd.DataFrame, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError("output must be CSV or Parquet")
    write_json(path.with_suffix(path.suffix + ".json"), metadata)


def _parse_report_selector(value: str) -> tuple[str, str]:
    parts = value.split(":", 1)
    if len(parts) != 2 or len(parts[0]) != 4 or not parts[0].isdigit():
        raise ValueError("report selector must use YEAR:TYPE")
    return parts[0], parts[1]


if __name__ == "__main__":
    main()
