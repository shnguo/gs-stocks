#!/usr/bin/env python3
"""Run a manual, repeatable daily research cycle without trading or training."""

import argparse
import json
from pathlib import Path

from quant_research.daily_loop import generate, read, register_model, review


def cycle(source, package, store, date, cutoff, mode, observed_through):
    """Generate today's immutable draft; append reviews of earlier drafts."""
    store = Path(store)
    generated = generate(source, package, store, date, cutoff, mode)
    results = []
    failures = []
    for f in sorted((store / "runs").glob("*/*/run.json")):
        run = read(f)
        if (
            run["status"] != "research_draft"
            or run["input"]["mode"] != mode
            or run["input"]["trade_date"] > observed_through
        ):
            continue
        try:
            # Clamp maturity to the original horizon; no automatic deadline extension.
            day = min(observed_through, run["input"]["horizon_dates"][-1])
            results.append(str(review(f.parent, source, store, day)))
        except (ValueError, OSError) as e:
            failures.append({"run": str(f.parent), "error": str(e)})
    return {
        "generated": str(generated),
        "status": read(generated / "run.json")["status"],
        "reviews": results,
        "review_errors": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--pilot", required=True, type=Path)
    reg.add_argument("--quality", required=True, type=Path)
    reg.add_argument("--store", required=True, type=Path)
    for name in ["generate", "cycle"]:
        p = sub.add_parser(name)
        p.add_argument("--source", required=True, type=Path)
        p.add_argument("--package", required=True, type=Path)
        p.add_argument("--store", required=True, type=Path)
        p.add_argument("--date", required=True)
        p.add_argument("--cutoff", required=True, help="ISO timestamp with timezone")
        p.add_argument(
            "--mode", choices=["replay", "prospective", "intraday_research"], default="prospective"
        )
        if name == "cycle":
            p.add_argument("--observed-through", required=True, help="Last completed session")
    p = sub.add_parser("review")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--observed-through", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    handlers = {"register": register_model, "generate": generate, "review": review, "cycle": cycle}
    if command == "generate":
        args["root"] = args.pop("source")
        args["trade_date"] = args.pop("date")
    result = handlers[command](**args)
    print(
        json.dumps(
            result if isinstance(result, dict) else {"path": str(result)},
            ensure_ascii=False,
            indent=2,
        )
    )
    if command == "cycle" and (result["review_errors"] or result["status"] == "blocked"):
        raise SystemExit(2)
    if command == "generate" and read(result / "run.json")["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
