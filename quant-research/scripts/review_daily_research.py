"""Refresh compact publication evidence and research scorecards without training."""
import argparse
from pathlib import Path

from quant_research.daily_loop import read
from quant_research.research_review import refresh_research


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/research-review-v1.json"))
    parser.add_argument("--event-overlay", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, help="Optional public dashboard JSON export")
    parser.add_argument("--execution-evidence", type=Path,
                        help="Immutable canonical execution bundle with archived coverage proofs")
    parser.add_argument("--include-execution", action="store_true", help="Explicitly run execution checks and simulation; default is analysis only")
    args = parser.parse_args()
    source = args.source or Path(read(args.store / "current.json")["source"])
    print(refresh_research(args.store, source, read(args.config), args.event_overlay,
                           args.output, args.execution_evidence,
                           analysis_only=not (args.include_execution or args.execution_evidence)))


if __name__ == "__main__":
    main()
