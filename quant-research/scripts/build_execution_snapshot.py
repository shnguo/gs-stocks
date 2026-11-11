"""Import archived execution evidence for one frozen decision; report every data gap."""
import argparse
from pathlib import Path

from quant_research.execution_snapshot import build_execution_snapshot
from quant_research.research_review import outcome_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    result = build_execution_snapshot(args.decision, outcome_snapshot(args.source / "snapshot"), args.evidence)
    print(result / "gap-report.json")


if __name__ == "__main__":
    main()
