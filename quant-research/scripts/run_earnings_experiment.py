"""Run an immutable offline earnings research bundle; never certify a real profit result."""
import argparse
import json
from pathlib import Path

from quant_research.earnings_runner import run_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_bundle(args.inputs, args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"], "formal_profit_qualified": False}))


if __name__ == "__main__":
    main()
