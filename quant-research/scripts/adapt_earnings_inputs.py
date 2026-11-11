"""Copy a daily training refresh into canonical research structure without qualification."""
import argparse
import json
from pathlib import Path

from quant_research.earnings_inputs import adapt_daily_training_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = adapt_daily_training_input(args.source, args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"], "formal_profit_qualified": False}))


if __name__ == "__main__":
    main()
