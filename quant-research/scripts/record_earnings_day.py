"""Manual research journal: engineering replay or actual-clock rejected-input observation."""
import argparse
import json
from pathlib import Path

from quant_research.earnings_continuation import (
    continue_engineering_replay,
    observe_unaccepted_inputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    replay = commands.add_parser("engineering-replay")
    replay.add_argument("--store", type=Path, required=True)
    replay.add_argument("--inputs", type=Path, required=True)
    observe = commands.add_parser("observe-unaccepted")
    observe.add_argument("--store", type=Path, required=True)
    observe.add_argument("--snapshot", type=Path, required=True)
    observe.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    result = (continue_engineering_replay(args.store, args.inputs) if args.mode == "engineering-replay"
              else observe_unaccepted_inputs(args.store, args.snapshot, args.protocol))
    print(json.dumps({k: result[k] for k in ["attempt_id", "kind", "status", "recorded_at", "formal_profit_qualified"]}))


if __name__ == "__main__":
    main()
