from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .research import run_research
from .sources import SourceCache, collect_filings, collect_prices, load_alpaca_credentials


def main():
    parser = argparse.ArgumentParser(description="Read-only US earnings research")
    parser.add_argument("command", choices=["collect", "backtest", "audit", "notebook", "all"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--credentials-file", type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    config_path = args.config.resolve() if args.config else args.root / "configs/pilot-v1.json"
    config = json.loads(config_path.read_text())
    if args.command in ("collect", "all"):
        out = args.root / "artifacts" / config.get("artifact_subdir", "")
        out.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        lock = out / "protocol.sha256"
        if lock.exists() and lock.read_text().strip() != digest:
            raise ValueError("Protocol changed; create a new protocol version/output root")
        lock.write_text(digest)
        cache = SourceCache(args.root / "data/raw")
        collect_prices(config, cache, load_alpaca_credentials(args.credentials_file), args.root / "data")
        collect_filings(config, cache, args.root / "data")
    if args.command in ("backtest", "all"):
        result = run_research(args.root, config_path)
        print(result["summary"].loc[result["summary"].period.str.contains("later evaluation")].to_string(index=False))
    if args.command in ("audit", "all"):
        from .audit import audit_run
        print(json.dumps(audit_run(args.root, config_path), indent=2))
    if args.command in ("notebook", "all"):
        from .notebook import build_notebook
        build_notebook(args.root, config_path)


if __name__ == "__main__":
    main()
