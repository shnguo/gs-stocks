from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    config.setdefault("universe_scope", "all_a_shares")
    if config["universe_scope"] != "all_a_shares":
        raise ValueError("Quantitative selection requires all_a_shares; personal stock lists are unsupported")
    if config.get("universe_catalog"):
        catalog = Path(config["universe_catalog"])
        config["universe_catalog"] = str((Path(path).resolve().parent / catalog).resolve())
    if set(config["exchanges"]) != {"XSHG", "XSHE", "XBSE"}:
        raise ValueError("The approved research universe must include all three A-share exchanges")
    # Preserve the meaning of archived v1 configurations without this field.
    config.setdefault("training_risk_policy", "exclude_st")
    if config["training_risk_policy"] not in {"include", "exclude_st"}:
        raise ValueError("Invalid training risk policy")
    if set(config["exclude_statuses"]) != {"ST", "*ST"}:
        raise ValueError("ST and *ST exclusions remain mandatory for simulated purchases")
    if config["unknown_status_policy"] != "exclude":
        raise ValueError("Unknown risk status must not be interpreted as normal")
    if config["lookback"] < 2 or set(config["horizons"]) != {5, 20}:
        raise ValueError("Invalid research horizons or lookback")
    p = config["portfolio"]
    if not (0 < p["max_weight"] <= 1 and 0 < p["max_drawdown"] < 1):
        raise ValueError("Invalid portfolio risk limits")
    if p["initial_cash"] <= 0 or p["max_positions"] < 1:
        raise ValueError("Invalid portfolio size")
    if not 0 < p["adv_participation"] <= 1 or not 0 <= p["industry_deviation"] <= 1:
        raise ValueError("Invalid industry or liquidity limit")
    m = config["model"]
    if (m["hidden_size"] % m["heads"] or min(m["layers"], m["hidden_size"], m["heads"],
           m["feedforward_size"], m["batch_size"], m["max_epochs"], m["patience"]) < 1):
        raise ValueError("Invalid Transformer shape or training budget")
    return config
