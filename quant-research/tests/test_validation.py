import copy

import pandas as pd
import pytest

from quant_research.storage import freeze, validate_tables


def test_string_boolean_cannot_certify_risk_history(snapshot, tmp_path):
    declaration = copy.deepcopy(snapshot.manifest["declaration"])
    declaration["st_history_verified"] = "false"
    with pytest.raises(ValueError, match="explicit boolean"):
        freeze(snapshot.path, tmp_path / "unsafe", declaration)


def test_overlapping_fee_regimes_are_rejected(snapshot):
    rules = snapshot.tables["rules"]
    snapshot.tables["rules"] = pd.concat([rules, rules.iloc[:1]])
    with pytest.raises(ValueError, match="Overlapping"):
        validate_tables(snapshot.tables)


def test_negative_fee_or_zero_lot_is_rejected(snapshot):
    snapshot.tables["rules"].loc[0, "buy_step"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        validate_tables(snapshot.tables)


def test_bar_outside_exchange_calendar_is_rejected(snapshot):
    snapshot.tables["bars"].loc[0, "date"] = "2020-01-01"
    with pytest.raises(ValueError, match="outside its exchange"):
        validate_tables(snapshot.tables)

