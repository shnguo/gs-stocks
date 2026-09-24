"""Independent, offline research comparison of the fixed September 7 captures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit outputs are immutable")
    lineage = {}

    def read(name: str) -> dict:
        raw = (args.artifacts / name).read_bytes()
        lineage[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def bao(name: str) -> pd.DataFrame:
        data = read(name)
        return pd.DataFrame(data["rows"], columns=data["fields"])

    def tushare(name: str) -> pd.DataFrame:
        root = read(name)
        assert root["code"] == 0
        return pd.DataFrame(root["data"]["items"], columns=root["data"]["fields"])

    daily = bao("p1-baostock-rust-600036-daily-20260907-v2/records.json").set_index("date")
    other = tushare("p1-tushare-daily-600036-20260907/response.json")
    other.index = pd.to_datetime(other.trade_date).dt.strftime("%Y-%m-%d")
    assert not daily.index.duplicated().any() and not other.index.duplicated().any()
    assert set(daily.index) == set(other.index)
    other = other.reindex(daily.index)
    prices = {}
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        bv = daily[field].astype(float)
        tv = other["vol" if field == "volume" else field].astype(float)
        tv *= 100 if field == "volume" else 1000 if field == "amount" else 1
        difference = (bv - tv).abs()
        tolerance = 0.0001 if field in {"open", "high", "low", "close"} else 1.01
        prices[field] = {"max_absolute_difference": float(difference.max()),
                         "max_relative_difference": float((difference / tv.abs()).max()),
                         "rows_above_comparison_tolerance": int((difference > tolerance).sum()),
                         "comparison_tolerance": tolerance}
    assert all(prices[x]["rows_above_comparison_tolerance"] == 0
               for x in ["open", "high", "low", "close", "volume"])

    calendar = bao("p1-baostock-rust-calendar-20260907-v2/records.json")
    cal = tushare("p1-calendar-sse-20260907/response.json")
    cal["calendar_date"] = pd.to_datetime(cal.cal_date).dt.strftime("%Y-%m-%d")
    assert set(calendar.calendar_date) == set(cal.calendar_date)
    common_calendar = calendar.merge(cal, on="calendar_date", validate="one_to_one")
    assert (common_calendar.is_trading_day.astype(int) == common_calendar.is_open).all()

    factor = bao("p1-baostock-rust-600036-factor-20260907-v1/records.json")
    bf = factor.set_index("dividOperateDate").backAdjustFactor.astype(float).sort_index()
    tf = tushare("p1-factor-sample-20260907/response.json")
    tf.index = pd.to_datetime(tf.trade_date).dt.strftime("%Y-%m-%d")
    tf = tf.sort_index().adj_factor
    expanded = bf.reindex(sorted(set(bf.index) | set(tf.index))).ffill().reindex(tf.index)
    assert expanded.notna().all()
    relative_bao, relative_tu = expanded / expanded.iloc[0], tf / tf.iloc[0]
    factor_drift = (relative_bao / relative_tu - 1).abs()
    change_drift = (relative_bao.pct_change() - relative_tu.pct_change()).abs()

    sdk_checks = {}
    for name, directory, probe in [
        ("daily", "p1-baostock-rust-600036-daily-20260907-v2", "600036-daily-probe.json"),
        ("calendar", "p1-baostock-rust-calendar-20260907-v2", "calendar-probe.json"),
        ("stock_basic", "p1-baostock-rust-stock-basic-20260907-v2", "stock_basic-probe.json"),
        ("factor", "p1-baostock-rust-600036-factor-20260907-v1", "factor-probe.json"),
    ]:
        left = bao(directory + "/records.json")
        reference = read("p1-baostock-probe-20260907/" + probe)
        right = pd.DataFrame(reference["rows"], columns=reference["fields"])
        if name == "stock_basic":
            raw_rows = [row for message in reference["response_messages"]
                        for row in json.loads(message[21:].split("\x01")[6])["record"]]
            pd.testing.assert_frame_equal(left, pd.DataFrame(raw_rows, columns=left.columns))
            pd.testing.assert_frame_equal(left.drop(columns="code_name"),
                                          right.drop(columns="code_name"))
            differences = left.code_name != right.code_name
            sdk_checks[name] = {"rows": len(left), "raw_message_exact_match": True,
                                "sdk_whitespace_removed_name_rows": int(differences.sum()),
                                "name_differences": [
                                    {"code": left.loc[i, "code"], "raw": left.loc[i, "code_name"],
                                     "sdk": right.loc[i, "code_name"]}
                                    for i in left.index[differences]]}
        else:
            pd.testing.assert_frame_equal(left, right)
            sdk_checks[name] = {"rows": len(left), "exact_rows_and_fields_match": True}

    master = bao("p1-baostock-rust-stock-basic-20260907-v2/records.json")
    inactive = master.loc[(master.type == "1") & (master.status == "0")].copy()
    inactive["ts_code"] = inactive.code.str[3:] + "." + inactive.code.str[:2].str.upper()
    retired = tushare("p1-delisted-20260907/response.json")
    retired = retired.loc[retired.exchange.isin(["SSE", "SZSE"])]
    shared = inactive.merge(retired, on="ts_code", validate="one_to_one")
    different_dates = shared.loc[shared.outDate.str.replace("-", "") != shared.delist_date]
    result = {
        "purpose": "independent_offline_research_audit", "complete_history_verified": False,
        "lineage_sha256": lineage, "official_sdk_replay": sdk_checks,
        "price_comparison": {"code": "600036.SH", "rows": len(daily),
                             "first_date": daily.index.min(), "last_date": daily.index.max(),
                             "units": {"volume": "shares", "amount": "CNY"}, "fields": prices},
        "calendar_comparison": {"civil_days": len(calendar),
                                "open_days": int(calendar.is_trading_day.astype(int).sum()),
                                "sse_mismatches": 0, "bse_verified": False},
        "factor_comparison": {"raw_factor_scales_are_different": True,
                              "max_normalized_curve_relative_difference": float(factor_drift.max()),
                              "max_daily_factor_change_difference": float(change_drift.max()),
                              "exact_equivalence_verified": False,
                              "interpretation": "source precision/method differences remain; no wealth-return certification"},
        "inactive_identity_comparison": {
            "baostock_inactive_stock_rows": len(inactive), "tushare_shsz_delisted_rows": len(retired),
            "shared_codes": len(shared),
            "only_tushare": retired.loc[~retired.ts_code.isin(inactive.ts_code),
                                        ["ts_code", "name", "list_date", "delist_date"]].to_dict("records"),
            "only_baostock": inactive.loc[~inactive.ts_code.isin(retired.ts_code),
                                           ["ts_code", "code_name", "ipoDate", "outDate"]].to_dict("records"),
            "date_conflicts": different_dates[["ts_code", "outDate", "delist_date"]].to_dict("records"),
            "automatic_merge_performed": False,
            "interpretation": "provider inactive status may include identity/code transitions; needs effective-date evidence"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "sdk_checks": sdk_checks,
                      "calendar_mismatches": 0, "price_days_compared": len(daily)}))


if __name__ == "__main__":
    main()
