"""Index full approved daily history and encode a date-balanced historical pool."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant_research.kronos_ranker import timestamps
from quant_research.price_strategy import price_labels
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_history import future_labels, partition_indices, stratified_pool
from quant_research.token_transformer import load_tokenizer, normalize_training, valid_bars

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
OUT = ART / "token-history-data-20260915-v1"
ROOT = ART / "full-market-training-20260910-v1"
INPUT = ART / "kronos-inputs-20260914-v3"
BUNDLE = ART / "kronos-comparison-20260910-v1"


def read(p):
    return json.loads(p.read_text())


def mmap(name, dtype, shape, fill=None):
    v = np.lib.format.open_memmap(OUT / name, mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        v[:] = fill
    return v


def main(output=None):
    global OUT
    if output is not None:
        OUT = Path(output).resolve()
    cfg = read(BASE / "configs/token-history-v1.json")
    OUT.mkdir()
    write_json(OUT / "protocol.json", cfg)
    sources = {}

    def check(p, expected):
        actual = file_hash(p)
        if actual != expected:
            raise ValueError(f"Source changed: {p}")
        sources[str(p)] = actual

    acceptance = read(ROOT / "development-acceptance.json")
    assert acceptance["passed"]
    for name, expected in acceptance["evidence"].items():
        check(ROOT / name, expected)
    pm = read(ROOT / "panel-h5/manifest.json")
    check(ROOT / "panel-h5/manifest.json", acceptance["panels"]["5"])
    check(ROOT / "panel-h5/samples.parquet", pm["files"]["samples.parquet"])
    sm = read(ROOT / "snapshot/manifest.json")
    for name in ["bars.parquet", "actions.parquet"]:
        check(ROOT / "snapshot" / name, sm["files"][name])
    im = read(INPUT / "manifest.json")
    check(INPUT / "values.npy", im["values_sha256"])
    assert im["dataset_id"] == pm["dataset_id"] == acceptance["dataset_id"]
    assert im["dates"][-1] == cfg["last_label"] < cfg["sealed_holdout_start"]
    dates, symbols = im["dates"], im["instruments"]
    assert dates == pm["dates"][: len(dates)] and symbols == pm["instruments"]
    shape = (len(symbols), len(dates))
    raw = np.load(INPUT / "values.npy", mmap_mode="r")
    good = valid_bars(raw[..., :6]) & np.isfinite(raw[..., 6]) & (raw[..., 6] > 0)
    cumulative = np.pad(np.cumsum(~good, axis=1), ((0, 0), (1, 0)))
    complete = np.zeros(shape, bool)
    complete[:, 59:] = (cumulative[:, 60:] - cumulative[:, :-60]) == 0
    del cumulative, good
    bars = pd.read_parquet(
        ROOT / "snapshot/bars.parquet", filters=[("date", "<=", cfg["last_label"])]
    )
    stock = pd.Categorical(bars.instrument_id, categories=symbols).codes.astype(np.int32)
    time = pd.Categorical(bars.date, categories=dates).codes.astype(np.int32)
    assert (stock >= 0).all() and (time >= 0).all()
    assert not bars.duplicated(["instrument_id", "date"]).any()
    quotes = mmap("quotes.npy", np.float64, (*shape, 7), np.nan)
    quotes[stock, time] = bars[
        ["open", "high", "low", "close", "volume", "amount", "factor"]
    ].to_numpy(float)
    priced = mmap("priced.npy", bool, shape, False)
    q = quotes[stock, time]
    priced[stock, time] = (
        valid_bars(q[:, :6])
        & np.isfinite(q[:, 6])
        & (q[:, 6] > 0)
        & (q[:, 4:6] > 0).all(1)
        & bars.source_trade_status.ne(0).to_numpy()
    )
    del q
    sequences = []
    for field in ["sequence_id", "label_sequence_id"]:
        seq = mmap(field + ".npy", np.int32, shape, -1)
        codes, _ = pd.factorize(bars[field].replace("", None), sort=False)
        seq[stock, time] = codes
        sequences.append(seq)
    actions_df = pd.read_parquet(
        ROOT / "snapshot/actions.parquet", filters=[("ex_date", "<=", cfg["last_label"])]
    )
    actions = mmap("actions.npy", bool, shape, False)
    asi = pd.Categorical(actions_df.instrument_id, categories=symbols).codes
    ati = pd.Categorical(actions_df.ex_date, categories=dates).codes
    ok = (asi >= 0) & (ati >= 0)
    actions[asi[ok], ati[ok]] = True
    candidates = pd.read_parquet(
        ROOT / "panel-h5/samples.parquet",
        filters=[("date", ">=", cfg["research_start"]), ("date", "<=", cfg["last_signal"])],
        columns=[
            "date",
            "instrument_id",
            "stock_index",
            "date_index",
            "risk_status",
            "trading_eligible",
            "source_is_st",
        ],
    )
    s, t = candidates[["stock_index", "date_index"]].to_numpy(int).T
    available = complete[s, t] & np.isfinite(quotes[s, t, 3]) & (quotes[s, t, 3] > 0)
    coverage = (
        candidates.assign(input_available=available)
        .groupby("date")
        .agg(
            inherited_eligible=("instrument_id", "size"),
            history_and_reference_available=("input_available", "sum"),
        )
    )
    coverage.to_csv(OUT / "daily-coverage.csv")
    candidates = (
        candidates.loc[available].sort_values(["date", "instrument_id"]).reset_index(drop=True)
    )
    coords = candidates[["stock_index", "date_index"]].to_numpy(np.int32)
    np.save(OUT / "all-eligible-coordinates.npy", coords)
    selected = []
    for day, g in candidates.groupby("date", sort=True):
        stock_ids = g.stock_index.to_numpy(int)
        chosen = stratified_pool(stock_ids, symbols, day, cfg["pool_per_date"], cfg["seed"])
        indexed = g.set_index("stock_index", drop=False)
        selected.append(indexed.loc[chosen].reset_index(drop=True))
    rows = pd.concat(selected, ignore_index=True)
    assert len(rows) > 0 and rows.date.nunique() == candidates.date.nunique(), (
        "Date sampling dropped eligible history"
    )
    rows["row_id"] = np.arange(len(rows))
    rows["label_end"] = [dates[t + 5] for t in rows.date_index]
    rows.to_parquet(OUT / "rows.parquet", index=False)
    total_candidates = len(candidates)
    del candidates, coords, complete, stock, time
    # Independent parity against the established label implementation before encoding.
    parity = rows.iloc[np.linspace(0, len(rows) - 1, 192, dtype=int)]
    labels = price_labels(bars, parity, dates, actions_df, "2025-07-30")
    ff, vv = future_labels(
        quotes,
        priced,
        *sequences,
        actions,
        parity.stock_index.to_numpy(),
        parity.date_index.to_numpy(),
    )
    np.testing.assert_array_equal(vv, labels["valid"])
    np.testing.assert_array_equal(ff, np.concatenate([labels["future"], labels["turnover"]], -1))
    write_json(OUT / "label-parity.json", dict(passed=True, rows=len(parity)))
    del bars, actions_df, labels, ff, vv
    for x in [quotes, priced, *sequences, actions]:
        x.flush()
    np.save(OUT / "calendar-stamps.npy", timestamps(dates).astype(np.int64))
    write_json(OUT / "calendar.json", dates)
    write_json(OUT / "instruments.json", symbols)
    folds = {}
    for fold in cfg["folds"]:
        parts = {}
        for part in ["train", "selection", "evaluation"]:
            ids = partition_indices(rows, dates, *fold[part])
            np.save(OUT / f"{fold['name']}-{part}.npy", ids)
            f = rows.iloc[ids]
            assert len(f) > 0 and f.label_end.max() < fold[part][1]
            parts[part] = dict(
                rows=len(ids),
                dates=int(f.date.nunique()),
                start=f.date.min(),
                end=f.date.max(),
                last_target=f.label_end.max(),
            )
        assert parts["train"]["last_target"] < parts["selection"]["start"]
        assert parts["selection"]["last_target"] < parts["evaluation"]["start"]
        folds[fold["name"]] = parts
    write_json(OUT / "split-audit.json", dict(passed=True, folds=folds))
    print(
        "Candidate windows",
        total_candidates,
        "selected pool",
        len(rows),
        "dates",
        rows.date.nunique(),
        flush=True,
    )
    n = len(rows)
    outputs = {
        "s1": mmap("s1.npy", np.int16, (n, 65)),
        "s2": mmap("s2.npy", np.int16, (n, 65)),
        "valid": mmap("valid.npy", bool, (n, 5)),
        "future": mmap("future.npy", np.float64, (n, 5, 6)),
        "mean": mmap("mean.npy", np.float32, (n, 1, 6)),
        "scale": mmap("scale.npy", np.float32, (n, 1, 6)),
        "last": mmap("last.npy", np.float32, (n, 6)),
    }
    torch.set_num_threads(4)
    tokenizer = load_tokenizer(BUNDLE, cfg["device"])
    for start in range(0, n, cfg["encoding_batch_size"]):
        end = min(start + cfg["encoding_batch_size"], n)
        sub = rows.iloc[start:end]
        s, t = sub[["stock_index", "date_index"]].to_numpy(int).T
        history = raw[s[:, None], t[:, None] + np.arange(-59, 1)].copy()
        future, known = future_labels(quotes, priced, *sequences, actions, s, t)
        normalized, valid, mean, scale = normalize_training(history, future, known, cfg["clip"])
        with torch.inference_mode():
            a, b = tokenizer.encode(torch.from_numpy(normalized).to(cfg["device"]), half=True)
        for key, value in dict(
            s1=a.cpu().numpy(),
            s2=b.cpu().numpy(),
            valid=valid,
            future=future,
            mean=mean,
            scale=scale,
            last=history[:, -1, :6],
        ).items():
            outputs[key][start:end] = value
        if start == 0:
            np.savez_compressed(
                OUT / "prefix-reference.npz",
                normalized=normalized[:2],
                s1=a[:2].cpu().numpy(),
                s2=b[:2].cpu().numpy(),
            )
        if start % (cfg["encoding_batch_size"] * 40) == 0:
            progress = dict(stage="encoding", rows=end, total=n, at=utc_now())
            write_json(OUT / "progress.json", progress)
            print(progress, flush=True)
    for x in outputs.values():
        x.flush()
    stamps = np.load(OUT / "calendar-stamps.npy")
    month_coverage = {}
    for fold in cfg["folds"]:
        ids = np.load(OUT / f"{fold['name']}-train.npy")
        t = rows.iloc[ids].date_index.to_numpy()
        months = np.unique(stamps[t[:, None] + np.arange(-59, 1), 4])
        assert months.tolist() == list(range(1, 13))
        month_coverage[fold["name"]] = months.tolist()
    provenance = read(BUNDLE / "pretrained-provenance.json")
    sources[str(BUNDLE / "pretrained-provenance.json")] = file_hash(
        BUNDLE / "pretrained-provenance.json"
    )
    sources[str(INPUT / "manifest.json")] = file_hash(INPUT / "manifest.json")
    write_json(OUT / "sources.json", sources)
    write_json(
        OUT / "summary.json",
        dict(
            candidate_windows=total_candidates,
            selected_pool=n,
            dates=int(rows.date.nunique()),
            known_rows=int(outputs["valid"].any(1).sum()),
            known_future_days=int(outputs["valid"].sum()),
            training_months=month_coverage,
            folds=folds,
            dataset_id=im["dataset_id"],
            created_at=utc_now(),
            tokenizer_revision=provenance["tokenizer_revision"],
            tokenizer_pretraining_dates_verified=False,
            sampling="Every eligible date indexed; outcome-independent 192-stock pool per date; 32 stocks/date/epoch rotate through known-label pool",
            limitations=acceptance["limitations"],
            sealed_holdout_accessed=False,
        ),
    )
    code = OUT / "code"
    shutil.copytree(BASE / "src", code / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (code / "scripts").mkdir()
    for name in [
        "prepare_token_history.py",
        "token_history_run.py",
        "evaluate_token_history.py",
        "compare_token_range.py",
        "compare_token_kronos_paths.py",
    ]:
        shutil.copy2(BASE / "scripts" / name, code / "scripts" / name)
    write_json(
        OUT / "completed.json",
        dict(
            passed=True,
            files={
                str(p.relative_to(OUT)): file_hash(p)
                for p in OUT.rglob("*")
                if p.is_file() and p.name != "progress.json"
            },
        ),
    )
    print("Completed historical dataset", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    main(parser.parse_args().output)
