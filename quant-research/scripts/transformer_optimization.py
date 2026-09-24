"""Selection-only neural diagnostic screen, then frozen matched confirmation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from context_transformer_run import finalize, fit
from price_pilot import verify_files

from quant_research.storage import file_hash, utc_now, write_json

VARIANTS = {
    "reference": {
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "dropout": 0.1,
        "price_auxiliary_weight": 0.1,
    },
    "lower_lr": {
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "dropout": 0.1,
        "price_auxiliary_weight": 0.1,
    },
    "regularized": {
        "learning_rate": 0.0003,
        "weight_decay": 0.1,
        "dropout": 0.3,
        "price_auxiliary_weight": 0.1,
    },
    "no_price_aux": {
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "dropout": 0.1,
        "price_auxiliary_weight": 0.0,
    },
}


def read(p):
    return json.loads(p.read_text())


def identity(variant, seed, fold):
    return f"{variant}-s{seed}-f{fold:02d}"


def choose_challenger(scores):
    """No evaluation metrics accepted; deterministic tie break by variant name."""
    if set(scores) != set(VARIANTS):
        raise ValueError("Incomplete selection screen")
    return min((scores[v], v) for v in scores if v != "reference")[1]


def freeze(args):
    root, parent = args.output.resolve(), args.parent.resolve()
    state = read(parent / "run-status.json")
    assert state["status"] == "completed" and read(parent / "verification.json")["passed"]
    verify_files(parent, state["files"])
    e = read(parent / "experiment.json")
    for key in ["prior", "features", "baseline"]:
        source = Path(e[key])
        verify_files(source, read(source / "run-status.json")["files"])
    panel = Path(e["panel"])
    verify_files(panel, read(panel / "manifest.json")["files"])
    root.mkdir()
    cfg = read(parent / "protocol.json")
    cfg.update(diagnostics=True, max_epochs=12, min_epochs=8)
    write_json(
        root / "protocol.json",
        dict(
            protocol_id="context-transformer-optimization-v1",
            parent=str(parent),
            parent_status_sha256=file_hash(parent / "run-status.json"),
            base_config=cfg,
            variants=VARIANTS,
            screen_fold=7,
            screen_seed=17,
            extra_seeds=[29, 43],
            confirmation_folds=[1, 15],
            selection="Lowest selection-only conditional-return MSE among three challengers; reference remains a matched control. No calibration/evaluation labels used to choose configuration.",
            primary="Seed17 challenger versus reference across folds1/7/15; raw return MSE",
            secondary="Reference versus chosen challenger at seeds17/29/43 in fold7; raw and calibrated metrics separately. This is not a full three-seed three-window grid.",
            epoch_budget="12 epochs maximum, 8 minimum; cap-limited runs explicitly flagged, not called converged",
            diagnostic="Initial constant-prior and epoch0 MSE; all-head training losses, auxiliary loss and a deterministic 1024-row training probe. Epoch0 is diagnostic only, not eligible for checkpoint selection.",
            quality_promotion=False,
            executable=False,
            created_at=utc_now(),
        ),
    )
    base = Path(__file__).resolve().parents[1]
    for part in ["src", "scripts", "tests"]:
        shutil.copytree(
            base / part, root / f"code/{part}", ignore=shutil.ignore_patterns("__pycache__")
        )
    shutil.copy2(base / "uv.lock", root / "code/uv.lock")
    write_json(
        root / "frozen-manifest.json",
        {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()},
    )


def make_trial(root, variant, seed, fold):
    protocol = read(root / "protocol.json")
    parent = Path(protocol["parent"])
    out = root / "trials" / identity(variant, seed, fold)
    out.mkdir(parents=True)
    config = dict(protocol["base_config"], **protocol["variants"][variant], seed=seed)
    config["protocol_id"] = "optimization-" + identity(variant, seed, fold)
    e = read(parent / "experiment.json")
    e.update(fold_indices=[fold], optimization_variant=variant, optimization_seed=seed)
    write_json(out / "protocol.json", config)
    write_json(out / "experiment.json", e)
    write_json(out / "context-manifest.json", read(parent / "context-manifest.json"))
    write_json(
        out / "frozen-manifest.json",
        {str(p.relative_to(out)): file_hash(p) for p in out.iterdir() if p.is_file()},
    )
    return out


def dispatch(root, stage, trial):
    verify_files(root, read(root / "frozen-manifest.json"))
    write_json(root / "progress.json", dict(stage=stage, trial=trial.name, updated_at=utc_now()))
    with (root / (trial.name + "." + stage + ".log")).open("x") as log:
        subprocess.run(
            [
                sys.executable,
                str(root / "code/scripts/transformer_optimization.py"),
                stage,
                "--output",
                str(root),
                "--trial",
                str(trial),
            ],
            env={**os.environ, "PYTHONPATH": str(root / "code/src")},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(stage, trial.name, "completed", flush=True)


def run(args):
    root = args.output.resolve()
    verify_files(root, read(root / "frozen-manifest.json"))
    p = read(root / "protocol.json")
    parent = Path(p["parent"])
    assert file_hash(parent / "run-status.json") == p["parent_status_sha256"]
    verify_files(parent, read(parent / "run-status.json")["files"])
    if (root / "run-status.json").exists():
        raise FileExistsError("Preserve attempts")
    write_json(
        root / "run-status.json", dict(status="running", pid=os.getpid(), started_at=utc_now())
    )
    try:
        screen, scores = {}, {}
        for variant in p["variants"]:
            t = make_trial(root, variant, p["screen_seed"], p["screen_fold"])
            dispatch(root, "train", t)
            screen[variant] = t
            scores[variant] = read(t / f"fold-{p['screen_fold']:02d}/training-result.json")[
                "selection_net_mse"
            ]
        chosen = choose_challenger(scores)
        selection_files = {}
        for t in screen.values():
            for name in [
                "training-result.json",
                "training-log.json",
                "transformer.pt",
                "trained.json",
            ]:
                path = t / f"fold-{p['screen_fold']:02d}" / name
                selection_files[str(path.relative_to(root))] = file_hash(path)
        write_json(
            root / "selection-lock.json",
            dict(
                chosen=chosen,
                selection_scores=scores,
                screen_files=selection_files,
                no_evaluation_forecasts_present=not any(
                    root.glob("trials/*/fold-*/*evaluation.npz")
                ),
                selected_at=utc_now(),
                quality_promotion=False,
            ),
        )
        chosen_trials = [screen["reference"], screen[chosen]]
        for seed in p["extra_seeds"]:
            for variant in ["reference", chosen]:
                t = make_trial(root, variant, seed, p["screen_fold"])
                dispatch(root, "train", t)
                chosen_trials.append(t)
        for fold in p["confirmation_folds"]:
            t = make_trial(root, chosen, p["screen_seed"], fold)
            dispatch(root, "train", t)
            chosen_trials.append(t)
        # Configuration is already locked. Only matched controls and that challenger are evaluated.
        for t in chosen_trials:
            dispatch(root, "evaluate", t)
            dispatch(root, "verify", t)
        summarize(root, chosen_trials)
        write_json(
            root / "run-status.json",
            dict(
                status="completed",
                finished_at=utc_now(),
                files={
                    str(f.relative_to(root)): file_hash(f)
                    for f in root.rglob("*")
                    if f.is_file()
                    and f != root / "run-status.json"
                    and "__pycache__" not in f.parts
                },
            ),
        )
    except BaseException as exc:
        write_json(root / "run-status.json", dict(status="failed", error=repr(exc), at=utc_now()))
        raise


def summarize(root, trials):
    p, lock = read(root / "protocol.json"), read(root / "selection-lock.json")
    original = pd.read_csv(Path(p["parent"]) / "head-metrics.csv")
    frames, decisions = [], []
    for t in trials:
        e = read(t / "experiment.json")
        for filename, target in [("head-metrics.csv", frames), ("decision-metrics.csv", decisions)]:
            frame = pd.read_csv(t / filename)
            frame = frame[
                frame.model.isin(["context_transformer", "context_transformer_raw"])
            ].copy()
            frame["variant"] = e["optimization_variant"]
            frame["seed"] = e["optimization_seed"]
            target.append(frame)
    old = original[
        original.model.isin(["context_transformer", "context_transformer_raw"])
        & original.window.isin(["fold-01", "fold-15"])
    ].copy()
    old["variant"], old["seed"] = "reference", 17
    frames.append(old)
    data = pd.concat(frames, ignore_index=True)
    assert not data.duplicated(["window", "date", "model", "head", "variant", "seed"]).any()
    data.to_csv(root / "head-metrics.csv", index=False)
    pd.concat(decisions, ignore_index=True).to_csv(root / "decision-metrics.csv", index=False)
    from quant_research.plan_value import compare_heads

    pairs = {}
    for suffix in ["_raw", ""]:
        x = data[data.seed.eq(17) & data.model.eq("context_transformer" + suffix)].copy()
        x["model"] = x.variant.map({"reference": "empirical", lock["chosen"]: "learned"})
        pairs["challenger" + suffix + "_vs_reference" + suffix] = compare_heads(x)
    write_json(
        root / "assessment.json",
        dict(
            comparisons=pairs,
            chosen=lock["chosen"],
            quality_promotion=False,
            executable=False,
            primary="challenger_raw_vs_reference_raw",
            limitation="Development-window tuning, reused dates and a partial seed/window grid; no independent holdout or profitability certification.",
        ),
    )
    seed_rows = []
    for (seed, model, head), x in data[data.window.eq("fold-07")].groupby(
        ["seed", "model", "head"]
    ):
        metric = "brier" if "probability" in head else "mse"
        means = x.groupby("variant")[metric].mean()
        seed_rows.append(
            dict(
                seed=int(seed),
                model=model,
                head=head,
                metric=metric,
                reference=float(means["reference"]),
                challenger=float(means[lock["chosen"]]),
                error_change_pct=float((means[lock["chosen"]] / means["reference"] - 1) * 100),
            )
        )
    write_json(root / "seed-comparison.json", seed_rows)
    write_json(
        root / "progress.json", dict(stage="completed", chosen=lock["chosen"], updated_at=utc_now())
    )


def worker(args):
    root, trial = args.output.resolve(), args.trial.resolve()
    verify_files(root, read(root / "frozen-manifest.json"))
    assert trial.parent == root / "trials"
    fold = read(trial / "experiment.json")["fold_indices"][0]
    if args.stage == "train":
        fit(SimpleNamespace(output=trial, fold=fold, train_only=True, inference_only=False))
    elif args.stage == "evaluate":
        lock = read(root / "selection-lock.json")
        verify_files(root, lock["screen_files"])
        assert read(trial / "experiment.json")["optimization_variant"] in [
            "reference",
            lock["chosen"],
        ]
        assert lock["no_evaluation_forecasts_present"]
        fit(SimpleNamespace(output=trial, fold=fold, train_only=False, inference_only=True))
        finalize(trial)
    elif args.stage == "verify":
        subprocess.run(
            [
                sys.executable,
                str(root / "code/scripts/verify_context_transformer.py"),
                "--output",
                str(trial),
            ],
            check=True,
            env=os.environ.copy(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "run", "train", "evaluate", "verify"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--trial", type=Path)
    args = parser.parse_args()
    if args.stage in ["freeze", "run"]:
        globals()[args.stage](args)
    else:
        worker(args)
