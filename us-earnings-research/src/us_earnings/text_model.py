"""A deliberately simple, local NLP baseline without historical LLM memorization.

The teacher is a lexical rule, not human truth. Evaluation of trading performance
does not validate extraction accuracy. All fitting uses only pre-2023 filings.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def guidance_context(text: str) -> str:
    spans = []
    for match in re.finditer(r"\b(?:guidance|outlook|forecast)\b", text, re.I):
        start, end = max(0, match.start() - 220), min(len(text), match.end() + 300)
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
        else:
            spans.append((start, end))
        if len(spans) >= 8:
            break
    context = " ".join(text[start:end] for start, end in spans)[:9000]
    # Do not train on identifiers, exact years or dollar amounts.
    return re.sub(r"\d+(?:[.,]\d+)*", " NUMBER ", context)


def fit_text_model(events: pd.DataFrame, data_dir: Path, settings: dict):
    known = events.loc[events.status == "ok"].copy()
    known["context"] = [(guidance_context((data_dir / path).read_text())) for path in known.text_file]
    training = known.filingDate <= settings["training_end"]
    train = known.loc[training]
    if train.raised_guidance.nunique() != 2 or int(train.raised_guidance.sum()) < 5:
        raise ValueError("Insufficient positive training examples for text baseline")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=settings["min_df"],
                                 max_features=settings["max_features"], sublinear_tf=True)),
        ("classifier", LogisticRegression(C=settings["C"], class_weight="balanced",
                                         solver="liblinear", random_state=settings["seed"], max_iter=1000)),
    ])
    model.fit(train.context, train.raised_guidance.astype(int))
    # Generate only genuinely later document predictions; no in-sample predictions.
    later = known.loc[known.filingDate >= settings["evaluation_start"]].copy()
    later["ml_score"] = model.predict_proba(later.context)[:, 1]
    later["ml_raised"] = (later.ml_score >= settings["threshold"]) & (later.context.str.len() > 0)
    later = later.drop(columns="context")
    input_hash = hashlib.sha256("\n".join(train.accessionNumber + ":" + train.source_sha256).encode()).hexdigest()
    metadata = {"settings": settings, "training_rows": len(train),
                "positive_training_rows": int(train.raised_guidance.sum()),
                "training_last_date": train.filingDate.max(), "training_inputs_sha256": input_hash,
                "later_predictions": len(later), "later_positives": int(later.ml_raised.sum()),
                "uses_returns_for_fitting": False,
                "feature_count": len(model.named_steps["tfidf"].vocabulary_)}
    return later, metadata, model


def add_model_signals(signals: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    positive = set(predictions.loc[predictions.ml_raised, "accessionNumber"])
    result = signals.copy()
    result["ml_guidance"] = [any(a in positive for a in accessions.split("|"))
                             for accessions in result.accession]
    return result


def export_model_audit(predictions: pd.DataFrame, metadata: dict, model, out: Path):
    import joblib

    predictions.to_csv(out / "text-model-predictions.csv", index=False)
    (out / "text-model-metadata.json").write_text(json.dumps(metadata, indent=2))
    joblib.dump(model, out / "text-model.joblib")
    coefficients = model.named_steps["classifier"].coef_[0]
    names = model.named_steps["tfidf"].get_feature_names_out()
    selected = np.r_[np.argsort(coefficients)[:15], np.argsort(coefficients)[-15:]]
    pd.DataFrame({"term": names[selected], "weight": coefficients[selected]}).to_csv(
        out / "text-model-features.csv", index=False)
