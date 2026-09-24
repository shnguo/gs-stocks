from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_start: str
    validation_start: str
    test_start: str
    test_end: str
    purpose: str = "development"

    def select(self, samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not self.train_start < self.validation_start < self.test_start <= self.test_end:
            raise ValueError("Fold dates must be strictly chronological")
        mature = samples.target.notna() & (samples.label_status == "available")
        train = samples.loc[mature & (samples.date >= self.train_start) &
                            (samples.date < self.validation_start) &
                            (samples.label_end < self.validation_start)].copy()
        validation = samples.loc[mature & (samples.date >= self.validation_start) &
                                 (samples.date < self.test_start) &
                                 (samples.label_end < self.test_start)].copy()
        test = samples.loc[(samples.date >= self.test_start) &
                           (samples.date <= self.test_end)].copy()
        if train.empty or validation.empty or test.empty:
            raise ValueError("Empty partition after label-maturity purge")
        return {"train": train, "validation": validation, "test": test}

    def to_dict(self) -> dict:
        return asdict(self)


def rolling_folds(dates: list[str], train_years: int = 5, validation_years: int = 1,
                  holdout_months: int = 12, refit_months: int = 3) -> dict:
    if not dates or min(train_years, validation_years, holdout_months, refit_months) <= 0:
        raise ValueError("Invalid rolling validation schedule")
    # The caller supplies only dates with mature labels, so the holdout is fully observable.
    start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    holdout_start = end - pd.DateOffset(months=holdout_months) + pd.DateOffset(days=1)
    test_start = start + pd.DateOffset(years=train_years + validation_years)
    folds = []
    while test_start < holdout_start:
        next_start = test_start + pd.DateOffset(months=refit_months)
        test_end = min(next_start, holdout_start) - pd.DateOffset(days=1)
        val_start = test_start - pd.DateOffset(years=validation_years)
        train_start = val_start - pd.DateOffset(years=train_years)
        folds.append(Fold(str(train_start.date()), str(val_start.date()), str(test_start.date()),
                          str(test_end.date())).to_dict())
        test_start = next_start
    return {"development": folds, "sealed_holdout_start": str(holdout_start.date()),
            "sealed_holdout_end": str(end.date()), "holdout_included": False}
