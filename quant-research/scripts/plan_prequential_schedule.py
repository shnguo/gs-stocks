"""Register historical-as-of fits and calibration dependencies, without training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.storage import file_hash, utc_now, write_json


def make_schedule(calendar, evaluations, training_start, sealed_start):
    axis = {d:i for i,d in enumerate(calendar)}
    fits, predictions = {}, []
    for day in evaluations:
        t = axis[day]
        if t+5 >= len(calendar) or calendar[t+5] >= sealed_start:
            raise ValueError('Evaluation horizon unavailable or sealed')
        if t < 60:
            raise ValueError('Insufficient past calibration dates')
        calibration = [calendar[t-lag] for lag in range(60, 0, -5)]
        for asof in [*calibration, day]:
            if asof in fits:
                continue
            current = axis[asof]
            if current < 20:
                raise ValueError('Insufficient selection history')
            selection = [calendar[current-lag] for lag in [20,15,10,5]]
            lower = max(training_start, str((pd.Timestamp(selection[0])-pd.DateOffset(years=1)).date()))
            eligible = [d for d in calendar if lower <= d < selection[0] and 59 <= axis[d]
                and axis[d]+5 < axis[selection[0]]]
            if len(eligible) < 96:
                raise ValueError('Insufficient fixed-density training dates')
            train = [eligible[i] for i in np.linspace(0,len(eligible)-1,96,dtype=int)]
            assert len(set(train)) == 96
            assert calendar[axis[train[-1]]+5] < selection[0]
            assert calendar[axis[selection[-1]]+5] <= asof
            fits[asof] = dict(asof_signal_close=asof, train=train, selection=selection,
                train_label_end=calendar[axis[train[-1]]+5], selection_label_end=calendar[axis[selection[-1]]+5],
                forecast_label_end=calendar[current+5], training_label_age_sessions=current-axis[train[-1]]-5)
        predictions.append(dict(date=day, fit_asof=day, calibration_prediction_dates=calibration,
            calibration_label_ends=[calendar[axis[c]+5] for c in calibration],
            evaluation_label_end=calendar[t+5]))
        assert all(axis[c]+5 <= t for c in calibration)
    return dict(fits={d:fits[d] for d in sorted(fits)}, predictions=predictions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    prior,out = a.prior.resolve(),a.output.resolve()
    protocol = json.loads((prior/'protocol.json').read_text())
    schedules = {}
    for f in protocol['fold_indices']:
        cfg = json.loads((prior/f'fold-{f:02d}/config.json').read_text())
        manifest = Path(cfg['root'])/'panel-h5/manifest.json'
        calendar = json.loads(manifest.read_text())['dates']
        schedules[f'fold-{f:02d}'] = make_schedule(calendar,cfg['dates']['evaluation'],cfg['fold']['train_start'],protocol['sealed_holdout_start'])
    out.mkdir()
    write_json(out/'protocol.json',dict(protocol_id='plan-prequential-schedule-v1',registered_at=utc_now(),
        parent=str(prior),parent_status_sha256=file_hash(prior/'run-status.json'),source_sha256=file_hash(Path(__file__)),
        train_dates=96,selection_dates=4,calibration_predictions=12,calibration_stride_sessions=5,
        decision_asof='After signal-date close, before next-session T opening; complete OHLC labels ending on signal date are available under this explicit daily-data assumption',
        separation='Train labels strictly before selection signals; selection labels through model as-of; calibration uses stored predictions made at each earlier as-of with labels matured by current as-of',
        comparability='Keep existing six windows, eight evaluation dates, 27 features, 12 plans, head definitions, costs, seed and tree parameters; shorter selection interval is part of the pipeline intervention',
        budget='Keep 60-tree cap for the first controlled comparison and report best-at-cap models; do not claim all capped models converged',
        calibration='Report raw forecasts and past-prediction offset/sigmoid calibration separately; no fits on future evaluation labels',
        retention='Future labels cannot affect previously produced forecasts; freeze each as-of model and prediction before later labels are loaded',
        status='schedule_only_not_trained',quality_promotion=False,executable=False,
        total_fits=sum(len(s['fits']) for s in schedules.values()),sealed_holdout_start=protocol['sealed_holdout_start']))
    write_json(out/'schedule.json',schedules)
    print('Registered',sum(len(s['fits']) for s in schedules.values()),'historical-as-of fits for',
        sum(len(s['predictions']) for s in schedules.values()),'unchanged evaluation dates')


if __name__ == '__main__':
    main()
