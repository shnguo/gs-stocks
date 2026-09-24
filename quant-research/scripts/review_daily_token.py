"""Review frozen forward watchlists once two/five-session outcomes are observable."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.daily_loop import digest, read, verify, write_manifest
from quant_research.daily_token import FIELDS, OPEN_CLOSE, panel, scenario_prices
from quant_research.forecast_audit import empirical_score
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import valid_bars


def review(store, source):
    store, source = Path(store), Path(source)
    meta, bars, _ = panel(source)
    observed = meta['price_data_through']
    groups = {s: g.set_index('date') for s, g in bars.groupby('instrument_id')}
    reports = []
    for spec in sorted((store / 'runs').glob('*/run.json')):
        run = spec.parent
        info = read(spec)
        if not info['prospective']:
            continue
        verify(run)
        publication = store / 'reference-publications' / (run.name+'.json')
        reference_ranks = None
        view_hash = None
        if publication.exists():
            pointer = read(publication)
            view = Path(pointer['report'])
            verify(view)
            delivery = read(view / 'delivery.json')
            if file_hash(view / 'manifest.json') != pointer['manifest_sha256'] or delivery['run_manifest_sha256'] != file_hash(run / 'manifest.json'):
                raise ValueError('Published ranking lineage changed')
            if not delivery['prospective']:
                continue
            view_hash = pointer['manifest_sha256']
            reference_ranks = pd.read_csv(view / 'ranking.csv').set_index('instrument_id')
        rows = pd.read_parquet(run / 'inputs/forecast-rows.parquet')
        positions = dict(zip(rows.instrument_id, range(len(rows))))
        forecast = pd.read_parquet(run / 'summary.parquet')
        ranks = pd.read_csv(run / 'ranking.csv').set_index('instrument_id')['rank'].to_dict()
        if reference_ranks is not None:
            ranks = reference_ranks['rank'].to_dict()
        models = {int(s): np.load(run / f'forecasts/seed{s}/paths.npy', mmap_mode='r') for s in info['checkpoints']}
        for h in [2, 5]:
            if info['horizon_dates'][h-1] > observed:
                continue
            source_hash = file_hash(source / 'snapshot/manifest.json')
            target = store / 'scorecards' / run.name / f'h{h}' / digest([source_hash, view_hash] if view_hash else source_hash)[:16]
            if (target / 'manifest.json').exists():
                verify(target)
                reports.append(read(target / 'scorecard.json'))
                continue
            records = []
            for prediction in forecast[forecast.horizon == h].itertuples():
                symbol = prediction.instrument_id
                if symbol not in groups:
                    continue
                dates = [info['signal_date'], *info['horizon_dates'][:h]]
                block = groups[symbol].reindex(dates)
                q = block[FIELDS].to_numpy(float)
                if not np.isfinite(q).all() or not valid_bars(q[:, :6]).all() or (q[:, 4:6] <= 0).any():
                    continue
                if block.source_trade_status.eq(0).any():
                    continue
                if not np.isclose(q[:, 6], q[0, 6], rtol=1e-8, atol=0).all():
                    continue
                if any(block[k].isna().any() or block[k].eq('').any() or block[k].nunique() != 1 for k in ['sequence_id', 'label_sequence_id']):
                    continue
                reference = q[0, 3]
                actual = np.array([(q[1:, 1].max()/reference-1)*100, (q[1:, 2].min()/reference-1)*100])
                error, crps, coverage, score = [], [], [], []
                for paths in models.values():
                    sample = paths[positions[symbol], :, :h].astype(float)
                    sample = sample[valid_bars(sample).all(-1)]
                    x = np.stack([(sample[:, :, 1].max(1)/reference-1)*100, (sample[:, :, 2].min(1)/reference-1)*100], axis=1)
                    stat = empirical_score(x, actual)
                    error.append(stat['mae'].mean())
                    crps.append(stat['crps'].mean())
                    coverage.append(stat['coverage80'].mean())
                    score.append(stat['interval_score80'].mean())
                entry, exit_price = scenario_prices(q[1:, :6], info.get('ranking_method', OPEN_CLOSE))
                realized = exit_price/entry-1-info['cost_scenario']
                expected, positive = prediction.expected_net_return, prediction.positive_fraction
                if reference_ranks is not None:
                    plan = reference_ranks.loc[symbol]
                    expected, positive = plan.expected_net_return, plan.positive_fraction
                    if plan.sell_reference_date not in dates:
                        realized = np.nan  # Price accuracy remains scored; the selected exit has not matured.
                    else:
                        buy_field, sell_field = (2, 1) if delivery['ranking_method'] == 't_low_to_future_high' else (0, 3)
                        realized = q[dates.index(plan.sell_reference_date), sell_field]/q[1, buy_field]-1-info['cost_scenario']
                records.append(dict(instrument_id=symbol, rank=ranks[symbol], expected=expected,
                    positive_fraction=positive, realized_net_scenario=realized,
                    endpoint_mae=float(np.mean(error)), persistence_mae=float(abs(actual).mean()),
                    endpoint_crps=float(np.mean(crps)), coverage80=float(np.mean(coverage)), interval_score80=float(np.mean(score)),
                    brier=(positive-float(realized > 0))**2 if np.isfinite(realized) else np.nan))
            frame = pd.DataFrame(records)
            target.mkdir(parents=True, exist_ok=True)
            frame.to_csv(target / 'rows.csv', index=False)
            summary = dict(run=run.name, horizon=h, observed_through=observed, inputs=len(forecast[forecast.horizon == h]),
                ranking_method=delivery['ranking_method'] if view_hash else info.get('ranking_method', OPEN_CLOSE), interpretation='Observed price-extrema potential on the selected dates when using reference-price ranking; not executed profit or trading win rate.',
                published_ranking_sha256=view_hash, ranking_score_method='reference_price_ratio' if view_hash else 'path_mean_return',
                known=len(frame), unknown=len(forecast[forecast.horizon == h])-len(frame), created_at=utc_now(),
                run_manifest_sha256=file_hash(run / 'manifest.json'), source_manifest_sha256=file_hash(source / 'snapshot/manifest.json'))
            if len(frame):
                matured = frame[frame.realized_net_scenario.notna()]
                top = matured[matured['rank'] <= info['top_n']]
                summary.update(endpoint_mae=float(frame.endpoint_mae.mean()), persistence_mae=float(frame.persistence_mae.mean()),
                    improvement_frequency=float((frame.endpoint_mae < frame.persistence_mae).mean()),
                    mean_error_change=float((frame.endpoint_mae-frame.persistence_mae).mean()),
                    endpoint_crps=float(frame.endpoint_crps.mean()), coverage80=float(frame.coverage80.mean()),
                    interval_score80=float(frame.interval_score80.mean()), brier=float(matured.brier.mean()) if len(matured) else None,
                    return_known=len(matured), return_pending=len(frame)-len(matured),
                    top_known=len(top), top_mean_net_scenario=float(top.realized_net_scenario.mean()) if len(top) else None,
                    top_positive_fraction=float((top.realized_net_scenario > 0).mean()) if len(top) else None,
                    pool_mean_net_scenario=float(matured.realized_net_scenario.mean()) if len(matured) else None,
                    rank_ic=float(matured.expected.rank().corr(matured.realized_net_scenario.rank())) if len(matured)>2 and matured.expected.nunique()>1 and matured.realized_net_scenario.nunique()>1 else None)
            write_json(target / 'scorecard.json', summary)
            write_manifest(target)
            reports.append(summary)
    write_json(store / 'evaluation-latest.json', dict(at=utc_now(), observed_through=observed, scorecards=reports,
        policy='Judge breadth and magnitude together; no unanimity or significance gate. These are forward scenario outcomes, not actual portfolio fills.'))
    return reports


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--source', type=Path)
    args = p.parse_args()
    source = args.source or Path(read(args.store / 'current.json')['source'])
    print(review(args.store, source))
