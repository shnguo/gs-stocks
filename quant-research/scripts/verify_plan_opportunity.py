"""Rebuild diagnostic ranks and statistics from frozen parents using pandas."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS
from quant_research.price_strategy import TradeAssumptions
from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output.resolve()
    state, protocol, parents = [read(out/n) for n in ['run-status.json', 'protocol.json', 'parents.json']]
    if state['status'] != 'completed' or (out/'verification.json').exists():
        raise ValueError('Requires completed, not previously verified diagnostic')
    counts = dict(hashes=0, plan_choices=0, groups=0, daily_values=0)
    completions=read(out/'source-completions.json') if (out/'source-completions.json').exists() else {}
    for name,spec in parents.items():
        root=Path(spec['root'])
        expected=completions.get(name,{}).get('status_sha256',spec.get('status_sha256'))
        assert expected is not None and file_hash(root/'run-status.json')==expected
        if name in completions:
            assert file_hash(root/'verification.json')==completions[name]['verification_sha256']
        if spec.get('frozen_sha256'):
            assert file_hash(root/'frozen-manifest.json')==spec['frozen_sha256']
    for root in [out, *[Path(s['root']) for s in parents.values()]]:
        for name, expected in read(root/'run-status.json')['files'].items():
            path = (root/name).resolve()
            if not path.is_relative_to(root) or file_hash(path) != expected:
                raise ValueError('Changed source or result')
            counts['hashes'] += 1
    all_daily = []
    for fold in protocol['fold_indices']:
        dest = out/f'fold-{fold:02d}'
        parent = Path(parents['lightgbm']['root'])/dest.name
        rows = pd.read_parquet(parent/'evaluation-rows.parquet')
        assert rows.label_end.max() < protocol['sealed_holdout_start']
        labels = load(parent/'evaluation.npz')
        a = TradeAssumptions()
        targets = plan_targets(labels)
        stress = plan_targets(labels, replace(a, commission=a.commission*2, minimum_fee=a.minimum_fee*2,
            sell_tax=a.sell_tax*2, slippage_bps=a.slippage_bps*2))
        for model in protocol['models']:
            name, version = model.rsplit('_', 1)
            root = Path(parents.get(name,parents['lightgbm'])['root'])
            prefix = 'learned' if name == 'lightgbm' else name
            suffix = '_raw' if version == 'raw' else ''
            pred = load(root/dest.name/f'{prefix}{suffix}-evaluation.npz')
            selected = load(dest/f'{model}-choices.npz')
            scores = pd.DataFrame(pred[HEADS[0]]*pred[HEADS[2]])
            valid = (pred[HEADS[0]] >= .3) & (pred[HEADS[1]] >= .9) & (pred[HEADS[2]] > 0)
            scores = scores.where(valid, -np.inf)
            expected_choice = scores.idxmax(axis=1).to_numpy()
            expected_choice[~valid.any(1)] = -1
            np.testing.assert_array_equal(expected_choice, selected['chosen'])
            counts['plan_choices'] += len(rows)
            ix = (np.arange(len(rows)), np.maximum(expected_choice, 0))
            table = rows[['date', 'instrument_id']].copy()
            table['eligible'] = expected_choice >= 0
            table['tie'] = table.instrument_id.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
            for h in HEADS:
                table[h] = pred[h][ix].astype(float)
            for k in ['filled', 'scenario_order_net_return', 'conditional_net_return', 'conditional_loss', 'conditional_downside', 'ambiguous']:
                table[k] = targets[k][ix]
            table['stress'] = stress['scenario_order_net_return'][ix]
            table['conditional_mean'] = table[HEADS[2]]
            # Preserve forecast-dtype multiplication for ranking; aggregate in float64.
            table['fill_times_conditional_mean'] = (pred[HEADS[0]][ix]*pred[HEADS[2]][ix]).astype(float)
            daily = pd.read_csv(dest/f'{model}-daily.csv')
            group_meta = read(dest/f'{model}-groups.json')
            expected_meta = [dict(date=date, score=score, kind=kind, value=value)
                for date in sorted(rows.date.unique()) for score in ['conditional_mean', 'fill_times_conditional_mean']
                for kind, values in [('top_fraction', protocol['fractions']), ('quintile_best_first', list(range(1, 6)))] for value in values]
            assert group_meta == expected_meta and len(daily) == len(group_meta)
            for i, g in enumerate(group_meta):
                candidates = table.loc[table.date.eq(g['date']) & table.eligible]
                ordered = candidates.sort_values([g['score'], 'tie'], ascending=[False, True])
                if g['kind'] == 'top_fraction':
                    expected_ids = ordered.index[:int(np.ceil(len(ordered)*g['value']))].to_numpy()
                else:
                    bin_number = np.floor(np.arange(len(ordered))*5/max(len(ordered), 1)).astype(int)+1
                    expected_ids = ordered.index[bin_number == g['value']].to_numpy()
                np.testing.assert_array_equal(selected[f'group_{i}'], expected_ids)
                s = table.loc[expected_ids]
                known = s.conditional_net_return.notna()
                expected = dict(cohort_stocks=int(table.date.eq(g['date']).sum()), eligible_stocks=len(candidates), selected=len(s),
                    fill_known=s.filled.count(), filled=s.filled.eq(1).sum(), return_known=s.scenario_order_net_return.count(),
                    return_unknown=s.scenario_order_net_return.isna().sum(), stress_return_known=s.stress.count(),
                    stress_return_unknown=s.stress.isna().sum(), ambiguous=s.ambiguous.sum(), conditional_known=known.sum(),
                    predicted_conditional_mean=s[HEADS[2]].mean(), predicted_conditional_mean_on_known=s.loc[known, HEADS[2]].mean(),
                    observed_conditional_mean=s.conditional_net_return.mean(), known_order_scenario_mean=s.scenario_order_net_return.mean(),
                    stress_known_order_scenario_mean=s.stress.mean(), observed_conditional_loss=s.conditional_loss.mean(),
                    predicted_loss_on_known=s.loc[known, HEADS[3]].mean(), observed_conditional_downside=s.conditional_downside.mean(),
                    conditional_loss_brier=((s[HEADS[3]]-s.conditional_loss)**2).mean(), observed_fill_rate=s.filled.mean(),
                    predicted_fill_probability=s[HEADS[0]].mean(), score_distinct_values=s[g['score']].nunique())
                record = daily.iloc[i]
                assert record.window == dest.name and record.model == model and record.date == g['date']
                assert record.score == g['score'] and record.group_kind == g['kind'] and record.group_value == g['value']
                for key, value in expected.items():
                    np.testing.assert_allclose(record[key], value, rtol=1e-10, atol=1e-12, equal_nan=True)
                    counts['daily_values'] += 1
                counts['groups'] += 1
            all_daily.append(daily)
    combined = pd.concat(all_daily, ignore_index=True)
    pd.testing.assert_frame_equal(combined, pd.read_csv(out/'daily.csv'))
    keys = ['model', 'score', 'group_kind', 'group_value']
    metrics = [c for c in combined.select_dtypes(include='number').columns if c != 'group_value']
    win = combined.groupby(['window', *keys])[metrics]
    for name, expected in [('window-means', win.mean()), ('window-known-date-counts', win.count()),
            ('overall-window-equal-means', win.mean().groupby(keys).mean()),
            ('overall-known-window-counts', win.mean().groupby(keys).count())]:
        pd.testing.assert_frame_equal(expected.reset_index(), pd.read_csv(out/f'{name}.csv'), check_exact=False, rtol=1e-10, atol=1e-12)
    write_json(out/'verification.json', dict(passed=True, checked_at=utc_now(), counts=counts,
        scope='All plan choices, prediction-only group membership, daily numeric statistics and equal-window aggregation; parent verified scenario labels reused; no strategy promotion'))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
