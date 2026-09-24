from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quant_research.return_calibration import fit_head, predict_head, rank_head, volatility_pp


def history():
    rng = np.random.default_rng(19)
    dates = pd.date_range('2024-01-01', periods=30, freq='7D')
    rows = []
    for day in dates:
        for i in range(40):
            predicted = rng.uniform(.01, .25)
            vol = rng.uniform(.5, 6)
            rows.append(dict(instrument_id=str(i), date=day.strftime('%Y-%m-%d'),
                label_end=(day.to_pydatetime()+timedelta(days=5)).strftime('%Y-%m-%d'),
                predicted=predicted, volatility_pp=vol, label_known=True,
                actual_extrema_scenario=predicted*.5+.003*vol,
                sell_offset=3, buy_reference=10., sell_reference=10*(1.0025+predicted)))
    return pd.DataFrame(rows)


def test_future_labels_and_unknown_labels_cannot_change_fit_or_ranking():
    frame = history()
    day = sorted(frame.date.unique())[23]
    head = fit_head(frame, day)
    poisoned = frame.copy()
    poisoned.loc[poisoned.label_end.ge(day), 'actual_extrema_scenario'] = 99999.
    assert fit_head(poisoned, day) == head
    current = frame[frame.date.eq(day)].copy()
    ranked = rank_head(current, head)
    current['label_known'] = False
    current['actual_extrema_scenario'] = np.nan
    unseen = rank_head(current, head)
    np.testing.assert_array_equal(ranked.instrument_id, unseen.instrument_id)
    np.testing.assert_array_equal(ranked.predicted, unseen.predicted)
    for key in ['sell_offset', 'buy_reference', 'sell_reference']:
        np.testing.assert_array_equal(ranked[key], frame.loc[ranked.index, key])
    # Even a known flag on a label ending at signal close cannot bypass the embargo.
    previous = frame[frame.label_end.lt(day)].copy()
    previous.loc[previous.date.eq(previous.date.max()), 'label_end'] = day
    assert fit_head(previous, day)['training_dates'] == head['training_dates']-1


def test_learned_correction_reduces_held_forward_error_and_roundtrips():
    import json
    frame = history()
    day = sorted(frame.date.unique())[-1]
    current = frame[frame.date.eq(day)]
    head = fit_head(frame, day)
    adjusted = predict_head(current, json.loads(json.dumps(head)))
    assert abs(adjusted-current.actual_extrema_scenario).mean() < .05*abs(current.predicted-current.actual_extrema_scenario).mean()
    with pytest.raises(ValueError, match='unavailable labels'):
        predict_head(frame, head)


def test_unknown_top20_remain_ranked_without_replacement():
    from token_return_calibration import evaluate_cross_section
    frame = history()
    day = sorted(frame.date.unique())[-1]
    head = fit_head(frame, day)
    current = frame[frame.date.eq(day)].copy()
    current['local_row'] = np.arange(len(current))
    current['seed'] = 'ensemble'
    current['arm'] = 'indicators_ranking'
    current['adverse_excursion_pct'] = 0.
    current['absolute_error'] = abs(current.predicted-current.actual_extrema_scenario)
    selected = rank_head(current, head).head(20).index
    current.loc[selected, 'label_known'] = False
    current.loc[selected, ['actual_extrema_scenario', 'absolute_error']] = np.nan
    ranks, metrics = evaluate_cross_section(current, head, 20)
    row = metrics[metrics.arm.eq('calibrated')].iloc[0]
    assert row.top_unknown == 20 and row.top_known == 0
    assert np.isnan(row.top_extrema_scenario_pct)
    assert len(ranks[ranks.arm.eq('calibrated')]) == len(current)


def test_bad_data_rejected_and_adjusted_volatility_matches_archive():
    frame = history()
    with pytest.raises(ValueError, match='Duplicate'):
        fit_head(pd.concat([frame, frame.iloc[:1]]), '2025-01-01')
    with pytest.raises(ValueError, match='finite'):
        fit_head(frame.assign(actual_extrema_scenario=np.nan), '2025-01-01')
    with pytest.raises(ValueError, match='Insufficient'):
        fit_head(frame, '2024-01-02')
    close = np.arange(21)+20.
    adj = np.linspace(1, 2, 21)
    assert volatility_pp(close, adj) == np.diff(np.log(close*adj)).std()*100
    with pytest.raises(ValueError, match='21 positive'):
        volatility_pp(close[:-1], adj[:-1])
