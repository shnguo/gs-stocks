import numpy as np
import pandas as pd
import pytest
from token_ranking_run import choose_shadow, daily_metrics, date_batches
from token_ranking_shadow import is_prospective, review_shadow

from quant_research.daily_loop import write_manifest
from quant_research.storage import write_json


def test_grouped_batches_preserve_rows_and_dates_reproducibly():
    rows = pd.DataFrame({'date': np.repeat(['a','b','c','d','e'], 7)})
    a = list(date_batches(rows, 17, 0, 2))
    b = list(date_batches(rows, 17, 0, 2))
    np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(len(rows)))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
        assert rows.iloc[x].date.nunique() <= 2
    for day, group in rows.groupby('date'):
        assert sum(set(group.index).issubset(set(batch)) for batch in a) == 1


def test_top20_is_fixed_count_and_return_ranking_not_realized_sort():
    frame = pd.DataFrame(dict(arm='a', seed='17', date='d', instrument_id=[f'cn.xshe.{i:06}' for i in range(100)],
        predicted=np.arange(100), actual_extrema_scenario=-np.arange(100)/100,
        absolute_error=1., volatility_pp=np.arange(100), adverse_excursion_pct=-1.))
    result = daily_metrics(frame).iloc[0]
    assert result.top_rows == 20
    assert result.top_extrema_scenario_pct == pytest.approx(-89.5)
    assert result.top_lift_pp == pytest.approx(-40.)
    assert result.top_high_volatility_fraction == 1


def test_nomination_only_uses_selection_ensemble_and_excludes_baseline():
    metrics = pd.DataFrame(dict(arm=['baseline','indicators','indicators_ranking','indicators_ranking'],
        seed=['ensemble','ensemble','ensemble','17'], top_lift_pp=[100,2,1,200]))
    assert choose_shadow(metrics, {}) == 'indicators'


def test_shadow_time_boundary_is_auction_and_requires_timely_baseline():
    assert is_prospective('2026-09-16T01:14:59Z', '2026-09-16', True)
    assert not is_prospective('2026-09-16T01:15:00Z', '2026-09-16', True)
    assert not is_prospective('2026-09-16T00:00:00Z', '2026-09-16', False)
    with pytest.raises(ValueError, match='Timezone-aware'):
        is_prospective('2026-09-16', '2026-09-16', True)


@pytest.mark.parametrize("control", ["current", "baseline"])
def test_shadow_review_preserves_exit_date_top20_and_excludes_late(tmp_path, control):
    dates = ['2026-09-16','2026-09-17','2026-09-18','2026-09-21','2026-09-22','2026-09-23']
    source, store = tmp_path/'source', tmp_path/'store'
    snapshot = source/'snapshot'
    snapshot.mkdir(parents=True)
    # The best actual high occurs later than the frozen selected day.
    bars = pd.DataFrame(dict(instrument_id='cn.xshe.000001', date=dates, open=10.,close=10.,
        high=[11.,11.,12.,13.,14.,20.], low=9.,volume=100.,amount=1000.,factor=1.,
        sequence_id='a',label_sequence_id='b',source_trade_status=1))
    bars.to_parquet(snapshot/'bars.parquet', index=False)
    pd.DataFrame(dict(instrument_id=['cn.xshe.000001'],name=['A'],current_member=[True])).to_parquet(snapshot/'instruments.parquet', index=False)
    write_manifest(snapshot)
    write_json(source/'panel-h5/manifest.json',dict(price_data_through=dates[-1],dates=dates))
    info = dict(signal_date=dates[0],horizon_dates=dates[1:],prospective=True,cost=.0025,top_n=20)
    if control == 'baseline':
        info['ranking_files'] = {'baseline': 'baseline-ranking.csv', 'candidate': 'indicators_ranking-ranking.csv'}
    files = info.get('ranking_files', {'current': 'current-ranking.csv', 'candidate': 'candidate-ranking.csv'})
    for suffix, timely in [('timely',True),('late',False)]:
        run = store/'runs'/suffix
        run.mkdir(parents=True)
        write_json(run/'publication.json',dict(info,prospective=timely))
        for arm, filename in files.items():
            pd.DataFrame(dict(instrument_id=['cn.xshe.000001','cn.xshe.000002'],rank=[1,2],
                expected_net_return=[.1,.2],sell_reference_date=[dates[2],dates[2]])).to_csv(run/filename,index=False)
        write_manifest(run)
    result=review_shadow(store,source)
    assert result[0]['status']=='late_excluded'
    summary=result[1]['metrics'][0]
    assert summary['arm'] == control
    assert summary['known']==1 and summary['unknown']==1 and summary['top_known']==1
    assert summary['top_extrema_scenario_pct']==pytest.approx((12/9-1-.0025)*100)
    assert review_shadow(store,source)==result
    # A future-dated publication stays pending; no premature outcome scoring.
    run=store/'runs'/'pending'
    run.mkdir()
    write_json(run/'publication.json',dict(info,horizon_dates=['2099-01-01']*5))
    write_manifest(run)
    assert any(r['status']=='pending_labels' for r in review_shadow(store,source))


def test_unknown_future_cannot_replace_a_top20_stock():
    from finalize_token_ranking import freeze_and_score
    frame=pd.DataFrame(dict(arm='a',seed='17',date='d',instrument_id=[f'cn.xshe.{i:06}' for i in range(30)],
        predicted=np.arange(30),actual_extrema_scenario=.1,absolute_error=.01,
        volatility_pp=1.,adverse_excursion_pct=-1.,label_known=True))
    frame.loc[29,['actual_extrema_scenario','absolute_error']]=np.nan
    frame.loc[29,'label_known']=False
    ranks,metrics=freeze_and_score(frame)
    top=ranks[ranks['rank']<=20]
    assert set(top.instrument_id)==set(frame.iloc[10:].instrument_id)
    assert top.iloc[0].instrument_id=='cn.xshe.000029'
    assert metrics.iloc[0].top_known==19 and metrics.iloc[0].top_unknown==1
    changed=frame.copy()
    changed['label_known']=False
    changed['actual_extrema_scenario']=np.nan
    other,_=freeze_and_score(changed)
    pd.testing.assert_series_equal(ranks.set_index('instrument_id')['rank'],other.set_index('instrument_id')['rank'])


def test_shadow_reconstruction_matches_daily_float32_normalization_layout(tmp_path, monkeypatch):
    import token_ranking_shadow as shadow

    from quant_research.daily_token import FIELDS
    from quant_research.storage import file_hash
    from quant_research.token_indicators import indicator_features
    from quant_research.token_transformer import normalized_history
    dates=pd.bdate_range('2026-01-01',periods=60).strftime('%Y-%m-%d').tolist()
    symbols=['cn.xshe.000001','cn.xshg.600001']
    rng=np.random.default_rng(27)
    raw=np.ones((2,60,7),float)
    close=np.exp(rng.normal(0,.02,(2,60)).cumsum(1))*13
    raw[...,0]=raw[...,3]=close
    raw[...,1]=close*1.03
    raw[...,2]=close*.97
    raw[...,4]=rng.uniform(1e6,1e8,(2,60))
    raw[...,5]=raw[...,4]*close
    bars=pd.concat([pd.DataFrame(raw[i],columns=FIELDS).assign(instrument_id=s,date=dates) for i,s in enumerate(symbols)],ignore_index=True)
    meta=dict(dates=dates,price_data_through=dates[-1])
    source=tmp_path/'source'
    write_json(source/'snapshot/manifest.json',{'test':True})
    run=tmp_path/'run'
    write_json(run/'inputs/input.json',dict(source=str(source),signal_date=dates[-1],snapshot_sha256=file_hash(source/'snapshot/manifest.json')))
    monkeypatch.setattr(shadow,'panel',lambda source:(meta,bars,None))
    _,mean,scale=normalized_history(raw.copy(order='C'))
    features,actual_source=shadow.prepare_shadow_features(run,{'mean':mean,'scale':scale},pd.DataFrame(dict(instrument_id=symbols)))
    np.testing.assert_array_equal(features,indicator_features(raw))
    assert actual_source==source
