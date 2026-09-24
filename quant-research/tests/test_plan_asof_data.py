import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_asof_data import AsOfData


@pytest.fixture
def source(tmp_path):
    root = tmp_path/'source'
    (root/'panel-h5').mkdir(parents=True)
    (root/'snapshot').mkdir()
    dates = pd.bdate_range('2024-01-01',periods=80).strftime('%Y-%m-%d').tolist()
    (root/'panel-h5/manifest.json').write_text(json.dumps(dict(dataset_id='fixture',dates=dates)))
    np.save(root/'panel-h5/values.npy',np.ones((2,80,3),np.float32))
    rows,bars = [],[]
    for si,symbol in enumerate(['cn.xshg.600000','cn.xbse.430001']):
        for t,day in enumerate(dates):
            rows.append(dict(date=day,instrument_id=symbol,stock_index=si,date_index=t,
                risk_status='known',trading_eligible=True,source_is_st=False,forward_return=999.))
            bars.append(dict(date=day,instrument_id=symbol,open=10.,high=11.,low=9.,close=10.,
                volume=100.,amount=1000.,factor=1.,upper_limit=np.nan,lower_limit=np.nan,
                sequence_id='one',label_sequence_id='one',source_trade_status=np.nan))
    pd.DataFrame(rows).to_parquet(root/'panel-h5/samples.parquet')
    pd.DataFrame(bars).to_parquet(root/'snapshot/bars.parquet')
    pd.DataFrame(dict(instrument_id=['cn.xshg.600000'],ex_date=[dates[79]])).to_parquet(root/'snapshot/actions.parquet')
    return root,dates


def test_no_label_read_before_maturity_and_no_future_sample_columns(source,tmp_path,monkeypatch):
    root,dates = source
    store = AsOfData(root,tmp_path/'cache','2025-01-01')
    calls=[]
    original=pd.read_parquet
    def tracked(path,**kwargs):
        calls.append((Path(path),kwargs))
        return original(path,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',tracked)
    with pytest.raises(ValueError,match='not matured'):
        store.labels(dates[60],dates[64])
    assert calls == []
    rows,data = store.inputs(dates[60],dates[60])
    assert len(rows)==2 and data['x'].shape==(2,3)
    for path,kwargs in calls:
        if path.name=='samples.parquet':
            assert 'forward_return' not in kwargs['columns']
        if path.name=='bars.parquet':
            assert kwargs['filters']==[('date','==',dates[60])]
    calls.clear()
    _,labels=store.labels(dates[60],dates[65])
    assert labels['valid'].all()  # Optional missing trade status must retain BSE.
    for path,kwargs in calls:
        if path.name in ['bars.parquet','actions.parquet']:
            assert max(kwargs['filters'][0][2])<=dates[65]


def test_later_events_and_cache_tampering(source,tmp_path):
    root,dates=source
    store=AsOfData(root,tmp_path/'cache','2025-01-01')
    _,before=store.labels(dates[60],dates[65])
    events=pd.read_parquet(root/'snapshot/actions.parquet')
    events.loc[0,'ex_date']=dates[70]
    events.to_parquet(root/'snapshot/actions.parquet')
    other=AsOfData(root,tmp_path/'other','2025-01-01')
    _,after=other.labels(dates[60],dates[65])
    for k in before:
        np.testing.assert_array_equal(before[k],after[k])
    with (tmp_path/'cache'/dates[60]/'labels.npz').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError,match='Changed cached'):
        store.labels(dates[60],dates[65])
    with pytest.raises(ValueError,match='Input date'):
        store.inputs(dates[61],dates[60])


def test_calibration_consumes_only_mature_earlier_predictions(source,tmp_path):
    from plan_prequential_run import calibrate_at

    from quant_research.plan_targets import plan_targets
    from quant_research.plan_value import HEADS
    from quant_research.storage import file_hash, write_json

    root,dates=source
    bars=pd.read_parquet(root/'snapshot/bars.parquet')
    bars['open'],bars['high'],bars['low']=9.7,10.1,9.6
    bars.to_parquet(root/'snapshot/bars.parquet')
    out=tmp_path/'experiment'
    store=AsOfData(root,out/'day-cache','2025-01-01')
    day=dates[65]
    past=[dates[i] for i in range(5,65,5)]
    for date in [*past,day]:
        dest=out/'fold-01'/'fits'/date
        dest.mkdir(parents=True)
        rows,_=store.inputs(date,date)
        rows.to_parquet(dest/'prediction-rows.parquet',index=False)
        np.savez_compressed(dest/'raw.npz',**{h:np.full((len(rows),12),v)
            for h,v in zip(HEADS,[.7,.9,.015,.45,.01])})
        write_json(dest/'config.json',dict(asof=date))
        write_json(dest/'raw-completed.json',dict(status='completed',files={n:file_hash(dest/n)
            for n in ['prediction-rows.parquet','raw.npz','config.json']}))
    spec=dict(calibration_prediction_dates=past,calibration_label_ends=[dates[store.axis[d]+5] for d in past])
    assert not (store.cache/day/'label-manifest.json').exists()
    calibrate_at(out,'fold-01',day,spec,store)
    dest=out/'fold-01'/'fits'/day
    with np.load(dest/'calibrated.npz') as z:
        corrected=z[HEADS[2]]
    _,labels=store.labels(past[-1],day)
    np.testing.assert_allclose(corrected,plan_targets(labels)['conditional_net_return'])
    assert not (store.cache/day/'label-manifest.json').exists()
    receipt=json.loads((dest/'calibration-receipts.json').read_text())
    assert set(receipt['earlier_raw_completions'])==set(past)
    assert receipt['maximum_label_end']==day
