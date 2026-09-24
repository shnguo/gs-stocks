"""Independent numerical and lineage checks for fixed-weight temporal transfer."""
from pathlib import Path

import numpy as np
import pandas as pd
import tokenizer_forecast_sensitivity as sensitivity
import torch
import verify_tokenizer_sensitivity as shared_check
from token_history_run import DATA, Dataset
from tokenizer_temporal_run import PRIOR, ROOT, read
from verify_token_pipeline_audit import compare_group, verify_aggregation
from verify_tokenizer_reconstruction import verify_reconstruction

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import restore_model


def verify_contrasts(cfg):
    source=pd.concat([pd.read_csv(ROOT/f'predictors/seed{s}/dense_3720k_equal/decoder-forecast/daily.csv').assign(seed=s)
        for s in cfg['forecast_seeds']],ignore_index=True)
    expected=pd.read_csv(ROOT/'forecast-results/target-paired.csv',dtype={'seed':str})
    for row in expected.itertuples(index=False):
        frame=source[source.horizon==row.horizon]
        if row.target!='mean':
            frame=frame[frame.target==row.target]
        if row.seed!='mean':
            frame=frame[frame.seed==int(row.seed)]
        dates=sorted(set(frame.date))
        b,x=[],[]
        for day in dates:
            p=frame[frame.date==day]
            b.append(p.loc[p.variant=='frozen',row.metric].mean())
            x.append(p.loc[p.variant=='adapted',row.metric].mean())
        b,x=np.asarray(b),np.asarray(x)
        np.testing.assert_allclose([row.baseline,row.adapted,row.delta,row.relative_change],
            [b.mean(),x.mean(),(x-b).mean(),x.mean()/b.mean()-1],atol=1e-10,rtol=1e-10)
        rng=np.random.default_rng(314159)
        starts=rng.integers(len(dates),size=(2000,int(np.ceil(len(dates)/10))))
        samples=[]
        for blocks in starts:
            indices=[(int(start)+i)%len(dates) for start in blocks for i in range(10)][:len(dates)]
            samples.append(float(sum((x-b)[indices])/len(dates)))
        np.testing.assert_allclose([row.ci_low,row.ci_high],np.quantile(samples,[.025,.975]),atol=1e-10,rtol=1e-10)
    primary=pd.read_csv(ROOT/'forecast-results/paired.csv',dtype={'seed':str})
    mean=expected[expected.target=='mean'].drop(columns='target')
    keys=['seed','horizon','metric']
    pd.testing.assert_frame_equal(primary.sort_values(keys).reset_index(drop=True),mean.sort_values(keys).reset_index(drop=True))
    return len(expected)


def main():
    assert read(ROOT/'run-completed.json')['passed']
    torch.set_num_threads(4)
    cfg=read(ROOT/'protocol.json')
    assert file_hash(ROOT/'decoder-selection/selection.json')==file_hash(PRIOR/'decoder-selection/selection.json')
    data_manifest=read(DATA/'completed.json')
    for name in ['protocol.json','rows.parquet','calendar-stamps.npy','s1.npy','s2.npy','future.npy','valid.npy','mean.npy','scale.npy','last.npy']:
        assert file_hash(DATA/name)==data_manifest['files'][name],name
    data=Dataset()
    counts={}
    for variant in ['frozen','adapted']:
        counts['reconstruction-'+variant]=verify_reconstruction(ROOT/'reconstruction-evaluation'/variant,data)
    eval_ids=np.load(ROOT/'evaluation-ids.npy')
    prior_ids=np.load(PRIOR/'evaluation-ids.npy')
    assert not set(eval_ids).intersection(prior_ids)
    r=data.rows.iloc[eval_ids]
    assert r.date.min()>=cfg['fold']['evaluation'][0]
    assert r.label_end.max()<cfg['fold']['evaluation'][1]<cfg['sealed_holdout_start']
    for part in ['train','selection']:
        np.testing.assert_array_equal(np.load(ROOT/f'{part}-ids.npy'),np.load(PRIOR/f'{part}-ids.npy'))
    for seed in cfg['forecast_seeds']:
        root=ROOT/f'predictors/seed{seed}/dense_3720k_equal'
        old=PRIOR/f'predictors/seed{seed}/dense_3720k_equal'
        assert file_hash(root/'training/best.pt')==file_hash(old/'training/best.pt')
        seen=np.load(root/'training/visited-row-ids.npy')
        assert set(seen).issubset(set(np.load(ROOT/'train-ids.npy')))
        assert data.a['valid'][seen].any(1).all()
        folder=root/'decoder-forecast'
        ids=np.load(folder/'row-ids.npy')
        assert set(ids).issubset(set(eval_ids))
        expected=pd.read_parquet(folder/'common.parquet')
        mapping={name:np.load(folder/f'{name}-paths.npy',mmap_mode='r') for name in ['frozen','adapted']}
        counts[f'forecast-{seed}']=compare_group(mapping,ids,expected,data.rows,data.a['future'],data.a['valid'],data.a['last'])
        verify_aggregation(expected,pd.read_csv(folder/'daily.csv'),pd.read_csv(folder/'metrics.csv'))
        model,_=restore_model(root/'training/best.pt','cpu')
        model.eval()
        with np.load(root/'training/reload-reference.npz') as z,torch.no_grad():
            a,b,stamps,_=data.tensors(z['ids'],'cpu')
            logits=model.forecast_logits(a[:,:-1],b[:,:-1],stamps[:,:-1],a[:,1:],59)
            np.testing.assert_array_equal(logits[0].numpy(),z['coarse'])
            np.testing.assert_array_equal(logits[1].numpy(),z['fine'])
        proof=read(folder/'verification.json')
        assert proof['first_batch_replay_exact'] and proof['shared_generated_tokens']
        assert proof['selection_sha256']==file_hash(ROOT/'decoder-selection/selection.json')
        assert proof['predictor_sha256']==file_hash(root/'training/best.pt')
        assert proof['decoder_sha256']==file_hash(Path(read(ROOT/'decoder-selection/selection.json')['checkpoint']))
        for name,h in proof['chunks'].items():
            assert file_hash(folder/'chunks'/name)==h
    counts['target_contrasts_and_intervals']=verify_contrasts(cfg)
    sensitivity.ROOT=ROOT
    sensitivity.main()
    shared_check.ROOT=ROOT
    shared_check.main()
    for name,h in read(ROOT/'source-manifest.json')['prior_files'].items():
        assert file_hash(Path(name))==h
    for name,h in read(ROOT/'source-manifest.json')['files'].items():
        assert file_hash(ROOT/name)==h
    files={}
    for done in ROOT.rglob('completed.json'):
        obj=read(done)
        assert obj['passed']
        for name,h in obj['files'].items():
            assert file_hash(done.parent/name)==h
        files[str(done)]=file_hash(done)
    write_json(ROOT/'independent-verification.json',dict(passed=True,at=utc_now(),counts=counts,files=files,
        original_checkpoints_unchanged=True,prior_training_selection_unchanged=True,previous_evaluation_disjoint=True,
        cpu_reload_logits_exact=True,common_draw_sensitivity_verified=True,sealed_holdout_opened=False))
    print('Temporal transfer verification passed',counts,flush=True)


if __name__=='__main__':
    main()
