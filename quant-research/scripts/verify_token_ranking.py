"""Independent artifact audit for the bounded ranking experiment (no training)."""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_features_run import read, verify_manifest
from token_history_run import Dataset
from tokenizer_reconstruction_run import load_decoder

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_cached_inference import decode_cached, generate_cached
from quant_research.token_indicators import FEATURES, indicator_features
from quant_research.token_transformer import (
    decode_paths,
    generate_tokens,
    restore_model,
    valid_bars,
)

BASE = Path(__file__).resolve().parents[1]


def independent_stats(paths, minimum):
    # Separate vectorized implementation; no reference_prices/ranking_rows call.
    x = np.asarray(paths, dtype=np.float64)
    valid = valid_bars(x).all(-1)
    count = valid.sum(1)
    peak = x[:, :, 1:, 1].argmax(2)
    votes = np.column_stack([(valid & (peak == i)).sum(1)/count.clip(1) for i in range(4)])
    buy = np.nanmedian(np.where(valid, x[:, :, 0, 2], np.nan), axis=1)
    sell = np.nanmedian(np.where(valid[:, :, None], x[:, :, 1:, 1], np.nan), axis=1)
    return count >= minimum, votes, buy, sell


def verify(root, replay=True):
    cfg = read(root/'protocol.json')
    verify_manifest(root, 'completed.json')
    for p, expected in read(root/'sources.json').items():
        assert file_hash(Path(p)) == expected, p
    for p, expected in read(root/'live-state.json').items():
        assert file_hash(Path(p)) == expected, p
    data = Dataset()
    counts, decoded_error, tokens_equal, replay_rows = 0, 0., 0, 0
    source = np.load(BASE/cfg['price_input']/'values.npy', mmap_mode='r')
    for part in ['training','selection','evaluation']:
        ids = np.load(root/f'{part}-ids.npy')
        rows = data.rows.iloc[ids]
        assert not rows.duplicated(['instrument_id','date']).any()
        assert rows.date.min() >= cfg[part][0] and rows.label_end.max() < cfg[part][1]
        saved = np.load(root/f'{part}-features.npy',mmap_mode='r')
        positions = np.random.default_rng(2718).choice(len(ids),min(128,len(ids)),replace=False)
        selected = ids[positions]
        ss, tt = data.rows.iloc[selected][['stock_index','date_index']].to_numpy(int).T
        raw = source[ss[:,None],tt[:,None]+np.arange(-59,1)].copy()
        np.testing.assert_array_equal(indicator_features(raw), saved[positions])
        modified = raw.copy()
        modified[:,40:,:4] *= 2
        np.testing.assert_array_equal(indicator_features(raw)[:,:40],indicator_features(modified)[:,:40])
        if part == 'training':
            prior = np.load(BASE/cfg['cohort_source']/f'{part}-ids.npy')
            np.testing.assert_array_equal(ids,prior)
        else:
            prior_ids=np.load(BASE/cfg['cohort_source']/f'{part}-ids.npy')
            assert set(rows.date)==set(data.rows.iloc[prior_ids].date)
            eligible=data.rows[data.rows.date.isin(rows.date.unique())].groupby('date').head(cfg['evaluation_per_date']).row_id.to_numpy()
            np.testing.assert_array_equal(ids,eligible)
    with np.load(root/'normalizer.npz') as norm:
        training=np.load(root/'training-features.npy',mmap_mode='r')
        # Independent exact fit definition used by feature branch.
        for j in range(len(FEATURES)):
            values=np.asarray(training[:,:,j],float)
            values=values[np.isfinite(values)]
            np.testing.assert_allclose(norm['center'][j],np.mean(values),rtol=1e-6,atol=1e-7)
            expected_scale=values.std() if values.std()>1e-6 else 1.
            np.testing.assert_allclose(norm['scale'][j],expected_scale,rtol=1e-6,atol=1e-7)
    for part in ['selection','evaluation']:
        ids=np.load(root/f'{part}-ids.npy')
        rows=data.rows.iloc[ids].reset_index(drop=True)
        actual=pd.read_csv(root/f'{part}-all-ranking.csv',dtype={'seed':str})
        for arm in cfg['variants']:
            arrays={str(s):np.load(root/f'seed{s}'/arm/part/'paths.npy',mmap_mode='r') for s in cfg['seeds']}
            for start in range(0,len(ids),256):
                end=min(start+256,len(ids))
                local=np.arange(start,end)
                stats={s:independent_stats(a[start:end],cfg['minimum_legal_paths']) for s,a in arrays.items()}
                for seed in [str(s) for s in cfg['seeds']]+['ensemble']:
                    used=list(stats.values()) if seed=='ensemble' else [stats[seed]]
                    okay=np.logical_and.reduce([x[0] for x in used]) & data.a['valid'][ids[local]].all(1)
                    day=np.mean([x[1] for x in used],axis=0).argmax(1)+1
                    buy=np.mean([x[2] for x in used],axis=0)
                    sell=np.mean([x[3] for x in used],axis=0)[np.arange(end-start),day-1]
                    pred=sell/buy-1-cfg['cost']
                    future=data.a['future'][ids[local]].astype(float)
                    realized=future[np.arange(end-start),day,1]/future[:,0,2]-1-cfg['cost']
                    found=actual[(actual.arm==arm)&(actual.seed==seed)&actual.local_row.ge(start)&actual.local_row.lt(end)].sort_values('local_row')
                    np.testing.assert_array_equal(found.local_row,local[okay])
                    np.testing.assert_array_equal(found.sell_offset,day[okay])
                    for column,expected in [('buy_reference',buy),('sell_reference',sell),('predicted',pred)]:
                        np.testing.assert_allclose(found[column],expected[okay],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(found.actual_extrema_scenario,realized[okay],rtol=1e-5,atol=2e-7)
                    counts+=len(found)
        paired=pd.read_csv(root/f'{part}-paired-ranking.csv',dtype={'seed':str})
        expected=actual.groupby('local_row').size()
        common=expected[expected==len(cfg['variants'])*(len(cfg['seeds'])+1)].index
        assert set(paired.local_row)==set(common)
        daily=pd.read_csv(root/f'{part}-daily.csv',dtype={'seed':str}).set_index(['arm','seed','date'])
        for key,g in paired.groupby(['arm','seed','date']):
            if len(g)<20:
                assert key not in daily.index
                continue
            top=g.sort_values(['predicted','instrument_id'],ascending=[False,True]).head(20)
            r=daily.loc[key]
            assert r.top_rows==20 and r.rows==len(g)
            np.testing.assert_allclose(r.top_extrema_scenario_pct,top.actual_extrema_scenario.mean()*100,rtol=1e-12)
            np.testing.assert_allclose(r.return_mae_pp,g.absolute_error.mean()*100,rtol=1e-12)
            np.testing.assert_allclose(r.top_lift_pp,(top.actual_extrema_scenario.mean()-g.actual_extrema_scenario.mean())*100,rtol=1e-12)
    selection=pd.read_csv(root/'selection-summary.csv',dtype={'seed':str})
    expected=selection[(selection.seed=='ensemble')&(selection.arm!='baseline')].sort_values(['top_lift_pp','arm'],ascending=[False,True]).iloc[0].arm
    assert read(root/'nomination.json')['arm']==expected
    torch.set_num_threads(4)
    decoder=load_decoder(read(root/'profiles.json')['decoder']['path'],cfg['device']) if replay else None
    for seed in cfg['seeds']:
        for arm in cfg['variants']:
            trained=root/f'seed{seed}'/arm
            t=read(trained/'training.json')
            assert t['examples']==cfg['epochs']*read(root/'splits.json')['training']['rows']
            assert len(t['history'])==cfg['epochs'] and t['reload_exact']
            model,saved=restore_model(trained/'model.pt',cfg['device'] if replay else 'cpu')
            assert saved['trained_labels_through']<cfg['selection'][0]
            assert all(torch.isfinite(x).all() for x in model.state_dict().values())
            if replay:
                ids=np.load(root/'evaluation-ids.npy')
                features=np.load(root/'evaluation-features.npy',mmap_mode='r')
                stored=np.load(trained/'evaluation/paths.npy',mmap_mode='r')
                for start in [0, len(ids)-cfg['forecast_batch']]:
                    ix=np.arange(start,start+cfg['forecast_batch'])
                    a,b,stamps,_=data.tensors(ids[ix],cfg['device'])
                    aux=None if arm=='baseline' else torch.tensor(np.array(features[ix]),device=cfg['device'])
                    args=(model,a[:,:60],b[:,:60],stamps[:,:60],stamps[:,60:])
                    kwargs=dict(samples=cfg['samples'],seed=17+start,top_p=1.,history_auxiliary=aux)
                    pairs=generate_cached(*args,**kwargs)
                    paths,legal=decode_cached(decoder,pairs,data.a['mean'][ids[ix]],data.a['scale'][ids[ix]],5)
                    np.testing.assert_array_equal(paths,stored[ix])
                    full=generate_tokens(*args,**kwargs)
                    same=all(torch.equal(x,y) for x,y in zip(pairs,full))
                    tokens_equal+=int(same)
                    # Logits already checked analytically in unit tests; tiny
                    # numerical sampling divergence is recorded, never hidden.
                    original,old_legal=decode_paths(decoder,pairs,data.a['mean'][ids[ix]],data.a['scale'][ids[ix]],5)
                    error=float(np.max(abs(paths-original)/data.a['scale'][ids[ix]][:,None]))
                    decoded_error=max(decoded_error,error)
                    assert error<2e-5, error
                    np.testing.assert_array_equal(legal,old_legal)
                    replay_rows+=len(ix)
            del model
            gc.collect()
            if cfg['device']=='mps':
                torch.mps.empty_cache()
    result=dict(passed=True,at=utc_now(),experiment_sha256=file_hash(root/'completed.json'),
        independently_recomputed_reference_rows=counts,feature_windows=384,replayed_inputs=replay_rows,
        original_sampler_exact_probe_batches=tokens_equal,total_probe_batches=18 if replay else 0,
        decoder_max_error_in_normalized_units=decoded_error,live_unchanged=True,
        nomination_uses_selection_only=True,code_sha256=file_hash(Path(__file__)))
    write_json(root.parent/(root.name+'-verification.json'),result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=BASE/'artifacts/token-ranking-20260916-v2')
    p.add_argument('--no-replay',action='store_true')
    args=p.parse_args()
    verify(args.root,not args.no_replay)
