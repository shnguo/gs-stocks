"""Independent artifact, split, checkpoint and forecast-step CE audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from quant_research.storage import file_hash, write_json
from quant_research.token_transformer import restore_model

BASE=Path('/Users/guo/Documents/stocks/quant-research')
DATA=BASE/'artifacts/token-history-data-20260915-v1'
OUT=BASE/'artifacts/token-history-experiment-20260915-v1'


def read(p):
    return json.loads(p.read_text())


def main():
    cfg=read(DATA/'protocol.json')
    assert read(OUT/'all-completed.json')['passed']
    assert read(BASE/'artifacts/token-history-prefix-independent-check-20260915.json')['passed']
    manifest=read(DATA/'completed.json')
    for path,digest in manifest['files'].items():
        assert file_hash(DATA/path)==digest,path
    rows=pd.read_parquet(DATA/'rows.parquet')
    dates=read(DATA/'calendar.json')
    stamps=np.load(DATA/'calendar-stamps.npy')
    arrays={k:np.load(DATA/f'{k}.npy',mmap_mode='r') for k in ['s1','s2','valid']}
    summary=read(DATA/'summary.json')
    assert rows.date.nunique()==summary['dates']
    assert rows.label_end.max()==cfg['last_label']<cfg['sealed_holdout_start']
    coverage=pd.read_csv(DATA/'daily-coverage.csv')
    assert int(coverage.history_and_reference_available.sum())==summary['candidate_windows']
    assert np.load(DATA/'all-eligible-coordinates.npy',mmap_mode='r').shape==(summary['candidate_windows'],2)
    for fold in cfg['folds']:
        parts=[]
        for part in ['train','selection','evaluation']:
            ids=np.load(DATA/f'{fold["name"]}-{part}.npy')
            selected=rows.iloc[ids]
            start,end=fold[part]
            assert selected.date.ge(start).all() and selected.date.lt(end).all()
            assert all(dates[t+5]<end for t in selected.date_index)
            parts.append(set(ids.tolist()))
        assert not (parts[0]&parts[1] or parts[1]&parts[2] or parts[0]&parts[2])
    torch.set_num_threads(4)
    records,checks=[],[]
    for fold in cfg['folds']:
        names=['dense_282k_equal','dense_3720k_equal','dense_3720k_weighted']
        ids=np.load(DATA/f'{fold["name"]}-evaluation.npy')
        t=rows.iloc[ids].date_index.to_numpy(int)
        calendar=stamps[t[:,None]+np.arange(-59,6)]
        for name in names:
            root=OUT/fold['name']/name
            for phase in ['training','forecast']:
                meta=read(root/phase/'completed.json')
                assert meta['passed']
                for path,digest in meta['files'].items():
                    assert file_hash(root/phase/path)==digest,(name,phase,path)
            # CPU checkpoint reload must reproduce the saved logits exactly.
            model,saved=restore_model(root/'training/best.pt','cpu')
            model.eval()
            with np.load(root/'training/reload-reference.npz') as z:
                probe=z['ids']
                a,b=[torch.as_tensor(arrays[k][probe].astype(np.int64)) for k in ['s1','s2']]
                tt=rows.iloc[probe].date_index.to_numpy(int)
                ss=torch.as_tensor(stamps[tt[:,None]+np.arange(-59,6)])
                with torch.inference_mode():
                    first,second=model.forecast_logits(a[:,:-1],b[:,:-1],ss[:,:-1],a[:,1:],59)
                np.testing.assert_array_equal(first.numpy(),z['coarse'])
                np.testing.assert_array_equal(second.numpy(),z['fine'])
            # Explicit CE arrays, without the new reduction helper, on every evaluation-pool row.
            model=model.to(cfg['device'])
            all_errors=[]
            for start in range(0,len(ids),cfg['batch_size']):
                batch=ids[start:start+cfg['batch_size']]
                a,b=[torch.as_tensor(arrays[k][batch].astype(np.int64),device=cfg['device']) for k in ['s1','s2']]
                ss=torch.as_tensor(calendar[start:start+len(batch)],device=cfg['device'])
                with torch.inference_mode():
                    logits=model.forecast_logits(a[:,:-1],b[:,:-1],ss[:,:-1],a[:,1:],59)
                    errors=[F.cross_entropy(p.transpose(1,2),target[:,60:],reduction='none').cpu().numpy()
                        for p,target in zip(logits,[a,b])]
                all_errors.append(np.stack(errors,-1))
            errors=np.concatenate(all_errors)
            valid=arrays['valid'][ids]
            for day in range(5):
                keep=valid[:,day]
                sub=pd.DataFrame(dict(date=rows.iloc[ids[keep]].date.to_numpy(),
                    coarse=errors[keep,day,0],fine=errors[keep,day,1]))
                daily=sub.groupby('date')[['coarse','fine']].mean()
                records.append(dict(fold=fold['name'],model=name,day=day+1,known_rows=int(keep.sum()),
                    dates=len(daily),coarse_ce=float(daily.coarse.mean()),fine_ce=float(daily.fine.mean()),
                    mean_ce=float(daily.to_numpy().mean())))
            probe_weights=np.asarray([1,.8,.6,.4,.2])
            known=valid.any(1)
            weighted=(errors.mean(-1)*valid*probe_weights).sum(1)/np.maximum((valid*probe_weights).sum(1),1e-12)
            equal=(errors.mean(-1)*valid).sum(1)/np.maximum(valid.sum(1),1)
            assert np.isfinite(weighted[known]).all() and np.isfinite(equal[known]).all()
            np.savez_compressed(OUT/f'{fold["name"]}-{name}-evaluation-ce.npz',row_ids=ids,
                per_day_head_ce=errors,valid=valid,equal_per_row=equal,weighted_per_row=weighted)
            stats=read(root/'training/summary.json')
            visited=np.load(root/'training/visited-row-ids.npy')
            pool=np.load(DATA/f'{fold["name"]}-train.npy')
            assert np.isin(visited,pool).all() and arrays['valid'][visited].any(1).all()
            assert len(visited)==stats['unique_examples_visited']
            assert rows.iloc[visited].date.nunique()==stats['training_dates']
            history=read(root/'training/history.json')
            selected_visits=history[saved['epoch']-1]['unique_examples_visited']
            assert selected_visits<=len(visited)
            checks.append(dict(fold=fold['name'],model=name,passed=True,parameters=sum(p.numel() for p in model.parameters()),
                checkpoint_epoch=saved['epoch'],cpu_reload_exact=True,evaluation_rows=len(ids),visited_rows=len(visited),
                selected_checkpoint_unique_rows=selected_visits))
            print('Verified',fold['name'],name,flush=True)
            del model
            torch.mps.empty_cache()
    pd.DataFrame(records).to_csv(OUT/'per-day-token-ce.csv',index=False)
    write_json(OUT/'independent-verification.json',dict(passed=True,checks=checks,
        exhaustive_partition_checks=True,source_manifest_verified=True,tokenizer_prefix_verified=True,
        sealed_holdout_accessed=False,notes='Forecast-step CE is teacher-forced and is separate from freely generated path accuracy'))


if __name__=='__main__':
    main()
