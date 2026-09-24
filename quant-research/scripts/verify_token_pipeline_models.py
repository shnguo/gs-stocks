"""Verify portable confirmation checkpoints and actual-data tokenizer prefix causality."""
from pathlib import Path

import numpy as np
import torch
from token_history_run import BUNDLE, DATA, Dataset

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import load_tokenizer, normalized_history, restore_model

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ROOT=BASE/'artifacts/token-pipeline-confirmation-20260915-v1'


def main():
    torch.set_num_threads(4)
    data=Dataset()
    records=[]
    for seed in [17,29,43]:
        root=ROOT/f'seed{seed}/dense_3720k_equal/training'
        model,meta=restore_model(root/'best.pt','cpu')
        model.eval()
        with np.load(root/'reload-reference.npz') as z,torch.inference_mode():
            a,b,stamps,_=data.tensors(z['ids'],'cpu')
            logits=model.forecast_logits(a[:,:-1],b[:,:-1],stamps[:,:-1],a[:,1:],59)
            np.testing.assert_array_equal(logits[0].numpy(),z['coarse'])
            np.testing.assert_array_equal(logits[1].numpy(),z['fine'])
        records.append(dict(seed=seed,epoch=meta['epoch'],checkpoint_sha256=file_hash(root/'best.pt'),
            parameters=sum(p.numel() for p in model.parameters()),cpu_reload_exact=True))
    tokenizer=load_tokenizer(BUNDLE,'cpu')
    ids=np.load(ROOT/'selection-ids.npy')[:4]
    r=data.rows.iloc[ids]
    stock,t=r[['stock_index','date_index']].to_numpy(int).T
    raw=np.load(BASE/'artifacts/kronos-inputs-20260914-v3/values.npy',mmap_mode='r')
    history,mean,scale=normalized_history(raw[stock[:,None],t[:,None]+np.arange(-59,1)].copy())
    with torch.inference_mode():
        prefix=tokenizer.encode(torch.from_numpy(history),half=True)
        arbitrary_suffix=torch.full((len(ids),5,6),3.)
        suffixed=tokenizer.encode(torch.cat([torch.from_numpy(history),arbitrary_suffix],1),half=True)
    for head,name in enumerate(['s1','s2']):
        np.testing.assert_array_equal(prefix[head].numpy(),data.a[name][ids,:60])
        np.testing.assert_array_equal(prefix[head].numpy(),suffixed[head][:,:60].numpy())
    write_json(ROOT/'model-verification.json',dict(passed=True,at=utc_now(),checkpoints=records,
        actual_history_prefix_rows=len(ids),future_suffix_does_not_change_past_tokens=True,
        training_data_manifest_sha256=file_hash(DATA/'completed.json')))
    print('Three CPU checkpoints and real-data token prefix causality verified',flush=True)


if __name__=='__main__':
    main()
