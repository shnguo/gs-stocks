"""Pinned Kronos-base native five-day paths on inherited stock/date cohort."""
import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from compare_token_kronos_paths import ART, ROOTS, legal, read, sha

BUNDLE = ART / 'kronos-comparison-20260910-v1'
WEIGHTS = ART / 'kronos-base-weights-20260914-v1'
PARENT = ART / 'token-kronos-path-comparison-20260914-v1'
OUT = ART / 'kronos-base-paths-20260914-v2'
REVISION = '2b554741eca47781b64468546e77fef3e85130e6'
OFFICIAL_SHA256 = 'abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83'


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def stamps(dates):
    d = pd.DatetimeIndex(dates)
    return np.column_stack([np.zeros(len(d)),np.zeros(len(d)),d.weekday,d.day,d.month]).astype(np.float32)


def prepare():
    OUT.mkdir()
    protocol = dict(model='NeoQuasar/Kronos-base', revision=REVISION, samples=32, horizon=5,
        lookback=60, max_context=512, temperature=1., top_p=.9, top_k=0, clip=5,
        batch_rows=4, seed=17, seed_rule='17 + batch starting row index', device='mps',
        sampling='Native upstream auto_regressive_inference; repeated independent paths and sample_count=1',
        cohort='All 1510 previously matched stock/date rows, before base path validity or future-label filtering',
        sealed_holdout='No labels read by generator; horizon must end before 2025-08-07',
        training='Frozen pretrained weights; no fine-tuning',
        expected_predictor_parameters=102310592, expected_registered_buffers=416)
    write(OUT / 'protocol.json', protocol)
    source = {}
    def check(path, expected):
        actual = sha(path)
        assert actual == expected, str(path)
        source[str(path)] = actual
    check(WEIGHTS / 'model.safetensors', OFFICIAL_SHA256)
    provenance = read(BUNDLE / 'pretrained-provenance.json')
    for name in ['upstream/model/__init__.py','upstream/model/kronos.py','upstream/model/module.py',
                 'tokenizer/config.json','tokenizer/model.safetensors']:
        check(BUNDLE / name, provenance['files'][name])
    for name in ['config.json','README.md']:
        source[str(WEIGHTS / name)] = sha(WEIGHTS / name)
    check(PARENT / 'matched-rows.parquet',read(PARENT / 'completed.json')['files']['matched-rows.parquet'])
    rows = pd.read_parquet(PARENT / 'matched-rows.parquet')
    assert not rows.duplicated(['date','instrument_id']).any()
    experiment = read(ROOTS['decoder_3720k'] / 'experiment.json')
    panel = Path(experiment['inputs'])
    check(panel / 'manifest.json', experiment['input_manifest_sha256'])
    manifest = read(panel / 'manifest.json')
    check(panel / 'values.npy', manifest['values_sha256'])
    raw = np.load(panel / 'values.npy', mmap_mode='r')
    si = {s:i for i,s in enumerate(manifest['instruments'])}
    di = {d:i for i,d in enumerate(manifest['dates'])}
    windows, past, future = [], [], []
    for row in rows.itertuples():
        t=di[row.date]
        assert t>=59 and manifest['dates'][t+5] < '2025-08-07'
        windows.append(raw[si[row.instrument_id], t-59:t+1])
        past.append(stamps(manifest['dates'][t-59:t+1]))
        future.append(stamps(manifest['dates'][t+1:t+6]))
    windows = np.stack(windows)
    assert np.isfinite(windows).all() and (windows[...,6]>0).all()
    assert legal(windows[:,None,:,:6]).all()
    x = windows[...,:6].astype(np.float32,copy=True)
    x[...,:4] *= windows[...,6:7] / windows[:,-1:,6:7]
    mean, scale = x.mean(1,keepdims=True), x.std(1,keepdims=True)+1e-5
    x = np.clip((x-mean)/scale,-5,5)
    np.savez_compressed(OUT / 'inputs.npz', x=x, mean=mean, scale=scale,
                        past=np.stack(past), future=np.stack(future))
    rows[['date','instrument_id']].to_parquet(OUT / 'rows.parquet',index=False)
    for filename in ['kronos_base_paths.py','compare_token_kronos_paths.py']:
        shutil.copy2(Path(__file__).parent / filename,OUT / filename)
    write(OUT / 'sources.json',source)
    write(OUT / 'provenance.json',dict(repo='NeoQuasar/Kronos-base',revision=REVISION,
        mirror='https://hf-mirror.com',official_weight_sha256=OFFICIAL_SHA256,
        official_hash_source='https://huggingface.co/NeoQuasar/Kronos-base/blob/main/model.safetensors',
        exact_pretraining_dates_independently_verified=False,
        upstream_commit=provenance['upstream_commit'],tokenizer_revision=provenance['tokenizer_revision']))
    write(OUT / 'prepared.json',dict(files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print('Prepared',len(rows),'rows',flush=True)


def checked():
    for name,expected in read(OUT/'prepared.json')['files'].items():
        assert sha(OUT/name)==expected,name
    for name,expected in read(OUT/'sources.json').items():
        assert sha(Path(name))==expected,name


def load():
    import torch
    from safetensors.torch import load_model
    sys.path.insert(0,str(BUNDLE/'upstream'))
    api=importlib.import_module('model')
    tokenizer=api.KronosTokenizer(**read(BUNDLE/'tokenizer/config.json'))
    model=api.Kronos(**read(WEIGHTS/'config.json'))
    load_model(tokenizer,str(BUNDLE/'tokenizer/model.safetensors'),strict=True)
    load_model(model,str(WEIGHTS/'model.safetensors'),strict=True)
    assert sum(p.numel() for p in model.parameters())==102310592
    assert sum(p.numel() for p in model.buffers())==416
    torch.set_num_threads(4)
    return tokenizer.eval().requires_grad_(False).to('mps'),model.eval().requires_grad_(False).to('mps')


def generate(tokenizer,model,data,start,end):
    import torch
    torch.manual_seed(17+start)
    np.random.seed(17+start)
    inference=importlib.import_module('model.kronos').auto_regressive_inference
    with torch.inference_mode():
        x,past,future=[torch.from_numpy(np.repeat(data[k][start:end],32,axis=0)).to('mps')
                       for k in ['x','past','future']]
        result=inference(tokenizer,model,x,past,future,max_context=512,pred_len=5,
                         clip=5,T=1.,top_k=0,top_p=.9,sample_count=1,verbose=False)
    result=result[:,-5:].reshape(end-start,32,5,6)
    return result*data['scale'][start:end,None]+data['mean'][start:end,None]


def run():
    import torch
    checked()
    tokenizer,model=load()
    with np.load(OUT/'inputs.npz') as z:
        data={k:z[k] for k in z.files}
    dest=OUT/'forecast'
    dest.mkdir()
    start_time=time.monotonic()
    count=len(data['x'])
    write(OUT/'runtime.json',dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version,
        device='mps',parameter_count=sum(p.numel() for p in model.parameters()),pid=os.getpid()))
    chunks=[]
    for start in range(0,count,4):
        end=min(start+4,count)
        paths=generate(tokenizer,model,data,start,end)
        valid=legal(paths)
        np.savez_compressed(dest/f'paths-{start:05d}.npz',paths=paths,valid_paths=valid,
                            row_indices=np.arange(start,end),seed=17+start)
        chunks.append(paths)
        elapsed=time.monotonic()-start_time
        write(OUT/'progress.json',dict(status='running',rows=end,total=count,elapsed_seconds=elapsed,
                                      estimated_remaining_seconds=elapsed/end*(count-end)))
        if start % 32==0:
            print('Rows',end,'/',count,'seconds',round(elapsed,1),flush=True)
    combined=np.concatenate(chunks)
    np.savez_compressed(dest/'paths.npz',paths=combined,valid_paths=legal(combined))
    summary=dict(rows=count,paths_shape=list(combined.shape),legal_paths=int(legal(combined).sum()),
                 rows_with_eight_legal_paths=int((legal(combined).sum(1)>=8).sum()),
                 elapsed_seconds=time.monotonic()-start_time)
    write(dest/'summary.json',summary)
    write(dest/'completed.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    write(OUT/'progress.json',dict(status='completed',**summary))
    print(json.dumps(summary),flush=True)


def verify():
    import torch
    checked()
    for name,expected in read(OUT/'forecast/completed.json')['files'].items():
        assert sha(OUT/'forecast'/name)==expected,name
    tokenizer,model=load()
    with np.load(OUT/'inputs.npz') as z:
        data={k:z[k] for k in z.files}
    with np.load(OUT/'forecast/paths.npz') as z:
        paths,valid=z['paths'],z['valid_paths']
    reproduced=generate(tokenizer,model,data,0,4)
    np.testing.assert_allclose(reproduced,paths[:4],atol=1e-6,rtol=1e-6)
    np.testing.assert_array_equal(legal(paths),valid)
    with torch.inference_mode():
        s1,s2=tokenizer.encode(torch.from_numpy(data['x'][:1]).to('mps'),half=True)
        past=torch.from_numpy(data['past'][:1]).to('mps')
        before,_=model.decode_s1(s1,s2,past)
        changed1,changed2=s1.clone(),s2.clone()
        changed1[:,40:]=(changed1[:,40:]+1)%1024
        changed2[:,40:]=(changed2[:,40:]+1)%1024
        after,_=model.decode_s1(changed1,changed2,past)
        np.testing.assert_allclose(before[:,:40].cpu(),after[:,:40].cpu(),atol=1e-5,rtol=1e-5)
    result=dict(passed=True,reloaded_paths_reproduced=128,causal_logits_checked=40*1024,
                shape=list(paths.shape),legal_paths=int(valid.sum()),
                completed_sha256=sha(OUT/'forecast/completed.json'))
    write(OUT/'verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','run','verify'])
    globals()[p.parse_args().stage]()
