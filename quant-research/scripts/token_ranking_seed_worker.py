"""One manually launched independent seed, staged atomically to avoid shared writes."""
import argparse
import gc
import os
import shutil
from pathlib import Path

import torch
from token_features_run import read, verify_manifest
from token_history_run import Dataset
from token_ranking_run import BASE, forecast, train
from tokenizer_reconstruction_run import load_decoder

from quant_research.storage import file_hash, utc_now, write_json


def run(root, seed):
    cfg=read(root/'protocol.json')
    if seed not in cfg['seeds']:
        raise ValueError('Seed is outside the fixed protocol')
    stage=root.parent/(root.name+f'-worker{seed}')
    target=root/f'seed{seed}'
    if stage.exists() or target.exists():
        raise ValueError('Worker stage or destination already exists; retain evidence')
    verify_manifest(root,'prepared.json')
    for name in ['scripts/token_ranking_run.py','src/quant_research/token_cached_inference.py','src/quant_research/return_ranking_loss.py']:
        if file_hash(BASE/name)!=file_hash(root/'code'/name):
            raise ValueError('Worker code differs from frozen training implementation')
    stage.mkdir()
    for p in root.iterdir():
        if p.is_file() and p.suffix in ['.json','.npy','.npz']:
            shutil.copy2(p,stage/p.name)
    torch.set_num_threads(4)
    data=Dataset()
    decoder=load_decoder(read(stage/'profiles.json')['decoder']['path'],cfg['device'])
    for arm in cfg['variants']:
        train(stage,cfg,data,seed,arm,decoder)
        for part in ['selection','evaluation']:
            forecast(stage,cfg,data,seed,arm,decoder,part)
    del decoder
    gc.collect()
    # Never replace a seed already started by the sequential coordinator.
    if target.exists():
        raise ValueError('Coordinator reached this seed; completed worker retained separately')
    os.rename(stage/f'seed{seed}',target)
    write_json(stage/'receipt.json',dict(at=utc_now(),seed=seed,destination=str(target),
        protocol_sha256=file_hash(root/'protocol.json'),code_sha256=file_hash(Path(__file__)),
        reason='Independent seed executed concurrently in an isolated stage; identical data, epochs, per-seed initialization and inference settings.'))
    print('Staged completed seed',seed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    a=p.parse_args()
    run(a.root.resolve(),a.seed)
