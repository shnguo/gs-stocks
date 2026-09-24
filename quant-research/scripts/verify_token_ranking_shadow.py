"""Independently audit one immutable manual candidate publication and replay it."""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_features_run import read
from tokenizer_reconstruction_run import load_decoder
from verify_token_ranking import independent_stats

from quant_research.daily_loop import verify
from quant_research.daily_token import load_arrays
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_cached_inference import decode_cached, generate_cached
from quant_research.token_transformer import restore_model


def audit(run,config):
    cfg=read(config)
    verify(run)
    info=read(run/'publication.json')
    parent=Path(info['current_run'])
    verify(parent)
    binding=read(run/'binding.json')
    assert binding['daily_manifest_sha256']==file_hash(parent/'manifest.json')
    rows=pd.read_parquet(parent/'inputs/forecast-rows.parquet')
    models={s:np.load(run/f'seed{s}/paths.npy',mmap_mode='r') for s in cfg['checkpoints']}
    records=[]
    for start in range(0,len(rows),256):
        end=min(start+256,len(rows))
        stats=[independent_stats(x[start:end],cfg['minimum_legal_paths']) for x in models.values()]
        okay=np.logical_and.reduce([x[0] for x in stats])
        frequency=np.mean([x[1] for x in stats],axis=0)
        day=frequency.argmax(1)+1
        buy=np.mean([x[2] for x in stats],axis=0)
        sell=np.mean([x[3] for x in stats],axis=0)[np.arange(end-start),day-1]
        for i in np.flatnonzero(okay):
            records.append(dict(instrument_id=rows.iloc[start+i].instrument_id,buy_reference_price=buy[i],
                sell_reference_price=sell[i],expected_net_return=sell[i]/buy[i]-1-cfg['cost'],
                sell_reference_date=info['horizon_dates'][day[i]],sell_date_frequency=frequency[i,day[i]-1]))
    expected=pd.DataFrame(records).sort_values(['expected_net_return','instrument_id'],ascending=[False,True])
    actual=pd.read_csv(run/'candidate-ranking.csv')
    np.testing.assert_array_equal(actual.instrument_id,expected.instrument_id)
    np.testing.assert_array_equal(actual['rank'],np.arange(1,len(actual)+1))
    np.testing.assert_array_equal(actual.sell_reference_date,expected.sell_reference_date)
    assert actual.buy_date.eq(info['horizon_dates'][0]).all()
    for name in ['buy_reference_price','sell_reference_price','expected_net_return','sell_date_frequency']:
        np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
    pointer=read(parent.parent.parent/'reference-publications'/(parent.name+'.json'))
    assert pointer['manifest_sha256']==binding['current_publication_sha256']
    current=Path(pointer['report'])
    verify(current)
    pd.testing.assert_frame_equal(pd.read_csv(run/'current-ranking.csv'),pd.read_csv(current/'ranking.csv'),check_exact=False,rtol=1e-12,atol=1e-12)
    delivery=read(current/'delivery.json')
    timely=bool(delivery['prospective'] and pd.Timestamp(info['completed_at'])<pd.Timestamp(info['horizon_dates'][0]+'T09:15:00',tz='Asia/Shanghai'))
    assert timely==info['prospective']
    torch.set_num_threads(4)
    decoder=load_decoder(cfg['decoder']['path'],cfg['device'])
    arrays=load_arrays(parent/'inputs','forecast')
    features=np.load(run/'history-features.npy',mmap_mode='r')
    size=cfg['forecast_batch']
    a,b=[torch.tensor(np.array(arrays[k][:size]),device=cfg['device'],dtype=torch.long) for k in ['s1','s2']]
    stamps=torch.tensor(np.array(arrays['stamps']),device=cfg['device'],dtype=torch.long)[None].expand(size,-1,-1)
    aux=torch.tensor(np.array(features[:size]),device=cfg['device'])
    for seed,spec in cfg['checkpoints'].items():
        assert file_hash(Path(spec['path']))==spec['sha256']
        model,_=restore_model(spec['path'],cfg['device'])
        pairs=generate_cached(model,a,b,stamps[:,:60],stamps[:,60:],samples=cfg['samples'],seed=17,
            top_p=1.,history_auxiliary=aux)
        replay,_=decode_cached(decoder,pairs,arrays['mean'][:size],arrays['scale'][:size],5)
        np.testing.assert_array_equal(replay,models[seed][:size])
        del model
        gc.collect()
        if cfg['device']=='mps':
            torch.mps.empty_cache()
    for path,expected_hash in read(Path(cfg['experiment'])/'live-state.json').items():
        assert file_hash(Path(path))==expected_hash,path
    result=dict(passed=True,at=utc_now(),rows=len(actual),replayed_inputs=size*len(models),
        prospective=timely,main_state_unchanged=True,run_manifest_sha256=file_hash(run/'manifest.json'),
        config_sha256=file_hash(config),code_sha256=file_hash(Path(__file__)))
    receipt=run.parent.parent/'verification'/(run.name+'.json')
    write_json(receipt,result)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    audit(args.run,args.config)
