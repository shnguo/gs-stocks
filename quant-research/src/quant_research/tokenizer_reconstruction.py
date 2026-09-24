"""Decoder adaptation with fixed token semantics and masked price-space objectives."""
import torch
import torch.nn.functional as F

TRAINABLE=('post_quant_embed.','decoder.','head.')


def enable_decoder_training(tokenizer):
    for name,p in tokenizer.named_parameters():
        p.requires_grad_(name.startswith(TRAINABLE))
    return [p for p in tokenizer.parameters() if p.requires_grad]


def prefix_mean(values,known):
    if values.shape!=known.shape or not known.any(1).all():
        raise ValueError('Each row must contain at least one observed step')
    if not torch.equal(known,known.to(torch.int64).cumprod(1).bool()):
        raise ValueError('Known targets must be a prefix')
    return torch.where(known,values,0).sum(1)/known.sum(1)


def tensor_extrema(raw,reference):
    high=raw[...,1].amax(1)
    low=raw[...,2].amin(1)
    return torch.stack([(high/reference-1)*100,(low/reference-1)*100,(high-low)/reference*100],1)


def reconstruction_loss(decoded,future,known,mean,scale,reference,objective,config):
    """Per-row losses; no target or gradient is taken from unknown suffixes."""
    if objective not in ['reconstruction','price_consistency']:
        raise ValueError('Unknown decoder objective')
    safe=torch.where(known[...,None],future,mean)
    target=(safe-mean)/scale
    clean=torch.where(known[...,None],decoded,torch.zeros_like(decoded))
    normalized=F.smooth_l1_loss(clean,target,reduction='none',beta=config['normalized_huber_beta']).mean(-1)
    result=prefix_mean(normalized,known)
    if objective=='reconstruction':
        return result
    raw=clean*scale+mean
    # Natural OHLC relationships are learned through the loss, never repaired at inference.
    order=(F.relu(raw[...,0]-raw[...,1])+F.relu(raw[...,3]-raw[...,1])+
           F.relu(raw[...,2]-raw[...,0])+F.relu(raw[...,2]-raw[...,3]))/reference[:,None]*100
    consistency=prefix_mean(order,known)
    extrema=[]
    for h in [2,5]:
        length=torch.minimum(known.sum(1),torch.full_like(known.sum(1),h))
        mask=torch.arange(known.shape[1],device=known.device)[None]<length[:,None]
        def boundaries(x):
            high=torch.where(mask,x[...,1],-torch.inf).amax(1)
            low=torch.where(mask,x[...,2],torch.inf).amin(1)
            return torch.stack([(high/reference-1)*100,(low/reference-1)*100,(high-low)/reference*100],1)
        error=F.smooth_l1_loss(boundaries(raw),boundaries(safe),reduction='none',beta=config['extrema_huber_beta_pp']).mean(1)
        extrema.append(error)
    return result+config['extrema_weight']*torch.stack(extrema).mean(0)+config['ohlc_order_weight']*consistency
