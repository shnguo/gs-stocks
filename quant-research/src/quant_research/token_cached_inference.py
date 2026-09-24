"""Isolated evaluation-only KV cache for the small causal model and pinned decoder.

No checkpoint changes. Supports forecasts fitting within the context window;
sliding windows use the original implementation instead. Floating-point kernel
shapes differ, so validate logits/decoded values and use the same backend for
all arms of a comparison. Training remains on the original implementation.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .token_transformer import forecast_auxiliary, sample_logits, valid_bars


def _heads(x, heads):
    return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)


def _project(attention, z):
    q, k, v = F.linear(z, attention.in_proj_weight, attention.in_proj_bias).chunk(3, -1)
    return tuple(_heads(x, attention.num_heads) for x in [q, k, v])


def _attend(attention, q, k, v, mask=None):
    z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
    return attention.out_proj(z.transpose(1, 2).reshape(z.shape[0], z.shape[2], -1))


class PredictorCache:
    def __init__(self, model, a, b, stamps, auxiliary, samples):
        self.model, self.samples, self.length = model, samples, a.shape[1]
        self.cache = []
        captured, hooks = [], []
        for layer in model.layers:
            hooks.append(layer.attention.register_forward_pre_hook(lambda module, args: captured.append(args[0])))
        try:
            hidden = model.encode_history(a, b, stamps, auxiliary)
        finally:
            for hook in hooks:
                hook.remove()
        for layer, z in zip(model.layers, captured):
            _, k, v = _project(layer.attention, z)
            self.cache.append((k.repeat_interleave(samples, 0), v.repeat_interleave(samples, 0)))
        self.hidden = hidden.repeat_interleave(samples, 0)
        attention = model.dependency
        width = model.config.width
        self.dep_k = _heads(F.linear(hidden, attention.in_proj_weight[width:2*width], attention.in_proj_bias[width:2*width]), attention.num_heads).repeat_interleave(samples, 0)
        self.dep_v = _heads(F.linear(hidden, attention.in_proj_weight[2*width:], attention.in_proj_bias[2*width:]), attention.num_heads).repeat_interleave(samples, 0)

    def coarse(self):
        return self.model.coarse_head(self.hidden[:, -1])

    def fine(self, first):
        model, attn = self.model, self.model.dependency
        width = model.config.width
        q = _heads(F.linear(model.emb_s1(first[:, None]), attn.in_proj_weight[:width], attn.in_proj_bias[:width]), attn.num_heads)
        z = _attend(attn, q, self.dep_k, self.dep_v)
        return model.fine_head(model.dependency_norm(self.hidden[:, -1:]+z))[:, 0]

    def append(self, first, second, stamp):
        model = self.model
        x = model.fusion(torch.cat([model.emb_s1(first[:, None]), model.emb_s2(second[:, None])], -1)*math.sqrt(model.config.width))
        x = x+model.position[:, self.length:self.length+1]
        for i, embedding in enumerate(model.time_embeddings):
            x = x+embedding(stamp[:, None, i])
        if model.config.auxiliary_features:
            # Future auxiliary values are missing: zero standardized value,
            # availability zero. Biases may still provide a learned residual.
            empty = torch.zeros((len(first), 1, len(model.config.auxiliary_features)*2), device=x.device, dtype=x.dtype)
            x = x+model.auxiliary_projection(empty)
        for i, layer in enumerate(model.layers):
            q, k, v = _project(layer.attention, layer.norm1(x))
            past_k, past_v = self.cache[i]
            k, v = torch.cat([past_k, k], 2), torch.cat([past_v, v], 2)
            x = x+_attend(layer.attention, q, k, v)
            x = x+layer.ff(layer.norm2(x))
            self.cache[i] = k, v
        hidden = model.norm(x)
        self.hidden = torch.cat([self.hidden, hidden], 1)
        attn, width = model.dependency, model.config.width
        k = _heads(F.linear(hidden, attn.in_proj_weight[width:2*width], attn.in_proj_bias[width:2*width]), attn.num_heads)
        v = _heads(F.linear(hidden, attn.in_proj_weight[2*width:], attn.in_proj_bias[2*width:]), attn.num_heads)
        self.dep_k, self.dep_v = torch.cat([self.dep_k, k], 2), torch.cat([self.dep_v, v], 2)
        self.length += 1


@torch.inference_mode()
def generate_cached(model, a, b, past_stamps, future_stamps, *, samples=64, seed=17,
                    temperature=1., top_p=1., top_k=0, history_auxiliary=None):
    if (samples < 1 or a.ndim != 2 or a.shape != b.shape or past_stamps.shape != (*a.shape, 5)
            or future_stamps.ndim != 3 or future_stamps.shape[0] != len(a) or future_stamps.shape[2] != 5
            or future_stamps.shape[1] < 1 or a.shape[1]+future_stamps.shape[1] > model.config.max_context):
        raise ValueError('Cached generation requires matching shapes within context')
    old = model.training
    model.eval()
    device = next(model.parameters()).device
    a, b, past_stamps, future_stamps = [x.to(device) for x in [a, b, past_stamps, future_stamps]]
    aux = forecast_auxiliary(history_auxiliary, a.shape[1], a.shape[1])
    if aux is not None:
        aux = aux.to(device)
    try:
        cache = PredictorCache(model, a, b, past_stamps, aux, samples)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        firsts, seconds = [], []
        for step in range(future_stamps.shape[1]):
            first = sample_logits(cache.coarse(), generator, temperature, top_p, top_k).to(device)
            second = sample_logits(cache.fine(first), generator, temperature, top_p, top_k).to(device)
            firsts.append(first)
            seconds.append(second)
            if step+1 < future_stamps.shape[1]:
                cache.append(first, second, future_stamps[:, step].repeat_interleave(samples, 0))
        return tuple(torch.cat([past.repeat_interleave(samples, 0), torch.stack(tail, 1)], 1).reshape(len(a), samples, -1)
                     for past, tail in [(a, firsts), (b, seconds)])
    finally:
        model.train(old)


def _rope(attention, q, k, offset):
    cos, sin = attention.rotary._update_cos_sin_cache(q, offset+q.shape[2])
    cos, sin = cos[:, :, offset:], sin[:, :, offset:]
    rotate = attention.rotary._rotate_half
    return q*cos+rotate(q)*sin, k*cos+rotate(k)*sin


@torch.inference_mode()
def decode_cached(decoder, pairs, mean, scale, horizon, max_context=512):
    a, b = pairs
    if (a.ndim != 3 or a.shape != b.shape or not 1 <= horizon < a.shape[-1]
            or a.shape[-1] > max_context or mean.shape != (len(a), 1, 6) or scale.shape != mean.shape):
        raise ValueError('Invalid cached decoder shapes')
    decoder.eval()
    rows, samples, length = a.shape
    history = length-horizon
    for token in [a, b]:
        if not torch.equal(token[:, :, :history], token[:, :1, :history].expand(-1, samples, -1)):
            raise ValueError('Cached decoder requires identical historical tokens within each row')
    pre = [x[:, 0, :history] for x in pairs]
    tail = [x[:, :, history:].reshape(-1, horizon) for x in pairs]
    x = decoder.post_quant_embed(decoder.indices_to_bits(pre, True))
    y = decoder.post_quant_embed(decoder.indices_to_bits(tail, True))
    for layer in decoder.decoder:
        attn = layer.self_attn
        z = layer.norm1(x)
        q, k, v = [_heads(fn(z), attn.n_heads) for fn in [attn.q_proj, attn.k_proj, attn.v_proj]]
        q, k = _rope(attn, q, k, 0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
        x = x+attn.out_proj(out.transpose(1, 2).reshape(rows, history, -1))
        x = x+layer.ffn(layer.norm2(x))
        z = layer.norm1(y)
        tq, tk, tv = [_heads(fn(z), attn.n_heads) for fn in [attn.q_proj, attn.k_proj, attn.v_proj]]
        tq, tk = _rope(attn, tq, tk, history)
        all_k = torch.cat([k.repeat_interleave(samples, 0), tk], 2)
        all_v = torch.cat([v.repeat_interleave(samples, 0), tv], 2)
        mask = torch.ones(horizon, length, dtype=torch.bool, device=y.device).tril(history)
        out = F.scaled_dot_product_attention(tq, all_k, all_v, attn_mask=mask, dropout_p=0.)
        y = y+attn.out_proj(out.transpose(1, 2).reshape(rows*samples, horizon, -1))
        y = y+layer.ffn(layer.norm2(y))
    paths = decoder.head(y).cpu().numpy().reshape(rows, samples, horizon, 6)
    paths = paths*np.asarray(scale)[:, None]+np.asarray(mean)[:, None]
    return paths, valid_bars(paths).all(-1)
