"""Small decoder-only model for Kronos-compatible hierarchical K-line tokens.

The backbone is trained from scratch. The separately loaded Kronos tokenizer is
frozen. Forecast heads predict market tokens; optional external training losses
can score observed prices and returns without adding trading-policy rules.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .storage import file_hash

FIELDS = ("open", "high", "low", "close", "volume", "amount")


@dataclass(frozen=True)
class TokenConfig:
    width: int = 48
    layers: int = 2
    heads: int = 4
    max_context: int = 512
    s1_bits: int = 10
    s2_bits: int = 10
    dropout: float = 0.1
    auxiliary_features: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "auxiliary_features", tuple(self.auxiliary_features))
        if (len(set(self.auxiliary_features)) != len(self.auxiliary_features)
                or any(not isinstance(x, str) or not x for x in self.auxiliary_features)):
            raise ValueError("Invalid auxiliary feature names")
        if (self.width < 4 or self.heads < 1 or self.width % self.heads
                or self.layers < 1 or self.max_context < 2
                or not 1 <= self.s1_bits <= 12 or not 1 <= self.s2_bits <= 12
                or not 0 <= self.dropout < 1):
            raise ValueError("Invalid decoder configuration")


def causal_mask(length, device):
    # MultiheadAttention uses True for forbidden positions, including in eval.
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, drop = config.width, config.dropout
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, config.heads, dropout=drop, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(width * 2, width),
        )
        self.dropout = nn.Dropout(drop)

    def forward(self, x, mask):
        z = self.norm1(x)
        z, _ = self.attention(z, z, z, attn_mask=mask, need_weights=False)
        x = x + self.dropout(z)
        return x + self.dropout(self.ff(self.norm2(x)))


class TokenTransformer(nn.Module):
    """Predict s1[t+1], then s2[t+1] conditioned on s1[t+1] and past tokens.

Uses the same coarse/fine factorization as Kronos; retains our small width and
depth, learned positions, LayerNorm and GELU rather than claiming weight or
block-level equivalence with the official RoPE/RMSNorm/SwiGLU backbone.
"""
    def __init__(self, config=TokenConfig()):
        super().__init__()
        self.config = config
        width = config.width
        self.emb_s1 = nn.Embedding(2 ** config.s1_bits, width)
        self.emb_s2 = nn.Embedding(2 ** config.s2_bits, width)
        self.fusion = nn.Linear(width * 2, width)
        self.position = nn.Parameter(torch.randn(1, config.max_context, width) * 0.02)
        # minute, hour, weekday, day of month, month: calendar only, no future prices.
        self.time_embeddings = nn.ModuleList(nn.Embedding(n, width) for n in [60, 24, 7, 32, 13])
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(CausalBlock(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(width)
        self.coarse_head = nn.Linear(width, 2 ** config.s1_bits)
        self.dependency = nn.MultiheadAttention(width, config.heads, batch_first=True)
        self.dependency_norm = nn.LayerNorm(width)
        self.fine_head = nn.Linear(width, 2 ** config.s2_bits)
        nn.init.normal_(self.emb_s1.weight, std=width ** -0.5)
        nn.init.normal_(self.emb_s2.weight, std=width ** -0.5)

        if config.auxiliary_features:
            count = len(config.auxiliary_features)
            self.register_buffer("auxiliary_center", torch.zeros(count))
            self.register_buffer("auxiliary_scale", torch.ones(count))
            self.auxiliary_projection = nn.Sequential(
                nn.Linear(count * 2, width), nn.GELU(), nn.Linear(width, width)
            )
            # Exact warm-start parity. The residual learns from the first update.
            nn.init.zeros_(self.auxiliary_projection[-1].weight)
            nn.init.zeros_(self.auxiliary_projection[-1].bias)

    def encode_history(self, s1, s2, stamps, auxiliary=None):
        if (s1.ndim != 2 or s1.shape != s2.shape or not 1 <= s1.shape[1] <= self.config.max_context
                or stamps.shape != (*s1.shape, 5)):
            raise ValueError("Token/context/calendar shape mismatch")
        if s1.dtype != torch.long or s2.dtype != torch.long:
            raise ValueError("Tokens must have int64 dtype")
        x = self.fusion(torch.cat([self.emb_s1(s1), self.emb_s2(s2)], -1)
                        * math.sqrt(self.config.width))
        x = x + self.position[:, :s1.shape[1]]
        for field, embedding in enumerate(self.time_embeddings):
            x = x + embedding(stamps[..., field].long())
        if self.config.auxiliary_features:
            if auxiliary is None or auxiliary.shape != (*s1.shape, len(self.config.auxiliary_features)):
                raise ValueError("Historical auxiliary features are required with matching shape")
            if torch.isinf(auxiliary).any():
                raise ValueError("Infinite auxiliary features")
            available = torch.isfinite(auxiliary)
            values = torch.where(available, auxiliary, self.auxiliary_center)
            values = ((values - self.auxiliary_center) / self.auxiliary_scale).clamp(-5, 5)
            x = x + self.auxiliary_projection(torch.cat([values, available.to(x.dtype)], -1))
        elif auxiliary is not None:
            raise ValueError("Baseline model does not accept auxiliary features")
        mask = causal_mask(s1.shape[1], x.device)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def decode_s1(self, s1, s2, stamps, auxiliary=None):
        hidden = self.encode_history(s1, s2, stamps, auxiliary)
        return self.coarse_head(hidden), hidden

    def decode_s2(self, hidden, next_s1):
        if next_s1.shape != hidden.shape[:2]:
            raise ValueError("Teacher-forced coarse target shape mismatch")
        z, _ = self.dependency(
            self.emb_s1(next_s1), hidden, hidden,
            attn_mask=causal_mask(hidden.shape[1], hidden.device), need_weights=False,
        )
        return self.fine_head(self.dependency_norm(hidden + z))

    def decode_last_s2(self, hidden, next_s1):
        # One query at the last position may attend to the entire known history.
        z, _ = self.dependency(self.emb_s1(next_s1[:, None]), hidden, hidden, need_weights=False)
        return self.fine_head(self.dependency_norm(hidden[:, -1:] + z))[:, 0]

    def forward(self, s1, s2, stamps, *, next_s1, auxiliary=None):
        coarse, hidden = self.decode_s1(s1, s2, stamps, auxiliary)
        return coarse, self.decode_s2(hidden, next_s1)

    def forecast_logits(self, s1, s2, stamps, next_s1, start, auxiliary=None):
        """Only project supervised positions; all historical keys remain visible."""
        hidden = self.encode_history(s1, s2, stamps, auxiliary)
        if not 0 <= start < hidden.shape[1] or next_s1.shape != s1.shape:
            raise ValueError("Invalid supervised forecast slice")
        z, _ = self.dependency(
            self.emb_s1(next_s1[:, start:]), hidden, hidden,
            attn_mask=causal_mask(hidden.shape[1], hidden.device)[start:], need_weights=False,
        )
        fine = self.fine_head(self.dependency_norm(hidden[:, start:] + z))
        return self.coarse_head(hidden[:, start:]), fine


def token_loss(model, s1, s2, stamps, future_valid, lookback, weights=None, *, history_auxiliary=None):
    """Next-token CE on forecast positions only; never score normalized history.

    All 60 history bars are known at the signal time and set normalization.
    Scoring historical positions would let their scale see later historical
    bars, so the first supervised position is the end of the history window.
    """
    if (lookback < 1 or s1.shape[1] != lookback + future_valid.shape[1]
            or future_valid.shape[0] != s1.shape[0] or future_valid.dtype != torch.bool):
        raise ValueError("Future mask and shifted token lengths differ")
    if (future_valid[:, 1:] & ~future_valid[:, :-1]).any():
        raise ValueError("Unknown future must mask the remaining suffix")
    coarse, fine = model.forecast_logits(
        s1[:, :-1], s2[:, :-1], stamps[:, :-1], s1[:, 1:], lookback - 1,
        auxiliary=forecast_auxiliary(history_auxiliary, lookback, s1.shape[1] - 1)
    )
    parts = []
    for logits, target in [(coarse, s1[:, lookback:]), (fine, s2[:, lookback:])]:
        error = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
        parts.append((error * future_valid).sum(1) / future_valid.sum(1).clamp_min(1))
    weight = future_valid.any(1).to(coarse.dtype)
    if weights is not None:
        if weights.shape != weight.shape or (weights < 0).any() or not torch.isfinite(weights).all():
            raise ValueError("Invalid date weights")
        weight = weight * weights
    if not bool(weight.sum() > 0):
        raise ValueError("No known forecast token targets")
    losses = [(p * weight).sum() / weight.sum() for p in parts]
    return (losses[0] + losses[1]) / 2, torch.stack(parts, -1)


def valid_bars(x):
    if x.shape[-1] != 6:
        raise ValueError("Expected OHLCVA fields")
    return (np.isfinite(x).all(-1) & (x[..., :4] > 0).all(-1)
            & (x[..., 4:] >= 0).all(-1)
            & (x[..., 1] >= x[..., :4].max(-1))
            & (x[..., 2] <= x[..., :4].min(-1)))


def normalized_history(raw, clip=5.0):
    """Signal-anchored adjustment and historical-only population statistics."""
    if raw.ndim != 3 or raw.shape[-1] != 7 or raw.shape[1] < 2:
        raise ValueError("Expected [sample, history, OHLCVA plus factor]")
    if (not valid_bars(raw[..., :6]).all() or not np.isfinite(raw[..., 6]).all()
            or (raw[..., 6] <= 0).any() or not np.isfinite(clip) or clip <= 0):
        raise ValueError("Invalid historical bars or factors")
    values = raw[..., :6].astype(np.float32, copy=True)
    values[..., :4] *= raw[..., 6:7] / raw[:, -1:, 6:7]
    mean = values.mean(1, keepdims=True)
    scale = values.std(1, keepdims=True) + 1e-5
    return np.clip((values - mean) / scale, -clip, clip), mean, scale


def normalize_training(raw, future, known, clip=5.0):
    history, mean, scale = normalized_history(raw, clip)
    if future.ndim != 3 or future.shape[0] != len(raw) or known.shape != future.shape[:2]:
        raise ValueError("Future shape mismatch")
    # A missing/corporate-action/invalid bar invalidates later autoregressive targets.
    valid = np.logical_and.accumulate(np.asarray(known, bool) & valid_bars(future), axis=1)
    values = np.where(valid[..., None], future, mean).astype(np.float32)
    normalized = np.clip((values - mean) / scale, -clip, clip)
    return np.concatenate([history, normalized], 1), valid, mean, scale


def load_tokenizer(bundle: Path, device="cpu"):
    """Load only the pinned official tokenizer, never the pretrained predictor."""
    from safetensors.torch import load_model

    bundle = Path(bundle).resolve()
    provenance = json.loads((bundle / "pretrained-provenance.json").read_text())
    required = ["upstream/model/__init__.py", "upstream/model/kronos.py",
                "upstream/model/module.py", "upstream/LICENSE",
                "tokenizer/config.json", "tokenizer/model.safetensors"]
    for name in required:
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or file_hash(path) != provenance["files"].get(name):
            raise ValueError(f"Tokenizer source/checkpoint hash mismatch: {name}")
    upstream = bundle / "upstream"
    existing = sys.modules.get("model")
    if existing and Path(existing.__file__).resolve().parent != upstream / "model":
        raise ValueError("Another package named model is already imported")
    sys.path.insert(0, str(upstream))
    api = importlib.import_module("model")
    config = json.loads((bundle / "tokenizer/config.json").read_text())
    tokenizer = api.KronosTokenizer(**config)
    load_model(tokenizer, str(bundle / "tokenizer/model.safetensors"), strict=True)
    return tokenizer.to(device).eval().requires_grad_(False)


def sample_logits(logits, generator, temperature=1.0, top_p=0.9, top_k=0):
    if (not np.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1
            or not isinstance(top_k, int) or top_k < 0):
        raise ValueError("Invalid sampling parameters")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite token logits")
    # CPU RNG is local and reproducible across save/reload; does not reset global RNG.
    logits = logits.detach().float().cpu() / temperature
    if top_k:
        limit = logits.topk(min(top_k, logits.shape[-1])).values[..., -1:]
        logits = logits.masked_fill(logits < limit, -torch.inf)
    ordered, indices = logits.sort(descending=True, dim=-1)
    remove = ordered.softmax(-1).cumsum(-1) > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    ordered = ordered.masked_fill(remove, -torch.inf)
    sampled = torch.multinomial(ordered.softmax(-1), 1, generator=generator)
    return indices.gather(-1, sampled).squeeze(-1)


@torch.inference_mode()
def generate_tokens(model, s1, s2, past_stamps, future_stamps, *, samples=32,
                    seed=17, temperature=1.0, top_p=0.9, top_k=0, history_auxiliary=None):
    """Retain sampled token pairs. No future targets enter this interface."""
    if (samples < 1 or s1.shape != s2.shape or s1.ndim != 2 or s1.shape[1] < 1
            or past_stamps.shape != (*s1.shape, 5) or future_stamps.ndim != 3
            or future_stamps.shape[0] != len(s1) or future_stamps.shape[2] != 5
            or future_stamps.shape[1] < 1 or s1.shape[1] > model.config.max_context):
        raise ValueError("Invalid generation input shapes")
    device = next(model.parameters()).device
    a, b = [v.to(device).repeat_interleave(samples, 0) for v in [s1, s2]]
    past, future = [v.to(device).repeat_interleave(samples, 0) for v in [past_stamps, future_stamps]]
    auxiliary = forecast_auxiliary(history_auxiliary, s1.shape[1], s1.shape[1])
    if auxiliary is not None:
        auxiliary = auxiliary.to(device).repeat_interleave(samples, 0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    previous_mode = model.training
    model.eval()
    try:
        for step in range(future.shape[1]):
            size = model.config.max_context
            coarse, hidden = model.decode_s1(a[:, -size:], b[:, -size:], past[:, -size:],
                auxiliary=None if auxiliary is None else auxiliary[:, -size:])
            first = sample_logits(coarse[:, -1], generator, temperature, top_p, top_k).to(device)
            fine = model.decode_last_s2(hidden, first)
            second = sample_logits(fine, generator, temperature, top_p, top_k).to(device)
            a, b = torch.cat([a, first[:, None]], 1), torch.cat([b, second[:, None]], 1)
            past = torch.cat([past, future[:, step:step + 1]], 1)
            if auxiliary is not None:
                auxiliary = torch.cat([auxiliary, torch.full_like(auxiliary[:, :1], torch.nan)], 1)
        shape = (len(s1), samples, a.shape[1])
        return a.reshape(shape), b.reshape(shape)
    finally:
        model.train(previous_mode)


@torch.inference_mode()
def decode_paths(tokenizer, token_pairs, mean, scale, horizon, max_context=512):
    a, b = token_pairs
    if (a.ndim != 3 or a.shape != b.shape or not 1 <= horizon <= a.shape[-1]
            or mean.shape != (len(a), 1, 6) or scale.shape != mean.shape):
        raise ValueError("Invalid token paths/scalers")
    device = next(tokenizer.parameters()).device
    tokenizer.eval()
    # Decode overlapping causal prefixes if generation exceeds the context window.
    flat_a, flat_b = [v.reshape(-1, v.shape[-1]).to(device) for v in [a, b]]
    if a.shape[-1] <= max_context:
        decoded = tokenizer.decode([flat_a, flat_b], half=True)[:, -horizon:]
    else:
        decoded = torch.stack([
            tokenizer.decode([flat_a[:, max(0, end-max_context):end],
                              flat_b[:, max(0, end-max_context):end]], half=True)[:, -1]
            for end in range(a.shape[-1]-horizon+1, a.shape[-1]+1)
        ], 1)
    paths = decoded.cpu().numpy().reshape(len(a), a.shape[1], horizon, 6)
    paths = paths * scale[:, None] + mean[:, None]
    # Keep impossible decoded paths unchanged and expose a mask; no fabricated repairs.
    return paths, valid_bars(paths).all(-1)


@torch.inference_mode()
def predict_paths(model, tokenizer, raw, past_stamps, future_stamps, *, clip=5.0,
                  history_auxiliary=None, **sampling):
    """Raw historical OHLCVA+factor -> [sample, path, future day, six fields]."""
    normalized, mean, scale = normalized_history(raw, clip)
    tokenizer.eval()
    device = next(tokenizer.parameters()).device
    s1, s2 = tokenizer.encode(torch.from_numpy(normalized).to(device), half=True)
    if (model.config.s1_bits != tokenizer.s1_bits or model.config.s2_bits != tokenizer.s2_bits):
        raise ValueError("Decoder vocabulary differs from tokenizer")
    pairs = generate_tokens(model, s1, s2, torch.as_tensor(past_stamps),
                            torch.as_tensor(future_stamps),
                            history_auxiliary=None if history_auxiliary is None else torch.as_tensor(
                                history_auxiliary, dtype=next(model.parameters()).dtype), **sampling)
    paths, valid = decode_paths(tokenizer, pairs, mean, scale, len(future_stamps[0]),
                                model.config.max_context)
    return {"paths": paths, "valid_paths": valid,
            "s1": pairs[0].cpu().numpy(), "s2": pairs[1].cpu().numpy()}


def checkpoint_payload(model, **metadata):
    return {"model_kind": "hierarchical_token_decoder_v1", "config": asdict(model.config),
            "state_dict": model.state_dict(), **metadata}


def restore_model(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("model_kind") != "hierarchical_token_decoder_v1":
        raise ValueError("Expected token decoder checkpoint; plan-head checkpoints are incompatible")
    model = TokenTransformer(TokenConfig(**saved["config"]))
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device), saved


def forecast_auxiliary(history, lookback, length):
    """Only accept historical features; future teacher-forced bars receive missing masks."""
    if history is None:
        return None
    if history.ndim != 3 or history.shape[1] != lookback or length < lookback:
        raise ValueError("Only lookback-length historical auxiliary features are accepted")
    if not history.is_floating_point():
        raise ValueError("Auxiliary features must be floating point")
    missing = history.new_full((len(history), length - lookback, history.shape[2]), torch.nan)
    return torch.cat([history, missing], 1)


def with_auxiliary(model, names, center, scale):
    """Expand an existing baseline checkpoint without changing initial predictions."""
    if model.config.auxiliary_features or not names:
        raise ValueError("Warm start expects a baseline model and nonempty feature names")
    device = next(model.parameters()).device
    expanded = TokenTransformer(replace(model.config, auxiliary_features=tuple(names))).to(device)
    result = expanded.load_state_dict(model.state_dict(), strict=False)
    if result.unexpected_keys or any(not k.startswith("auxiliary_") for k in result.missing_keys):
        raise ValueError("Unexpected checkpoint incompatibility")
    center, scale = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in [center, scale]]
    if (center.shape != (len(names),) or scale.shape != center.shape
            or not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Invalid training-only auxiliary normalization")
    expanded.auxiliary_center.copy_(center)
    expanded.auxiliary_scale.copy_(scale)
    return expanded.train(model.training)
