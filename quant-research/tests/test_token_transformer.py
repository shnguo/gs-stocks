import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from quant_research.kronos_ranker import timestamps
from quant_research.token_transformer import (
    TokenConfig,
    TokenTransformer,
    checkpoint_payload,
    decode_paths,
    generate_tokens,
    load_tokenizer,
    normalize_training,
    normalized_history,
    predict_paths,
    restore_model,
    sample_logits,
    token_loss,
)


def tiny():
    torch.manual_seed(41)
    return TokenTransformer(TokenConfig(width=16, heads=4, layers=2,
                                        max_context=16, s1_bits=3, s2_bits=3, dropout=0))


def token_inputs(n=2, length=8):
    a, b = torch.randint(0, 8, (n, length)), torch.randint(0, 8, (n, length))
    stamps = torch.from_numpy(timestamps(pd.bdate_range("2025-01-01", periods=length)))
    return a, b, stamps[None].expand(n, -1, -1)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_both_heads_are_causal_in_training_and_evaluation(training, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    model = tiny().to(device).train(training)
    a, b, stamps = [x.to(device) for x in token_inputs()]
    target = a.roll(-1, dims=1)
    before = model(a, b, stamps, next_s1=target)
    aa, bb, tt, ss = a.clone(), b.clone(), target.clone(), stamps.clone()
    aa[:, 4:], bb[:, 4:], tt[:, 4:], ss[:, 4:, 3] = 7-aa[:, 4:], 7-bb[:, 4:], 7-tt[:, 4:], 28
    after = model(aa, bb, ss, next_s1=tt)
    for x, y in zip(before, after):
        torch.testing.assert_close(x[:, :4], y[:, :4], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(before[0][:, 4:], after[0][:, 4:])
    # Same-day target conditions the fine head, never the coarse head.
    different_target = model(a, b, stamps, next_s1=7-target)
    torch.testing.assert_close(before[0], different_target[0])
    assert not torch.allclose(before[1], different_target[1])


def test_last_step_dependency_matches_causal_teacher_forcing():
    model = tiny().eval()
    a, b, stamps = token_inputs()
    _, hidden = model.decode_s1(a, b, stamps)
    fine = model.decode_s2(hidden, b)
    torch.testing.assert_close(fine[:, -1], model.decode_last_s2(hidden, b[:, -1]),
                               atol=1e-6, rtol=1e-5)


def test_forecast_slice_preserves_full_head_logits_and_gradients():
    model = tiny().double()
    a, b, stamps = token_inputs()
    full = model(a, b, stamps, next_s1=b)
    full_loss = sum(x[:, 4:].square().mean() for x in full)
    full_loss.backward()
    gradients = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad()
    sliced = model.forecast_logits(a, b, stamps, b, 4)
    for actual, expected in zip(sliced, full):
        torch.testing.assert_close(actual, expected[:, 4:], atol=1e-12, rtol=1e-10)
    sum(x.square().mean() for x in sliced).backward()
    for k, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[k], atol=1e-12, rtol=1e-9)


def test_future_targets_are_shifted_and_unknown_suffix_has_no_loss_or_gradient():
    model = tiny()
    a, b, stamps = token_inputs()
    known = torch.tensor([[True, False, False], [True, True, False]])
    loss, _ = token_loss(model, a, b, stamps, known, 5)
    loss.backward()
    gradient = model.coarse_head.weight.grad.clone()
    aa, bb = a.clone(), b.clone()
    for i, valid in enumerate(known):
        aa[i, 5:][~valid] = 7-aa[i, 5:][~valid]
        bb[i, 5:][~valid] = 7-bb[i, 5:][~valid]
    model.zero_grad()
    altered, _ = token_loss(model, aa, bb, stamps, known, 5)
    altered.backward()
    torch.testing.assert_close(loss, altered)
    torch.testing.assert_close(gradient, model.coarse_head.weight.grad)
    assert model.fine_head.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="remaining suffix"):
        token_loss(model, a, b, stamps, torch.tensor([[True, False, True]]*2), 5)
    with pytest.raises(ValueError, match="No known"):
        token_loss(model, a, b, stamps, torch.zeros((2, 3), dtype=torch.bool), 5)


def raw_inputs(n=2, length=6):
    raw = np.ones((n, length, 7), dtype=np.float32)
    close = 10 + np.arange(length, dtype=np.float32) * 0.1
    raw[..., :4] = close[None, :, None]
    raw[..., 1] += 0.1
    raw[..., 2] -= 0.1
    raw[..., 4], raw[..., 5] = 10000, 100000
    return raw


def test_normalization_uses_only_history_and_masks_unknown_prefix():
    raw = raw_inputs()
    future = raw[:, :3, :6].copy()
    x, valid, mean, scale = normalize_training(raw, future, np.ones((2, 3), bool))
    changed = future.copy()
    changed *= 3
    y, _, mean2, scale2 = normalize_training(raw, changed, np.ones((2, 3), bool))
    np.testing.assert_array_equal(x[:, :6], y[:, :6])
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(scale, scale2)
    future[:, 1, 0] = np.nan
    z, valid, _, _ = normalize_training(raw, future, np.ones((2, 3), bool))
    np.testing.assert_array_equal(valid, [[True, False, False]]*2)
    assert np.isfinite(z).all()
    np.testing.assert_array_equal(z[:, 7:], 0)
    # Historical adjustment can vary, but its anchor is the last known factor.
    split = raw.copy()
    split[:, :3, :4] *= 2
    split[:, 3:, 6] = 2
    np.testing.assert_allclose(normalized_history(split)[0], normalized_history(raw)[0], atol=1e-5)


def test_training_improves_known_token_sequence_and_checkpoint_is_reproducible(tmp_path):
    torch.set_num_threads(2)
    model = tiny()
    a, b, stamps = token_inputs()
    a[:], b[:] = 3, 6
    valid = torch.ones((2, 3), dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    first = float(token_loss(model, a, b, stamps, valid, 5)[0].detach())
    for _ in range(12):
        optimizer.zero_grad()
        loss, _ = token_loss(model, a, b, stamps, valid, 5)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < first * 0.5
    model.eval()
    before = model(a, b, stamps, next_s1=a)
    path = tmp_path / "model.pt"
    torch.save(checkpoint_payload(model), path)
    restored, _ = restore_model(path)
    restored.eval()
    for expected, actual in zip(before, restored(a, b, stamps, next_s1=a)):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    torch.save({"state_dict": model.state_dict()}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="incompatible"):
        restore_model(tmp_path / "old.pt")


def test_sampling_retains_paths_uses_local_seed_and_rolls_context():
    model = tiny().train()
    a, b, stamps = token_inputs(length=16)
    rng = torch.get_rng_state().clone()
    args = (model, a, b, stamps, stamps[:, :5])
    result = generate_tokens(*args, samples=4, seed=9)
    replay = generate_tokens(*args, samples=4, seed=9)
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for x, y, history in zip(result, replay, [a, b]):
        assert x.shape == (2, 4, 21)
        torch.testing.assert_close(x, y)
        torch.testing.assert_close(x[:, :, :16], history[:, None].expand(-1, 4, -1))
    assert not torch.equal(result[0][:, 0, -5:], result[0][:, 1, -5:])
    generator = torch.Generator().manual_seed(2)
    selected = sample_logits(torch.tensor([[0., 10., 1.]]*8), generator, top_k=1)
    assert (selected == 1).all()
    with pytest.raises(ValueError, match="sampling"):
        sample_logits(torch.ones(1, 8), generator, temperature=0)


class DummyTokenizer(nn.Module):
    s1_bits = s2_bits = 3

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)

    def encode(self, values, half):
        assert half and not self.training
        return ((values[..., 0] + 4).long().clamp(0, 7),
                (values[..., 3] + 4).long().clamp(0, 7))

    def decode(self, tokens, half):
        assert half and not self.training
        a, b = tokens
        x = (a.float() - 4) / 5
        return torch.stack([x, x+0.3, x-0.3, x, b.float(), b.float()], -1)


def test_end_to_end_ohlcva_outputs_do_not_require_plan_labels():
    model, tokenizer = tiny(), DummyTokenizer()
    raw = raw_inputs()
    stamps = timestamps(pd.bdate_range("2025-01-01", periods=11))[None].repeat(2, 0)
    result = predict_paths(model, tokenizer, raw, stamps[:, :6], stamps[:, 6:], samples=8)
    assert result["paths"].shape == (2, 8, 5, 6)
    assert result["valid_paths"].shape == (2, 8)
    assert result["s1"].shape == (2, 8, 11)
    assert set(result) == {"paths", "valid_paths", "s1", "s2"}
    broken = torch.zeros((2, 3, 5), dtype=torch.long)
    paths, valid = decode_paths(tokenizer, [broken, broken],
        np.zeros((2, 1, 6)), np.ones((2, 1, 6)), 5)
    assert not valid.any()  # Negative price stays negative, never silently repaired.
    assert (paths[..., 0] < 0).all()


def runner_module():
    spec = importlib.util.spec_from_file_location("token_runner", Path(__file__).parents[1] / "scripts/token_transformer_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_and_input_sampling_preserve_temporal_boundaries():
    runner = runner_module()
    cfg = json.loads((Path(__file__).parents[1] / "configs/token-transformer-v1.json").read_text())
    runner.validate_protocol(cfg)
    cfg["rows_per_date"] = 1
    frame = pd.DataFrame({"instrument_id": ["a", "b", "a", "b"],
                           "date": ["2024-06-01"]*2 + ["2025-01-02"]*2})
    chosen = runner.select_rows(frame, cfg, "train")
    assert len(chosen) == 1 and chosen.iloc[0].date == "2025-01-02"
    assert chosen.iloc[0].source_row in [2, 3]
    cfg["sealed_holdout_start"] = "2025-08-08"
    with pytest.raises(ValueError, match="boundary"):
        runner.validate_protocol(cfg)


def test_official_tokenizer_encoding_is_causal_and_frozen():
    bundle = Path(__file__).parents[1] / "artifacts/kronos-comparison-20260910-v1"
    if not (bundle / "pretrained-provenance.json").exists() or not importlib.util.find_spec("safetensors"):
        pytest.skip("Pinned tokenizer integration requires the isolated Kronos environment")
    tokenizer = load_tokenizer(bundle)
    assert not tokenizer.training and not any(p.requires_grad for p in tokenizer.parameters())
    raw = raw_inputs()
    normalized, _, _, _ = normalize_training(raw, raw[:, :3, :6], np.ones((2, 3), bool))
    with torch.inference_mode():
        full = tokenizer.encode(torch.from_numpy(normalized), half=True)
        prefix = tokenizer.encode(torch.from_numpy(normalized[:, :6]), half=True)
    for f, p in zip(full, prefix):
        torch.testing.assert_close(f[:, :6], p, atol=0, rtol=0)
    model = TokenTransformer(TokenConfig(width=16, layers=1, heads=4, max_context=32))
    stamps = timestamps(pd.bdate_range("2025-01-01", periods=11))[None].repeat(2, 0)
    result = predict_paths(model, tokenizer, raw, stamps[:, :6], stamps[:, 6:], samples=2)
    assert result["paths"].shape == (2, 2, 5, 6)
