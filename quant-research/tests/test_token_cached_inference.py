import numpy as np
import pytest
import torch

from quant_research.token_cached_inference import PredictorCache, generate_cached
from quant_research.token_transformer import (
    TokenConfig,
    TokenTransformer,
    generate_tokens,
    with_auxiliary,
)


@pytest.mark.parametrize('has_auxiliary', [False, True])
def test_cached_logits_and_samples_match_full_context(has_auxiliary):
    torch.manual_seed(19)
    model = TokenTransformer(TokenConfig(width=16, heads=2, layers=2, s1_bits=3, s2_bits=3))
    aux = None
    if has_auxiliary:
        model = with_auxiliary(model, ('a', 'b'), np.array([1., 2.]), np.array([2., 3.]))
        torch.nn.init.normal_(model.auxiliary_projection[-1].weight, std=.1)
        aux = torch.randn(2, 12, 2)
        aux[:, 2, 0] = float('nan')
    a, b = torch.randint(8, (2, 12)), torch.randint(8, (2, 12))
    past, future = torch.zeros(2, 12, 5, dtype=torch.long), torch.zeros(2, 5, 5, dtype=torch.long)
    expected = generate_tokens(model, a, b, past, future, samples=4, seed=123, top_p=1., history_auxiliary=aux)
    actual = generate_cached(model, a, b, past, future, samples=4, seed=123, top_p=1., history_auxiliary=aux)
    assert model.training  # Sampler restores mode.
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    model.eval()
    with torch.inference_mode():
        cache = PredictorCache(model, a, b, past, aux, 4)
        for step in range(5):
            aa, bb = [v.reshape(8, -1)[:, :12+step] for v in actual]
            stamp = torch.zeros(8, 12+step, 5, dtype=torch.long)
            history_aux = None if aux is None else torch.cat([aux.repeat_interleave(4, 0), torch.full((8, step, 2), float('nan'))], 1)
            logits, hidden = model.decode_s1(aa, bb, stamp, auxiliary=history_aux)
            torch.testing.assert_close(cache.coarse(), logits[:, -1], atol=3e-6, rtol=3e-6)
            first, second = [v.reshape(8, -1)[:, 12+step] for v in actual]
            torch.testing.assert_close(cache.fine(first), model.decode_last_s2(hidden, first), atol=3e-6, rtol=3e-6)
            if step < 4:
                cache.append(first, second, torch.zeros(8, 5, dtype=torch.long))


def test_cache_rejects_context_overflow():
    model = TokenTransformer(TokenConfig(width=8, heads=2, layers=1, max_context=8, s1_bits=2, s2_bits=2))
    with pytest.raises(ValueError, match='within context'):
        generate_cached(model, torch.zeros(1, 7, dtype=torch.long), torch.zeros(1, 7, dtype=torch.long),
                        torch.zeros(1, 7, 5, dtype=torch.long), torch.zeros(1, 5, 5, dtype=torch.long))
