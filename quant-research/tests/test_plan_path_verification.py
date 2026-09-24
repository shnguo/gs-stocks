import importlib.util
from pathlib import Path

import numpy as np

from quant_research.plan_paths import path_heads
from quant_research.plan_value import HEADS


def test_scalar_audit_matches_float32_paths_and_abstentions():
    path = Path(__file__).resolve().parents[1]/'scripts/verify_plan_kronos.py'
    spec = importlib.util.spec_from_file_location('verify_paths_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(91)
    paths = np.zeros((3, 32, 5, 6), np.float32)
    paths[..., 0] = 100+rng.normal(0, 2, (3, 32, 5))
    paths[..., 3] = paths[..., 0]+rng.normal(0, 2, (3, 32, 5))
    paths[..., 1] = np.maximum(paths[..., 0], paths[..., 3])+rng.uniform(.1, 4, (3, 32, 5))
    paths[..., 2] = np.minimum(paths[..., 0], paths[..., 3])-rng.uniform(.1, 4, (3, 32, 5))
    paths[..., 4:] = 1000
    paths[1, 3:] = np.nan
    paths[2, :5, 0, 2] = 110
    predictions, _ = path_heads(paths, np.full(3, 100.))
    for i in range(3):
        independent, _ = module.scalar_heads(paths[i], 100., 8, 4)
        for h, name in enumerate(HEADS):
            np.testing.assert_allclose(independent[:, h], predictions[name][i], rtol=1e-10, atol=1e-12, equal_nan=True)
