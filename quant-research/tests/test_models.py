import torch

from quant_research.models import TemporalRanker, seed_everything


def test_encoder_layers_are_independent_and_output_one_score_per_stock(config):
    seed_everything(17)
    model = TemporalRanker(25, 60, config["model"])
    assert not torch.equal(model.layers[0].linear1.weight, model.layers[1].linear1.weight)
    model.eval()
    x = torch.ones(3, 60, 25)
    assert model(x).shape == (3,)
    # No stock identifier embedding: identical stock histories produce identical scores.
    assert torch.allclose(model(x), model(x)[0].expand(3))


def test_lightgbm_uses_fresh_process_without_loading_torch(tmp_path):
    import json

    import numpy as np

    from quant_research.tree_runner import isolated_tree
    rng = np.random.default_rng(17)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    y = x[:, 0] * 0.1
    path = tmp_path / "tree.txt"
    scores = isolated_tree(x[:120], y[:120], x[120:160], y[120:160], x[160:], 17, path)
    assert scores.shape == (40,)
    assert np.isfinite(scores).all()
    assert json.loads(path.with_suffix(".runtime.json").read_text())["torch_loaded"] is False


def test_weighted_tree_selects_validation_daily_rank_ic(tmp_path):
    import json

    import numpy as np

    from quant_research.tree_runner import isolated_tree
    rng = np.random.default_rng(29)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    y = x[:, 0] * 0.1
    path = tmp_path / 'weighted-tree.txt'
    scores = isolated_tree(x[:120], y[:120], x[120:160], y[120:160], x[160:], 29,
                           path, np.ones(120), np.ones(40),
                           np.array(['2024-01-01'] * 20 + ['2024-01-02'] * 20))
    assert np.isfinite(scores).all()
    runtime = json.loads(path.with_suffix('.runtime.json').read_text())
    assert runtime['selection_metric'] == 'mean_daily_rank_ic'
    assert runtime['best_iteration'] > 0
    assert runtime['torch_loaded'] is False
