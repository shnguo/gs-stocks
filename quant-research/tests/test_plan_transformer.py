"""Run Torch checks in isolation from other native model runtimes."""
import subprocess
import sys


def test_plan_training_masks_scaler_and_checkpoint(tmp_path):
    code = r'''
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from quant_research.plan_transformer import (
    PlanTransformer, auxiliary_weights, fit_input_scaler, plan_loss, predict, target_arrays, train,
)

torch.set_num_threads(1)
n = 8
filled = np.ones((n, 12))
filled[0] = np.nan
filled[1] = 0
net = np.broadcast_to(np.linspace(-.03, .04, n)[:, None], (n, 12)).copy()
net[:2] = np.nan
outcomes = dict(filled=filled, conditional_net_return=net,
    conditional_loss=np.where(np.isfinite(net), net < 0, np.nan),
    conditional_downside=np.where(np.isfinite(net), np.maximum(-net, 0), np.nan))
dates = np.array(['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'])
y, w, center, scale, prior = target_arrays(outcomes, dates)
assert not w[0].any()
assert not w[1, :, 1:].any()
for h in range(5):
    np.testing.assert_allclose(w[dates == 'a', :, h].sum(0), w[dates == 'b', :, h].sum(0))
np.testing.assert_allclose((y*w).sum(0)[:, [2, 4]], 0, atol=1e-6)
# Missing labels receive no gradient even with extreme logits.
z = torch.zeros((n, 12, 5), requires_grad=True)
loss = plan_loss(z, torch.from_numpy(y), torch.from_numpy(w))
loss.backward()
assert torch.isfinite(z.grad).all()
assert (z.grad[torch.from_numpy(w == 0)] == 0).all()

rng = np.random.default_rng(17)
values = rng.normal(size=(2, 70, 27)).astype(np.float32)
coords = np.array([[0, 59], [0, 60], [1, 59]])
m, s, count = fit_input_scaler(values, coords)
assert count == 121
changed = values.copy()
changed[:, 61:] = 1e6
changed[1, 60] = 1e6
m2, s2, _ = fit_input_scaler(changed, coords)
np.testing.assert_array_equal(m, m2)
np.testing.assert_array_equal(s, s2)
prices = rng.normal(0, .02, (n, 5, 4)).astype(np.float32)
prices[0] = np.nan
pw = auxiliary_weights(prices, dates)
assert pw[0] == 0
np.testing.assert_allclose(pw[dates == 'a'].sum(), pw[dates == 'b'].sum())
model = PlanTransformer(center, scale, prior, width=8)
z, q = model(torch.from_numpy(values[[0, 1], :60]))
assert q.shape == (2, 5, 4, 3)
assert (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all()
p = model.decode(z)
assert ((p[..., [0, 1, 3]] >= 0) & (p[..., [0, 1, 3]] <= 1)).all()
loss = plan_loss(z, torch.from_numpy(y[2:4]), torch.from_numpy(w[2:4]))
loss.backward()
assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)

rows = pd.DataFrame(dict(stock_index=np.arange(n)%2, date_index=np.full(n, 60), date=dates))
cfg = dict(seed=17, device='cpu', width=8, learning_rate=.001, batch_size=4,
    max_epochs=2, min_epochs=2, patience=1, min_delta=1e9, price_auxiliary_weight=.1)
out = Path(sys.argv[1])
model, mean, std = train(values, rows, {'targets': prices}, outcomes,
    rows, {'targets': prices}, outcomes, out, cfg)
state = torch.load(out/'transformer.pt', weights_only=True)
assert state['selected_epoch'] == 1
assert json.loads((out/'training-result.json').read_text())['epochs_completed'] == 2
restored = PlanTransformer(center, scale, prior, width=8)
restored.load_state_dict(state['state_dict'])
a, _ = predict(model, values, rows, mean, std)
b, _ = predict(restored, values, rows, mean, std)
for h in a:
    np.testing.assert_array_equal(a[h], b[h])
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True, timeout=90)
