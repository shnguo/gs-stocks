"""Small past-window encoder trained on masked OHLC quantile loss."""
import copy
import time

import numpy as np
import torch
from torch import nn

from .storage import utc_now, write_json


class PriceTransformer(nn.Module):
    def __init__(self, features=27, lookback=60, width=48):
        super().__init__()
        self.input = nn.Linear(features, width)
        self.position = nn.Parameter(torch.randn(1, lookback, width) * .02)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            width, 4, width * 2, .1, batch_first=True, norm_first=True,
            activation="gelu") for _ in range(2)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 60))
        nn.init.normal_(self.output[-1].weight, std=.001)
        with torch.no_grad():
            bias = self.output[-1].bias.view(5, 4, 3)
            bias[..., 0] = 0
            bias[..., 1:] = -3.5

    def forward(self, x):
        x = self.input(x) + self.position[:, :x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        z = self.output(x[:, -1]).reshape(-1, 5, 4, 3)
        median = z[..., 0]
        widths = nn.functional.softplus(z[..., 1:])
        return torch.stack((median - widths[..., 0], median, median + widths[..., 1]), -1)


def masked_pinball(prediction, target):
    valid = torch.isfinite(target)
    error = torch.where(valid, target, torch.zeros_like(target))[..., None] - prediction
    q = prediction.new_tensor([.1, .5, .9])
    loss = torch.maximum(q * error, (q - 1) * error)
    loss = torch.where(valid[..., None], loss, torch.zeros_like(loss))
    per_row = loss.sum((1, 2, 3)) / (valid.sum((1, 2)).clamp_min(1) * 3)
    return per_row, valid.any((1, 2))


def windows(values, rows, ids, mean, scale):
    coords = rows.iloc[ids][["stock_index", "date_index"]].to_numpy(int)
    if (coords[:, 1] < 59).any():
        raise ValueError("Insufficient past window")
    raw = values[coords[:, 0, None], coords[:, 1, None] + np.arange(-59, 1)]
    if not np.isfinite(raw).all():
        raise ValueError("Invalid historical feature window")
    return np.clip((raw - mean) / scale, -10, 10).astype(np.float32)


def predict(model, values, rows, mean, scale, batch=512):
    model.eval()
    out = np.empty((len(rows), 5, 4, 3), np.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            ids = np.arange(start, min(start + batch, len(rows)))
            x = torch.from_numpy(windows(values, rows, ids, mean, scale)).to(device)
            out[ids] = model(x).cpu().numpy()
    return out


def train(values, train_rows, train_data, selection_rows, selection_data, out, config):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    rng = np.random.default_rng(config["seed"])
    device = torch.device(config["device"])
    model = PriceTransformer().to(device)
    mean = train_data["x"].mean(0).astype(np.float32)
    scale = np.maximum(train_data["x"].std(0), 1e-4).astype(np.float32)
    np.savez(out / 'transformer-scaler.npz', mean=mean, scale=scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    y = train_data["targets"].astype(np.float32)
    known = np.isfinite(y).any((1, 2))
    counts = train_rows.loc[known].groupby('date').size()
    weights = train_rows.date.map(1 / counts).fillna(0).to_numpy(np.float32)
    weights *= known.sum() / weights.sum()
    best_loss, best_state, bad, log = float('inf'), None, 0, []
    started = time.monotonic()
    batch = config['batch_size']
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        shuffled = rng.permutation(np.flatnonzero(known))
        for step, start in enumerate(range(0, len(shuffled), batch), 1):
            ids = shuffled[start:start + batch]
            x = torch.from_numpy(windows(values, train_rows, ids, mean, scale)).to(device)
            target = torch.from_numpy(y[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = masked_pinball(model(x), target)
            loss = (loss * torch.from_numpy(weights[ids]).to(device)).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 1 or step % 40 == 0:
                progress = dict(epoch=epoch, step=step, steps_per_epoch=int(np.ceil(len(shuffled)/batch)),
                                loss=float(loss.detach()), elapsed_seconds=time.monotonic()-started,
                                updated_at=utc_now(), status='training')
                write_json(out / 'progress.json', progress)
                print(progress, flush=True)
        pred = predict(model, values, selection_rows, mean, scale, batch)
        error = selection_data['targets'][..., None] - pred
        q = np.array([.1, .5, .9])
        daily = []
        for day in selection_rows.date.unique():
            e = error[selection_rows.date.eq(day)]
            daily.append(float(np.nanmean(np.maximum(q * e, (q-1)*e))))
        score = float(np.mean(daily))
        log.append(dict(epoch=epoch, selection_pinball=score, elapsed_seconds=time.monotonic()-started))
        print(log[-1], flush=True)
        if score < best_loss:
            best_loss, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            torch.save({'state_dict': best_state, 'config': config, 'selected_epoch': epoch,
                        'selection_pinball': score, 'mean': torch.from_numpy(mean),
                        'scale': torch.from_numpy(scale)}, out / 'transformer.pt')
        else:
            bad += 1
        write_json(out / 'training-log.json', log)
        if bad >= 2:
            break
    model.load_state_dict(best_state)
    return model, mean, scale, log
