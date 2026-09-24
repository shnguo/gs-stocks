"""Past-window Transformer with plan means, risks and auxiliary price quantiles."""

import time

import numpy as np
import torch
from torch import nn

from .plan_value import HEADS, head_targets
from .price_transformer import PriceTransformer, masked_pinball
from .storage import utc_now, write_json


class PlanTransformer(PriceTransformer):
    def __init__(self, center, scale, prior, width=48, context_features=0, dropout=0.1):
        super().__init__(features=27, lookback=60, width=width)
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        for layer in self.layers:
            layer.self_attn.dropout = dropout
            for module in layer.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
        self.plan_output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 60))
        nn.init.normal_(self.plan_output[-1].weight, std=0.001)
        nn.init.zeros_(self.plan_output[-1].bias)
        self.register_buffer("target_center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("target_scale", torch.as_tensor(scale, dtype=torch.float32))
        with torch.no_grad():
            bias = self.plan_output[-1].bias.view(12, 5)
            for h in [0, 1, 3]:
                p = np.clip(prior[:, h], 1e-4, 1 - 1e-4)
                bias[:, h] = torch.from_numpy(np.log(p / (1 - p)).astype(np.float32))
        self.context_features = context_features
        self.context_encoder = (
            nn.Sequential(
                nn.Linear(context_features * 2, width), nn.GELU(), nn.Linear(width, width)
            )
            if context_features
            else None
        )

    def forward(self, x, context=None):
        x = self.input(x) + self.position[:, : x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        hidden = x[:, -1]
        if self.context_encoder is not None:
            if context is None or context.shape != (len(x), self.context_features * 2):
                raise ValueError("Missing or incorrectly shaped signal-date context")
            hidden = hidden + self.context_encoder(context)
        elif context is not None:
            raise ValueError("Unexpected context for sequence-only model")
        plan = self.plan_output(hidden).reshape(-1, 12, 5)
        z = self.output(hidden).reshape(-1, 5, 4, 3)
        widths = nn.functional.softplus(z[..., 1:])
        return plan, torch.stack(
            [z[..., 0] - widths[..., 0], z[..., 0], z[..., 0] + widths[..., 1]], -1
        )

    def decode(self, z):
        result = z * self.target_scale + self.target_center
        result = result.clone()
        for h in [0, 1, 3]:
            result[..., h] = torch.sigmoid(z[..., h])
        result[..., 4] = result[..., 4].clamp_min(0)
        return result


def target_arrays(outcomes, dates, center=None, scale=None):
    ys = head_targets(outcomes)
    y = np.stack([ys[h] for h in HEADS], axis=-1).astype(np.float32)
    weight = np.zeros_like(y)
    fit = center is None
    if fit:
        center, scale = np.zeros((12, 5), np.float32), np.ones((12, 5), np.float32)
    prior = np.zeros((12, 5), np.float32)
    dates = np.asarray(dates)
    for plan in range(12):
        for head in range(5):
            good = np.isfinite(y[:, plan, head])
            unique, counts = np.unique(dates[good], return_counts=True)
            if not len(unique):
                raise ValueError("No observed target dates")
            weights = {d: len(y) / len(unique) / c for d, c in zip(unique, counts)}
            weight[good, plan, head] = [weights[d] for d in dates[good]]
            w = weight[good, plan, head].astype(float)
            values = y[good, plan, head].astype(float)
            mean = np.average(values, weights=w)
            prior[plan, head] = mean
            if fit and head in [2, 4]:
                center[plan, head] = mean
                scale[plan, head] = max(np.sqrt(np.average((values - mean) ** 2, weights=w)), 0.005)
    normalized = np.where(np.isfinite(y), (y - center) / scale, 0).astype(np.float32)
    return normalized, weight, center, scale, prior


def fit_input_scaler(values, coordinates):
    mask = np.zeros(values.shape[:2], bool)
    for stock, date in coordinates:
        if date < 59 or date >= values.shape[1]:
            raise ValueError("Insufficient historical context")
        mask[stock, date - 59 : date + 1] = True
    total, square, count = np.zeros(27), np.zeros(27), 0
    for stock in np.flatnonzero(mask.any(axis=1)):
        data = np.asarray(values[stock, mask[stock]], float)
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite training context")
        total += data.sum(axis=0)
        square += (data * data).sum(axis=0)
        count += len(data)
    if not count:
        raise ValueError("Empty training context")
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean * mean, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-4).astype(np.float32), count


def window_batch(values, coordinates, ids, mean, scale):
    c = coordinates[ids]
    raw = values[c[:, 0, None], c[:, 1, None] + np.arange(-59, 1)]
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite historical context")
    return np.clip((raw - mean) / scale, -10, 10).astype(np.float32)


def plan_loss_elements(z, y, weights):
    loss = (z - y) ** 2
    for h in [0, 1, 3]:
        loss[..., h] = nn.functional.binary_cross_entropy_with_logits(
            z[..., h], y[..., h], reduction="none"
        )
    return loss * weights


def plan_loss_components(z, y, weights):
    return plan_loss_elements(z, y, weights).mean(dim=(0, 1))


def plan_loss(z, y, weights):
    return plan_loss_elements(z, y, weights).mean()


def auxiliary_weights(targets, dates):
    known = np.isfinite(targets).any((1, 2))
    dates = np.asarray(dates)
    unique, counts = np.unique(dates[known], return_counts=True)
    result = np.zeros(len(dates), np.float32)
    if len(unique):
        for date, count in zip(unique, counts):
            result[(dates == date) & known] = len(dates) / len(unique) / count
    return result


def fit_context_scaler(context):
    context = np.asarray(context, dtype=np.float64)
    if context.ndim != 2 or np.isinf(context).any():
        raise ValueError("Invalid training context")
    valid = np.isfinite(context)
    count = valid.sum(axis=0)
    mean = np.divide(
        np.where(valid, context, 0).sum(axis=0),
        count,
        out=np.zeros(context.shape[1]),
        where=count > 0,
    )
    variance = np.divide(
        np.where(valid, (context - mean) ** 2, 0).sum(axis=0),
        count,
        out=np.ones(context.shape[1]),
        where=count > 0,
    )
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-4).astype(np.float32), count


def normalize_context(context, mean, scale):
    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 2 or context.shape[1:] != mean.shape or np.isinf(context).any():
        raise ValueError("Invalid signal-date context")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid training-only context scaler")
    valid = np.isfinite(context)
    values = np.where(valid, np.clip((context - mean) / scale, -10, 10), 0)
    return np.concatenate([values, (~valid).astype(np.float32)], axis=1).astype(np.float32)


def predict(
    model, values, rows, mean, scale, batch=512, context=None, context_mean=None, context_scale=None
):
    model.eval()
    if bool(model.context_features) != (context is not None):
        raise ValueError("Model/context configuration mismatch")
    if context is not None and len(context) != len(rows):
        raise ValueError("Context cohort length differs")
    coordinates = rows[["stock_index", "date_index"]].to_numpy(int)
    result = np.empty((len(rows), 12, 5), np.float32)
    prices = np.empty((len(rows), 5, 4, 3), np.float32)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            ids = np.arange(start, min(start + batch, len(rows)))
            x = torch.from_numpy(window_batch(values, coordinates, ids, mean, scale)).to(device)
            extra = (
                torch.from_numpy(normalize_context(context[ids], context_mean, context_scale)).to(
                    device
                )
                if context is not None
                else None
            )
            z, q = model(x, extra)
            result[ids], prices[ids] = model.decode(z).cpu().numpy(), q.cpu().numpy()
    return {h: result[:, :, j] for j, h in enumerate(HEADS)}, prices


def train(
    values,
    train_rows,
    train_data,
    train_outcomes,
    select_rows,
    select_data,
    select_outcomes,
    out,
    config,
    train_context=None,
    select_context=None,
):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = torch.device(config["device"])
    rng = np.random.default_rng(config["seed"])
    coordinates = train_rows[["stock_index", "date_index"]].to_numpy(int)
    mean, std, count = fit_input_scaler(values, coordinates)
    y, weights, center, scale, prior = target_arrays(train_outcomes, train_rows.date)
    price_weights = auxiliary_weights(train_data["targets"], train_rows.date)
    _, val_weights, _, _, _ = target_arrays(select_outcomes, select_rows.date, center, scale)
    val_raw = head_targets(select_outcomes)[HEADS[2]]
    context_features = config.get("context_features", 0)
    context_stats = {}
    context_mean = context_std = None
    if context_features:
        if (
            train_context is None
            or select_context is None
            or train_context.shape != (len(train_rows), context_features)
            or select_context.shape != (len(select_rows), context_features)
        ):
            raise ValueError("Context cohorts differ from original plan rows")
        context_mean, context_std, context_count = fit_context_scaler(train_context)
        context_stats = dict(
            context_mean=context_mean, context_scale=context_std, context_count=context_count
        )
    elif train_context is not None or select_context is not None:
        raise ValueError("Unexpected context in baseline configuration")
    model = PlanTransformer(
        center, scale, prior, config["width"], context_features, config.get("dropout", 0.1)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    np.savez(
        out / "scalers.npz",
        mean=mean,
        scale=std,
        target_center=center,
        target_scale=scale,
        prior=prior,
        **context_stats,
    )
    write_json(
        out / "training-shape.json",
        {
            "train_rows": len(train_rows),
            "selection_rows": len(select_rows),
            "unique_training_tokens": count,
            "parameters": sum(p.numel() for p in model.parameters()),
            "head_weight_totals": weights.sum(axis=0).tolist(),
            "scaler_fit": "unique training-window tokens only",
            "context_features": context_features,
            "context_scaler_fit": "training signal rows only; finite values; explicit missing masks",
            "device": str(device),
            "parameter_device": str(next(model.parameters()).device),
        },
    )
    batch, best, best_epoch, bad, log = config["batch_size"], float("inf"), 0, 0, []
    started = time.monotonic()
    diagnostics = config.get("diagnostics", False)
    probe_ids = np.unique(
        np.linspace(0, len(train_rows) - 1, min(1024, len(train_rows)), dtype=int)
    )

    def diagnostic_scores():
        forecasts, _ = predict(
            model,
            values,
            select_rows,
            mean,
            std,
            batch,
            context=select_context,
            context_mean=context_mean,
            context_scale=context_std,
        )
        errors = np.where(np.isfinite(val_raw), forecasts[HEADS[2]] - val_raw, 0)
        score = float(np.mean(errors**2 * val_weights[:, :, 2]))
        details = {}
        if diagnostics:
            probe, _ = predict(
                model,
                values,
                train_rows.iloc[probe_ids],
                mean,
                std,
                batch,
                context=train_context[probe_ids] if train_context is not None else None,
                context_mean=context_mean,
                context_scale=context_std,
            )
            raw = head_targets(train_outcomes)[HEADS[2]][probe_ids]
            delta = np.where(np.isfinite(raw), probe[HEADS[2]] - raw, 0)
            details = dict(
                training_probe_net_mse=float(np.mean(delta**2 * weights[probe_ids, :, 2])),
                selection_plan_net_mse=np.mean(errors**2 * val_weights[:, :, 2], axis=0).tolist(),
            )
        return score, details

    if diagnostics:
        initial, details = diagnostic_scores()
        write_json(
            out / "initial-diagnostics.json",
            dict(
                epoch=0,
                selection_net_mse=initial,
                **details,
                probe_ids=probe_ids.tolist(),
                selection_constant_prior_net_mse=float(
                    np.mean(
                        np.where(np.isfinite(val_raw), prior[:, 2] - val_raw, 0) ** 2
                        * val_weights[:, :, 2]
                    )
                ),
                checkpoint_eligible=False,
            ),
        )
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        shuffled = rng.permutation(len(train_rows))
        epoch_totals = torch.zeros(6, device=device)
        epoch_lr = optimizer.param_groups[0]["lr"]
        for step, start in enumerate(range(0, len(shuffled), batch), 1):
            ids = shuffled[start : start + batch]
            x = torch.from_numpy(window_batch(values, coordinates, ids, mean, std)).to(device)
            extra = (
                torch.from_numpy(
                    normalize_context(train_context[ids], context_mean, context_std)
                ).to(device)
                if train_context is not None
                else None
            )
            z, prices = model(x, extra)
            elements = plan_loss_elements(
                z, torch.from_numpy(y[ids]).to(device), torch.from_numpy(weights[ids]).to(device)
            )
            loss = elements.mean()
            price_loss, _ = masked_pinball(
                prices, torch.from_numpy(train_data["targets"][ids].astype(np.float32)).to(device)
            )
            weighted_price = (price_loss * torch.from_numpy(price_weights[ids]).to(device)).mean()
            loss = loss + config["price_auxiliary_weight"] * weighted_price / 0.05
            if diagnostics:
                components = elements.mean(dim=(0, 1))
                epoch_totals += torch.cat(
                    [components.detach(), weighted_price.detach()[None]]
                ) * len(ids)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite optimization objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                write_json(
                    out / "progress.json",
                    {
                        "status": "training",
                        "epoch": epoch,
                        "step": step,
                        "steps_per_epoch": int(np.ceil(len(train_rows) / batch)),
                        "loss": float(loss.detach()),
                        "updated_at": utc_now(),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
        score, details = diagnostic_scores()
        if not np.isfinite(score):
            raise ValueError("Nonfinite selection score")
        scheduler.step(score)
        improved = score < best - config["min_delta"]
        if improved:
            best, best_epoch, bad = score, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": state,
                    "config": config,
                    "selected_epoch": epoch,
                    "selection_net_mse": score,
                },
                out / "transformer.pt",
            )
        else:
            bad += 1
        if diagnostics:
            losses = (epoch_totals / len(train_rows)).cpu().numpy()
            details.update(
                training_head_losses=dict(zip(HEADS, losses[:5].astype(float).tolist())),
                training_price_pinball=float(losses[5]),
                training_total_objective=float(
                    losses[:5].mean() + config["price_auxiliary_weight"] * losses[5] / 0.05
                ),
                epoch_learning_rate=epoch_lr,
            )
        log.append(
            {
                "epoch": epoch,
                "selection_net_mse": score,
                "best_epoch": best_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic() - started,
                **details,
            }
        )
        write_json(out / "training-log.json", log)
        print(log[-1], flush=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "rng_state": torch.get_rng_state(),
                "numpy_rng_state": rng.bit_generator.state,
            },
            out / "last-checkpoint.pt",
        )
        if epoch >= config["min_epochs"] and bad >= config["patience"]:
            break
    checkpoint = torch.load(out / "transformer.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    write_json(
        out / "training-result.json",
        {
            "selected_epoch": best_epoch,
            "epochs_completed": len(log),
            "selection_net_mse": best,
            "budget_limited": len(log) == config["max_epochs"] and bad < config["patience"],
            "stop_reason": "early_stopping" if bad >= config["patience"] else "epoch_cap",
        },
    )
    return model, mean, std
