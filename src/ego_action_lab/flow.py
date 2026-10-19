"""Masked residual Flow Matching and five-step human sampling."""

import mlx.core as mx

from .model import translate_torch_key


def load_torch(model, path):
    weights = mx.load(str(path))
    model.load_weights([(translate_torch_key(k), v) for k, v in weights.items()], strict=True)
    mx.eval(model.parameters())


def convert(batch):
    return {k: mx.array(v) for k, v in batch.items()}


def reference(batch):
    return mx.where(batch["state_mask"], batch["state"], 0)[:, None]


def path(batch, noise, tau):
    mask = batch["action_mask"]
    safe = mx.where(mask, batch["action"] - reference(batch), 0)
    t = tau[:, None, None]
    return mx.where(mask, t * safe + (1 - t) * noise, noise), mx.where(mask, safe - noise, 0)


def loss_components(pred, target, mask):
    error = (mx.where(mask, pred, 0) - mx.where(mask, target, 0)) ** 2

    def group(lo, hi):
        n = mx.sum(mask[:, :, lo:hi], axis=(1, 2))
        return mx.sum(error[:, :, lo:hi], axis=(1, 2)) / mx.maximum(n, 1), n > 0

    def available(parts):
        v = mx.stack([a for a, p in parts], axis=-1)
        p = mx.stack([p for a, p in parts], axis=-1)
        n = mx.sum(p, axis=-1)
        return mx.sum(v * p, axis=-1) / mx.maximum(n, 1), n > 0

    pos = [group(s, s + 3) for s in (0, 24)]
    rot = [group(s + 3, s + 9) for s in (0, 24)]
    wrists = available(pos + rot)
    tips = available([group(s + 9 + j * 3, s + 12 + j * 3) for s in (0, 24) for j in range(5)])
    loss = mx.mean(available([wrists, tips])[0])

    def present_mean(part):
        return mx.sum(part[0] * part[1]) / mx.maximum(mx.sum(part[1]), 1)

    return loss, mx.stack(
        [
            present_mean(available(pos)),
            present_mean(available(rot)),
            present_mean(tips),
            mx.sum(error) / mx.maximum(mx.sum(mask), 1),
        ]
    )


def loss_fn(model, batch, noise, tau):
    noisy, target = path(batch, noise, tau)
    pred = model({**batch, "noisy": noisy, "tau": tau})
    return loss_components(pred, target, batch["action_mask"])


def sample(model, batch, noise, steps=5):
    action = noise
    for i in range(steps):
        tau = mx.full((len(noise),), i / steps, dtype=mx.float32)
        action = action + model({**batch, "noisy": action, "tau": tau}) / steps
    return action + reference(batch)
