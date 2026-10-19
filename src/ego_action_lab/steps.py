"""Compiled full FP32 update; optimizer state captured explicitly."""

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from .flow import loss_fn


def make_optimizer(model):
    opt = optim.AdamW(
        learning_rate=1e-4, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01, bias_correction=True
    )
    opt.init(model.trainable_parameters())
    return opt


def make_step(model, opt, compiled=True):
    vg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def step(batch, noise, tau):
        (loss, parts), grads = vg(model, batch, noise, tau)
        grads, norm = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss, parts, norm

    if compiled:
        step = mx.compile(step, inputs=state, outputs=state)

    def execute(batch, noise, tau):
        result = step(batch, noise, tau)
        mx.eval(result, state)
        return result

    return execute
