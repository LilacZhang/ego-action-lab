"""Fast MLX objective and checkpoint invariants; full 131M smoke is a separate CLI run."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

from ego_action_lab.checkpoint import restore, save_resume
from ego_action_lab.flow import loss_components, path

pytestmark = pytest.mark.mlx


def test_masked_flow_ignores_invalid_targets():
    state = mx.ones((1, 48))
    action = mx.full((1, 16, 48), 3.0)
    mask = mx.ones((1, 16, 48), dtype=mx.bool_)
    mask[:, :, 0] = False
    b = {
        "state": state,
        "state_mask": mx.ones_like(state).astype(mx.bool_),
        "action": action,
        "action_mask": mask,
    }
    noise = mx.ones_like(action)
    tau = mx.array([0.5])
    noisy, target = path(b, noise, tau)
    assert np.allclose(np.array(noisy[:, :, 1:]), 1.5)
    b["action"] = mx.where(mask, action, mx.full(action.shape, float("nan")))
    n2, t2 = path(b, noise, tau)
    np.testing.assert_array_equal(np.array(noisy), np.array(n2))
    np.testing.assert_array_equal(np.array(target), np.array(t2))
    loss, _ = loss_components(target, target, mask)
    assert float(loss) == 0


def test_resume_preserves_optimizer_and_rng(tmp_path):
    mx.random.seed(7)
    model = nn.Linear(3, 2)
    opt = optim.AdamW(0.001)
    opt.init(model.parameters())
    rng = np.random.default_rng(47)

    def step(m, o, r):
        x = mx.array(r.normal(size=(2, 3)).astype("float32"))
        value, grads = nn.value_and_grad(m, lambda m: mx.mean(m(x) ** 2))(m)
        o.update(m, grads)
        mx.eval(m.parameters(), o.state, value)

    step(model, opt, rng)
    save_resume(model, opt, tmp_path, {"step": 1, "rng": rng.bit_generator.state})
    step(model, opt, rng)
    expected = dict(tree_flatten(model.parameters()))
    fresh = nn.Linear(3, 2)
    resumed_opt = optim.AdamW(0.001)
    resumed_opt.init(fresh.parameters())
    meta = restore(fresh, resumed_opt, tmp_path)
    resumed_rng = np.random.default_rng()
    resumed_rng.bit_generator.state = meta["rng"]
    step(fresh, resumed_opt, resumed_rng)
    for key, value in tree_flatten(fresh.parameters()):
        np.testing.assert_array_equal(np.array(value), np.array(expected[key]))
    save_resume(fresh, resumed_opt, tmp_path, {"step": 2, "rng": resumed_rng.bit_generator.state})
    assert restore(fresh, resumed_opt, tmp_path)["step"] == 2


def test_corrupt_checkpoint_rejected(tmp_path):
    model = nn.Linear(2, 1)
    opt = optim.AdamW(0.001)
    opt.init(model.parameters())
    save_resume(model, opt, tmp_path, {"step": 0})
    p = tmp_path / "resume_0.safetensors"
    with p.open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        restore(model, opt, tmp_path)
