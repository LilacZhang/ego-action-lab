import json

import numpy as np
import pytest

from ego_action_lab.data import Data
from ego_action_lab.demo import demo
from ego_action_lab.geometry import RawEpisode, normalize
from ego_action_lab.metrics import angle_degrees, paired_session_interval
from ego_action_lab.prepare import fit_normalization, prepare


@pytest.fixture
def toy(tmp_path):
    directory = tmp_path / "toy"
    demo(directory)
    return directory


def test_end_to_end_and_integrity(toy):
    train, val = Data(toy / "prepared", "train"), Data(toy / "prepared", "validation")
    assert len(train) == len(val) == 4
    b = train.batch(np.array([0, 1]))
    assert b["image"].shape == (2, 196, 384)
    assert b["action"].shape == (2, 16, 48)
    assert np.isfinite(b["action"]).all()
    assert not b["action_mask"].all()
    assert train.manifest["future_visual_conditions"] is False
    # Every cache mapping must refer to that window's current frame, never a future image.
    assert train.feature_rows.tolist() == [0, 1, 2, 3]
    assert train.normalization["fit_split"] == "train"
    path = toy / "prepared/normalization.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        Data(toy / "prepared", "train")


def test_session_and_duplicate_rejection(toy):
    path = toy / "episodes.json"
    records = json.loads(path.read_text())
    records["episodes"][1]["session"] = records["episodes"][0]["session"]
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="leakage"):
        prepare(path, toy / "bad", synthetic=True)
    records["episodes"][1]["session"] = "toy-validation"
    records["episodes"][1]["hdf5"] = records["episodes"][0]["hdf5"]
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="duplicate"):
        prepare(path, toy / "bad", synthetic=True)


def test_train_only_normalization_masks_unseen():
    m = np.ones((1, 17, 48), bool)
    m[:, :, 47] = False
    v = np.ones((1, 17, 48)) * 3
    v[:, :, 47] = np.nan
    norm = fit_normalization([{"values": v, "masks": m}])
    assert norm["mean"][0] == 3 and norm["supported"][47] is False
    z = normalize(v, m, norm["mean"], norm["scale"])
    assert np.isfinite(z).all() and (z == 0).all()


def test_camera_coordinates_and_masks(toy):
    raw = RawEpisode(toy / "train.hdf5")
    v, m = raw.window(np.arange(17))
    assert not m[0, 21:24].any()  # Unobserved little fingertip, wrist still valid.
    assert m[0, :9].all()
    original = v.copy()
    # Translate camera and all tracked joints equally: anchor coordinates are invariant.
    raw.camera[:, :3, 3] += np.array([1.0, 2.0, 3.0])
    for tf in raw.transform.values():
        tf[:, :3, 3] += np.array([1.0, 2.0, 3.0])
    translated, _ = raw.window(np.arange(17))
    np.testing.assert_allclose(translated, original, atol=1e-6, equal_nan=True)


def test_metrics_zero_baseline_and_rotation():
    rows = [{"session": "a", "sample_sse": 1.0, "hold_sse": 0.0, "count": 3}]
    assert paired_session_interval(rows)["sample_to_hold_rmse_ratio_ci95"] is None
    identity = np.array([1.0, 0, 0, 0, 1, 0])
    angle, valid = angle_degrees(identity, identity)
    assert valid and angle == 0
    angle, valid = angle_degrees(np.zeros(6), identity)
    assert not valid and angle == 180


def test_real_encoder_path_preserves_synthetic_label(toy, monkeypatch):
    # Avoid downloading/running a model: replace only the encoder factory.
    import sys
    from types import SimpleNamespace

    fake_mx = SimpleNamespace(array=np.asarray, eval=lambda x: None)
    fake_vision = SimpleNamespace(make_encoder=lambda p: lambda x: np.zeros((len(x), 196, 384)))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "ego_action_lab.vision", fake_vision)
    weights = toy / "fake-weights"
    weights.write_bytes(b"not a model")
    prepare(toy / "episodes.json", toy / "encoded", weights=weights)
    assert Data(toy / "encoded", "train").manifest["synthetic_data"] is True
