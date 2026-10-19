"""Portable refit helpers; these tests never access evaluation labels."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shared_recipe", ROOT / "training/train_shared.py")
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def graph(n):
    return dict(node=torch.zeros(n, 35), point=torch.zeros(n, 24, 11),
        point_mask=torch.ones(n, 24, dtype=torch.bool), atom=torch.zeros(n, 24, 8, 19),
        edge=torch.zeros(n, 20), edge_index=torch.stack((torch.arange(n), torch.arange(n))),
        fit_y=torch.zeros(n, 4), mask=torch.ones(n, 4, dtype=torch.bool))


def test_batch_offsets_and_finite_gradients():
    batch = recipe.batch_graphs([graph(2), graph(3)])
    assert torch.equal(batch["edge_index"], torch.stack((torch.arange(5), torch.arange(5))))
    model = recipe.SharedWaterModel()
    loss = recipe.objective(model, batch, [1., 2., 3., 4.])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_objective_is_task_averaged():
    batch = graph(2)
    prediction = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 4.]])
    expected = (.75 + 4 + 9 + 16) / 4  # SmoothL1(beta=.5), then three MSEs.
    assert recipe.objective(lambda g: prediction, batch, [1.] * 4).item() == pytest.approx(expected)
    batch["mask"][:] = False
    with pytest.raises(ValueError, match="Empty training mask"):
        recipe.objective(lambda g: prediction, batch, [1.] * 4)


def test_fit_membership_and_frozen_schedule():
    protocol = json.loads((ROOT / "training/shared_protocol.json").read_text())
    roles = json.loads((ROOT / "training/development_split.json").read_text())
    assert len(roles) == 800
    assert len(protocol["fit_ids"]) == 800
    assert set(protocol["fit_ids"]) == set(roles)
    assert protocol["feature_version"] == "terminal_carboxylate_v2"
    assert {s: v["steps"] for s, v in protocol["seeds"].items()} == {"17": 2100, "29": 2200, "43": 2100}
    for seed in protocol["seeds"].values():
        assert set(map(int, seed["learning_rate_at_block_end"])) == set(range(100, seed["steps"] + 1, 100))


def test_corrupt_cache_rejected_before_load(tmp_path):
    path = tmp_path / "not_a_cache.pt"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum mismatch"):
        recipe.load_training(path, {"cache": {"sha256": "wrong"}})
