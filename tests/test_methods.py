"""Losses, metrics, and label-budget rules on small synthetic data (no real data needed).

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import context  # noqa: E402
import evaluate_rep  # noqa: E402
from models.nets import MultiTower, infonce, supcon, vicreg  # noqa: E402


def fake_ctx(labels, train, val, test):
    ctx = object.__new__(context.Context)
    ctx.data = {"labels": {"y": np.array(labels, dtype=object)}}
    ctx.excluded = {}
    ctx.train, ctx.val, ctx.test = np.array(train), np.array(val), np.array(test)
    ctx.fit_idx = np.sort(np.concatenate([ctx.train, ctx.val]))
    return ctx


class TestMetrics(unittest.TestCase):
    def test_foscttm_perfect_and_random(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(200, 16))
        m = evaluate_rep.matching_metrics(Z, Z, np.arange(200))
        self.assertEqual(m["foscttm"], 0.0)
        self.assertEqual(m["recall@1"], 1.0)
        r = evaluate_rep.matching_metrics(Z, rng.normal(size=(200, 16)), np.arange(200))
        self.assertAlmostEqual(r["foscttm"], 0.5, delta=0.06)

    def test_collapse_detects_low_rank(self):
        rng = np.random.default_rng(0)
        full = evaluate_rep.collapse(rng.normal(size=(500, 32)))
        low = evaluate_rep.collapse(rng.normal(size=(500, 2)) @ rng.normal(size=(2, 32)))
        self.assertGreater(full["effective_rank"], 20)
        self.assertLess(low["effective_rank"], 3)
        dead = np.c_[rng.normal(size=(500, 8)), np.zeros((500, 8))]
        self.assertEqual(evaluate_rep.collapse(dead)["low_variance_dim_fraction"], 0.5)


class TestLosses(unittest.TestCase):
    def test_losses_finite_and_aligned_pairs_score_better(self):
        torch.manual_seed(0)
        za = torch.randn(32, 8)
        far = torch.randn(32, 8)
        scale = torch.tensor(np.log(1 / 0.07), dtype=torch.float32)
        self.assertLess(float(infonce(za, za, scale)), float(infonce(za, far, scale)))
        self.assertLess(float(vicreg(za, za)), float(vicreg(za, far)))
        labels = torch.tensor([0, 1] * 16)
        self.assertTrue(torch.isfinite(supcon(za, far, labels, 0.1)))
        self.assertTrue(torch.isfinite(supcon(za, far, torch.full((32,), -1), 0.1)))

    def test_modality_dropout_keeps_one(self):
        net = MultiTower({"a": 4, "b": 4}, "linear", [], 3, 0.0)
        net.train()
        zs = {"a": torch.ones(64, 3), "b": torch.ones(64, 3) * 3}
        z = net.fused(zs, modality_dropout=0.9)
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(set(np.unique(z.numpy().round(4))) <= {1.0, 2.0, 3.0})


class TestLabelBudget(unittest.TestCase):
    def test_draw_respects_budget_and_pool(self):
        labels = ["a"] * 50 + ["b"] * 50 + ["c"] * 3 + [None] * 10
        n = len(labels)
        ctx = fake_ctx(labels, train=range(0, n, 2), val=[], test=range(1, n, 2))
        idx, taken, unfilled = ctx.draw_labeled("y", 5, 0, pool=ctx.train)
        self.assertTrue(set(idx) <= set(ctx.train))
        self.assertEqual(taken["a"], 5)
        self.assertEqual(unfilled, ["c"])
        again, _, _ = ctx.draw_labeled("y", 5, 0, pool=ctx.train)
        self.assertTrue(np.array_equal(idx, again))  # same draw for training and evaluation

    def test_group_split_keeps_groups_together(self):
        n = 120
        groups = np.array([f"g{i // 10}" for i in range(n)], dtype=object)
        data = {"ids": np.arange(n).astype(str), "split": None, "labels": {"slide": groups}}
        sp = context.make_split(data, {"source": "random", "test_fraction": 0.25, "group_by": "slide", "val_fraction": 0.2, "seed": 0})
        sides = [set(groups[sp[k]]) for k in ("train", "val", "test")]
        self.assertFalse(sides[0] & sides[1] or sides[0] & sides[2] or sides[1] & sides[2])
        self.assertEqual(sum(len(sp[k]) for k in sp), n)

    def test_excluded_class_is_missing(self):
        ctx = fake_ctx(["a", "b", "c"], train=[0, 1, 2], val=[], test=[])
        ctx.excluded = {"y": {"c"}}
        self.assertIsNone(ctx.labels("y")[2])


if __name__ == "__main__":
    unittest.main()
