"""Run context: the loaded dataset, the split, and the labeled draws, shared by features.py,
   train.py, evaluate_rep.py, and baseline_check.py.

   Split (plan.json -> "split"):
     {"source": "declared"}                     use the dataset's declared split (train/test)
     {"source": "random", "test_fraction": 0.2, "stratify_label": "<label>" | null}
     {"source": "random", "test_fraction": 0.2, "group_by": "<label>"}   whole groups per side
     both take "val_fraction" (carved from training, stratified like the test split) and "seed".
   The split is written to outputs/<d>/split.json and reused, so every script sees the same one.

   Held-out test samples are never used to fit anything: featurizers, models, and probes fit on
   `fit_idx` (train, or train+val) only. Validation samples are used for early stopping and for
   the baseline check.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import data_spec
import loaders
import plan_schema

OUTPUTS = data_spec.REPO / "outputs"


def _strata(values, idx):
    """Stratification keys for idx: the label, with missing and very small classes pooled."""
    keys = np.array(["__missing__" if values[i] is None else str(values[i]) for i in idx], dtype=object)
    uniq, counts = np.unique(keys, return_counts=True)
    small = set(uniq[counts < 4])
    return np.array(["__small__" if k in small else k for k in keys], dtype=object)


def make_split(data, spec):
    from sklearn.model_selection import train_test_split
    n = len(data["ids"])
    seed = int(spec.get("seed", 0))
    label = spec.get("stratify_label")
    values = data["labels"][label] if label else None
    all_idx = np.arange(n)
    if spec["source"] == "declared":
        if data["split"] is None:
            raise ValueError("split.source is 'declared' but the dataset declares no split")
        test = all_idx[data["split"] == "test"]
        train = all_idx[data["split"] != "test"]
    elif spec["source"] == "random" and spec.get("group_by"):
        # Whole groups (e.g. slides, patients, donors) go to one side, so near-duplicate samples from
        # one group never straddle the split. group_by names a declared label used as the group id.
        from sklearn.model_selection import GroupShuffleSplit
        groups = np.array([str(g) for g in data["labels"][spec["group_by"]]], dtype=object)
        gss = GroupShuffleSplit(n_splits=1, test_size=float(spec["test_fraction"]), random_state=seed)
        train, test = next(gss.split(all_idx, groups=groups))
        vf = float(spec.get("val_fraction", 0.15))
        gss = GroupShuffleSplit(n_splits=1, test_size=vf, random_state=seed)
        tr2, va2 = next(gss.split(train, groups=groups[train]))
        return {"train": np.sort(train[tr2]), "val": np.sort(train[va2]), "test": np.sort(test)}
    elif spec["source"] == "random":
        strat = _strata(values, all_idx) if values is not None else None
        train, test = train_test_split(all_idx, test_size=float(spec["test_fraction"]), random_state=seed, stratify=strat)
    else:
        raise ValueError(f"unknown split source {spec['source']}")
    vf = float(spec.get("val_fraction", 0.15))
    strat = _strata(values, train) if values is not None else None
    train, val = train_test_split(np.sort(train), test_size=vf, random_state=seed, stratify=strat)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


class Context:
    """Everything a script needs about one dataset and one plan's split."""

    def __init__(self, dataset, plan, objective=None):
        self.dataset = dataset
        self.plan = plan
        self.objective = objective or {}
        self.out = OUTPUTS / dataset
        self.data = loaders.load(dataset)
        self.ids = self.data["ids"]
        self.n = len(self.ids)
        self.spec = data_spec.parse(dataset)
        self.split = self._load_or_make_split(plan["split"])
        self.train, self.val, self.test = self.split["train"], self.split["val"], self.split["test"]
        self.fit_idx = np.sort(np.concatenate([self.train, self.val]))
        self.excluded = {k: set(v.get("classes", [])) for k, v in (plan.get("excluded_classes") or {}).items()}

    def _load_or_make_split(self, spec):
        path = self.out / "split.json"
        key = hashlib.sha1(json.dumps({k: v for k, v in spec.items() if k != "reason"}, sort_keys=True).encode()).hexdigest()[:12]
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("key") == key:
                pos = {s: i for i, s in enumerate(self.ids)}
                return {k: np.array(sorted(pos[s] for s in saved[k])) for k in ("train", "val", "test")}
        split = make_split(self.data, spec)
        self.out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "spec": spec, **{k: [str(self.ids[i]) for i in v] for k, v in split.items()},
                                    "sizes": {k: int(len(v)) for k, v in split.items()}}, indent=1) + "\n")
        return split

    def modality_kind(self, name):
        return self.data["modalities"][name]["kind"]

    def labels(self, name):
        """Label values with excluded classes set to None."""
        v = self.data["labels"][name].copy()
        ex = self.excluded.get(name, set())
        if ex:
            v[np.array([x in ex for x in v])] = None
        return v

    def labeled(self, name, idx):
        v = self.labels(name)
        return np.array([i for i in idx if v[i] is not None], dtype=int)

    def draw_labeled(self, name, budget, draw, pool=None):
        """Indices of a seeded, stratified labeled subset of `pool` (default: train+val) with `budget`
        samples per class ('all' = every labeled sample). Returns (idx, {class: n_taken}, unfilled)."""
        pool = self.fit_idx if pool is None else pool
        v = self.labels(name)
        lab = self.labeled(name, pool)
        if budget == plan_schema.BUDGET_ALL:
            cls, cnt = np.unique(v[lab].astype(str), return_counts=True)
            return lab, {str(c): int(k) for c, k in zip(cls, cnt)}, []
        rng = np.random.default_rng(10_000 + 97 * int(budget) + draw)
        out, taken, unfilled = [], {}, []
        for c in sorted(set(v[lab])):
            members = lab[v[lab] == c]
            k = min(int(budget), len(members))
            if k < budget:
                unfilled.append(str(c))
            out.append(rng.choice(members, size=k, replace=False))
            taken[str(c)] = int(k)
        return np.sort(np.concatenate(out)), taken, unfilled
