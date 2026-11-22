"""Representations for methods with no trainable weights (plan_schema.BASELINE_METHODS).

     build(ctx, entry) -> (rep, info)   rep: {modality: (n_samples, d)}, info: what was chosen and why

   single_modality_features, frozen_probe, linear_supervised: each input modality's features as-is
     (computed featurizer or cached frozen encoder). linear_supervised adds PLS / logistic regression
     at evaluation time.
   linear_alignment: CCA between two modalities' standardized features (each reduced to at most 50
     PCA components first, fit on the training split). Fit on the training split for
     each candidate number of components, pick the one with the lowest validation FOSCTTM, then refit
     on train+val with that number.
   linear_alignment honors `seeds`: seed 0 fits on all training rows, seed s > 0 on a seeded 90%
     subsample, so the baseline gets a spread too.
   A frozen_probe whose inputs all come from one cached encoder that embeds every one of them was
   pretrained jointly on those modalities, so its embeddings share a space (zero-shot matching).
"""
import numpy as np

import evaluate_rep
import features
import plan_schema


def _standardize(ctx, Z):
    mu, sd = Z[ctx.fit_idx].mean(0), Z[ctx.fit_idx].std(0) + 1e-6
    return (Z - mu) / sd


CCA_MAX_INPUT_DIM = 50


def cca_inputs(Z, rows, max_dim=CCA_MAX_INPUT_DIM):
    """Standardize on `rows`, then reduce to at most max_dim PCA components fit on `rows`. CCA on
    hundreds of correlated dimensions is ill-conditioned (sklearn's solver fails to converge on
    512- and 768-dim frozen embeddings), so both sides are reduced first."""
    from sklearn.decomposition import PCA
    Z = (Z - Z[rows].mean(0)) / (Z[rows].std(0) + 1e-6)
    if Z.shape[1] <= max_dim:
        return Z
    return PCA(n_components=max_dim, random_state=0).fit(Z[rows]).transform(Z)


BOOTSTRAP_FRACTION = 0.9


def build(ctx, entry, seed=0):
    method = entry["method"]
    rep = {m: features.featurize(ctx, m, s["encoder"], s.get("params")) for m, s in entry["inputs"].items()}
    info = {"inputs": {m: s["encoder"] for m, s in entry["inputs"].items()}}
    if method in ("single_modality_features", "linear_supervised"):
        return rep, {**info, "shared_space": False}
    if method == "frozen_probe":
        encs = {s["encoder"] for s in entry["inputs"].values()}
        joint = len(rep) > 1 and len(encs) == 1 and next(iter(encs)).startswith("cache:")
        return rep, {**info, "shared_space": joint}
    if method == "linear_alignment":
        from sklearn.cross_decomposition import CCA
        a, b = list(rep)
        tr, fit = ctx.train, ctx.fit_idx
        if seed:  # seed > 0: fit on a seeded subsample, so repeated fits show how much CCA itself varies
            rng = np.random.default_rng(seed)
            tr = np.sort(rng.choice(ctx.train, int(BOOTSTRAP_FRACTION * len(ctx.train)), replace=False))
            fit = np.sort(np.concatenate([tr, ctx.val]))
        A, B = cca_inputs(rep[a], tr), cca_inputs(rep[b], tr)
        hp = entry.get("hyperparameters", {})
        cands = hp.get("n_components_candidates", plan_schema.CCA_COMPONENT_CANDIDATES)
        cands = [k for k in cands if k <= min(A.shape[1], B.shape[1])]
        scores = {}
        for k in cands:
            cca = CCA(n_components=k, max_iter=2000).fit(A[tr], B[tr])
            Pa, Pb = cca.transform(A, B)
            scores[k] = evaluate_rep.matching_metrics(Pa, Pb, ctx.val)["foscttm"]
        best = min(scores, key=scores.get)
        cca = CCA(n_components=best, max_iter=2000).fit(A[fit], B[fit])
        Pa, Pb = cca.transform(A, B)
        return {a: np.asarray(Pa, np.float32), b: np.asarray(Pb, np.float32)}, {
            **info, "shared_space": True, "n_components": int(best), "seed": seed,
            "training_rows": int(len(tr)),
            "validation_foscttm_by_components": {str(k): v for k, v in scores.items()}}
    raise ValueError(f"{method} is not a baseline method")
