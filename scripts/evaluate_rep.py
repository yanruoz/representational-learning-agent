"""Evaluate one representation (a trained model's embeddings, or a baseline's features) on the
   objective's tasks. Held-out test samples only; inputs only from objective.inference_modalities.

     evaluate(ctx, rep, method, key, only_budget=None) -> metrics dict (also written to metrics/<key>.json)

   rep: {modality: (n_samples, d) array}, rows aligned with the dataset's sample ids.

   Label prediction (each label task and secondary task):
     For each per-class budget, seeded stratified draws of labeled TRAINING samples (the same draws
     train.py gives label-using models), then a kNN classifier (cosine, k = min(10, budget)) and a
     linear probe (logistic regression on standardized embeddings). Macro-F1, accuracy, per-class F1
     on labeled test samples, mean and SD over draws, next to the majority-class baseline.
   Clustering: KMeans with k = number of test classes; ARI and NMI against the label.
   Cross-modal prediction: kNN averaging (k = plan_schema.KNN_PREDICT_K, cosine, reference = train+val
     samples embedded from the inference modalities) of the target modality's log-normalized values;
     for linear_supervised, PLS regression from the inference features instead. Per-feature Pearson
     correlation over test samples: median, quartiles, mean, and the per-feature values (.npy).
   Retrieval (shared-space methods with a second modality in rep): test queries from the inference
     modality, gallery = the other modality's test embeddings; recall@1/5/10, median rank, chance.
   Collapse: per embedding, standard deviation per dimension, the share of dimensions with SD below
     1% of the largest, and effective rank (exp of the entropy of normalized singular values), on
     test rows.
"""
import json

import numpy as np

import features
import plan_schema

LOW_VAR_FRACTION = 0.01


def _unit(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def inference_embedding(ctx, rep):
    mods = ctx.objective["inference_modalities"]
    missing = [m for m in mods if m not in rep]
    if missing:
        raise ValueError(f"representation lacks inference modalities {missing}")
    return np.concatenate([_unit(rep[m]) for m in mods], axis=1) if len(mods) > 1 else rep[mods[0]]


def collapse(Z):
    Z = np.asarray(Z, dtype=np.float64)
    sd = Z.std(0)
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return {"dim": int(Z.shape[1]), "effective_rank": round(float(np.exp(-(p * np.log(p)).sum())), 2),
            "low_variance_dim_fraction": round(float(np.mean(sd < LOW_VAR_FRACTION * sd.max())), 4) if sd.max() > 0 else 1.0,
            "sd_per_dim": {"min": round(float(sd.min()), 5), "median": round(float(np.median(sd)), 5), "max": round(float(sd.max()), 5)}}


def _probe(Ztr, ytr, Zte, budget):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    k = min(10, len(ytr) if budget == plan_schema.BUDGET_ALL else int(budget))
    knn = KNeighborsClassifier(n_neighbors=max(1, k), metric="cosine").fit(Ztr, ytr).predict(Zte)
    sc = StandardScaler().fit(Ztr)
    lin = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Ztr), ytr).predict(sc.transform(Zte))
    return {"knn": knn, "linear": lin}


def label_metrics(ctx, Z, label, budgets, only_budget=None):
    from sklearn.metrics import accuracy_score, f1_score
    vals = ctx.labels(label)
    te = ctx.labeled(label, ctx.test)
    yte = vals[te].astype(str)
    classes = sorted(set(yte))
    out = {"label": label, "n_test": int(len(te)), "classes": classes, "budgets": {}}
    for b in budgets:
        if only_budget is not None and b != only_budget:
            continue
        n_draws = 1 if only_budget is not None else plan_schema.draws_for_budget(b)
        rows = {"knn": [], "linear": [], "majority": []}
        per_class = {"knn": [], "linear": []}
        info = None
        for d in range(n_draws):
            idx, taken, unfilled = ctx.draw_labeled(label, b, d, pool=ctx.train)
            info = {"per_class": taken, "unfilled_classes": unfilled, "n_labeled": int(len(idx))}
            ytr = vals[idx].astype(str)
            preds = _probe(Z[idx], ytr, Z[te], b)
            maj = np.full(len(te), max(set(ytr), key=list(ytr).count))
            for name, yp in [*preds.items(), ("majority", maj)]:
                rows[name].append({"macro_f1": f1_score(yte, yp, average="macro", labels=classes, zero_division=0),
                                   "accuracy": accuracy_score(yte, yp)})
                if name in per_class:
                    per_class[name].append(f1_score(yte, yp, average=None, labels=classes, zero_division=0))
        res = {"draws": n_draws, **info}
        for name, r in rows.items():
            for metric in ("macro_f1", "accuracy"):
                v = np.array([x[metric] for x in r])
                res[f"{name}_{metric}"] = {"mean": round(float(v.mean()), 4), "sd": round(float(v.std()), 4)}
        for name, pc in per_class.items():
            res[f"{name}_per_class_f1"] = dict(zip(classes, np.round(np.mean(pc, axis=0), 4).tolist()))
        out["budgets"][str(b)] = res
    return out


def clustering_metrics(ctx, Z, label, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    te = ctx.labeled(label, ctx.test)
    y = ctx.labels(label)[te].astype(str)
    k = len(set(y))
    pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(_unit(Z[te]))
    return {"label": label, "k": k, "ari": round(float(adjusted_rand_score(y, pred)), 4),
            "nmi": round(float(normalized_mutual_info_score(y, pred)), 4)}


def _pearson_per_feature(P, T):
    P = P - P.mean(0)
    T = T - T.mean(0)
    den = np.sqrt((P ** 2).sum(0) * (T ** 2).sum(0))
    r = np.full(P.shape[1], np.nan)
    ok = den > 0
    r[ok] = (P * T).sum(0)[ok] / den[ok]
    return r


def prediction_metrics(ctx, Z, task, method, rep_features=None, key=None):
    from sklearn.neighbors import NearestNeighbors
    T, names = features.target_matrix(ctx, task["target"], int(task.get("target_features", plan_schema.DEFAULT_TARGET_FEATURES)))
    ref, te = ctx.fit_idx, ctx.test
    if method == "linear_supervised":
        from sklearn.cross_decomposition import PLSRegression
        Xs = rep_features if rep_features is not None else Z
        pls = PLSRegression(n_components=plan_schema.PLS_COMPONENTS, scale=False).fit(Xs[ref], T[ref])
        P, how = pls.predict(Xs[te]), f"PLS regression ({plan_schema.PLS_COMPONENTS} components)"
    else:
        nn_ = NearestNeighbors(n_neighbors=plan_schema.KNN_PREDICT_K, metric="cosine").fit(Z[ref])
        _, nb = nn_.kneighbors(Z[te])
        P, how = T[ref][nb].mean(1), f"kNN averaging (k={plan_schema.KNN_PREDICT_K}, cosine)"
    r = _pearson_per_feature(P, T[te])
    if key:
        (ctx.out / "metrics").mkdir(parents=True, exist_ok=True)
        np.save(ctx.out / "metrics" / f"{key}__{task['target']}_pearson.npy", r)
    v = r[~np.isnan(r)]
    return {"source": task["source"], "target": task["target"], "how": how, "n_features": int(len(r)),
            "n_features_constant_in_test": int(np.isnan(r).sum()),
            "pearson_median": round(float(np.median(v)), 4), "pearson_mean": round(float(v.mean()), 4),
            "pearson_p25": round(float(np.percentile(v, 25)), 4), "pearson_p75": round(float(np.percentile(v, 75)), 4),
            "fraction_features_r_above_0.3": round(float(np.mean(v > 0.3)), 4)}


def retrieval_metrics(ctx, rep, query, gallery):
    te = ctx.test
    Q, G = _unit(rep[query][te]), _unit(rep[gallery][te])
    S = Q @ G.T
    rank = (S > S[np.arange(len(te)), np.arange(len(te))][:, None]).sum(1) + 1
    n = len(te)
    return {"query": query, "gallery": gallery, "n": int(n),
            **{f"recall@{k}": round(float(np.mean(rank <= k)), 4) for k in (1, 5, 10)},
            **{f"chance_recall@{k}": round(k / n, 4) for k in (1, 5, 10)},
            "median_rank": float(np.median(rank))}


def evaluate(ctx, rep, method, key, only_budget=None, shared_space=None, rep_features=None):
    obj = ctx.objective
    Z = inference_embedding(ctx, rep)
    out = {"key": key, "method": method, "inference_modalities": obj["inference_modalities"], "tasks": {}}
    shared = plan_schema.METHODS[method]["shared_space"] if shared_space is None else shared_space
    for t in obj.get("tasks", []):
        name = t["task"]
        if name == "label_prediction":
            budgets = t.get("label_budgets_per_class", [plan_schema.BUDGET_ALL])
            out["tasks"].setdefault("label_prediction", {})[t["label"]] = label_metrics(ctx, Z, t["label"], budgets, only_budget)
            out["tasks"].setdefault("clustering", {})[t["label"]] = clustering_metrics(ctx, Z, t["label"])
        elif name == "cross_modal_prediction":
            out["tasks"].setdefault("cross_modal_prediction", {})[t["target"]] = prediction_metrics(ctx, Z, t, method, rep_features, key)
        elif name == "cross_modal_retrieval" or (name == "joint_embedding" and shared):
            pass
    if shared:
        others = [m for m in rep if m not in obj["inference_modalities"]]
        for g in others:
            for q in obj["inference_modalities"]:
                out["tasks"].setdefault("cross_modal_retrieval", {})[f"{q}->{g}"] = retrieval_metrics(ctx, rep, q, g)
    if only_budget is None:
        for t in obj.get("secondary_tasks", []):
            if t["task"] == "label_prediction":
                out["tasks"].setdefault("secondary_label_prediction", {})[t["label"]] = label_metrics(
                    ctx, Z, t["label"], t.get("label_budgets_per_class", [plan_schema.BUDGET_ALL]))
    out["collapse"] = {m: collapse(rep[m][ctx.test]) for m in rep}
    (ctx.out / "metrics").mkdir(parents=True, exist_ok=True)
    (ctx.out / "metrics" / f"{key}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out
