"""Featurizers: turn one modality into a fixed feature matrix for every sample, fit only on the
   given fit rows (never on held-out test rows). Also builds cross-modal prediction targets.

     featurize(ctx, modality, encoder, params=None, fit_idx=None) -> (n_samples, d) float32
     target_matrix(ctx, modality, n_features, fit_idx=None)       -> ((n_samples, n_features), names)

   Encoders (plan_schema.COMPUTED_ENCODERS, plus "cache:<encoder_id>"):
     pca         counts: library-size normalize, log1p, top-variance features, centered PCA
     lsi         counts: TF-IDF on features detected in enough samples, truncated SVD, optionally
                 drop the first component (it tracks sequencing depth), standardize components
     pixels_pca  image: grayscale, resized to size x size, PCA
     tfidf_svd   text: TF-IDF (sublinear, min_df) + truncated SVD
     cache:<id>  frozen pretrained encoder embeddings from cache/<dataset>/<id>/ (not refit)
   Any encoder takes params.reduce_to = k: PCA to k components on top (fit on the fit rows).
   Results are cached in outputs/<d>/features/ keyed by encoder, params, and fit rows.
"""
import hashlib
import json

import numpy as np
import scipy.sparse as sp

import data_spec
import extract_embeddings
import loaders
import plan_schema

SEED = 0


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _lognorm(X):
    tot = np.asarray(X.sum(axis=1)).ravel()
    Xn = sp.diags(1e4 / np.maximum(tot, 1)) @ X
    return Xn.log1p().tocsr()


def _top_variance(Xn, fit_idx, n):
    F = Xn[fit_idx]
    mean = np.asarray(F.mean(axis=0)).ravel()
    var = np.asarray(F.multiply(F).mean(axis=0)).ravel() - mean ** 2
    return np.sort(np.argsort(-var)[:n])


def _pca(ctx, m, p, fit_idx):
    from sklearn.decomposition import PCA
    Xn = _lognorm(ctx.data["modalities"][m]["X"])
    cols = _top_variance(Xn, fit_idx, p["n_features"])
    D = Xn[:, cols].toarray()
    pca = PCA(n_components=p["n_components"], random_state=SEED).fit(D[fit_idx])
    return pca.transform(D)


def _lsi(ctx, m, p, fit_idx):
    from sklearn.decomposition import TruncatedSVD
    X = ctx.data["modalities"][m]["X"]
    det = np.bincount(X[fit_idx].indices, minlength=X.shape[1]) / len(fit_idx)
    X = X[:, det >= p["min_detected_fraction"]].tocsr()
    tf = sp.diags(1 / np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1)) @ X
    idf = len(fit_idx) / np.maximum(np.asarray(X[fit_idx].sum(axis=0)).ravel(), 1)
    T = (tf @ sp.diags(idf)).tocsr()
    T.data = np.log1p(T.data * 1e4)
    k = p["n_components"] + (1 if p.get("drop_first", True) else 0)
    svd = TruncatedSVD(n_components=k, random_state=SEED).fit(T[fit_idx])
    Z = svd.transform(T)
    if p.get("drop_first", True):
        Z = Z[:, 1:]
    mu, sd = Z[fit_idx].mean(0), Z[fit_idx].std(0) + 1e-8
    return (Z - mu) / sd


def _pixels(ctx, m, size):
    path = ctx.out / "features" / f"pixels_{m}_{size}.npy"
    if path.exists():
        return np.load(path)
    mod = ctx.data["modalities"][m]
    P = np.zeros((ctx.n, size * size), dtype=np.float32)
    for start, imgs in loaders.iter_images(mod["refs"], mod["column"], batch_size=512):
        for k, im in enumerate(imgs):
            P[start + k] = np.asarray(im.convert("L").resize((size, size)), dtype=np.float32).ravel() / 255.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, P)
    return P


def _pixels_pca(ctx, m, p, fit_idx):
    from sklearn.decomposition import PCA
    P = _pixels(ctx, m, p["size"])
    return PCA(n_components=p["n_components"], random_state=SEED).fit(P[fit_idx]).transform(P)


def _tfidf_svd(ctx, m, p, fit_idx):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = ctx.data["modalities"][m]["texts"]
    vec = TfidfVectorizer(sublinear_tf=True, min_df=p["min_df"]).fit([texts[i] for i in fit_idx])
    T = vec.transform(texts)
    return TruncatedSVD(n_components=p["n_components"], random_state=SEED).fit(T[fit_idx]).transform(T)


def _cached(ctx, m, enc_id):
    folder = data_spec.REPO / "cache" / ctx.dataset / enc_id
    ok, why = extract_embeddings.cache_status(folder, ctx.dataset)
    if not ok:
        raise ValueError(f"cache:{enc_id} not usable: {why}")
    import csv
    with open(folder / "ids.csv") as f:
        row_of = {r["pair_id"]: int(r["row"]) for r in csv.DictReader(f)}
    missing = [s for s in ctx.ids if s not in row_of]
    if missing:
        raise ValueError(f"cache:{enc_id} lacks {len(missing)} sample ids (e.g. {missing[:2]})")
    E = np.load(folder / f"{m}.npy")
    return E[[row_of[s] for s in ctx.ids]]


def available_encoders(ctx, modality):
    """Candidate encoders for a modality: computed ones for its kind plus ready caches."""
    kind = ctx.modality_kind(modality)
    out = sorted(plan_schema.COMPUTED_ENCODERS.get(kind, set()))
    for enc_id, enc in sorted(extract_embeddings.ENCODERS.items()):
        mods = extract_embeddings.applicable(ctx.spec, enc)
        if mods and modality in mods.values():
            ok, _ = extract_embeddings.cache_status(data_spec.REPO / "cache" / ctx.dataset / enc_id, ctx.dataset)
            if ok:
                out.append(f"cache:{enc_id}")
    return out


def featurize(ctx, modality, encoder, params=None, fit_idx=None):
    """params may include reduce_to: k, which applies a PCA to k components (fit on the fit rows) to
    the encoder's output. It gives a same-dimension control for a learned projection."""
    fit_idx = ctx.fit_idx if fit_idx is None else np.sort(np.asarray(fit_idx))
    params = dict(params or {})
    k = params.pop("reduce_to", None)
    Z = _featurize(ctx, modality, encoder, params, fit_idx)
    if k:
        from sklearn.decomposition import PCA
        Z = PCA(n_components=int(k), random_state=SEED).fit(Z[fit_idx]).transform(Z).astype(np.float32)
    return Z


def _featurize(ctx, modality, encoder, params, fit_idx):
    if encoder.startswith("cache:"):
        return _cached(ctx, modality, encoder.split(":", 1)[1]).astype(np.float32)
    kind = ctx.modality_kind(modality)
    if encoder not in plan_schema.COMPUTED_ENCODERS.get(kind, set()):
        raise ValueError(f"encoder {encoder} does not apply to {kind} modality {modality}")
    p = {**plan_schema.ENCODER_DEFAULT_PARAMS[encoder], **(params or {})}
    path = ctx.out / "features" / f"{modality}__{encoder}__{_key(p, fit_idx.tolist())}.npy"
    if path.exists():
        return np.load(path)
    fn = {"pca": _pca, "lsi": _lsi, "pixels_pca": _pixels_pca, "tfidf_svd": _tfidf_svd}[encoder]
    Z = np.asarray(fn(ctx, modality, p, fit_idx), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, Z)
    return Z


def target_matrix(ctx, modality, n_features=plan_schema.DEFAULT_TARGET_FEATURES, fit_idx=None):
    """Cross-modal prediction target: log-normalized values of the n highest-variance features
    (chosen on fit rows) of a counts modality. Returns (dense matrix, feature names)."""
    fit_idx = ctx.fit_idx if fit_idx is None else np.sort(np.asarray(fit_idx))
    mod = ctx.data["modalities"][modality]
    if mod["kind"] != "counts":
        raise ValueError(f"cross-modal prediction targets must be a counts modality, not {mod['kind']}")
    Xn = _lognorm(mod["X"])
    cols = _top_variance(Xn, fit_idx, n_features)
    return Xn[:, cols].toarray().astype(np.float32), mod["feature_names"][cols]
