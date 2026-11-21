"""Dataset loading. One loader for every dataset: what to load is declared in the '## Loading'
   section of data/<name>/DATA_DESCRIPTION.md (parsed by data_spec.py). Nothing here or downstream
   branches on which dataset is loaded.

   load(dataset) returns a dict:
     ids         np.ndarray[str] (n,)       sample ids, shared by every modality (rows are paired)
     split       np.ndarray[str] (n,)|None  declared split per sample, if the dataset has one
     modalities  {name: {...}}              per modality, by kind:
                   counts: X (scipy CSR float32, n x features), feature_names (np.ndarray[str])
                   text:   texts (list[str])
                   image:  refs (list of (file, row_in_file)); read with iter_images / read_images
     labels      {name: np.ndarray[object]} None marks a missing label
     other_columns  sample-level columns present in the files but not loaded (never model inputs)

   Images are not decoded at load time (they can be several GB); use iter_images or read_images.
   This module never touches the network: fetching is scripts/fetch_data.py's job.
"""
import io
from pathlib import Path

import numpy as np

import data_spec

DATA_ROOT = data_spec.DATA_ROOT


def _label_values(raw, label_spec):
    """Plain column -> values as-is (NaN/empty -> None). first_code_of -> label of the first
    listed code found in each row's list of codes."""
    codes = label_spec.get("codes")
    out = np.empty(len(raw), dtype=object)
    for i, v in enumerate(raw):
        if codes is not None:
            found = None
            for c in (list(v) if v is not None and not isinstance(v, str) else [v]):
                if c in codes:
                    found = codes[c]
                    break
            out[i] = found
        else:
            missing = v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "")
            out[i] = None if missing else str(v)
    return out


def _load_h5ad(dataset, spec, data_root):
    import anndata as ad
    import scipy.sparse as sp

    mods, ids, obs = {}, None, None
    for name, m in spec["modalities"].items():
        if m["kind"] != "counts":
            raise ValueError(f"{dataset}/{name}: h5ad files hold count matrices; kind={m['kind']} is not supported")
        (path,) = data_spec.files_for(dataset, name, spec, data_root)
        a = ad.read_h5ad(path)
        these = np.asarray(a.obs_names, dtype=str)
        if ids is None:
            ids, obs = these, a.obs
        elif not np.array_equal(ids, these):
            raise ValueError(f"{dataset}: modality '{name}' has different sample ids or order than the first "
                             "modality; rows would not be paired")
        X = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
        mods[name] = {"kind": "counts", "X": X.tocsr().astype(np.float32),
                      "feature_names": np.asarray(a.var_names, dtype=str), "file": path.name}
    labels = {}
    for lname, l in spec["labels"].items():
        if l["column"] not in obs.columns:
            raise ValueError(f"{dataset}: label column '{l['column']}' not in the sample annotations")
        labels[lname] = _label_values(obs[l["column"]].tolist(), l)
    used = {l["column"] for l in spec["labels"].values()}
    other = [c for c in obs.columns if c not in used]
    return ids, None, mods, labels, other


def _load_parquet(dataset, spec, data_root):
    import pyarrow.parquet as pq

    file_sets = {name: data_spec.files_for(dataset, name, spec, data_root) for name in spec["modalities"]}
    files = next(iter(file_sets.values()))
    if any(fs != files for fs in file_sets.values()):
        raise ValueError(f"{dataset}: parquet modalities must come from the same files so rows are paired")
    if not spec["id_column"]:
        raise ValueError(f"{dataset}: '## Loading' needs id_column for parquet data")

    text_cols = {n: m["column"] for n, m in spec["modalities"].items() if m["kind"] == "text"}
    label_cols = {n: l["column"] for n, l in spec["labels"].items()}
    columns = sorted({spec["id_column"], *text_cols.values(), *label_cols.values()})
    ids, split, refs = [], [], []
    texts = {n: [] for n in text_cols}
    raw_labels = {n: [] for n in label_cols}
    all_columns = None
    for path in files:
        pf = pq.ParquetFile(path)
        all_columns = all_columns or pf.schema_arrow.names
        t = pf.read(columns=columns).to_pydict()
        n = len(t[spec["id_column"]])
        ids += [str(x) for x in t[spec["id_column"]]]
        split += [data_spec.split_of(path, spec)] * n
        refs += [(path, i) for i in range(n)]
        for name, col in text_cols.items():
            texts[name] += [x or "" for x in t[col]]
        for name, col in label_cols.items():
            raw_labels[name] += t[col]

    mods = {}
    for name, m in spec["modalities"].items():
        if m["kind"] == "text":
            mods[name] = {"kind": "text", "texts": texts[name], "column": m["column"]}
        elif m["kind"] == "image":
            mods[name] = {"kind": "image", "refs": refs, "column": m["column"]}
        else:
            raise ValueError(f"{dataset}/{name}: kind=counts from parquet is not supported")
    labels = {n: _label_values(raw_labels[n], spec["labels"][n]) for n in label_cols}
    loaded = set(columns) | {m["column"] for m in spec["modalities"].values()}
    other = [c for c in (all_columns or []) if c not in loaded]
    has_split = spec.get("split") is not None
    return np.asarray(ids, dtype=str), (np.asarray(split, dtype=str) if has_split else None), mods, labels, other


def load(dataset, data_root=DATA_ROOT, description_root=DATA_ROOT):
    """data_root: where the data files are; description_root: where DATA_DESCRIPTION.md is (they differ
    on a cluster, where data lives outside the repo)."""
    spec = data_spec.parse(dataset, description_root)
    suffixes = {Path(m["file"]).suffix.lower() for m in spec["modalities"].values()}
    if suffixes == {".h5ad"}:
        ids, split, mods, labels, other = _load_h5ad(dataset, spec, data_root)
    elif suffixes == {".parquet"}:
        ids, split, mods, labels, other = _load_parquet(dataset, spec, data_root)
    else:
        raise ValueError(f"{dataset}: unsupported or mixed file types {sorted(suffixes)} (supported: .h5ad, .parquet)")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{dataset}: sample ids are not unique")
    return {"dataset": dataset, "ids": ids, "split": split, "modalities": mods, "labels": labels,
            "other_columns": other}


def read_images(refs, column, as_pil=True):
    """Decode the images at the given (file, row_in_file) refs, in the given order."""
    import pyarrow.parquet as pq
    from PIL import Image

    by_file = {}
    for k, (path, row) in enumerate(refs):
        by_file.setdefault(path, []).append((row, k))
    out = [None] * len(refs)
    for path, items in by_file.items():
        col = pq.ParquetFile(path).read(columns=[column]).column(column)
        for row, k in items:
            v = col[row].as_py()
            b = v["bytes"] if isinstance(v, dict) else v
            out[k] = Image.open(io.BytesIO(b)) if as_pil else b
    return out


def iter_images(refs, column, batch_size=256):
    """Yields (start_index, list of PIL images) over all refs in order, streaming each file once.
    Refs must be in file order with consecutive rows from 0 (as load() returns them)."""
    import pyarrow.parquet as pq
    from PIL import Image

    start = 0
    files = list(dict.fromkeys(p for p, _ in refs))
    for path in files:
        rows = [r for p, r in refs if p == path]
        if rows != list(range(len(rows))):
            raise ValueError("iter_images needs refs in file order; use read_images for a subset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=[column]):
            vals = batch.column(0).to_pylist()
            imgs = [Image.open(io.BytesIO(v["bytes"] if isinstance(v, dict) else v)) for v in vals]
            yield start, imgs
            start += len(imgs)
