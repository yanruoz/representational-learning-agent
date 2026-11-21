"""Per-modality dataset profiler. Generic: everything it knows about a dataset comes from
   loaders.load() and the DATA_DESCRIPTION '## Loading' declaration; no dataset-specific branches.

     venv/bin/python scripts/profiler.py --dataset <d> [--out outputs/<d>/profile.json]

   Writes profile.json and profile_summary.md (next to it). Sections:
     samples     count, id uniqueness, declared split sizes
     modalities  per modality, by kind:
                   counts: shape, sparsity, integer check, per-sample totals and detected features,
                           near-binary check, rarely detected features
                   text:   length distribution, empty and duplicate texts, vocabulary size
                   image:  sizes, color modes, formats, grayscale-stored-as-color (seeded sample)
     labels      class counts (overall and per split), missing labels, imbalance, small classes,
                 and how often a class name appears verbatim in each text modality (a leakage hint)
     other_columns  sample-level columns present but not loaded (never model inputs)
     cache       frozen-encoder cache status for every encoder that applies
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import data_spec
import extract_embeddings
import loaders

SEED = 0
IMAGE_SAMPLE = 400            # images decoded for size/mode statistics
INTEGER_CHECK_ROWS = 500      # rows checked for integer-valued counts
SMALL_CLASS = 20              # classes below this many samples give unstable per-class metrics
SMALL_TEST_CLASS = 10         # same, within a declared test split
RARE_FEATURE_FRACTION = 0.01  # features detected in fewer than 1% of samples


def _q(x, qs=(5, 50, 95)):
    x = np.asarray(x, dtype=float)
    return {f"p{q}": round(float(np.percentile(x, q)), 2) for q in qs}


def profile_counts(m):
    X = m["X"]
    n, d = X.shape
    nnz = X.nnz
    rows = X[: min(n, INTEGER_CHECK_ROWS)]
    integer = bool(np.allclose(rows.data, np.round(rows.data)))
    totals = np.asarray(X.sum(axis=1)).ravel()
    detected = np.diff(X.indptr)
    per_feature = np.bincount(X.indices, minlength=d) / n
    return {
        "kind": "counts",
        "file": m.get("file"),
        "n_samples": n,
        "n_features": d,
        "feature_name_examples": [str(f) for f in m["feature_names"][:3]],
        "sparsity": round(1 - nnz / (n * d), 4),
        "integer_valued": integer,
        "max_value": float(X.data.max()) if nnz else 0.0,
        "nonzero_fraction_equal_to_1": round(float(np.mean(X.data == 1)), 4) if nnz else None,
        "per_sample_total": _q(totals),
        "per_sample_total_ratio_p95_p5": round(float(np.percentile(totals, 95) / max(np.percentile(totals, 5), 1e-9)), 2),
        "per_sample_detected_features": _q(detected),
        "samples_with_zero_total": int((totals == 0).sum()),
        "fraction_features_detected_in_under_1pct": round(float(np.mean(per_feature < RARE_FEATURE_FRACTION)), 4),
        "features_never_detected": int((per_feature == 0).sum()),
    }


def _words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def profile_text(m):
    texts = m["texts"]
    words = [_words(t) for t in texts]
    lengths = [len(w) for w in words]
    vocab = Counter(w for ws in words for w in set(ws))
    return {
        "kind": "text",
        "n_samples": len(texts),
        "empty_texts": int(sum(1 for t in texts if not t.strip())),
        "duplicate_texts": int(len(texts) - len(set(texts))),
        "words_per_text": {**_q(lengths), "max": int(max(lengths)) if lengths else 0},
        "vocabulary_size": len(vocab),
        "words_in_at_least_1pct_of_texts": int(sum(1 for c in vocab.values() if c >= 0.01 * len(texts))),
    }


def profile_image(m, rng):
    refs = m["refs"]
    idx = np.sort(rng.choice(len(refs), size=min(IMAGE_SAMPLE, len(refs)), replace=False))
    imgs = loaders.read_images([refs[i] for i in idx], m["column"])
    widths, heights, modes, formats, gray_as_color = [], [], Counter(), Counter(), 0
    for im in imgs:
        widths.append(im.width)
        heights.append(im.height)
        modes[im.mode] += 1
        formats[im.format or "unknown"] += 1
        if im.mode in ("RGB", "RGBA"):
            a = np.asarray(im.convert("RGB").resize((32, 32)), dtype=np.int16)
            if np.abs(a[..., 0] - a[..., 1]).max() <= 2 and np.abs(a[..., 1] - a[..., 2]).max() <= 2:
                gray_as_color += 1
    aspect = np.asarray(widths) / np.maximum(np.asarray(heights), 1)
    return {
        "kind": "image",
        "n_samples": len(refs),
        "n_decoded_for_profile": len(imgs),
        "width": _q(widths),
        "height": _q(heights),
        "aspect_ratio_w_over_h": _q(aspect),
        "modes": dict(modes.most_common()),
        "formats": dict(formats.most_common()),
        "color_images_that_are_grayscale": round(gray_as_color / max(1, sum(modes[k] for k in ("RGB", "RGBA"))), 3),
    }


def profile_labels(values, split, text_mods):
    present = np.array([v is not None for v in values])
    counts = Counter(v for v in values if v is not None)
    out = {
        "n_classes": len(counts),
        "n_labeled": int(present.sum()),
        "n_missing": int((~present).sum()),
        "class_counts": dict(counts.most_common()),
        "imbalance_ratio_max_over_min": round(max(counts.values()) / min(counts.values()), 1) if counts else None,
        "classes_under_%d" % SMALL_CLASS: sorted(c for c, k in counts.items() if k < SMALL_CLASS),
    }
    if split is not None:
        out["class_counts_by_split"] = {
            s: dict(Counter(v for v, sp in zip(values, split) if sp == s and v is not None).most_common())
            for s in sorted(set(split))
        }
        out["missing_by_split"] = {s: int(sum(1 for v, sp in zip(values, split) if sp == s and v is None))
                                   for s in sorted(set(split))}
        if "test" in out["class_counts_by_split"]:
            out["test_classes_under_%d" % SMALL_TEST_CLASS] = sorted(
                c for c, k in out["class_counts_by_split"]["test"].items() if k < SMALL_TEST_CLASS)
    # Leakage hint: how often a sample's own class name appears verbatim in its text.
    hints = {}
    for name, texts in text_mods.items():
        per_class = {}
        for c in counts:
            pat = re.compile(r"\b" + re.escape(c.lower()) + r"\b")
            own = [pat.search(t.lower()) is not None for t, v in zip(texts, values) if v == c]
            per_class[c] = round(float(np.mean(own)), 3) if own else None
        overall = [re.search(r"\b" + re.escape(v.lower()) + r"\b", t.lower()) is not None
                   for t, v in zip(texts, values) if v is not None]
        hints[name] = {"overall": round(float(np.mean(overall)), 3) if overall else None, "per_class": per_class}
    if hints:
        out["own_class_name_in_text_fraction"] = hints
    return out


def profile(dataset, data_root=loaders.DATA_ROOT, description_root=loaders.DATA_ROOT):
    rng = np.random.default_rng(SEED)
    spec = data_spec.parse(dataset, description_root)
    d = loaders.load(dataset, data_root, description_root)
    ids, split = d["ids"], d["split"]
    p = {
        "dataset": dataset,
        "samples": {
            "n": int(len(ids)),
            "ids_unique": bool(len(set(ids.tolist())) == len(ids)),
            "id_examples": [str(i) for i in ids[:3]],
            "paired_modalities": sorted(d["modalities"]),
            "split_declared": split is not None,
            "split_sizes": dict(Counter(split.tolist())) if split is not None else None,
        },
        "modalities": {},
        "labels": {},
        "other_columns": d["other_columns"],
        "cache": {},
    }
    for name, m in d["modalities"].items():
        if m["kind"] == "counts":
            p["modalities"][name] = profile_counts(m)
        elif m["kind"] == "text":
            p["modalities"][name] = profile_text(m)
        else:
            p["modalities"][name] = profile_image(m, rng)
    texts = {n: m["texts"] for n, m in d["modalities"].items() if m["kind"] == "text"}
    for lname, values in d["labels"].items():
        p["labels"][lname] = profile_labels(values, split, texts)
    for enc_id, enc in sorted(extract_embeddings.ENCODERS.items()):
        mods = extract_embeddings.applicable(spec, enc)
        if mods is None:
            continue
        ok, why = extract_embeddings.cache_status(data_spec.REPO / "cache" / dataset / enc_id, dataset)
        p["cache"][enc_id] = {"ready": ok, "status": why, "modalities": mods, "description": enc["description"]}
    return p


def summary_md(p):
    L = [f"# Profile: {p['dataset']}", ""]
    s = p["samples"]
    L += [f"- Samples: {s['n']} (ids unique: {s['ids_unique']}), paired modalities: {', '.join(s['paired_modalities'])}",
          f"- Declared split: {s['split_sizes'] if s['split_declared'] else 'none (the plan must choose one)'}", ""]
    for name, m in p["modalities"].items():
        L.append(f"## Modality `{name}` ({m['kind']})")
        if m["kind"] == "counts":
            L += [f"- {m['n_samples']} x {m['n_features']} features, sparsity {m['sparsity']}, integer-valued: {m['integer_valued']}, max {m['max_value']}",
                  f"- Nonzero entries equal to 1: {m['nonzero_fraction_equal_to_1']}",
                  f"- Per-sample total: {m['per_sample_total']} (p95/p5 ratio {m['per_sample_total_ratio_p95_p5']})",
                  f"- Detected features per sample: {m['per_sample_detected_features']}",
                  f"- Features detected in under 1% of samples: {m['fraction_features_detected_in_under_1pct']}; never detected: {m['features_never_detected']}"]
        elif m["kind"] == "text":
            L += [f"- {m['n_samples']} texts, {m['empty_texts']} empty, {m['duplicate_texts']} duplicates",
                  f"- Words per text: {m['words_per_text']}; vocabulary {m['vocabulary_size']} words"]
        else:
            L += [f"- {m['n_samples']} images ({m['n_decoded_for_profile']} decoded for this profile)",
                  f"- Width {m['width']}, height {m['height']}, aspect {m['aspect_ratio_w_over_h']}",
                  f"- Modes {m['modes']}, formats {m['formats']}; color images that are grayscale: {m['color_images_that_are_grayscale']}"]
        L.append("")
    for lname, l in p["labels"].items():
        L += [f"## Label `{lname}`",
              f"- {l['n_classes']} classes, {l['n_labeled']} labeled, {l['n_missing']} missing, imbalance max/min {l['imbalance_ratio_max_over_min']}",
              f"- Counts: {l['class_counts']}"]
        if "class_counts_by_split" in l:
            for sp, c in l["class_counts_by_split"].items():
                L.append(f"- {sp}: {c} (missing {l['missing_by_split'][sp]})")
        small = [k for k in l if k.startswith(("classes_under", "test_classes_under")) and l[k]]
        L += [f"- {k.replace('_', ' ')}: {l[k]}" for k in small]
        for t, h in l.get("own_class_name_in_text_fraction", {}).items():
            L.append(f"- Own class name appears in `{t}`: {h['overall']} of labeled samples")
        L.append("")
    if p["other_columns"]:
        L += ["## Other sample-level columns (not loaded, never model inputs)", ", ".join(p["other_columns"]), ""]
    if p["cache"]:
        L.append("## Frozen-encoder cache")
        L += [f"- `{k}`: {'ready' if v['ready'] else 'NOT ready'} ({v['status']})" for k, v in p["cache"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Profile a multimodal dataset; writes profile.json + profile_summary.md.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", help="default outputs/<dataset>/profile.json")
    args = ap.parse_args()
    out = Path(args.out) if args.out else data_spec.REPO / "outputs" / args.dataset / "profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prof = profile(args.dataset)
    out.write_text(json.dumps(prof, indent=2) + "\n")
    md = out.with_name("profile_summary.md")
    md.write_text(summary_md(prof))
    print(md.read_text())
    print(f"Wrote {out}\nWrote {md}")
