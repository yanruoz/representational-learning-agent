"""Setup step: embeds a dataset's image and text modalities with a frozen pretrained encoder and
   caches the result, so the pipeline trains only small heads on fixed vectors.

   Runs the same on CUDA (Longleaf), MPS (Mac), or CPU. run.sh calls it with --if-missing before the
   agent starts, so a machine without the Longleaf cache computes the embeddings itself. The agent
   never runs it (it only reads the cache).

     venv/bin/python scripts/extract_embeddings.py --dataset <d> [--encoder biomedclip_frozen]
         [--data-root data] [--cache-root cache] [--device auto] [--if-missing] [--limit N]

   Writes cache/<dataset>/<encoder>/:
     <modality>.npy   float32, one row per pair, not normalized (callers L2-normalize for cosine)
     ids.csv          row, pair_id, split, source_file, row_in_file: row i of every .npy is this pair
     manifest.json    encoder, versions, device, data file checksums, counts, timing, `complete: true`
   Files are written to a temporary folder and renamed at the end, so a half-finished run never
   looks like a cache.

   An encoder applies to a dataset when the dataset declares modalities of every kind it needs
   (e.g. an image modality and a text modality read from the same files, so rows are paired).
"""
import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_spec  # noqa: E402
import fetch_data  # noqa: E402

REPO = data_spec.REPO

# Registry of frozen encoders. `kinds`: the modality kinds the encoder embeds.
ENCODERS = {
    "biomedclip_frozen": {
        "kinds": ["image", "text"],
        "model": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "context_length": 256,
        "description": "BiomedCLIP (ViT-B/16 image tower, PubMedBERT text tower), contrastively "
                       "pretrained on PubMed Central figure-caption pairs; frozen, no fine-tuning.",
    },
}


def pick_device(name):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def applicable(spec, encoder):
    """{kind: modality name} if the dataset has one modality of each kind the encoder needs, read
    from the same files (so rows pair up); else None."""
    by_kind = {}
    for name, m in spec["modalities"].items():
        if m["kind"] in encoder["kinds"]:
            by_kind.setdefault(m["kind"], []).append(name)
    if set(by_kind) != set(encoder["kinds"]) or any(len(v) != 1 for v in by_kind.values()):
        return None
    chosen = {k: v[0] for k, v in by_kind.items()}
    if len({spec["modalities"][m]["file"] for m in chosen.values()}) != 1:
        return None
    return chosen


def cache_status(out_dir, dataset):
    """(ok, reason): ok when the manifest says complete and lists exactly the declared data files
    with their declared checksums (so a cache built from other data is never reused)."""
    try:
        m = json.loads((Path(out_dir) / "manifest.json").read_text())
    except (OSError, ValueError):
        return False, "no readable manifest.json"
    if m.get("complete") is not True:
        return False, "manifest says incomplete (a --limit test run?)"
    declared = data_checksums(dataset)
    used = m.get("data_files") or {}
    if used != {f: declared.get(f) for f in used} or not used:
        return False, "built from data files whose checksums differ from DATA_DESCRIPTION.md"
    return True, f"complete: {m.get('n_pairs')} pairs, built on {m.get('device')}"


def data_checksums(dataset):
    """Declared checksums from DATA_DESCRIPTION '## Files' (fetch_data verified them on download)."""
    entries = fetch_data.parse_files_section(data_spec.DATA_ROOT / dataset / "DATA_DESCRIPTION.md")
    return {e["file"]: e.get("sha256") or e.get("md5") for e in entries}


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def extract(dataset, encoder_id, data_root, cache_root, device_name, batch_size, limit):
    import open_clip
    import pyarrow.parquet as pq
    import torch
    from PIL import Image

    enc = ENCODERS[encoder_id]
    spec = data_spec.parse(dataset)
    mods = applicable(spec, enc)
    if mods is None:
        raise ValueError(f"{encoder_id} needs one modality of each kind {enc['kinds']} from the same files; "
                         f"{dataset} declares {spec['modalities']}")
    if not spec["id_column"]:
        raise ValueError(f"{dataset}: '## Loading' needs id_column for paired parquet modalities")
    img_col = spec["modalities"][mods["image"]]["column"]
    txt_col = spec["modalities"][mods["text"]]["column"]
    files = data_spec.files_for(dataset, mods["image"], spec, data_root)

    device = pick_device(device_name)
    t0 = time.time()
    model, preprocess = open_clip.create_model_from_pretrained(enc["model"])
    tokenizer = open_clip.get_tokenizer(enc["model"])
    model = model.to(device).eval()
    load_s = time.time() - t0
    print(f"[extract] {encoder_id} on {device}, model loaded in {load_s:.1f}s", flush=True)

    def encode(images, texts):
        x = torch.stack([preprocess(Image.open(io.BytesIO(b)).convert("RGB")) for b in images]).to(device)
        t = tokenizer(texts, context_length=enc["context_length"]).to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            return model.encode_image(x).float().cpu().numpy(), model.encode_text(t).float().cpu().numpy()

    img_out, txt_out, ids = [], [], []
    t0 = time.time()
    for path in files:
        pf = pq.ParquetFile(path)
        row_in_file = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=[spec["id_column"], img_col, txt_col]):
            rows = batch.to_pylist()
            if limit is not None:
                rows = rows[: max(0, limit - len(ids))]
            if not rows:
                break
            images = [r[img_col]["bytes"] if isinstance(r[img_col], dict) else r[img_col] for r in rows]
            texts = [r[txt_col] or "" for r in rows]
            i_emb, t_emb = encode(images, texts)
            img_out.append(i_emb)
            txt_out.append(t_emb)
            for r in rows:
                ids.append((r[spec["id_column"]], data_spec.split_of(path, spec), path.name, row_in_file))
                row_in_file += 1
            if len(ids) % (batch_size * 20) < len(rows):
                rate = len(ids) / (time.time() - t0)
                print(f"[extract] {len(ids)} pairs, {rate:.1f}/s", flush=True)
        if limit is not None and len(ids) >= limit:
            break
    encode_s = time.time() - t0

    if len({i[0] for i in ids}) != len(ids):
        raise ValueError(f"{dataset}: {spec['id_column']} is not unique; cannot key the cache by pair id")

    final = Path(cache_root) / dataset / encoder_id
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / f"{mods['image']}.npy", np.concatenate(img_out).astype(np.float32))
    np.save(tmp / f"{mods['text']}.npy", np.concatenate(txt_out).astype(np.float32))
    with open(tmp / "ids.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "pair_id", "split", "source_file", "row_in_file"])
        for k, (pid, split, src, r) in enumerate(ids):
            w.writerow([k, pid, split, src, r])

    import torch as _t
    splits = {}
    for _, s, _, _ in ids:
        splits[s] = splits.get(s, 0) + 1
    manifest = {
        "complete": limit is None,
        "dataset": dataset,
        "encoder": encoder_id,
        "model": enc["model"],
        "description": enc["description"],
        "frozen": True,
        "modalities": {mods["image"]: "image", mods["text"]: "text"},
        "embedding_dim": int(img_out[0].shape[1]),
        "normalized": False,
        "n_pairs": len(ids),
        "n_by_split": splits,
        "limit": limit,
        "data_files": {p.name: data_checksums(dataset).get(p.name) for p in files},
        "device": str(device),
        "precision": "float16 autocast" if device.type == "cuda" else "float32",
        "versions": {"torch": _t.__version__, "open_clip": open_clip.__version__},
        "seconds": {"model_load": round(load_s, 1), "encode": round(encode_s, 1)},
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", choices=sorted(ENCODERS), help="default: every encoder that applies")
    ap.add_argument("--data-root", default=str(data_spec.DATA_ROOT))
    ap.add_argument("--cache-root", default=str(REPO / "cache"))
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps, or cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--if-missing", action="store_true", help="skip encoders whose complete cache exists")
    ap.add_argument("--limit", type=int, help="embed only the first N pairs (testing; cache marked incomplete)")
    args = ap.parse_args()

    spec = data_spec.parse(args.dataset)
    wanted = [args.encoder] if args.encoder else sorted(ENCODERS)
    todo = [e for e in wanted if applicable(spec, ENCODERS[e])]
    if not todo:
        print(f"[extract] no pretrained encoder applies to {args.dataset}'s modalities; nothing to do")
        return
    for e in todo:
        out = Path(args.cache_root) / args.dataset / e
        ok, why = cache_status(out, args.dataset)
        if args.if_missing and ok:
            print(f"[extract] {out}: {why}; skipping")
            continue
        if out.exists():
            print(f"[extract] {out}: {why}; rebuilding")
        try:
            final, m = extract(args.dataset, e, args.data_root, args.cache_root, args.device, args.batch_size, args.limit)
        except ValueError as err:
            sys.exit(f"ERROR: {err}")
        print(f"[extract] wrote {final}: {m['n_pairs']} pairs {m['n_by_split']}, dim {m['embedding_dim']}, "
              f"{m['seconds']['encode']}s on {m['device']}")


if __name__ == "__main__":
    main()
