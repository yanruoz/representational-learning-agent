"""Setup step: embeds a dataset's image and/or text modalities with frozen pretrained encoders and
   caches the result, so the pipeline trains only small heads on fixed vectors. Each encoder is a
   candidate the agent can choose (or not) in its plan, with evidence from baseline_check.py.

   Runs the same on CUDA (Longleaf), MPS (Mac), or CPU. run.sh calls it with --if-missing before the
   agent starts, so a machine without the Longleaf cache computes the embeddings itself. The agent
   never runs it (it only reads the cache).

     venv/bin/python scripts/extract_embeddings.py --dataset <d> [--encoder biomedclip_frozen]
         [--data-root data] [--cache-root cache] [--device auto] [--if-missing] [--limit N]

   Writes cache/<dataset>/<encoder>/:
     <modality>.npy   float32, one row per sample, not normalized (callers L2-normalize for cosine)
     ids.csv          row, pair_id, split, source_file, row_in_file: row i of every .npy is this sample
     manifest.json    encoder, pretraining data, versions, device, data checksums, counts, `complete`
   Files are written to a temporary folder and renamed at the end, so a half-finished run never
   looks like a cache.

   An encoder applies to a dataset when the dataset declares exactly one modality of every kind the
   encoder embeds, all read from the same files (so rows pair up). --download-only fetches the
   weights and exits (for clusters whose compute nodes run offline).
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

# Registry of frozen encoders. `kinds`: the modality kinds the encoder embeds. `pretraining_data`
# is recorded in the manifest so the profile, critic, and report can check for overlap with the
# evaluation data (the data description says where a dataset comes from).
ENCODERS = {
    "biomedclip_frozen": {
        "kinds": ["image", "text"],
        "backend": "open_clip",
        "model": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "context_length": 256,
        "description": "BiomedCLIP: ViT-B/16 image tower and PubMedBERT text tower trained together "
                       "with a CLIP-style contrastive loss on figure-caption pairs; frozen.",
        "pretraining_data": "PMC-15M: about 15 million figure-caption pairs from PubMed Central "
                            "Open Access articles (images and captions).",
    },
    "dinov2_vitb14_frozen": {
        "kinds": ["image"],
        "backend": "transformers",
        "model": "facebook/dinov2-base",
        "description": "DINOv2 ViT-B/14: self-supervised image encoder (self-distillation, no text, "
                       "no labels); frozen; embedding = final-layer CLS token after layer norm.",
        "pretraining_data": "LVD-142M: about 142 million web images selected by visual similarity to "
                            "curated image datasets; no captions or text were used.",
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
    """{kind: modality name} if the dataset has exactly one modality of each kind the encoder
    embeds, all from the same files (so rows pair up); else None."""
    by_kind = {}
    for name, m in spec["modalities"].items():
        if m["kind"] in encoder["kinds"]:
            by_kind.setdefault(m["kind"], []).append(name)
    if set(by_kind) != set(encoder["kinds"]) or any(len(v) != 1 for v in by_kind.values()):
        return None
    chosen = {k: v[0] for k, v in by_kind.items()}
    if any(spec["modalities"][m].get("column") is None for m in chosen.values()):
        return None  # cached encoders read parquet columns
    if len({spec["modalities"][m]["file"] for m in chosen.values()}) != 1:
        return None
    return chosen


def load_encoder(enc, device):
    """Returns ({kind: fn(list of raw values) -> np.ndarray}, versions, model_revision)."""
    import torch
    from PIL import Image

    amp = dict(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")

    def pil(b):
        return Image.open(io.BytesIO(b)).convert("RGB")

    if enc["backend"] == "open_clip":
        import open_clip
        model, preprocess = open_clip.create_model_from_pretrained(enc["model"])
        tokenizer = open_clip.get_tokenizer(enc["model"])
        model = model.to(device).eval()

        def image_fn(values):
            x = torch.stack([preprocess(pil(b)) for b in values]).to(device)
            with torch.inference_mode(), torch.autocast(**amp):
                return model.encode_image(x).float().cpu().numpy()

        def text_fn(values):
            t = tokenizer([v or "" for v in values], context_length=enc["context_length"]).to(device)
            with torch.inference_mode(), torch.autocast(**amp):
                return model.encode_text(t).float().cpu().numpy()

        return {"image": image_fn, "text": text_fn}, {"open_clip": open_clip.__version__}, None

    if enc["backend"] == "transformers":
        import transformers
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(enc["model"])
        model = AutoModel.from_pretrained(enc["model"]).to(device).eval()

        def image_fn(values):
            x = processor(images=[pil(b) for b in values], return_tensors="pt")["pixel_values"].to(device)
            with torch.inference_mode(), torch.autocast(**amp):
                return model(pixel_values=x).pooler_output.float().cpu().numpy()

        revision = getattr(model.config, "_commit_hash", None)
        return {"image": image_fn}, {"transformers": transformers.__version__}, revision

    raise ValueError(f"unknown backend {enc['backend']}")


def _raw(value):
    return value["bytes"] if isinstance(value, dict) else value


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
    import pyarrow.parquet as pq
    import torch

    enc = ENCODERS[encoder_id]
    spec = data_spec.parse(dataset)
    mods = applicable(spec, enc)
    if mods is None:
        raise ValueError(f"{encoder_id} needs one modality of each kind {enc['kinds']} from the same files; "
                         f"{dataset} declares {spec['modalities']}")
    if not spec["id_column"]:
        raise ValueError(f"{dataset}: '## Loading' needs id_column for parquet modalities")
    cols = {kind: spec["modalities"][name]["column"] for kind, name in mods.items()}
    files = data_spec.files_for(dataset, next(iter(mods.values())), spec, data_root)

    device = pick_device(device_name)
    t0 = time.time()
    fns, versions, revision = load_encoder(enc, device)
    load_s = time.time() - t0
    print(f"[extract] {encoder_id} on {device}, model loaded in {load_s:.1f}s", flush=True)

    out = {kind: [] for kind in mods}
    ids = []
    t0 = time.time()
    for path in files:
        row_in_file = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                        columns=[spec["id_column"], *cols.values()]):
            rows = batch.to_pylist()
            if limit is not None:
                rows = rows[: max(0, limit - len(ids))]
            if not rows:
                break
            for kind, col in cols.items():
                out[kind].append(fns[kind]([_raw(r[col]) for r in rows]))
            for r in rows:
                ids.append((r[spec["id_column"]], data_spec.split_of(path, spec), path.name, row_in_file))
                row_in_file += 1
            if len(ids) % (batch_size * 20) < len(rows):
                print(f"[extract] {len(ids)} samples, {len(ids) / (time.time() - t0):.1f}/s", flush=True)
        if limit is not None and len(ids) >= limit:
            break
    encode_s = time.time() - t0

    if len({i[0] for i in ids}) != len(ids):
        raise ValueError(f"{dataset}: {spec['id_column']} is not unique; cannot key the cache by sample id")

    final = Path(cache_root) / dataset / encoder_id
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    dims = {}
    for kind, name in mods.items():
        arr = np.concatenate(out[kind]).astype(np.float32)
        np.save(tmp / f"{name}.npy", arr)
        dims[name] = int(arr.shape[1])
    with open(tmp / "ids.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "pair_id", "split", "source_file", "row_in_file"])
        for k, (pid, split, src, r) in enumerate(ids):
            w.writerow([k, pid, split, src, r])

    splits = {}
    for _, s, _, _ in ids:
        splits[s] = splits.get(s, 0) + 1
    manifest = {
        "complete": limit is None,
        "dataset": dataset,
        "encoder": encoder_id,
        "model": enc["model"],
        "model_revision": revision,
        "description": enc["description"],
        "pretraining_data": enc["pretraining_data"],
        "frozen": True,
        "modalities": {name: kind for kind, name in mods.items()},
        "embedding_dim": dims,
        "normalized": False,
        "n_pairs": len(ids),
        "n_by_split": splits,
        "limit": limit,
        "data_files": {p.name: data_checksums(dataset).get(p.name) for p in files},
        "device": str(device),
        "precision": "float16 autocast" if device.type == "cuda" else "float32",
        "versions": {"torch": torch.__version__, **versions},
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
    ap.add_argument("--limit", type=int, help="embed only the first N samples (testing; cache marked incomplete)")
    ap.add_argument("--download-only", action="store_true", help="download the weights of applicable encoders and exit")
    args = ap.parse_args()

    spec = data_spec.parse(args.dataset)
    wanted = [args.encoder] if args.encoder else sorted(ENCODERS)
    todo = [e for e in wanted if applicable(spec, ENCODERS[e])]
    if not todo:
        print(f"[extract] no pretrained encoder applies to {args.dataset}'s modalities; nothing to do")
        return
    if args.download_only:
        import torch
        for e in todo:
            load_encoder(ENCODERS[e], torch.device("cpu"))
            print(f"[extract] {e}: weights cached")
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
        print(f"[extract] wrote {final}: {m['n_pairs']} samples {m['n_by_split']}, dims {m['embedding_dim']}, "
              f"{m['seconds']['encode']}s on {m['device']}")


if __name__ == "__main__":
    main()
