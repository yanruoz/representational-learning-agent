"""GPU smoke test for Longleaf: CUDA works, BiomedCLIP loads, and it embeds real ROCOv2 pairs.

Writes a small JSON with the device, timings, and the matched-pair vs mismatched-pair cosine
similarity on a batch of test images (matched should be clearly higher if everything is wired up).
Not pipeline code: it only checks the environment the GPU jobs will use.
"""
import argparse
import io
import json
import time
from pathlib import Path

import torch

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocov2-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    result = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        Path(args.out).write_text(json.dumps(result, indent=2))
        raise SystemExit("CUDA not available")
    dev = torch.device("cuda")
    result["gpu"] = torch.cuda.get_device_name(0)

    t = time.time()
    a = torch.randn(4096, 4096, device=dev)
    for _ in range(10):
        a @ a
    torch.cuda.synchronize()
    result["matmul_10x4096_s"] = round(time.time() - t, 3)

    import open_clip
    import pyarrow.parquet as pq
    from PIL import Image

    t = time.time()
    model, preprocess = open_clip.create_model_from_pretrained(MODEL)
    tokenizer = open_clip.get_tokenizer(MODEL)
    model = model.to(dev).eval()
    result["model_load_s"] = round(time.time() - t, 2)

    shards = sorted(Path(args.rocov2_dir).glob("test-*.parquet"))
    if not shards:
        raise SystemExit(f"no test-*.parquet in {args.rocov2_dir}")
    table = pq.read_table(shards[0], columns=["image", "caption"]).slice(0, args.n).to_pylist()
    images = torch.stack([preprocess(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")) for r in table])
    texts = tokenizer([r["caption"] for r in table], context_length=256)

    t = time.time()
    with torch.no_grad():
        img = model.encode_image(images.to(dev))
        txt = model.encode_text(texts.to(dev))
    torch.cuda.synchronize()
    img = torch.nn.functional.normalize(img, dim=-1)
    txt = torch.nn.functional.normalize(txt, dim=-1)
    sim = (img @ txt.T).float().cpu()
    n = sim.shape[0]
    matched = sim.diag().mean().item()
    mismatched = (sim.sum() - sim.diag().sum()).item() / (n * n - n)
    top1 = (sim.argmax(dim=1) == torch.arange(n)).float().mean().item()

    result.update({
        "n_pairs": n,
        "embed_dim": img.shape[1],
        "encode_s": round(time.time() - t, 3),
        "mean_cos_matched": round(matched, 4),
        "mean_cos_mismatched": round(mismatched, 4),
        "image_to_text_top1_in_batch": round(top1, 4),
        "chance_top1": round(1 / n, 4),
    })
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
