"""Train one model from plan.json at one seed (and, for label-using methods, one label budget),
   then save every sample's per-modality embeddings.

   Usage: venv/bin/python scripts/train.py --plan outputs/<d>/plan.json --model-id <id> --seed <s> [--budget B]

   Writes outputs/<d>/embeddings/<key>.npz (one array per modality, rows aligned with the dataset's
   sample ids) and outputs/<d>/runs/<key>.json (history, stop reason, temperature, timing).
   key = <id>_s<seed>, plus _b<budget> for methods that use labels in training.

   Data use:
   - Featurizers are fit on train+val; the model trains on the training split, filtered by
     plan.training_pool; validation pairs give the early-stopping loss.
   - Label-using losses (supcon, cross_entropy) see only a seeded draw of `budget` labeled training
     samples per class (draw 0, the same draw the evaluation's probe uses for that budget). They do
     not early-stop: a validation loss would need more labels than the budget allows, so they train
     for the planned number of epochs.
   - Test samples are embedded after training and never touch training or model selection.
   Hyperparameter concat_input: true saves [input features, learned embedding] (each unit-norm) as
   the embedding, for objectives where a probe may need what the alignment compressed away.
   A wall-clock cap (plan hyperparameter max_minutes, default 3 x estimated_minutes, at least 2) stops
   any run that overruns.
"""
import argparse
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import context
import features
import plan_schema
from models.nets import MultiTower, infonce, supcon, vicreg

HP_DEFAULTS = {"embed_dim": 64, "hidden_dims": [256], "epochs": 100, "batch_size": 256, "lr": 1e-3,
               "weight_decay": 1e-4, "dropout": 0.1, "temperature_init": 0.07, "supcon_temperature": 0.1,
               "modality_dropout": 0.3, "patience": 15}


def pick_device(name="auto"):
    """'auto' = CPU. The models here are small MLPs on <= ~1k-dim inputs; on an Apple M4, 5 epochs on
    ~6.5k pairs took 0.2-0.8 s on CPU vs 1.0-1.8 s on MPS (kernel launch overhead dominates)."""
    return torch.device("cpu" if name == "auto" else name)


def model_entry(plan, model_id):
    for m in plan["models"]:
        if m["id"] == model_id:
            return m
    raise ValueError(f"no model '{model_id}' in plan.json")


def run_key(entry, seed, budget=None):
    key = f"{entry['id']}_s{seed}"
    return key + (f"_b{budget}" if budget is not None else "")


def config_hash(plan, entry, seed, budget):
    cfg = {"split": plan["split"], "pool": plan.get("training_pool"), "excluded": plan.get("excluded_classes"),
           "entry": {k: v for k, v in entry.items() if k not in ("status", "reason", "estimated_minutes")},
           "seed": seed, "budget": budget}
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def label_name(objective, entry):
    """The label a label-using model trains on: the model's own `label`, else the primary label task's."""
    if entry.get("label"):
        return entry["label"]
    for t in objective.get("tasks", []):
        if t["task"] == "label_prediction":
            return t["label"]
    raise ValueError(f"model {entry['id']} uses labels but no label task or model label is given")


def training_rows(ctx, idx):
    pool = ctx.plan.get("training_pool") or {"filter": "all"}
    if pool["filter"] == "all":
        return idx
    if pool["filter"] == "label_present":
        return ctx.labeled(pool["label"], idx)
    raise ValueError(f"unknown training_pool filter {pool['filter']}")


def train(ctx, entry, seed, budget=None, device_name="auto"):
    method, loss_name = entry["method"], entry["loss"]
    if method not in plan_schema.LOSSES or loss_name not in plan_schema.LOSSES[method]:
        raise ValueError(f"{method} cannot train with loss {loss_name}")
    hp = {**HP_DEFAULTS, **entry.get("hyperparameters", {})}
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device(device_name)
    t_start = time.time()
    max_minutes = float(hp.get("max_minutes", max(2.0, 3 * float(entry.get("estimated_minutes", 1)))))

    # Inputs: standardized features per modality (stats from train+val).
    X = {}
    for m, spec in entry["inputs"].items():
        Z = features.featurize(ctx, m, spec["encoder"], spec.get("params"))
        mu, sd = Z[ctx.fit_idx].mean(0), Z[ctx.fit_idx].std(0) + 1e-6
        X[m] = torch.tensor((Z - mu) / sd, dtype=torch.float32)
    mods = list(X)
    if method == "two_tower" and len(mods) != 2:
        raise ValueError("two_tower needs exactly two input modalities")

    tr = training_rows(ctx, ctx.train)
    va = training_rows(ctx, ctx.val)
    uses_labels = loss_name in plan_schema.LOSSES_NEEDING_LABELS
    y_all = torch.full((ctx.n,), -1, dtype=torch.long)
    classes, label_info = [], None
    if uses_labels:
        if budget is None:
            raise ValueError(f"{entry['id']} uses labels ({loss_name}); a label budget is required")
        lname = label_name(ctx.objective, entry)
        lab_idx, taken, unfilled = ctx.draw_labeled(lname, budget, 0, pool=ctx.train)
        vals = ctx.labels(lname)
        classes = sorted({vals[i] for i in lab_idx})
        cidx = {c: k for k, c in enumerate(classes)}
        y_all[torch.tensor(lab_idx)] = torch.tensor([cidx[vals[i]] for i in lab_idx])
        label_info = {"label": lname, "budget": budget, "labels_used": int(len(lab_idx)), "per_class": taken,
                      "unfilled_classes": unfilled}
        if loss_name == "cross_entropy":
            tr = lab_idx  # a classifier trains on labeled samples only

    in_dims = {m: X[m].shape[1] for m in mods}
    net = MultiTower(in_dims, entry["inputs"][mods[0]].get("head", "mlp"), hp["hidden_dims"], hp["embed_dim"],
                     hp["dropout"], decoders=(loss_name == "reconstruction"),
                     n_classes=len(classes) if loss_name == "cross_entropy" else 0,
                     temperature_init=hp["temperature_init"])
    # Per-modality heads may differ: rebuild towers whose head differs from the first.
    for m in mods[1:]:
        h = entry["inputs"][m].get("head", "mlp")
        if h != entry["inputs"][mods[0]].get("head", "mlp"):
            from models.nets import tower
            net.towers[m] = tower(in_dims[m], h, hp["hidden_dims"], hp["embed_dim"], hp["dropout"])
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    def batch_loss(rows):
        xs = {m: X[m][rows].to(device) for m in mods}
        zs = net.embed(xs)
        if loss_name == "infonce":
            return infonce(zs[mods[0]], zs[mods[1]], net.logit_scale)
        if loss_name == "vicreg":
            return vicreg(zs[mods[0]], zs[mods[1]])
        if loss_name == "supcon":
            return supcon(zs[mods[0]], zs[mods[1]], y_all[rows].to(device), hp["supcon_temperature"])
        z = net.fused(zs, hp["modality_dropout"])
        if loss_name == "reconstruction":
            return sum(F.mse_loss(net.decoders[m](z), xs[m]) for m in mods) / len(mods)
        return F.cross_entropy(net.classifier(z), y_all[rows].to(device))

    @torch.no_grad()
    def eval_loss(rows):
        net.eval()
        vals = []
        g = torch.Generator().manual_seed(0)
        perm = torch.tensor(rows)[torch.randperm(len(rows), generator=g)]
        for i in range(0, len(perm) - 1, hp["batch_size"]):
            b = perm[i:i + hp["batch_size"]]
            if len(b) > 1:
                vals.append(batch_loss(b).item())
        net.train()
        return float(np.mean(vals)) if vals else float("nan")

    early_stop = not uses_labels
    gen = torch.Generator().manual_seed(seed)
    tr_t = torch.tensor(tr)
    history, best, best_state, since, stop = [], float("inf"), None, 0, "max_epochs"
    net.train()
    for epoch in range(1, int(hp["epochs"]) + 1):
        perm = tr_t[torch.randperm(len(tr_t), generator=gen)]
        losses = []
        for i in range(0, len(perm), hp["batch_size"]):
            b = perm[i:i + hp["batch_size"]]
            if len(b) < 2:
                continue
            loss = batch_loss(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if early_stop:
            row["val_loss"] = eval_loss(va)
            if row["val_loss"] < best - 1e-4:
                best, since = row["val_loss"], 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                since += 1
        history.append(row)
        if early_stop and since >= hp["patience"]:
            stop = f"early_stopping (no validation improvement for {hp['patience']} epochs)"
            break
        if (time.time() - t_start) / 60 > max_minutes:
            stop = f"time_cap ({max_minutes:.1f} min)"
            break
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    emb = {}
    with torch.no_grad():
        for m in mods:
            out = []
            for i in range(0, ctx.n, 2048):
                out.append(net.towers[m](X[m][i:i + 2048].to(device)).cpu())
            emb[m] = torch.cat(out).numpy().astype(np.float32)
    if hp.get("concat_input"):
        # Embedding = [unit-norm input features, unit-norm learned embedding]: alignment compresses the
        # input (lower effective rank); keeping the input lets a probe use what the alignment dropped.
        for m in mods:
            raw = features.featurize(ctx, m, entry["inputs"][m]["encoder"], entry["inputs"][m].get("params"))
            unit = lambda Z: Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            emb[m] = np.concatenate([unit(raw), unit(emb[m])], axis=1).astype(np.float32)
    record = {
        "key": run_key(entry, seed, budget), "model_id": entry["id"], "method": method, "loss": loss_name,
        "seed": seed, "budget": budget, "label_use": label_info,
        "inputs": {m: {k: v for k, v in s.items() if k in ("encoder", "head", "params")} for m, s in entry["inputs"].items()},
        "hyperparameters": hp, "device": str(device), "n_train": int(len(tr)), "n_val": int(len(va)),
        "epochs_run": len(history), "stop_reason": stop, "early_stopping": early_stop,
        "best_val_loss": None if best == float("inf") else best,
        "final_temperature": (1 / net.logit_scale.detach().exp().clamp(max=100)).item() if loss_name == "infonce" else None,
        "seconds": round(time.time() - t_start, 1), "history": history,
    }
    return emb, record


def save(ctx, key, emb, record, chash):
    (ctx.out / "embeddings").mkdir(parents=True, exist_ok=True)
    (ctx.out / "runs").mkdir(parents=True, exist_ok=True)
    np.savez(ctx.out / "embeddings" / f"{key}.npz", **emb)
    record["config_hash"] = chash
    (ctx.out / "runs" / f"{key}.json").write_text(json.dumps(record, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Train one plan model at one seed (and label budget).")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", help="per-class label budget (integer or 'all'), for label-using methods")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    plan = json.loads(open(args.plan).read())
    objective = json.loads((context.OUTPUTS / plan["dataset"] / "objective.json").read_text())
    ctx = context.Context(plan["dataset"], plan, objective)
    entry = model_entry(plan, args.model_id)
    budget = None if args.budget is None else (args.budget if args.budget == "all" else int(args.budget))
    emb, rec = train(ctx, entry, args.seed, budget, args.device)
    save(ctx, rec["key"], emb, rec, config_hash(plan, entry, args.seed, budget))
    print(f"[train] {rec['key']}: {rec['epochs_run']} epochs, {rec['stop_reason']}, {rec['seconds']}s on {rec['device']}")


if __name__ == "__main__":
    main()
