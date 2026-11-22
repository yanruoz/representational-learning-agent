"""Figures for one dataset's run, from metrics/summary.json, runs/*.json, and embeddings.

   Usage: venv/bin/python scripts/visualize.py --dataset <d>

   Writes outputs/<d>/figures/*.png (200 dpi; only the figures whose tasks exist):
     label_budget_<label>.png   macro-F1 vs labels per class (linear probe and kNN panels), per model
     matching_<q>-<g>.png       FOSCTTM and recall@k per model with chance
     transfer_<a>-<b>.png       label-transfer macro-F1 per model
     training_curves.png        validation (or training) loss per epoch, seed 0 of each trained model
     collapse.png               effective rank of each trained model's embedding vs its input features
     embedding_2d.png           t-SNE of test embeddings: best trained model vs no-pretraining baseline

   Colors: one categorical hue per model, in plan order, never reused (validated default palette from
   the dataviz skill, light mode). Baselines are gray; chance and majority-class references are
   dashed muted lines. One y-axis per panel.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import context
import plan_schema

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE_GRAYS = ["#5f5e5a", "#8a8984", "#b4b2a9", "#3d3d3a"]
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e6e4dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def colors_for(plan):
    out, k, g = {}, 0, 0
    for m in plan["models"]:
        if m["method"] in plan_schema.BASELINE_METHODS:
            out[m["id"]] = BASELINE_GRAYS[g % len(BASELINE_GRAYS)]
            g += 1
        else:
            out[m["id"]] = PALETTE[k] if k < len(PALETTE) else "#3d3d3a"
            k += 1
    return out


def _series(summary, metric_fn):
    """{model_id: {budget: (mean over seeds, sd over seeds)}} for a metric path built per budget."""
    out = {}
    for name, row in summary.items():
        for key, v in row["metrics"].items():
            b = metric_fn(key)
            if b is not None:
                out.setdefault(row["model_id"], {})[b] = (v["mean"], v["sd"])
    return out


def label_budget_figure(out, summary, label, budgets, colors, task_key="label_prediction"):
    order = [str(b) for b in budgets]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    maj = None
    for ax, probe in zip(axes, ("linear", "knn")):
        prefix = f"{task_key}.{label}.budgets."
        data = _series(summary, lambda k: k[len(prefix):].split(".")[0]
                       if k.startswith(prefix) and k.endswith(f".{probe}_macro_f1") else None)
        for mid, pts in data.items():
            xs = [i for i, b in enumerate(order) if b in pts]
            ys = [pts[order[i]][0] for i in xs]
            es = [pts[order[i]][1] for i in xs]
            if len(xs) == 1:
                ax.errorbar(xs, ys, yerr=es, fmt="o", ms=6, color=colors.get(mid, INK), label=mid, capsize=3, lw=1.5)
            else:
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=2, color=colors.get(mid, INK), label=mid, capsize=3)
        mk = _series(summary, lambda k: k[len(prefix):].split(".")[0]
                     if k.startswith(prefix) and k.endswith(".majority_macro_f1") else None)
        if mk:
            maj = np.mean([v[0] for pts in mk.values() for v in pts.values()])
            ax.axhline(maj, ls="--", lw=1, color=MUTED)
            ax.text(len(order) - 1, maj, " majority class", va="bottom", ha="right", color=MUTED, fontsize=8)
        ax.set_xticks(x, [b if b == "all" else f"{b}/class" for b in order])
        ax.set_title(f"{'Linear probe' if probe == 'linear' else 'kNN classifier'}", color=INK, fontsize=10)
        ax.set_xlabel("labeled training examples")
    axes[0].set_ylabel(f"macro-F1, {label} (test)")
    axes[0].set_ylim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(4, len(l)), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path = out / "figures" / f"{'label' if task_key == 'label_prediction' else 'secondary'}_budget_{label}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def matching_figure(out, summary, pair, colors):
    base = f"cross_modal_retrieval.{pair}."
    rows = [(r["model_id"], r["metrics"]) for r in summary.values() if f"{base}foscttm" in r["metrics"]]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ids = [m for m, _ in rows]
    y = np.arange(len(ids))
    f = [m[f"{base}foscttm"] for _, m in rows]
    axes[0].barh(y, [v["mean"] for v in f], xerr=[v["sd"] for v in f], color=[colors.get(i, INK) for i in ids], height=0.6, capsize=3)
    axes[0].set_yticks(y, ids)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("FOSCTTM (lower is better; chance 0.5)")
    axes[0].set_title("Fraction of samples closer than the true match", fontsize=10, color=INK)
    ks = [k for k in plan_schema.RECALL_KS if f"{base}recall@{k}" in rows[0][1]]
    w = 0.8 / len(ids)
    for j, (mid, m) in enumerate(rows):
        axes[1].bar(np.arange(len(ks)) + j * w, [m[f"{base}recall@{k}"]["mean"] for k in ks], width=w * 0.9,
                    yerr=[m[f"{base}recall@{k}"]["sd"] for k in ks], color=colors.get(mid, INK), label=mid, capsize=2)
    chance = [rows[0][1][f"{base}chance_recall@{k}"]["mean"] for k in ks]
    for i, c in enumerate(chance):
        axes[1].hlines(c, i - 0.05, i + 0.85, ls="--", lw=1, color=MUTED)
    axes[1].set_xticks(np.arange(len(ks)) + 0.4 - w / 2, [f"recall@{k}" for k in ks])
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"{pair}: true match within the top k (dashed: chance)", fontsize=10, color=INK)
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path = out / "figures" / f"matching_{pair.replace('->', '-')}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def transfer_figure(out, summary, pair, colors):
    base = f"label_transfer.{pair}.budgets.all."
    rows = [(r["model_id"], r["metrics"]) for r in summary.values() if f"{base}knn_macro_f1" in r["metrics"]]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ids = [m for m, _ in rows]
    y = np.arange(len(ids))
    for j, probe in enumerate(("knn", "linear")):
        v = [m[f"{base}{probe}_macro_f1"] for _, m in rows]
        ax.barh(y + (j - 0.5) * 0.38, [x["mean"] for x in v], xerr=[x["sd"] for x in v], height=0.36,
                color=[colors.get(i, INK) for i in ids], alpha=1.0 if probe == "knn" else 0.55, capsize=2)
    ax.set_yticks(y, ids)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"macro-F1 of labels transferred {pair} (solid: kNN, light: linear)")
    fig.tight_layout()
    path = out / "figures" / f"transfer_{pair.replace('->', '-')}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def training_curves(out, plan, colors):
    trained = [m for m in plan["models"] if m["method"] not in plan_schema.BASELINE_METHODS]
    if not trained:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for m in trained:
        runs = sorted((out / "runs").glob(f"{m['id']}_s0*.json"))
        if not runs:
            continue
        r = json.loads(runs[-1].read_text())
        h = r["history"]
        key = "val_loss" if h and "val_loss" in h[0] else "train_loss"
        y = np.array([e[key] for e in h])
        y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
        ax.plot([e["epoch"] for e in h], y, lw=2, color=colors.get(m["id"], INK),
                label=f"{m['id']} ({'validation' if key == 'val_loss' else 'training'}; {r['stop_reason'].split(' (')[0]})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss, rescaled to [0, 1] per model")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path = out / "figures" / "training_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def collapse_figure(out, colors):
    p = out / "metrics" / "collapse.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    if not c:
        return None
    labels, inp, lrn, cols = [], [], [], []
    for mid, v in c.items():
        for mod, x in v["inputs"].items():
            learned = [r[mod]["effective_rank"] for r in v["learned"].values() if mod in r]
            labels.append(f"{mid}\n{mod}")
            inp.append(x["effective_rank"])
            lrn.append(np.mean(learned) if learned else np.nan)
            cols.append(colors.get(mid, INK))
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels)), 3.4))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, inp, width=0.38, color="#b4b2a9", label="input features")
    ax.bar(x + 0.2, lrn, width=0.38, color=cols, label="learned embedding (mean over runs)")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("effective rank (test samples)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "figures" / "collapse.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def embedding_2d(ctx, plan, summary, colors):
    from sklearn.manifold import TSNE
    obj = ctx.objective
    label = next((t["label"] for t in obj["tasks"] if t.get("label")), None)
    mod = obj["inference_modalities"][-1]
    primary = next(t for t in obj["tasks"] if t.get("primary"))
    trained = [m["id"] for m in plan["models"] if m["method"] not in plan_schema.BASELINE_METHODS]
    if not trained or label is None:
        return None
    if primary["task"] == "cross_modal_retrieval":
        key, better = f"cross_modal_retrieval.{primary['query']}->{primary['gallery']}.foscttm", min
    else:
        b = str(primary.get("label_budgets_per_class", ["all"])[0])
        key, better = f"label_prediction.{primary['label']}.budgets.{b}.knn_macro_f1", max
    scored = {r["model_id"]: r["metrics"][key]["mean"] for r in summary.values() if key in r["metrics"] and r["model_id"] in trained}
    if not scored:
        return None
    best = better(scored, key=scored.get)
    emb_file = sorted((ctx.out / "embeddings").glob(f"{best}_s0*.npz"))[-1]
    Zb = np.load(emb_file)[mod]
    import features
    base = next((m for m in plan["models"] if m["method"] in plan_schema.BASELINE_METHODS and mod in m["inputs"]
                 and m["inputs"][mod]["encoder"] in plan_schema.NO_PRETRAINING), None)
    panels = [(f"{best} ({mod})", Zb)]
    if base:
        panels.insert(0, (f"{base['id']} ({mod}, no pretraining)", features.featurize(ctx, mod, base["inputs"][mod]["encoder"])))
    te = ctx.labeled(label, ctx.test)
    y = ctx.labels(label)[te].astype(str)
    classes = sorted(set(y), key=lambda c: -np.sum(y == c))
    pal = PALETTE + ["#3d3d3a"]
    cmap = {c: pal[min(i, len(pal) - 1)] for i, c in enumerate(classes)}
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.3))
    axes = np.atleast_1d(axes)
    for ax, (title, Z) in zip(axes, panels):
        P = TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(Z[te])
        for c in classes:
            sel = y == c
            ax.scatter(P[sel, 0], P[sel, 1], s=4, color=cmap[c], label=c if len(classes) <= 8 else None, linewidths=0)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    if len(classes) > 8:
        fig.text(0.5, 0.01, f"{len(classes)} classes of '{label}'; the 8 largest have their own colors, the rest are dark gray",
                 ha="center", fontsize=8, color=MUTED)
    else:
        axes[-1].legend(markerscale=3, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    path = ctx.out / "figures" / "embedding_2d.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="Draw the run's figures.")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    out = context.OUTPUTS / args.dataset
    plan = json.loads((out / "plan.json").read_text())
    obj = json.loads((out / "objective.json").read_text())
    summary = json.loads((out / "metrics" / "summary.json").read_text())
    (out / "figures").mkdir(exist_ok=True)
    colors = colors_for(plan)
    made = []
    for t in obj["tasks"]:
        if t["task"] == "label_prediction":
            made.append(label_budget_figure(out, summary, t["label"], t.get("label_budgets_per_class", ["all"]), colors))
        elif t["task"] == "cross_modal_retrieval":
            made.append(matching_figure(out, summary, f"{t['query']}->{t['gallery']}", colors))
        elif t["task"] == "label_transfer":
            made.append(transfer_figure(out, summary, f"{t['from']}->{t['to']}", colors))
    for t in obj.get("secondary_tasks", []):
        made.append(label_budget_figure(out, summary, t["label"], t.get("label_budgets_per_class", ["all"]), colors,
                                        task_key="secondary_label_prediction"))
    made.append(training_curves(out, plan, colors))
    made.append(collapse_figure(out, colors))
    made.append(embedding_2d(context.Context(args.dataset, plan, obj), plan, summary, colors))
    for p in made:
        if p:
            print(f"[visualize] {p}")


if __name__ == "__main__":
    main()
