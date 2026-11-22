"""Assemble reports/<dataset>/generated_report_<N>.pdf from the artifacts in outputs/<dataset>/.
   The only file in reports/ is the PDF; this script is the only thing that writes there.

   Usage: venv/bin/python scripts/report.py --dataset <d> --report-number <N>

   Pages (built from JSON; the agent writes only findings.md). A Contents page comes first:
     Summary                      objective (verbatim), decision, headline table vs baselines, verdicts
     How to read this report      abbreviations, split, seeds and label draws, what "±" means
     Objective interpretation     objective.json: tasks, inference vs training-only modalities, criterion
     Data and split               profile.json, split sizes, training pool, excluded classes, reasons
     Evidence before choosing     metrics/baseline_check.json (validation split only)
     Methods: encoders and losses every model: method, per-modality encoder and head, training loss,
                                  hyperparameters, each with its reason and the evidence it cites
     Results                      metrics/summary.json per task, mean ± SD over seeds
     Training and collapse        runs/*.json and metrics/collapse.json
     Figures, Findings (findings.md), Downstream use (recommendation.json), Critique, Limitations,
     Reproducibility
   Every decimal number in findings.md must match a value in the JSON artifacts at the precision
   written, or the script exits 1 without writing (--allow-untraced overrides and lists them under
   Limitations). Pages are also saved as PNGs in outputs/<d>/report_pages/ for inspection.
   Text-page machinery (measured-width wrapping, contents, page numbers) is copied from Project 2.
"""
import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

ROOT = Path(__file__).resolve().parent.parent
PAGE_SIZE = (8.5, 11)
plt.rcParams["figure.max_open_warning"] = 0  # a report holds ~20 page figures until written
TEXT_W = 488  # usable line width in points (8.5in page minus margins); lines are wrapped by measured width
LINE_H, MIN_Y = 0.0185, 0.05
INK, MUTED = "#1f1f1e", "#5f5e5a"


# ── Text page helpers (adapted from Project 1's report.py) ──

_T2P = TextToPath()
_FONTS = {"p": FontProperties(family="sans-serif", size=8.6), "m": FontProperties(family="monospace", size=7.2),
          "h": FontProperties(family="sans-serif", size=11, weight="bold"), "c": FontProperties(family="sans-serif", size=8)}


def _width(text, kind="p"):
    return _T2P.get_text_width_height_descent(text.replace("$", "S"), _FONTS[kind], ismath=False)[0]


def _wrap(text, kind="p", indent="", max_w=TEXT_W):
    """Greedy word wrap by rendered width, so wide characters or long headings never run off the page.
    Splits on single spaces, so aligned columns in monospace rows keep their spacing."""
    if _width(text, kind) <= max_w:
        return [text]
    lines, cur = [], ""
    for w in text.split(" "):
        cand = f"{cur} {w}" if cur else w
        if cur.strip() and _width(cand, kind) > max_w:
            lines.append(cur.rstrip())
            cur = indent + w
        else:
            cur = cand
    lines.append(cur.rstrip())
    out = []
    for line in lines:  # a single word wider than the line (e.g. a long path) is cut by characters
        while _width(line, kind) > max_w and len(line) > 1:
            k = len(line)
            while k > 1 and _width(line[:k], kind) > max_w:
                k -= 1
            out.append(line[:k]); line = indent + line[k:]
        out.append(line)
    return out


def _block_groups(heading, paras):
    """One group of lines per paragraph (the heading joins the first), so a short paragraph is never split
    across pages and a heading never sits alone at the bottom of a page."""
    groups = []
    for i, p in enumerate(paras or [""]):
        lines = []
        if heading and i == 0:
            hl = _wrap(heading, "h")
            lines = [("h", hl[0])] + [("hc", l) for l in hl[1:]]  # a long heading continues on the next line
        mono = p.startswith("```")
        body = p.strip("`") if mono else p
        for raw_line in body.splitlines() or [""]:
            if mono:
                lines += [("m", l) for l in _wrap(raw_line, "m", indent="    ")]
            else:
                indent = "  " if raw_line.startswith("- ") else ""
                lines += [("p", l) for l in _wrap(raw_line, indent=indent)]
        lines.append(("sgap" if p.startswith("- ") else "gap", ""))  # list items sit closer together
        groups.append(lines)
    return groups


def _height(kind):
    return {"h": LINE_H * 1.6, "hc": LINE_H * 1.25, "gap": LINE_H * 0.6, "sgap": LINE_H * 0.2}.get(kind, LINE_H)


def text_pages(title, blocks):
    """blocks: [(heading or None, [paragraph, ...])]. Paragraphs starting with ``` render monospace.
    Splits across pages as needed; never silently drops content."""
    pages, cur, y = [], [], 0.0
    capacity = 0.93 - MIN_Y
    for heading, paras in blocks:
        for group in _block_groups(heading, paras):
            g_h = sum(_height(k) for k, _ in group)
            if y + g_h > capacity and cur and g_h <= capacity / 4:  # start short paragraphs on a fresh page
                pages.append(cur); cur, y = [], 0.0
            for kind, text in group:  # long paragraphs still split line by line
                h = _height(kind)
                if y + h > capacity and cur:
                    pages.append(cur); cur, y = [], 0.0
                cur.append((kind, text)); y += h
    pages.append(cur)
    pages = [pg for pg in pages if any(kind not in ("gap", "sgap") for kind, _ in pg)] or [[]]  # no spacing-only pages
    figs = []
    for i, lines in enumerate(pages):
        fig = plt.figure(figsize=PAGE_SIZE)
        ax = fig.add_axes([0.04, 0.02, 0.92, 0.95]); ax.axis("off")  # use the full page, not default axes margins
        ax.text(0.05, 0.97, title + (" (cont.)" if i else ""), transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", color=INK)
        y = 0.93
        for kind, text in lines:
            if kind == "h":
                y -= LINE_H * 0.4
                ax.text(0.05, y, text, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", color=INK)
                y -= LINE_H * 1.2
            elif kind == "hc":
                ax.text(0.05, y, text, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", color=INK)
                y -= LINE_H * 1.25
            elif kind in ("gap", "sgap"):
                y -= _height(kind)
            else:
                ax.text(0.06, y, text, transform=ax.transAxes, va="top", color=INK,
                        fontsize=7.2 if kind == "m" else 8.6, family="monospace" if kind == "m" else "sans-serif")
                y -= LINE_H
        figs.append(fig)
    return figs


def figure_page(png, caption):
    img = plt.imread(png)
    h, w = img.shape[:2]
    fig_w = 8.0
    fig = plt.figure(figsize=(fig_w, min(fig_w * h / w + 0.8, 10.5)))
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.9]); ax.axis("off")
    ax.imshow(img, interpolation="none")  # embed the 300-dpi source pixels as-is, no resampling
    fig.text(0.04, 0.02, "\n".join(_wrap(caption, "c", max_w=fig_w * 72 * 0.92)), fontsize=8, color=MUTED, va="bottom")
    return fig



# ── Number tracing ──

def _load(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()


def _numbers_in(obj, acc):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, acc)




# ── Artifacts ──

def load_all(dataset):
    out = ROOT / "outputs" / dataset
    a = {"out": out, "dataset": dataset}
    for k, f in [("profile", "profile.json"), ("objective", "objective.json"), ("plan", "plan.json"), ("run_log", "run_log.json"),
                 ("recommendation", "recommendation.json"), ("critique", "critique.json"), ("summary", "metrics/summary.json"),
                 ("collapse", "metrics/collapse.json"), ("baseline_check", "metrics/baseline_check.json"), ("contract", "summary.json"),
                 ("split", "split.json")]:
        a[k] = _load(out / f, {} if k not in ("recommendation", "critique") else None)
    a["findings"] = _load(out / "findings.md", "")
    a["desc"] = _load(ROOT / "data" / dataset / "DATA_DESCRIPTION.md", "")
    ok = [r["key"] for r in (a["run_log"] or {}).get("runs", []) if r.get("status") in ("ok", "reused")]
    a["runs"] = {k: _load(out / "runs" / f"{k}.json") for k in ok if (out / "runs" / f"{k}.json").exists()}
    a["metrics"] = {k: _load(out / "metrics" / f"{k}.json") for k in ok if (out / "metrics" / f"{k}.json").exists()}
    return a


def untraced_numbers(text, a):
    pool = []
    for k in ("profile", "summary", "collapse", "baseline_check", "plan", "run_log", "recommendation", "split"):
        _numbers_in(a.get(k) or {}, pool)
    _numbers_in(list(a["metrics"].values()), pool)
    _numbers_in(list(a["runs"].values()), pool)
    missing = []
    for m in re.finditer(r"(?<![\w.])(-?\d+\.\d+)(%?)", text):
        s, pct = m.group(1), m.group(2) == "%"
        q, dec = float(s), len(s.split(".")[1])
        cands = [q, q / 100] if pct else [q]
        if not any(any(round(v, d) == round(c, d) for v in pool) for c, d in zip(cands, [dec, dec + 2])):
            missing.append(m.group(0))
    return sorted(set(missing))


# ── Helpers ──

GLOSSARY = [
    ("kNN", "k-nearest-neighbor classifier (cosine distance) fit on the labeled training embeddings"),
    ("linear probe", "logistic regression on standardized embeddings, fit on the labeled training embeddings"),
    ("macro-F1", "F1 score averaged over classes with equal weight, so small classes count as much as large ones"),
    ("FOSCTTM", "fraction of samples closer than the true match: for each held-out sample, the share of other held-out samples whose other-modality embedding is closer than its own; averaged over both directions. 0 is perfect, 0.5 is chance"),
    ("recall@k", "share of held-out queries whose true partner is among the k nearest in the other modality"),
    ("ARI / NMI", "adjusted Rand index / normalized mutual information between KMeans clusters and the labels"),
    ("effective rank", "exp of the entropy of the normalized singular values of an embedding; how many directions carry variance. Far below the input's suggests collapse"),
    ("PCA / LSI / SVD", "principal component analysis / latent semantic indexing (TF-IDF then truncated singular value decomposition)"),
    ("TF-IDF", "term frequency times inverse document frequency weighting"),
    ("CCA", "canonical correlation analysis: the linear projections of two modalities that are most correlated (linear alignment baseline)"),
    ("PLS", "partial least squares regression (linear supervised baseline for cross-modal prediction)"),
    ("InfoNCE", "contrastive loss (CLIP-style): each sample's other modality is its positive, other samples in the batch are negatives"),
    ("VICReg", "variance-invariance-covariance loss: pulls paired embeddings together without negatives; variance and covariance terms prevent collapse"),
    ("SupCon", "supervised contrastive loss: same-label samples are also positives (uses labels in training)"),
    ("budget", "number of labeled training examples per class used to fit a probe (and, for label-using losses, the model)"),
]


def _f(v, d=4):
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def _ms(m):
    return f"{m['mean']:.4f} ± {m['sd']:.4f}" + (f" (n={m['n']})" if m.get("n", 1) > 1 else "")


def _primary_metric(a):
    obj = a["objective"]
    t = next((t for t in obj.get("tasks", []) if t.get("primary")), None)
    if not t:
        return None, None, None
    if t["task"] == "cross_modal_retrieval":
        return f"cross_modal_retrieval.{t['query']}->{t['gallery']}.foscttm", "FOSCTTM (lower is better)", t
    if t["task"] == "label_transfer":
        return f"label_transfer.{t['from']}->{t['to']}.budgets.all.knn_macro_f1", "transfer kNN macro-F1", t
    if t["task"] == "label_prediction":
        b = str(t.get("label_budgets_per_class", ["all"])[0])
        return f"label_prediction.{t['label']}.budgets.{b}.linear_macro_f1", f"linear-probe macro-F1 at budget {b}", t
    if t["task"] == "cross_modal_prediction":
        return f"cross_modal_prediction.{t['target']}.pearson_median", "median per-feature Pearson", t
    return None, None, t


def _models(a):
    return {m["id"]: m for m in a["plan"].get("models", [])}


# ── Pages ──

def summary_page(a, n):
    obj, rec = a["objective"], a["recommendation"] or {}
    key, desc, t = _primary_metric(a)
    lines = [f"Dataset: {a['dataset']}. Report {n}.", "", "Objective (verbatim):", obj.get("objective_text", ""), "",
             f"Decision: {obj.get('decision')}. {obj.get('decision_reason', '')}",
             f"Inference modalities (inputs for new samples): {', '.join(obj.get('inference_modalities', []))}. "
             f"Training-only modalities: {', '.join(obj.get('training_only_modalities', [])) or 'none'}."]
    sc = obj.get("success_criterion") or {}
    if sc:
        lines.append(f"Success criterion: {sc.get('metric')} vs {sc.get('baseline')}: {sc.get('comparison')}.")
    blocks = [(None, lines)]
    if key:
        rows = []
        for name, r in (a["summary"] or {}).items():
            if key in r["metrics"]:
                kind = "baseline" if r["method"] in ("single_modality_features", "linear_supervised", "frozen_probe", "linear_alignment") else "learned"
                rows.append(f"{name:28s} {kind:9s} {_ms(r['metrics'][key])}")
        blocks.append((f"Primary result: {desc}", ["```" + "\n".join(rows) + "```",
                                                     "Mean ± SD over seeds on held-out test samples; n = number of seeds. Baselines have no trainable weights (one run)."]))
    if rec:
        blocks.append(("Recommendation", [f"Chosen representation: {rec.get('chosen_model')}. {rec.get('chosen_reason', '')}"] +
                       [f"- {u['verdict'].replace('_', ' ')}: {u['use']}" for u in rec.get("uses", [])]))
    return text_pages("Summary", blocks)


def guide_pages(a):
    sp = (a["split"] or {}).get("sizes", {})
    lines = [f"Split: {sp.get('train')} training, {sp.get('val')} validation, {sp.get('test')} held-out test samples. "
             "Featurizers and models are fit on training (+ validation for featurizers); validation picks epochs and CCA size; "
             "every reported result is on held-out test samples, which never touch fitting or model selection.",
             "Seeds: each learned model is trained with several seeds; tables show mean ± SD over seeds. "
             "Label budgets: probes are fit on seeded, class-stratified draws of k labeled training samples per class "
             "(several draws for small k); the draw SD is in the per-run metrics files.",
             "Baselines: no-pretraining features (e.g. PCA, LSI, TF-IDF, pixel PCA), frozen pretrained encoders, "
             "CCA (linear alignment), and PLS / logistic regression (linear supervised), as the plan lists."]
    return text_pages("How to read this report", [("Method of evaluation", lines),
                                                   ("Abbreviations and terms", [f"- {k}: {v}" for k, v in GLOSSARY])])


def objective_page(a):
    obj = a["objective"]
    tl = []
    for t in obj.get("tasks", []):
        extra = ", ".join(f"{k}: {v}" for k, v in t.items() if k not in ("task", "primary"))
        tl.append(f"- {t['task']}{' (primary)' if t.get('primary') else ''}: {extra}")
    for t in obj.get("secondary_tasks", []):
        tl.append(f"- secondary {t['task']}: label {t.get('label')}, modality {t.get('modality', 'inference')}")
    sc = obj.get("success_criterion") or {}
    blocks = [("Tasks", tl), ("Success criterion", [f"Metric: {sc.get('metric')}", f"Baseline: {sc.get('baseline')}",
                                                    f"Comparison: {sc.get('comparison')}", f"Reason: {sc.get('reason')}"])]
    if obj.get("interpretation_notes"):
        blocks.append(("Interpretation notes", [obj["interpretation_notes"]]))
    return text_pages("Objective interpretation", blocks)


def data_page(a):
    p, plan = a["profile"], a["plan"]
    lines = [f"{p.get('samples', {}).get('n')} samples; paired modalities: {', '.join(p.get('samples', {}).get('paired_modalities', []))}."]
    for m, d in p.get("modalities", {}).items():
        if d["kind"] == "counts":
            lines.append(f"- {m} (counts): {d['n_features']} features, sparsity {d['sparsity']}, integer-valued {d['integer_valued']}")
        elif d["kind"] == "text":
            lines.append(f"- {m} (text): median {d['words_per_text']['p50']} words per text, vocabulary {d['vocabulary_size']}")
        else:
            lines.append(f"- {m} (image): median size {d['width']['p50']} x {d['height']['p50']} px")
    for lname, l in p.get("labels", {}).items():
        lines.append(f"- label {lname}: {l['n_classes']} classes, {l['n_labeled']} labeled, {l['n_missing']} missing; counts {l['class_counts']}")
    sp, pool = plan.get("split", {}), plan.get("training_pool", {})
    dec = [f"Split: {json.dumps({k: v for k, v in sp.items() if k != 'reason'})}. Reason: {sp.get('reason')}",
           f"Training pool for representation learning: {pool.get('filter')}{' (' + pool['label'] + ')' if pool.get('label') else ''}. Reason: {pool.get('reason')}"]
    for lab, ex in (plan.get("excluded_classes") or {}).items():
        dec.append(f"Excluded classes of {lab}: {ex.get('classes')}. Reason: {ex.get('reason')}")
    for lab, bn in (plan.get("budget_not_filled") or {}).items():
        dec.append(f"Budget not filled for {lab}: {bn.get('classes')}. Reason: {bn.get('reason')}")
    return text_pages("Data and split", [("Data", lines), ("Decisions", dec)])


def evidence_pages(a):
    b = a["baseline_check"] or {}
    if not b:
        return []
    blocks = [(None, [b.get("note", "")])]
    for m, encs in b.get("encoders", {}).items():
        rows = []
        for enc, e in encs.items():
            lab = "; ".join(f"{label}: " + ", ".join(f"b{bud} lin {r['linear_macro_f1']['mean']:.3f} / kNN {r['knn_macro_f1']['mean']:.3f}"
                                                      for bud, r in budgets.items()) for label, budgets in e.get("labels", {}).items())
            rows.append(f"- {enc} (dim {e['dim']}, effective rank {e['effective_rank']}{', no pretraining' if e['no_pretraining'] else ''}){': ' + lab if lab else ''}")
        blocks.append((f"Candidate encoders for {m} (validation macro-F1)", rows))
    for pair, combos in b.get("cross_modal", {}).items():
        blocks.append((f"Linear (CCA) matching {pair} on validation", [f"- {c}: best FOSCTTM {v.get('best_foscttm')} with {v.get('best_components')} components"
                                                                         for c, v in combos.items()]))
    for enc, d in b.get("zero_shot", {}).items():
        blocks.append((f"Zero-shot matching, {enc}", [f"- {k}: FOSCTTM {v['foscttm']}, recall@10 {v['recall@10']}" for k, v in d.items()]))
    red = b.get("redundancy", {})
    if red:
        blocks.append(("Redundancy (near-duplicates among the 10 nearest neighbors)", [
            f"- {k}: {v['mean_fraction_of_knn_that_are_near_duplicates']} of neighbors are near-duplicates (random pairs: {v.get('random_pairs_at_or_above_cutoff')})"
            for k, v in red.items()]))
    return text_pages("Evidence before choosing (baseline check)", blocks)


def _ev(items):
    return "; ".join(f"{i.get('file')}:{i.get('key')} = {i.get('value')}" for i in items or [])


def methods_pages(a):
    blocks = []
    for m in a["plan"].get("models", []):
        lines = [f"Method: {m['method']}. Status: {m.get('status', 'planned')}. Seeds: {m.get('seeds', [0])}.", f"Why: {m.get('reason', '')}"]
        for mod, s in m.get("inputs", {}).items():
            lines.append(f"- {mod}: encoder {s.get('encoder')}" + (f", head {s['head']}" if s.get("head") else "") + f". Reason: {s.get('reason', '')}")
            if s.get("evidence"):
                lines.append(f"  Evidence: {_ev(s['evidence'])}")
        if m.get("loss"):
            lines.append(f"- training loss: {m['loss']}. Reason: {m.get('loss_reason', '')}")
            if m.get("loss_evidence"):
                lines.append(f"  Evidence: {_ev(m['loss_evidence'])}")
        for k, v in (m.get("hyperparameters") or {}).items():
            lines.append(f"- {k} = {v}: {(m.get('hyperparameter_reasons') or {}).get(k, '')}")
        blocks.append((f"{m['id']}", lines))
    rev = a["plan"].get("critique_applied")
    if rev:
        blocks.append(("Revision after the critique", [f"Plan revision {a['plan'].get('revision')}: {rev}"]))
    return text_pages("Methods: encoders, losses, and why", blocks)


def results_pages(a):
    summ = a["summary"] or {}
    groups = {}
    for name, r in summ.items():
        for k, v in r["metrics"].items():
            parts = k.split(".")
            if parts[0] == "clustering" or k.endswith(("_accuracy", "chance_foscttm", "median_rank", "n", "n_test", "draws", "n_labeled")):
                continue
            if "per_class" in k or "chance_recall" in k or ".per_class." in k:
                continue
            groups.setdefault(".".join(parts[:2]), {}).setdefault(name, {})[".".join(parts[2:])] = v
    blocks = []
    for g, rows in groups.items():
        lines = []
        for name, mets in rows.items():
            lines.append(f"{name}:")
            for k, v in sorted(mets.items()):
                if any(x in k for x in ("macro_f1", "foscttm", "recall@", "pearson_median", "ari", "nmi")):
                    lines.append(f"    {k:42s} {_ms(v)}")
        blocks.append((g.replace(".", ": "), ["```" + "\n".join(lines) + "```"]))
    clus = [(n, r["metrics"]) for n, r in summ.items() if any(k.startswith("clustering") for k in r["metrics"])]
    if clus:
        blocks.append(("clustering (KMeans on test embeddings)", ["```" + "\n".join(
            f"{n:28s} " + "  ".join(f"{k.split('.')[-1]} {_ms(v)}" for k, v in m.items() if k.startswith("clustering") and k.endswith(("ari", "nmi")))
            for n, m in clus) + "```"]))
    return text_pages("Results (held-out test samples)", blocks)


def training_pages(a):
    lines = []
    for k, r in a["runs"].items():
        if "history" not in r:
            continue
        t = f", final temperature {r['final_temperature']:.3f}" if r.get("final_temperature") else ""
        lu = r.get("label_use")
        lab = f", {lu['labels_used']} labels used ({lu['label']}, budget {lu['budget']})" if lu else ""
        lines.append(f"- {k}: {r['epochs_run']} epochs, {r['stop_reason']}, {r['n_train']} training pairs, {r['seconds']} s{t}{lab}")
    col = []
    for mid, v in (a["collapse"] or {}).items():
        for mod, x in v["inputs"].items():
            ranks = [r[mod]["effective_rank"] for r in v["learned"].values() if mod in r]
            low = [r[mod]["low_variance_dim_fraction"] for r in v["learned"].values() if mod in r]
            col.append(f"- {mid} / {mod}: input effective rank {x['effective_rank']} (dim {x['dim']}); learned {min(ranks) if ranks else '-'}-{max(ranks) if ranks else '-'}"
                       f" (dim {next(iter(v['learned'].values()))[mod]['dim'] if v['learned'] else '-'}); near-constant dimensions {max(low) if low else '-'}")
    fails = [f"- {r['key']}: {r.get('error')}" for r in (a["run_log"] or {}).get("runs", []) if r.get("status") == "failed"]
    blocks = [("Training runs", lines or ["(no trained models)"]), ("Collapse check", col or ["(no trained models)"])]
    if fails:
        blocks.append(("Failed runs", fails))
    return text_pages("Training and collapse", blocks)


CAPTIONS = {
    "label_budget": "Macro-F1 on held-out test samples vs labeled training examples per class, linear probe (left) and kNN (right). Points: mean over seeds; bars: SD over seeds. Dashed: majority class.",
    "secondary_budget": "Secondary label, same layout: frozen and learned representations, held-out test samples.",
    "matching": "Held-out matching between modalities. Left: FOSCTTM (0 perfect, 0.5 chance). Right: recall@k with chance (dashed). Bars: SD over seeds.",
    "transfer": "Labels transferred across modalities: classifier fit on one modality's training embeddings, applied to the other modality's test embeddings.",
    "training_curves": "Loss per epoch for seed 0 of each trained model, rescaled to [0, 1] per model; the legend gives the stopping reason.",
    "collapse": "Effective rank of input features vs the learned embedding (test samples). A large drop is compression; near zero would be collapse.",
    "embedding_2d": "t-SNE of held-out test embeddings colored by label: no-pretraining baseline vs the best trained model on the primary metric.",
}


def figure_pages(a):
    figs = []
    for png in sorted((a["out"] / "figures").glob("*.png")):
        cap = next((v for k, v in CAPTIONS.items() if png.stem.startswith(k)), png.stem)
        figs.append(figure_page(png, cap))
    return figs


def findings_pages(a):
    text = a["findings"] or "(findings.md was not written)"
    blocks, head, paras = [], None, []
    for line in text.splitlines():
        if line.startswith("#"):
            if head or paras:
                blocks.append((head, paras))
            head, paras = line.lstrip("# ").strip(), []
        elif line.strip():
            paras.append(line.strip())
    blocks.append((head, paras))
    return text_pages("Findings", blocks)


def recommendation_pages(a):
    rec = a["recommendation"]
    if not rec:
        return []
    lines = [f"Chosen representation: {rec.get('chosen_model')}. {rec.get('chosen_reason', '')}", f"Evidence: {_ev(rec.get('chosen_evidence'))}"]
    blocks = [(None, lines)]
    for u in rec.get("uses", []):
        blocks.append((f"{u['verdict'].replace('_', ' ').capitalize()}: {u['use']}", [u.get("reason", ""), f"Evidence: {_ev(u.get('evidence'))}"]))
    c = a["contract"] or {}
    if c:
        blocks.append(("Files for downstream tools", [f"- embeddings: {c.get('embeddings')} (rows: {c.get('embedding_index')})", f"- summary: outputs/{a['dataset']}/summary.json"]))
    return text_pages("Downstream use", blocks)


def critique_pages(a):
    c = a["critique"]
    if not c:
        return text_pages("Critique and revision", [(None, ["No critique.json was written; the critic did not run or wrote nothing."])])
    blocks = [(None, [c.get("summary", "")])]
    for f in c.get("findings", []):
        blocks.append((f"{f.get('check', '')}: {f.get('severity', '')}", [f.get("finding", ""), f"Suggested fix: {f.get('suggested_fix', '-')}"]))
    return text_pages("Critique and revision", blocks)


def limitations_pages(a, untraced):
    auto = []
    for enc, c in (a["profile"].get("cache") or {}).items():
        used = any(s.get("encoder") == f"cache:{enc}" for m in a["plan"].get("models", []) for s in m.get("inputs", {}).values())
        if used:
            auto.append(f"- Frozen encoder {enc} was pretrained on: {c.get('pretraining_data')} Check the data description for overlap with this dataset.")
    caveats = re.search(r"^##\s+Known caveats\s*$(.*?)(?=^##\s|\Z)", a["desc"] or "", flags=re.M | re.S)
    if caveats:
        auto.append("Known caveats of the dataset (from its data description):")
        auto += [ln for ln in caveats.group(1).strip().splitlines() if ln.strip()]
    for r in (a["run_log"] or {}).get("runs", []):
        if r.get("status") in ("not_run_budget", "not_run_unavailable", "failed"):
            auto.append(f"- {r.get('model_id') or r.get('key')}: {r['status']} ({r.get('reason') or r.get('error', '')})")
    for k, r in a["runs"].items():
        if r.get("stop_reason", "").startswith(("max_epochs", "time_cap")) and r.get("early_stopping"):
            auto.append(f"- {k} stopped at {r['stop_reason']}; validation loss may still have been improving.")
    if untraced:
        auto.append(f"- findings.md contains numbers not found in any artifact: {', '.join(untraced)}")
    own = re.search(r"^##\s+Limitations\s*$(.*?)(?=^##\s|\Z)", a["findings"] or "", flags=re.M | re.S)
    blocks = [("Collected automatically", auto or ["(none)"])]
    if own:
        blocks.append(("Stated in the findings", [ln.strip() for ln in own.group(1).strip().splitlines() if ln.strip()]))
    return text_pages("Limitations", blocks)


def repro_pages(a, n):
    try:
        import numpy, sklearn, torch
        versions = f"python {platform.python_version()}, torch {torch.__version__}, numpy {numpy.__version__}, scikit-learn {sklearn.__version__}"
    except Exception:
        versions = f"python {platform.python_version()}"
    rl = a["run_log"] or {}
    lines = [f"Command: ./run.sh {a['dataset']} \"<objective above>\" --report-number {n}",
             f"Plan revision {a['plan'].get('revision')}; budget {a['plan'].get('budget_min')} min; model runs took {rl.get('minutes')} min.",
             f"Split key {(a['split'] or {}).get('key')}; {versions}."]
    lines += [f"- {k}: config hash {r.get('config_hash', '-')}, {r.get('seconds', '-')} s on {r.get('device', 'cpu')}" for k, r in a["runs"].items()]
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')}. All intermediate files: outputs/{a['dataset']}/.")
    return text_pages("Reproducibility", [(None, lines)])


def contents_page(sections, n, dataset):
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_axes([0.04, 0.02, 0.92, 0.95]); ax.axis("off")
    ax.text(0.05, 0.97, f"Generated report {n}: {dataset}", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", color=INK)
    ax.text(0.05, 0.925, "Contents", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", color=INK)
    y, page = 0.88, 2
    for title, figs in sections:
        if not figs:
            continue
        ax.text(0.07, y, title, transform=ax.transAxes, fontsize=10, va="top", color=INK)
        ax.text(0.90, y, str(page), transform=ax.transAxes, fontsize=10, va="top", ha="right", color=INK)
        y -= 0.032
        page += len(figs)
    return [fig]


def build(dataset, n, allow_untraced=False):
    a = load_all(dataset)
    if not a["plan"] or not a["objective"]:
        sys.exit("ERROR: plan.json and objective.json are required")
    untraced = untraced_numbers(a["findings"] or "", a)
    if untraced and not allow_untraced:
        print("UNTRACED numbers in findings.md (not found in any artifact at the precision written):")
        for u in untraced:
            print(f"  {u}")
        sys.exit("ERROR: fix findings.md so every number comes from the JSON artifacts, then rerun.")
    sections = [("Summary", summary_page(a, n)), ("How to read this report", guide_pages(a)),
                ("Objective interpretation", objective_page(a)), ("Data and split", data_page(a)),
                ("Evidence before choosing", evidence_pages(a)), ("Methods: encoders, losses, and why", methods_pages(a)),
                ("Results", results_pages(a)), ("Training and collapse", training_pages(a)), ("Figures", figure_pages(a)),
                ("Findings", findings_pages(a)), ("Downstream use", recommendation_pages(a)),
                ("Critique and revision", critique_pages(a)), ("Limitations", limitations_pages(a, untraced)),
                ("Reproducibility", repro_pages(a, n))]
    pages = contents_page(sections, n, dataset)
    for _, figs in sections:
        pages += figs
    for i, fig in enumerate(pages, 1):
        fig.text(0.5, 0.008, f"{i} / {len(pages)}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    pdf_path = ROOT / "reports" / dataset / f"generated_report_{n}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_dir = a["out"] / "report_pages"
    png_dir.mkdir(exist_ok=True)
    for old in png_dir.glob("page_*.png"):
        old.unlink()
    with PdfPages(pdf_path) as pdf:
        for i, fig in enumerate(pages, 1):
            pdf.savefig(fig)
            fig.savefig(png_dir / f"page_{i:02d}.png", dpi=70)
            plt.close(fig)
    if a["contract"]:
        c = a["contract"]
        c["report"] = f"reports/{dataset}/generated_report_{n}.pdf"
        (a["out"] / "summary.json").write_text(json.dumps(c, indent=1) + "\n")
    print(f"Wrote {pdf_path} ({len(pages)} pages); page images in {png_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the generated report PDF for one dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report-number", type=int, required=True)
    parser.add_argument("--allow-untraced", action="store_true", help="build even if findings.md has untraceable numbers (listed under Limitations)")
    args = parser.parse_args()
    build(args.dataset, args.report_number, args.allow_untraced)
