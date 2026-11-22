"""Assemble reports/<dataset>/generated_report_<N>.pdf from the artifacts in outputs/<dataset>/.
   This script is the only thing that writes reports/.

   Usage: venv/bin/python scripts/report.py --dataset <d> --report-number <N>

   Structure (each fact stated once; target 10-12 pages, hard cap MAX_PAGES):
     Contents
     Summary            objective (verbatim), short answer (findings.md "## Answer"), primary-task table,
                        chosen representation and one line per use verdict
     Setup              objective interpretation, data, split, classes; abbreviations
     Decisions          one table: decision | choice | evidence | alternative rejected (internal ids only
                        here); each distinct reason stated once below it
     Results            primary-task table first, then the other tasks, secondary labels, collapse.
                        Readable model labels; identical baselines merged; 3 significant figures
     Figures            two per page, each caption with a one-sentence takeaway computed from the data
     Findings           findings.md (evidence sections; its Answer and Limitations go elsewhere)
     Downstream use     use cases, verdict, one line of reason, no numbers restated
     Critique           table: finding | severity | what changed | effect on the final result
                        (critique_response.json, final numbers only; the full critique stays in critique.json)
     Limitations        one section: data caveats, pretraining overlap, run issues, findings.md's own list
     Reproducibility    half a page pointing to the files that hold logs, hashes, hyperparameters
   Machine-oriented material (evidence path = value lines, per-seed logs, config hashes,
   hyperparameter reasons) goes to deliverables files, not the PDF (scripts/package_deliverables.py).

   Every decimal number in findings.md and critique_response.json must match a value in the JSON
   artifacts at the precision written, or the script exits 1 without writing (--allow-untraced
   overrides and lists them under Limitations). Above MAX_PAGES it also exits 1 without writing.
   Pages are saved as PNGs in outputs/<d>/report_pages/. Text-page machinery is from Project 2.
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

import labels
import plan_schema

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

MAX_PAGES = 15


def load_all(dataset):
    out = ROOT / "outputs" / dataset
    a = {"out": out, "dataset": dataset}
    for k, f in [("profile", "profile.json"), ("objective", "objective.json"), ("plan", "plan.json"), ("run_log", "run_log.json"),
                 ("recommendation", "recommendation.json"), ("critique", "critique.json"), ("critique_response", "critique_response.json"),
                 ("summary", "metrics/summary.json"), ("collapse", "metrics/collapse.json"), ("baseline_check", "metrics/baseline_check.json"),
                 ("contract", "summary.json"), ("split", "split.json")]:
        a[k] = _load(out / f, None if k in ("recommendation", "critique", "critique_response") else {})
    a["findings"] = _load(out / "findings.md", "")
    a["desc"] = _load(ROOT / "data" / dataset / "DATA_DESCRIPTION.md", "")
    ok = [r["key"] for r in (a["run_log"] or {}).get("runs", []) if r.get("status") in ("ok", "reused")]
    a["runs"] = {k: _load(out / "runs" / f"{k}.json") for k in ok if (out / "runs" / f"{k}.json").exists()}
    a["metrics"] = {k: _load(out / "metrics" / f"{k}.json") for k in ok if (out / "metrics" / f"{k}.json").exists()}
    a["labels"] = labels.model_labels(a["plan"], a["objective"])
    return a


def number_pool(a):
    pool = []
    for k in ("profile", "summary", "collapse", "baseline_check", "plan", "run_log", "recommendation", "split"):
        _numbers_in(a.get(k) or {}, pool)
    _numbers_in(list(a["metrics"].values()), pool)
    _numbers_in(list(a["runs"].values()), pool)
    return pool


def untraced_numbers(text, a, pool=None):
    pool = number_pool(a) if pool is None else pool
    missing = []
    for m in re.finditer(r"(?<![\w.])(-?\d+\.\d+)(%?)", text):
        s_, pct = m.group(1), m.group(2) == "%"
        q, dec = float(s_), len(s_.split(".")[1])
        cands = [q, q / 100] if pct else [q]
        if not any(any(round(v, d) == round(c, d) for v in pool) for c, d in zip(cands, [dec, dec + 2])):
            missing.append(m.group(0))
    return sorted(set(missing))


def sections_of(md):
    """{heading: text} of a markdown file's '## ' sections."""
    out, head, buf = {}, None, []
    for line in (md or "").splitlines():
        if line.startswith("## "):
            if head is not None:
                out[head] = "\n".join(buf).strip()
            head, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if head is not None:
        out[head] = "\n".join(buf).strip()
    return out


def _paras(text):
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


GLOSSARY = [
    ("kNN / linear probe", "nearest-neighbor classifier (cosine) / logistic regression, fit on labeled training embeddings"),
    ("macro-F1", "F1 averaged over classes with equal weight"),
    ("FOSCTTM", "fraction of samples closer than the true match (0 perfect, 0.5 chance)"),
    ("recall@k", "share of queries whose true partner is among the k nearest in the other modality"),
    ("ARI / NMI", "agreement of KMeans clusters with the labels"),
    ("effective rank", "how many directions carry an embedding's variance; far below its input's suggests collapse"),
    ("PCA, LSI, TF-IDF, CCA, PLS", "principal components; TF-IDF + truncated SVD; term weighting; canonical correlation (linear alignment); partial least squares"),
    ("InfoNCE / VICReg / SupCon", "contrastive loss with in-batch negatives / loss without negatives (variance and covariance terms) / contrastive loss with same-label positives"),
    ("budget", "labeled training examples per class used to fit a probe"),
]


def _primary(a):
    t = next((t for t in a["objective"].get("tasks", []) if t.get("primary")), None)
    if not t:
        return None, None
    if t["task"] == "cross_modal_retrieval":
        return f"cross_modal_retrieval.{t['query']}->{t['gallery']}", t
    if t["task"] == "label_transfer":
        return f"label_transfer.{t['from']}->{t['to']}", t
    if t["task"] == "label_prediction":
        return f"label_prediction.{t['label']}", t
    return f"{t['task']}.{t.get('target', '')}", t


def _table(header, rows, w0=None, w=13):
    w0 = w0 or min(52, max([len(header[0])] + [len(r[0]) for r in rows]) + 2)
    rows = [[r[0] if len(r[0]) < w0 else r[0][: w0 - 2] + "…"] + list(r[1:]) for r in rows]
    lines = [f"{header[0]:{w0}s}" + "".join(f"{h:>{w}s}" for h in header[1:])]
    lines += [f"{r[0]:{w0}s}" + "".join(f"{c:>{w}s}" for c in r[1:]) for r in rows]
    return "```" + "\n".join(lines) + "```"


def _group(a, prefix):
    """Merged rows [(label, {subkey: value})] for metrics under one task prefix."""
    rows = []
    for lab, m in labels.merge_identical(a["summary"] or {}, a["labels"]):
        sub = {k[len(prefix) + 1:]: v for k, v in m.items() if k.startswith(prefix + ".")}
        if sub:
            rows.append((lab, sub))
    return rows


def task_table(a, prefix, probe_pref=None, one_probe=False):
    rows = _group(a, prefix)
    if not rows:
        return []
    keys = set().union(*[set(m) for _, m in rows])
    budgets = sorted({k.split(".")[1] for k in keys if k.startswith("budgets.")}, key=lambda b: 10**9 if b == "all" else int(b))
    paras = []
    if budgets:
        probes = [("linear_macro_f1", "linear probe"), ("knn_macro_f1", "kNN")]
        if probe_pref == "knn":
            probes.reverse()
        if one_probe:
            probes = probes[:1]
        for probe, pname in probes:
            trs = [[lab] + [labels.cell(m.get(f"budgets.{b}.{probe}")) for b in budgets] for lab, m in rows
                   if any(f"budgets.{b}.{probe}" in m for b in budgets)]
            if trs:
                paras.append(_table([f"macro-F1, {pname}"] + ["all" if b == "all" else f"{b}/class" for b in budgets], trs))
        m0 = rows[0][1]
        maj = [f"{'all' if b == 'all' else b + '/class'} {labels.sig3(m0[f'budgets.{b}.majority_macro_f1']['mean'])}"
               for b in budgets if f"budgets.{b}.majority_macro_f1" in m0]
        if maj:
            paras.append("Majority class: " + ", ".join(maj) + ".")
    else:
        cols = [c for c in ("foscttm", "recall@1", "recall@10", "recall@50", "pearson_median", "ari", "nmi") if c in keys]
        paras.append(_table(["model"] + cols, [[lab] + [labels.cell(m.get(c)) for c in cols] for lab, m in rows]))
        ch = [f"{c.replace('chance_', '')} {labels.sig3(rows[0][1][c]['mean'])}" for c in sorted(keys) if c.startswith("chance_")]
        if ch:
            paras.append("Chance: " + ", ".join(ch) + ".")
    return paras


# ── Pages ──

def summary_page(a, n):
    obj, rec = a["objective"], a["recommendation"] or {}
    answer = sections_of(a["findings"]).get("Answer", "")
    sc = obj.get("success_criterion") or {}
    blocks = [(None, ["Objective (verbatim): " + obj.get("objective_text", "")]),
              ("Answer", _paras(answer) or ["(findings.md has no Answer section)"])]
    blocks.append(("Success criterion", [f"{sc.get('metric')}" + (f", {sc['probe']} probe" if sc.get("probe") else "")
                                         + f", against {sc.get('baseline')}: {sc.get('comparison')}. The full primary-task table opens the Results section."]))
    if rec:
        blocks.append(("Recommendation", [f"Chosen representation: {a['labels'].get(rec.get('chosen_model'), rec.get('chosen_model'))}."] +
                       [f"- {u['verdict'].replace('_', ' ')}: {u['use']}" for u in rec.get("uses", [])]))
    return text_pages("Summary", blocks)


def setup_pages(a):
    obj, p, plan = a["objective"], a["profile"], a["plan"]
    tasks = [f"- {t['task'].replace('_', ' ')}{' (primary)' if t.get('primary') else ''}: " +
             ", ".join(f"{k} {v}" for k, v in t.items() if k not in ("task", "primary")) for t in obj.get("tasks", [])]
    tasks += [f"- secondary: {t['task'].replace('_', ' ')} of {t.get('label')}" for t in obj.get("secondary_tasks", [])]
    data = [f"{p.get('samples', {}).get('n')} paired samples; modalities: " + ", ".join(
        f"{m} ({d['kind']})" for m, d in p.get("modalities", {}).items()) + ". Inference (test-time) modalities: "
        + ", ".join(obj.get("inference_modalities", [])) + "; training-only: " + (", ".join(obj.get("training_only_modalities", [])) or "none") + "."]
    sp = (a["split"] or {}).get("sizes", {})
    data.append(f"Split ({plan.get('split', {}).get('source')}): {sp.get('train')} training, {sp.get('val')} validation, {sp.get('test')} test. "
                "Validation picks epochs and the CCA size and feeds the evidence check; every reported result is on the test samples.")
    for lname, l in p.get("labels", {}).items():
        data.append(f"Label {lname}: {l['n_classes']} classes, {l['n_labeled']} labeled. Excluded: "
                    + (", ".join((plan.get("excluded_classes") or {}).get(lname, {}).get("classes", [])) or "none") + ".")
    return text_pages("Setup", [("Tasks", tasks), ("Data and split", data),
                                ("Abbreviations", [f"- {k}: {v}" for k, v in GLOSSARY])])


def _first_sentence(text):
    """First sentence; extended to the second when the first is a short lead-in ("Judgment call.")."""
    text = (text or "").strip()
    ends = [m.start() for m in re.finditer(r"(?<=[.!?])\s", text)]
    if not ends:
        return text
    cut = ends[0] if ends[0] >= 40 or len(ends) == 1 else ends[1]
    return text[:cut]


KEY_WORDS = {"mean_fraction_of_knn_that_are_near_duplicates": "near-duplicate neighbors", "samples_with_a_near_duplicate": "samples with a near-duplicate",
             "random_pairs_at_or_above_cutoff": "random pairs above cutoff", "best_foscttm": "CCA FOSCTTM", "foscttm": "FOSCTTM",
             "linear_macro_f1": "linear macro-F1", "knn_macro_f1": "kNN macro-F1", "effective_rank": "effective rank"}


def _ev_short(items, k=2):
    out = []
    for i in (items or [])[:k]:
        parts = [x for x in str(i.get("key", "")).split(".") if x not in ("mean", "metrics")]
        word = KEY_WORDS.get(parts[-1], parts[-1].replace("_", " ")) if parts else ""
        v = i.get("value")
        out.append(f"{word} {labels.sig3(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v}")
    return "; ".join(out) or "-"


def decisions(a):
    """Rows (decision, choice, evidence, alternative rejected) and the distinct reasons, once each."""
    plan, b = a["plan"], a["baseline_check"] or {}
    rows, reasons = [], {}

    def add_reason(text, who):
        if text:
            reasons.setdefault(_first_sentence(text), []).append(who)

    sp = plan.get("split", {})
    sz = (a["split"] or {}).get("sizes", {})
    rows.append(("split", sp.get("source", ""), f"{sz.get('train')}/{sz.get('val')}/{sz.get('test')} train/val/test", "random" if sp.get("source") == "declared" else "declared / grouped"))
    add_reason(sp.get("reason"), "split")
    pool = plan.get("training_pool", {})
    rows.append(("training pool", pool.get("filter", "") + (f" ({pool['label']})" if pool.get("label") else ""), "-",
                 "label_present" if pool.get("filter") == "all" else "all"))
    add_reason(pool.get("reason"), "training pool")
    for lab, ex in (plan.get("excluded_classes") or {}).items():
        ev = ", ".join(f"{i.get('value')}" for i in ex.get("evidence", []))
        rows.append((f"exclude {lab} classes", ", ".join(ex.get("classes", [])), f"counts {ev}", "keep them"))
        add_reason(ex.get("reason"), f"exclusion ({lab})")
    seen = set()
    for m in plan.get("models", []):
        for mod, s in m.get("inputs", {}).items():
            key = (mod, s["encoder"])
            if key in seen or not plan_schema.METHODS.get(m["method"], {}).get("trains"):
                continue
            seen.add(key)
            cands = [c for c in b.get("encoders", {}).get(mod, {}) if c != s["encoder"]]
            ev = _ev_short(s.get("evidence"))
            rows.append((f"{mod} encoder", labels.enc_name(s["encoder"]), ev, ", ".join(labels.enc_name(c) for c in cands) or "-"))
            add_reason(s.get("reason"), f"{mod} encoder {labels.enc_name(s['encoder'])}")
    losses = {}
    for m in plan.get("models", []):
        if m.get("loss"):
            losses.setdefault(m["loss"], []).append(m)
    for loss, ms in losses.items():
        others = sorted(plan_schema.LOSSES.get(ms[0]["method"], set()) - set(losses))
        ev = _ev_short(ms[0].get("loss_evidence"))
        rows.append((f"loss ({len(ms)} model{'s' if len(ms) > 1 else ''})", labels.LOSS_NAMES.get(loss, loss), ev, ", ".join(labels.LOSS_NAMES.get(o, o) for o in others) or "-"))
        for x in ms:
            add_reason(x.get("loss_reason"), f"loss {labels.LOSS_NAMES.get(loss, loss)}")
    rec = a["recommendation"] or {}
    if rec:
        rows.append(("exported representation", rec.get("chosen_model", ""), "see Results", "the other models"))
        add_reason(rec.get("chosen_reason"), "exported representation")
    return rows, reasons


def decisions_pages(a):
    rows, reasons = decisions(a)
    lines = ["```" + "\n".join(f"{r[0][:24]:24s} {r[1][:16]:16s} {r[2][:44]:44s} {r[3][:26]}" for r in
                               [("decision", "choice", "evidence", "alternative rejected")] + rows) + "```"]
    rs = [f"- {', '.join(dict.fromkeys(who))}: {text}" for text, who in reasons.items()]
    names = ["; ".join(f"{mid} = {lab}" for mid, lab in a["labels"].items()) + "."]
    return text_pages("Decisions", [(None, lines), ("Reasons (each stated once)", rs), ("Model ids used in the files", names)])


def results_pages(a):
    obj = a["objective"]
    pk, t = _primary(a)
    order = [pk] + [p for p in (
        [f"cross_modal_retrieval.{x['query']}->{x['gallery']}" for x in obj.get("tasks", []) if x["task"] == "cross_modal_retrieval"] +
        [f"label_transfer.{x['from']}->{x['to']}" for x in obj.get("tasks", []) if x["task"] == "label_transfer"] +
        [f"label_prediction.{x['label']}" for x in obj.get("tasks", []) if x["task"] == "label_prediction"] +
        [f"cross_modal_prediction.{x['target']}" for x in obj.get("tasks", []) if x["task"] == "cross_modal_prediction"] +
        [f"clustering.{x['label']}" for x in obj.get("tasks", []) if x["task"] == "label_prediction"]) if p != pk]
    sec = [f"secondary_label_prediction.{x['label']}" for x in obj.get("secondary_tasks", [])]
    blocks = []
    for i, prefix in enumerate(order + sec):
        paras = task_table(a, prefix, (obj.get("success_criterion") or {}).get("probe"), one_probe=prefix in sec)
        if prefix in sec and paras:
            paras.append("Secondary label: only the probe named in the success criterion is shown; both are in evaluation.csv.")
        if paras:
            title = ("Primary: " if i == 0 else ("Secondary: " if prefix in sec else "")) + prefix.replace("_", " ").replace(".", ", ")
            blocks.append((title, paras))
    col = []
    for mid, v in (a["collapse"] or {}).items():
        for mod, x in v["inputs"].items():
            ranks = [r[mod]["effective_rank"] for r in v["learned"].values() if mod in r]
            if ranks:
                col.append([f"{a['labels'].get(mid, mid)} / {mod}", labels.sig3(x["effective_rank"]), f"{labels.sig3(min(ranks))}-{labels.sig3(max(ranks))}"])
    if col:
        blocks.append(("Collapse check", [_table(["embedding", "input rank", "learned rank"], col, w=14)]))
    blocks.append((None, ["Cells: mean ± SD over seeds (no SD: one run); 3 significant figures. Full precision and label-draw SD: evaluation.csv and metrics/*.json."]))
    return text_pages("Results (held-out test samples)", blocks)


def _takeaway(a, stem):
    summ = a["summary"] or {}
    L = a["labels"]
    best = lambda key, low=False: (sorted(((r["metrics"][key]["mean"], labels.row_label(n, r, L)) for n, r in summ.items() if key in r["metrics"]),
                                          reverse=not low) or [(None, None)])[0]
    if stem.startswith("matching_"):
        pair = stem[len("matching_"):].replace("-", "->", 1)
        v, who = best(f"cross_modal_retrieval.{pair}.foscttm", low=True)
        return f"Lowest FOSCTTM: {who} ({labels.sig3(v)}); chance is 0.5." if who else ""
    if stem.startswith("transfer_"):
        pair = stem[len("transfer_"):].replace("-", "->", 1)
        v, who = best(f"label_transfer.{pair}.budgets.all.knn_macro_f1")
        return f"Highest kNN transfer macro-F1: {who} ({labels.sig3(v)})." if who else ""
    if stem.startswith(("label_budget_", "secondary_budget_")):
        lab = stem.split("_budget_", 1)[1]
        pre = ("label_prediction." if stem.startswith("label") else "secondary_label_prediction.") + lab
        ks = sorted({k.split(".")[3] for r in summ.values() for k in r["metrics"] if k.startswith(pre + ".budgets.")},
                    key=lambda b: 10**9 if b == "all" else int(b))
        if not ks:
            return ""
        v, who = best(f"{pre}.budgets.{ks[0]}.linear_macro_f1")
        return f"With the fewest labels ({ks[0]}{'/class' if ks[0] != 'all' else ''}), the best linear probe is {who} ({labels.sig3(v)})."
    if stem == "training_curves":
        stops = sorted({r.get("stop_reason", "").split(" (")[0] for r in a["runs"].values() if "stop_reason" in r})
        return "Stopping reasons seen: " + ", ".join(stops) + "."
    if stem == "collapse":
        return "A learned rank far below the input's means the alignment compressed the features."
    if stem == "embedding_2d":
        return "Classes that form separate clusters are easy to label with few examples."
    return ""


FIG_TEXT = {"label_budget": "Macro-F1 vs labeled examples per class (linear probe left, kNN right); bars: SD over seeds.",
            "secondary_budget": "Secondary label, same layout.",
            "matching": "Held-out matching: FOSCTTM (left) and recall@k with chance dashed (right).",
            "transfer": "Labels transferred across modalities (solid: kNN, light: linear).",
            "training_curves": "Loss per epoch, seed 0 of each trained model, rescaled per model.",
            "collapse": "Effective rank, input features vs learned embedding.",
            "embedding_2d": "t-SNE of test embeddings colored by label: no-pretraining baseline vs best trained model."}


def figure_pages(a):
    pngs = sorted((a["out"] / "figures").glob("*.png"), key=lambda p: (not p.stem.startswith(("matching", "label_budget")), p.stem))
    figs = []
    for i in range(0, len(pngs), 2):
        fig = plt.figure(figsize=PAGE_SIZE)
        for j, png in enumerate(pngs[i:i + 2]):
            img = plt.imread(png)
            top = 0.97 - j * 0.48
            ax = fig.add_axes([0.06, top - 0.38, 0.88, 0.37]); ax.axis("off")
            ax.imshow(img, interpolation="none")
            txt = next((v for k, v in FIG_TEXT.items() if png.stem.startswith(k)), png.stem) + " " + _takeaway(a, png.stem)
            fig.text(0.07, top - 0.395, "\n".join(_wrap(txt, "c", max_w=0.86 * 612)), fontsize=8, color=MUTED, va="top")
        figs.append(fig)
    return figs


def findings_pages(a):
    secs = sections_of(a["findings"])
    blocks = [(h, _paras(t)) for h, t in secs.items() if h not in ("Answer", "Limitations", "Critique and revision")]
    return text_pages("Findings", blocks) if blocks else []


def recommendation_pages(a, n=None):
    rec = a["recommendation"]
    if not rec:
        return []
    lines = []
    for u in rec.get("uses", []):
        r = _first_sentence(u.get("reason", ""))
        r = re.split(r"[,;(]?\s*[^,;(]*\d+\.\d+", r)[0].strip(" ,;(") if re.search(r"\d+\.\d+", r) else r
        lines.append(f"- {u['verdict'].replace('_', ' ').capitalize()}: {u['use']}." + (f" {r}" if r else ""))
    c = a["contract"] or {}
    files = [f"Embeddings and index: deliverables/generated_report_{n}/representations/ (also {c.get('embeddings')}, {c.get('embedding_index')})."] if c else []
    return text_pages("Downstream use", [(None, lines + files)])


def _strip_decimals(text):
    t = _first_sentence(text)
    return re.split(r"\s*[(\[]?[^.;:()]*\d+\.\d+", t)[0].rstrip(" ,;:(") + ("" if not re.search(r"\d+\.\d+", t) else " ...")


def critique_pages(a):
    c, resp = a["critique"], a["critique_response"]
    if not c:
        return text_pages("Critique", [(None, ["No critique.json was written; the critic did not run or wrote nothing."])])
    rows = []
    if resp:
        for r in resp:
            rows.append(f"- [{r.get('severity')}] {r.get('finding_title')}: changed: {r.get('change')}. Effect on the final result: {r.get('effect_on_final')}")
    else:
        for f in c.get("findings", []):
            title = f.get("title") or _strip_decimals(f.get("finding", ""))
            rows.append(f"- [{f.get('severity')}] {title} What changed / effect on the final result: not recorded in this run "
                        f"(critique_response.json missing); see Findings and Limitations.")
    return text_pages("Critique", [(None, rows + [f"Full critique: outputs/{a['dataset']}/critique.json (numbers there are from the revision it reviewed)."])])


def limitation_items(a, untraced):
    items = []
    for enc, c in (a["profile"].get("cache") or {}).items():
        if any(s.get("encoder") == f"cache:{enc}" for m in a["plan"].get("models", []) for s in m.get("inputs", {}).values()):
            items.append(f"Pretraining data of {labels.enc_name('cache:' + enc)}: {c.get('pretraining_data')}")
    cav = re.search(r"^##\s+Known caveats\s*$(.*?)(?=^##\s|\Z)", a["desc"] or "", flags=re.M | re.S)
    if cav:
        items += [" ".join(x.split()) for x in re.split(r"\n- ", "\n" + cav.group(1).strip()) if x.strip()]
    for r in (a["run_log"] or {}).get("runs", []):
        if r.get("status") in ("not_run_budget", "not_run_unavailable", "failed"):
            items.append(f"{r.get('model_id') or r.get('key')}: {r['status']} ({r.get('reason') or r.get('error', '')})")
    for k, r in a["runs"].items():
        if r.get("early_stopping") and str(r.get("stop_reason", "")).startswith(("max_epochs", "time_cap")):
            items.append(f"{k} stopped at {r['stop_reason']}; validation loss may still have been improving.")
    own = sections_of(a["findings"]).get("Limitations", "")
    items += [x.lstrip("- ").strip() for x in _paras(own)]
    if untraced:
        items.append("findings contain numbers not found in any artifact: " + ", ".join(untraced))
    seen, out = set(), []
    for x in items:
        key = re.sub(r"\W+", " ", x.lower())[:80]
        if x and key not in seen:
            seen.add(key)
            out.append("- " + x.lstrip("- "))
    return out


def repro_pages(a, n):
    rl = a["run_log"] or {}
    lines = [f"Command: ./run.sh {a['dataset']} \"<objective above>\" --report-number {n}. Plan revision {a['plan'].get('revision')}; "
             f"model runs took {rl.get('minutes')} min. Per-seed training logs, config hashes, hyperparameters with reasons, "
             f"evidence values and versions: deliverables/generated_report_{n}/methods_summary.md, traceability.md, and "
             f"outputs/{a['dataset']}/ (runs/, run_log.json, plan.json)."]
    return lines


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


def check_page_cap(n_pages, cap=MAX_PAGES):
    if n_pages > cap:
        raise SystemExit(f"ERROR: report would have {n_pages} pages, above the cap of {cap}. Shorten findings.md "
                         "(each fact once; tables and figures carry the numbers) and rebuild.")


def build(dataset, n, allow_untraced=False):
    a = load_all(dataset)
    if not a["plan"] or not a["objective"]:
        sys.exit("ERROR: plan.json and objective.json are required")
    text = (a["findings"] or "") + "\n" + json.dumps(a["critique_response"] or [])
    untraced = untraced_numbers(text, a)
    if untraced and not allow_untraced:
        print("UNTRACED numbers in findings.md / critique_response.json (not found in any artifact at the precision written):")
        for u in untraced:
            print(f"  {u}")
        sys.exit("ERROR: fix them so every number comes from the JSON artifacts, then rerun.")
    sections = [("Summary", summary_page(a, n)), ("Setup", setup_pages(a)), ("Decisions", decisions_pages(a)),
                ("Results", results_pages(a)), ("Figures", figure_pages(a)), ("Findings", findings_pages(a)),
                ("Downstream use", recommendation_pages(a, n)), ("Critique", critique_pages(a)),
                ("Limitations and reproducibility", limitations_and_repro(a, untraced, n))]
    pages = contents_page(sections, n, dataset)
    for _, figs in sections:
        pages += figs
    check_page_cap(len(pages))
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
    return len(pages)


def limitations_and_repro(a, untraced, n):
    return text_pages("Limitations and reproducibility", [(None, limitation_items(a, untraced)), ("Reproducibility", repro_pages(a, n))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the generated report PDF for one dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report-number", type=int, required=True)
    parser.add_argument("--allow-untraced", action="store_true", help="build even if findings have untraceable numbers (listed under Limitations)")
    args = parser.parse_args()
    build(args.dataset, args.report_number, args.allow_untraced)
