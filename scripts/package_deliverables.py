"""Package one dataset's run into deliverables/generated_report_<N>/ (the five required outputs).

   Usage: venv/bin/python scripts/package_deliverables.py --dataset <d> --report-number <N>
   Called by run.sh after the report is built; the agent never writes deliverables/.

   deliverables/generated_report_<N>/
     INDEX.md                  maps the five required outputs to files
     generated_report_<N>.pdf  the final analysis report
     representations/          learned representation of the chosen model: embeddings_<modality>.npy,
                               embedding_index.csv (row, sample_id, split), model_note.md
     evaluation.csv            every metric of every model: mean, SD over seeds, SD over label draws
     downstream_results.csv    each downstream use: verdict, reason, cited values
     methods_summary.md        decisions with full reasons, encoders, losses, hyperparameters with
                               reasons, per-run training log, config hashes, versions
     traceability.md           every evidence value the plan and recommendation cite, and every decimal
                               number in findings.md / critique_response.json with the artifact and key it
                               matches (report.py refuses numbers that match nothing)
   deliverables/ is tracked in git, so the submission archive (git archive) includes it.
"""
import argparse
import csv
import json
import platform
import re
import shutil
from pathlib import Path

import numpy as np

import labels
import report

ROOT = Path(__file__).resolve().parent.parent


def _leaves(obj, path=""):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield path, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")


def trace_numbers(text, a):
    """[(number as written, 'file: key' of the first artifact value it matches, or None)]."""
    sources = [("metrics/summary.json", a["summary"]), ("profile.json", a["profile"]), ("metrics/baseline_check.json", a["baseline_check"]),
               ("metrics/collapse.json", a["collapse"]), ("plan.json", a["plan"]), ("run_log.json", a["run_log"]),
               ("recommendation.json", a["recommendation"] or {}), ("split.json", {"sizes": (a["split"] or {}).get("sizes", {})})]
    sources += [(f"metrics/{k}.json", m) for k, m in a["metrics"].items()] + [(f"runs/{k}.json", r) for k, r in a["runs"].items()]
    leaves = [(f, p, v) for f, obj in sources for p, v in _leaves(obj)]
    out = []
    for m in sorted(set(re.findall(r"(?<![\w.])-?\d+\.\d+%?", text)), key=lambda x: text.index(x)):
        pct = m.endswith("%")
        q = float(m.rstrip("%"))
        dec = len(m.rstrip("%").split(".")[1])
        cands = [(q, dec)] + ([(q / 100, dec + 2)] if pct else [])
        hit = next((f"{f}: {p}" for f, p, v in leaves if any(round(v, d) == round(c, d) for c, d in cands)), None)
        out.append((m, hit))
    return out


def evaluation_rows(a):
    rows = []
    for name, r in (a["summary"] or {}).items():
        lab = labels.row_label(name, r, a["labels"])
        for k, v in r["metrics"].items():
            if k.endswith("_draw_sd"):
                continue
            parts = k.split(".")
            budget = parts[3] if len(parts) > 4 and parts[2] == "budgets" else ""
            dsd = r["metrics"].get(f"{k}_draw_sd", {}).get("mean", "")
            rows.append({"model": lab, "model_id": r["model_id"], "trained_with_labels_per_class": r.get("budget") or "",
                         "task": parts[0], "target": parts[1] if len(parts) > 1 else "", "label_budget": budget,
                         "metric": f"{parts[-2]} [{parts[-1]}]" if len(parts) > 1 and parts[-2].endswith("per_class_f1") else parts[-1],
                         "mean": v["mean"], "sd_over_seeds": v["sd"], "n_seeds": v.get("n", 1), "sd_over_label_draws": dsd})
    return rows


def methods_md(a, n):
    rows, reasons = report.decisions(a)
    L = [f"# Methods summary: {a['dataset']} (report {n})", "", f"Objective: {a['objective'].get('objective_text', '')}", "",
         "## Decisions", "", "| decision | choice | evidence | alternative rejected |", "|---|---|---|---|"]
    L += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows]
    L += ["", "## Models: encoders, losses, reasons, hyperparameters", ""]
    for m in a["plan"].get("models", []):
        L += [f"### {a['labels'].get(m['id'], m['id'])} (`{m['id']}`)", "", f"- Method: {m['method']}. {m.get('reason', '')}"]
        for mod, s in m.get("inputs", {}).items():
            L.append(f"- {mod}: encoder `{s['encoder']}`" + (f", params {s['params']}" if s.get("params") else "")
                     + (f", head {s['head']}" if s.get("head") else "") + f". {s.get('reason', '')}")
        if m.get("loss"):
            L.append(f"- Loss: {m['loss']}. {m.get('loss_reason', '')}")
        for k, v in (m.get("hyperparameters") or {}).items():
            L.append(f"- `{k}` = {v}: {(m.get('hyperparameter_reasons') or {}).get(k, '')}")
        L.append("")
    if a["plan"].get("critique_applied"):
        L += ["## Revision after the critique", "", str(a["plan"]["critique_applied"]), ""]
    L += ["## Training log (per run)", "", "| run | epochs | stop | train pairs | seconds | config hash |", "|---|---|---|---|---|---|"]
    for k, r in a["runs"].items():
        if "history" in r:
            L.append(f"| {k} | {r['epochs_run']} | {r['stop_reason']} | {r['n_train']} | {r['seconds']} | {r.get('config_hash', '')} |")
    try:
        import sklearn
        import torch
        L += ["", f"Versions: python {platform.python_version()}, torch {torch.__version__}, numpy {np.__version__}, scikit-learn {sklearn.__version__}."]
    except Exception:
        pass
    return "\n".join(L) + "\n"


def traceability_md(a):
    L = ["# Traceability", "", "## Evidence cited by decisions (file: key = value)", ""]
    plan, rec = a["plan"], a["recommendation"] or {}
    for m in plan.get("models", []):
        for mod, s in m.get("inputs", {}).items():
            for i in s.get("evidence", []):
                L.append(f"- {m['id']} / {mod} encoder: {i.get('file')}: {i.get('key')} = {i.get('value')}")
        for i in m.get("loss_evidence", []):
            L.append(f"- {m['id']} / loss: {i.get('file')}: {i.get('key')} = {i.get('value')}")
    for lab, ex in (plan.get("excluded_classes") or {}).items():
        for i in ex.get("evidence", []):
            L.append(f"- excluded {lab}: {i.get('file')}: {i.get('key')} = {i.get('value')}")
    for i in rec.get("chosen_evidence", []):
        L.append(f"- chosen representation: {i.get('file')}: {i.get('key')} = {i.get('value')}")
    for u in rec.get("uses", []):
        for i in u.get("evidence", []):
            L.append(f"- use '{u['use'][:60]}': {i.get('file')}: {i.get('key')} = {i.get('value')}")
    text = (a["findings"] or "") + "\n" + json.dumps(a["critique_response"] or [])
    L += ["", "## Numbers in findings.md and critique_response.json", "",
          "Each decimal number and the first artifact value it matches at the precision written (files under outputs/<dataset>/).", ""]
    for num, hit in trace_numbers(text, a):
        L.append(f"- {num}: {hit or 'NOT FOUND (report.py refuses to build with untraced numbers)'}")
    return "\n".join(L) + "\n"


def package(dataset, n):
    a = report.load_all(dataset)
    pdf = ROOT / "reports" / dataset / f"generated_report_{n}.pdf"
    if not pdf.exists():
        raise SystemExit(f"ERROR: {pdf} does not exist; build the report first")
    d = ROOT / "deliverables" / f"generated_report_{n}"
    if d.exists():
        shutil.rmtree(d)
    (d / "representations").mkdir(parents=True)
    shutil.copy(pdf, d / pdf.name)

    out = a["out"]
    c = a["contract"] or {}
    mods = c.get("embedding_modalities") or a["objective"].get("inference_modalities", [])
    rep_files = []
    for m in mods:
        src = out / f"embeddings_{m}.npy"
        if src.exists():
            shutil.copy(src, d / "representations" / src.name)
            rep_files.append(src.name)
    if (out / "embedding_index.csv").exists():
        shutil.copy(out / "embedding_index.csv", d / "representations" / "embedding_index.csv")
    chosen = (a["recommendation"] or {}).get("chosen_model")
    entry = next((m for m in a["plan"].get("models", []) if m["id"] == chosen), {})
    note = [f"# Learned representation: {a['labels'].get(chosen, chosen)} (`{chosen}`)", "",
            f"Method: {entry.get('method')}; loss: {entry.get('loss') or 'none (no training)'}; encoders: "
            + ", ".join(f"{m} {s['encoder']}" for m, s in entry.get("inputs", {}).items()) + ".", "",
            "Files: " + ", ".join(rep_files) + " (float32, one row per sample, rows aligned with embedding_index.csv; "
            "the split column says which samples were held out).", "",
            "Load: `np.load('embeddings_<modality>.npy')`; L2-normalize rows for cosine similarity.", "",
            f"Why this one: {(a['recommendation'] or {}).get('chosen_reason', '')}"]
    (d / "representations" / "model_note.md").write_text("\n".join(note) + "\n")

    rows = evaluation_rows(a)
    with open(d / "evaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)
    with open(d / "downstream_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["use", "verdict", "reason", "evidence"])
        for u in (a["recommendation"] or {}).get("uses", []):
            w.writerow([u["use"], u["verdict"], u.get("reason", ""),
                        "; ".join(f"{i.get('file')}: {i.get('key')} = {i.get('value')}" for i in u.get("evidence", []))])
    (d / "methods_summary.md").write_text(methods_md(a, n))
    (d / "traceability.md").write_text(traceability_md(a))
    index = [f"# Deliverables: {dataset}, generated report {n}", "",
             f"Objective: {a['objective'].get('objective_text', '')}", "",
             "| Required output | File |", "|---|---|",
             "| Learned multimodal representations | `representations/` (" + ", ".join(rep_files) + ", embedding_index.csv, model_note.md) |",
             "| Quantitative evaluation results | `evaluation.csv` (also the Results section of the PDF) |",
             "| Downstream analysis results | `downstream_results.csv` (also the Downstream use section) |",
             "| Summary of the selected methods and preprocessing | `methods_summary.md` (also the Decisions section) |",
             f"| Final analysis report | `generated_report_{n}.pdf` |", "",
             "`traceability.md` links every cited value and every number in the agent's text to the file and key it comes from.",
             f"All intermediate files: `outputs/{dataset}/` (not in the archive; rebuilt by `./run.sh`)."]
    (d / "INDEX.md").write_text("\n".join(index) + "\n")
    print(f"[package] wrote {d}: " + ", ".join(sorted(p.name for p in d.iterdir())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Package the five required outputs for one dataset.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--report-number", type=int, required=True)
    args = ap.parse_args()
    package(args.dataset, args.report_number)
