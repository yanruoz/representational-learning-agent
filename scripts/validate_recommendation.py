"""Validate outputs/<d>/recommendation.json, then export the machine-readable result for callers.

   Usage: venv/bin/python scripts/validate_recommendation.py outputs/<d>/recommendation.json

   Schema:
     chosen_model: a plan model id that ran (its embeddings become embeddings.npy)
     chosen_reason, chosen_evidence: why this representation, with values from metrics/summary.json
     uses: [{use, verdict: recommended | conditional | not_recommended, reason, evidence}]
   Checks: verdicts; chosen model ran; every evidence value matches its file; a recommended or
   conditional use cites at least two values (the model's and a baseline or reference to compare).

   On success, writes the caller contract (Project 4 reads these):
     outputs/<d>/embeddings.npy        chosen model, seed 0 (budget 'all' if label-using): the
                                       inference modalities' embedding, rows = embedding_index.csv
     outputs/<d>/embedding_index.csv   row, sample_id, split
     outputs/<d>/summary.json          chosen model, paths, headline metrics with baselines, verdicts
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import evaluate_rep
import evidence
import loaders

VERDICTS = {"recommended", "conditional", "not_recommended"}


def validate(rec, out):
    errors = []
    dataset = out.name
    summary = json.loads((out / "metrics" / "summary.json").read_text())
    run_log = json.loads((out / "run_log.json").read_text())
    ran = {r["model_id"] for r in run_log["runs"] if r.get("status") in ("ok", "reused")}
    if rec.get("chosen_model") not in ran:
        errors.append(f"chosen_model {rec.get('chosen_model')!r} did not run successfully (ran: {sorted(ran)})")
    evidence.check_reason(rec.get("chosen_reason"), "chosen_reason", errors)
    evidence.check_items(rec.get("chosen_evidence"), dataset, "chosen_evidence", errors)
    uses = rec.get("uses") or []
    if not uses:
        errors.append("uses must list at least one use with a verdict")
    for i, u in enumerate(uses):
        w = f"uses[{i}]"
        if u.get("verdict") not in VERDICTS:
            errors.append(f"{w}: verdict must be one of {sorted(VERDICTS)}")
        evidence.check_reason(u.get("reason"), f"{w}.reason", errors)
        n = evidence.check_items(u.get("evidence"), dataset, w, errors)
        if u.get("verdict") in ("recommended", "conditional") and n < 2:
            errors.append(f"{w}: a {u.get('verdict')} use needs at least two evidence values (the model and what it is compared with)")
    return errors, summary


def export(rec, out, summary):
    plan = json.loads((out / "plan.json").read_text())
    objective = json.loads((out / "objective.json").read_text())
    split = json.loads((out / "split.json").read_text())
    mid = rec["chosen_model"]
    entry = next(m for m in plan["models"] if m["id"] == mid)
    data = loaders.load(plan["dataset"])
    ids = data["ids"]
    side = {s: k for k in ("train", "val", "test") for s in split[k]}
    emb_path = out / "embeddings" / f"{mid}_s0.npz"
    if not emb_path.exists():
        cands = sorted((out / "embeddings").glob(f"{mid}_s0_b*.npz"))
        emb_path = next((c for c in cands if c.stem.endswith("_ball")), cands[-1] if cands else None)
    if emb_path is not None and emb_path.exists():
        rep = dict(np.load(emb_path))
        source = emb_path.name
    else:  # a baseline: rebuild its representation deterministically
        import baselines
        import context
        rep, _ = baselines.build(context.Context(plan["dataset"], plan, objective), entry)
        source = f"rebuilt baseline {mid}"
    mods = objective["inference_modalities"]
    np.save(out / "embeddings.npy", np.concatenate([rep[m] for m in mods], axis=1).astype(np.float32))
    with open(out / "embedding_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "sample_id", "split"])
        for k, s in enumerate(ids):
            w.writerow([k, s, side.get(s, "")])
    per_mod = {}
    for m in rep:
        np.save(out / f"embeddings_{m}.npy", rep[m].astype(np.float32))
        per_mod[m] = f"outputs/{plan['dataset']}/embeddings_{m}.npy"
    model_rows = {k: v for k, v in summary.items() if v["model_id"] == mid}
    contract = {
        "dataset": plan["dataset"], "objective": objective.get("objective_text"), "chosen_model": mid,
        "method": entry["method"], "loss": entry.get("loss"),
        "encoders": {m: s["encoder"] for m, s in entry["inputs"].items()},
        "embeddings": f"outputs/{plan['dataset']}/embeddings.npy", "embedding_modalities": mods,
        "embeddings_per_modality": per_mod, "embedding_source": source,
        "embedding_index": f"outputs/{plan['dataset']}/embedding_index.csv",
        "metrics": {k: v["metrics"] for k, v in model_rows.items()},
        "all_models_summary": f"outputs/{plan['dataset']}/metrics/summary.json",
        "uses": [{"use": u["use"], "verdict": u["verdict"]} for u in rec["uses"]],
        "report": f"reports/{plan['dataset']}/generated_report_<N>.pdf",
    }
    (out / "summary.json").write_text(json.dumps(contract, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recommendation")
    args = ap.parse_args()
    path = Path(args.recommendation)
    rec = json.loads(path.read_text())
    errors, summary = validate(rec, path.parent)
    (path.parent / "recommendation_validation.json").write_text(json.dumps({"passed": not errors, "errors": errors}, indent=1) + "\n")
    print("PASS" if not errors else "FAIL", f"recommendation.json ({len(errors)} errors)")
    for e in errors:
        print("  ERROR:", e)
    if not errors:
        export(rec, path.parent, summary)
        print(f"Wrote {path.parent / 'summary.json'}, embeddings.npy, embedding_index.csv")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
