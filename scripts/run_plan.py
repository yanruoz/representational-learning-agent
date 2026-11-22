"""Execute a validated plan.json: for every planned model, build or train its representation and
   evaluate it, then aggregate across seeds and write the run log. Pattern from Project 2.

   Usage: venv/bin/python scripts/run_plan.py outputs/<d>/plan.json [--only <model_id> ...]

   - Refuses a plan that hasn't passed validate_plan.py (outputs/<d>/plan_validation.json must say
     passed and be newer than plan.json). --allow-unvalidated exists for development only.
   - Baselines (no trainable weights) run once (key <id>_s0), except linear_alignment, which runs per
     seed on a seeded 90% subsample of the training rows so it has a spread like the learned models.
   - Earlier revisions' run_log / summary / collapse are archived in outputs/<d>/history/revision_<n>/. Trained models run per seed, and
     label-using losses per seed x label budget (key <id>_s<seed>_b<budget>).
   - Reuse: a trained run whose runs/<key>.json config_hash matches and whose embeddings exist is not
     retrained. Evaluation always reruns, so metrics never go stale when evaluation code changes.
   - A failing model is retried once, then recorded as failed; one failure never stops the others.
     Models with status "not_run_budget" or "not_run_unavailable" are recorded, not run.
   - Writes metrics/<key>.json (per run), metrics/summary.json (mean / SD / n over seeds for every
     numeric metric; metrics averaged over label draws also get a <metric>_draw_sd entry), metrics/collapse.json (learned embeddings next to their input features),
     and run_log.json.
"""
import argparse
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

import baselines
import context
import evaluate_rep
import features
import plan_schema
import train


def _flatten(d, prefix=""):
    """Numeric leaves of a metrics dict as {path: value}; {mean, sd} pairs contribute their mean."""
    out = {}
    if isinstance(d, dict):
        if set(d) >= {"mean", "sd"} and isinstance(d["mean"], (int, float)):
            # the SD inside one run is over label draws; keep it next to the mean
            return {prefix: float(d["mean"]), f"{prefix}_draw_sd": float(d["sd"])}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = float(d)
    return out


def _budgets(ctx, entry):
    lname = train.label_name(ctx.objective, entry)
    for t in ctx.objective.get("tasks", []):
        if t.get("label") == lname and "label_budgets_per_class" in t:
            return t["label_budgets_per_class"]
    return [plan_schema.BUDGET_ALL]


def run_entry(ctx, plan, entry, log):
    method = entry["method"]
    if entry.get("status", "planned") != "planned":
        log.append({"model_id": entry["id"], "status": entry["status"], "reason": entry.get("reason")})
        return
    jobs = []
    if method == "linear_alignment":
        jobs = [(s, None) for s in entry.get("seeds", [0])]
    elif method in plan_schema.BASELINE_METHODS:
        jobs = [(0, None)]
    else:
        uses_labels = entry["loss"] in plan_schema.LOSSES_NEEDING_LABELS
        budgets = _budgets(ctx, entry) if uses_labels else [None]
        jobs = [(s, b) for s in entry.get("seeds", [0]) for b in budgets]
    for seed, budget in jobs:
        key = f"{entry['id']}_s{seed}" + (f"_b{budget}" if budget is not None else "")
        t0 = time.time()
        for attempt in (1, 2):
            try:
                if method in plan_schema.BASELINE_METHODS:
                    rep, info = baselines.build(ctx, entry, seed)
                    rec = {"key": key, "model_id": entry["id"], "method": method, "seed": seed, "built": info}
                    (ctx.out / "runs").mkdir(parents=True, exist_ok=True)
                    (ctx.out / "runs" / f"{key}.json").write_text(json.dumps(rec, indent=1) + "\n")
                    m_out = evaluate_rep.evaluate(ctx, rep, method, key, shared_space=info["shared_space"],
                                                  rep_features=rep if method == "linear_supervised" else None)
                    status = "ok" if m_out["tasks"] else "ok_no_applicable_task"
                else:
                    chash = train.config_hash(plan, entry, seed, budget)
                    run_path, emb_path = ctx.out / "runs" / f"{key}.json", ctx.out / "embeddings" / f"{key}.npz"
                    reused = False
                    if run_path.exists() and emb_path.exists() and json.loads(run_path.read_text()).get("config_hash") == chash:
                        emb, reused = dict(np.load(emb_path)), True
                    else:
                        emb, rec = train.train(ctx, entry, seed, budget)
                        train.save(ctx, key, emb, rec, chash)
                    m_out = evaluate_rep.evaluate(ctx, emb, method, key, only_budget=budget)
                    status = ("reused" if reused else "ok") if m_out["tasks"] else "ok_no_applicable_task"
                log.append({"key": key, "model_id": entry["id"], "seed": seed, "budget": budget, "status": status,
                            "attempts": attempt, "seconds": round(time.time() - t0, 1)})
                print(f"   [{status}] {key} ({time.time() - t0:.1f}s)", flush=True)
                break
            except Exception as e:  # one bounded retry, then record and move on
                if attempt == 2:
                    log.append({"key": key, "model_id": entry["id"], "seed": seed, "budget": budget, "status": "failed",
                                "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
                    print(f"   [failed] {key}: {type(e).__name__}: {e}", flush=True)


def aggregate(ctx, plan, log):
    groups = defaultdict(list)
    for r in log:
        if r.get("status") in ("ok", "reused"):
            groups[(r["model_id"], r.get("budget"))].append(json.loads((ctx.out / "metrics" / f"{r['key']}.json").read_text()))
    summary = {}
    for (mid, budget), runs in groups.items():
        flat = [_flatten(m["tasks"]) for m in runs]
        keys = sorted(set().union(*flat))
        name = mid + (f"_b{budget}" if budget is not None else "")
        summary[name] = {"model_id": mid, "budget": budget, "n_seeds": len(runs), "method": runs[0]["method"],
                         "metrics": {k: {"mean": round(float(np.mean([f[k] for f in flat if k in f])), 5),
                                         "sd": round(float(np.std([f[k] for f in flat if k in f])), 5),
                                         "n": sum(k in f for f in flat)} for k in keys}}
    (ctx.out / "metrics" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    collapse = {}
    for entry in plan["models"]:
        if entry["method"] in plan_schema.BASELINE_METHODS:
            continue
        inputs = {}
        for m, s in entry["inputs"].items():
            Z = features.featurize(ctx, m, s["encoder"], s.get("params"))
            inputs[m] = evaluate_rep.collapse(Z[ctx.test])
        learned = {r["key"]: json.loads((ctx.out / "metrics" / f"{r['key']}.json").read_text())["collapse"]
                   for r in log if r.get("model_id") == entry["id"] and r.get("status") in ("ok", "reused")}
        collapse[entry["id"]] = {"inputs": inputs, "learned": learned}
    (ctx.out / "metrics" / "collapse.json").write_text(json.dumps(collapse, indent=1) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run every planned model and evaluate it.")
    ap.add_argument("plan")
    ap.add_argument("--only", nargs="*", help="run only these model ids")
    ap.add_argument("--allow-unvalidated", action="store_true", help="development only")
    args = ap.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    out = context.OUTPUTS / plan["dataset"]
    val = out / "plan_validation.json"
    if not args.allow_unvalidated:
        ok = val.exists() and json.loads(val.read_text()).get("passed") and val.stat().st_mtime >= plan_path.stat().st_mtime
        if not ok:
            raise SystemExit("ERROR: plan.json has not passed validate_plan.py (run it first)")
    objective = json.loads((out / "objective.json").read_text())
    old_log = out / "run_log.json"
    if old_log.exists():  # keep earlier revisions' results instead of overwriting them
        prev = json.loads(old_log.read_text())
        if prev.get("plan_revision") != plan.get("revision"):
            hist = out / "history" / f"revision_{prev.get('plan_revision')}"
            hist.mkdir(parents=True, exist_ok=True)
            for f in (old_log, out / "metrics" / "summary.json", out / "metrics" / "collapse.json"):
                if f.exists():
                    (hist / f.name).write_text(f.read_text())
            print(f"[run] archived revision {prev.get('plan_revision')} results in {hist}")
    ctx = context.Context(plan["dataset"], plan, objective)
    log, t0 = [], time.time()
    for entry in plan["models"]:
        if args.only and entry["id"] not in args.only:
            continue
        print(f"[run] {entry['id']} ({entry['method']})", flush=True)
        run_entry(ctx, plan, entry, log)
    aggregate(ctx, plan, log)
    (out / "run_log.json").write_text(json.dumps({"plan_revision": plan.get("revision"), "minutes": round((time.time() - t0) / 60, 2),
                                                  "runs": log}, indent=1) + "\n")
    failed = [r["key"] for r in log if r.get("status") == "failed"]
    empty = [r["key"] for r in log if r.get("status") == "ok_no_applicable_task"]
    if empty:
        print(f"[run] WARNING: no objective task applies to {empty} (their inputs don't cover any task's modalities)")
    print(f"[run] done in {(time.time() - t0) / 60:.1f} min; {len(log)} runs, {len(failed)} failed {failed}")


if __name__ == "__main__":
    main()
