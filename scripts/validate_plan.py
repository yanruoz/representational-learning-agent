"""Validate outputs/<d>/plan.json against objective.json, profile.json, the baseline check, and the
   actual split. Writes outputs/<d>/plan_validation.json; run_plan.py refuses an unvalidated plan.

   Usage: venv/bin/python scripts/validate_plan.py outputs/<d>/plan.json

   Plan-level: dataset, revision, budget_min, split {source, val_fraction, seed, stratify_label |
   group_by, reason}, training_pool {filter: all | label_present, label?, reason}, excluded_classes
   {label: {classes, reason, evidence}}, budget_not_filled {label: {classes, reason}}, models [...].

   Per model: id ([a-z0-9_]), method, inputs {modality: {encoder, head?, params?, reason, evidence}},
   loss + loss_reason + loss_evidence (trained methods), hyperparameters + hyperparameter_reasons (one
   per key), seeds, serves_tasks, estimated_minutes, status (planned | not_run_budget |
   not_run_unavailable), reason. Optional: label (for label-using losses), concat_input.

   Checks:
   - methods, losses, heads, encoders exist and fit the modality kind; cached encoders are ready
   - every choice has a specific reason; every evidence value matches its file
   - every objective task is served by a trained model whose method has that capability, embeds
     the needed modalities alone, and (if it uses labels) has a label task
   - required baselines: for each task, a no-pretraining baseline on the inference modality
     (label tasks), a linear_alignment model (retrieval, label transfer), a linear_supervised model
     (cross-modal prediction)
   - integer label budgets: every non-excluded class can fill them from the training split, or is
     listed in budget_not_filled with a reason
   - planned minutes (estimated_minutes x seeds x budgets for label-using losses) fit budget_min
"""
import argparse
import json
import re
import sys
from pathlib import Path

import context
import evidence
import features
import plan_schema


def _check_model(m, plan, objective, ctx, errors, warnings):
    w = f"model '{m.get('id')}'"
    if not re.fullmatch(r"[a-z0-9_]+", str(m.get("id", ""))):
        errors.append(f"{w}: id must be [a-z0-9_]+")
    method = m.get("method")
    if method not in plan_schema.METHODS:
        errors.append(f"{w}: unknown method (allowed: {sorted(plan_schema.METHODS)})")
        return
    if m.get("status", "planned") not in ("planned", "not_run_budget", "not_run_unavailable"):
        errors.append(f"{w}: status must be planned, not_run_budget, or not_run_unavailable")
    evidence.check_reason(m.get("reason"), f"{w}.reason", errors)
    trains = plan_schema.METHODS[method]["trains"]
    inputs = m.get("inputs") or {}
    if not inputs:
        errors.append(f"{w}: needs inputs")
    for mod, spec in inputs.items():
        wi = f"{w}.inputs.{mod}"
        if mod not in ctx.data["modalities"]:
            errors.append(f"{wi}: not a declared modality")
            continue
        avail = features.available_encoders(ctx, mod)
        if spec.get("encoder") not in avail:
            errors.append(f"{wi}: encoder {spec.get('encoder')!r} not available for this modality (available: {avail})")
        if trains and spec.get("head") not in ("linear", "mlp"):
            errors.append(f"{wi}: trained methods need head 'linear' or 'mlp'")
        evidence.check_reason(spec.get("reason"), f"{wi}.reason", errors)
        if trains or method == "frozen_probe":
            evidence.check_items(spec.get("evidence"), plan["dataset"], f"{wi}", errors)
    if method in ("two_tower", "linear_alignment") and len(inputs) != 2:
        errors.append(f"{w}: {method} needs exactly two input modalities")
    if trains:
        loss = m.get("loss")
        if loss not in plan_schema.LOSSES.get(method, set()):
            errors.append(f"{w}: loss {loss!r} not allowed for {method} (allowed: {sorted(plan_schema.LOSSES.get(method, []))})")
        evidence.check_reason(m.get("loss_reason"), f"{w}.loss_reason", errors)
        evidence.check_items(m.get("loss_evidence"), plan["dataset"], f"{w}.loss_evidence", errors)
        hp, hr = m.get("hyperparameters") or {}, m.get("hyperparameter_reasons") or {}
        for k in hp:
            if k not in hr:
                errors.append(f"{w}: hyperparameter '{k}' has no entry in hyperparameter_reasons")
            else:
                evidence.check_reason(hr[k], f"{w}.hyperparameter_reasons.{k}", errors, min_len=15)
        if not m.get("seeds"):
            errors.append(f"{w}: needs seeds (a list; several seeds make spread visible)")
        if loss in plan_schema.LOSSES_NEEDING_LABELS:
            labels_in_tasks = [t.get("label") for t in objective.get("tasks", []) if t.get("label")]
            if not (m.get("label") or labels_in_tasks):
                errors.append(f"{w}: {loss} needs labels, but the objective has no label task")
    for t in m.get("serves_tasks", []):
        if t not in plan_schema.METHOD_TASKS[method]:
            errors.append(f"{w}: {method} cannot serve task {t}")
    if not isinstance(m.get("estimated_minutes", 0), (int, float)):
        errors.append(f"{w}: estimated_minutes must be a number")


def _modalities_of(t):
    return [t[f] for f in ("modality", "from", "to", "query", "gallery", "source") if t.get(f)]


def validate(plan, objective):
    errors, warnings = [], []
    sp = plan.get("split") or {}
    if sp.get("source") not in ("declared", "random"):
        errors.append("split.source must be 'declared' or 'random'")
    evidence.check_reason(sp.get("reason"), "split.reason", errors)
    pool = plan.get("training_pool") or {}
    if pool.get("filter") not in ("all", "label_present"):
        errors.append("training_pool.filter must be 'all' or 'label_present'")
    evidence.check_reason(pool.get("reason"), "training_pool.reason", errors)
    if errors:
        return errors, warnings
    ctx = context.Context(plan["dataset"], plan, objective)
    for lab, ex in (plan.get("excluded_classes") or {}).items():
        evidence.check_reason(ex.get("reason"), f"excluded_classes.{lab}.reason", errors)
        evidence.check_items(ex.get("evidence"), plan["dataset"], f"excluded_classes.{lab}", errors)

    models = plan.get("models", [])
    for m in models:
        _check_model(m, plan, objective, ctx, errors, warnings)
    planned = [m for m in models if m.get("status", "planned") == "planned" and m.get("method") in plan_schema.METHODS]

    for i, t in enumerate(objective.get("tasks", [])):
        name, w = t["task"], f"objective task {i} ({t['task']})"
        mods = _modalities_of(t) or objective["inference_modalities"]
        trained = [m for m in planned if plan_schema.METHODS[m["method"]]["trains"] and name in plan_schema.METHOD_TASKS[m["method"]]
                   and all(x in m.get("inputs", {}) for x in mods)]
        if not trained:
            errors.append(f"{w}: no planned trained model can serve it with inputs {mods}")
        if name in ("label_prediction", "clustering"):
            base = [m for m in planned if m["method"] in plan_schema.BASELINE_METHODS and any(
                x in m["inputs"] and m["inputs"][x]["encoder"] in plan_schema.NO_PRETRAINING for x in mods)]
            if not base:
                errors.append(f"{w}: needs a baseline with no pretraining on {mods} (e.g. single_modality_features)")
        if name in ("cross_modal_retrieval", "label_transfer") and not any(m["method"] == "linear_alignment" for m in planned):
            errors.append(f"{w}: needs a linear_alignment (CCA) baseline")
        if name == "cross_modal_prediction" and not any(m["method"] == "linear_supervised" for m in planned):
            errors.append(f"{w}: needs a linear_supervised (PLS) baseline")
        for b in t.get("label_budgets_per_class", []):
            if b == plan_schema.BUDGET_ALL:
                continue
            _, taken, unfilled = ctx.draw_labeled(t["label"], b, 0, pool=ctx.train)
            allowed = set((plan.get("budget_not_filled") or {}).get(t["label"], {}).get("classes", []))
            bad = [c for c in unfilled if c not in allowed]
            if bad:
                errors.append(f"{w}: budget {b} per class can't be filled for {bad} from the training split "
                              f"({ {c: taken[c] for c in bad} }); exclude them (excluded_classes) or list them in budget_not_filled with a reason")
    for lab, bn in (plan.get("budget_not_filled") or {}).items():
        evidence.check_reason(bn.get("reason"), f"budget_not_filled.{lab}.reason", errors)

    total = 0.0
    for m in planned:
        if not plan_schema.METHODS[m["method"]]["trains"]:
            continue
        n_b = 1
        if m.get("loss") in plan_schema.LOSSES_NEEDING_LABELS:
            n_b = max([len(t.get("label_budgets_per_class", [1])) for t in objective.get("tasks", []) if t.get("label")] or [1])
        total += float(m.get("estimated_minutes", 0)) * len(m.get("seeds", [0])) * n_b
    if total > float(plan.get("budget_min", 30)):
        errors.append(f"planned minutes {total:.1f} exceed budget_min {plan.get('budget_min', 30)}; mark models not_run_budget")
    return errors, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    args = ap.parse_args()
    path = Path(args.plan)
    plan = json.loads(path.read_text())
    objective = json.loads((path.parent / "objective.json").read_text())
    errors, warnings = validate(plan, objective)
    (path.parent / "plan_validation.json").write_text(json.dumps({"passed": not errors, "errors": errors, "warnings": warnings}, indent=1) + "\n")
    print("PASS" if not errors else "FAIL", f"plan.json ({len(errors)} errors, {len(warnings)} warnings)")
    for e in errors:
        print("  ERROR:", e)
    for w in warnings:
        print("  WARNING:", w)
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
