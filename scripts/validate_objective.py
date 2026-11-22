"""Validate outputs/<d>/objective.json against the profile and the objective text.

   Usage: venv/bin/python scripts/validate_objective.py outputs/<d>/objective.json --objective "<text>"

   Schema:
     dataset, objective_text (verbatim), decision (proceed | proceed_adapted | decline), decision_reason,
     inference_modalities, training_only_modalities (declared modalities; disjoint),
     tasks: [{task, ...task fields, primary?}], secondary_tasks: [...],
     success_criterion: {metric, baseline, comparison, reason}, interpretation_notes (free text)
   Task fields:
     label_prediction: label, modality? (an inference modality), label_budgets_per_class? ([int..., "all"])
     label_transfer: label, from, to (inference modalities), label_budgets_per_class?
     cross_modal_retrieval: query, gallery (inference modalities)
     cross_modal_prediction: source (inference modality), target (counts modality)
     clustering / joint_embedding: label? for scoring
   Checks: verbatim text; declared modalities and labels; exactly one primary task; budgets are
   per-class integers or "all"; secondary tasks are label_prediction on other declared labels;
   no test-time input from a training-only modality; reasons are specific.
   Writes outputs/<d>/objective_validation.json.
"""
import argparse
import json
import sys
from pathlib import Path

import data_spec
import evidence
import plan_schema


def validate(obj, profile, text=None):
    errors, warnings = [], []
    if text is not None and obj.get("objective_text", "").strip() != text.strip():
        errors.append("objective_text must be the objective verbatim, as given in the run prompt")
    if obj.get("decision") not in plan_schema.DECISIONS:
        errors.append(f"decision must be one of {sorted(plan_schema.DECISIONS)}")
    evidence.check_reason(obj.get("decision_reason"), "decision_reason", errors)
    if obj.get("decision") == "decline":
        return errors, warnings

    declared = set(profile.get("modalities", {}))
    labels = profile.get("labels", {})
    inf, tro = obj.get("inference_modalities", []), obj.get("training_only_modalities", [])
    if not inf:
        errors.append("inference_modalities must list the modalities available for new samples")
    for m in inf + tro:
        if m not in declared:
            errors.append(f"modality '{m}' is not declared (declared: {sorted(declared)})")
    if set(inf) & set(tro):
        errors.append("a modality cannot be both an inference and a training-only modality")

    tasks = obj.get("tasks", [])
    if sum(bool(t.get("primary")) for t in tasks) != 1:
        errors.append("exactly one task must have primary: true")
    for i, t in enumerate(tasks):
        w = f"tasks[{i}] ({t.get('task')})"
        name = t.get("task")
        if name not in plan_schema.TASKS:
            errors.append(f"{w}: unknown task (allowed: {sorted(plan_schema.TASKS)})")
            continue
        if "label" in t and t["label"] not in labels:
            errors.append(f"{w}: label '{t['label']}' is not declared (declared: {sorted(labels)})")
        if name in ("label_prediction", "label_transfer") and "label" not in t:
            errors.append(f"{w}: needs a label")
        for f in {"label_prediction": ["modality"], "label_transfer": ["from", "to"], "cross_modal_retrieval": ["query", "gallery"],
                  "cross_modal_prediction": ["source"]}.get(name, []):
            if f not in t:
                if f != "modality":
                    errors.append(f"{w}: needs '{f}'")
            elif t[f] not in inf:
                errors.append(f"{w}: '{f}' = {t[f]!r} must be an inference modality; training-only modalities are never test-time inputs")
        if name == "cross_modal_prediction" and t.get("target") not in declared:
            errors.append(f"{w}: target must be a declared modality")
        for b in t.get("label_budgets_per_class", []):
            if not (b == plan_schema.BUDGET_ALL or (isinstance(b, int) and b > 0)):
                errors.append(f"{w}: label budgets are per-class positive integers or 'all', got {b!r}")
    for i, t in enumerate(obj.get("secondary_tasks", [])):
        if t.get("task") != "label_prediction" or t.get("label") not in labels:
            errors.append(f"secondary_tasks[{i}]: must be label_prediction on a declared label")
        if t.get("modality") and t["modality"] not in inf:
            errors.append(f"secondary_tasks[{i}]: modality must be an inference modality")

    sc = obj.get("success_criterion") or {}
    for f in ("metric", "baseline", "comparison"):
        if not sc.get(f):
            errors.append(f"success_criterion.{f} is required (which metric, against which baseline, what margin)")
    evidence.check_reason(sc.get("reason"), "success_criterion.reason", errors)
    if obj.get("dataset") != profile.get("dataset"):
        errors.append("dataset does not match profile.json")
    return errors, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("objective")
    ap.add_argument("--objective", dest="text", help="the objective text, verbatim")
    args = ap.parse_args()
    path = Path(args.objective)
    obj = json.loads(path.read_text())
    profile = json.loads((path.parent / "profile.json").read_text())
    errors, warnings = validate(obj, profile, args.text)
    (path.parent / "objective_validation.json").write_text(json.dumps({"passed": not errors, "errors": errors, "warnings": warnings}, indent=1) + "\n")
    print("PASS" if not errors else "FAIL", f"objective.json ({len(errors)} errors, {len(warnings)} warnings)")
    for e in errors:
        print("  ERROR:", e)
    for w in warnings:
        print("  WARNING:", w)
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
