"""Evidence for choosing encoders and losses, computed before any model is chosen.

   Usage: venv/bin/python scripts/baseline_check.py outputs/<d>/plan.json
   (needs objective.json and a plan with its split; models may be empty)

   Everything is fit on the TRAINING split and scored on the VALIDATION split. The test split is never
   used, so choices made from these numbers don't leak test information.

   Writes outputs/<d>/metrics/baseline_check.json:
     encoders.<modality>.<encoder>      dim, effective rank, and for each label task that embeds this
                                        modality: kNN / linear-probe macro-F1 per label budget (mean
                                        and SD over draws), with the majority-class reference
     cross_modal.<a>|<b>.<encA>|<encB>  CCA matching (validation FOSCTTM, recall@10) per number of
                                        components, for every candidate pair of encoders: how much
                                        the two modalities share, as linear methods can find it
     zero_shot.<encoder>.<a>-><b>       matching on validation for a cached encoder pretrained jointly
                                        on both modalities (no training on this dataset)
     redundancy                         copied from profile.json (near-duplicates per representation)
"""
import argparse
import json
from pathlib import Path

import numpy as np

import context
import evaluate_rep
import extract_embeddings
import features
import plan_schema
from baselines import cca_inputs

MAX_DRAWS = 3


def main():
    ap = argparse.ArgumentParser(description="Frozen-feature evidence per candidate encoder (train -> validation).")
    ap.add_argument("plan")
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    out = context.OUTPUTS / plan["dataset"]
    objective = json.loads((out / "objective.json").read_text())
    ctx = context.Context(plan["dataset"], plan, objective)
    tr, va = ctx.train, ctx.val
    modalities = list(dict.fromkeys(objective["inference_modalities"] + objective.get("training_only_modalities", [])))

    label_tasks = []
    for t in objective.get("tasks", []) + objective.get("secondary_tasks", []):
        if t["task"] == "label_prediction":
            label_tasks.append((t["label"], t.get("modality"), t.get("label_budgets_per_class", ["all"])))
        elif t["task"] == "label_transfer":
            label_tasks.append((t["label"], t["from"], t.get("label_budgets_per_class", ["all"])))

    res = {"dataset": plan["dataset"], "note": "fit on the training split, scored on the validation split; test never used",
           "split_sizes": {k: int(len(v)) for k, v in ctx.split.items()}, "encoders": {}, "cross_modal": {}, "zero_shot": {}}
    feats = {}
    for m in modalities:
        res["encoders"][m] = {}
        for enc in features.available_encoders(ctx, m):
            Z = features.featurize(ctx, m, enc, fit_idx=tr)
            feats[(m, enc)] = Z
            entry = {"dim": int(Z.shape[1]), "effective_rank": evaluate_rep.collapse(Z[va])["effective_rank"],
                     "no_pretraining": enc in plan_schema.NO_PRETRAINING, "labels": {}}
            for label, lmod, budgets in label_tasks:
                if lmod not in (None, m) or (lmod is None and m not in objective["inference_modalities"]):
                    continue
                lm = evaluate_rep.label_metrics(ctx, Z, label, budgets, eval_idx=va, max_draws=MAX_DRAWS)
                entry["labels"][label] = {b: {k: v for k, v in r.items() if k.endswith(("macro_f1", "accuracy")) or k == "unfilled_classes"}
                                          for b, r in lm["budgets"].items()}
            res["encoders"][m][enc] = entry
            print(f"[baseline_check] {m} / {enc}: dim {entry['dim']}", flush=True)

    from sklearn.cross_decomposition import CCA
    for i, a in enumerate(modalities):
        for b in modalities[i + 1:]:
            pair = res["cross_modal"].setdefault(f"{a}|{b}", {})
            for ea in features.available_encoders(ctx, a):
                for eb in features.available_encoders(ctx, b):
                    A, B = cca_inputs(feats[(a, ea)], tr), cca_inputs(feats[(b, eb)], tr)
                    by_k = {}
                    for k in plan_schema.CCA_COMPONENT_CANDIDATES:
                        if k > min(A.shape[1], B.shape[1]):
                            continue
                        try:
                            Pa, Pb = CCA(n_components=k, max_iter=1000).fit(A[tr], B[tr]).transform(A, B)
                        except Exception as e:  # record, don't stop the check
                            by_k[str(k)] = {"error": f"{type(e).__name__}: {e}"}
                            continue
                        mm = evaluate_rep.matching_metrics(Pa, Pb, va)
                        by_k[str(k)] = {"foscttm": mm["foscttm"], "recall@10": mm["recall@10"]}
                    by_k_ok = {k: v for k, v in by_k.items() if "foscttm" in v}
                    if not by_k_ok:
                        pair[f"{ea}|{eb}"] = {"cca_validation_by_components": by_k, "error": "CCA failed for every size"}
                        continue
                    best = min(by_k_ok, key=lambda k: by_k_ok[k]["foscttm"])
                    pair[f"{ea}|{eb}"] = {"cca_validation_by_components": by_k, "best_components": int(best),
                                          "best_foscttm": by_k[best]["foscttm"], "chance_foscttm": 0.5}
                    print(f"[baseline_check] CCA {a}:{ea} | {b}:{eb}: best FOSCTTM {by_k_ok[best]['foscttm']}", flush=True)

    for enc_id, enc in sorted(extract_embeddings.ENCODERS.items()):
        mods = extract_embeddings.applicable(ctx.spec, enc)
        key = f"cache:{enc_id}"
        if not mods or len(mods) < 2:
            continue
        names = [n for n in mods.values() if (n, key) in feats]
        if len(names) == 2:
            a, b = names
            res["zero_shot"][key] = {f"{a}->{b}": evaluate_rep.matching_metrics(feats[(a, key)], feats[(b, key)], va)}

    prof = out / "profile.json"
    if prof.exists():
        res["redundancy"] = json.loads(prof.read_text()).get("redundancy", {})
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "metrics" / "baseline_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"[baseline_check] wrote {out / 'metrics' / 'baseline_check.json'}")


if __name__ == "__main__":
    main()
