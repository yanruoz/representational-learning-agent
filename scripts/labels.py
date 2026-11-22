"""Readable names and number formatting shared by report.py and package_deliverables.py.

   model_labels(plan, objective) -> {model_id: readable label}, e.g.
     two_tower        "DINOv2 + caption alignment (InfoNCE)"
     frozen_probe     "Frozen BiomedCLIP (reference)" or "Frozen DINOv2 (control: no partner modality)"
     linear_alignment "CCA on LSI / LSI (linear baseline)"
   Labels are built from the plan (method, encoders, loss, options), never from dataset names, and
   made unique by adding the partner encoder, then the id, when two would read the same.

   merge_identical(rows) -> rows where baselines with identical metric values share one row.
   sig3(x) -> 3 significant figures (full precision stays in the JSON).
"""
import plan_schema

ENCODER_NAMES = {"pca": "PCA", "lsi": "LSI", "pixels_pca": "pixel PCA", "tfidf_svd": "TF-IDF",
                 "cache:dinov2_vitb14_frozen": "DINOv2", "cache:biomedclip_frozen": "BiomedCLIP"}
LOSS_NAMES = {"infonce": "InfoNCE", "vicreg": "VICReg", "supcon": "SupCon", "reconstruction": "reconstruction",
              "cross_entropy": "cross-entropy"}


def enc_name(enc, params=None):
    name = ENCODER_NAMES.get(enc, enc.split(":", 1)[-1])
    if params and params.get("reduce_to"):
        name += f" reduced to {params['reduce_to']} dims"
    return name


def model_labels(plan, objective):
    inf = objective.get("inference_modalities", [])
    trained_encoders = {(m, s["encoder"]) for e in plan.get("models", []) if plan_schema.METHODS.get(e["method"], {}).get("trains")
                        for m, s in e["inputs"].items()}
    base, partner, base_inputs = {}, {}, {e["id"]: e.get("inputs", {}) for e in plan.get("models", [])}
    for e in plan.get("models", []):
        ins, meth = e.get("inputs", {}), e["method"]
        mods = list(ins)
        if meth == "two_tower":
            main = next((m for m in mods if m in inf and len(inf) == 1), mods[0])
            other = [m for m in mods if m != main][0]
            if len(inf) > 1:
                lab = f"Two-tower {main}/{other} on {enc_name(ins[main]['encoder'])}"
            else:
                lab = f"{enc_name(ins[main]['encoder'], ins[main].get('params'))} + {other}"
            lab += f" ({LOSS_NAMES.get(e.get('loss'), e.get('loss'))}"
            if (e.get("hyperparameters") or {}).get("concat_input"):
                lab += ", input kept"
            lab += ")"
            partner[e["id"]] = enc_name(ins[other]["encoder"])
        elif meth in ("multimodal_autoencoder", "supervised_fusion"):
            lab = f"{'Autoencoder' if meth == 'multimodal_autoencoder' else 'Supervised fusion'} on " + \
                  " + ".join(f"{m} {enc_name(s['encoder'])}" for m, s in ins.items())
        elif meth == "linear_alignment":
            lab = "CCA, " + "/".join(enc_name(s["encoder"]) for s in ins.values())
        elif meth == "frozen_probe":
            (m, s), = list(ins.items())[:1]
            control = any((m, s["encoder"]) == te for te in trained_encoders)
            lab = f"Frozen {enc_name(s['encoder'], s.get('params'))} " + ("(control)" if control else "(reference)")
        elif meth == "linear_supervised":
            lab = "/".join(enc_name(s["encoder"]) for s in ins.values()) + " + logistic/PLS"
        else:
            lab = "/".join(enc_name(s["encoder"], s.get("params")) for s in ins.values()) + " (no training)"
        base[e["id"]] = lab
    out, seen = {}, {}
    for mid, lab in base.items():
        seen[lab] = seen.get(lab, 0) + 1
    for mid, lab in base.items():
        if seen[lab] > 1 and mid in partner:
            lab = lab[:-1] + f", {partner[mid]} {next(m for m in base_inputs[mid] if base_inputs[mid][m]['encoder'] != None and enc_name(base_inputs[mid][m]['encoder']) == partner[mid])})"
        out[mid] = lab
    counts = {}
    for lab in out.values():
        counts[lab] = counts.get(lab, 0) + 1
    return {mid: (lab if counts[lab] == 1 else f"{lab} [{mid}]") for mid, lab in out.items()}


def row_label(name, summary_row, labels):
    lab = labels.get(summary_row["model_id"], summary_row["model_id"])
    b = summary_row.get("budget")
    return lab + (f", trained with {'all labels' if b == 'all' else str(b) + ' labels/class'}" if b is not None else "")


def summary_row_id(r):
    return r["model_id"]


def merge_identical(summary, labels):
    """[(label, metrics)] with baselines whose every shared metric mean is equal merged into one row."""
    rows = []
    for name, r in summary.items():
        lab = row_label(name, r, labels)
        is_base = r["method"] in plan_schema.BASELINE_METHODS
        for i, (olab, om, obase) in enumerate(rows):
            if is_base and obase and set(om) == set(r["metrics"]) and all(
                    round(om[k]["mean"], 6) == round(r["metrics"][k]["mean"], 6) for k in om):
                rows[i] = (f"{olab} (= {lab})" if len(olab) + len(lab) < 44 else f"{olab} (same as {summary_row_id(r)})", om, True)
                break
        else:
            rows.append((lab, r["metrics"], is_base))
    return [(lab, m) for lab, m, _ in rows]


def sig3(x):
    if x is None:
        return "-"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    if x == 0:
        return "0"
    return f"{x:.3g}"


def cell(v):
    """mean ± SD (SD only when there are several seeds), 3 significant figures."""
    if v is None:
        return "-"
    return sig3(v["mean"]) + (f"±{v['sd']:.1g}" if v.get("n", 1) > 1 else "")
