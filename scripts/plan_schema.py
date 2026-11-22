"""Shared constants for objective.json and plan.json: tasks, methods and their capabilities,
   encoders per modality kind, losses, heads. Imported by the validators, features.py, train.py,
   evaluate_rep.py, baseline_check.py, and run_plan.py, so these lists can't drift apart.

   Pattern copied from Project 2's plan_schema.py (fixed tables + validator); contents rewritten
   for multimodal representation learning. Nothing here names a dataset.
"""

# ── Objectives ──
TASKS = {
    "label_prediction",        # predict a declared label from the inference modalities' embedding
    "cross_modal_prediction",  # estimate a training-only modality's values from the inference modalities
    "cross_modal_retrieval",   # find a sample's partner in another modality
    "clustering",              # unsupervised structure that agrees with a label
    "joint_embedding",         # a representation for downstream use, judged by the above
}
DECISIONS = {"proceed", "proceed_adapted", "decline"}
BUDGET_ALL = "all"

# ── Methods ──
# labels: needs labels in representation training. single: can embed one modality alone.
# paired: needs paired samples for training. shared_space: per-modality embeddings live in one space
# (so cross-modal retrieval and kNN across modalities make sense). trains: has trainable weights.
METHODS = {
    "single_modality_features": {"labels": False, "single": True, "paired": False, "shared_space": False, "trains": False},
    "linear_supervised":        {"labels": False, "single": True, "paired": False, "shared_space": False, "trains": False},
    "frozen_probe":             {"labels": False, "single": True, "paired": False, "shared_space": False, "trains": False},
    "two_tower":                {"labels": False, "single": True, "paired": True,  "shared_space": True,  "trains": True},
    "multimodal_autoencoder":   {"labels": False, "single": True, "paired": True,  "shared_space": True,  "trains": True},
    "supervised_fusion":        {"labels": True,  "single": True, "paired": True,  "shared_space": True,  "trains": True},
}
BASELINE_METHODS = {"single_modality_features", "linear_supervised", "frozen_probe"}
# Which tasks each method can serve. A method serves label_prediction / clustering / joint_embedding
# through its embedding; cross_modal_prediction through kNN averaging (all) or PLS (linear_supervised);
# retrieval only with a shared space.
METHOD_TASKS = {
    m: {"label_prediction", "clustering", "joint_embedding", "cross_modal_prediction"}
       | ({"cross_modal_retrieval"} if caps["shared_space"] else set())
    for m, caps in METHODS.items()
}

LOSSES = {
    "two_tower": {"infonce", "vicreg", "supcon"},
    "multimodal_autoencoder": {"reconstruction"},
    "supervised_fusion": {"cross_entropy"},
}
LOSSES_NEEDING_LABELS = {"supcon", "cross_entropy"}
HEADS = {"none", "linear", "mlp"}

# ── Encoders (featurizers) per modality kind ──
# Computed ones are fit on training samples only. Cached ones come from extract_embeddings.ENCODERS
# and are referred to as "cache:<encoder_id>".
COMPUTED_ENCODERS = {
    "counts": {"pca", "lsi"},
    "image": {"pixels_pca"},
    "text": {"tfidf_svd"},
}
NO_PRETRAINING = {"pca", "lsi", "pixels_pca", "tfidf_svd"}
ENCODER_DEFAULT_PARAMS = {
    "pca": {"n_features": 2000, "n_components": 50},
    "lsi": {"min_detected_fraction": 0.005, "n_components": 50, "drop_first": True},
    "pixels_pca": {"size": 32, "n_components": 64},
    "tfidf_svd": {"min_df": 2, "n_components": 128},
}

# Per-class label budgets: draws per budget (more draws where the probe is noisier).
def draws_for_budget(budget):
    if budget == BUDGET_ALL:
        return 1
    return 10 if budget <= 20 else 5


# Cross-modal prediction: kNN averaging neighbors, PLS components, target features.
KNN_PREDICT_K = 20
PLS_COMPONENTS = 20
DEFAULT_TARGET_FEATURES = 2000
