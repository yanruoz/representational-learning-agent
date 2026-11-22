---
name: evaluation
description: Which metric answers which task, which baselines and references each needs, and how to read seed and draw spread. Used when planning (step 6 of AGENT.md) and when reading results (step 8).
---

# Evaluation

All evaluation runs on held-out test samples, with inputs only from the inference modalities (`scripts/evaluate_rep.py`). You choose which comparisons exist by choosing models; the scripts compute every number.

## Metric per task

| Task | Metric (file: `metrics/summary.json`, key prefix) | Reference |
|---|---|---|
| `cross_modal_retrieval` | `foscttm` (lower is better), `recall@k`, `median_rank` | chance FOSCTTM 0.5, chance recall k / n; a `linear_alignment` (CCA) baseline |
| `label_transfer` | `knn_macro_f1`, `linear_macro_f1` (transfer from one modality's embeddings to the other's) | CCA transfer; a classifier trained on the target modality's own labels is an upper reference only |
| `label_prediction` | `linear_macro_f1`, `knn_macro_f1` per budget; `per_class_f1` | majority class; a no-pretraining baseline; frozen pretrained encoders as references |
| `cross_modal_prediction` | `pearson_median` per target feature | kNN averaging in a no-pretraining space; PLS (`linear_supervised`) |
| `clustering` | `ari`, `nmi` | the same on a no-pretraining baseline |

Macro-F1 weights every class equally, so a few small classes can move it a lot. Read `per_class_f1` before explaining a difference.

## Spread and "better"

- `summary.json` gives mean ± SD over **seeds** (trained models); each run's metrics file also has SD over **label draws** for small budgets.
- Call a model better only if the gap is larger than both spreads. Otherwise write "no clear difference". Gaps between single-run baselines can't be judged against a spread; say so.
- If kNN and the linear probe disagree (one improves, the other doesn't), report both. It tells you what changed in the geometry of the embedding.

## Label use

- A label-using loss (`supcon`, `cross_entropy`) trains one model per budget, using only that budget's labels. Its label metrics answer "with this many labels"; never compare it at a budget it wasn't trained for.
- Clustering and label scores of a label-trained representation are not label-free results; say so.

## Collapse and adequacy (`metrics/collapse.json`, `runs/*.json`)

- Effective rank far below the input's, or many near-constant dimensions, means collapsed or over-compressed embeddings, even if probe scores look fine.
- `stop_reason: max_epochs` on an early-stopping run means the validation loss may still have been improving.
- An InfoNCE run whose final temperature sits at its bound may be miscalibrated.
