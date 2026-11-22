---
name: method-selection
description: Choose the representation-learning method, the encoder for each modality, and the training loss for the validated objective, citing evidence from profile.json and metrics/baseline_check.json. Used when writing plan.json, after the data-profiling skill has set the split and preprocessing.
---

# Method, encoder, and loss selection

You make three choices per model, and each one is a real choice between candidates:

1. **Method**: how modalities are combined (table below).
2. **Encoder per modality**: what turns raw data into input features (a computed featurizer or a cached frozen pretrained encoder), plus the trainable head on top.
3. **Training loss**: for two-tower methods, `infonce`, `vicreg`, or `supcon`.

Every choice gets a `reason` and an `evidence` list. Each evidence item names a file (`profile.json` or `metrics/baseline_check.json`), a key path, and the value you read there; `validate_plan.py` checks that the value matches. A reason that would read the same for any dataset of this shape is not a reason. "Default", "standard", or "commonly used" is never a reason; the validator rejects it.

## Model entry schema (plan.json → models)

```json
{
  "id": "<short_id>", "method": "two_tower", "loss": "infonce", "label": null,
  "inputs": {
    "<modality>": {"encoder": "<from baseline_check candidates>", "head": "mlp", "reason": "...",
                   "evidence": [{"file": "metrics/baseline_check.json", "key": "encoders.<modality>.<encoder>....", "value": 0.0}]}
  },
  "loss_reason": "...",
  "loss_evidence": [{"file": "profile.json", "key": "redundancy.<representation>.mean_fraction_of_knn_that_are_near_duplicates", "value": 0.0}],
  "hyperparameters": {"embed_dim": 128, "hidden_dims": [512, 256], "epochs": 300, "batch_size": 256, "lr": 0.0003, "patience": 30},
  "hyperparameter_reasons": {"embed_dim": "...", "hidden_dims": "...", "epochs": "...", "batch_size": "...", "lr": "...", "patience": "..."},
  "seeds": [0, 1, 2], "serves_tasks": ["cross_modal_retrieval"], "estimated_minutes": 0.5, "status": "planned", "reason": "..."
}
```

- Baselines (`single_modality_features`, `frozen_probe`, `linear_supervised`, `linear_alignment`) need `inputs` with `encoder` and `reason` (`frozen_probe` also `evidence`) and no loss. `linear_alignment` takes `seeds`: seed 0 fits on all training rows, others on seeded 90% subsamples, so the baseline has a spread to compare against. Give it as many seeds as the learned models.
- A baseline only produces metrics for tasks whose modalities are among its inputs; plan one per modality the tasks need.
- Other hyperparameters: `weight_decay`, `dropout`, `temperature_init` (infonce), `supcon_temperature`, `modality_dropout` (autoencoder, fusion), `concat_input`, `max_minutes`. Every key you set needs a reason.
- `label`: the label a `supcon` / `cross_entropy` model trains on (default: the objective's label task). It is trained once per label budget.
- `estimated_minutes` per run: small heads on fixed features take well under a minute on CPU for about 10k samples; label-using losses run once per budget, so multiply.
- Status `not_run_budget` keeps a model you would run but that doesn't fit `budget_min`; the report lists it.

## Step 1: what the objective needs at prediction time

Read `objective.json`: which modalities are available for **new** samples (`inference_modalities`) and which are available only for training. This rules methods in or out before anything else:

- If new samples have only one modality, the model must embed that modality **alone**. Two-tower models qualify (use that modality's tower). An autoencoder qualifies only if trained with modality dropout. Supervised fusion of all modalities does not, unless trained with modality dropout.
- If the objective asks for one modality's values to be estimated from another (cross-modal prediction), the method needs a path from the input modality to the target: a decoder, or nearest neighbors in a shared space whose target values are averaged.
- If the objective asks whether a training-only modality helps, plan a matched pair: the same encoder and head trained with and without that modality.

## Step 2: method (capabilities)

| Method | Needs labels | Embeds one modality alone | Cross-modal prediction | Notes |
|---|---|---|---|---|
| `single_modality_features` (baseline) | no | yes | via nearest neighbors in its own space | always planned for each inference modality |
| `linear_supervised` (baseline) | yes, or a target modality | yes | yes (PLS regression) | logistic regression for labels, PLS for cross-modal targets |
| `frozen_probe` | for the probe only | yes | no | cached encoder features + kNN / linear probe, no training |
| `linear_alignment` (baseline) | no | yes (one projection per modality) | via nearest neighbors in the shared space | CCA; size chosen on validation; the required baseline for matching and label transfer |
| `two_tower` | only with `supcon` | yes | via nearest neighbors in the shared space | loss: `infonce`, `vicreg`, `supcon` |
| `multimodal_autoencoder` | no | with modality dropout | yes (decode the other modality) | reconstruction loss per modality |
| `supervised_fusion` | yes | with modality dropout only | no | embedding = fused layer before the classifier |

## Step 3: encoder per modality

Candidates are listed in `metrics/baseline_check.json` under `encoders`: computed featurizers (e.g. PCA, LSI, TF-IDF + SVD, downsampled pixels + PCA) and every cached frozen encoder whose cache is ready (`profile.json` → `cache`). For each candidate, the baseline check reports frozen-feature performance on a validation split carved from the training data, never the test split.

- Prefer the candidate with better validation performance for the objective's task, and cite both numbers. If the difference is within the seed spread the check reports, say so and decide on another ground (dimension, speed, what the comparison in Step 1 needs).
- **Pretraining overlap.** `profile.json` → `cache.<encoder>.pretraining_data` says what each encoder was trained on; `DATA_DESCRIPTION.md` says where the dataset comes from. If they may overlap, that encoder's numbers can be inflated. Keep a candidate without that overlap in the plan, and state the risk in the reason.
- **Confounded comparisons.** If the objective asks whether training with modality B helps modality A, an encoder for A that was itself pretrained together with B already contains B's influence. For the clean comparison use an A-encoder pretrained without B; the jointly pretrained encoder can still be reported as a reference.
- Always keep one encoder with no pretraining per inference modality as a baseline.

## Step 4: training loss for two-tower models

Paired modalities are natural views of the same sample: each sample's other modality is its positive, so no data augmentation is needed.

- **`infonce`** (CLIP-style, symmetric): every other sample in the batch is a negative. When many training samples are near-duplicates of each other, those negatives are false negatives, and the loss pushes apart samples that should be close. Evidence: `profile.json` → `redundancy.<modality>.mean_fraction_of_knn_that_are_near_duplicates` and `samples_with_a_near_duplicate`, and, if labels exist, the expected number of same-class samples per batch (batch size x class fraction, from `labels.<name>.class_counts`). The near-duplicate cutoff is a fixed cosine, and representations differ in how close unrelated samples sit: compare each number with `random_pairs_at_or_above_cutoff` and `random_pair_cosine` for the same representation, and judge redundancy in the representation the loss will actually be trained on. Low redundancy favors `infonce`. With high redundancy, prefer `vicreg` or `supcon`, or justify keeping `infonce` (e.g. a smaller batch).
- **`vicreg`**: no negatives. Invariance between the two views plus a variance term and a covariance term that prevent collapse. Robust to redundancy. Check the collapse metrics after training (`effective_rank`, low-variance dimensions).
- **`supcon`**: same-label samples are positives. **Needs labels on the training samples it uses.** If the objective evaluates a small labeled fraction, supcon may use only that fraction's labels, or the evaluation is leaked. It cannot be the only method when labels are missing for most samples.
- Alignment compresses: a learned embedding usually has a much lower effective rank than its input features (`metrics/collapse.json`). If a linear probe on the frozen input beats the aligned embedding, `concat_input: true` keeps both; plan it as a separate model with a reason, not as a silent default.
- State the batch size and temperature (for `infonce` / `supcon`) with reasons tied to the sample count and the redundancy numbers.

**Sample size.** Compare the number of labeled training samples with the size of a supervised head. With a few hundred labeled samples or fewer, supervised heads and fusion classifiers overfit. Prefer label-free losses, small linear heads, and probes on fixed features, and say so in the reason.

## Step 5: baselines and comparisons (required)

For each task in `objective.json`, the plan includes at least:
- the no-pretraining single-modality baseline on the inference modality, and
- a linear supervised baseline (logistic regression for labels, PLS regression for a cross-modal target).
- for matching (`cross_modal_retrieval`) and `label_transfer`: a `linear_alignment` (CCA) model on no-pretraining features.

The learned model is judged against both, with the seed spread. A frozen pretrained encoder is reported as a reference, and as a bar only if the objective asks to beat it.
