---
name: objective-interpretation
description: Turn the free-text objective into objective.json (tasks, inference vs training-only modalities, label budgets, success criterion). Used in step 3 of AGENT.md.
---

# Objective interpretation

## objective.json schema

```json
{
  "dataset": "<dataset>",
  "objective_text": "<the objective, verbatim>",
  "decision": "proceed | proceed_adapted | decline",
  "decision_reason": "...",
  "inference_modalities": ["<modalities available for new samples>"],
  "training_only_modalities": ["<modalities available only in this training data>"],
  "tasks": [
    {"task": "cross_modal_retrieval", "query": "<m1>", "gallery": "<m2>", "primary": true},
    {"task": "label_transfer", "label": "<label>", "from": "<m1>", "to": "<m2>", "label_budgets_per_class": ["all"]},
    {"task": "label_prediction", "label": "<label>", "modality": "<m>", "label_budgets_per_class": [5, 20, 100, "all"]},
    {"task": "cross_modal_prediction", "source": "<m1>", "target": "<m2>"}
  ],
  "secondary_tasks": [{"task": "label_prediction", "label": "<other label>", "modality": "<m>", "label_budgets_per_class": ["all"]}],
  "success_criterion": {"metric": "...", "baseline": "...", "comparison": "...", "reason": "..."},
  "interpretation_notes": "ambiguities and how you resolved them"
}
```

Validate: `venv/bin/python scripts/validate_objective.py outputs/<dataset>/objective.json --objective "<text>"`.

## Step 1: who has which modality, when

The most important reading of the objective is **what a new sample will have**.
- "Future samples will have only X", "from X alone", "X-only" → `inference_modalities: [X]`; the others are `training_only_modalities`.
- "Measured with X in one experiment and Y in another" → both are inference modalities, but **each is embedded on its own** (no test sample has both). Tasks that compare them are `cross_modal_retrieval` (matching) and `label_transfer`.
- A modality the objective describes as "used during training", "available for training", or present only in the archive is training-only.

## Step 2: tasks

| The objective asks to... | Task |
|---|---|
| match or find the same sample across modalities | `cross_modal_retrieval` (FOSCTTM, recall@k) |
| move annotations from one modality's samples to another's | `label_transfer` (from the annotated modality to the other) |
| identify a declared label from some modalities | `label_prediction` with `modality` |
| estimate one modality's values from another | `cross_modal_prediction` (target must be a count modality) |
| find structure / groups | `clustering` (scored against a label if one exists) |

Exactly one task is `primary`: the one the objective's main question is about. If the objective asks for a label-free result first, the label-free task is primary.

## Step 3: label budgets

- If the objective says labels are scarce ("a few labeled examples", "only some are annotated"), use **per-class** budgets that span very few to all (e.g. a handful, a few dozen, about a hundred, and `"all"`). Check them against `profile.json → labels.<label>.class_counts_by_split.train`: every integer budget should be fillable by the classes you keep after reserving a validation share. Classes that can't fill a budget are handled in the plan (excluded with a reason, or listed as not filled).
- Otherwise use `["all"]`.

## Step 4: secondary tasks

Other declared labels that the objective does not ask about become `secondary_tasks` (label prediction from the inference modalities). They are reported as findings, never as the main result.

## Step 5: decision and success criterion

- `decline` only when the data cannot answer the objective (e.g. a required modality or label is not declared). Say exactly what is missing.
- `proceed_adapted` when you had to change the task (e.g. the objective asks for a label that does not exist, so you substitute the closest declared one). Say what changed.
- The success criterion names the metric, the baseline it must beat (a baseline with no pretraining, a linear method such as CCA or PLS, or a single-modality model), and the comparison, including the seed spread (e.g. "lower FOSCTTM than CCA by more than the SD over seeds"). A frozen pretrained encoder is a reference, not the bar, unless the objective asks to beat it.
- If the data description says labels were derived from some modality, note in `interpretation_notes` how that biases label-based comparisons.
