---
name: data-profiling
description: Decide the split, the training pool, class exclusions, and budget notes for plan.json from profile.json and DATA_DESCRIPTION.md. Used in step 4 of AGENT.md, before the baseline check.
---

# Data decisions: split, training pool, classes

Read `profile_summary.md` first, then the relevant parts of `profile.json` and the data description's "Known caveats".

## plan.json (data part)

```json
{
  "dataset": "<dataset>", "revision": 1, "budget_min": 30,
  "split": {"source": "declared | random", "test_fraction": 0.2, "val_fraction": 0.15, "seed": 0,
            "stratify_label": "<label or null>", "group_by": "<label used as group id, or omit>", "reason": "..."},
  "training_pool": {"filter": "all | label_present", "label": "<label, if label_present>", "reason": "..."},
  "excluded_classes": {"<label>": {"classes": ["..."], "reason": "...", "evidence": [{"file": "profile.json", "key": "...", "value": 0}]}},
  "budget_not_filled": {"<label>": {"classes": ["..."], "reason": "..."}},
  "models": []
}
```

## Split

- **Declared split** (`profile.json → samples.split_declared`): use it (`source: declared`) unless you have a stated reason not to. It is usually the split others report on.
- **Random split**: stratify by the primary task's label when there is one, so small classes appear in every part.
- **Groups.** If samples share a source (same patient, slide, donor, article, batch) and that source is declared as a label, split by it (`group_by`). Samples from one source are near-duplicates, and splitting them across train and test inflates every metric. If the data description says samples share sources but no group label is declared, say so as a limitation.
- The validation part (`val_fraction`) is carved from training: it picks epochs (early stopping), the CCA size, and feeds the baseline check. It is never a reported result.

## Training pool (representation learning only)

Which training samples the representation is trained on. Labels and test data are not involved.
- `all`: every paired training sample. More pairs usually help alignment.
- `label_present` with a label: only samples where that label exists, which is closer to the target domain when the label only applies to part of the data (e.g. one sub-type of samples).
This is a real trade-off (more data vs closer domain). State which you chose and why; if it is close, say it is a judgment call. Planning both as separate models is a valid answer when the budget allows.

## Classes

- `profile.json → labels.<label>.classes_under_<k>` and `test_classes_under_<k>` flag classes too small for stable metrics.
- A flagged class can be **excluded** (its samples become unlabeled for that label) or kept. Excluding removes noise from macro-F1, but the result no longer covers that class. Decide per class, with the counts as evidence.
- If an integer label budget can't be filled by a kept class, list it in `budget_not_filled` with a reason. `validate_plan.py` checks this against the actual training split.

## Reading the profile for later steps

- **Count modalities.** Sparsity, integer values, and the share of nonzero values equal to 1 tell you whether log-normalized PCA or TF-IDF/LSI suits the features. The baseline check measures both; let it decide.
- **Images and text.** Note the size spread, grayscale stored as color, and very short or duplicated texts.
- **Redundancy** (`redundancy`). Near-duplicates per representation; the method-selection skill uses it for the loss choice.
- **Cache** (`cache`). Which frozen pretrained encoders are ready and what they were pretrained on, for the overlap check.
- **`other_columns`**. Columns that are never model inputs. If the data description says some of them were derived from the labels, say so in the limitations.
