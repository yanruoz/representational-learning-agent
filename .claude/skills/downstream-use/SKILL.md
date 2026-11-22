---
name: downstream-use
description: Write recommendation.json - which representation to export and what it can and cannot be used for, each verdict backed by cited metric values. Used in step 10 of AGENT.md.
---

# Downstream use

## recommendation.json

```json
{
  "chosen_model": "<plan model id>",
  "chosen_reason": "...",
  "chosen_evidence": [{"file": "metrics/summary.json", "key": "<model>.metrics.<metric path>", "value": 0.0}],
  "uses": [
    {"use": "...", "verdict": "recommended | conditional | not_recommended", "reason": "...",
     "evidence": [{"file": "metrics/summary.json", "key": "...", "value": 0.0},
                  {"file": "metrics/summary.json", "key": "<baseline>.metrics...", "value": 0.0}]}
  ]
}
```

Validate (this also exports the chosen embeddings and `summary.json` for other tools): `venv/bin/python scripts/validate_recommendation.py outputs/<dataset>/recommendation.json`.

## Choosing the representation

- Choose by the **primary task's** metric on held-out samples. If no learned model beats its baseline by more than the spread, choose the baseline and say so. A simpler representation that is as good is the better choice.
- Values in `summary.json` are means over seeds. Quote them at the precision the validator checks (e.g. 4 decimals); `key` is the full path such as `<model>.metrics.<task>.<...>.<metric>`.

## Uses and verdicts

One entry per use the objective implies, plus any use a reader would naturally try:
- `recommended`: the evidence clearly supports it. Cite the model's value and the baseline or reference it beats.
- `conditional`: it works under a stated condition (e.g. only with enough labels per class, only for the larger classes, only with the kNN classifier).
- `not_recommended`: it fails, or nothing measured it. Say what would be needed.

Never recommend a use for a modality the model wasn't evaluated on. If the frozen pretrained reference beats the learned model on a use, say that the reference is the better choice for that use, and keep the pretraining-overlap caveat.
