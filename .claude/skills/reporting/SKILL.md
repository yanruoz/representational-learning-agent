---
name: reporting
description: Write findings.md for the generated report - reader-first, every number traceable to an artifact, decisions explained, limitations stated. Used in step 11 of AGENT.md.
---

# Reporting

`report.py` builds every table and figure page from the JSON files. You write only `outputs/<dataset>/findings.md`, the narrative a reader needs to understand what was decided and what it means.

## Structure of findings.md

```
## Answer
2-4 sentences answering the objective directly, with the headline numbers vs their baselines and spread.

## How the agent decided
- Inference vs training-only modalities, and the split (one line each, with the reason).
- Encoder per modality: which candidates the baseline check compared, which won, and why.
- Training loss: which, and the evidence (redundancy, labels, matching numbers).
- Baselines and references included, and why each matters.

## Results
One short paragraph per task: the learned model vs each baseline, with mean ± SD and the spread rule. Mention kNN vs linear disagreements and per-class effects that explain the average.

## Secondary findings
Results on secondary labels and anything unexpected (e.g. a frozen encoder already near the ceiling).

## Critique and revision
What the critic flagged, and what changed (or why nothing could).

## Limitations
- One bullet per limitation: data caveats from the description (label construction, pretraining overlap, small classes), training adequacy, what was not tested.
```

## Rules

- **Every decimal number must appear in an artifact** (metrics, summary, profile, baseline check, runs, plan) at the precision you write. `report.py` refuses to build otherwise. Copy numbers; never round differently or compute new ones (no differences, no ratios, no percent changes).
- Spell out an abbreviation the first time unless the report's glossary defines it (kNN, macro-F1, FOSCTTM, recall@k, ARI, NMI, PCA, LSI, TF-IDF, CCA, PLS, InfoNCE, VICReg, SupCon, effective rank).
- Use model ids as they appear in the plan, and say in plain words what each is the first time.
- No field names or JSON paths in prose.
- Say "no clear difference" when a gap is within the spread. Never claim more than the numbers show.
- After `report.py`, look at the page images in `outputs/<dataset>/report_pages/`. Fix any page that is empty, cut off, or unreadable, then rebuild.
