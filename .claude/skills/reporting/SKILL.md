---
name: reporting
description: Write findings.md for the generated report - reader-first, every number traceable to an artifact, decisions explained, limitations stated. Used in step 11 of AGENT.md.
---

# Reporting

`report.py` builds every table and figure page from the JSON files. You write only `outputs/<dataset>/findings.md`, the narrative a reader needs to understand what was decided and what it means.

## How the report is built (so you know what not to repeat)

`report.py` builds a report of at most 15 pages (target 10-12; it refuses to build above 15). Each fact appears once:

| Section | Comes from | Holds |
|---|---|---|
| Summary | `findings.md` → `## Answer`, objective, recommendation verdicts | the short answer |
| Setup | objective, profile, split | tasks, data, split, abbreviations |
| Decisions | plan.json, baseline check | one table: decision, choice, evidence, alternative rejected; each distinct reason once |
| Results | metrics | primary-task table first, then other tasks, secondary labels, collapse; readable model labels; 3 significant figures |
| Figures | figures/ | two per page, each with a computed one-sentence takeaway |
| Findings | `findings.md` (all sections except Answer, Critique, Limitations) | the evidence and its interpretation |
| Downstream use | recommendation.json | one line per use: verdict and reason, no numbers |
| Critique | `critique_response.json` | finding, severity, what changed, effect on the final result |
| Limitations and reproducibility | data description caveats, run issues, `findings.md` → `## Limitations` | one list, duplicates removed; pointers to files |

Machine-oriented material (evidence path = value, per-run logs, config hashes, hyperparameter reasons) goes to `deliverables/` files, never into your prose.

## Structure of findings.md

```
## Answer
3-5 sentences answering the objective directly: the headline result against its baseline, with spread, and the recommendation in one clause. This is the Summary; don't repeat it below.

## Evidence
One short paragraph per task, in the order primary task first: the learned model vs each baseline (mean ± SD), the spread rule, kNN vs linear disagreements, the per-class effects that explain an average. Refer to models by what they are ("the image encoder aligned to the text with InfoNCE", "the frozen pretrained reference"), not by plan ids.

## Secondary findings
Results on secondary labels and anything unexpected (e.g. a frozen encoder already near the ceiling).

## Limitations
One bullet per limitation that the data description and the run log don't already state.
```

Do not write a "How the agent decided" or "Critique" section: the Decisions and Critique pages are built from plan.json and critique_response.json. Keep `findings.md` under about 900 words.

## Rules

- **Every decimal number must appear in an artifact** (metrics, summary, profile, baseline check, runs, plan) at the precision you write. `report.py` refuses to build otherwise, and `traceability.md` lists where each number comes from. Copy numbers; never compute new ones (no differences, ratios, or percent changes).
- **Use final numbers only.** Numbers from an earlier revision belong nowhere in the report; `history/` keeps them.
- State each fact once. The tables carry the full numbers; prose cites only the few that support a claim.
- Readable names in prose; plan ids appear only in the Decisions table and the files.
- Spell out an abbreviation the first time unless the glossary defines it (kNN, macro-F1, FOSCTTM, recall@k, ARI, NMI, PCA, LSI, TF-IDF, CCA, PLS, InfoNCE, VICReg, SupCon, effective rank).
- Say "no clear difference" when a gap is within the spread. Never claim more than the numbers show.
- After `report.py`, look at the page images in `outputs/<dataset>/report_pages/`. If it refuses because of length, shorten `findings.md` (remove repetition, not findings) and rebuild.
