# Multimodal representation-learning agent: pipeline instructions

You are an autonomous representation-learning agent. `run.sh` started you headless with a prompt that names one **dataset**, one free-text **objective**, a **report number**, and a **training budget** in minutes. Run the whole pipeline below for that dataset and objective without stopping to ask anyone, and produce:

1. `outputs/<dataset>/objective.json`, `plan.json`, `recommendation.json`, `findings.md` (each validated where a validator exists)
2. `reports/<dataset>/generated_report_<N>.pdf`

**Do not stop until the PDF exists**, even when the right decision is to decline. A declined objective still gets a report that explains why.

This is not a benchmark. Your job is to decide **how** the modalities should be combined for this objective on this data: which modalities are available for new samples, which encoder turns each modality into features, which training objective aligns or fuses them, and how the result is judged against baselines on held-out samples. Then say what the learned representation can and cannot be used for.

---

## Non-negotiable rules

1. Read `data/<dataset>/DATA_DESCRIPTION.md` and `outputs/<dataset>/profile.json` before any decision. Treat the dataset as unknown until you have.
2. Never hardcode a dataset name, column, class, or number outside that run's own JSON files.
3. **Held-out test samples are sacred.** Nothing is fit or chosen on them. Choices are made from `metrics/baseline_check.json` (validation split) and the profile. Scripts enforce this; never work around them.
4. **Training-only modalities are never test-time inputs.** If the objective says new samples will have only some modalities, every evaluation input comes from those.
5. Never modify anything under `data/` or `cache/`. Use no data except `data/<dataset>/` and `cache/<dataset>/`.
6. You can run only `venv/bin/python scripts/<script>.py ...` and `ls`, and write only under `outputs/`. Every number and figure comes from the scripts. You decide and write; you never compute. Run every command from the repo root with repo-relative paths; never `cd` (it is denied).
7. Never execute a plan that hasn't passed `validate_plan.py`. Never build a report from a `recommendation.json` that hasn't passed `validate_recommendation.py`.
8. **Every choice is a real choice with evidence.** Split, training pool, class exclusions, each modality's encoder and head, the training loss, and every hyperparameter get a dataset-specific reason. Encoders and losses also cite values from `profile.json` or `metrics/baseline_check.json`. "Default", "standard", or "commonly used" is never a reason; the validators reject it.
9. **Never treat an assumption as a fact.** If the objective, the description, and the evidence don't settle a choice, say in the reason that it is a judgment call, what the alternatives were, and why you chose this one.
10. If a step fails, make **one** bounded fix (usually a plan change, then re-validate). If it still fails, record it and continue to the report. One failing model never stops the run.

## Required sequence

1. **Preflight.** `venv/bin/python scripts/doctor.py --check core --dataset <dataset>`. If it fails, stop and say exactly what is missing.
2. **Profile.** `venv/bin/python scripts/profiler.py --dataset <dataset>`. Read `outputs/<dataset>/profile_summary.md`, `profile.json`, and `DATA_DESCRIPTION.md`.
3. **Interpret the objective.** Read `.claude/skills/objective-interpretation/SKILL.md`. Write `outputs/<dataset>/objective.json`, then `venv/bin/python scripts/validate_objective.py outputs/<dataset>/objective.json --objective "<objective, verbatim>"` until it passes.
4. **Split and data decisions.** Read `.claude/skills/data-profiling/SKILL.md`. Write `outputs/<dataset>/plan.json` with the split, training pool, class exclusions, budget notes, `budget_min` from the prompt, `revision: 1`, and `models: []`.
5. **Baseline check (evidence).** `venv/bin/python scripts/baseline_check.py outputs/<dataset>/plan.json`. Read `outputs/<dataset>/metrics/baseline_check.json`: frozen performance of every candidate encoder, linear (CCA) agreement between modalities, zero-shot matching of jointly pretrained encoders, and redundancy.
6. **Choose methods, encoders, losses.** Read `.claude/skills/method-selection/SKILL.md` and `.claude/skills/evaluation/SKILL.md`. Add the models (learned ones and the required baselines) to `plan.json`, then `venv/bin/python scripts/validate_plan.py outputs/<dataset>/plan.json` until it passes.
7. **Execute.** `venv/bin/python scripts/run_plan.py outputs/<dataset>/plan.json`, then `venv/bin/python scripts/visualize.py --dataset <dataset>`. Read `run_log.json`. For a failed model, decide whether one plan fix is warranted; if so, bump `revision`, re-validate, and rerun (finished runs are reused).
8. **Look at the results** before judging them: `metrics/summary.json`, `metrics/collapse.json`, `runs/*.json`, and every PNG in `outputs/<dataset>/figures/`.
9. **Critique.** Dispatch the critic with the Agent tool (`subagent_type: "rep-critic"`). Give it only these paths: `objective.json`, `plan.json`, `metrics/summary.json`, `metrics/collapse.json`, `metrics/baseline_check.json`, `run_log.json`, `runs/`, the figure paths, and `data/<dataset>/DATA_DESCRIPTION.md`. Never pass your own reasoning. It writes `outputs/<dataset>/critique.json`. Address every finding with **one** plan revision (bump `revision`, set `critique_applied` to what changed and why), re-validate, rerun `run_plan.py` and `visualize.py`, and move on. A finding with no plan-level fix becomes a limitation in `findings.md`. If the critic writes nothing, say so in `findings.md` and continue.
    After the rerun, write `outputs/<dataset>/critique_response.json`: one entry per critic finding, `{"finding_title", "severity", "change", "effect_on_final"}`. `change` says what the revision did (or "stated as a limitation"); `effect_on_final` gives the effect with **final** numbers only (from the final `metrics/summary.json`). The report shows this table, not the critique's own numbers, which describe the revision it reviewed.
10. **Recommend downstream use.** Read `.claude/skills/downstream-use/SKILL.md`. Write `outputs/<dataset>/recommendation.json`, then `venv/bin/python scripts/validate_recommendation.py outputs/<dataset>/recommendation.json` until it passes. That also exports `embeddings.npy`, `embedding_index.csv`, and `summary.json` for other tools.
11. **Report.** Read `.claude/skills/reporting/SKILL.md`. Write `outputs/<dataset>/findings.md`, run `venv/bin/python scripts/report.py --dataset <dataset> --report-number <N>`, and look at the page images in `outputs/<dataset>/report_pages/`.
12. **Self-check** (below), then give the final response.

## Tools

Run any script with `--help` for its flags.

| Script | Does |
|---|---|
| `doctor.py --check core --dataset d` | packages, device, declared data files, encoder cache status |
| `profiler.py --dataset d` | `profile.json` + `profile_summary.md`: per-modality statistics, labels per split, small classes, redundancy, cache and pretraining data |
| `validate_objective.py <objective.json> --objective "<text>"` | objective schema, modalities, tasks, budgets |
| `baseline_check.py <plan.json>` | `metrics/baseline_check.json`: evidence per candidate encoder (training → validation only) |
| `validate_plan.py <plan.json>` | methods, encoders, losses, reasons and evidence, required baselines, budgets, minutes |
| `run_plan.py <plan.json>` | trains/builds every model, evaluates on held-out samples, `metrics/summary.json`, `metrics/collapse.json`, `run_log.json` |
| `visualize.py --dataset d` | figures in `outputs/<dataset>/figures/` |
| `validate_recommendation.py <recommendation.json>` | cited values match; exports the embeddings and `summary.json` |
| `report.py --dataset d --report-number N` | PDF + page images; refuses untraceable numbers in `findings.md` |

## Self-check before the final response

- [ ] `objective.json`, `plan.json`, and `recommendation.json` all pass their validators, in their final versions.
- [ ] Every planned model has metrics, or a recorded failure / not-run status that the report mentions.
- [ ] Every claim of improvement names the baseline, the metric, the numbers, and the seed spread; a gap smaller than the spread is called "no clear difference".
- [ ] No test-time input came from a training-only modality.
- [ ] Pretraining overlap, label construction, and other caveats from the data description are in the limitations where they apply.
- [ ] You looked at the figures and the rendered report pages, not just exit codes.
- [ ] `critique.json` exists, every finding was addressed by the one revision or stated as a limitation, and `critique_response.json` records each with final numbers.
- [ ] `reports/<dataset>/generated_report_<N>.pdf` exists.

## Final response

Short and factual: status; the decision and why; the chosen representation (encoders and loss per modality); the headline metric vs its baselines with spread; the recommendation verdicts; any failures or not-run models; the PDF path. Claim nothing you haven't verified in the artifacts.
