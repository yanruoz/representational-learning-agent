# Handoff: Project 2 (generative-modeling agent) → Projects 3 and 4

For a fresh agent building **Project 3** (multimodal representation learning on 10x Multiome PBMC RNA+ATAC and a ROCOv2 subset) and **Project 4** (general biomedical analysis agent that calls Project 3 as a tool). Both are due **Nov 30, 11:59 PM ET**, with about **7 hours** of work time, so reuse beats redesign. Source repo: `gen-model-agent/` (GitHub `yanruoz/gen-model-agent`). Graded 15/15. Copy code in; never import across project folders.

## 1. Harness: how one command runs the agent

`./run.sh <dataset> "<objective>" [--report-number N] [--budget-min M] [--use-api-key]`
1. Re-executes itself under `caffeinate -is` (a Mac sleep kills the headless session), creates `venv/` from pinned `requirements.txt`, fetches the data file if missing (`scripts/fetch_data.py`, URL + MD5 from the dataset's `DATA_DESCRIPTION.md`).
2. Runs `claude -p` (Claude Code headless, the user's login; the API key only with `--use-api-key`) with `AGENT.md` appended to the system prompt, `--setting-sources local`, `--permission-mode dontAsk`, `--strict-mcp-config`, `--agents` from `bin/agents_json.py` (the critic), and `--settings` from `bin/runtime_settings.py`. Stream-JSON output goes through `bin/agent_stream.py` for a readable log.
3. Exits nonzero if the PDF was not written.

**Who decides what.** Claude makes every decision and writes it as JSON with a dataset-specific reason: `objective.json` (task, success criterion, proceed / adapt / decline), `plan.json` (split, preprocessing, models, hyperparameters, seeds), `recommendation.json`, and `findings.md`. Deterministic, seeded scripts compute every number. Validators (`scripts/validate_{objective,plan,recommendation}.py`) check the JSON (schema, capability vs task, budget, "default" is never a reason, cited metric values match). Skills (`.claude/skills/*/SKILL.md`) hold the decision guidance, read just in time. A critic subagent (`.claude/agents/gen-critic.md`) sees only objective, plan, metrics, logs, and figures, never the builder's reasoning, writes `critique.json`, and the agent makes exactly one plan revision. This held up: in two runs of the same PathMNIST command the agent answered the same critique differently (a GAN comparator vs a longer retrain).

**Why.** Every stage reads and writes `outputs/<dataset>/`, so it can be rerun alone, and every decision plus its reason ends up in the report. `CLAUDE.md` is for dev sessions only; pipeline behavior lives in `AGENT.md`, skills, and scripts.

**Permissions.** `.claude/pipeline-settings.json` allows only `venv/bin/python scripts/*`, `ls`, reads, and writes under `outputs/`; denies `.env`, writes to `data/` and `reports/`, `pip install`, `rm`, `git`, and `cd`. `.claude/settings.json` is the dev config (denies `git push`; the user pushes with `! git push`). `tests/test_settings.py` checks that every dev allow rule is denied in the pipeline.

## 2. Reuse map (paths in `gen-model-agent/`)

| Reuse | Files | Change needed |
|---|---|---|
| As-is | `run.sh`, `bin/{agent_stream,agents_json,runtime_settings}.py`, `scripts/fetch_data.py`, `tests/test_settings.py` | Rename the critic and env var names |
| Small change | `scripts/loaders.py` (csv/tsv/npy/npz/h5ad via the `## Loading` spec), `scripts/profiler.py` (modality, counts, sparsity, class balance), `scripts/doctor.py` (subprocess-isolated imports, MPS/CPU check) | Multiome needs two matrices (RNA + ATAC peaks) from one `.h5`/`.h5ad`; ROCOv2 needs image files + caption text. Add a per-modality profile |
| Small change | `scripts/plan_schema.py`, `validate_*.py`, `run_plan.py` (loop, reuse finished runs by config hash, retry once, `run_log.json`, keys `<model>_s<seed>`) | New method list and capability table (contrastive, CLIP-style, self-supervised, supervised, fine-tuning) |
| Small change | `scripts/report.py` (matplotlib `PdfPages`, Contents page, measured-width wrapping, page PNGs in `outputs/<d>/report_pages/`, untraced-number check on `findings.md`), `scripts/visualize.py` | Replace the metric groups and page builders; keep the text/figure page machinery |
| Pattern only | `.claude/skills/*`, `.claude/agents/gen-critic.md`, `AGENT.md` (rules, required sequence, tool table, self-check) | Rewrite content for representation learning |
| Rewrite | `scripts/models/*`, `train.py`, `generate.py`, `evaluate_gen.py`, objective tests | Encoders, contrastive losses, retrieval/kNN/linear-probe evaluation |

**Project 4 calling Project 3.** Treat Project 3 as a command-line tool with JSON in and out (e.g. `run.sh <dataset> "<objective>"` writing `outputs/<d>/embeddings.npy` + `metrics/*.json`), the same way Project 2's scripts are called. Copy it in, don't import across folders.

## 3. Lessons still relevant (from Projects 1 and 2)
- **Every command through `venv/bin/python`.** A bare `pip install` once modified a system conda env.
- **One bounded fix per failure**, then record it and continue; one failed model never stops the report.
- **"Default" or "standard choice" is never a reason**; enforce it in the validator, not the prompt.
- **Enforce boundaries in code.** Held-out data is never used for fitting; only scripts write `reports/`.
- **Key outputs by `<model>_s<seed>`** so seeds coexist, and give every metric a real-vs-real or baseline reference.
- **Read the figures and rendered pages**, not just exit codes.

## 4. What failed, wasted time, or needed rework
- **Dataset details leaked into the agent's instructions** (commit `a77fb2e`). Skills and script comments named PBMC genes, cell counts, and numbers, and `CLAUDE.md` (with the handoff imported) loaded into pipeline runs, so decisions could look pre-written. Fix: scrub dataset specifics from skills, run with `--setting-sources local`, verified by a full run with the handoff file removed. Write skills generically from the start.
- **Fixed rules crowd out judgment.** The model-selection skill named one primary model per modality, so the GAN never ran in rehearsals. Leave real choices open, and give the agent evidence (e.g. a headroom check) to choose with.
- **`cd` broke write permissions.** `./outputs/**` rules resolve against the shell's current directory, which persists between calls. Fix: absolute path rules (`bin/runtime_settings.py`), deny `cd`, and an `AGENT.md` rule to use repo-relative paths.
- **Report readability took three rounds**: undefined abbreviations, repeated tables, field names in prose, lines cut off the page (character-count wrapping). Design for a reader from the start: a glossary page, plain labels, a Contents page.
- **Both main objectives failed** (PBMC synthetic cells: classifier two-sample AUC 0.84–0.87 vs 0.58 between real samples; PathMNIST train-on-synthetic utility 0.72). It held under 9 PBMC model variants and 5 sampling settings. The agent reported it honestly; choose objectives the methods can plausibly meet.
- **Planned, never built:** an uncertainty-quantification test (always declined), categorical tabular features.

## 5. Environment and setup quirks
- Apple M4, 16 GB, no CUDA: PyTorch on **MPS**. Keep models small. PathMNIST: 20k-image subsample, diffusion took about 20 min; PBMC about 10 min total.
- **Lid open, power plugged in** (`caffeinate -s` only holds on AC power).
- The auto-mode permission classifier sometimes returns no verdict: retry once. `rm` with a variable path is blocked; move files to the scratchpad instead.
- `scverse` returns HTTP 403 without a User-Agent header (handled in `fetch_data.py`).
- **The fresh-clone check** (the user cloned from GitHub and followed the README) caught stale generated reports still tracked in git, and confirmed `run.sh` fetches data into an empty `data/`. Do this check yourself before the user's final run.

## 6. How the reports were made
- **Generated reports:** `scripts/report.py --dataset d --report-number N` builds `reports/<d>/generated_report_<N>.pdf` from the JSON files. The agent writes only `findings.md`; every decimal in it must match an artifact or the build fails. It can be rerun from saved outputs without rerunning the agent.
- **`report.pdf`:** hand-edited LaTeX at `docs/report.tex`, built with `tectonic docs/report.tex` (4 pages). Results were first drafted from rehearsal runs, then replaced with the clean-run numbers.
- **Submission:** `git archive --format=zip -o ../<repo>-source.zip HEAD` (excludes `.env`, `venv/`, data, outputs, checkpoints), plus the two generated reports and `report.pdf`.
