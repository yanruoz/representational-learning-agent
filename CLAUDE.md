# CLAUDE.md (dev sessions only)

This file is for development sessions. Pipeline behavior lives in `AGENT.md`, the skills, and the scripts. The pipeline must not read this file.

## Project
Project 3: an agent for multimodal representation learning (BIOS774). Spec: `docs/PROJECT3_REQUIREMENTS.md`. Build plan, milestones, and cut list: `docs/PROJECT3_PLAN.md`. Datasets: `pbmc_multiome` (report 1) and `rocov2` (report 2).

## Schedule
- Due **Mon Nov 30, 2026, 11:59 PM ET**.
- Work sessions: about 2 hours each on Sun Nov 22, Mon Nov 23, and Tue Nov 24. Mostly away Nov 25 to Sun Nov 29.
- By the end of **Tue Nov 24**, `run.sh` must produce `generated_report_1` and `generated_report_2` end to end, even if rough.

## Hard rules
- **The pipeline never reads `docs/**`, `longleaf/**`, `README.md`** (it has the run commands with the objectives), or this file. Enforce this with deny rules in `.claude/pipeline-settings.json` and a check in `tests/test_settings.py`, not only with a prompt instruction. Reason: in Project 2, dataset details leaked into the agent's instructions, so its decisions could look pre-written.
- Skills, `AGENT.md`, and script comments stay generic: no dataset names, genes, counts, or numbers from these datasets. Dataset facts belong only in `data/<d>/DATA_DESCRIPTION.md`.
- Reuse code from `../gen-model-agent` by copying it in. Never import across project folders.
- Run Python through `venv/bin/python` (Mac) or the Longleaf env's python. Never a bare `pip install`.
- `run.sh` never submits SLURM jobs. Longleaf results come in as files under `cache/<dataset>/` with a `manifest.json`. If the frozen-encoder cache is missing, `run.sh` builds it locally with `scripts/extract_embeddings.py --if-missing` (graders have no Longleaf). The agent only reads `cache/`.
- Don't push. The user pushes with `! git push`.
