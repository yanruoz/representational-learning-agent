# Multimodal representation-learning agent

Given a multimodal biomedical dataset and a free-text objective, an AI agent (Claude Code, run headless) inspects the modalities, chooses preprocessing, encoders, a training objective, and evaluation, trains small models on top of fixed features, evaluates the learned representations against baselines, and writes a PDF report. Every number comes from deterministic, seeded scripts; the agent writes decisions and reasons as JSON that validators check.

## Requirements

- macOS or Linux, Python 3.12+ (`python3` on PATH), about 6 GB free disk.
- [Claude Code](https://claude.com/claude-code) installed and logged in (`claude` on PATH).
- Apple silicon (MPS) or an NVIDIA GPU is recommended; CPU works but is slower.

## Run

From the repo root:

```
./run.sh pbmc_multiome "I have cells measured with RNA in one experiment and with ATAC in another. Learn a shared representation where a cell’s RNA and ATAC profiles can be matched, so cell-type annotations can be transferred from RNA-annotated cells to ATAC-only cells. Tell me how well matching works and how accurate the transferred labels are." --report-number 1

./run.sh rocov2 "Our archive has captions for every radiograph but expert labels for only a few. Learn image representations that identify the body region from the image alone when only a few labeled examples per region are available, and tell me whether using the captions during training helps." --report-number 2
```

Each command:
1. creates `venv/` from the pinned `requirements.txt` (first run only),
2. downloads the dataset's files if missing and checks their checksums (about 200 MB for `pbmc_multiome`, 2.4 GB for `rocov2`),
3. for image and text modalities, computes frozen pretrained-encoder embeddings into `cache/<dataset>/` if they are not already there (about 8 minutes per image encoder on an Apple M4),
4. runs the agent, which writes everything to `outputs/<dataset>/` and the report to `reports/<dataset>/generated_report_<N>.pdf`.

Options: `--budget-min M` (training time budget, default 30), `--use-api-key` (use `ANTHROPIC_API_KEY` from `.env` instead of the Claude Code login), `--smoke` in place of the objective (setup and a permission check only).

## Outputs (also the interface for other tools)

`outputs/<dataset>/` holds `profile.json`, `objective.json`, `plan.json`, `metrics/*.json`, `embeddings/<model>_s<seed>.npy`, `embeddings.npy` (the chosen representation, rows aligned with `embedding_index.csv`), `summary.json` (machine-readable: chosen model, paths, headline metrics with baselines), `findings.md`, and figures.

## Optional: GPU cluster

`longleaf/` has SLURM scripts that compute the same embedding cache on a GPU cluster; copy `cache/<dataset>/` back and `run.sh` reuses it (it checks the data checksums in each cache's `manifest.json`). Not needed to run the pipeline.

## Tests

```
venv/bin/python -m unittest discover -s tests -v
```
