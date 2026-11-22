# Project 3 build plan

Dev-only document. The pipeline must never read `docs/**` or `longleaf/**`; this is enforced by deny rules in `.claude/pipeline-settings.json` (see §1) and checked by `tests/test_settings.py`.

Source for reuse: `../gen-model-agent` (Project 2). Code is copied in, never imported across folders. Requirements: `docs/PROJECT3_REQUIREMENTS.md`.

## 1. Architecture and where the GPU fits

```
./run.sh <dataset> "<objective>" --report-number N
  └─ claude -p  (AGENT.md + skills + validators + rep-critic subagent)
       ├─ profiler.py           per-modality profile (counts / images / text), pairing, label and leakage columns
       ├─ baseline_check.py     evidence before choosing a method: single-modality and concatenation
       │                         baselines (kNN, clustering), cross-modal agreement (CCA), cached-encoder zero-shot
       ├─ objective.json → plan.json (validated)
       ├─ run_plan.py           preprocess → encode inputs (raw features or cached encoder features)
       │                         → train small heads per <model>_s<seed> → evaluate_rep.py → figures
       ├─ critic → one plan revision
       ├─ recommendation.json → findings.md → report.py → reports/<d>/generated_report_<N>.pdf
       └─ summary.json          machine-readable contract for Project 4
```

**Compute split.**

| Step | Where | Why |
|---|---|---|
| PBMC: everything (HVG/LSI input, MLP encoders, all methods) | Mac, inside `run.sh` | Inputs are about 9.6k cells x a few thousand features. Small MLPs train in minutes on MPS. No pretrained encoder needed. |
| ROCOv2: frozen BiomedCLIP image and text embeddings (10,539 pairs) | Longleaf GPU job (`longleaf/extract_embeddings.sbatch`), copied back to `cache/rocov2/` | The same script runs on MPS: `run.sh` calls `scripts/extract_embeddings.py --if-missing` before the agent starts, measured at 466 s for all 10,539 pairs (DINOv2: 336 s). Graders without Longleaf get the same cache. The MPS output matches the CUDA smoke test (83% in-batch top-1 on the first 64 test pairs). |
| ROCOv2: frozen DINOv2 ViT-B/14 image embeddings (second image encoder candidate) | Same script and sbatch job (`ENCODER=dinov2_vitb14_frozen`), or MPS in `run.sh` (about 5 min) | DINOv2 never saw captions, so it gives the clean "do captions help" comparison; BiomedCLIP's image tower was trained with captions. |
| ROCOv2: light fine-tune (last few transformer blocks of both towers, contrastive loss) | Longleaf GPU job only, saved to `cache/rocov2/` | Too slow on 16 GB MPS. |
| ROCOv2: CLIP-style projection heads, supervised heads, autoencoder on frozen features | Mac, inside `run.sh` | Trains on 512-dim cached vectors, takes seconds. |

**Cache contract.** `scripts/extract_embeddings.py` (and later the fine-tune job) writes `cache/<dataset>/<encoder_id>/<modality>.npy` (float32, not normalized), `ids.csv` (row, pair_id, split, source_file, row_in_file; row i of every .npy is that pair), and `manifest.json` (`complete`, model, device, precision, versions, declared data checksums, counts by split, git commit, SLURM job id). It writes to a temporary folder and renames at the end, so a partial run never looks like a cache. `run.sh` never submits SLURM jobs; it only runs the local extraction when the cache is missing or incomplete. `validate_plan.py` accepts an encoder only if its manifest exists and its checksums match the current data. A plan that wants a missing cache gets `status: not_run_unavailable` with the reason, and the report says so.

**Honesty point for the report.** Fine-tune hyperparameters are fixed before the agent runs (one or two precomputed variants). The agent decides *whether* to use a fine-tuned encoder and compares it against the frozen one; it does not choose fine-tune hyperparameters. Say this in limitations.

**Project 4 contract.** `outputs/<d>/summary.json` holds the chosen model id, paths to `embeddings.npy` (chosen joint embedding, rows aligned with `embedding_index.csv`), per-modality embeddings when they exist, `metrics/*.json`, headline metrics with their baselines, and the report path. All embeddings for every `<model>_s<seed>` stay under `outputs/<d>/embeddings/`.

**Permissions (pipeline).** Copied from P2, plus denies: `Read(./docs/**)`, `Read(./longleaf/**)`, `Read(./CLAUDE.md)`, `Read(./README.md)` (it has the run commands with the objectives), `ls` of `docs`/`longleaf`, `Edit/Write(./cache/**)`, and running `fetch_data.py` / `extract_embeddings.py`. `bin/runtime_settings.py` adds the absolute-path variants. `tests/test_settings.py` additionally asserts these read denies exist.

## 2. Reuse / adapt / rewrite by file

| File(s) in this repo | From P2 | Action |
|---|---|---|
| `run.sh` | `run.sh` | **Reuse**: rename env vars (`GENAGENT_*` → `REPAGENT_*`), critic name, fetch loop for multi-file datasets. |
| `bin/agent_stream.py`, `bin/agents_json.py`, `bin/runtime_settings.py` | same | **Reuse**: critic name only. |
| `.claude/pipeline-settings.json`, `.claude/settings.json`, `tests/test_settings.py` | same | **Reuse + extend**: deny `docs/**`, `longleaf/**`, `CLAUDE.md`, writes to `cache/**`; test for these. |
| `scripts/fetch_data.py` | same | **Done tonight (adapted)**: multi-file `## Files` section, sha256 or md5, `--data-root`, stdlib only. |
| `scripts/doctor.py` | same | **Small change**: add `pyarrow`, `pillow`; optional check that a cache manifest matches the data. |
| `scripts/loaders.py` | same | **Adapt**: return a dict of modalities `{name: {"X", "kind", "ids"}}` + labels + metadata. Kinds: `counts` (h5ad), `image` (parquet bytes), `text` (parquet string). Declared in `## Loading` (`modalities:` mapping file/column → kind, `label_column`, `leakage_columns`, `split_column`). |
| `scripts/profiler.py` | same | **Adapt**: per-modality profile (counts: sparsity, library size, near-binary check; image: size/mode distribution; text: length distribution, vocabulary), pairing check across modalities, label balance, flags declared leakage columns. |
| `scripts/plan_schema.py` | same | **Rewrite contents, keep pattern**: tasks, methods, capability table (§4), preprocessing steps per kind, metric groups. |
| `scripts/validate_objective.py`, `validate_plan.py`, `validate_recommendation.py` | same | **Adapt**: keep "default is never a reason", budget, cited-metric match; new checks: method capability vs task, labels required, baseline present for every task, cache manifest present, no leakage column as input. |
| `scripts/run_plan.py`, `scripts/model_io.py` | same | **Small change**: same loop, config-hash reuse, retry once, `run_log.json`, `<model>_s<seed>` keys; call new train/evaluate. |
| `scripts/preprocess.py` | same | **Adapt**: counts: normalize, log1p, HVG, PCA; peaks: binarize, TF-IDF, LSI; image/text: none (cached features) or tokenization note. Fit on train split only. |
| `scripts/headroom_check.py` | | **Replace** with `scripts/baseline_check.py` (the evidence step, §5 of the critic lessons: give the agent evidence, not rules). |
| `scripts/models/*`, `train.py` | | **Rewrite**: `encoders.py` (MLP for vectors, linear/MLP head for cached features), `losses.py` (InfoNCE with learnable temperature, reconstruction (MSE / Bernoulli / NB), cross-entropy), `fusion.py` (concat, mean, gated). `train.py` dispatches on method. |
| `generate.py`, `evaluate_gen.py`, `anomaly_test.py`, `imputation_test.py`, `image_eval.py` | | **Drop**. |
| `scripts/evaluate_rep.py`, `scripts/downstream_rep.py` | `downstream_test.py` (pattern) | **New**: kNN and linear probe (macro-F1, accuracy), KMeans ARI/NMI, retrieval recall@1/5/10 + median rank + chance, cross-modal prediction (Pearson per feature vs kNN-regression baseline), embedding collapse check (effective rank). Every metric written next to its baseline and chance value. |
| `scripts/extract_embeddings.py` | | **New, runs on cuda or mps**: frozen or fine-tuned encoder → cache with manifest. Called by `longleaf/*.sbatch`, never by the agent. |
| `scripts/finetune_encoder.py` | | **New, Longleaf only**: partial unfreeze + contrastive loss on the training split only; writes embeddings for all splits to the cache. |
| `scripts/visualize.py` | same | **Adapt**: UMAP-free 2D (PCA / t-SNE from sklearn) colored by label, retrieval rank histograms, metric-vs-baseline bars, training curves. |
| `scripts/report.py` | same | **Adapt**: keep PdfPages machinery, Contents, measured-width wrapping, page PNGs, untraced-number check; replace metric groups and page builders. Glossary from the start. |
| `AGENT.md` | same | **Rewrite content, keep shape**: rules, required sequence, tool table, self-check. |
| `.claude/skills/*`, `.claude/agents/rep-critic.md` | same | **Rewrite** (§3). |
| `requirements.txt` | same | **Reuse + add** `pyarrow`, `pillow`. |
| `longleaf/*` | | **New tonight**: setup script, test job, smoke test, GPU requirements. Later: `extract_embeddings.sbatch`, `finetune.sbatch`, `sync_cache.sh` (rsync `cache/` back to the Mac). |

## 3. Skills (generic; no dataset names, genes, counts, or numbers from these datasets)

| Skill | Used at step | Contents |
|---|---|---|
| `data-profiling` | profile → plan | How to read the per-modality profile; preprocessing choices by modality kind (count matrices, sparse near-binary counts, images, free text) and when each is wrong; pairing and split rules (split by sample, held-out never used for fitting, preprocessing fit on train only); how to spot label-leakage columns and derived labels. |
| `objective-interpretation` | objective.json | Map free text to tasks (`joint_embedding`, `cross_modal_retrieval`, `cross_modal_prediction`, `clustering`, `label_prediction`); proceed / adapt / decline; how to write a success criterion that names a metric, a baseline, and a margin the methods can plausibly meet; what to do when labels are missing or not independent of a modality. |
| `method-selection` | plan.json | **Written** (`.claude/skills/method-selection/SKILL.md`): inference modalities first; method capabilities; encoder per modality from the baseline check's candidates (pretraining overlap, confounded comparisons, a no-pretraining baseline always kept); two-tower loss `infonce` / `vicreg` / `supcon` from redundancy and label evidence; required baselines. Every choice cites evidence. |
| `evaluation` | plan + results | Which metric answers which task; baselines and chance references required for each; seed spread vs gap rule; how labels derived from a modality bias comparisons; embedding collapse signs. |
| `downstream-use` | recommendation.json | What the embeddings can and cannot be used for; the `summary.json` contract for callers; verdict wording. |
| `reporting` | findings.md | Reader-first structure, glossary, every number traced to an artifact, limitations required. |

Critic `rep-critic` (sees objective, plan, metrics, run logs, figures only). Six checks: (1) does a metric test the success criterion, (2) baselines and chance references present and fair (same split, same dimensionality budget), (3) seed stability vs gaps, (4) training adequacy and **collapse**: loss still falling at the last epoch, temperature at its bound, and, from `metrics/collapse.json`, per-dimension variance (fraction of embedding dimensions with standard deviation below 1% of the largest) and effective rank (exp of the entropy of the normalized singular values) of every learned embedding, compared with its input features. A learned embedding whose effective rank is far below its input's, or with many near-constant dimensions, is flagged even if its probe scores look fine, (5) leakage: label-derived inputs; labels derived from a modality treated as independent truth; test data used in fine-tuning; and **pretraining overlap**, when the data description says the evaluation data may share a source with a pretrained encoder's training data. In that case the critic flags any conclusion that rests only on that encoder's zero-shot or fine-tuned numbers, and checks that a baseline with no pretraining is reported next to it. (6) Reasons that would read the same on any dataset. The critic learns about overlap only from `DATA_DESCRIPTION.md` and the cache manifest, so the critic prompt stays generic.

## 4. Methods, encoders, losses, and the plan schema

The agent makes three real choices per model, each with a `reason` and an `evidence` list (file, key path, value; `validate_plan.py` checks that each value matches the file). The generated report's methods page prints, for every model and every modality, the encoder, head, and loss with their reasons.

**Methods** (`plan_schema.METHOD_CAPABILITIES`):

| Method | Needs labels | Embeds one modality alone | Cross-modal prediction | Losses |
|---|---|---|---|---|
| `single_modality_features` (baseline) | no | yes | kNN averaging in its own space | none |
| `linear_supervised` (baseline) | labels or a target modality | yes | **PLS regression** | logistic regression (labels), PLS (targets) |
| `linear_alignment` (baseline) | no | yes (one projection per modality) | kNN averaging in the shared space | CCA; number of components chosen on validation |
| `frozen_probe` | probe only | yes | no | none (cached encoder + kNN / linear probe) |
| `two_tower` | only with `supcon` | yes (one tower) | kNN averaging in the shared space | `infonce` \| `vicreg` \| `supcon` |
| `multimodal_autoencoder` | no | with modality dropout | yes (decode the other modality) | reconstruction per modality (MSE, Bernoulli, or negative binomial) |
| `supervised_fusion` | yes | with modality dropout only | no | cross-entropy |
| `pretrained_finetune` | no | yes | no | `infonce`; from cache only, made on Longleaf (first cut) |

**Encoders** (candidates per modality kind; all fit on training samples only):

| Modality kind | No pretraining (always a candidate and a baseline) | Cached frozen encoders |
|---|---|---|
| counts | PCA of log-normalized counts; LSI (TF-IDF + truncated SVD) for sparse low-count features | none |
| image | downsampled grayscale pixels + PCA | `biomedclip_frozen` (image tower), `dinov2_vitb14_frozen` |
| text | TF-IDF + truncated SVD | `biomedclip_frozen` (text tower) |

Heads: `none`, `linear`, or a small `mlp` (hidden sizes in the hyperparameters).

**Two-tower losses.**
- `infonce`: other samples in the batch are negatives. False negatives when many samples are near-duplicates.
- `vicreg`: no negatives; variance and covariance terms prevent collapse.
- `supcon`: same-label positives; needs training labels, and only from the labeled fraction the evaluation allows.

Paired modalities are the two views, so no augmentations. The guidance is in the method-selection skill; the evidence is `profile.json → redundancy` and class counts.

**Plan schema (one model entry):**

```json
{
  "id": "tt_vicreg",
  "method": "two_tower",
  "inputs": {
    "<modality>": {"encoder": "lsi", "head": "mlp", "reason": "...",
                   "evidence": [{"file": "metrics/baseline_check.json", "key": "encoders.<modality>.lsi.val_macro_f1", "value": 0.81}]}
  },
  "loss": "vicreg",
  "loss_reason": "...",
  "loss_evidence": [{"file": "profile.json", "key": "redundancy.<modality>.mean_fraction_of_knn_that_are_near_duplicates", "value": 0.0}],
  "inference_modalities": ["<modality>"],
  "trained_without": null,
  "hyperparameters": {"embed_dim": 64, "hidden_dims": [256], "epochs": 100, "batch_size": 256, "lr": 0.001},
  "hyperparameter_reasons": {"embed_dim": "...", "...": "one per key"},
  "seeds": [0, 1, 2],
  "serves_tasks": ["label_prediction", "cross_modal_prediction"],
  "estimated_minutes": 1.0,
  "status": "planned",
  "reason": "..."
}
```

`trained_without` names the modality left out of a matched "without" model, for objectives that ask whether a training-only modality helps.

**Baseline check** (`scripts/baseline_check.py`, run before the agent picks models). On a validation split carved from the training samples, never the test split:
- **Every candidate encoder, frozen, per modality:** kNN and linear-probe macro-F1 for each label, at the label fractions the objective needs (seeded draws).
- **Cross-modal agreement:** CCA between the inference modality's candidate features and the other modality's.
- **Frozen cross-modal retrieval:** for jointly pretrained encoders (zero-shot).
- **Seed spread:** of each number.
- **Redundancy:** copied from the profile, so the evidence is in one file.

**Evaluation** (`scripts/evaluate_rep.py`, held-out samples, inference modalities only):
- **Labels:** kNN and linear probe (macro-F1, per class) at each label fraction.
- **Clustering:** KMeans ARI / NMI.
- **Retrieval:** recall@1/5/10 with chance.
- **Cross-modal prediction:** per-feature Pearson (median and distribution).
- **Collapse:** `metrics/collapse.json` gives per-dimension variance and effective rank for every embedding.

Every metric is written next to its baselines (single-modality features, linear supervised) and the seed spread.

**Validator rules.**
- A task needs a method with the capability, and a method that embeds the inference modalities alone.
- `supcon` and `supervised_fusion` need labels, limited to the allowed labeled fraction.
- Cached encoders need a complete manifest matching the data checksums.
- Each task has both required baselines.
- Every encoder, head, loss, and hyperparameter has a reason, and every evidence value matches its file.
- Objective-specific expectations (label budgets fillable per class, inference vs training-only modalities, per-budget label use, training pool, class exclusions, secondary labels): see `docs/OBJECTIVES.md` → Validator expectations.

## 5. Objectives

Full text, intended evaluation, and success criteria are in `docs/OBJECTIVES.md` (the pipeline cannot read it). In short:

- **pbmc_multiome** (report 1, changed Nov 22): **cross-modal matching and label transfer.** RNA-only and ATAC-only cells from separate experiments. Learn a shared space where a cell's RNA and ATAC profiles match, and transfer cell-type labels from RNA to ATAC.
  - Primary (label-free): FOSCTTM and recall@k on held-out cells vs a CCA baseline (components chosen on validation) and chance.
  - Secondary: RNA-trained classifier applied to ATAC embeddings vs CCA transfer, with ATAC-alone kNN as an upper reference.
  - Expression estimation is dropped; its noise ceiling (0.286) goes in the report.
  - Feasibility check: InfoNCE FOSCTTM 0.015 vs CCA 0.020 vs chance 0.5; recall@10 0.47 vs 0.38; transfer macro-F1 0.88-0.92 vs CCA 0.88 and reference 0.916. Details in `docs/OBJECTIVES.md`.

- **rocov2** (report 2): **body region** from the image alone with **5, 20, 100, or all labeled images per class**; do captions used in training help?
  - Image-only test-set macro-F1 at each budget.
  - Matched pair: frozen DINOv2 vs a DINOv2 image tower aligned to captions. DINOv2 never saw captions, so this comparison is clean.
  - BiomedCLIP as a reference: its image tower was trained with captions, and PubMed Central is the source of both, so pretraining overlap is a risk.
  - Baselines: majority class, pixel PCA, and logistic regression on frozen features.
  - The agent chooses the caption training pool (all 8,884 pairs or the 2,513 X-ray pairs) with a reason. Breast (9 train / 4 test) is the agent's call.
  - Imaging modality is a secondary result (frozen probe only).

  Profile evidence: near-duplicates are rare for captions and BiomedCLIP images (0.8% or less of the 10 nearest neighbors). In DINOv2 space, though, 18.8% are near-duplicates, against 0.5% of random pairs above the cutoff. So a DINOv2-based two-tower model faces InfoNCE false negatives, which makes `vicreg` a real option.

**Why body region, not modality (changed Nov 22).** On 252 test images, 1-NN leave-one-out modality accuracy from frozen features is already 0.956 (BiomedCLIP) and 0.948 (DINOv2), against 0.409 for the majority class. That left no room to show a caption effect on modality, so modality became a secondary finding.

**Limitations to carry into report.pdf and both generated reports:**
- Pretraining overlap: BiomedCLIP was trained on PubMed Central figure-caption pairs, the same source as ROCOv2. Frozen and fine-tuned BiomedCLIP results can be inflated. DINOv2 and the no-pretraining baselines don't share this risk to the same degree.
- PBMC labels come from a joint RNA+ATAC clustering, which favors representations that use RNA. That is the comparison the objective asks about, so the report must state it.

## 6. Milestones

Time is your supervision time; I write the code during those sessions.

**Fri Nov 20, tonight (1 h).** Done: requirements doc, this plan, `scripts/fetch_data.py`, both `DATA_DESCRIPTION.md`, `longleaf/` setup. You: commit, push, start the Mac data download, start the Longleaf setup (see §7).

**Sun Nov 22 (2 h). Harness + data in.**
- Check the Longleaf test job output (GPU name, matched vs mismatched cosine).
- Copy and rename the harness: `run.sh`, `bin/*`, settings with the new denies, `tests/test_settings.py`, `doctor.py`, `requirements.txt`, `CLAUDE.md` (dev).
- `loaders.py` + `profiler.py` for multi-modality; tests on both datasets.
- `scripts/extract_embeddings.py` + `longleaf/extract_embeddings.sbatch`; submit the frozen BiomedCLIP job. `longleaf/sync_cache.sh`.
- Exit check: `./run.sh <d> --smoke` passes for both; profiles look right; frozen cache on the Mac.

**Mon Nov 23 (2 h). Methods + evaluation without the agent.**
- `plan_schema.py` (§4), `preprocess.py`, `models/` (encoders, losses, fusion), `train.py`, `evaluate_rep.py`, `baseline_check.py`, `run_plan.py`, `visualize.py`.
- Hand-written plans run end to end on both datasets; read the figures.
- Dry run of the skills on `docs/DRYRUN_DATASETS.md` (STimage-1K4M, TCGA-BRCA): walk each objective through the skills and record what assumes our two datasets.
- Write and submit `finetune_encoder.py` on Longleaf (runs while you are away if needed).
- Exit check: hand plans give metrics and figures for every method on both datasets; numbers make sense against baselines.

**Monday milestone: done Nov 22.**
- Built: `context.py` (split incl. `group_by`, label draws), `features.py`, `models/nets.py`, `train.py`, `baselines.py` (incl. CCA), `evaluate_rep.py`, `baseline_check.py`, `run_plan.py`.
- Hand plans (`docs/handplans/`) run end to end: PBMC 14 runs in 3.0 min, ROCOv2 24 runs in 2.2 min, none failed.
- PBMC: InfoNCE FOSCTTM 0.0156 ± 0.0006 vs CCA 0.0200; transfer kNN macro-F1 0.902 vs CCA 0.881.
- ROCOv2: captions help body-region kNN at 5 per class (0.604 vs frozen DINOv2 0.487) but not the linear probe. `concat_input` added as a plan option.
- Dry run results are in `docs/DRYRUN_DATASETS.md`.
- Figures (`visualize.py`) move to Tuesday with the report.

**Tue Nov 24 (2 h). Agent + reports.**
- `AGENT.md`, six skills, `rep-critic.md`, three validators, `report.py` page builders, `summary.json`.
- `./run.sh pbmc_multiome "<objective>" --report-number 1` and `./run.sh rocov2 "<objective>" --report-number 2`.
- Exit check (hard deadline): both generated reports exist, end to end, even rough. Commit and push.

**Sun Nov 29 – Mon Nov 30.** Sync the fine-tune cache if it finished; read both reports and fix the worst problems; clean runs; fresh-clone check; `docs/report.tex` (4 pages); submission zip.

## 7. Cut list (cut from the top when behind)

1. Fine-tuning (keep frozen BiomedCLIP + trained heads; mention as not run in the report).
2. Longleaf entirely for ROCOv2 (fall back to `extract_embeddings.py` on MPS).
3. The plan revision after the critic (critic findings become limitations only).
4. Multiple seeds for slow methods (keep 3 seeds only for the small heads).
5. Cross-modal prediction on ROCOv2 (retrieval is the downstream task there).
6. `multimodal_autoencoder` on ROCOv2 (keep it for PBMC).
7. Report polish beyond the P2 machinery.

Never cut: baselines for every metric, train-only fitting, the held-out split, validators rejecting "default" reasons, the `docs/**` read deny, `summary.json` for Project 4.

## 8. Tonight's commands

**Mac** (stdlib only, no venv needed; about 2.6 GB, ROCOv2 dominates):
```
python3 scripts/fetch_data.py --dataset pbmc_multiome
python3 scripts/fetch_data.py --dataset rocov2
```

**Longleaf** (after `git push`):
```
ssh <onyen>@longleaf.unc.edu
git clone https://github.com/yanruoz/representational-learning-agent.git ~/representational-learning-agent
cd ~/representational-learning-agent
nohup bash longleaf/setup_longleaf.sh > ~/repagent_setup.log 2>&1 &
tail -f ~/repagent_setup.log       # Ctrl-C to stop watching; the setup keeps running
```
Sunday: `cat ~/repagent_setup.log`, then the test job's log and JSON (paths printed at the end of the setup log). If the default partition is wrong, rerun with `PARTITION=<name>`; every completed step is skipped.
