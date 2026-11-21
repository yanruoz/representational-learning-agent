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
| ROCOv2: frozen BiomedCLIP image and text embeddings (10,539 pairs) | Longleaf GPU job (`longleaf/extract_embeddings.sbatch`), copied back to `cache/rocov2/` | The same script runs on MPS: `run.sh` calls `scripts/extract_embeddings.py --if-missing` before the agent starts, measured at about 8 minutes for all pairs (256 pairs in 11.5 s). Graders without Longleaf get the same cache. The MPS output matches the CUDA smoke test (83% in-batch top-1 on the first 64 test pairs). |
| ROCOv2: light fine-tune (last few transformer blocks of both towers, contrastive loss) | Longleaf GPU job only, saved to `cache/rocov2/` | Too slow on 16 GB MPS. |
| ROCOv2: CLIP-style projection heads, supervised heads, autoencoder on frozen features | Mac, inside `run.sh` | Trains on 512-dim cached vectors, takes seconds. |

**Cache contract.** `scripts/extract_embeddings.py` (and later the fine-tune job) writes `cache/<dataset>/<encoder_id>/<modality>.npy` (float32, not normalized), `ids.csv` (row, pair_id, split, source_file, row_in_file; row i of every .npy is that pair), and `manifest.json` (`complete`, model, device, precision, versions, declared data checksums, counts by split, git commit, SLURM job id). It writes to a temporary folder and renames at the end, so a partial run never looks like a cache. `run.sh` never submits SLURM jobs; it only runs the local extraction when the cache is missing or incomplete. `validate_plan.py` accepts an encoder only if its manifest exists and its checksums match the current data. A plan that wants a missing cache gets `status: not_run_unavailable` with the reason, and the report says so.

**Honesty point for the report.** Fine-tune hyperparameters are fixed before the agent runs (one or two precomputed variants). The agent decides *whether* to use a fine-tuned encoder and compares it against the frozen one; it does not choose fine-tune hyperparameters. Say this in limitations.

**Project 4 contract.** `outputs/<d>/summary.json` holds the chosen model id, paths to `embeddings.npy` (chosen joint embedding, rows aligned with `embedding_index.csv`), per-modality embeddings when they exist, `metrics/*.json`, headline metrics with their baselines, and the report path. All embeddings for every `<model>_s<seed>` stay under `outputs/<d>/embeddings/`.

**Permissions (pipeline).** Copy P2's `.claude/pipeline-settings.json` and add denies: `Read(./docs/**)`, `Read(./longleaf/**)`, `Read(./CLAUDE.md)`, `Edit/Write(./cache/**)`. `bin/runtime_settings.py` adds the absolute-path variants. `tests/test_settings.py` additionally asserts these read denies exist.

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
| `method-selection` | plan.json | The capability table (§4) and plan schema; how to read `baseline_check` output to choose (e.g. one modality already near ceiling → fusion gains are bounded; weak cross-modal agreement → alignment will be hard; small labeled set → supervised heads overfit); encoder choice (raw features vs cached pretrained encoder) and loss; hyperparameter reasons; which comparisons each method needs. No method is the default for any modality. |
| `evaluation` | plan + results | Which metric answers which task; baselines and chance references required for each; seed spread vs gap rule; how labels derived from a modality bias comparisons; embedding collapse signs. |
| `downstream-use` | recommendation.json | What the embeddings can and cannot be used for; the `summary.json` contract for callers; verdict wording. |
| `reporting` | findings.md | Reader-first structure, glossary, every number traced to an artifact, limitations required. |

Critic `rep-critic` (sees objective, plan, metrics, run logs, figures only). Six checks: (1) does a metric test the success criterion, (2) baselines and chance references present and fair (same split, same dimensionality budget), (3) seed stability vs gaps, (4) training adequacy (loss still falling, temperature collapse, effective-rank collapse), (5) leakage: label-derived inputs; labels derived from a modality treated as independent truth; test data used in fine-tuning; and **pretraining overlap**, when the data description says the evaluation data may share a source with a pretrained encoder's training data. In that case the critic flags any conclusion that rests only on that encoder's zero-shot or fine-tuned numbers, and checks that a baseline with no pretraining is reported next to it. (6) Reasons that would read the same on any dataset. The critic learns about overlap only from `DATA_DESCRIPTION.md` and the cache manifest, so the critic prompt stays generic.

## 4. Method capability table (goes into `plan_schema.METHOD_CAPABILITIES`)

| Method | Needs labels | Needs paired samples | Joint embedding | Per-modality embeddings in a shared space | Cross-modal retrieval | Cross-modal prediction | Can use cached pretrained encoder | Runs in `run.sh` |
|---|---|---|---|---|---|---|---|---|
| `concat_baseline` (per-modality PCA/LSI, concatenated) | no | yes | yes | no | no | no | yes | yes |
| `supervised_fusion` (per-modality encoders → fused → classifier; embedding = fused layer) | **yes** | yes | yes | no | no | no | yes | yes |
| `contrastive_alignment` (two encoders, symmetric InfoNCE, CLIP-style) | no | **yes** | yes (mean/concat of the two) | **yes** | **yes** | via nearest neighbor in the other modality | yes | yes |
| `multimodal_autoencoder` (self-supervised; per-modality encoders → shared latent → per-modality decoders, modality dropout) | no | yes for training | yes | partial (each encoder alone maps into the shared latent) | weak | **yes** (encode one, decode the other) | yes | yes |
| `pretrained_zero_shot` (frozen pretrained multimodal encoder) | no | no | yes | yes | yes | no | is one | yes (from cache) |
| `pretrained_finetune` (partial unfreeze, contrastive) | no | yes | yes | yes | yes | no | is one | **no**: from cache, made on Longleaf |

Validator rules: a task needs a method with the capability; `supervised_fusion` needs a label column, and its label metrics must be computed on the held-out split; `pretrained_*` need a cache manifest; every plan includes `concat_baseline` or a single-modality baseline for each task.

## 5. Objectives (one per dataset, chosen so they are plausibly met)

- **pbmc_multiome** (report 1): *"Learn a joint RNA + ATAC cell embedding. Check whether it identifies cell types at least as well as the better single modality, and whether each modality can find the same cell in the other."* Success: on held-out cells, joint kNN macro-F1 ≥ best single-modality baseline minus the seed spread, and ATAC-to-RNA recall@10 well above chance (chance = 10 / number of held-out cells). Plausible because RNA alone is already strong and contrastive alignment of paired multiome cells is known to work at this scale. Risk: the labels came from a joint RNA+ATAC clustering, which favors the joint embedding; the data description says so and the agent should report it.
- **rocov2** (report 2): *"Learn image and caption representations of radiology figures that identify the imaging modality from the image. Check how much the caption adds, and how well captions retrieve their images."* Success: on the official test pairs, imaging-modality macro-F1 from the learned image (or joint) embedding is well above the majority-class baseline and above a baseline with no pretraining (downsampled-pixel PCA for images, TF-IDF + SVD for captions). Frozen BiomedCLIP (its linear probe and its zero-shot retrieval) is reported as a **strong reference, not the bar to beat**: the Longleaf smoke test gave 83% image-to-text top-1 within batches of 64 (chance 1.6%). Some of that is likely pretraining overlap, because ROCOv2 and BiomedCLIP's pretraining data both come from PubMed Central. Retrieval recall@k is reported against chance and against that reference, as a secondary result.

**ROCOv2 modality labels** (derived by the loader from the concept list; see `data/rocov2/DATA_DESCRIPTION.md`). In this copy every row has 1 to 3 concepts. The first is always the manually curated modality concept, and no modality code ever appears in another position. Coverage in the subset (train / test / total):

| Code | Modality | Train | Test | Total |
|---|---|---|---|---|
| C0040405 | CT | 3031 | 622 | 3653 |
| C1306645 | Plain X-ray | 2513 | 428 | 2941 |
| C0024485 | MRI | 1431 | 251 | 1682 |
| C0041618 | Ultrasound | 1202 | 260 | 1462 |
| C0002978 | Angiography | 589 | 72 | 661 |
| C0032743 | PET | 47 | 5 | 52 |
| `nan` (no code) | missing label | 71 | 17 | 88 |
| | **all pairs** | **8884** | **1655** | **10539** |

Rows with no modality code are kept for retrieval and left out of label metrics. PET has only 5 test pairs, so its per-class F1 is unstable; the agent should say so or merge it, and give a reason either way. Captions usually name the modality ("CT of the chest ..."), so caption-only or joint modality accuracy is close to trivial. Image-only results are the informative ones, and the report must say this.

**Limitations to carry into report.pdf and both generated reports:**
- Pretraining overlap: BiomedCLIP was trained on PubMed Central figure-caption pairs, the same source as ROCOv2, so some test pairs may have been seen in pretraining. Frozen and fine-tuned BiomedCLIP results can be inflated by this. The no-pretraining baselines are not affected.
- PBMC labels come from a joint RNA+ATAC clustering, which favors joint embeddings.

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
- Write and submit `finetune_encoder.py` on Longleaf (runs while you are away if needed).
- Exit check: hand plans give metrics and figures for every method on both datasets; numbers make sense against baselines.

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
