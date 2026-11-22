# Dry-run datasets (not run)

Dev-only. These two datasets are never downloaded or run. On Monday we use them to check that the skills, the method capability table, and the validators are generic: for each objective, we walk through the decisions the skills would lead to, and check that nothing in them assumes `pbmc_multiome` or `rocov2`.

## STimage-1K4M

Histopathology image patches paired with spatial transcriptomics: each spot has an image tile and a gene-expression profile.

**Objective:**

> From a few slides of one tissue, learn a joint representation of H&E patches and spot gene expression so marker-gene expression can be predicted from histology on an unseen slide.

**Notes:**
- Split by slide, not by spot: neighboring spots on one slide are near-duplicates, so a spot-level split leaks.
- The strong pathology encoders (UNI, CONCH) are gated (access must be requested), so the registry would offer only openly available image encoders.
- Expected shape: inference modality image, training-only modality expression, task cross-modal prediction (per-gene Pearson on held-out slides), with a group-aware split.

## TCGA-BRCA

Breast cancer histopathology whole-slide images paired with clinical variables per patient.

**Objective:**

> Using images and clinical variables, learn a patient representation that predicts molecular subtype better than clinical variables alone.

**Notes:**
- About 1,000 patients have slide images on GDC. The TCIA collection hosts MRI for only about 140 patients.
- Small n. Whole-slide images need tile features and multiple-instance learning (many tiles per patient, one label).
- Split by patient.
- Expected shape: inference modalities image and clinical (both available at prediction time), task label prediction, baseline clinical variables alone.

## What the dry run checks (per dataset)

1. **Loading.** Can `## Loading` declare these modalities with the existing kinds (`counts`, `image`, `text`)? Tabular clinical variables and whole-slide images (many tiles per patient) may need a new kind. Note it as a known limitation rather than building it.
2. **Objective interpretation.** Does the objective map onto the task list (`joint_embedding`, `cross_modal_retrieval`, `cross_modal_prediction`, `clustering`, `label_prediction`)? Which modalities are available at prediction time?
3. **Method selection.** Walk the five steps of `.claude/skills/method-selection/SKILL.md`. Is every rule stated in terms of evidence (profile, baseline check), with no rule that only makes sense for single-cell or radiology data?
4. **Encoders.** Which registry encoders would apply (`extract_embeddings.applicable`)? Is a no-pretraining baseline defined for each modality kind?
5. **Loss.** Do the loss guidance and the redundancy evidence still make sense (for example, neighboring tiles from one slide as near-duplicates)?
6. **Evaluation and critic.** Are the required baselines and the critic checks (leakage, pretraining overlap, collapse) defined for these data?

Record each gap found, and whether it was fixed in a skill or recorded as a limitation.

## Results (Nov 22)

| Check | STimage-1K4M | TCGA-BRCA |
|---|---|---|
| 1. Loading | **Gap.** Expression (counts) and image patches would come from different file types (h5ad + images), and the loader refuses mixed types; image folders (not parquet) are unsupported. Limitation. | **Gap.** No `tabular` kind for clinical variables, and no multi-instance kind (many tiles per patient). Limitation; would need tile features + MIL pooling. |
| 2. Objective | Maps to `cross_modal_prediction` (image → expression), inference modality image, expression training-only. Supported (kNN averaging, PLS). | Maps to `label_prediction` with inference modalities image + clinical; baseline clinical alone = `single_modality_features`. Supported once the kinds exist. |
| Split | **Gap fixed:** split by slide needs a group-aware split. Added `split.group_by` (a declared label used as group id; `GroupShuffleSplit`), with a test. The data-profiling skill must tell the agent to use it whenever samples share a source (slide, patient, donor). | Same: `group_by` patient. |
| 3. Method selection | Steps 1-5 apply as written. No rule assumes single-cell or radiology. | Step 1 (both modalities at inference) points to fusion; the skill should warn that small n (~1,000) makes supervised heads overfit. **To add** to the skill (generic: compare n with model size). |
| 4. Encoders | DINOv2 and BiomedCLIP apply to image patches; pixel PCA as no-pretraining baseline. UNI/CONCH are gated, so not in the registry; noted. | Same image encoders on tiles; clinical needs a tabular featurizer (standardize + one-hot). Gap with the kind. |
| 5. Loss | Redundancy evidence applies: neighboring spots are near-duplicates, which the profile would show, so `vicreg` gets a real case. The group split matters more than the loss. | Not a two-tower problem (both modalities at inference); `supervised_fusion` with modality dropout fits. |
| 6. Critic | Leakage check must include "split ignores declared groups". **To add** to the critic. | Same. |

Fixed now: group-aware split. To add with Tuesday's skills and critic: group-split rule (data-profiling skill), small-n overfitting guidance (method-selection), group leakage check (critic). Recorded as limitations: mixed file types, tabular kind, multi-instance kind.
