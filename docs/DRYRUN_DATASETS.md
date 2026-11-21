# Dry-run datasets (not run)

Dev-only. These two datasets are never downloaded or run. On Monday we use them to check that the skills, the method capability table, and the validators are generic: for each objective, we walk through the decisions the skills would lead to, and check that nothing in them assumes `pbmc_multiome` or `rocov2`.

## STimage-1K4M

Histopathology image patches paired with spatial transcriptomics: each spot has an image tile and a gene-expression profile.

**Objective (to be supplied):**

> _TODO: objective text from the user._

## TCGA-BRCA

Breast cancer histopathology whole-slide images paired with clinical variables per patient.

**Objective (to be supplied):**

> _TODO: objective text from the user._

## What the dry run checks (per dataset)

1. **Loading.** Can `## Loading` declare these modalities with the existing kinds (`counts`, `image`, `text`)? Tabular clinical variables and whole-slide images (many tiles per patient) may need a new kind. Note it as a known limitation rather than building it.
2. **Objective interpretation.** Does the objective map onto the task list (`joint_embedding`, `cross_modal_retrieval`, `cross_modal_prediction`, `clustering`, `label_prediction`)? Which modalities are available at prediction time?
3. **Method selection.** Walk the five steps of `.claude/skills/method-selection/SKILL.md`. Is every rule stated in terms of evidence (profile, baseline check), with no rule that only makes sense for single-cell or radiology data?
4. **Encoders.** Which registry encoders would apply (`extract_embeddings.applicable`)? Is a no-pretraining baseline defined for each modality kind?
5. **Loss.** Do the loss guidance and the redundancy evidence still make sense (for example, neighboring tiles from one slide as near-duplicates)?
6. **Evaluation and critic.** Are the required baselines and the critic checks (leakage, pretraining overlap, collapse) defined for these data?

Record each gap found, and whether it was fixed in a skill or recorded as a limitation.
