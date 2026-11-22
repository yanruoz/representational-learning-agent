# Objectives for the two generated reports

Dev-only. The pipeline cannot read this file (`docs/**` is denied) or `README.md`, which repeats the run commands. The agent sees an objective only as the text `run.sh` passes it. Everything below on how to evaluate and what counts as success is for us to check the agent's choices against. The agent has to reach those choices itself from the skills, the data description, the profile, and the baseline check.

## Report 1: `pbmc_multiome`

**Objective (passed verbatim):**

> Future samples will be profiled with scATAC-seq only. Using this paired RNA+ATAC data, learn a representation that lets ATAC-only cells be annotated by cell type and have their gene expression estimated, and show how much better that is than working from ATAC alone.

**Intended evaluation** (ATAC is the only input for held-out cells; RNA and labels are used in training only):
- Cell-type annotation: kNN macro-F1 on held-out cells, from the learned ATAC embedding, compared with kNN on an ATAC-LSI baseline (TF-IDF + truncated SVD fit on training cells) and a linear supervised baseline (logistic regression on ATAC-LSI).
- Expression estimation: per-gene Pearson correlation on highly variable genes (chosen on training cells) between estimated and measured expression in held-out cells. Compared with (a) averaging the expression of the k nearest training cells in ATAC-LSI space and (b) PLS regression from ATAC-LSI to expression.
- Report the median and the distribution over genes, and the spread over seeds.

**What should count as success:** the learned representation beats the ATAC-LSI kNN baseline on macro-F1, and beats both expression baselines on median per-gene Pearson, by more than the seed spread. This is plausible because RNA supervises which ATAC differences matter. Risk: ATAC-LSI already separates the major lineages, so gains may show up mainly in the finer T-cell and B-cell subtypes. The per-class F1 should show where the gain comes from. The labels came from a joint RNA+ATAC clustering (see the data description), so the comparison with ATAC-only baselines is somewhat in the learned model's favor.

## Report 2: `rocov2`

**Objective (passed verbatim):**

> Our archive has captions for every radiograph but expert labels for only a few. Learn image representations that identify the body region from the image alone when only a few labeled examples per region are available, and tell me whether using the captions during training helps.

(Changed on Nov 22 from imaging modality to body region: frozen image features already identify modality almost perfectly, see below.)

**Concept counts beyond modality** (this subset; train / test / total). All are on plain X-ray images. The copy keeps at most 3 concepts per image (modality, body region, view direction), so there are no finding concepts.

| Code | Kind | Meaning | Train | Test | Total |
|---|---|---|---|---|---|
| C0817096 | body region | chest | 876 | 155 | 1031 |
| C0037303 | body region | cranium (mostly jaw and dental panoramic) | 468 | 53 | 521 |
| C0023216 | body region | lower limb | 302 | 71 | 373 |
| C0000726 | body region | abdomen | 277 | 43 | 320 |
| C0030797 | body region | pelvis | 194 | 32 | 226 |
| C0037949 | body region | spine | 172 | 19 | 191 |
| C1140618 | body region | upper limb | 167 | 26 | 193 |
| C0006141 | body region | breast | 9 | 4 | 13 |
| C1999039 | view | anteroposterior | 1080 | 213 | 1293 |
| C1996865 | view | posteroanterior | 544 | 72 | 616 |
| C0205129 | view | lateral / sagittal | 276 | 44 | 320 |
| C0205106 | view | (another view; captions mention axillary views) | 4 | 2 | 6 |

The names come from UMLS and were checked against the most common words in the matching captions. C0205106's exact meaning is unconfirmed.

**Label budgets (per class, not percentages): 5, 20, 100, and all.** Checked against the counts above:
- Seven regions can fill every budget. With 15% of training held out for validation (for model selection and the baseline check), the smallest training class has 141 images (upper limb), so 100 per class fits.
- Test classes have 19 (spine) to 155 (chest) images. Per-class F1 for spine and upper limb rests on 19 and 26 images.
- **Breast** has 9 training and 4 test images, so it can fill only the 5 budget. It stays declared, and the profile flags it as a small class. Dropping or keeping it is the agent's decision, with a reason.

**Training pool for captions.** Captions are used in training only; test inputs are images alone. Body region exists only on X-ray images (2,513 training pairs), but captions exist for all 8,884 training pairs. Training the caption-aligned image model on all pairs or on X-ray only is the agent's choice, with a reason (more pairs vs pairs closer to the target domain).

**Intended evaluation** (image only, on the official test split):
- Body-region macro-F1 and per-class F1 from a kNN classifier and a linear probe on the image embedding, fit on 5, 20, or 100 labeled training images per class (seeded, stratified draws; several draws for the small budgets), or on all labeled training images.
- **"Do captions help":** the same image encoder with and without caption-based training at each budget. The clean pair is DINOv2 (never trained on text): frozen features vs a DINOv2 image tower aligned to captions. BiomedCLIP's image tower was trained with captions, so it is a reference, not a clean "no captions" condition.
- Baselines: majority class; downsampled-pixel PCA (no pretraining); logistic regression on the frozen features.

**Secondary result: imaging modality.** On 252 test images, 1-NN leave-one-out modality accuracy from frozen features is 0.956 (BiomedCLIP) and 0.948 (DINOv2), against 0.409 for the majority class. The report should state as a finding that frozen image features already separate modality, so captions have little room to help there. Mechanism: the objective-interpretation skill tells the agent to evaluate other declared labels with a frozen probe as secondary results (generic rule, no dataset names).

**Evidence already in the profile** (seeded sample of 3,000; near-duplicate = cosine ≥ 0.95; share of the 10 nearest neighbors that are near-duplicates, vs the share of random pairs above the cutoff):

| Representation | kNN near-duplicates | Random pairs ≥ cutoff |
|---|---|---|
| caption, TF-IDF | 0.0014 | 0.0 |
| image, BiomedCLIP | 0.0047 | 0.0 |
| caption, BiomedCLIP | 0.0077 | 0.0001 |
| image, DINOv2 | **0.1879** | 0.0053 |
| (PBMC) RNA, PCA | 0.0067 | 0.0 |
| (PBMC) ATAC, PCA | 0.0 | 0.0 |

In DINOv2 space many images sit almost on top of each other, so a DINOv2-based two-tower model faces InfoNCE false negatives. That gives the agent a real reason to consider `vicreg`.

**What should count as success:** at 5 and 20 labels per class, a caption-trained image representation beats the same encoder's frozen features on body-region macro-F1 by more than the spread over label draws and seeds. At 100 and all, the gap may close; that is an acceptable, informative result. Risks: pretraining overlap for BiomedCLIP (PubMed Central); small test classes (spine 19, upper limb 26).

## Validator expectations

What `validate_objective.py` and `validate_plan.py` must enforce for these objectives (implemented generically, never by dataset name):
- **Label budgets** in `objective.json` are per class (`label_budgets_per_class`, integers plus `"all"`). Each integer budget must be fillable by every retained class's training count after the validation carve-out. Classes that can't fill it must be excluded by the plan, with a reason, or flagged in the report as "budget not filled".
- **Inference vs training-only modalities.** `inference_modalities` (ATAC; image) and `training_only_modalities` (RNA; caption). No evaluation input may come from a training-only modality, and no metric may use a training-only modality of a test sample.
- **Label use.** Methods that use labels in representation training (`supcon`, `supervised_fusion`) may use only labels inside the budget being evaluated. Each budget gets its own model.
- **Training pool.** `plan.json` → `training_pool` (all paired training samples, or only samples with a given label present) has a reason.
- **Class exclusions.** `plan.json` → `excluded_classes` (e.g. a class under the small-class flag) has a reason that cites the profile counts.
- **Secondary labels.** Other declared labels appear in `objective.json` → `secondary_tasks`, evaluated with frozen probes only.

## Run commands

```
./run.sh pbmc_multiome "Future samples will be profiled with scATAC-seq only. Using this paired RNA+ATAC data, learn a representation that lets ATAC-only cells be annotated by cell type and have their gene expression estimated, and show how much better that is than working from ATAC alone." --report-number 1

./run.sh rocov2 "Our archive has captions for every radiograph but expert labels for only a few. Learn image representations that identify the body region from the image alone when only a few labeled examples per region are available, and tell me whether using the captions during training helps." --report-number 2
```
