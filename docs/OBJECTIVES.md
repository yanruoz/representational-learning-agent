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

> Our archive has captions for every figure but expert labels for only a few. Learn image representations that identify imaging modality (and body region or finding if there are enough examples) from the image alone when only a small fraction of images is labeled, and tell me whether using captions during training helps.

**Concept counts beyond modality** (this subset; train / test / total). All are on plain X-ray images. The copy keeps at most 3 concepts per image (modality, body region, view direction), so there are **no finding concepts**.

| Code | Kind | Meaning | Train | Test | Total |
|---|---|---|---|---|---|
| C0817096 | body region | chest | 876 | 155 | 1031 |
| C0037303 | body region | cranium (mostly jaw and dental panoramic) | 468 | 53 | 521 |
| C0023216 | body region | lower limb | 302 | 71 | 373 |
| C0000726 | body region | abdomen | 277 | 43 | 320 |
| C0030797 | body region | pelvis | 194 | 32 | 226 |
| C1140618 | body region | upper limb | 167 | 26 | 193 |
| C0037949 | body region | spine | 172 | 19 | 191 |
| C0006141 | body region | breast | 9 | 4 | 13 |
| C1999039 | view | anteroposterior | 1080 | 213 | 1293 |
| C1996865 | view | posteroanterior | 544 | 72 | 616 |
| C0205129 | view | lateral / sagittal | 276 | 44 | 320 |
| C0205106 | view | (another view; captions mention axillary views) | 4 | 2 | 6 |

The names come from UMLS and were checked against the most common words in the matching captions. C0205106's exact meaning is unconfirmed.

**Proposed label set:**
- `modality`: 6 classes on all labeled images (10,451). PET has 5 test images; report its F1 separately or merge it, with a reason.
- `body_region`: X-ray images only, 7 regions (breast left out: 13 examples, 4 in the test split). 2,855 images; 19 to 155 test images per class. At 1% labels this is about 24 labeled training images across 7 classes, so the 1% result for body region is expected to be noisy.
- Not proposed: view direction (not asked for) and findings (none in this copy).

Both labels are declared in `data/rocov2/DATA_DESCRIPTION.md`, including breast. The profile flags breast as a class with under 20 examples, so dropping it is the agent's decision, with that evidence.

**Intended evaluation** (image only on the official test split; captions are used in training only):
- For each label: linear probe (and kNN) macro-F1 on the test set, training the probe on 1%, 10%, and 100% of the labeled training images. Draw the labeled subsets stratified by class with at least one example per class, with several seeds at 1% and 10%.
- "Do captions help": compare the same image encoder with and without caption-based training. The cleanest pair is DINOv2 (never trained on captions): frozen features vs a caption-aligned head trained on them. BiomedCLIP's image tower was itself trained with captions, so it is a reference, not a clean "no captions" condition.
- Baselines: majority class; downsampled-pixel PCA (no pretraining).

**Evidence already in the profile** (seeded sample of 3,000; near-duplicate = cosine ≥ 0.95; share of the 10 nearest neighbors that are near-duplicates, vs the share of random pairs above the cutoff):

| Representation | kNN near-duplicates | Random pairs ≥ cutoff |
|---|---|---|
| caption, TF-IDF | 0.0014 | 0.0 |
| image, BiomedCLIP | 0.0047 | 0.0 |
| caption, BiomedCLIP | 0.0077 | 0.0001 |
| image, DINOv2 | **0.1879** | 0.0053 |
| (PBMC) RNA, PCA | 0.0067 | 0.0 |
| (PBMC) ATAC, PCA | 0.0 | 0.0 |

In DINOv2 space many images sit almost on top of each other (similar-looking scans), so a two-tower model built on DINOv2 features faces InfoNCE false negatives. That gives the agent a real reason to consider `vicreg`. In every other representation, near-duplicates are rare.

**Frozen features are already strong for modality:** 1-NN leave-one-out accuracy on 252 test images is 0.956 (BiomedCLIP) and 0.948 (DINOv2), against 0.409 for the majority class. Captions may not help modality. Body region is where an effect is more likely.

**What should count as success:** at 1% and 10% labels, a caption-trained image representation beats the same encoder's frozen features on modality macro-F1 by more than the seed spread. At 100% the gap may close; that is an acceptable, informative result. Risks: pretraining overlap for BiomedCLIP (PubMed Central), and the 1% body-region condition has very few labels.

## Run commands

```
./run.sh pbmc_multiome "Future samples will be profiled with scATAC-seq only. Using this paired RNA+ATAC data, learn a representation that lets ATAC-only cells be annotated by cell type and have their gene expression estimated, and show how much better that is than working from ATAC alone." --report-number 1

./run.sh rocov2 "Our archive has captions for every figure but expert labels for only a few. Learn image representations that identify imaging modality (and body region or finding if there are enough examples) from the image alone when only a small fraction of images is labeled, and tell me whether using captions during training helps." --report-number 2
```
