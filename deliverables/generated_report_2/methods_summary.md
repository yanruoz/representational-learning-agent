# Methods summary: rocov2 (report 2)

Objective: Our archive has captions for every radiograph but expert labels for only a few. Learn image representations that identify the body region from the image alone when only a few labeled examples per region are available, and tell me whether using the captions during training helps.

## Decisions

| decision | choice | evidence | alternative rejected |
|---|---|---|---|
| split | declared | 7551/1333/1655 train/val/test | random |
| training pool | all | - | label_present |
| exclude body_region classes | breast | counts 9, 4 | keep them |
| image encoder | DINOv2 | linear macro-F1 0.674; linear macro-F1 0.809 | pixel PCA, BiomedCLIP |
| caption encoder | BiomedCLIP | CCA FOSCTTM 0.0471; CCA FOSCTTM 0.149 | TF-IDF |
| caption encoder | TF-IDF | CCA FOSCTTM 0.149; CCA FOSCTTM 0.0471 | BiomedCLIP |
| image encoder | pixel PCA | kNN macro-F1 0.307 | BiomedCLIP, DINOv2 |
| loss (4 models) | InfoNCE | near-duplicate neighbors 0.0077; random pairs above cutoff 0.0001 | SupCon |
| loss (1 model) | VICReg | near-duplicate neighbors 0.188; chest 1031 | SupCon |
| exported representation | bmc_frozen | see Results | the other models |

## Models: encoders, losses, reasons, hyperparameters

### pixel PCA (no training) (`pix_single`)

- Method: single_modality_features. Required no-pretraining single-modality baseline on the only inference modality (image), and the without-captions control for tt_pix_tfidf_infonce.
- image: encoder `pixels_pca`. The no-pretraining image baseline: downsampled pixels + PCA (64 dims) learned only from this archive. Its validation kNN macro-F1 at 5 labels per region is 0.307, far above the majority class (0.0753) but far below pretrained encoders, so it shows what the images give with no outside knowledge.

### pixel PCA + logistic/PLS (`pix_logreg`)

- Method: linear_supervised. Required linear supervised baseline for the label task. In revision 1 it reported exactly the numbers of pix_single's linear probe (same logistic regression on the same pixel features), so the report treats the two as one baseline.
- image: encoder `pixels_pca`. Logistic regression on the no-pretraining pixel PCA features, trained only on the budgeted body_region labels; validation linear macro-F1 0.2664 at 5 per region and 0.4815 with all labels shows how much a purely supervised linear model gets from this archive.

### Frozen DINOv2 (control) (`dino_frozen`)

- Method: frozen_probe. Without-captions control: same frozen DINOv2 features the two-tower models start from, probed with kNN and logistic regression at each budget.
- image: encoder `cache:dinov2_vitb14_frozen`. The image-only representation without captions, and the matched control for every DINOv2 caption-aligned model. DINOv2 was pretrained on web images with no text, so it is the clean A-encoder for asking whether captions help: validation linear macro-F1 0.6739 at 5 per region and 0.7395 at 20, below BiomedCLIP's 0.8095 and 0.8578 but with no caption influence; LVD-142M is web-retrieved, so overlap with these PubMed Central figures is not documented either way, but it was not built from PubMed Central figure-caption pairs.

### Frozen BiomedCLIP (reference) (`bmc_frozen`)

- Method: frozen_probe. Reference for what large-scale caption-supervised pretraining (with possible overlap) achieves; not the success bar.
- image: encoder `cache:biomedclip_frozen`. Reference only. BiomedCLIP's image tower has the best validation body_region scores (linear macro-F1 0.8095 at 5 per region vs 0.6739 for DINOv2), but it was trained together with captions on PMC-15M, which may contain these very PubMed Central figures, so its numbers may be inflated by overlap and already include caption influence; it cannot answer whether captions from this archive help.

### CCA, DINOv2/BiomedCLIP (`cca_dino_bmccap`)

- Method: linear_alignment. Linear caption-alignment baseline: if the nonlinear two-tower models do not beat a CCA image projection, the caption benefit (if any) does not need a learned model. Seed 0 fits all training rows and seeds 1-2 fit seeded 90% subsamples, so its SD mixes subsampling with seed and is a rough spread, not a seed-only SD.
- image: encoder `cache:dinov2_vitb14_frozen`. Linear way of using captions: CCA projects DINOv2 image features onto directions shared with the captions. DINOv2 with BiomedCLIP caption features reached validation FOSCTTM 0.04713 vs 0.14936 with TF-IDF captions, so the caption side carries more image-related signal in that representation.
- caption: encoder `cache:biomedclip_frozen`. BiomedCLIP text features of the training captions (training only, never a test input); best CCA partner for DINOv2 on validation (FOSCTTM 0.04713 vs 0.14936 for TF-IDF + SVD).

### DINOv2 + caption (InfoNCE, BiomedCLIP caption) (`tt_dino_bmccap_infonce`)

- Method: two_tower. Main caption-using model: DINOv2 image tower aligned to captions with InfoNCE; its image tower alone embeds new images. Compared with dino_frozen (same encoder, no captions) at each label budget.
- image: encoder `cache:dinov2_vitb14_frozen`, head mlp. DINOv2 is the image encoder pretrained without text (no captions in LVD-142M), so any change from dino_frozen is caused by training on this archive's captions. Its validation linear macro-F1 at 5 per region (0.6739) leaves room to improve below BiomedCLIP's 0.8095.
- caption: encoder `cache:biomedclip_frozen`, head mlp. Training-only caption tower. BiomedCLIP text features align with DINOv2 images much better than TF-IDF on validation (CCA FOSCTTM 0.04713 vs 0.14936). Judgment call: these text features come from a model pretrained on PubMed Central pairs, so the caption signal is partly outside knowledge; tt_dino_tfidf_infonce is the archive-only alternative.
- Loss: infonce. Judgment call. InfoNCE treats each image's own caption as the positive and the other pairs in the batch as negatives, in both directions. On the caption side false negatives are rare: 0.0077 of BiomedCLIP-caption nearest neighbors are near-duplicates and 0.0001 of random pairs pass the cutoff. On the image side they are not rare: 0.2327 of DINOv2 images have a near-duplicate and 0.1879 of neighbors are near-duplicates (with 0.0053 of random DINOv2 pairs also above the cutoff, so part of this is inflated cosine), so the image-to-caption direction will sometimes push apart near-identical panels. I keep InfoNCE because the captions, which supply the training signal, are distinct, and tt_dino_bmccap_vicreg is the negative-free alternative that tests whether the image-side false negatives matter.
- `embed_dim` = 64: CCA between DINOv2 and BiomedCLIP captions was best at 20 of 50 components on validation (FOSCTTM 0.04713) and worse at 50 (0.09357), so the shared signal is low-dimensional; 64 leaves room for nonlinear structure beyond those 20 directions while compressing DINOv2's effective rank of 442.71 to what the captions can supervise.
- `hidden_dims` = [512]: One hidden layer of 512 on the 768-d DINOv2 input (about 0.43M parameters per tower) for 7551 training pairs. In revision 1 this tower already reached its best validation loss at epoch 9 and then overfit, so a deeper tower is not warranted.
- `epochs` = 200: 7551 pairs at batch 256 is about 30 steps per epoch; 200 epochs (about 6000 steps) is an upper limit that early stopping on the 1333-pair validation split cuts short (revision 1 stopped at 29 epochs).
- `batch_size` = 256: 256 gives 255 caption negatives per image; with 0.0077 near-duplicate captions among neighbors, fewer than 2 caption false negatives are expected per batch.
- `lr` = 0.0003: 3e-4 with AdamW on fixed features. Revision 1 at this rate reached its best validation loss at epoch 9 (about 270 steps), so a higher rate would only reach the overfitting regime sooner; runs take about 4 seconds.
- `patience` = 20: About 30 steps per epoch make epoch-level validation loss noisy; 20 epochs (about 600 steps) avoids stopping on noise.
- `temperature_init` = 0.07: Judgment call: learnable temperature starting at 0.07, the value BiomedCLIP-style image-caption models train toward; the alternative 0.1 is softer. The final value is checked against its bound in runs/*.json.
- `dropout` = 0.2: 0.2 because the head has about 0.43M parameters per tower against 7551 pairs; dropout limits memorizing individual pairs.
- `weight_decay` = 0.001: 1e-3 for the same parameter-to-pair ratio; it also keeps the image tower close to a smooth map of DINOv2 features, which matters because the probe later sees only 5 labels per region.

### DINOv2 + caption (InfoNCE, input kept) (`tt_dino_bmccap_infonce_concat`)

- Method: two_tower. Tests whether caption-aligned dimensions add information on top of the frozen DINOv2 features; compared with dino_frozen and tt_dino_bmccap_infonce.
- image: encoder `cache:dinov2_vitb14_frozen`, head mlp. Same DINOv2 input as tt_dino_bmccap_infonce (validation linear macro-F1 0.8245 with all labels on the frozen features). That strong frozen linear probe can be lost when a 64-d aligned embedding compresses 442.71 effective dimensions, so this variant keeps the frozen features next to the aligned ones.
- caption: encoder `cache:biomedclip_frozen`, head mlp. Same training-only caption tower as tt_dino_bmccap_infonce (CCA FOSCTTM with DINOv2 0.04713 vs 0.14936 for TF-IDF).
- Loss: infonce. Same InfoNCE as tt_dino_bmccap_infonce so only concat_input differs: caption near-duplicates are rare (0.0077 of neighbors) and 0.0001 of random caption pairs pass the cutoff, so in-batch negatives are mostly true negatives.
- `embed_dim` = 64: Matched to tt_dino_bmccap_infonce (64) so only concat_input differs.
- `hidden_dims` = [512]: Matched to tt_dino_bmccap_infonce ([512]).
- `epochs` = 200: Matched to tt_dino_bmccap_infonce: about 30 steps per epoch on 7551 pairs with early stopping.
- `batch_size` = 256: Matched to tt_dino_bmccap_infonce (256; fewer than 2 expected false-negative captions per batch).
- `lr` = 0.0003: Matched to tt_dino_bmccap_infonce (3e-4).
- `patience` = 20: Matched to tt_dino_bmccap_infonce (20 epochs, about 600 steps).
- `temperature_init` = 0.07: Matched to tt_dino_bmccap_infonce (0.07, a judgment call).
- `dropout` = 0.2: Matched to tt_dino_bmccap_infonce (0.2).
- `weight_decay` = 0.001: Matched to tt_dino_bmccap_infonce (1e-3).
- `concat_input` = True: The frozen DINOv2 linear probe reaches validation macro-F1 0.8245 with all labels; if the 64-d aligned embedding loses region detail, concatenating the frozen 768-d input keeps it, so this tests whether caption alignment adds to, rather than replaces, the frozen features.

### DINOv2 + caption (VICReg) (`tt_dino_bmccap_vicreg`)

- Method: two_tower. Loss ablation for the main caption model: same encoders, heads and regularization, trained without negatives.
- image: encoder `cache:dinov2_vitb14_frozen`, head mlp. Same DINOv2 input as tt_dino_bmccap_infonce so only the loss differs; DINOv2 is the text-free image encoder (validation linear macro-F1 0.6739 at 5 per region).
- caption: encoder `cache:biomedclip_frozen`, head mlp. Same training-only caption tower as tt_dino_bmccap_infonce (CCA FOSCTTM with DINOv2 0.04713).
- Loss: vicreg. Judgment call, a matched loss ablation. VICReg uses no negatives, so images of the same region in a batch are not pushed apart; 0.1879 of DINOv2 nearest neighbors are near-duplicates and chest alone is 1031 of 2868 region labels, so if many same-region images share a batch, InfoNCE could separate them and hurt the region probe. The caption-side redundancy (0.0077) argues the opposite, so this checks which matters.
- `embed_dim` = 64: Matched to tt_dino_bmccap_infonce (64) so only the loss differs.
- `hidden_dims` = [512]: Matched to tt_dino_bmccap_infonce ([512]).
- `epochs` = 200: Matched to tt_dino_bmccap_infonce: early stopping on validation; at batch 512 there are about 15 steps per epoch, and revision 1 ran about 100 epochs before stopping, well under 200.
- `batch_size` = 512: Raised from 256 to 512 after the critique: in revision 1 the VICReg image embedding had effective rank about 23.7 of 64 (InfoNCE about 52), and 256 samples estimate the 64x64 covariance term from only 4 samples per dimension; 512 doubles that. This breaks the batch match with InfoNCE, which is not a comparable quantity anyway (InfoNCE batch sets negatives, VICReg batch sets covariance estimation).
- `lr` = 0.0003: Matched to tt_dino_bmccap_infonce (3e-4).
- `patience` = 20: Matched to tt_dino_bmccap_infonce (20 epochs).
- `dropout` = 0.2: Matched to tt_dino_bmccap_infonce (0.2) so regularization does not confound the loss comparison.
- `weight_decay` = 0.001: Matched to tt_dino_bmccap_infonce (1e-3).

### DINOv2 + caption (InfoNCE, TF-IDF caption) (`tt_dino_tfidf_infonce`)

- Method: two_tower. Archive-only caption variant: separates the effect of this archive's captions from BiomedCLIP's text pretraining.
- image: encoder `cache:dinov2_vitb14_frozen`, head mlp. Same text-free DINOv2 image input as tt_dino_bmccap_infonce (validation linear macro-F1 0.6739 at 5 per region), so only the caption encoder differs.
- caption: encoder `tfidf_svd`, head mlp. Archive-only caption features (TF-IDF + SVD, 128 dims, no pretraining), so any gain comes from this archive's captions and not from BiomedCLIP's PubMed Central text pretraining. The cost is a weaker caption-image link: CCA FOSCTTM 0.14936 vs 0.04713 with BiomedCLIP captions.
- Loss: infonce. InfoNCE with TF-IDF caption negatives: 0.0014 of TF-IDF caption neighbors are near-duplicates and 0.0 of random pairs pass the cutoff, the lowest redundancy of any representation here, so false negatives are negligible.
- `embed_dim` = 64: Matched to tt_dino_bmccap_infonce (64); CCA with TF-IDF captions was best at only 10 components (FOSCTTM 0.14936), so 64 is ample.
- `hidden_dims` = [512]: Matched to tt_dino_bmccap_infonce ([512]) so only the caption encoder differs.
- `epochs` = 200: Matched: about 30 steps per epoch on 7551 pairs with early stopping.
- `batch_size` = 256: Matched (256); with 0.0014 near-duplicate caption neighbors, false negatives per batch are well under 1.
- `lr` = 0.0003: Matched to tt_dino_bmccap_infonce (3e-4).
- `patience` = 20: Matched (20 epochs).
- `temperature_init` = 0.07: Matched (0.07, a judgment call).
- `dropout` = 0.2: Matched to tt_dino_bmccap_infonce (0.2) so only the caption encoder differs.
- `weight_decay` = 0.001: Matched to tt_dino_bmccap_infonce (1e-3) so only the caption encoder differs.

### pixel PCA + caption (InfoNCE) (`tt_pix_tfidf_infonce`)

- Method: two_tower. Fully archive-only test of whether captions help: compared with pix_single and pix_logreg on the same pixel features.
- image: encoder `pixels_pca`, head mlp. No-pretraining pixel features (validation kNN macro-F1 0.307 at 5 per region): with no outside knowledge on either side, this tests whether captions help when the image features start weak, compared with pix_single.
- caption: encoder `tfidf_svd`, head mlp. No-pretraining caption features, so the whole model uses only this archive. CCA between pixels and TF-IDF reaches validation FOSCTTM 0.30264 (chance 0.5), a weak but real link.
- Loss: infonce. InfoNCE: the TF-IDF caption representation has the lowest redundancy measured (0.0014 of neighbors near-duplicates, 0.0 random pairs above the cutoff), so in-batch negatives are true negatives.
- `embed_dim` = 32: The pixel input has only 64 dims with effective rank 45.2, and CCA with TF-IDF was best at 10 components (FOSCTTM 0.30264), so 32 is enough.
- `hidden_dims` = [256]: One hidden layer of 256 on a 64-d input, sized down from the DINOv2 models because the input is 12 times smaller.
- `epochs` = 200: About 30 steps per epoch on 7551 pairs; 200 epochs is an upper limit cut by early stopping.
- `batch_size` = 256: 256, with well under 1 expected false-negative caption per batch (0.0014 near-duplicate neighbors).
- `lr` = 0.0003: 3e-4, matched to the DINOv2 caption models; runs are short.
- `patience` = 20: 20 epochs (about 600 steps) because the weak pixel-caption link makes validation loss noisy.
- `temperature_init` = 0.07: Judgment call: 0.07, matched to the other InfoNCE models; final value checked against its bound.
- `dropout` = 0.2: 0.2, matched to the other two-tower models.
- `weight_decay` = 0.001: 1e-3, matched to the other two-tower models.

## Revision after the critique

One revision for all critique findings. (1) Probes disagree: objective.json now names the linear probe as primary (the stronger probe on frozen DINOv2 in validation, 0.6739 vs 0.5051 kNN at 5/class) and requires a gap larger than both seed and label-draw SD; kNN stays a reported secondary probe. (2) VICReg partial collapse (image effective rank about 23.7 of 64 vs about 52 for InfoNCE): tt_dino_bmccap_vicreg batch_size raised from 256 to 512 so the 64x64 covariance term is estimated from 8 rather than 4 samples per dimension; everything else stays matched. (3) Reasons corrected: the InfoNCE loss_reason now accepts image-side false negatives (0.2327 of DINOv2 images have a near-duplicate) instead of dismissing them; the embed_dim reason no longer compares 64 dims with 7 classes; reasons citing a previous pipeline run are rewritten from this run's evidence; the CCA reason states that seed 0 fits all rows and seeds 1-2 fit 90% subsamples; the dino_frozen reason says PMC overlap is undocumented rather than absent. (4) modality_dropout 0.3 and supcon_temperature 0.1 in runs/*.json are train.py defaults that infonce and vicreg two-tower losses never read (train.py uses modality_dropout only in fused models and supcon_temperature only for supcon), so they are not set in the plan. Not fixable in the plan (no caption-free trained projection, PCA-reduced encoder, or shuffled-pair option exists among the available methods and encoders) and stated as limitations in findings.md: the kNN gain may come from the low-dimensional projection rather than caption content; bmc_frozen beats every learned model (with PMC overlap caveat); frozen baselines are single runs with label-draw SD only in the per-run metrics; BiomedCLIP caption overlap and labels partly extracted from captions; near-duplicate panels with no article id; fast InfoNCE overfitting (handled by early stopping); pix_logreg equals pix_single and the concat model shares its towers with the main model; all-budget gaps under about 0.02 are not ranked.

## Training log (per run)

| run | epochs | stop | train pairs | seconds | config hash |
|---|---|---|---|---|---|
| tt_dino_bmccap_infonce_s0 | 29 | early_stopping (no validation improvement for 20 epochs) | 7551 | 4.2 | 9d8c703297c20f47 |
| tt_dino_bmccap_infonce_s1 | 30 | early_stopping (no validation improvement for 20 epochs) | 7551 | 3.4 | 2ee406a22a6e63e2 |
| tt_dino_bmccap_infonce_s2 | 29 | early_stopping (no validation improvement for 20 epochs) | 7551 | 3.3 | 1dc6c4e183a58a7f |
| tt_dino_bmccap_infonce_concat_s0 | 29 | early_stopping (no validation improvement for 20 epochs) | 7551 | 3.7 | 2776673ad9f00959 |
| tt_dino_bmccap_infonce_concat_s1 | 30 | early_stopping (no validation improvement for 20 epochs) | 7551 | 3.7 | 2a091e883e76c3be |
| tt_dino_bmccap_infonce_concat_s2 | 29 | early_stopping (no validation improvement for 20 epochs) | 7551 | 3.6 | f209006fe043566c |
| tt_dino_bmccap_vicreg_s0 | 119 | early_stopping (no validation improvement for 20 epochs) | 7551 | 10.3 | 2b1250a0dc5a61a4 |
| tt_dino_bmccap_vicreg_s1 | 145 | early_stopping (no validation improvement for 20 epochs) | 7551 | 12.1 | 216b2f03a3cbf36b |
| tt_dino_bmccap_vicreg_s2 | 122 | early_stopping (no validation improvement for 20 epochs) | 7551 | 10.1 | 2f8375122a652c90 |
| tt_dino_tfidf_infonce_s0 | 26 | early_stopping (no validation improvement for 20 epochs) | 7551 | 3.1 | 489813673f38408f |
| tt_dino_tfidf_infonce_s1 | 29 | early_stopping (no validation improvement for 20 epochs) | 7551 | 2.8 | 3a39515c3b1b1f5c |
| tt_dino_tfidf_infonce_s2 | 29 | early_stopping (no validation improvement for 20 epochs) | 7551 | 2.9 | 468ced6bc04451a6 |
| tt_pix_tfidf_infonce_s0 | 36 | early_stopping (no validation improvement for 20 epochs) | 7551 | 1.8 | ee35167d7bf30dd6 |
| tt_pix_tfidf_infonce_s1 | 39 | early_stopping (no validation improvement for 20 epochs) | 7551 | 2.0 | 0af1858a867c1d4f |
| tt_pix_tfidf_infonce_s2 | 38 | early_stopping (no validation improvement for 20 epochs) | 7551 | 1.9 | 229fc3f2106eed07 |

Versions: python 3.13.5, torch 2.14.0, numpy 2.5.3, scikit-learn 1.9.1.
