# Methods summary: pbmc_multiome (report 1)

Objective: I have cells measured with RNA in one experiment and with ATAC in another. Learn a shared representation where a cell’s RNA and ATAC profiles can be matched, so cell-type annotations can be transferred from RNA-annotated cells to ATAC-only cells. Tell me how well matching works and how accurate the transferred labels are.

## Decisions

| decision | choice | evidence | alternative rejected |
|---|---|---|---|
| split | random | 6548/1156/1927 train/val/test | declared / grouped |
| training pool | all | - | label_present |
| exclude cell_type classes | HSPC, Plasma | counts 17, 12 | keep them |
| rna encoder | LSI | CCA FOSCTTM 0.0165; CCA FOSCTTM 0.0213 | PCA |
| atac encoder | LSI | CCA FOSCTTM 0.0165; CCA FOSCTTM 0.0416 | PCA |
| loss (1 model) | InfoNCE | near-duplicate neighbors 0.0067; near-duplicate neighbors 0 | SupCon |
| loss (1 model) | VICReg | CD14 Mono 2554; near-duplicate neighbors 0.0067 | SupCon |
| exported representation | tt_infonce_lsi | see Results | the other models |

## Models: encoders, losses, reasons, hyperparameters

### LSI/LSI (no training) (`lsi_single`)

- Method: single_modality_features. No-pretraining reference for both modalities in one entry: its ATAC features are scored on the secondary task (kNN and linear probe trained with ATAC's own labels), the upper reference for transfer, with the test majority class next to it.
- rna: encoder `lsi`. No-pretraining RNA features, the same LSI the alignment models use, so any gain is attributable to alignment. LSI beat PCA for RNA on validation kNN macro-F1 (0.8362 vs 0.7658); linear macro-F1 is within 0.002 (0.8714 vs 0.8733), so kNN decides.
- atac: encoder `lsi`. No-pretraining ATAC features. LSI is chosen over PCA on the validation comparison: CCA with ATAC-LSI reached FOSCTTM 0.01652 against 0.04157 with ATAC-PCA (both with RNA-LSI).

### LSI/LSI + logistic/PLS (`lsi_logreg`)

- Method: linear_supervised. Linear supervised baseline required for label tasks; restores the ATAC-side logistic-regression reference that failed in revision 1.
- rna: encoder `lsi`. RNA-LSI is listed because run_plan.py takes linear_supervised features from the first inference modality (rna); same LSI as the other models (RNA kNN macro-F1 0.8362 vs 0.7658 for PCA on validation).
- atac: encoder `lsi`. ATAC-LSI with logistic regression on ATAC's training labels: the linear supervised reference for the transfer target; LSI beat PCA in validation CCA FOSCTTM (0.01652 vs 0.04157).

### CCA, LSI/LSI (`cca_lsi`)

- Method: linear_alignment. Required linear baseline and the success-criterion bar: a learned shared space has to beat CCA on FOSCTTM by more than the seed spread. Number of components is chosen on validation.
- rna: encoder `lsi`. RNA-LSI with ATAC-LSI gave the lowest validation CCA FOSCTTM of the four encoder pairs (0.01652 vs 0.02131 for RNA-PCA with ATAC-LSI).
- atac: encoder `lsi`. ATAC-LSI gave validation CCA FOSCTTM 0.01652 vs 0.04157 for ATAC-PCA (paired with RNA-LSI).

### Two-tower rna/atac on LSI (InfoNCE) (`tt_infonce_lsi`)

- Method: two_tower. Main learned model: nonlinear towers that embed RNA and ATAC separately into one space, so an ATAC-only cell can be matched to RNA cells and given their labels. Compared with cca_lsi on the same inputs.
- rna: encoder `lsi`, head mlp. LSI input for the same reason as CCA (best validation FOSCTTM 0.01652 of the four pairs); an MLP head lets the tower learn a nonlinear map, which is the only thing that distinguishes this model from CCA.
- atac: encoder `lsi`, head mlp. ATAC-LSI beat ATAC-PCA in validation CCA FOSCTTM (0.01652 vs 0.04157); MLP head for a nonlinear map into the shared space.
- Loss: infonce. The primary task is to identify each cell's own partner among all others, which is exactly what InfoNCE optimizes (each RNA profile must pick its ATAC partner out of the batch). False negatives from near-duplicates are rare here: in the RNA representation 0.0067 of nearest neighbors and in ATAC 0.0 are near-duplicates, with 0.0 of random pairs above the cutoff in either. Same-class cells are frequent in a batch (CD14 Mono is 2554 of 9631 cells, about 68 per batch of 256), but for cell-level matching these must be told apart, so treating them as negatives is what the objective wants; the label-transfer cost of that is checked against VICReg in tt_vicreg_lsi.
- `embed_dim` = 32: CCA reached its best validation FOSCTTM at 20 of 50 components (0.01652) and got worse at 50 (0.02259), so the shared RNA-ATAC signal lives in roughly 20 dimensions; 32 leaves room for nonlinear structure. The revision-1 InfoNCE embeddings had effective rank about 22 of 32 with no low-variance dimensions, consistent with that.
- `hidden_dims` = [128]: Revision 1 with [256, 128] overfit: validation loss bottomed at epoch 12 (1.9258) and rose to 2.2422 by epoch 37 while training loss kept falling. One hidden layer of 128 on the 50-d LSI input cuts tower parameters to about 11k, for 6548 training pairs.
- `epochs` = 300: 6548 training pairs at batch 256 is about 26 steps per epoch, so 300 epochs is about 7800 steps; at the lower learning rate convergence is slower and early stopping on the 1156-cell validation split ends training when it plateaus.
- `batch_size` = 256: 256 of 6548 pairs gives 255 negatives per query, enough to make partner identification hard, while expected same-class negatives (about 68 for CD14 Mono) are acceptable for a cell-level matching objective.
- `lr` = 0.0003: Lowered from 1e-3 to 3e-4 because the revision-1 runs reached their best validation loss within about 12 epochs (about 312 steps) and then overfit; each run took about 3 seconds, so the slower learning rate costs nothing that matters against the 30-minute budget.
- `patience` = 25: With only about 26 steps per epoch, validation loss is noisy epoch to epoch; 25 epochs (about 650 steps) avoids stopping on noise.
- `temperature_init` = 0.1: Judgment call: learnable temperature starting at 0.1. In revision 1 it moved from 0.1 to 0.072-0.078 across seeds, well above the 0.01 bound, so the start value is not limiting; the alternative 0.07 would start near where training ends but gives no evidence-based advantage.
- `dropout` = 0.2: Raised from the implicit 0.1 to 0.2 as one of the regularizers answering the revision-1 overfit (training loss 0.952 vs validation 2.242 at epoch 37).
- `weight_decay` = 0.001: Raised from the implicit 1e-4 to 1e-3 for the same revision-1 overfit; with about 11k parameters per tower and 6548 pairs, a stronger penalty limits memorizing individual pairs.

### Two-tower rna/atac on LSI (VICReg) (`tt_vicreg_lsi`)

- Method: two_tower. Loss ablation for the main model: same inputs, heads, sizes and regularization as tt_infonce_lsi, trained without negatives.
- rna: encoder `lsi`, head mlp. Same LSI input and MLP head as tt_infonce_lsi so that only the loss differs; LSI had the best validation CCA FOSCTTM (0.01652).
- atac: encoder `lsi`, head mlp. Same LSI input and head as tt_infonce_lsi; ATAC-LSI beat ATAC-PCA in validation CCA FOSCTTM (0.01652 vs 0.04157).
- Loss: vicreg. Judgment call, a matched loss ablation for InfoNCE. VICReg uses no negatives, so it does not push apart the many same-class cells sharing a batch (CD14 Mono is 2554 of 9631 cells); the hypothesis was that this could keep cell types compact for label transfer. Revision 1 did not support it (worse matching and transfer than InfoNCE), and its effective rank was about 13 of 32, so this run tests whether the same regularization changes that picture; a loss-level conclusion is drawn only if VICReg is not under-trained. Redundancy is low (RNA near-duplicate kNN fraction 0.0067), so VICReg is not needed to avoid false negatives.
- `embed_dim` = 32: Matched to tt_infonce_lsi (32) so only the loss differs; CCA's best validation size of 20 components suggests about 20 shared dimensions.
- `hidden_dims` = [128]: Matched to tt_infonce_lsi ([128]), shrunk after the revision-1 InfoNCE overfit so the two losses keep the same towers.
- `epochs` = 300: Matched to tt_infonce_lsi: about 26 steps per epoch on 6548 pairs; early stopping on the 1156-cell validation split.
- `batch_size` = 256: 256 so the variance and covariance terms are estimated from enough cells (256 samples for a 32x32 covariance) and to match tt_infonce_lsi.
- `lr` = 0.0003: Matched to tt_infonce_lsi (3e-4), lowered after the revision-1 overfit; runs took about 6 seconds, so the budget is not a constraint.
- `patience` = 25: Matched to tt_infonce_lsi: 25 epochs; the revision-1 VICReg validation loss was noisy (best at epoch 67 of 92), so a short patience would stop on noise.
- `dropout` = 0.2: Matched to tt_infonce_lsi (0.2) so the loss comparison is not confounded by regularization.
- `weight_decay` = 0.001: Matched to tt_infonce_lsi (1e-3) so the loss comparison is not confounded by regularization.

## Revision after the critique

One revision for all critique findings. (1) Empty references: objective.json gained a secondary task, cell_type prediction from ATAC with ATAC's own labels, and the single-modality and logistic-regression baselines were merged into lsi_single and lsi_logreg with both rna and atac LSI inputs, so they are scored on the test split (ATAC-native upper reference, test majority class, per-class F1) and the rna input keeps run_plan.py's linear_supervised path working. (2) InfoNCE overfitting (validation loss lowest at epoch 12, 1.9258, then rising to 2.2422 by epoch 37): towers shrunk to one hidden layer of 128, dropout raised from the implicit 0.1 to 0.2, weight_decay from the implicit 1e-4 to 1e-3, lr lowered from 1e-3 to 3e-4; the same values are applied to VICReg so the loss comparison stays matched. (3) Unreasoned hyperparameters: dropout and weight_decay are now set with reasons; modality_dropout is not set because train.py uses it only for fused models and two-tower towers never fuse. (4) Reasons contradicted by results were rewritten: hidden_dims, lr, temperature_init, VICReg loss reason, and the ATAC encoder reason (the low share of 1s reflects cut-site doubling, not depth, so it is no longer cited). Not fixable in the plan and stated as limitations instead: the CCA validation FOSCTTM differs between baseline_check.json and the cca_lsi run (the scripts fit it separately); learned vs CCA label transfer; VICReg effective rank; label construction from joint RNA+ATAC clustering; no per-class F1 for transfer.

## Training log (per run)

| run | epochs | stop | train pairs | seconds | config hash |
|---|---|---|---|---|---|
| tt_infonce_lsi_s0 | 150 | early_stopping (no validation improvement for 25 epochs) | 6548 | 5.6 | ab284e9efb6681ff |
| tt_infonce_lsi_s1 | 138 | early_stopping (no validation improvement for 25 epochs) | 6548 | 4.7 | 529ffd9834046a29 |
| tt_infonce_lsi_s2 | 134 | early_stopping (no validation improvement for 25 epochs) | 6548 | 4.0 | a71fc90693dba983 |
| tt_vicreg_lsi_s0 | 202 | early_stopping (no validation improvement for 25 epochs) | 6548 | 5.9 | 39df26c8b35d829b |
| tt_vicreg_lsi_s1 | 190 | early_stopping (no validation improvement for 25 epochs) | 6548 | 5.4 | ac677d8eb831f917 |
| tt_vicreg_lsi_s2 | 241 | early_stopping (no validation improvement for 25 epochs) | 6548 | 6.6 | 610efb7a968967c2 |

Versions: python 3.13.5, torch 2.14.0, numpy 2.5.3, scikit-learn 1.9.1.
