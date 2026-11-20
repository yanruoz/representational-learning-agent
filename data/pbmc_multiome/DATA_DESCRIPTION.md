# DATA_DESCRIPTION: pbmc_multiome

## What this is
Peripheral blood mononuclear cells (PBMCs) from one healthy donor, profiled with
10x Genomics Single Cell Multiome ATAC + Gene Expression (10x sample
"pbmc_granulocyte_sorted_10k"). Each cell has two modalities measured in the
same nucleus: gene expression (RNA) and chromatin accessibility (ATAC). The
files are the processed copies distributed with the GLUE paper (Cao and Gao,
Nature Biotechnology 2022), one AnnData file per modality.

## Files
- file: pbmc10k_RNA.h5ad | url: https://download.gao-lab.org/GLUE/dataset/10x-Multiome-Pbmc10k-RNA.h5ad | md5: e48f7b4d8d2755df943ceb470003737a
- file: pbmc10k_ATAC.h5ad | url: https://download.gao-lab.org/GLUE/dataset/10x-Multiome-Pbmc10k-ATAC.h5ad | md5: 387ac0687e32967f29fd3a31bb402bf3

## Structure
- 9,631 cells, the same cells in the same order in both files (`obs_names` are
  10x barcodes and match across files).
- RNA: 9,631 cells x 29,095 genes. Values are raw UMI counts (integer-valued,
  stored as float32), about 93% zeros. `var` includes gene coordinates and a
  precomputed `highly_variable` flag.
- ATAC: 9,631 cells x 107,194 peaks. Values are raw fragment counts per peak
  (integer-valued, small: maximum 64 in the first 200 cells), about 93% zeros.
  Peaks are named `chrN:start-end`.
- Total size about 200 MB.

## Labels
`obs['cell_type']`: 19 PBMC cell types, identical in both files. Strongly
imbalanced: the largest class has 2,554 cells and the two smallest have 17 and
12 cells.

## Known caveats
- The `cell_type` labels were assigned by the original authors from a joint
  RNA+ATAC (weighted nearest neighbor) clustering followed by annotation. They
  are not independent ground truth, and because they used both modalities they
  may favor joint representations over single-modality ones.
- `obs` also holds columns derived from that same clustering and from both
  modalities (`seurat_clusters`, `wsnn_res.0.8`, `sub.cluster`, `SCT.weight`,
  `ATAC.weight`, per-modality count totals). Using any of them as model input
  leaks label information.
- ATAC peak counts are much sparser per feature than RNA and are usually treated
  as near-binary accessibility.
- Classes with fewer than about 20 cells give unstable per-class metrics.
- One donor and one batch: no batch effect to correct, and no way to test
  generalization across donors.
