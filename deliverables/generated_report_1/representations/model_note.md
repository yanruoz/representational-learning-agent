# Learned representation: Two-tower rna/atac on LSI (InfoNCE) (`tt_infonce_lsi`)

Method: two_tower; loss: infonce; encoders: rna lsi, atac lsi.

Files: embeddings_rna.npy, embeddings_atac.npy (float32, one row per sample, rows aligned with embedding_index.csv; the split column says which samples were held out).

Load: `np.load('embeddings_<modality>.npy')`; L2-normalize rows for cosine similarity.

Why this one: The primary task is matching ATAC-only cells to RNA cells. On 1927 held-out cells the InfoNCE two-tower model on LSI features has FOSCTTM 0.01059 (SD 0.00003 over 3 seeds) against 0.01522 for CCA on the same LSI features, a gap far larger than the seed spread, and recall@1 0.17313 (SD 0.0024) against 0.1287. On the second task, kNN label transfer, it is not worse than CCA (macro-F1 0.89163 vs 0.8886, CCA is a single deterministic fit). It therefore serves both tasks at least as well as the linear baseline and better on the primary one.
