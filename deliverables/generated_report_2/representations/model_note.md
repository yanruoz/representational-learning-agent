# Learned representation: Frozen BiomedCLIP (reference) (`bmc_frozen`)

Method: frozen_probe; loss: none (no training); encoders: image cache:biomedclip_frozen.

Files: embeddings_image.npy (float32, one row per sample, rows aligned with embedding_index.csv; the split column says which samples were held out).

Load: `np.load('embeddings_<modality>.npy')`; L2-normalize rows for cosine similarity.

Why this one: No learned model beats the frozen encoders on the primary probe. On 399 held-out X-ray images with 5 labels per region, the linear-probe macro-F1 of the caption-aligned DINOv2 model is 0.59943 (SD 0.00674 over 3 seeds) against 0.5983 for frozen DINOv2 without captions: no clear difference, given a label-draw SD of 0.0262. Frozen BiomedCLIP image features score 0.7322 at 5 per region and 0.8154 with all labels, higher than every model trained here, so it is the representation to export. It is a single deterministic run whose 0.7322 has no seed spread. Caveat: BiomedCLIP was pretrained on PubMed Central figure-caption pairs that may include these images, so its numbers may be inflated, and it already contains caption supervision from outside this archive.
