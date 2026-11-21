# DATA_DESCRIPTION: rocov2

## What this is
A subset of ROCOv2 (Radiology Objects in COntext, version 2; Rückert et al.,
Scientific Data 2024): radiology images taken from figures in PubMed Central
Open Access articles, each paired with its figure caption and a list of UMLS
medical concept codes (CUIs). Two modalities per sample: an image and a free-text
caption. Taken from the Hugging Face copy `eltorio/ROCOv2-radiology`, pinned to
revision `80ffeef4eb8d34d27cb5c2815305f1d8aee8a83c`.

The subset is the first 4 of the 27 official training shards plus the first of
the 6 official test shards, so the official train/test split is kept.

## Files
- file: train-00000-of-00027.parquet | url: https://huggingface.co/datasets/eltorio/ROCOv2-radiology/resolve/80ffeef4eb8d34d27cb5c2815305f1d8aee8a83c/data/train-00000-of-00027.parquet | sha256: 932e29780894f30227b01969632f459056dfacc9c2ecb3812f99cdacb31ff0a2
- file: train-00001-of-00027.parquet | url: https://huggingface.co/datasets/eltorio/ROCOv2-radiology/resolve/80ffeef4eb8d34d27cb5c2815305f1d8aee8a83c/data/train-00001-of-00027.parquet | sha256: 1c6fbe4b15a4f61444c0148f5ae47fd2e65224a062bed7d008a5d54f63ca588e
- file: train-00002-of-00027.parquet | url: https://huggingface.co/datasets/eltorio/ROCOv2-radiology/resolve/80ffeef4eb8d34d27cb5c2815305f1d8aee8a83c/data/train-00002-of-00027.parquet | sha256: 3dab4588ba81228c51f6b39f7e80e6344e067d0d9927973e58eeffe197f4d0b0
- file: train-00003-of-00027.parquet | url: https://huggingface.co/datasets/eltorio/ROCOv2-radiology/resolve/80ffeef4eb8d34d27cb5c2815305f1d8aee8a83c/data/train-00003-of-00027.parquet | sha256: f1eb6104002160e90f627897f2a12c4346ed462dce59d652bb11211c27c24c3c
- file: test-00000-of-00006.parquet | url: https://huggingface.co/datasets/eltorio/ROCOv2-radiology/resolve/80ffeef4eb8d34d27cb5c2815305f1d8aee8a83c/data/test-00000-of-00006.parquet | sha256: e6cb01b149566e971debf7d0bd7508125ae803821611511346b543bda7a3b875

## Structure
- Parquet files with four columns: `image` (encoded image bytes), `image_id`
  (e.g. `ROCOv2_2023_train_000001`), `caption` (free text), `cui` (list of
  UMLS concept codes).
- 10,539 pairs: 8,884 from the official training split and 1,655 from the
  official test split. The shard name records the split.
- About 2.4 GB in total.
- Images vary in size, aspect ratio, and color mode (most are grayscale scans
  saved as image files). Captions vary from a few words to several sentences.

## Labels
No single label column. Each row's `cui` list holds 1 to 3 UMLS concept codes.
The first is the manually curated imaging-modality concept; for plain X-ray
images, body-region and directionality concepts follow. In this copy the
modality codes never appear in any other position. A few rows have the string
`nan` instead of a modality code (no label).

Modality concepts:
- C0040405: X-ray computed tomography (CT)
- C1306645: plain radiography (X-ray)
- C0024485: magnetic resonance imaging (MRI)
- C0041618: ultrasonography
- C0002978: angiography
- C0032743: positron emission tomography (PET)

The loader turns these into a `modality` label (see `## Loading`).

## Known caveats
- Captions are written by article authors, not radiologists, and often describe
  findings from the surrounding paper. Many name the imaging modality
  explicitly ("CT of the chest ..."), so a modality label is easy to predict
  from text alone.
- The concept codes were partly extracted from the captions, so concept labels
  are not independent of the text.
- Pretrained biomedical image-text models were trained on PubMed Central
  figure-caption pairs. Some of these images may have been in their training
  data, which can inflate zero-shot results.
- Images are figure panels from papers and can contain arrows, letters, and
  other annotations.
- License: CC BY-NC-SA 4.0 (non-commercial).
