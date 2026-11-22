# Deliverables: rocov2, generated report 2

Objective: Our archive has captions for every radiograph but expert labels for only a few. Learn image representations that identify the body region from the image alone when only a few labeled examples per region are available, and tell me whether using the captions during training helps.

| Required output | File |
|---|---|
| Learned multimodal representations | `representations/` (embeddings_image.npy, embedding_index.csv, model_note.md) |
| Quantitative evaluation results | `evaluation.csv` (also the Results section of the PDF) |
| Downstream analysis results | `downstream_results.csv` (also the Downstream use section) |
| Summary of the selected methods and preprocessing | `methods_summary.md` (also the Decisions section) |
| Final analysis report | `generated_report_2.pdf` |

`traceability.md` links every cited value and every number in the agent's text to the file and key it comes from.
All intermediate files: `outputs/rocov2/` (not in the archive; rebuilt by `./run.sh`).
