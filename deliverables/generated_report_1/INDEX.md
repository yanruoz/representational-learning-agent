# Deliverables: pbmc_multiome, generated report 1

Objective: I have cells measured with RNA in one experiment and with ATAC in another. Learn a shared representation where a cell’s RNA and ATAC profiles can be matched, so cell-type annotations can be transferred from RNA-annotated cells to ATAC-only cells. Tell me how well matching works and how accurate the transferred labels are.

| Required output | File |
|---|---|
| Learned multimodal representations | `representations/` (embeddings_rna.npy, embeddings_atac.npy, embedding_index.csv, model_note.md) |
| Quantitative evaluation results | `evaluation.csv` (also the Results section of the PDF) |
| Downstream analysis results | `downstream_results.csv` (also the Downstream use section) |
| Summary of the selected methods and preprocessing | `methods_summary.md` (also the Decisions section) |
| Final analysis report | `generated_report_1.pdf` |

`traceability.md` links every cited value and every number in the agent's text to the file and key it comes from.
All intermediate files: `outputs/pbmc_multiome/` (not in the archive; rebuilt by `./run.sh`).
