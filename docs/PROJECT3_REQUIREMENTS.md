# Project 3: Building an AI Agent for Multimodal Representation Learning

In this project, you will build an AI agent that automatically performs multimodal representation learning for biomedical data. Unlike a traditional benchmarking study, your goal is to develop an automated system that determines how multiple data modalities should be integrated to learn meaningful joint representations for downstream analysis.

Given a multimodal dataset and a user objective, your agent should automatically determine an appropriate representation learning workflow, train or fine-tune suitable models, evaluate the learned representations, and generate a summary of its findings.

## Task

Your agent should automatically perform the following steps:

1. Inspect the available data modalities.
2. Recommend and perform appropriate preprocessing for each modality.
3. Select an appropriate multimodal representation learning strategy.
4. Select appropriate encoders and training objectives.
5. Train or fine-tune the selected model.
6. Evaluate the learned representations using appropriate quantitative or qualitative criteria.
7. Perform one or more downstream analyses (e.g., classification, clustering, retrieval, cross-modal prediction).
8. Produce a final analysis report summarizing the decisions and findings.

The objective is to build a system that can automatically determine how multimodal representations should be learned, rather than manually implementing a fixed pipeline.

## Representation Learning Methods

Your agent should be capable of using the representation learning methods covered in class. These may include, but are not limited to:

- Contrastive learning
- Self-supervised learning
- Supervised representation learning
- CLIP-style multimodal learning
- Fine-tuning pretrained multimodal foundation models

Your agent does not need to apply every method to every dataset. Instead, it should determine which approach is appropriate for the given data and objective, and explain its choices.

## Datasets

Evaluate your agent on at least two multimodal biomedical datasets. You may use your own data (ideally related to your dissertation), or select from the following publicly available datasets:

- Histopathology images + spatial transcriptomics (STimage-1K4M)
- Radiology images + text reports (ROCOv2)
- Single-cell RNA-seq + ATAC-seq (10x Genomics Multiome PBMC)
- Histopathology images + clinical variables (TCGA-BRCA)

## Deliverables

Submit the following materials through Canvas.

### Source code

Your code should execute the complete pipeline with a single command and require minimal manual intervention after providing a dataset and a user objective.

### Generated outputs

For each of the two chosen datasets, your agent should automatically generate:

- learned multimodal representations,
- quantitative evaluation results,
- downstream analysis results,
- a summary of the selected methods and preprocessing steps,
- a final analysis report.

The exact format of these outputs is up to you. The two reports should be named `generated_report_1` and `generated_report_2`.

### Report

Submit a PDF report named `report` of at most four pages (excluding references and appendices).

Your report should include:

- an overview of your system architecture,
- how the agent makes representation learning decisions,
- the tools or AI models used,
- experimental results on all datasets,
- strengths and limitations of your system.

The report should focus on the design and behavior of the AI agent rather than providing an exhaustive benchmark of individual multimodal representation learning methods.

## Evaluation

Projects will be evaluated based on the following criteria:

1. Degree of automation and autonomy of the representation learning pipeline.
2. Quality of the analysis and modeling decisions made by the agent.
3. Correct use of multimodal representation learning methods.
4. Quality of the learned representations.
5. Quality of the downstream analysis.
6. Quality of the generated reports and interpretations.
7. Robustness on different datasets.
8. Overall software design and reproducibility.

The emphasis of this project is on building an intelligent and reusable multimodal representation learning system, not on achieving the highest performance on a particular benchmark.

## Tools

You may use any publicly available software packages or AI models. Libraries such as PyTorch, TensorFlow, Hugging Face Transformers, OpenCLIP, OpenAI APIs, Anthropic APIs, LangChain, LangGraph, or other agent frameworks are all acceptable.

The specific implementation is entirely up to you.

## Submission

- **Due date:** November 30, 2026, 11:59 PM EST.
- Submit your report and source code through Canvas.
- **Late policy:** late submissions receive a 20% deduction from the total possible project score for each day (or portion of a day) that the submission is late. Submissions more than five days late receive no credit.
