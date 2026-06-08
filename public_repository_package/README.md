# Public reproducibility package

## Paper title
Leakage-safe and calibration-aware model-threshold selection for multi-sensor robotic-arm anomaly diagnosis

## Repository purpose
This package is a repository-ready release bundle for reproducing the aggregate tables, statistical comparisons, figures, LaTeX tables and supplementary files supporting the manuscript. It is intended for upload to GitHub, Gitee, Zenodo or OSF.

## What is included
- Public-release scripts for data adaptation, benchmark aggregation, threshold analysis, statistics, table generation and figure generation.
- Configuration files for datasets, split protocols, model settings and validation gates.
- Aggregated result tables, statistical comparison files, split summaries, threshold summaries, FAR/MDR sensitivity outputs, figure source maps and supplementary tables.
- Command manifests and environment notes.
- Dataset acquisition instructions and expected file trees for third-party datasets.

## What is not included
Third-party raw datasets and derivative processed datasets are not redistributed here unless their redistribution rights are explicitly confirmed by the authors. Use the acquisition instructions to download datasets from the official sources and place them locally.

## Exact reproduction scope
- Raw-data reproduction: supported through official download and data-placement instructions when redistribution is restricted or unclear.
- Processed-data reproduction: supported through adapter/preprocessing scripts and expected file-tree documentation.
- Aggregate-table reproduction: supported from released result summaries and command manifests.
- Figure/table reproduction: supported from figure source maps, released CSV summaries and manuscript table files.
- Statistical analysis reproduction: supported from released statistical comparison tables, cluster-bootstrap details, sign tests, effect sizes and wins/losses where available.

## External dataset requirements
IMAD-DS, RoAD, NIST UR and KUKA dataset access remains subject to the original dataset providers and licences. See dataset_acquisition/.

## Environment setup
Install dependencies listed in environment/requirements.txt or adapt environment/environment_notes.md to your platform.

## Command manifest overview
See command_manifests/ for reproduction commands and expected outputs.

## Citation
Cite the associated manuscript and the original dataset records. See CITATION.cff.

## License notes
A final repository license must be selected by the authors before upload. See LICENSE_TO_BE_SELECTED_BY_AUTHORS.txt and DATA_LICENSE_AUDIT.md.

## Contact
Corresponding author: Yapeng Wang. Repository URL: REPOSITORY_URL_TO_BE_FILLED_AFTER_PUBLIC_UPLOAD.
