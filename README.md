# Toxic Comment Detection on Civil Comments

CSCE 3602 / DSCI 3415 Fundamentals of Machine Learning, The American University in Cairo.
Moaz Allam, Kareem Elhenawy.

A moderation aid that scores a comment on six toxicity attributes (toxicity, insult,
identity_attack, obscene, threat, sexual_explicit), trained on the Civil Comments corpus
(about 2 million comments).

## Contents

| Path | What it is |
|---|---|
| `Toxic_Comment_Detection_Report.docx` | Phase 1: literature review and dataset survey |
| `analysis/` | Phase 1 scripts and figures (dataset statistics) |
| `Phase2_Data_Preparation_Report.docx` / `.pdf` | Phase 2: feature analysis, cleaning, pre-processing, final features |
| `prep/` | Phase 2 pipeline: download, analysis, cleaning, Word2Vec, correlation checks, figures. See [prep/README.md](prep/README.md) |

## Data

The raw and cleaned data are not in this repository (the files are 0.3 to 1.3 GB, above
GitHub's limit). `prep/00_download.py` fetches the raw release, and the rest of the pipeline
rebuilds `data/clean/` from it. The cleaned dataset is also shared on Google Drive (link in
the Phase 2 report).

## Reproduce Phase 2

    pip install -U huggingface_hub pandas pyarrow scipy scikit-learn gensim matplotlib
    python prep/00_download.py
    python prep/01_load.py
    python prep/02_feature_analysis.py
    python prep/03_clean_preprocess.py
    python prep/04_word2vec.py
    python prep/05_correlation_checks.py
    python prep/06_make_figures.py
