# Phase 2: data preparation, cleaning and feature generation

Every number and figure in `Phase2_Data_Preparation_Report.docx` comes from the
scripts in this folder.

## Run order

    pip install -U huggingface_hub pandas pyarrow scipy scikit-learn gensim matplotlib
    python prep/00_download.py            # ~305 MB, SHA-256 checked
    python prep/01_load.py                # ~2 min
    python prep/02_feature_analysis.py    # ~3 min
    python prep/03_clean_preprocess.py    # ~5 min
    python prep/04_word2vec.py            # ~15 min
    python prep/05_correlation_checks.py  # ~2 min
    python prep/06_make_figures.py

Tested on Windows 11, Python 3.13, 7.4 GB RAM. The scripts keep text in Arrow
memory and work in chunks so they fit in that.

| Step | What it does | Main output |
|---|---|---|
| `00_download.py` | Fetches the full 45-column release (the three Kaggle "Jigsaw Unintended Bias" files, via the Hugging Face mirror `shuttie/jigsaw-unintended-bias`) and checks every SHA-256 | `data/jigsaw_unintended_bias/` |
| `01_load.py` | Loads the three files as train / validation / test into one table | `data/interim/civil_comments_full.parquet`, `out/01_load.json` |
| `02_feature_analysis.py` | Type, missing values, unique values, distribution and label relation of every column (train only) | `out/02_analysis.json` |
| `03_clean_preprocess.py` | Text cleaning, duplicate and leakage removal (train only), labels, feature selection, encoding and scaling | `data/clean/*.parquet`, `data/clean/preprocessing_params.json`, `out/03_cleaning.json` |
| `04_word2vec.py` | Skip-gram Word2Vec trained on train; mean-pooled, standardised document vectors | `data/clean/w2v/`, `out/04_word2vec.json` |
| `05_correlation_checks.py` | Distributions before/after scaling, feature-to-feature correlation of all 133 final features, correlation of every final feature with every target; fails if any reaches abs(r) >= 0.85 | `out/05_correlations.json` |
| `06_make_figures.py` | All report figures | `figures/*.png` |
| `common.py`, `textfeat.py` | Paths, column groups, the 15 text statistics | |
| `build_report.js` | Writes the report .docx from `out/` and `figures/` (`npm install docx`, then `node prep/build_report.js`) | |

`out/*_output.txt` are the console logs of the runs used in the report.

## The cleaned dataset (`data/clean/`)

| File | Rows | Content |
|---|---|---|
| `train.parquet` | 1,777,414 | one row per comment |
| `validation.parquet` | 97,320 | the Kaggle public test set |
| `test.parquet` | 97,320 | the Kaggle private test set |
| `w2v/{split}.npy` | same | float32, 100 standardised Word2Vec dimensions, rows in parquet order |
| `sample_train_1000.csv` | 1,000 | first training rows, for a quick look |
| `preprocessing_params.json` | | every fitted parameter (clips, means, stds, publication categories) |
| `w2v/word2vec.kv`, `w2v/scaler_*.npy` | | the fitted Word2Vec model and its scaler |

Parquet columns:

- `id`, `text` (cleaned, case kept), `text_bow` (normalised, the Word2Vec input)
- seven scores (`toxicity` ... `sexual_explicit`) and seven 0/1 labels `y_*` at 0.5; targets are the six without `severe_toxicity`
- `n_annotators`: kept for reference, never an input
- `f_*`: the 33 scaled tabular input features (12 text statistics, `f_is_reply`, 20 publication one-hot columns)
- `identity_annotated` + 24 identity scores (NaN when not annotated): bias audit only, never inputs
- `publication_id`, `created_date`, `rating`: metadata, never inputs

Load example:

```python
import numpy as np, pandas as pd
tr = pd.read_parquet("data/clean/train.parquet")
X = np.hstack([tr.filter(like="f_").to_numpy(), np.load("data/clean/w2v/train.npy")])
y = tr[["y_toxicity", "y_insult", "y_identity_attack", "y_obscene", "y_threat", "y_sexual_explicit"]]
```
