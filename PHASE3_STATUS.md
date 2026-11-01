# Phase 3 status: Moaz's part

Snapshot of the work, including what is still running. Numbers below are read from `outputs/*.json` (validation split, `toxicity` label). The full metrics for every result, including test, are in `outputs/summary.csv`.

## Finished

- All four non-tree models on Word2Vec, all 6 labels.
- Shared code: `common/evaluate.py`, `common/sample.py`, `common/data.py`.
- Doc2Vec script: `prep/07_doc2vec.py`, tested on 50,000 comments.
- Model scripts: `linear_regression`, `logistic_regression`, `perceptron`, `naive_bayes`, `decision_tree_id3`, `decision_tree_cart` (in `models/`).
- Results merge: `models/summarize.py`.

## Results so far (Word2Vec features)

| Model | Features | Labels done | F1 | Precision | Recall | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| linear_regression | word2vec | 6 of 6 | 0.335 | 0.216 | 0.746 | 0.838 |
| logistic_regression | word2vec | 6 of 6 | 0.352 | 0.229 | 0.763 | 0.849 |
| naive_bayes | word2vec | 6 of 6 | 0.281 | 0.176 | 0.703 | 0.766 |
| perceptron | word2vec | 6 of 6 | 0.264 | 0.158 | 0.802 | 0.797 |

## Still running on Moaz's laptop

- Doc2Vec full run, about 45 minutes. Not finished, so there are no Doc2Vec results yet.
- When Doc2Vec finishes, `models/after_doc2vec.sh` runs the four non-tree models on it automatically.

## To run on Kareem's laptop

ID3 and CART need all 1.78M rows in RAM, so they have no results yet. Wait until Doc2Vec is finished, because the run covers both feature sets.

1. Pull this repository.
2. Copy into `data/clean/`: `train.parquet`, `validation.parquet`, `test.parquet`, `preprocessing_params.json`, `w2v/`, and `doc2vec/` (the `.npy` files).
3. Install: `pip3 install numpy pandas pyarrow scikit-learn`
4. From the project folder, run `bash models/run_trees.sh`. If memory runs out, use `JOBS=3 bash models/run_trees.sh`.
5. Send back `outputs/decision_tree_*.json`.

Run time on the full data is not measured yet. Finished results are skipped, so the run can be stopped and restarted.

## Changes from the handout

- Result files are named `outputs/<model>_<features>_<label>.json`, because each model runs on two feature sets.
- Doc2Vec vectors are inferred for all three splits, including training.
- Logistic regression picks `C` on a 200,000-row sample, then refits on all rows, instead of a 5-fold search.
- Trees use balanced class weights, like the other models, and pick depth on a 200,000-row sample.
- Linear regression and Naive Bayes each also try one balancing option, kept if validation F1 is better.
- The perceptron reports ROC-AUC from its decision value.

## Open points

- F1 is low on the rare labels. Models use the default decision cut, and balanced weights favour recall.
- The handout tabs for Kareem's 6 models are not run yet.
