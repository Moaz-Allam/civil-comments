# Phase 3 status: Moaz's part

Numbers below are read from `outputs/*.json` (validation split, `toxicity` label). The full metrics for every result, including test, are in `outputs/summary.csv`.

## Finished

- All four non-tree models on all three feature sets, all 6 labels: 72 results.
- Shared code: `common/evaluate.py`, `common/sample.py`, `common/data.py`.
- Doc2Vec: `prep/07_doc2vec.py`, both modes finished. Vectors are in `data/clean/doc2vec/` (PV-DM) and `data/clean/doc2vec_dbow/` (PV-DBOW).
- Model scripts: `linear_regression`, `logistic_regression`, `perceptron`, `naive_bayes`, `decision_tree_id3`, `decision_tree_cart` (in `models/`).
- Results merge: `models/summarize.py`.

## Results so far

| Model | Features | Labels done | F1 | Precision | Recall | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| linear_regression | doc2vec | 6 of 6 | 0.312 | 0.198 | 0.731 | 0.810 |
| linear_regression | doc2vec_dbow | 6 of 6 | 0.346 | 0.222 | 0.788 | 0.853 |
| linear_regression | word2vec | 6 of 6 | 0.335 | 0.216 | 0.746 | 0.838 |
| logistic_regression | doc2vec | 6 of 6 | 0.313 | 0.199 | 0.730 | 0.812 |
| logistic_regression | doc2vec_dbow | 6 of 6 | 0.349 | 0.225 | 0.782 | 0.853 |
| logistic_regression | word2vec | 6 of 6 | 0.352 | 0.229 | 0.763 | 0.849 |
| naive_bayes | doc2vec | 6 of 6 | 0.203 | 0.122 | 0.610 | 0.663 |
| naive_bayes | doc2vec_dbow | 6 of 6 | 0.304 | 0.200 | 0.632 | 0.752 |
| naive_bayes | word2vec | 6 of 6 | 0.281 | 0.176 | 0.703 | 0.766 |
| perceptron | doc2vec | 6 of 6 | 0.227 | 0.139 | 0.626 | 0.699 |
| perceptron | doc2vec_dbow | 6 of 6 | 0.275 | 0.171 | 0.704 | 0.775 |
| perceptron | word2vec | 6 of 6 | 0.264 | 0.158 | 0.802 | 0.797 |

## Doc2Vec vs Word2Vec

Three text features were compared on the four finished models, 24 results each.

| Features | Best toxicity F1 | Mean ROC-AUC |
| --- | --- | --- |
| word2vec | 0.352 | 0.855 |
| doc2vec_dbow (PV-DBOW) | 0.349 | 0.856 |
| doc2vec (PV-DM) | 0.313 | 0.801 |

PV-DBOW and Word2Vec are about level, and both beat PV-DM. All three pass the leakage rule.

## Nothing is running now

All jobs on Moaz's laptop are finished.

## To run on Kareem's laptop

ID3 and CART need all 1.78M rows in RAM, so they have no results yet. Everything they need is ready.

1. Pull this repository.
2. Copy into `data/clean/`: `train.parquet`, `validation.parquet`, `test.parquet`, `preprocessing_params.json`, `w2v/train.npy`, `w2v/validation.npy`, `w2v/test.npy`, and `doc2vec/train.npy`, `doc2vec/validation.npy`, `doc2vec/test.npy`. Moaz has to upload the three Doc2Vec files first, because they are not on Drive yet.
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
