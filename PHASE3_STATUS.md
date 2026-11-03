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

ID3 and CART need all 1.78M rows in RAM, so they have no results yet. They run on Word2Vec and PV-DBOW, the two best text features.

1. Pull this repository.
2. Copy into `data/clean/`: `train.parquet`, `validation.parquet`, `test.parquet`, `preprocessing_params.json`, `w2v/train.npy`, `w2v/validation.npy`, `w2v/test.npy`, and `doc2vec_dbow/train.npy`, `doc2vec_dbow/validation.npy`, `doc2vec_dbow/test.npy`. Moaz has to upload the three PV-DBOW files first, because they are not on Drive yet.
3. Install: `pip3 install numpy pandas pyarrow scikit-learn`
4. From the project folder, run `bash models/run_trees.sh`. If memory runs out, use `JOBS=3 bash models/run_trees.sh`.
5. Send back `outputs/decision_tree_*.json`.

Run time on the full data is not measured yet. Finished results are skipped, so the run can be stopped and restarted.

## Changes from the handout

- Result files are named `outputs/<model>_<features>_<label>.json`, because each model runs on two feature sets.
- Doc2Vec vectors are inferred for all three splits, including training.
- Both Doc2Vec modes were run. PV-DBOW replaces PV-DM for the remaining models.
- Logistic regression picks `C` on a 200,000-row sample, then refits on all rows, instead of a 5-fold search.
- Trees use balanced class weights, like the other models, and pick depth on a 200,000-row sample.
- Linear regression and Naive Bayes each also try one balancing option, kept if validation F1 is better.
- The perceptron reports ROC-AUC from its decision value.

## Models not run

- Bayesian network: skipped on purpose, and explained in the report. It needs a known structure and discrete variables; our 133 features are continuous and have no causal meaning. Naive Bayes covers the Bayesian family.

## Notes for the ACM report

Write these into the report:

- **Bayesian network, not run.** A Bayesian network needs a known structure, built from causal relationships or domain knowledge, and it stores a probability table for each node (Lecture 11). Our 133 features do not fit this. The 100 text dimensions and the scaled statistics are continuous, and embedding dimensions have no causal meaning, so no structure can be drawn from domain knowledge. Learning a structure without that knowledge depends heavily on the feature ordering, and the probability tables would first require binning every feature, which discards information. Naive Bayes, the special case with a fixed structure that treats features as independent given the label, was run instead and represents the Bayesian family.
- **Doc2Vec.** Follows the Phase 2 feedback. Compare Word2Vec, PV-DBOW and PV-DM (table above). Vectors were inferred the same way for all splits, because stored training vectors differ from inferred ones (mean cosine 0.87 for PV-DM, 0.94 for PV-DBOW).
- **Changes from the Phase 1 plan.** TF-IDF was removed in Phase 2, and DistilBERT is not used because only course models are allowed.
- **Tree substitutions.** scikit-learn has no true ID3 or C4.5, so state what stands in for each. ID3 is an entropy tree without pruning.
- **Linear regression.** Fitted on the 0/1 label and cut at 0.5: a linear probability model, not regression on the raw score.
- **Sampling.** State the rows each model used. `n_train` and `sampled` are in every result file.
- **Low F1 on rare labels.** Models use the default decision cut, and balanced weights favour recall. Report ROC-AUC beside F1.
- **Phase 1 commitments.** Repeat the evaluation at label thresholds 0.4 and 0.6, and run the identity-subgroup check on the top 3.
- **Changes from the handout.** Listed in this file.

## Open points

- F1 is low on the rare labels. Models use the default decision cut, and balanced weights favour recall.
- Kareem's 6 models are not run yet. They should use the same two feature sets, Word2Vec and PV-DBOW.
