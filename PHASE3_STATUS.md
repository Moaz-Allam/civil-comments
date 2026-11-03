# Phase 3 status

Phase 3 is finished. The report is `Phase3_Model_Comparison_Report.docx` / `.pdf` (due Wednesday 4 November 2026).

## What is done

- All 13 course models on all 6 labels: 180 result files in `outputs/`, merged into `outputs/summary.csv`.
  - Moaz: linear regression, logistic regression, perceptron, Naive Bayes (Word2Vec, PV-DBOW, PV-DM), ID3 and CART (Word2Vec, PV-DBOW).
  - Kareem: C4.5, random forest, linear SVM, RBF SVM, MLP, AdaBoost, KNN (Word2Vec, PV-DBOW).
- Bayesian network: not run, explained in the report (Section 3.14).
- Top 3 (chosen by the authors): RBF SVM, MLP, logistic regression.
- Variability study of the top 3: `models/top3.py`, results in `outputs/top3/`.
  - `cv`: 5-fold stratified CV on a 62,500-row pool per label, same folds for all three.
  - `seeds`: MLP with 5 seeds on all rows, RBF SVM on 5 different 50,000-row samples (toxicity, threat).
  - `thresholds`: toxicity label rebuilt at 0.4 / 0.5 / 0.6 (Phase 1 commitment).
  - `subgroups`: identity-group bias metrics (subgroup AUC, BPSN, BNSP) and the decision-cut sweep.
  - `tests`: bootstrap 95% intervals and McNemar tests on the test split.
- Chosen technique: **MLP** (Section 5 of the report).

## Rebuilding the report

```
python models/summarize.py
python report/collect.py
python report/make_figures.py
node report/build_report.js        # needs the "docx" npm package
```

Every number in the report is read from `report/out/numbers.json`, which `collect.py` builds from `outputs/`.

## Rows used per model

| Model | Training rows |
| --- | --- |
| Linear / logistic regression, perceptron, Naive Bayes, C4.5, linear SVM, MLP | 1,777,414 (all) |
| ID3, CART | 600,000 (sample; full data did not fit in 7.4 GB RAM) |
| Random forest, AdaBoost | 300,000 |
| KNN | 150,000 |
| RBF SVM | 50,000 |

ID3 and CART can be rerun on all rows on a 16 GB laptop with `bash models/run_trees.sh`; then rebuild the report.

## Changes from the handout

- Result files are named `outputs/<model>_<features>_<label>.json`, because each model runs on two feature sets.
- Doc2Vec vectors are inferred for all three splits, including training.
- Both Doc2Vec modes were run. PV-DBOW replaces PV-DM for the remaining models.
- Logistic regression picks `C` on a 200,000-row sample, then refits on all rows, instead of a 5-fold search.
- Trees use balanced class weights, like the other models, and pick depth on a 200,000-row sample.
- Linear regression and Naive Bayes each also try one balancing option, kept if validation F1 is better.
- The perceptron reports ROC-AUC from its decision value.

## Notes for Phase 4

- Choose the decision cut per label on validation (the MLP's best toxicity cut is 0.243, not 0.5).
- Average several MLP seeds; oversample rare positives (sklearn's MLP has no class weights).
- Keep checking identity-group bias: all three models rank clean comments that mention "white" or "black" too high.
- Keep logistic regression as the interpretable baseline.
