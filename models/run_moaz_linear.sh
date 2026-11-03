#!/bin/bash
# Linear regression, Naive Bayes, perceptron and logistic regression: bash models/run_moaz_linear.sh word2vec
export PYTHONIOENCODING=utf-8
for m in linear_regression naive_bayes perceptron logistic_regression; do
  python -u models/$m.py --features $1 >> outputs/logs/${m}_$1.txt 2>&1 || echo "FAILED $m $1"
  echo "done $m $1 $(date +%T)"
done
