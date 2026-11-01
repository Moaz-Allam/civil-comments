#!/bin/bash
# PV-DBOW comparison: builds the features, then runs the four chunked models on them.
export PYTHONIOENCODING=utf-8
python -u prep/07_doc2vec.py --dm 0 --out doc2vec_dbow > prep/out/07_doc2vec_dbow_output.txt 2>&1 || { echo "FAILED doc2vec"; exit 1; }
python -m common.data
bash models/run_moaz_linear.sh doc2vec_dbow
