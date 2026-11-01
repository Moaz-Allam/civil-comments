#!/bin/bash
# Waits for the Doc2Vec features, builds their cache, then runs the four chunked models on them.
export PYTHONIOENCODING=utf-8
until [ -f prep/out/07_doc2vec.json ]; do sleep 30; done
python -m common.data
bash models/run_moaz_linear.sh doc2vec
