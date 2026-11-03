#!/bin/bash
# Kareem's models: decision_tree_c45, random_forest, svm, mlp, adaboost and knn.
#     bash models/run_kareem.sh            all six
#     bash models/run_kareem.sh knn svm    only these
#     FEATURES=doc2vec_dbow bash models/run_kareem.sh    one feature set
# Finished results are skipped, so it can be stopped and restarted.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
MODELS=${@:-svm random_forest decision_tree_c45 knn adaboost mlp}
FEATURES=${FEATURES:-word2vec doc2vec_dbow}
mkdir -p outputs/logs
python3 -c "import sys; sys.path[:0]=['models']; import model_common as m; m.require('$FEATURES'.split())"
for m in $MODELS; do
  echo "=== $m  $(date '+%H:%M')"
  python3 -u models/$m.py --features $FEATURES 2>&1 | tee -a outputs/logs/$m.txt
done
python3 models/summarize.py
