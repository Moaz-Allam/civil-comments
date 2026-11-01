#!/bin/bash
# ID3 and CART on every training row, for the 16 GB laptop. Run from the project folder:
#     bash models/run_trees.sh
# Needs in data/clean: the 3 parquet files, preprocessing_params.json, and train/validation/test.npy
# in both w2v/ and doc2vec/.
# Finished results are skipped, so it is safe to stop it and run it again.
# Lower JOBS if the laptop runs out of memory.
set -e
JOBS=${JOBS:-6}
export OMP_NUM_THREADS=1
mkdir -p outputs/logs
python3 -m common.data
for m in decision_tree_cart decision_tree_id3; do
  python3 -u models/$m.py --jobs "$JOBS" 2>&1 | tee -a outputs/logs/$m.txt
done
echo "Done. Copy outputs/decision_tree_*.json back to Moaz."
