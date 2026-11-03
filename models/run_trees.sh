#!/bin/bash
# ID3 and CART on all training rows (needs about 16 GB of RAM). Run: bash models/run_trees.sh
set -e
JOBS=${JOBS:-6}
export OMP_NUM_THREADS=1
mkdir -p outputs/logs
python3 -m common.data
for m in decision_tree_cart decision_tree_id3; do
  python3 -u models/$m.py --jobs "$JOBS" 2>&1 | tee -a outputs/logs/$m.txt
done
echo "Done. Copy outputs/decision_tree_*.json back to Moaz."
