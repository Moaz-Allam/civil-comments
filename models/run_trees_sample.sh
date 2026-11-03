#!/bin/bash
# ID3 and CART on a 600,000-row sample, for a machine with less than 16 GB of RAM.
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1
mkdir -p outputs/logs
for m in decision_tree_cart decision_tree_id3; do
  python -u models/$m.py --final-rows 600000 --jobs 2 >> outputs/logs/$m.txt 2>&1 || echo "FAILED $m"
  echo "done $m $(date +%T)"
done
