"""
Decision tree, ID3 entry (Lecture 9): splits chosen by entropy / information gain, no pruning.

scikit-learn has no true ID3. It builds binary trees on numeric thresholds, so
criterion="entropy" is the stand-in for ID3's splitting rule. Our features are continuous,
which ID3 itself cannot handle without binning them first (Lecture 10).

Run:  python models/decision_tree_id3.py [--features word2vec doc2vec]
"""
from tree_common import cli

if __name__ == "__main__":
    cli("decision_tree_id3", "entropy")
