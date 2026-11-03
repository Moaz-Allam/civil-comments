"""Decision tree, ID3 entry: entropy splits, no pruning (scikit-learn has no true ID3)."""
from tree_common import cli

if __name__ == "__main__":
    cli("decision_tree_id3", "entropy")
