"""
Decision tree, CART entry (Lecture 10): binary splits chosen by the Gini index.

Run:  python models/decision_tree_cart.py [--features word2vec doc2vec]
"""
from tree_common import cli

if __name__ == "__main__":
    cli("decision_tree_cart", "gini")
