"""
00_download.py -- fetch the full 45-column Civil Comments release (the three
Kaggle "Jigsaw Unintended Bias in Toxicity Classification" files, from the
Hugging Face mirror shuttie/jigsaw-unintended-bias, because Kaggle requires a
signed-in account) and check each file's SHA-256 against the value Hugging Face
publishes for it.

    data/jigsaw_unintended_bias/data/   train.csv.gz, test_public_expanded.csv.gz,
                                        test_private_expanded.csv.gz

The Xet transfer backend stalled on this machine with files at full size but
missing chunks, so it is switched off and plain HTTP is used instead.

Run:  python prep/00_download.py
"""
import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, sha256

FILES = [
    ("shuttie/jigsaw-unintended-bias", "jigsaw_unintended_bias", "data/train.csv.gz",
     "82d217ec5c09cf25703fec20726e40def415367f37a8ea097976dacbd6a11260"),
    ("shuttie/jigsaw-unintended-bias", "jigsaw_unintended_bias", "data/test_public_expanded.csv.gz",
     "a5a8fba390af3f17639b1bcef9b1c1b38373d3e45728f7eb814bfe5f3ea1c3d2"),
    ("shuttie/jigsaw-unintended-bias", "jigsaw_unintended_bias", "data/test_private_expanded.csv.gz",
     "994cda8329554422d63a6691b66dcce838bc596f8975540f8e2d8b53cd993482"),
]


def main():
    from huggingface_hub import hf_hub_download
    bad = 0
    for repo, folder, path, expected in FILES:
        local = os.path.join(DATA, folder, path.replace("/", os.sep))
        if not (os.path.exists(local) and sha256(local) == expected):
            hf_hub_download(repo, path, repo_type="dataset", local_dir=os.path.join(DATA, folder))
        got = sha256(local)
        ok = got == expected
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {os.path.getsize(local):>12,}  {repo}/{path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
