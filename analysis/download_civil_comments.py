"""
download_civil_comments.py -- fetch google/civil_comments into the project folder.

WHY YOU HAVE TO RUN THIS YOURSELF
    huggingface.co is blocked by the network policy on the machines Claude can
    reach, so the download has to happen from your own Windows machine. This
    script does the whole thing in one step.

WHAT YOU GET (422 MB total, four Parquet files)
    data/civil_comments/data/train-00000-of-00002.parquet   194 MB
    data/civil_comments/data/train-00001-of-00002.parquet   187 MB
    data/civil_comments/data/validation-00000-of-00001.parquet  21 MB
    data/civil_comments/data/test-00000-of-00001.parquet     20.8 MB

    Splits: train 1,804,874 / validation 97,320 / test 97,320 rows.
    Columns: text, plus seven float scores in [0, 1] -- toxicity,
    severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit.
    Each score is the fraction of annotators who said the attribute applied.
    Licence: CC0-1.0.

    IMPORTANT: this Hugging Face version does NOT include the identity columns
    (male, female, muslim, black, ...). Those live in the Kaggle release of
    "Jigsaw Unintended Bias in Toxicity Classification" and in the
    CivilCommentsIdentities config of TensorFlow Datasets. If your project
    includes a bias audit, you need one of those instead of, or as well as,
    this one.

HOW TO RUN (Windows, from a terminal in the project folder)
    pip install -U huggingface_hub
    python analysis\\download_civil_comments.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
TARGET = os.path.join(PROJECT, "data", "civil_comments")

REPO = "google/civil_comments"
EXPECTED = [
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
    "data/validation-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
]


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Run:")
        print("    pip install -U huggingface_hub")
        return 1

    os.makedirs(TARGET, exist_ok=True)
    print(f"Downloading {REPO} into:")
    print(f"    {TARGET}")
    print("About 422 MB. This takes a few minutes on a normal connection.\n")

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=TARGET,
        allow_patterns=["data/*.parquet", "README.md"],
    )

    print("\nDownloaded files:")
    missing = []
    total = 0
    for rel in EXPECTED:
        p = os.path.join(TARGET, rel.replace("/", os.sep))
        if os.path.exists(p):
            size = os.path.getsize(p)
            total += size
            print(f"  {size:>12,} bytes  {rel}")
        else:
            missing.append(rel)
    print(f"  {total:>12,} bytes  TOTAL")

    if missing:
        print("\nMISSING (re-run the script, the download resumes):")
        for m in missing:
            print("  " + m)
        return 1

    print("\nDone. Next step:")
    print("    pip install pandas pyarrow")
    print("    python analysis\\civil_comments_stats.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
