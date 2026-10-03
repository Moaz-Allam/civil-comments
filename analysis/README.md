# Analysis assets for the toxic-comment report

Every number in the report comes from one of the scripts here. Nothing is typed
in from memory. The project's dataset is **Civil Comments**; the Jigsaw Toxic
Comment set is kept as the smaller corpus to develop the pipeline against.

## Civil Comments (the project's dataset)

| File | What it is |
|---|---|
| `download_civil_comments.py` | Downloads `google/civil_comments` from Hugging Face into `../data/civil_comments` (422 MB of Parquet). **Run this on your own machine** — Hugging Face is not reachable from the machine the analysis ran on. |
| `civil_comments_hf_stats.py` | Derives every Civil Comments number in the report from the Hugging Face dataset server's published statistics, with the raw API responses recorded inline. Runs its own integrity checks and fails loudly if any of them break. |
| `cc_stats.json` | Output of the script above. The figures read from it, so text and figures cannot disagree. |
| `cc_stats_output.txt` | Console output of the run on 20 Sep 2026. |
| `make_cc_figures.py` | Draws Figures 1 and 2 from `cc_stats.json`. |
| `fig1_cc_labels_at_050.png` | Comments scoring >= 0.5 on each attribute (log axis — the counts span four orders of magnitude). |
| `fig2_cc_threshold.png` | How the number of "toxic" comments changes as the threshold moves from 0.1 to 0.9. |
| `civil_comments_stats.py` | Recomputes all of it from the downloaded Parquet files, and goes further than the histograms can — per-comment label combinations, co-occurrence, exact zero counts. **Run this once the download finishes.** |

### Why there are two Civil Comments scripts

`civil_comments_hf_stats.py` works without the data, because the dataset server
publishes per-column histograms over the full training split. Bin edges are
0, 0.1, ... 1.0, so the count at or above any tenth is a sum of bins.

That reading is checked, not assumed. An independent row-count query for
`toxicity >= 0.5` returns 144,334, which is exactly what the last five toxicity
bins sum to, and each of the seven histograms sums to 1,804,874. Both checks are
`assert` statements in the script.

`civil_comments_stats.py` is the better source once the Parquet files are on
disk. If the two ever disagree, trust the local one.

### Headline numbers (training split, 1,804,874 comments, threshold 0.5)

    toxicity          144,334   7.9969%
    insult            106,534   5.9026%
    identity_attack    13,410   0.7430%
    obscene             9,603   0.5321%
    sexual_explicit     4,686   0.2596%
    threat              4,280   0.2371%
    severe_toxicity        13   0.0007%

`severe_toxicity` is not a typo. At this threshold it cannot be trained on.

Moving the threshold from 0.4 to 0.6 changes the toxicity positive set from
201,476 to 96,410 — a factor of 2.09.

## Jigsaw Toxic Comment (the comparison corpus)

| File | What it is |
|---|---|
| `jigsaw_stats.py` | Computes every Jigsaw statistic from `train.csv`, `test.csv`, `test_labels.csv`. Prints file SHA-256 first. |
| `jigsaw_stats.json`, `stats_output.txt` | Its output. |
| `make_figures.py`, `fig1_label_distribution.png`, `fig2_labels_per_comment.png` | The Jigsaw figures. Not used in the current report — kept in case the comparison is wanted. |

Get the files from the Kaggle competition page after accepting the rules, then
`python jigsaw_stats.py <folder>`. These SHA-256 values identify the exact files
the numbers came from:

    train.csv        68,802,655 bytes  bd4084611bd27c939ba98e5e63bc3e5a2c1a4e99477dcba46c829e4c986c429d
    test.csv         60,354,593 bytes  c2513ce4abb98c4d1d216e3ca0d4377d57589a0989aa8c06a840509a16c786e8
    test_labels.csv   4,976,930 bytes  2a56dcbeba5c05f965a636f56cb5ae972bad60c3b952c239b49be18d7ab70f49

### An independent cross-check

Van Aken et al. (ALW2 2018) describe their Wikipedia corpus as 223,549 comments.
`jigsaw_stats.py` counts 159,571 training comments and 63,978 scored test
comments; 159,571 + 63,978 = 223,549. A 2018 paper and a script run in 2026
agreeing to the unit is good evidence that both the download and the counting
are right.

## Bibliography

`references_verified.bib` — entries that were each opened and checked against
the publisher's page, every one carrying a note saying what was confirmed.
The Borkan et al. 2019 entry used in the current report is not yet in this file.

## Order to run things

    pip install -U huggingface_hub pandas pyarrow matplotlib
    python analysis\download_civil_comments.py
    python analysis\civil_comments_stats.py
    python analysis\make_cc_figures.py
