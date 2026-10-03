"""
textfeat.py -- the hand-crafted text statistics, defined once so the analysis
(02) and the cleaning script (03) compute exactly the same thing.

All counts use Arrow's vectorised string kernels through pandas' "string[pyarrow]"
dtype, so 1.8 million comments take seconds rather than minutes. Arrow uses the
RE2 regex engine, which has no back-references, so "repeated letter" style
features are not expressible here and are left out.
"""
import numpy as np
import pandas as pd

# name -> (regex counted per comment, one-line meaning for the report)
COUNT_PATTERNS = {
    "n_words":        (r"\S+",                       "whitespace-separated tokens"),
    "n_letters":      (r"[A-Za-z]",                  "ASCII letters"),
    "n_upper":        (r"[A-Z]",                     "upper-case letters"),
    "n_caps_words":   (r"\b[A-Z]{3,}\b",             "fully capitalised words of 3+ letters (shouting)"),
    "n_exclaim":      (r"!",                         "exclamation marks"),
    "n_question":     (r"\?",                        "question marks"),
    "n_punct_runs":   (r"[!?]{2,}",                  "runs of 2+ '!' or '?'"),
    "n_masked":       (r"\w\*+\w|\w\*{2,}",          "asterisk-masked words such as f**k"),
    "n_urls":         (r"(?i)https?://|www\.",       "links"),
    "n_digits":       (r"[0-9]",                     "digits"),
    "n_newlines":     (r"\n",                        "line breaks"),
    "n_non_ascii":    (r"[^\x00-\x7F]",              "non-ASCII characters (accents, emoji, curly quotes)"),
}


def text_stats(text: pd.Series) -> pd.DataFrame:
    """Return one row of statistics per comment."""
    s = text.fillna("").astype("string[pyarrow]")
    out = pd.DataFrame(index=text.index)
    out["n_chars"] = s.str.len().astype("int32")
    for name, (pat, _) in COUNT_PATTERNS.items():
        out[name] = s.str.count(pat).astype("int32")
    letters = out["n_letters"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["upper_ratio"] = np.where(letters > 0, out["n_upper"] / letters, 0.0).astype("float32")
    out["mean_word_len"] = np.where(out["n_words"] > 0,
                                    out["n_chars"] / out["n_words"].clip(lower=1), 0.0).astype("float32")
    return out


DERIVED = {
    "n_chars":       "characters",
    "upper_ratio":   "upper-case letters / all letters",
    "mean_word_len": "characters / tokens",
}
