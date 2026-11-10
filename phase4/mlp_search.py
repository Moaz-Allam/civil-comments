"""
Phase 4 MLP search (PyTorch, made for a Colab GPU; runs on CPU too, slowly).

Every setting is scored by 5-fold cross-validation on the pool (train + validation), grouped by
article. Selection metric: mean PR-AUC (average precision) over the 6 labels on out-of-fold scores.
Each finished fold is appended to results.jsonl, so a cut-off Colab session resumes where it stopped.

    python mlp_search.py --pack /content/pack --out /content/drive/MyDrive/phase4 stage1
    python mlp_search.py ... focal         # focal gamma x alpha at the base architecture (joins stage 1)
    python mlp_search.py ... stage2        # needs stage 1 finished
    python mlp_search.py ... separate      # six single-label networks on the best setting
    python mlp_search.py ... final         # 5 seeds on the whole pool, cuts from out-of-fold scores, test once
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS, PATIENCE, STOP_SHARE = 30, 3, 0.05

BASE = {"text": "both", "hidden": [256, 128], "activation": "relu", "dropout": 0.2, "lr": 1e-3,
        "batch": 2048, "weight_decay": 0.0, "loss": "bce", "gamma": 0.0, "alpha": None, "identity_weight": 0.0,
        "outputs": "joint"}


# ---------------------------------------------------------------- search spaces
def stage1_settings(n=15, seed=0):
    """Loss, architecture and text features; everything else at BASE."""
    losses = [{"loss": "bce"}, {"loss": "wbce"}] + [{"loss": "focal", "gamma": g, "alpha": a}
                                                   for g in [1.0, 2.0, 3.0] for a in [0.25, "balanced"]]
    hidden = [[256], [512], [256, 128], [512, 256], [512, 256, 128]]
    texts = ["word2vec", "doc2vec_dbow", "both"]
    grid = [{**l, "hidden": h, "text": t} for l, h, t in itertools.product(losses, hidden, texts)]
    random.Random(seed).shuffle(grid)
    must = [{"loss": "bce"}, {"loss": "wbce"}, {"loss": "focal", "gamma": 2.0, "alpha": 0.25}]
    must = [{**m, "hidden": BASE["hidden"], "text": BASE["text"]} for m in must]   # every loss seen at BASE
    picked = must + [g for g in grid if g not in must][:n - len(must)]
    return [{**BASE, **p} for p in picked]


def stage2_settings(best, n=10, seed=0):
    """Activation, dropout, learning rate, batch size, L2 and identity weights around the stage-1 winner."""
    space = {"activation": ["relu", "leaky_relu", "tanh", "sigmoid"], "dropout": [0.0, 0.1, 0.3],
             "lr": [3e-4, 1e-3, 3e-3], "batch": [1024, 4096], "weight_decay": [0.0, 1e-5, 1e-4]}
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n - 2:
        s = {**best, **{k: rng.choice(v) for k, v in space.items()}}
        key = json.dumps(s, sort_keys=True)
        if key not in seen and s != best:
            seen.add(key)
            out.append(s)
    return out + [{**best, "identity_weight": w} for w in [1.0, 3.0]]


# ---------------------------------------------------------------- data
class Pack:
    def __init__(self, path, limit=0):
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        load = lambda n: np.load(os.path.join(path, n), mmap_mode="r")
        keep = np.sort(np.random.default_rng(0).choice(self.meta["rows"]["pool"], limit, replace=False)) if limit else slice(None)
        keep_t = slice(0, limit // 10) if limit else slice(None)
        self.X, self.y, self.fold = [np.array(load(n)[keep]) for n in ["X_pool.npy", "y_pool.npy", "fold_pool.npy"]]
        self.ident = np.asarray(load("ident_pool.npy")[keep]).astype(np.float32)
        self.Xt, self.yt = np.array(load("X_test.npy")[keep_t]), np.array(load("y_test.npy")[keep_t])
        self.ident_t = np.asarray(load("ident_test.npy")[keep_t]).astype(np.float32)
        self.mention = (np.nan_to_num(self.ident) >= 0.5).any(axis=1)
        self.Xg = torch.from_numpy(self.X).to(DEV)            # float16 on the GPU, cast per batch
        self.yg = torch.from_numpy(self.y).to(DEV)

    def cols(self, text):
        c = self.meta["columns"]
        keep = list(range(*c["tabular"]))
        if text in ("word2vec", "both"):
            keep += list(range(*c["word2vec"]))
        if text in ("doc2vec_dbow", "both"):
            keep += list(range(*c["doc2vec_dbow"]))
        return torch.tensor(keep, device=DEV)


# ---------------------------------------------------------------- model and losses
ACT = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_out, activation, dropout, prior):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), ACT[activation](), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_out)
        with torch.no_grad():                       # prior init (Lin et al. 2017): start at each label's rate
            self.head.bias.copy_(torch.log(prior / (1 - prior)))

    def forward(self, x):
        return self.head(self.body(x))


def loss_fn(logits, y, w, s, rates):
    y = y.float()
    if s["loss"] == "focal":
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = p * y + (1 - p) * (1 - y)
        a = (1 - rates) if s["alpha"] == "balanced" else torch.full_like(rates, s["alpha"])
        at = a * y + (1 - a) * (1 - y)
        per = at * (1 - pt).clamp(min=1e-6) ** s["gamma"] * ce      # clamp: gamma < 1 has an infinite gradient at pt = 1
    else:
        pos_weight = (1 - rates) / rates if s["loss"] == "wbce" else None
        per = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return (per.mean(dim=1) * w).sum() / w.sum()


# ---------------------------------------------------------------- training
def row_weights(pack, idx, s):
    w = np.ones(len(idx), dtype=np.float32)
    if s["identity_weight"]:
        w += s["identity_weight"] * pack.mention[idx]
    return torch.from_numpy(w).to(DEV)


def predict(model, X, cols, batch=65536):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(X), batch):
            xb = X[b:b + batch] if torch.is_tensor(X) else torch.from_numpy(X[b:b + batch]).to(DEV)
            out.append(torch.sigmoid(model(xb[:, cols].float())).cpu())
    return torch.cat(out).numpy()


def train(pack, idx, s, labels, seed):
    """Fit on pool rows idx; early stopping on a random 5% of them, by mean PR-AUC."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    n_stop = int(len(idx) * STOP_SHARE)
    stop, fit = idx[:n_stop], idx[n_stop:]
    cols = pack.cols(s["text"])
    lab = torch.tensor(labels, device=DEV)
    rates = pack.yg[torch.from_numpy(fit).to(DEV)][:, lab].float().mean(0).clamp(1e-4, 1 - 1e-4)
    model = MLP(len(cols), s["hidden"], len(labels), s["activation"], s["dropout"], rates).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    fit_g, w_all = torch.from_numpy(fit).to(DEV), row_weights(pack, fit, s)
    best, best_state, bad, epochs = -1.0, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(fit_g), device=DEV)
        for b in range(0, len(perm), s["batch"]):
            j = perm[b:b + s["batch"]]
            rows = fit_g[j]
            logits = model(pack.Xg[rows][:, cols].float())
            loss = loss_fn(logits, pack.yg[rows][:, lab], w_all[j], s, rates)
            opt.zero_grad()
            loss.backward()
            opt.step()
        p = predict(model, pack.Xg[torch.from_numpy(stop).to(DEV)], cols)
        score = np.mean([average_precision_score(pack.y[stop, l], p[:, k]) for k, l in enumerate(labels)])
        epochs = epoch + 1
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, epochs


# ---------------------------------------------------------------- scoring
def label_scores(y, p):
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


def bias_scores(y, p, ident, min_n=500):
    """Borkan et al. 2019 on the toxicity label: subgroup AUC, BPSN, BNSP per identity mentioned >= min_n times."""
    out = {}
    m_all = np.nan_to_num(ident) >= 0.5
    for k in range(ident.shape[1]):
        m = m_all[:, k]
        if m.sum() < min_n or y[m].min() == y[m].max():
            continue
        bpsn = (~m & (y == 1)) | (m & (y == 0))
        bnsp = (~m & (y == 0)) | (m & (y == 1))
        out[k] = {"n": int(m.sum()), "subgroup_auc": float(roc_auc_score(y[m], p[m])),
                  "bpsn_auc": float(roc_auc_score(y[bpsn], p[bpsn])), "bnsp_auc": float(roc_auc_score(y[bnsp], p[bnsp]))}
    return out


def best_cut(y, p):
    """Cut with the highest F1 (Lipton et al. 2014: chosen on out-of-fold scores, never on test)."""
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    k = np.arange(1, len(p) + 1)
    f1 = 2 * tp / (k + y.sum())
    i = int(np.argmax(f1))
    return float(p[order][i]), float(f1[i])


# ---------------------------------------------------------------- runs (resumable)
def key_of(s):
    return json.dumps(s, sort_keys=True)


def load_results(out):
    path = os.path.join(out, "results.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf8") if l.strip()]


def run_cv(pack, out, stage, s, labels=None, seed=0):
    """5-fold CV of one setting; saves each fold, then the out-of-fold scores."""
    labels = labels or list(range(len(pack.meta["targets"])))
    tag = f"{stage}|{key_of(s)}|{labels}"
    done = {r["fold"] for r in load_results(out) if r["tag"] == tag}
    oof_path = os.path.join(out, "oof", hashlib.md5(tag.encode()).hexdigest()[:16] + ".npy")
    os.makedirs(os.path.dirname(oof_path), exist_ok=True)
    oof = np.load(oof_path) if os.path.exists(oof_path) else np.full((len(pack.y), len(labels)), np.nan, np.float32)
    for f in range(pack.meta["folds"]):
        if f in done:
            continue
        t0 = time.time()
        tr, te = np.flatnonzero(pack.fold != f), np.flatnonzero(pack.fold == f)
        model, epochs = train(pack, tr, s, labels, seed + f)
        p = predict(model, pack.Xg[torch.from_numpy(te).to(DEV)], pack.cols(s["text"]))
        oof[te] = p
        np.save(oof_path, oof)
        r = {"tag": tag, "stage": stage, "setting": s, "labels": labels, "fold": f, "epochs": epochs,
             "seconds": round(time.time() - t0, 1),
             "scores": {pack.meta["targets"][l]: label_scores(pack.y[te, l], p[:, k]) for k, l in enumerate(labels)}}
        with open(os.path.join(out, "results.jsonl"), "a", encoding="utf8") as fh:
            fh.write(json.dumps(r) + "\n")
        print(f"{stage} fold {f} {r['seconds']}s epochs {epochs} mean PR-AUC "
              f"{np.mean([v['pr_auc'] for v in r['scores'].values()]):.4f}", flush=True)
    return oof


def summary(out, stage):
    """Mean PR-AUC over folds and labels for every finished setting of a stage."""
    rows = {}
    for r in load_results(out):
        if r["stage"] == stage:
            rows.setdefault(r["tag"], []).append(r)
    table = []
    for tag, rs in rows.items():
        if len(rs) == 5:
            table.append({"setting": rs[0]["setting"], "labels": rs[0]["labels"],
                          "mean_pr_auc": float(np.mean([np.mean([v["pr_auc"] for v in r["scores"].values()]) for r in rs])),
                          "std_pr_auc": float(np.std([np.mean([v["pr_auc"] for v in r["scores"].values()]) for r in rs], ddof=1)),
                          "mean_roc_auc": float(np.mean([np.mean([v["roc_auc"] for v in r["scores"].values()]) for r in rs]))})
    return sorted(table, key=lambda t: -t["mean_pr_auc"])


def save(out, name, obj):
    with open(os.path.join(out, name), "w", encoding="utf8") as f:
        json.dump(obj, f, indent=1)
    print("wrote", name, flush=True)


def stage1(pack, out):
    for s in stage1_settings():
        run_cv(pack, out, "stage1", s)
    save(out, "stage1_summary.json", summary(out, "stage1"))


def focal_check(pack, out):
    """Fair focal test at the stage-1 base architecture and text; saved as stage 1 so stage 2 starts from the overall winner."""
    for g in [0.5, 1.0, 2.0, 3.0]:
        for a in [0.25, 0.5]:
            run_cv(pack, out, "stage1", {**BASE, "loss": "focal", "gamma": g, "alpha": a})
    save(out, "stage1_summary.json", summary(out, "stage1"))


def stage2(pack, out):
    best = summary(out, "stage1")[0]["setting"]
    oofs = {}
    for s in [best] + stage2_settings(best):
        oofs[key_of(s)] = run_cv(pack, out, "stage2", s)
    table = summary(out, "stage2")
    tox = pack.meta["targets"].index("toxicity")
    for t in table:                                # bias metrics on out-of-fold toxicity scores
        b = bias_scores(pack.y[:, tox], oofs[key_of(t["setting"])][:, tox], pack.ident)
        t["bias"] = {pack.meta["identities"][k]: v for k, v in b.items()}
        t["mean_bpsn"] = float(np.mean([v["bpsn_auc"] for v in b.values()]))
    save(out, "stage2_summary.json", table)


def separate(pack, out):
    best = summary(out, "stage2")[0]["setting"]
    joint = summary(out, "stage2")[0]
    per = {}
    for l, name in enumerate(pack.meta["targets"]):
        run_cv(pack, out, "separate", {**best, "outputs": "separate"}, labels=[l])
        per[name] = next(t for t in summary(out, "separate") if t["labels"] == [l])["mean_pr_auc"]
    save(out, "separate_summary.json", {"joint_mean_pr_auc": joint["mean_pr_auc"], "separate_pr_auc": per,
                                         "separate_mean_pr_auc": float(np.mean(list(per.values())))})


def final(pack, out, seeds=5):
    sep = json.load(open(os.path.join(out, "separate_summary.json")))
    best = summary(out, "stage2")[0]["setting"]
    use_separate = sep["separate_mean_pr_auc"] > sep["joint_mean_pr_auc"]
    targets = pack.meta["targets"]
    groups = [[l] for l in range(len(targets))] if use_separate else [list(range(len(targets)))]
    stage = "separate" if use_separate else "stage2"
    s = {**best, "outputs": "separate" if use_separate else "joint"}
    # cuts from out-of-fold scores of the chosen setting
    oof = np.column_stack([run_cv(pack, out, stage, s, labels=g) for g in groups]) if use_separate \
        else run_cv(pack, out, stage, s)
    cuts = {t: best_cut(pack.y[:, l], oof[:, l]) for l, t in enumerate(targets)}
    # 5 seeds on the whole pool, scores averaged
    all_rows = np.arange(len(pack.y))
    p_test = np.zeros((len(pack.yt), len(targets)), np.float32)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    for seed in range(seeds):
        for g in groups:
            model, _ = train(pack, all_rows, s, g, 100 + seed)
            torch.save(model.state_dict(), os.path.join(out, "models", f"mlp_seed{seed}_{'-'.join(map(str, g))}.pt"))
            p_test[:, g] += predict(model, pack.Xt, pack.cols(s["text"])) / seeds
    np.save(os.path.join(out, "test_scores.npy"), p_test)
    res = {"setting": s, "seeds": seeds, "cuts": {t: c[0] for t, c in cuts.items()},
           "oof_best_f1": {t: c[1] for t, c in cuts.items()}, "test": {}}
    for l, t in enumerate(targets):
        y, p = pack.yt[:, l], p_test[:, l]
        pred = (p >= cuts[t][0]).astype(int)
        tp, fp, fn, tn = [int(((pred == a) & (y == b)).sum()) for a, b in [(1, 1), (1, 0), (0, 1), (0, 0)]]
        res["test"][t] = {**label_scores(y, p), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                          "f1": 2 * tp / max(2 * tp + fp + fn, 1), "accuracy": (tp + tn) / len(y)}
    tox = targets.index("toxicity")
    res["test_bias"] = {pack.meta["identities"][k]: v for k, v in
                        bias_scores(pack.yt[:, tox], p_test[:, tox], pack.ident_t, min_n=100).items()}
    save(out, "final.json", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["stage1", "focal", "stage2", "separate", "final"])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use this many pool rows")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    a = ap.parse_args()
    MAX_EPOCHS = a.max_epochs
    os.makedirs(a.out, exist_ok=True)
    print("device", DEV, flush=True)
    pack = Pack(a.pack, a.limit)
    {"stage1": stage1, "focal": focal_check, "stage2": stage2, "separate": separate, "final": final}[a.step](pack, a.out)
