"""TabFM transfer, leak-fixed per peer review (scratch/peer_review_1.md).

All five models answered the SAME 500 TriviaQA questions, so any pooled or
cross-model split that ignores question identity leaks question difficulty
from train to test. Fixes here:

  1. Question-disjoint folds everywhere: a question never appears on both
     sides of a split, within or across models.
  2. Normalization is train-side only. Two variants reported:
       strict   - held-out model normalized with averaged train-model stats
                  (zero target-side information; hard-rule-5 pure)
       unlabeled - held-out model normalized with its OWN feature stats from
                  train-fold questions only (deployment-realistic: you can
                  always collect unlabeled traffic; still label-free)
  3. Bootstrap 95% CIs (1000 resamples) on every AUC.

Tests:
  A. Pooled 4-Gemma CV, question-grouped folds
  B. Leave-one-Gemma-out, question-disjoint
  C. All Gemmas -> Qwen, question-disjoint

Run with the ufc_bet tabfm venv.
"""

import json
import os

import numpy as np

WEIGHTS = os.environ.get(
    "TABFM_WEIGHTS", "C:/Users/18632/Desktop/tabfm/classification/model.safetensors")
TRACE = "data/uncertainty_trivia_{}.jsonl"
GEMMAS = ["gemma-4-e4b-it", "gemma-4-12b-it", "huihui-gemma-4-12b-it-abliterated",
          "gemma-4-26b-a4b-it"]
QWEN = "qwen3.6-27b"
N_TRAJ = 16
SETS = ["logprob", "workspace", "combined"]

BASELINE = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob", "bl_answer_len"]
WS_SCALARS = ["ignition_frac", "ignition_depth", "mean_log_rank_answer",
              "band_agreement", "mean_entropy", "best_hedge_rank_log"]


def load(model: str):
    rows = [json.loads(l) for l in open(TRACE.format(model), encoding="utf-8") if l.strip()]

    def traj(r):
        e = np.asarray(r["layer_entropies"], dtype=float)
        return np.interp(np.linspace(0, 1, N_TRAJ), np.linspace(0, 1, len(e)), e)

    X_bl = np.array([[r[k] for k in BASELINE] for r in rows])
    X_ws = np.hstack([
        np.array([[r[k] for k in WS_SCALARS] for r in rows]),
        np.array([traj(r) for r in rows]),
    ])
    y = np.array([0 if r["correct"] else 1 for r in rows])
    q = np.array([r["q"] for r in rows])
    return {"logprob": X_bl, "workspace": X_ws,
            "combined": np.hstack([X_bl, X_ws])}, y, q


def zstats(X):
    return X.mean(0), X.std(0) + 1e-9


def apply_z(X, mu, sd):
    return (X - mu) / sd


def boot_ci(y, p, n=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


def make_tabfm():
    import torch
    from safetensors.torch import load_file
    from tabfm import TabFMClassifier
    from tabfm.src.pytorch.model import TabFM
    from tabfm.src.pytorch.tabfm_v1_0_0 import ClassificationConfig

    model = TabFM(**ClassificationConfig().to_dict())
    miss, unexp = model.load_state_dict(load_file(WEIGHTS), strict=False)
    assert not miss and not unexp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda" and torch.cuda.mem_get_info()[0] / 1e9 < 3.0:
            device = "cpu"
    except Exception:  # noqa: BLE001
        device = "cpu"
    model = model.to(device).eval()
    return lambda: TabFMClassifier(model=model, n_estimators=32, random_state=0)


def question_folds(questions, k=5, seed=0):
    """Disjoint question groups: fold id per unique question."""
    uniq = sorted(set(questions))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[i]: order_i % k for order_i, i in enumerate(order)}
    return np.array([fold_of[q] for q in questions])


def main() -> None:
    from sklearn.metrics import roc_auc_score

    make_clf = make_tabfm()
    data = {m: load(m) for m in GEMMAS + [QWEN]}
    results = {}

    print("== A. pooled 4-Gemma CV, question-grouped folds ==")
    for name in SETS:
        Xs, ys, qs = [], [], []
        for m in GEMMAS:
            sets, y, q = data[m]
            Xs.append(sets[name]); ys.append(y); qs.append(q)
        # per-model z with TRAIN-fold stats happens inside the fold loop
        y_all = np.concatenate(ys)
        q_all = np.concatenate(qs)
        folds = question_folds(q_all)
        oof = np.zeros(len(y_all))
        model_id = np.concatenate([[i] * len(ys[i]) for i in range(len(GEMMAS))])
        X_raw = np.vstack(Xs)
        for f in range(5):
            tr, te = folds != f, folds == f
            Xtr = X_raw.copy()
            # z-score each model's rows with stats from its TRAIN rows only
            for i in range(len(GEMMAS)):
                mrows = model_id == i
                mu, sd = zstats(X_raw[mrows & tr])
                Xtr[mrows] = apply_z(X_raw[mrows], mu, sd)
            clf = make_clf()
            clf.fit(Xtr[tr], y_all[tr])
            oof[te] = clf.predict_proba(Xtr[te])[:, 1]
        auc = roc_auc_score(y_all, oof)
        lo, hi = boot_ci(y_all, oof)
        results[f"pooled|{name}"] = {"auc": auc, "ci": [lo, hi]}
        print(f"  {name:>10}: {auc:.3f} [{lo:.3f}, {hi:.3f}] (n={len(y_all)})")

    def transfer(train_models, test_model, tag):
        sets_te, y_te, q_te = data[test_model]
        folds = question_folds(q_te)
        for name in SETS:
            for variant in ["strict", "unlabeled"]:
                oof = np.zeros(len(y_te))
                for f in range(5):
                    te = folds == f
                    tr_q_ok = folds != f  # question-disjoint mask by fold
                    Xtr_list, ytr_list, stats = [], [], []
                    for m in train_models:
                        sets_m, y_m, q_m = data[m]
                        keep = np.isin(q_m, q_te[tr_q_ok])
                        mu, sd = zstats(sets_m[name][keep])
                        stats.append((mu, sd))
                        Xtr_list.append(apply_z(sets_m[name][keep], mu, sd))
                        ytr_list.append(y_m[keep])
                    if variant == "strict":
                        mu = np.mean([s[0] for s in stats], axis=0)
                        sd = np.mean([s[1] for s in stats], axis=0)
                    else:  # unlabeled target stats from train-fold questions
                        mu, sd = zstats(sets_te[name][tr_q_ok])
                    clf = make_clf()
                    clf.fit(np.vstack(Xtr_list), np.concatenate(ytr_list))
                    oof[te] = clf.predict_proba(
                        apply_z(sets_te[name][te], mu, sd))[:, 1]
                auc = roc_auc_score(y_te, oof)
                lo, hi = boot_ci(y_te, oof)
                results[f"{tag}|{name}|{variant}"] = {"auc": auc, "ci": [lo, hi]}
                print(f"  {name:>10} {variant:>9}: {auc:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\n== B. leave-one-Gemma-out, question-disjoint ==")
    for held in GEMMAS:
        print(f" hold out {held}:")
        transfer([m for m in GEMMAS if m != held], held, f"logo|{held}")

    print("\n== C. all Gemmas -> Qwen, question-disjoint ==")
    transfer(GEMMAS, QWEN, "toqwen")

    with open("out/tabfm_transfer.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote out/tabfm_transfer.json")


if __name__ == "__main__":
    main()
