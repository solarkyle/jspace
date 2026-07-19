"""The 31B blind test: frozen sweep config (32|native|z1|ext0), never tuned
on any 31B data. Per-model 5-fold CV, all three feature sets, plus logistic
reference. The theory's prediction, written before running: TabFM sees the
native per-layer entropies as separate columns, so unlike the band-mean it
should find the depth-migrated signal on its own."""
import json
import numpy as np

BASE = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob", "bl_answer_len"]
WS = ["ignition_frac", "ignition_depth", "mean_log_rank_answer",
      "band_agreement", "mean_entropy", "best_hedge_rank_log"]

rows = [json.loads(l) for l in open("data/uncertainty_triviaq4_gemma-4-31b-it.jsonl", encoding="utf-8") if l.strip()]
E = np.array([r["layer_entropies"] for r in rows])  # native resolution
Xb = np.array([[r[k] for k in BASE] for r in rows])
Xw = np.hstack([np.array([[r[k] for k in WS] for r in rows]), E])
y = np.array([0 if r["correct"] else 1 for r in rows])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from safetensors.torch import load_file
from tabfm import TabFMClassifier
from tabfm.src.pytorch.model import TabFM
from tabfm.src.pytorch.tabfm_v1_0_0 import ClassificationConfig
m = TabFM(**ClassificationConfig().to_dict())
m.load_state_dict(load_file("tabfm/classification/model.safetensors"), strict=False)
m = m.to("cuda" if torch.cuda.is_available() else "cpu").eval()

def cv(X, y, model_kind):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if model_kind == "tabfm":
            clf = TabFMClassifier(model=m, n_estimators=32, random_state=0)
            clf.fit(Xtr, y[tr])
            oof[te] = clf.predict_proba(Xte)[:, 1]
        else:
            clf = LogisticRegression(max_iter=2000).fit(Xtr, y[tr])
            oof[te] = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, oof)

for name, X in [("logprob", Xb), ("workspace", Xw), ("combined", np.hstack([Xb, Xw]))]:
    print(f"{name:>10}: logistic {cv(X, y, 'lr'):.3f}  TabFM(frozen) {cv(X, y, 'tabfm'):.3f}")
