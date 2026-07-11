"""Temporal model: does trajectory ORDER add signal beyond static summaries?

The honest version of the handoff's temporal CNN. Codex spec'd dilations over a
length-5 prefix sequence (receptive field >> sequence). Instead this runs a 1D
CNN over the real sequence we capture for free: the per-band-layer entropy
trajectory (the depth axis, ~24 points). The 31B depth-migration result showed
this trajectory has structure that fixed-fraction summaries miss.

Fair test: temporal CNN on the raw layer trajectory vs LightGBM on the STATIC
summaries of that same trajectory (ent_early/mid/late/slope/std/max_jump...),
both under leave-one-dataset-out. If the CNN wins cross-dataset, order matters.

    python -m campaign.train_temporal --input out/campaign/stage1_features.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from campaign.split_groups import leave_one_dataset_out
from campaign.train_baselines import _matrix, _auc

TRAJ_SUMMARY = ["ent_early", "ent_mid", "ent_late", "ent_slope", "ent_std",
                "ent_max_jump", "ent_n_dir_changes"]


def _pad(seq, L):
    a = np.asarray(seq, dtype=np.float32)
    if len(a) >= L:
        return a[:L]
    return np.concatenate([a, np.full(L - len(a), a[-1] if len(a) else 0.0, np.float32)])


class _CNN:
    """Tiny 1D CNN over the layer-entropy trajectory. Kept self-contained."""

    def __init__(self, L, seed=0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch = torch
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(16, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1),
        )
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.dev)

    def fit(self, X, y, epochs=40):
        t = self.torch
        Xt = t.tensor(X[:, None, :], device=self.dev)
        yt = t.tensor(y[:, None].astype(np.float32), device=self.dev)
        w = float((y == 0).sum()) / max(1, (y == 1).sum())  # pos weight
        opt = t.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = t.nn.BCEWithLogitsLoss(pos_weight=t.tensor([w], device=self.dev))
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            lossf(self.net(Xt), yt).backward()
            opt.step()
        return self

    def predict_proba(self, X):
        t = self.torch
        self.net.eval()
        with t.no_grad():
            p = t.sigmoid(self.net(t.tensor(X[:, None, :], device=self.dev)))
        return p.cpu().numpy().ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if "label" in r and r.get("layer_entropies_onset")]
    L = max(len(r["layer_entropies_onset"]) for r in labeled)
    traj = np.array([_pad(r["layer_entropies_onset"], L) for r in labeled])
    # z-normalize each trajectory (shape, not absolute level)
    traj = (traj - traj.mean(1, keepdims=True)) / (traj.std(1, keepdims=True) + 1e-6)
    y = np.array([r["label"] for r in labeled])

    print(f"{len(labeled)} rows, trajectory length {L}")
    print("== temporal CNN (raw layer trajectory) vs LightGBM (static summaries), LODO ==")
    print(f"{'held-out':>12} {'CNN':>7} {'LGBM-static':>12}")
    cnn_per, lgbm_per = {}, {}
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    Xs = _matrix(labeled, TRAJ_SUMMARY)
    for held, tr, te in leave_one_dataset_out(labeled):
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            continue
        c = _CNN(L).fit(traj[tr], y[tr]).predict_proba(traj[te])
        cnn_per[held] = _auc(y[te], c)
        sc = StandardScaler().fit(Xs[tr])
        g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               class_weight="balanced", verbosity=-1).fit(
            sc.transform(Xs[tr]), y[tr]).predict_proba(sc.transform(Xs[te]))[:, 1]
        lgbm_per[held] = _auc(y[te], g)
        print(f"{held:>12} {cnn_per[held]:>7.3f} {lgbm_per[held]:>12.3f}")
    cm = np.mean([v for v in cnn_per.values()])
    gm = np.mean([v for v in lgbm_per.values()])
    print(f"{'MEAN':>12} {cm:>7.3f} {gm:>12.3f}")
    print(f"\ntrajectory-order advantage (CNN - static): {cm - gm:+.4f}")
    print("-> order carries extra signal" if cm - gm > 0.01 else
          "-> static summaries capture the trajectory (order adds nothing)")
    if args.out:
        json.dump({"cnn": cnn_per, "lgbm_static": lgbm_per,
                   "cnn_mean": cm, "lgbm_mean": gm}, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
