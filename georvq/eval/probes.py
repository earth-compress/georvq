"""Lightweight frozen-embedding probes, per AlphaEarth Foundations.

Three probes -- kNN(k=1), kNN(k=3), linear -- over a geographic-block train/test
split, swept across sparse training budgets (the paper's "max trial size" regime,
where embedding quality shows up most). Classification reports balanced accuracy;
regression reports R^2. The same fitted-from-scratch probe is applied to original
and reconstructed embeddings so only the codec changes between the two columns.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def geo_block_split(groups, test_frac=0.3, seed=0):
    """Split by group id (one group = one AEF tile ~ 82 km block) so no tile
    straddles train and test. Returns boolean (train_mask, test_mask)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_groups = set(uniq[perm[:n_test]].tolist())
    test_mask = np.array([g in test_groups for g in groups])
    return ~test_mask, test_mask


def _budget_indices(y, train_idx, budget, task, rng):
    """Sub-sample train indices to a sparse budget: `budget` samples per class
    for classification, `budget` total for regression. None/<=0 keeps all."""
    if not budget or budget <= 0:
        return train_idx
    if task == "classification":
        out = []
        for cls in np.unique(y[train_idx]):
            idx = train_idx[y[train_idx] == cls]
            take = min(budget, len(idx))
            out.append(rng.choice(idx, take, replace=False))
        return np.concatenate(out)
    take = min(budget, len(train_idx))
    return rng.choice(train_idx, take, replace=False)


def _fit_eval(probe, task, Xtr, ytr, Xte, yte):
    if probe == "linear":
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        if task == "classification":
            m = LogisticRegression(max_iter=2000, C=1.0)
        else:
            m = Ridge(alpha=1.0)
    else:                                              # knn1 / knn3
        k = int(probe[3:])
        Cls = KNeighborsClassifier if task == "classification" \
            else KNeighborsRegressor
        m = Cls(n_neighbors=k)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    if task == "classification":
        return float(balanced_accuracy_score(yte, pred))
    return float(r2_score(yte, pred))


def run_probes(X, y, groups, task, budgets=(50, 200, None),
               probes=("knn1", "knn3", "linear"), test_frac=0.3, seed=0):
    """Run every (probe, budget) cell and return a list of result rows. The
    train/test split is fixed by `seed`; budgets only thin the train set, so a
    single shared test set scores every cell -- comparable across orig/recon."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    tr_mask, te_mask = geo_block_split(groups, test_frac, seed)
    train_idx = np.nonzero(tr_mask)[0]
    Xte, yte = X[te_mask], y[te_mask]
    rng = np.random.default_rng(seed + 1)
    metric = "balanced_accuracy" if task == "classification" else "r2"
    rows = []
    for budget in budgets:
        idx = _budget_indices(y, train_idx, budget, task, rng)
        Xtr, ytr = X[idx], y[idx]
        for probe in probes:
            val = _fit_eval(probe, task, Xtr, ytr, Xte, yte)
            rows.append({"probe": probe, "budget": budget or "all",
                         "metric": metric, "value": val,
                         "n_train": len(idx), "n_test": len(yte)})
    return rows
