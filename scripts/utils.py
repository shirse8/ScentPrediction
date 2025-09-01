from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics


def multilabel_signature(Y: pd.DataFrame) -> pd.Series:
    """Concatenate binary label columns into a per-row string (e.g., '1010')."""
    return Y.astype(int).astype(str).agg("".join, axis=1)


def train_val_test_split(X: pd.DataFrame, Y: pd.DataFrame, test_size=0.1, val_size=0.1, seed=17):
    """Split X/Y into train/val/test with optional stratification on label signature."""
    sig = multilabel_signature(Y)
    # Use stratification only if not too many unique signatures
    strat = sig if sig.nunique() < len(sig) * 0.5 else None

    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)

    # Recompute stratification for the train subset
    if strat is not None:
        strat2 = sig.iloc[tr_idx]
        strat2 = strat2 if strat2.nunique() < len(strat2) * 0.5 else None
    else:
        strat2 = None

    # val size is relative to remaining train pool
    tr_idx, va_idx = train_test_split(tr_idx, test_size=val_size/(1.0-test_size), random_state=seed, stratify=strat2)
    return tr_idx, va_idx, te_idx


def evaluate_multilabel(y_true, y_prob, thresholds=None, label_names=None):
    """Compute per-label and macro metrics (ROC-AUC, PR-AUC, F1, Precision, Recall)."""
    n_labels = y_true.shape[1]
    import numpy as np, pandas as pd
    if thresholds is None:
        thresholds = np.array([0.5]*n_labels)

    y_pred = (y_prob >= thresholds).astype(int)

    roc_auc, pr_auc, f1, prec, rec = [], [], [], [], []
    for k in range(n_labels):
        # Skip metrics that require both classes when label is constant
        if len(np.unique(y_true[:,k])) < 2:
            roc_auc.append(np.nan); pr_auc.append(np.nan); f1.append(np.nan); prec.append(np.nan); rec.append(np.nan)
            continue
        roc_auc.append(metrics.roc_auc_score(y_true[:,k], y_prob[:,k]))
        pr_auc.append(metrics.average_precision_score(y_true[:,k], y_prob[:,k]))
        f1.append(metrics.f1_score(y_true[:,k], y_pred[:,k]))
        prec.append(metrics.precision_score(y_true[:,k], y_pred[:,k], zero_division=0))
        rec.append(metrics.recall_score(y_true[:,k], y_pred[:,k], zero_division=0))

    out = {
        "per_label": pd.DataFrame({
            "label": label_names if label_names else [f"label_{i}" for i in range(n_labels)],
            "roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1, "precision": prec, "recall": rec, "threshold": thresholds,
        }),
        "macro": {
            "roc_auc": np.nanmean(roc_auc), "pr_auc": np.nanmean(pr_auc), "f1": np.nanmean(f1),
            "precision": np.nanmean(prec), "recall": np.nanmean(rec),
        }
    }
    return out


def optimize_thresholds(y_true, y_prob):
    """Per-label threshold sweep (0..1) maximizing F1 on validation data."""
    import numpy as np
    from sklearn import metrics
    n_labels = y_true.shape[1]
    thresholds = np.zeros(n_labels, dtype=float)

    for k in range(n_labels):
        best_t, best_f1 = 0.5, -1.0
        # Grid search over thresholds
        for t in np.linspace(0, 1, 101):
            y_pred = (y_prob[:,k] >= t).astype(int)
            f1 = metrics.f1_score(y_true[:,k], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[k] = best_t

    return thresholds
