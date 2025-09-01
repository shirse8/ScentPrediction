from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

from scripts.features import build_mordred_cache
from scripts.utils import train_val_test_split, evaluate_multilabel, optimize_thresholds


LABEL_COLS = ["label_pungent","label_sweet","label_floral","label_minty"]


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load CSV and ensure binary label columns exist.
    Falls back to prob_* >= 0.5 if label_* is missing.
    """
    df = pd.read_csv(path)
    for lab in LABEL_COLS:
        if lab not in df.columns:
            pc = lab.replace("label_","prob_")
            if pc in df.columns:
                df[lab] = (df[pc] >= 0.5).astype(int)
            else:
                raise ValueError(f"Missing {lab} and no {pc} to derive it.")
    return df


def build_X(df: pd.DataFrame, mordred_parquet: Path) -> pd.DataFrame:
    """Load Mordred features and align to df index."""
    X = pd.read_parquet(mordred_parquet)
    X = X.loc[df.index]
    return X


def predict_proba_ovr(model, X):
    """
    Predict per-label probabilities for One-vs-Rest models.
    Uses predict_proba/decision_function as available.
    """
    import numpy as np
    ps = []
    for est in model.estimators_:
        clf = est.named_steps.get("clf", est)
        if hasattr(clf, "predict_proba"):
            p = est.predict_proba(X)[:,1]
        else:
            if hasattr(est, "decision_function"):
                from scipy.special import expit
                p = expit(est.decision_function(X))
            else:
                p = est.predict(X)
        ps.append(p)
    return np.vstack(ps).T


def main(args):
    """
    Train OVR baselines (logreg, RF) with val-threshold tuning.
    Saves test metrics, per-label metrics, and ROC/PR plots.
    """
    data_csv = Path(args.data_csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_csv)
    y = df[LABEL_COLS].astype(int).values
    label_names = [c.replace("label_","") for c in LABEL_COLS]

    mordred_parquet = Path(args.mordred_parquet) if args.mordred_parquet else out_dir.parent / "features/mordred.parquet"
    mordred_parquet = build_mordred_cache(data_csv, smiles_col="SMILES", out_parquet=mordred_parquet, n_jobs=args.n_jobs)
    X = build_X(df, mordred_parquet)

    # split (stratification is handled inside helper if implemented)
    tr_idx, va_idx, te_idx = train_val_test_split(X, pd.DataFrame(y, columns=label_names), test_size=0.1, val_size=0.1, seed=args.seed)
    Xtr, Xva, Xte = X.iloc[tr_idx], X.iloc[va_idx], X.iloc[te_idx]
    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

    # models/pipelines
    base_lr = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=True)), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=args.n_jobs))])
    model_lr = OneVsRestClassifier(base_lr, n_jobs=args.n_jobs)

    base_rf = Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=args.seed, n_jobs=args.n_jobs))])
    model_rf = OneVsRestClassifier(base_rf, n_jobs=args.n_jobs)

    results = {}
    for name, model in [("logreg", model_lr), ("rf", model_rf)]:
        print(f"\nTraining {name}...")
        model.fit(Xtr, ytr)
        p_va = predict_proba_ovr(model, Xva)
        thr = optimize_thresholds(yva, p_va)

        p_te = predict_proba_ovr(model, Xte)
        ev = evaluate_multilabel(yte, p_te, thresholds=thr, label_names=label_names)
        results[name] = ev

        # save thresholds, probabilities, and metrics
        (out_dir/f"{name}_thresholds.npy").write_bytes(thr.tobytes())
        pd.DataFrame(p_te, columns=[f"prob_{l}" for l in label_names], index=df.index[te_idx]).to_csv(out_dir/f"{name}_test_probs.csv", index_label="row_id")
        ev["per_label"].to_csv(out_dir/f"{name}_per_label_metrics.csv", index=False)
        with open(out_dir/f"{name}_macro.json","w") as f: json.dump(ev["macro"], f, indent=2)

        # ROC plot per label
        plt.figure(figsize=(6,4))
        for i,l in enumerate(label_names):
            if len(np.unique(yte[:,i]))<2: continue
            fpr, tpr, _ = metrics.roc_curve(yte[:,i], p_te[:,i])
            plt.plot(fpr, tpr, label=l)
        plt.plot([0,1],[0,1],'--',alpha=0.5)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {name}"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir/f"{name}_roc.png", dpi=160); plt.close()

        # PR plot per label
        plt.figure(figsize=(6,4))
        for i,l in enumerate(label_names):
            if len(np.unique(yte[:,i]))<2: continue
            prec, rec, _ = metrics.precision_recall_curve(yte[:,i], p_te[:,i])
            plt.plot(rec, prec, label=l)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {name}"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir/f"{name}_pr.png", dpi=160); plt.close()

    # macro summary
    summary = pd.DataFrame({m: results[m]["macro"] for m in results}).T
    summary.to_csv(out_dir/"summary_macro.csv")
    print("\n=== Macro summary on test ===\n", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", default="data/molecule_scent_labels_with_probs.csv", type=str)
    p.add_argument("--out_dir", default="artifacts/classical", type=str)
    p.add_argument("--mordred_parquet", default="features/mordred.parquet", type=str)
    p.add_argument("--seed", default=17, type=int)
    p.add_argument("--n_jobs", default=1, type=int)
    args = p.parse_args()
    main(args)
