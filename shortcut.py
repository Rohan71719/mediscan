import argparse, glob
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils import OUT, compute_metrics, set_seed, get_logger

log = get_logger("shortcut")

def find_cache():
    hits = glob.glob("/kaggle/input/**/meta.csv", recursive=True)
    return str(__import__("pathlib").Path(hits[0]).parent) if hits else str(OUT / "cache")

def down(X, k):                       # 96 -> k x k by block averaging
    f = 96 // k
    return X.reshape(len(X), k, f, k, f).mean((2, 4)).reshape(len(X), -1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", default=None)
    a = ap.parse_args(); cache = a.cache or find_cache()
    set_seed(0)
    X = np.load(f"{cache}/X96.npy").astype(np.float32) / 255.0
    meta = pd.read_csv(f"{cache}/meta.csv"); y = meta.label.values
    tr, va = (meta.split == "train").values, (meta.split == "val").values

    Xmask = X.copy(); Xmask[:, 16:80, 16:80] = 0          # blank the central lung field
    feats = {
        "16x16 downsample": down(X, 16),
        "lungs masked (32x32 border only)": down(Xmask, 32),
        "metadata only (aspect, area, mean)": meta[["aspect", "area", "mean_int"]].values,
    }
    rows = [dict(model="majority class (PNEUMONIA)", **compute_metrics(y[va], np.ones(va.sum()), 0.5))]
    for name, F in feats.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(F[tr], y[tr])
        p = clf.predict_proba(F[va])[:, 1]
        rows.append(dict(model=name, **compute_metrics(y[va], p, 0.5)))
        log.info(f"{name}: acc={rows[-1]['accuracy']:.3f} bal_acc={rows[-1]['balanced_acc']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "shortcut_results.csv", index=False)
    print(res[["model", "accuracy", "balanced_acc", "recall", "specificity", "roc_auc", "pr_auc"]].round(3).to_string(index=False))
