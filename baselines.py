import argparse, glob
import numpy as np, pandas as pd
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from src.utils import OUT, compute_metrics, set_seed, get_logger, find_files

log = get_logger("baselines")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    hits = find_files("meta.csv")
    cache = a.cache or str(__import__("pathlib").Path(hits[0]).parent)
    set_seed(0)
    X = np.load(f"{cache}/X96.npy"); meta = pd.read_csv(f"{cache}/meta.csv"); y = meta.label.values
    tr, va = (meta.split == "train").values, (meta.split == "val").values

    H = np.stack([hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for x in tqdm(X)])
    log.info(f"HOG features: {H.shape}")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    clf.fit(H[tr], y[tr])
    p = clf.predict_proba(H[va])[:, 1]

    rows = [dict(model="majority class (PNEUMONIA)", **compute_metrics(y[va], np.ones(va.sum()), 0.5)),
            dict(model="HOG + logistic regression", **compute_metrics(y[va], p, 0.5))]
    res = pd.DataFrame(rows); res.to_csv(OUT / "baselines.csv", index=False)
    print(res[["model", "accuracy", "balanced_acc", "precision", "recall", "f1", "pr_auc"]].round(3).to_string(index=False))
