import argparse, hashlib
import numpy as np, pandas as pd
from scipy.stats import beta
from sklearn.model_selection import StratifiedGroupKFold
from src.utils import OUT, get_logger, set_seed

log = get_logger("split")
NEAR_THR = 2
SEED = 42

def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

def build_groups(df):
    """Union-Find over: same patient OR near-duplicate image."""
    parent = list(range(len(df)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    # same patient
    for _, idx in df.groupby("patient").indices.items():
        for i in idx[1:]: union(idx[0], i)
    # near-duplicates and exact duplicates
    H = np.array([[int(b) for b in format(int(h, 16), "064b")] for h in df.phash], dtype=np.float32)
    ham = H.shape[1] - (H @ H.T + (1 - H) @ (1 - H).T)
    np.fill_diagonal(ham, 99)
    lab = df.label.values
    for i, j in np.argwhere(np.triu(ham <= NEAR_THR, k=1)):
        if lab[i] == lab[j]:
            union(i, j)
    return np.array([find(i) for i in range(len(df))])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=str(OUT / "inventory.csv"))
    a = ap.parse_args()
    set_seed(SEED)

    # ---- 1. why 100% on n=16 means little ----
    lo, hi = clopper_pearson(16, 16)
    print(f"Observed 16/16 = 100%. 95% Clopper-Pearson CI: [{lo:.3f}, {hi:.3f}]")
    pd.DataFrame([{"k": 16, "n": 16, "lo": lo, "hi": hi}]).to_csv(OUT / "val16_ci.csv", index=False)

    # ---- 2. pool everything ----
    df = pd.read_csv(a.inventory)
    df["orig_split"] = df["split"]
    n0 = len(df)
    df = df.drop_duplicates("md5", keep="first").reset_index(drop=True)   # exact duplicates
    log.info(f"Pool: {n0} -> {len(df)} after removing exact duplicates")
    df["group"] = build_groups(df)
    print("Groups:", df.group.nunique(), "for", len(df), "images")

    # ---- 3. locked test (15%) ----
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)   # ~14.3% held out
    pool_idx, test_idx = next(sgkf.split(df, df.label, df.group))
    df["split"] = "pool"; df.loc[test_idx, "split"] = "test"

    # ---- 4. validation (~15% of total) from the pool ----
    pool = df[df.split == "pool"]
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED + 1)
    tr_i, va_i = next(sgkf2.split(pool, pool.label, pool.group))
    df.loc[pool.index[va_i], "split"] = "val"
    df.loc[pool.index[tr_i], "split"] = "train"

    # ---- 5. 5-fold CV assignment on the non-test pool (train+val) ----
    df["fold"] = -1
    cvpool = df[df.split != "test"]
    sgkf3 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 2)
    for f, (_, te) in enumerate(sgkf3.split(cvpool, cvpool.label, cvpool.group)):
        df.loc[cvpool.index[te], "fold"] = f

    # ---- 6. verify no leakage between split sets ----
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        g1 = set(df[df.split == s1].group); g2 = set(df[df.split == s2].group)
        pa = set(df[df.split == s1].patient); pb = set(df[df.split == s2].patient)
        assert not (g1 & g2), f"group leak {s1}/{s2}"
        assert not (pa & pb), f"patient leak {s1}/{s2}"
    print("Leakage check passed: no group or patient shared across train/val/test")

    summary = df.groupby(["split", "cls"]).size().unstack()
    summary["pneumonia_frac"] = summary["PNEUMONIA"] / (summary["NORMAL"] + summary["PNEUMONIA"])
    print(summary); summary.to_csv(OUT / "split_summary.csv")
    print(df[df.split != "test"].groupby(["fold", "cls"]).size().unstack())

    keep = ["path", "fname", "cls", "label", "patient", "group", "split", "fold", "orig_split",
            "h", "w", "aspect", "area", "mean_int", "std_int", "phash"]
    df[keep].to_csv(OUT / "split.csv", index=False)
    h = hashlib.md5(open(OUT / "split.csv", "rb").read()).hexdigest()
    open(OUT / "split.md5", "w").write(h)
    log.info(f"split.csv written, md5={h}")
