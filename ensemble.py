import argparse, re
from itertools import combinations
import numpy as np, pandas as pd
from src.utils import OUT, compute_metrics

COLS = ["accuracy", "balanced_acc", "recall", "specificity", "roc_auc", "pr_auc"]
AROGYA = ["arogya_full_224_aug", "arogya_no_res_224_aug", "arogya_no_se_224_aug", "arogya_meta_224_aug"]
SETS = {"soft vote: ArogyaNet variants": AROGYA, "soft vote: ArogyaNet variants + ResNet18": AROGYA + ["transfer"]}
SEED_ENS = ["arogya_full_224_aug", "transfer"]

def load(paths):
    df = pd.read_csv(paths[0])
    for p in paths[1:]:
        o = pd.read_csv(p)
        assert (o.fname.values == df.fname.values).all() and (o.label.values == df.label.values).all(), f"{p}: row order differs"
        for c in o.columns:
            if c not in df.columns: df[c] = o[c].values
    return df

def groups(df):
    g = {}
    for c in df.columns:
        m = re.match(r"(.+)_s(\d+)$", c)
        if m and "saved" not in m.group(1) and "_pw" not in m.group(1):
            g.setdefault(m.group(1), {})[int(m.group(2))] = c
    return g

def row(name, runs, y, thr):
    R = pd.DataFrame([compute_metrics(y, p, thr) for p in runs]); r = dict(model=name, runs=len(R))
    for c in COLS: r[c] = f"{R[c].mean():.4f} ± {R[c].std():.4f}" if len(R) > 1 else f"{R[c].iloc[0]:.4f}"
    r["_bal"], r["_auc"] = R.balanced_acc.mean(), R.roc_auc.mean()
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", default=["outputs/val_preds.csv"]); ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    df = load(a.preds); y = df.label.values.astype(int); g = groups(df)
    P = lambda n, s: df[g[n][s]].values
    print("models found:", {k: sorted(v) for k, v in g.items()})
    rows = [row(n, [P(n, s) for s in sorted(g[n])], y, a.thr) for n in sorted(g)]
    for label, members in SETS.items():
        members = [m for m in members if m in g]
        seeds = sorted(set.intersection(*[set(g[m]) for m in members])) if members else []
        if len(members) < 2 or not seeds: print("skipping:", label); continue
        rows.append(row(f"{label} ({len(members)} models)", [np.mean([P(m, s) for m in members], 0) for s in seeds], y, a.thr))
    for m in SEED_ENS:
        if m in g: rows.append(row(f"soft vote: 3 seeds of {m}", [np.mean([P(m, s) for s in sorted(g[m])], 0)], y, a.thr))
    T = pd.DataFrame(rows); T.drop(columns=["_bal", "_auc"]).to_csv(OUT / "ensemble_table.csv", index=False)
    print(f"\n== validation, threshold {a.thr} ==\n", T.drop(columns=["_bal", "_auc"]).to_string(index=False))
    mem = T[T.model.isin(sum(SETS.values(), []) + [n for n in g if n.startswith("emb_")])]
    best = mem.sort_values("_bal", ascending=False).iloc[0]
    for _, e in T[T.model.str.startswith("soft vote")].iterrows():
        print(f"{e.model}: bal acc {e._bal:.4f}, ROC-AUC {e._auc:.4f}  |  best single member {best.model}: {best._bal:.4f}, {best._auc:.4f}")
    members = [m for m in AROGYA + ["transfer"] if m in g and 0 in g[m]]
    if len(members) >= 2:
        err = {m: set(np.where((P(m, 0) >= a.thr) != y)[0]) for m in members}
        jac = [len(err[i] & err[j]) / max(len(err[i] | err[j]), 1) for i, j in combinations(members, 2)]
        rho = [pd.Series(P(i, 0)).corr(pd.Series(P(j, 0)), method="spearman") for i, j in combinations(members, 2)]
        print(f"\nseed 0: errors per member {[len(e) for e in err.values()]}; mean pairwise error overlap (Jaccard) "
              f"{np.mean(jac):.2f}; wrong in every member {len(set.intersection(*err.values()))}; wrong in at least one "
              f"{len(set.union(*err.values()))}; mean Spearman correlation of probabilities {np.mean(rho):.3f}")
