import numpy as np, pandas as pd
from src.utils import OUT
from external_eval import paired

d = pd.read_csv(OUT / "external_preds.csv")
ind = ~d.flagged.values; vir = d.flagged.values & (d.label.values == 1)      # vir: viral images that duplicate primary data
rows = []
for s in range(3):
    pdv, pfv = d[f"p_default_s{s}"].values, d[f"p_fix_s{s}"].values
    for thr in [0.4158, 0.5498]:
        rows.append(dict(seed=s, threshold=thr, default_spec_indep=(pdv[ind] < thr).mean(), fix_spec_indep=(pfv[ind] < thr).mean(),
                         default_recall_dupviral=(pdv[vir] >= thr).mean(), fix_recall_dupviral=(pfv[vir] >= thr).mean()))
R = pd.DataFrame(rows); R.to_csv(OUT / "external_posthoc.csv", index=False)
print(R.groupby("threshold").mean().drop(columns="seed").round(4).T)
p0, p1 = d.p_default_s0.values[ind], d.p_fix_s0.values[ind]
for thr in [0.4158, 0.5498]:
    g, lo, hi = paired(np.zeros(ind.sum()), p0, thr, p1, thr, "specificity")
    print(f"seed 0, both models at threshold {thr}: specificity gain of the fix {g:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]")
