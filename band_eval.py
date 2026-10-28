import argparse, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--preds", default="outputs/val_preds.csv")
ap.add_argument("--col", default="arogya_full_224_aug_saved_s0"); a = ap.parse_args()
df = pd.read_csv(a.preds); y = df.label.values; p = df[a.col].values; rows = []
for lo in [0.05, 0.10, 0.20, 0.30, 0.4158]:
    ref = p >= lo; rec = ref[y == 1].mean(); spec = (~ref)[y == 0].mean()
    rows.append(dict(refer_if_p_at_least=lo, recall=rec, specificity=spec, missed_per_1000=100 * (1 - rec),
                     false_alarms_per_1000=900 * (1 - spec), cost_per_1000=2000 * (1 - rec) * 0.1 * 10 + 900 * (1 - spec),
                     share_of_validation_in_band_lo_to_0p9=((p >= lo) & (p < 0.9)).mean()))
print(pd.DataFrame(rows).round(4).to_string(index=False))
err = ((p >= 0.4158) != (y == 1)); inb = (p > 0.1) & (p < 0.9)
print(f"errors at 0.4158: {err.sum()}; {inb[err].mean():.0%} of errors but {inb[~err].mean():.0%} of correct predictions lie in 0.1 to 0.9")
