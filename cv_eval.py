import argparse, numpy as np, pandas as pd, torch
from src.utils import OUT, set_seed, compute_metrics, get_logger
from src.data import Data, make_augment
from src.models import ArogyaNet
from src.train import train_model

log = get_logger("cv")
PI = 0.10                                              # camp prevalence
COLS = ["accuracy", "balanced_acc", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
        "camp_ppv", "camp_cost_per_1000"]

def camp(m):
    r, s = m["recall"], m["specificity"]
    return dict(camp_ppv=PI * r / (PI * r + (1 - PI) * (1 - s)),
                camp_cost_per_1000=1000 * (PI * (1 - r) * 20 + (1 - PI) * (1 - s)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.4158)   # frozen in threshold.py (operating_point.json)
    a = ap.parse_args()
    d = Data(224); fold = d.meta.fold.values                      # fold -1 = locked test set, never used here
    rows, oof = [], []
    for k in range(5):
        tr, te = np.where((fold >= 0) & (fold != k))[0], np.where(fold == k)[0]
        set_seed(k)
        model = ArogyaNet().cuda()
        model, _, probs, _ = train_model(model, d, tr, te, k, epochs=a.epochs, lr=2e-3, wd=1e-4, bs=64,
                                         augment=make_augment(), select_best=False, verbose=False)
        y = d.y[torch.as_tensor(te, device=d.device)].cpu().numpy()
        for thr in [a.threshold, 0.5]:
            m = compute_metrics(y, probs, thr); m.update(camp(m))
            rows.append(dict(fold=k, threshold=thr, n=len(te), **m))
        oof.append(pd.DataFrame(dict(fname=d.meta.fname.values[te], fold=k, label=y, prob=probs)))
        log.info(f"fold {k}: " + ", ".join(f"{c}={rows[-2][c]:.4f}" for c in ["balanced_acc", "recall", "specificity", "roc_auc"]))
    R = pd.DataFrame(rows); R.to_csv(OUT / "cv_folds.csv", index=False)
    pd.concat(oof).to_csv(OUT / "cv_oof_preds.csv", index=False)
    for thr, g in R.groupby("threshold"):
        S = pd.DataFrame({"mean": g[COLS].mean(), "sd": g[COLS].std()}).round(4)
        S.to_csv(OUT / f"cv_summary_thr{thr:.4f}.csv")
        print(f"\n== 5-fold CV, threshold {thr:.4f} (mean ± SD over folds) ==\n", S)
        print("pooled confusion matrix over all out-of-fold images: TN %d FP %d FN %d TP %d" %
              tuple(int(g[c].sum()) for c in ["tn", "fp", "fn", "tp"]))
